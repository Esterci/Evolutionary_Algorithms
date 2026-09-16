"""Behavioral checks for the weight-vector differential evolution stage."""

import math
import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from methods.differential_evolution import DEConfig, differential_evolution


class DifferentialEvolutionTests(unittest.TestCase):
    def test_progress_callback_receives_independent_history_rows(self):
        recorded = []
        result = differential_evolution(
            lambda x: x.square().sum(),
            torch.tensor([0.5], dtype=torch.float64),
            DEConfig(population_size=4, generations=2),
            on_generation=recorded.append,
        )
        self.assertEqual(recorded, result.history)
        self.assertEqual([row["generation"] for row in recorded], [0, 1, 2])
        recorded[0]["best_fitness"] = -1.0
        self.assertNotEqual(recorded[0], result.history[0])

    def test_reproducibility_and_global_rng_independence(self):
        initial = torch.tensor([0.8, -0.6], dtype=torch.float64)
        config = DEConfig(population_size=8, generations=5, seed=42)
        global_state = torch.random.get_rng_state().clone()
        first = differential_evolution(lambda x: x.square().sum(), initial, config)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        try:
            torch.manual_seed(987)
            second = differential_evolution(lambda x: x.square().sum(), initial, config)
        finally:
            torch.random.set_rng_state(global_state)
        self.assertTrue(torch.equal(first.population, second.population))
        self.assertTrue(torch.equal(first.fitness, second.fitness))
        self.assertTrue(torch.equal(first.mutation_factors, second.mutation_factors))
        self.assertTrue(torch.equal(first.crossover_rates, second.crossover_rates))
        self.assertEqual(first.history, second.history)

    def test_elitism_fitness_consistency_bounds_and_exact_budget(self):
        config = DEConfig(population_size=8, generations=12, seed=17)
        result = differential_evolution(
            lambda x: x.square().sum(),
            torch.tensor([0.9, 0.9], dtype=torch.float64),
            config,
        )
        best_history = [row["best_fitness"] for row in result.history]
        self.assertTrue(all(a >= b for a, b in zip(best_history, best_history[1:])))
        self.assertLess(best_history[-1], best_history[0])
        self.assertEqual(len(result.history), config.generations + 1)
        self.assertEqual(
            result.fitness_evaluations,
            config.population_size * (config.generations + 1),
        )
        for generation, row in enumerate(result.history):
            self.assertEqual(row["generation"], generation)
            self.assertEqual(
                row["fitness_evaluations"], config.population_size * (generation + 1)
            )
            self.assertIsInstance(row["best_index"], int)
            self.assertGreaterEqual(row["best_index"], 0)
            self.assertLess(row["best_index"], config.population_size)
        torch.testing.assert_close(
            result.population.square().sum(dim=1), result.fitness
        )
        self.assertEqual(result.best_fitness, result.fitness.min().item())
        self.assertEqual(
            result.history[-1]["best_index"], result.fitness.argmin().item()
        )
        self.assertAlmostEqual(
            result.best_fitness, result.best_vector.square().sum().item()
        )
        self.assertTrue(torch.all(result.population >= config.lower_bound))
        self.assertTrue(torch.all(result.population <= config.upper_bound))
        self.assertTrue(
            torch.all(
                (result.mutation_factors >= 0.3) & (result.mutation_factors <= 0.9)
            )
        )
        self.assertTrue(
            torch.all((result.crossover_rates >= 0.0) & (result.crossover_rates <= 1.0))
        )
        self.assertEqual(result.nonfinite_evaluations, 0)

    def test_control_adaptation_works_for_large_genomes_and_preserves_rejected_controls(
        self,
    ):
        dimension = 512
        config = DEConfig(
            population_size=12, generations=1, seed=31, adaptation_probability=1.0
        )
        initial = torch.zeros(dimension, dtype=torch.float64)
        generator = torch.Generator().manual_seed(config.seed)
        torch.rand(
            config.population_size, dimension, dtype=initial.dtype, generator=generator
        )
        original_factors = 0.3 + 0.6 * torch.rand(
            config.population_size, dtype=initial.dtype, generator=generator
        )
        original_rates = torch.rand(
            config.population_size, dtype=initial.dtype, generator=generator
        )
        # Keep a large genome but use one active objective coordinate to
        # exercise acceptance and rejection with this fixed seed. Progress on
        # the 512-D sphere is not guaranteed within a single generation.
        result = differential_evolution(lambda x: x[0].square(), initial, config)
        row = result.history[1]
        self.assertEqual(row["control_adaptation_attempts"], config.population_size)
        self.assertGreater(row["accepted_trials"], 0)
        self.assertEqual(row["control_adaptation_acceptances"], row["accepted_trials"])
        self.assertEqual(
            int((result.mutation_factors != original_factors).sum()),
            row["accepted_trials"],
        )
        self.assertEqual(
            int((result.crossover_rates != original_rates).sum()),
            row["accepted_trials"],
        )
        # The exact minimizer cannot be improved: its proposed controls
        # must be rejected together with its trial genome.
        self.assertEqual(result.mutation_factors[0], original_factors[0])
        self.assertEqual(result.crossover_rates[0], original_rates[0])

    def test_local_initialization_mixed_bounds_and_reproducibility(self):
        initial = torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64)
        lower = torch.tensor([-1.0, -3.0, 0.0], dtype=initial.dtype)
        upper = torch.tensor([1.0, -1.0, 1.0], dtype=initial.dtype)
        config = DEConfig(
            population_size=12,
            generations=2,
            seed=31,
            initialization="local",
            initial_spread=0.05,
        )
        evaluated = []

        def objective(vector):
            evaluated.append(vector.clone())
            return ((vector - initial) / (upper - lower)).square().sum()

        result = differential_evolution(
            objective, initial, config, bounds=(lower, upper)
        )
        again = differential_evolution(
            objective, initial, config, bounds=(lower, upper)
        )
        torch.testing.assert_close(result.population, again.population, rtol=0, atol=0)
        self.assertEqual(result.history, again.history)
        initialized = torch.stack(evaluated[: config.population_size])
        torch.testing.assert_close(initialized[0], initial, rtol=0, atol=0)
        self.assertTrue(
            torch.all(
                (initialized - initial).abs() <= config.initial_spread * (upper - lower)
            )
        )
        self.assertTrue(torch.all(initialized >= lower))
        self.assertTrue(torch.all(initialized <= upper))
        self.assertTrue(torch.all(result.population >= lower))
        self.assertTrue(torch.all(result.population <= upper))
        self.assertTrue(torch.all(initialized[1:, 2] <= 1.0))
        # Mixed-unit diversity is reported after dividing each coordinate's
        # standard deviation by its own bound width.
        expected_spread = (
            (initialized.std(0, unbiased=False) / (upper - lower)).mean().item()
        )
        self.assertAlmostEqual(result.history[0]["population_spread"], expected_spread)
        for row in result.history:
            self.assertLessEqual(
                row["control_adaptation_acceptances"], row["accepted_trials"]
            )
            self.assertLessEqual(
                row["control_adaptation_acceptances"],
                row["control_adaptation_attempts"],
            )
        self.assertEqual(result.history[0]["accepted_trials"], 0)
        self.assertEqual(result.history[0]["control_adaptation_attempts"], 0)

    def test_legacy_adaptation_retains_work3_control_ranges(self):
        config = DEConfig(
            population_size=8, generations=12, seed=17, control_adaptation="legacy"
        )
        result = differential_evolution(
            lambda x: x.square().sum(),
            torch.tensor([0.9, 0.9], dtype=torch.float64),
            config,
        )
        self.assertTrue(
            torch.all((result.crossover_rates >= 0.9) & (result.crossover_rates <= 1.0))
        )
        self.assertLess(
            result.history[-1]["best_fitness"], result.history[0]["best_fitness"]
        )
        self.assertGreater(
            sum(row["control_adaptation_attempts"] for row in result.history), 0
        )

    def test_unbounded_genes_escape_initialization_box_while_network_stays_bounded(self):
        initial = torch.zeros(2, dtype=torch.float64)
        lower = torch.tensor([-1.0, -0.2], dtype=initial.dtype)
        upper = -lower
        config = DEConfig(population_size=16, generations=50, seed=42)
        evaluated = []

        def objective(vector):
            evaluated.append(vector.clone())
            return vector[0].square() + (vector[1] - 3.0).square()

        result = differential_evolution(
            objective,
            initial,
            config,
            bounds=(lower, upper),
            bound_mask=torch.tensor([True, False]),
        )
        candidates = torch.stack(evaluated)
        initialized = candidates[: config.population_size]
        self.assertTrue(torch.all(initialized >= lower))
        self.assertTrue(torch.all(initialized <= upper))
        self.assertTrue(torch.all(candidates[:, 0].abs() <= 1.0))
        self.assertGreater(result.best_vector[1].item(), upper[1].item())
        expected_spread = (
            result.population.std(0, unbiased=False) / (upper - lower)
        ).mean().item()
        self.assertAlmostEqual(result.history[-1]["population_spread"], expected_spread)
        self.assertEqual(len(evaluated), result.fitness_evaluations)

    def test_explicit_all_bounded_mask_preserves_seeded_default_behavior(self):
        initial = torch.tensor([0.8, -0.6], dtype=torch.float64)
        for control_adaptation in ("jde", "legacy"):
            with self.subTest(control_adaptation=control_adaptation):
                config = DEConfig(
                    population_size=8,
                    generations=5,
                    seed=42,
                    control_adaptation=control_adaptation,
                )
                default = differential_evolution(
                    lambda x: x.square().sum(), initial, config
                )
                masked = differential_evolution(
                    lambda x: x.square().sum(),
                    initial,
                    config,
                    bound_mask=torch.ones_like(initial, dtype=torch.bool),
                )
                torch.testing.assert_close(
                    default.population, masked.population, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    default.fitness, masked.fitness, rtol=0, atol=0
                )
                self.assertEqual(default.history, masked.history)

    def test_bound_mask_requires_boolean_vector_on_genome_device(self):
        initial = torch.zeros(2, dtype=torch.float64)
        config = DEConfig(population_size=4, generations=1)
        for mask in (
            [True, False],
            torch.tensor(True),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.bool, device="meta"),
        ):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "bound_mask"):
                differential_evolution(
                    lambda x: x.square().sum(), initial, config, bound_mask=mask
                )
        # Unbounded evolution still requires a finite and valid initial box.
        with self.assertRaisesRegex(ValueError, "inside the DE bounds"):
            differential_evolution(
                lambda x: x.square().sum(),
                torch.tensor([0.0, 2.0]),
                config,
                bound_mask=torch.tensor([True, False]),
            )

    def test_unbounded_overflow_is_counted_and_never_enters_population(self):
        finite_candidates = []

        def objective(vector):
            finite = bool(torch.isfinite(vector).all())
            finite_candidates.append(finite)
            # Even a finite returned fitness cannot make an invalid genome win.
            return -vector[1] / 1e38 if finite else -1e20

        config = DEConfig(population_size=12, generations=20, seed=1)
        result = differential_evolution(
            objective,
            torch.zeros(2),
            config,
            bounds=(torch.tensor([-1.0, -1.5e38]), torch.tensor([1.0, 1.5e38])),
            bound_mask=torch.tensor([True, False]),
        )
        self.assertGreater(result.nonfinite_evaluations, 0)
        self.assertEqual(result.nonfinite_evaluations, finite_candidates.count(False))
        self.assertEqual(result.fitness_evaluations, len(finite_candidates))
        self.assertEqual(
            result.fitness_evaluations,
            config.population_size * (config.generations + 1),
        )
        self.assertTrue(torch.isfinite(result.population).all())
        self.assertTrue(torch.isfinite(result.fitness).all())
        self.assertTrue(
            all(math.isfinite(row["population_spread"]) for row in result.history)
        )

    def test_explicit_bounds_reject_broadcasting_mismatch_and_invalid_intervals(self):
        initial = torch.zeros(2, dtype=torch.float64)
        config = DEConfig(population_size=4, generations=1)
        for bounds in (
            (torch.tensor(-1.0), torch.tensor(1.0)),
            (torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.float32)),
            (torch.zeros(2, dtype=initial.dtype), torch.zeros(2, dtype=initial.dtype)),
            (
                torch.tensor([-1.0, math.nan], dtype=initial.dtype),
                torch.ones(2, dtype=initial.dtype),
            ),
            (
                torch.ones(2, dtype=initial.dtype),
                2 * torch.ones(2, dtype=initial.dtype),
            ),
            (-1.0, 1.0),
            (torch.zeros(2, dtype=initial.dtype),),
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                differential_evolution(
                    lambda x: x.square().sum(), initial, config, bounds=bounds
                )

    def test_initial_genome_equal_fitness_survival_and_no_aliasing(self):
        initial = torch.tensor([0.5, -0.5], requires_grad=True)
        original = initial.detach().clone()
        evaluated = []

        def constant_objective(vector):
            evaluated.append(vector.clone())
            vector.fill_(0.75)
            return 1.0

        result = differential_evolution(
            constant_objective, initial, DEConfig(population_size=4, generations=2)
        )
        torch.testing.assert_close(evaluated[0], original)
        torch.testing.assert_close(result.population[0], original)
        torch.testing.assert_close(result.best_vector, original)
        torch.testing.assert_close(initial, original)
        self.assertFalse(result.population.requires_grad)
        self.assertFalse(result.best_vector.requires_grad)
        self.assertTrue(all(row["best_index"] == 0 for row in result.history))
        self.assertEqual(result.population.dtype, initial.dtype)
        result.best_vector.add_(0.1)
        torch.testing.assert_close(result.population[0], original)

    def test_residual_autograd_is_available(self):
        def objective(vector):
            points = torch.tensor(
                [[0.2], [0.8]], dtype=vector.dtype, requires_grad=True
            )
            predictions = vector[0] * points.square()
            derivative = torch.autograd.grad(
                predictions.sum(), points, create_graph=True
            )[0]
            return derivative.square().mean()

        with torch.no_grad():
            result = differential_evolution(
                objective,
                torch.tensor([0.5], dtype=torch.float64),
                DEConfig(population_size=4, generations=1),
            )
        self.assertTrue(math.isfinite(result.best_fitness))

    def test_nonfinite_candidates_are_counted_and_cannot_win(self):
        failed_values = []

        def objective(vector):
            value = (
                vector.square().sum() if vector[0] >= 0.0 else torch.tensor(math.nan)
            )
            failed_values.append(not bool(torch.isfinite(value)))
            return value

        result = differential_evolution(
            objective,
            torch.tensor([0.5], dtype=torch.float64),
            DEConfig(population_size=8, generations=3, seed=12),
        )
        self.assertGreater(result.nonfinite_evaluations, 0)
        self.assertEqual(result.nonfinite_evaluations, sum(failed_values))
        self.assertEqual(result.fitness_evaluations, len(failed_values))
        self.assertTrue(math.isfinite(result.best_fitness))
        self.assertGreaterEqual(result.best_vector[0].item(), 0.0)
        self.assertTrue(
            all(not math.isnan(row["std_fitness"]) for row in result.history)
        )

    def test_all_nonfinite_initial_population_raises(self):
        calls = []

        def objective(vector):
            calls.append(vector)
            return -math.inf

        with self.assertRaisesRegex(ValueError, "no finite"):
            differential_evolution(
                objective, torch.zeros(2), DEConfig(population_size=4, generations=1)
            )
        self.assertEqual(len(calls), 4)

    def test_objective_errors_and_vector_outputs_are_not_swallowed(self):
        def broken_objective(vector):
            raise RuntimeError("residual implementation failed")

        with self.assertRaisesRegex(RuntimeError, "residual implementation failed"):
            differential_evolution(broken_objective, torch.zeros(2), DEConfig())
        with self.assertRaisesRegex(ValueError, "real scalar"):
            differential_evolution(lambda x: x.square(), torch.zeros(2), DEConfig())

    def test_invalid_config_and_initial_vectors_raise(self):
        for kwargs in (
            {"population_size": 3},
            {"population_size": 4.5},
            {"generations": 0},
            {"generations": True},
            {"seed": -1},
            {"lower_bound": math.nan},
            {"upper_bound": math.inf},
            {"lower_bound": 1.0, "upper_bound": 1.0},
            {"control_adaptation": "unknown"},
            {"initialization": "unknown"},
            {"adaptation_probability": -0.1},
            {"adaptation_probability": 1.1},
            {"adaptation_probability": math.nan},
            {"initial_spread": 0.0},
            {"initial_spread": math.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DEConfig(**kwargs)
        for initial in (
            torch.tensor([]),
            torch.zeros(2, 2),
            torch.tensor([1, 2]),
            torch.tensor([math.nan]),
            torch.tensor([1.01]),
        ):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                differential_evolution(lambda x: x.square().sum(), initial, DEConfig())


if __name__ == "__main__":
    unittest.main()
