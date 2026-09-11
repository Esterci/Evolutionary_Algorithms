"""Mathematical and optimizer-handoff checks on tiny conservative Burgers PINNs.

Run from the repository root with::

    python -m unittest discover -s work_4/tests -p 'test_pinn_de_adam.py'

These reduced checks validate implementation behavior, not solution accuracy.
"""

import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from differential_evolution import DEConfig, differential_evolution
from pinn_de_adam import PINNFitness, make_adam_trainer
from fisiocomPinn.Trainer import AdaptiveLossWeights
from pinn_model import (
    BurgersProblem,
    build_trainer,
    burgers_residual,
    zero_gradient_boundary_residual,
)


class PolynomialVelocity(torch.nn.Module):
    """Manufactured field with nonzero first and second spatial derivatives."""

    def forward(self, coordinates):
        t, x, y = coordinates.split(1, dim=1)
        u = t.square() + x.square() + y.square()
        v = t.square() + 2 * x.square() + 3 * y.square()
        return torch.cat((u, v), dim=1)


def tiny_trainer(dtype=torch.float64, adaptive=False, split_pde=False):
    mesh = dict(h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02])
    constants = dict(
        nu=0.01,
        boundary="zero_gradient",
        mean_velocity_u=0.5,
        mean_velocity_v=0.25,
        perturbation_amplitude=1.0,
    )
    torch.manual_seed(81)
    problem = BurgersProblem(mesh, constants, dtype)
    _, trainer = build_trainer(
        problem,
        hidden_sizes=(4,),
        device="cpu",
        epochs=2,
        adaptive=adaptive,
        print_steps=1000,
        split_pde=split_pde,
    )
    return trainer


class MathematicalResidualTests(unittest.TestCase):
    def test_conservative_residual_matches_manufactured_polynomial(self):
        coordinates = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.8, 0.4], [0.8, 0.6, 0.9]],
            dtype=torch.float64,
            requires_grad=True,
        )
        t, x, y = coordinates.split(1, dim=1)
        u = t.square() + x.square() + y.square()
        v = t.square() + 2 * x.square() + 3 * y.square()
        viscosity = 0.07
        # Expand both conservative fluxes independently of the residual code.
        expected_u = 2 * t + 2 * x * u + 2 * y * v + 6 * y * u - 4 * viscosity
        expected_v = 2 * t + 2 * x * v + 4 * x * u + 6 * y * v - 10 * viscosity
        actual = burgers_residual(coordinates, PolynomialVelocity(), viscosity)
        self.assertEqual(actual.shape, (3, 2))
        torch.testing.assert_close(
            actual, torch.cat((expected_u, expected_v), dim=1), rtol=1e-13, atol=1e-13
        )

    def test_boundary_operator_selects_spatial_normal_derivatives(self):
        coordinates = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.8, 0.4], [0.8, 0.6, 0.9]],
            dtype=torch.float64,
            requires_grad=True,
        )
        derivative_columns = torch.tensor([1, 2, 1])
        expected = torch.tensor(
            [[0.4, 0.8], [0.8, 2.4], [1.2, 2.4]], dtype=torch.float64
        )
        actual = zero_gradient_boundary_residual(
            (coordinates, derivative_columns), PolynomialVelocity()
        )
        self.assertEqual(actual.shape, (3, 2))
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


class PINNFitnessTests(unittest.TestCase):
    def test_genome_copy_preserves_parameters_buffers_and_ownership(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                trainer = tiny_trainer(dtype)
                fitness = PINNFitness(trainer)
                parameter_ids = [id(p) for p in trainer.model.parameters()]
                buffers = {
                    name: value.clone() for name, value in trainer.model.named_buffers()
                }
                original = fitness.vector()
                candidate = original + 0.03125
                expected = candidate.clone()
                fitness.load(candidate)
                candidate.zero_()
                torch.testing.assert_close(fitness.vector(), expected, rtol=0, atol=0)
                self.assertEqual(
                    parameter_ids, [id(p) for p in trainer.model.parameters()]
                )
                for name, value in trainer.model.named_buffers():
                    torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
                self.assertEqual(fitness.vector().dtype, dtype)
                self.assertEqual(fitness.vector().device.type, "cpu")
                self.assertEqual(
                    fitness.dimension,
                    sum(p.numel() for p in trainer.model.parameters()),
                )
                # A returned snapshot must not expose writable parameter storage.
                snapshot = fitness.vector()
                snapshot.zero_()
                torch.testing.assert_close(fitness.vector(), expected, rtol=0, atol=0)

    def test_deterministic_weighted_fitness_keeps_input_autograd_enabled(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                trainer = tiny_trainer(dtype)
                fitness = PINNFitness(trainer)
                candidate = fitness.vector()
                # DE may run in no_grad; PDE input derivatives still require graphs.
                with torch.no_grad():
                    first = fitness(candidate)
                    second = fitness(candidate)
                    terms = fitness.terms()
                self.assertEqual(first, second)
                self.assertEqual(
                    first, 10 * terms["Initial"] + terms["Boundary"] + terms["PDE"]
                )
                self.assertTrue(
                    torch.isfinite(torch.tensor(list(terms.values()))).all()
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in trainer.model.parameters()
                    )
                )

    def test_adaptive_weights_and_wrong_genome_layout_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "adaptive=False"):
            PINNFitness(tiny_trainer(adaptive=True))
        fitness = PINNFitness(tiny_trainer())
        candidate = fitness.vector()
        with self.assertRaisesRegex(ValueError, "genome shape"):
            fitness.load(candidate.reshape(1, -1))
        with self.assertRaisesRegex(ValueError, "device and dtype"):
            fitness.load(candidate.float())
        torch.testing.assert_close(fitness.vector(), candidate, rtol=0, atol=0)


class EvolvedLossWeightTests(unittest.TestCase):
    def test_three_genes_roundtrip_initial_bounds_and_parameter_ownership(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                trainer = tiny_trainer(dtype)
                objective = PINNFitness(trainer, evolve_loss_weights=True)
                parameter_ids = [id(p) for p in trainer.model.parameters()]
                candidate = objective.vector()
                self.assertEqual(objective.dimension, objective.network_dimension + 3)
                self.assertEqual(objective.reference_weights, (1.0, 1.0, 1.0))
                candidate[-3:] = torch.tensor([-1.0, 2.0, 0.5], dtype=dtype)
                expected = candidate.clone()
                objective.load(candidate)
                candidate.zero_()
                torch.testing.assert_close(objective.vector(), expected, rtol=0, atol=0)
                self.assertEqual(parameter_ids, [id(p) for p in trainer.model.parameters()])
                self.assertEqual(
                    objective.weights, tuple(torch.exp(-expected[-3:]).tolist())
                )
                snapshot = objective.loss_log_vars
                snapshot.zero_()
                torch.testing.assert_close(
                    objective.loss_log_vars, expected[-3:], rtol=0, atol=0
                )
                lower, upper = objective.bounds(-2.0, 2.0, 4.0)
                self.assertEqual(lower.dtype, dtype)
                self.assertEqual(upper.shape, expected.shape)
                self.assertTrue((lower[:-3] == -2.0).all())
                self.assertTrue((upper[:-3] == 2.0).all())
                self.assertTrue((lower[-3:] == -4.0).all())
                self.assertTrue((upper[-3:] == 4.0).all())
                self.assertEqual(
                    [item["name"] for item in objective.layout[-3:]],
                    ["loss_log_var_Initial", "loss_log_var_Boundary", "loss_log_var_PDE"],
                )
                self.assertEqual(objective.layout[-1]["stop"], objective.dimension)
                # These are initialization bounds, not adaptive-weight constraints.
                expected[-3:] = torch.tensor([5.0, -6.0, 7.0], dtype=dtype)
                self.assertTrue(math.isfinite(objective(expected)))

    def test_adaptive_formula_exactly_matches_framework_for_all_three_terms(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                trainer = tiny_trainer(dtype)
                objective = PINNFitness(trainer, evolve_loss_weights=True)
                baseline = AdaptiveLossWeights(3).to(dtype=dtype)
                losses = [loss.forward(trainer.model) for loss in trainer.losses]
                raw = {
                    loss.name: float(value.detach())
                    for loss, value in zip(trainer.losses, losses)
                }
                candidate = objective.vector()
                for genes in ([0.0, 0.0, 0.0], [-1.0, 0.5, 2.0], [0.75, -0.25, -1.5]):
                    with self.subTest(genes=genes):
                        candidate[-3:] = torch.tensor(genes, dtype=dtype)
                        objective.load(candidate)
                        with torch.no_grad():
                            baseline.log_vars.copy_(candidate[-3:])
                            expected, _ = baseline(losses)
                        self.assertEqual(objective.score(raw), float(expected))
                # Initial-condition coefficient 10 is ignored by both adaptive methods.
                candidate[-3:] = 0
                objective.load(candidate)
                self.assertEqual(objective.weights, (1.0, 1.0, 1.0))
                tolerance = 4 * torch.finfo(dtype).eps * max(1, sum(raw.values()))
                self.assertAlmostEqual(objective.score(raw), sum(raw.values()), delta=tolerance)
                self.assertNotEqual(
                    objective.score(raw),
                    10 * raw["Initial"] + raw["Boundary"] + raw["PDE"],
                )
                zero_score = objective.score(raw)
                for index in range(3):
                    candidate[-3:] = 0
                    candidate[-3 + index] = 1.0
                    objective.load(candidate)
                    self.assertNotEqual(objective.score(raw), zero_score)

    def test_pde_is_one_mse_over_both_residual_columns(self):
        objective = PINNFitness(tiny_trainer(), evolve_loss_weights=True)
        pde = objective.losses[-1]
        coordinates, targets = pde.batch_generator(pde.batch_size, pde.device)
        residual = pde.eval_func(coordinates, objective.model, *pde.eval_args)
        self.assertEqual(residual.shape, (pde.batch_size, 2))
        self.assertEqual(targets.shape, residual.shape)
        component_u = residual[:, 0].square().mean()
        component_v = residual[:, 1].square().mean()
        self.assertAlmostEqual(
            objective.terms()["PDE"], float(0.5 * (component_u + component_v)), places=13
        )
        self.assertEqual([loss.name for loss in objective.losses], ["Initial", "Boundary", "PDE"])

    def test_regularization_and_reference_loss_are_consistent_for_every_term(self):
        objective = PINNFitness(tiny_trainer(), evolve_loss_weights=True)
        candidate = objective.vector()
        raw = dict(Initial=1.0, Boundary=1.0, PDE=1.0)
        optimum = objective.score(raw)
        for index in range(3):
            for log_var in (-4.0, -0.5, 0.5, 4.0):
                candidate[-3:] = 0
                candidate[-3 + index] = log_var
                objective.load(candidate)
                self.assertGreater(objective.score(raw), optimum)
                self.assertEqual(objective.reference_fitness(raw), 3.0)
        # For positive L_i, each weight optimum is s_i = log(L_i).
        raw = dict(Initial=math.exp(1.0), Boundary=math.exp(-1.0), PDE=math.exp(0.5))
        candidate[-3:] = torch.tensor([1.0, -1.0, 0.5], dtype=candidate.dtype)
        objective.load(candidate)
        optimum = objective.score(raw)
        for offset in (-0.25, 0.25):
            perturbed = candidate.clone()
            perturbed[-3:] += offset
            objective.load(perturbed)
            self.assertGreater(objective.score(raw), optimum)

    def test_reload_preserves_deterministic_fitness_and_raw_diagnostics(self):
        objective = PINNFitness(tiny_trainer(), evolve_loss_weights=True)
        first = objective.vector()
        first[-3:] = torch.tensor([-0.5, 0.25, 0.75], dtype=first.dtype)
        with torch.no_grad():
            cached = objective(first)
            expected = objective.last_diagnostics.copy()
            alternative = first.clone()
            alternative[-3:] = torch.tensor([1.25, -1.0, -0.5], dtype=first.dtype)
            objective(alternative)
            self.assertNotEqual(objective.last_diagnostics["fitness"], cached)
            self.assertEqual(
                objective.last_diagnostics["reference_fitness"],
                expected["reference_fitness"],
            )
            self.assertEqual(objective(first), cached)
        self.assertEqual(objective.last_diagnostics, expected)
        for name, gene, weight in zip(
            ("Initial", "Boundary", "PDE"), first[-3:], torch.exp(-first[-3:])
        ):
            self.assertEqual(expected[f"loss_log_var_{name}"], float(gene))
            self.assertEqual(expected[f"weight_{name}"], float(weight))
            self.assertIn(f"loss_{name}", expected)
        self.assertTrue(all(parameter.grad is None for parameter in objective.parameters))

    def test_nonfinite_candidates_and_exponential_overflow_are_failed_evaluations(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                objective = PINNFitness(tiny_trainer(dtype), evolve_loss_weights=True)
                original = objective.vector()
                for index in (0, -3, -2, -1):
                    for invalid in (math.inf, -math.inf, math.nan):
                        candidate = original.clone()
                        candidate[index] = invalid
                        self.assertEqual(objective(candidate), math.inf)
                        self.assertEqual(
                            objective.last_diagnostics,
                            dict(fitness=math.inf, reference_fitness=math.inf),
                        )
                        torch.testing.assert_close(objective.vector(), original, rtol=0, atol=0)
                candidate = original.clone()
                candidate[-1] = -1000.0
                self.assertEqual(objective(candidate), math.inf)
                self.assertEqual(objective.weights[-1], math.inf)
                self.assertEqual(
                    objective.score(dict(Initial=1.0, Boundary=1.0, PDE=0.0)), math.inf
                )
                self.assertTrue(math.isfinite(objective(original)))

    def test_adaptation_requires_grouped_terms_and_finite_initial_bounds(self):
        with self.assertRaisesRegex(ValueError, "Initial, Boundary, PDE"):
            PINNFitness(tiny_trainer(split_pde=True), evolve_loss_weights=True)
        objective = PINNFitness(tiny_trainer(), evolve_loss_weights=True)
        for bounds in (
            (-1.0, 1.0, 0.0), (-1.0, 1.0, math.inf),
            (1.0, -1.0, 4.0), (-1.0, 1.0, 1000.0),
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                objective.bounds(*bounds)

    def test_de_search_varies_all_three_loss_genes_and_preserves_best(self):
        objective = PINNFitness(tiny_trainer(), evolve_loss_weights=True)
        initial = objective.vector()
        evaluated = []

        def record_candidate(candidate):
            evaluated.append(candidate.detach().clone())
            return objective(candidate)

        result = differential_evolution(
            record_candidate, initial,
            DEConfig(
                population_size=4, generations=2, lower_bound=-2.0,
                upper_bound=2.0, seed=2027, initialization="local",
            ),
            bounds=objective.bounds(-2.0, 2.0),
        )
        self.assertEqual(result.fitness_evaluations, 12)
        self.assertEqual(result.nonfinite_evaluations, 0)
        self.assertTrue((result.population[:, -3:].std(dim=0) > 0).all())
        initial_genes = torch.stack(evaluated[:4])[:, -3:]
        trial_genes = torch.stack(evaluated[4:8])[:, -3:]
        self.assertTrue((trial_genes != initial_genes).any(dim=0).all())
        self.assertEqual(objective(result.best_vector), result.best_fitness)
        self.assertEqual(result.history[-1]["best_fitness"], result.best_fitness)


class AdamHandoffTests(unittest.TestCase):
    def test_baseline_pinn_and_handoff_match_from_identical_adaptive_state(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                baseline = tiny_trainer(dtype, adaptive=True)
                de_trainer = tiny_trainer(dtype)
                objective = PINNFitness(de_trainer, evolve_loss_weights=True)
                continuation = make_adam_trainer(
                    de_trainer, epochs=2, loss_log_vars=objective.loss_log_vars,
                    print_steps=1000,
                )
                self.assertEqual(baseline.adaptive_weights.log_vars.dtype, dtype)
                self.assertEqual(baseline.adaptive_weights.log_vars.device.type, "cpu")
                self.assertEqual(
                    [loss.name for loss in baseline.losses],
                    [loss.name for loss in continuation.losses],
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    _, baseline_history = baseline.train()
                    _, continuation_history = continuation.train()
                self.assertEqual(baseline_history, continuation_history)
                for left, right in zip(
                    baseline.model.parameters(), continuation.model.parameters()
                ):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                torch.testing.assert_close(
                    baseline.adaptive_weights.log_vars,
                    continuation.adaptive_weights.log_vars,
                    rtol=0, atol=0,
                )
                self.assertTrue((baseline.adaptive_weights.log_vars != 0).all())

    def test_all_three_best_genes_continue_adapting_with_exact_framework_adam_step(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                de_trainer = tiny_trainer(dtype)
                objective = PINNFitness(de_trainer, evolve_loss_weights=True)
                expected_trainer = tiny_trainer(dtype)
                expected_objective = PINNFitness(expected_trainer, evolve_loss_weights=True)
                best = objective.vector()
                best[:-3] += 0.03125
                best[-3:] = torch.tensor([-1.0, 0.5, 0.25], dtype=dtype)
                objective(best)
                raw_before = objective.terms()
                objective(best + 0.125)
                objective.load(best)
                supplied_genes = objective.loss_log_vars
                trainer = make_adam_trainer(
                    de_trainer, epochs=1, print_steps=1000, loss_log_vars=supplied_genes,
                )
                supplied_genes.zero_()
                self.assertTrue(trainer.adaptive)
                self.assertEqual(trainer.lossesW, [])
                self.assertEqual(len(trainer.optimizer.state), 0)
                self.assertIsInstance(trainer.adaptive_weights, AdaptiveLossWeights)
                self.assertEqual(trainer.adaptive_weights.log_vars.dtype, dtype)
                torch.testing.assert_close(
                    trainer.adaptive_weights.log_vars, best[-3:], rtol=0, atol=0
                )
                optimizer_parameter_ids = {
                    id(p) for group in trainer.optimizer.param_groups
                    for p in group["params"]
                }
                expected_parameter_ids = {
                    id(p) for p in trainer.model.parameters()
                } | {id(trainer.adaptive_weights.log_vars)}
                self.assertEqual(optimizer_parameter_ids, expected_parameter_ids)
                self.assertTrue(all(
                    left is right for left, right in zip(trainer.losses, de_trainer.losses)
                ))
                expected_objective.load(best)
                expected_weights = AdaptiveLossWeights(3).to(dtype=dtype)
                with torch.no_grad():
                    expected_weights.log_vars.copy_(best[-3:])
                expected_optimizer = torch.optim.Adam(
                    list(expected_trainer.model.parameters()) + list(expected_weights.parameters()),
                    lr=1e-3, betas=(0.9, 0.999),
                )
                weight_module = trainer.adaptive_weights
                for step in (1, 2):
                    expected_loss, _ = expected_weights([
                        loss.forward(expected_trainer.model)
                        for loss in expected_trainer.losses
                    ])
                    if step == 1:
                        self.assertEqual(float(expected_loss.detach()), objective.score(raw_before))
                    expected_optimizer.zero_grad()
                    expected_loss.backward()
                    expected_optimizer.step()
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, history = trainer.train()
                    self.assertIs(trainer.adaptive_weights, weight_module)
                    torch.testing.assert_close(
                        objective.vector()[:-3], expected_objective.vector()[:-3],
                        rtol=0, atol=0,
                    )
                    torch.testing.assert_close(
                        trainer.adaptive_weights.log_vars, expected_weights.log_vars,
                        rtol=0, atol=0,
                    )
                    self.assertTrue((trainer.adaptive_weights.log_vars != best[-3:]).all())
                    self.assertFalse(torch.equal(objective.vector()[:-3], best[:-3]))
                    if step == 1:
                        self.assertEqual(
                            history, {name: [value] for name, value in raw_before.items()}
                        )
                    self.assertTrue(all(
                        state["step"].item() == step
                        for state in trainer.optimizer.state.values()
                    ))
                # Adam owns an independent module; DE's cached genotype is preserved.
                torch.testing.assert_close(objective.loss_log_vars, best[-3:], rtol=0, atol=0)

    def test_adaptive_handoff_rejects_wrong_genome_lengths_and_nonfinite_genes(self):
        trainer = tiny_trainer()
        for genes in ([1.0, 2.0], [0.0, math.nan, 0.0], [0.0, 0.0, math.inf]):
            with self.subTest(genes=genes), self.assertRaisesRegex(ValueError, "three finite"):
                make_adam_trainer(trainer, epochs=1, loss_log_vars=genes)

    def test_real_de_then_adam_uses_best_genome_with_exact_stage_budgets(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                de_trainer = tiny_trainer(dtype)
                fitness = PINNFitness(de_trainer)
                initial = fitness.vector()
                initial_fitness = fitness(initial)
                evaluated = []

                def counted_fitness(candidate):
                    value = fitness(candidate)
                    evaluated.append(value)
                    return value

                result = differential_evolution(
                    counted_fitness,
                    initial,
                    DEConfig(
                        population_size=4,
                        generations=2,
                        lower_bound=-2,
                        upper_bound=2,
                        seed=2027,
                    ),
                )
                self.assertEqual(result.fitness_evaluations, 12)
                self.assertEqual(len(evaluated), 12)
                self.assertEqual(result.nonfinite_evaluations, 0)
                self.assertEqual(len(result.history), 3)
                best_history = [row["best_fitness"] for row in result.history]
                self.assertTrue(
                    all(
                        after <= before
                        for before, after in zip(best_history, best_history[1:])
                    )
                )
                fitness.load(result.best_vector)
                terms_at_best = fitness.terms()
                reevaluated = sum(
                    weight * terms_at_best[loss.name]
                    for loss, weight in zip(fitness.losses, fitness.weights)
                )
                tolerance = 4 * torch.finfo(dtype).eps * max(1.0, abs(reevaluated))
                self.assertAlmostEqual(
                    reevaluated, result.best_fitness, delta=tolerance
                )
                self.assertLessEqual(reevaluated, initial_fitness + tolerance)
                adam_trainer = make_adam_trainer(
                    de_trainer, epochs=1, lr=1e-3, print_steps=1000
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    _, history = adam_trainer.train()
                for name, value in terms_at_best.items():
                    self.assertEqual(history[name], [value])
                self.assertTrue(torch.isfinite(fitness.vector()).all())
                self.assertFalse(torch.equal(fitness.vector(), result.best_vector))
                self.assertTrue(
                    all(
                        state["step"].item() == 1
                        for state in adam_trainer.optimizer.state.values()
                    )
                )

    def test_restored_best_is_adam_start_and_same_parameters_are_updated(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                de_trainer = tiny_trainer(dtype)
                fitness = PINNFitness(de_trainer)
                parameter_ids = [id(p) for p in de_trainer.model.parameters()]
                best = fitness.vector()
                expected_terms = fitness.terms()
                # Evaluating the final DE trial leaves the model at that trial,
                # which need not be the best surviving individual.
                fitness(best + 0.125)
                self.assertFalse(torch.equal(fitness.vector(), best))
                fitness.load(best)
                adam_trainer = make_adam_trainer(
                    de_trainer, epochs=2, lr=1e-3, print_steps=1000
                )
                optimizer = adam_trainer.optimizer
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(len(optimizer.state), 0)
                self.assertIs(adam_trainer.model, de_trainer.model)
                self.assertEqual(adam_trainer.lossesW, [10.0, 1.0, 1.0])
                self.assertTrue(
                    all(
                        left is right
                        for left, right in zip(adam_trainer.losses, de_trainer.losses)
                    )
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    trained_model, history = adam_trainer.train()
                self.assertIs(trained_model, de_trainer.model)
                self.assertIs(adam_trainer.optimizer, optimizer)
                self.assertEqual(
                    parameter_ids, [id(p) for p in trained_model.parameters()]
                )
                for name, expected in expected_terms.items():
                    self.assertEqual(history[name][0], expected)
                    self.assertEqual(len(history[name]), 2)
                self.assertFalse(torch.equal(fitness.vector(), best))
                for parameter in trained_model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertEqual(optimizer.state[parameter]["step"].item(), 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
