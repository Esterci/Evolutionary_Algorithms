"""Reduced end-to-end artifact checks; these are not accuracy experiments."""

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_pinn_de_adam import main, parse_args
from train_pinn import main as train_pinn_main
from pinn_model import train_with_loss_weight_history


class DERunnerTests(unittest.TestCase):
    def test_cli_smoke_records_genes_raw_losses_budgets_and_fvm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mesh_properties.json").write_text(
                json.dumps(
                    dict(h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02])
                )
            )
            (root / "constant_properties.json").write_text(
                json.dumps(
                    dict(
                        nu=0.01,
                        boundary="zero_gradient",
                        mean_velocity_u=0.5,
                        mean_velocity_v=0.25,
                        perturbation_amplitude=1.0,
                    )
                )
            )
            for evolved in (False, True):
                with self.subTest(evolved=evolved), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    output = root / str(evolved)
                    arguments = [
                        "--config-directory",
                        str(root),
                        "--output-directory",
                        str(output),
                        "--epochs",
                        "4",
                        "--de-generations",
                        "2",
                        "--population-size",
                        "4",
                        "--hidden-sizes",
                        "4",
                        "--device",
                        "cpu",
                        "--dtype",
                        "float64",
                    ]
                    if not evolved:
                        arguments += [
                            "--no-evolve-loss-weights",
                            "--initialization",
                            "uniform",
                            "--control-adaptation",
                            "legacy",
                        ]
                    main(arguments)
                    (run,) = output.iterdir()
                    metadata = json.loads((run / "metadata.json").read_text())
                    self.assertEqual(metadata["status"], "complete")
                    self.assertEqual(metadata["de"]["fitness_evaluations"], 12)
                    self.assertEqual(metadata["de_adaptive_loss_weights"], evolved)
                    self.assertEqual(
                        metadata["genome_dimension"],
                        metadata["network_dimension"] + (3 if evolved else 0),
                    )
                    with np.load(run / "de_history.npz") as history:
                        self.assertEqual(
                            history["fitness_evaluations"].tolist(), [4, 8, 12]
                        )
                        self.assertTrue((np.diff(history["best_fitness"]) <= 0).all())
                        self.assertAlmostEqual(
                            history["selected_reference_fitness"][-1],
                            metadata["de_best_reference_fitness"],
                        )
                        for name, coefficient in metadata["de_best_loss_weights"].items():
                            self.assertAlmostEqual(
                                history[f"best_weight_{name}"][-1], coefficient
                            )
                    with np.load(run / "losses.npz") as history:
                        for name, value in metadata["de_best_losses"].items():
                            self.assertEqual(len(history[name]), 2)
                            self.assertEqual(history[name][0], value)
                    with np.load(run / "comparison.npz") as comparison:
                        self.assertTrue(
                            all(
                                np.isfinite(comparison[name]).all()
                                for name in comparison.files
                            )
                        )
                    population = torch.load(run / "de_population.pt", weights_only=True)
                    self.assertEqual(
                        population["best_vector"].numel(), metadata["genome_dimension"]
                    )
                    self.assertEqual(metadata['loss_weight_names'], ['Initial', 'Boundary', 'PDE'])
                    self.assertEqual(metadata['adaptive_loss'], evolved)
                    self.assertIsNone(metadata['loss_log_variance_bounds'])
                    self.assertEqual(population['bound_mask'][-3:].tolist(),
                                     [not evolved] * 3)
                    if evolved:
                        names = metadata['loss_weight_names']
                        genes = population['best_vector'][-3:].tolist()
                        self.assertEqual(genes, list(metadata['de_best_loss_log_vars'].values()))
                        for name, gene in zip(names, genes):
                            self.assertAlmostEqual(metadata['de_best_loss_weights'][name], math.exp(-gene))
                        self.assertEqual(metadata['reference_loss_weights'], dict.fromkeys(names, 1.0))
                        state = torch.load(run / 'adaptive_weights.pt', weights_only=True)
                        final_genes = state['log_vars'].cpu().numpy()
                        np.testing.assert_array_equal(final_genes, list(metadata['final_loss_log_vars'].values()))
                        with np.load(run / 'adam_loss_weights.npz') as history:
                            self.assertEqual(history['loss_names'].tolist(), names)
                            self.assertEqual(history['log_vars'].shape, (2, 3))
                            self.assertTrue(np.all(history['log_vars'][0] != np.asarray(genes)))
                            np.testing.assert_array_equal(history['log_vars'][-1], final_genes)
                            np.testing.assert_allclose(history['weights'], np.exp(-history['log_vars']))
                        optimizer = torch.load(run / 'adam_optimizer.pt', weights_only=True)
                        self.assertEqual(len(optimizer['state']), len(metadata['genome_layout']) - 3 + 1)
                    final_score = sum(
                        metadata['final_loss_weights'][name] * value
                        for name, value in metadata['final_losses'].items()
                    ) + sum(metadata['final_loss_log_vars'].values())
                    self.assertAlmostEqual(final_score, metadata['final_regularized_fitness'])

    def test_cli_defaults_and_invalid_weight_search_parameters(self):
        args = parse_args([])
        self.assertTrue(args.evolve_loss_weights)
        self.assertEqual(args.initialization, "local")
        self.assertEqual(args.control_adaptation, "jde")
        for arguments in (
            ["--log-weight-init-bound", "nan"],
            ["--log-weight-init-bound", "0"],
            ["--adaptation-probability", "2"],
            ["--initial-spread", "-1"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)

    def test_old_pde_flags_are_explicit_aliases_and_old_bounds_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()) as messages:
            self.assertTrue(parse_args(['--evolve-pde-weights']).evolve_loss_weights)
            self.assertFalse(parse_args(['--no-evolve-pde-weights']).evolve_loss_weights)
        self.assertIn('Initial, Boundary and combined PDE', messages.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as messages:
            with self.assertRaises(SystemExit):
                parse_args(['--log-weight-bound', '4'])
        self.assertIn('unconstrained', messages.getvalue())

    def test_adam_only_runner_preserves_matching_adaptive_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'mesh_properties.json').write_text(json.dumps(dict(
                h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02])))
            (root / 'constant_properties.json').write_text(json.dumps(dict(
                nu=0.01, boundary='zero_gradient', mean_velocity_u=0.5,
                mean_velocity_v=0.25, perturbation_amplitude=1.0)))
            output = root / 'adam_only'
            with contextlib.redirect_stdout(io.StringIO()):
                train_pinn_main(['--config-directory', str(root), '--output-directory', str(output),
                                 '--epochs', '2', '--hidden-sizes', '4', '--device', 'cpu',
                                 '--dtype', 'float64'])
            run, = output.iterdir()
            metadata = json.loads((run / 'metadata.json').read_text())
            self.assertTrue(metadata['adaptive_loss'])
            self.assertEqual(metadata['loss_weight_names'], ['Initial', 'Boundary', 'PDE'])
            self.assertEqual(metadata['initial_loss_log_vars'], dict(Initial=0, Boundary=0, PDE=0))
            state = torch.load(run / 'adaptive_weights.pt', weights_only=True)
            self.assertEqual(state['log_vars'].dtype, torch.float64)
            with np.load(run / 'adam_loss_weights.npz') as history:
                self.assertEqual(history['log_vars'].shape, (2, 3))
                np.testing.assert_array_equal(history['log_vars'][-1], state['log_vars'].numpy())
                np.testing.assert_allclose(history['weights'][-1],
                                            list(metadata['final_loss_weights'].values()))

    def test_adam_coefficient_overflow_records_failed_run_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'mesh_properties.json').write_text(json.dumps(dict(
                h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02])))
            (root / 'constant_properties.json').write_text(json.dumps(dict(
                nu=0.01, boundary='zero_gradient', mean_velocity_u=0.5,
                mean_velocity_v=0.25, perturbation_amplitude=1.0)))

            def overflow_after_training(trainer):
                result = train_with_loss_weight_history(trainer)
                with torch.no_grad():
                    trainer.adaptive_weights.log_vars.fill_(-1000)
                return result

            output = root / 'failed'
            with patch('train_pinn_de_adam.train_with_loss_weight_history',
                       side_effect=overflow_after_training), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(FloatingPointError, 'adaptive coefficients'):
                    main(['--config-directory', str(root), '--output-directory', str(output),
                          '--epochs', '2', '--de-generations', '1', '--population-size', '4',
                          '--hidden-sizes', '4', '--device', 'cpu', '--dtype', 'float64'])
            run, = output.iterdir()
            metadata = json.loads((run / 'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['error_type'], 'FloatingPointError')
            self.assertNotIn('final_loss_weights', metadata)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
