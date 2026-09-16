"""Behavioral checks for scheduling and early stopping through real trainers."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.pinn_model import BurgersProblem, build_trainer, train_with_loss_weight_history
from methods.pinn_de_adam import make_adam_trainer
from methods.training_controls import LogarithmicLR, save_training_controls
from runners.train_pinn_de_adam import parse_args
from utils.analyze_paired_study import run_schedule


class TrainingControlTests(unittest.TestCase):
    def test_logarithmic_endpoints_monotonicity_and_state_restore(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=0.001)
        scheduler = LogarithmicLR(optimizer, 5, 0.1)
        rates = []
        for _ in range(5):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], 0.001)
        self.assertAlmostEqual(rates[-1], 0.0001)
        self.assertTrue(all(a > b for a, b in zip(rates, rates[1:])))
        other = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        restored = LogarithmicLR(other, 5)
        other.load_state_dict(optimizer.state_dict())
        restored.load_state_dict(scheduler.state_dict())
        optimizer.step(); scheduler.step()
        other.step(); restored.step()
        self.assertEqual(scheduler.get_last_lr(), restored.get_last_lr())

    def test_single_update_and_invalid_ratios(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        scheduler = LogarithmicLR(optimizer, 1)
        self.assertEqual(scheduler.get_last_lr(), [0.001])
        for ratio in (0, -1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                LogarithmicLR(optimizer, 5, ratio)
        for flag, value in [('--lr-final-ratio', 'nan'),
                            ('--early-stopping-patience', '0'),
                            ('--early-stopping-tolerance', '-1')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, value])

    def test_real_pinn_and_handoff_stop_and_preserve_executed_budget(self):
        problem = BurgersProblem(
            dict(h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02]),
            dict(nu=0.01, boundary='zero_gradient', mean_velocity_u=0.5,
                 mean_velocity_v=0.25, perturbation_amplitude=1.0), torch.float64)
        for adaptive in (False, True):
            for hybrid in (False, True):
                for enabled in (False, True):
                    with self.subTest(adaptive=adaptive, hybrid=hybrid, enabled=enabled):
                        torch.manual_seed(81)
                        config = dict(scheduler='logarithmic', final_ratio=0.1,
                                      early_stopping=enabled, patience=1, tolerance=1e6)
                        _, trainer = build_trainer(problem, (4,), epochs=5, adaptive=adaptive if not hybrid else False,
                                                   training_controls=config if not hybrid else None)
                        if hybrid:
                            trainer = make_adam_trainer(trainer, 5, training_controls=config,
                                                        loss_log_vars=torch.zeros(3) if adaptive else None)
                        with contextlib.redirect_stdout(io.StringIO()):
                            model, history, logs = train_with_loss_weight_history(trainer)
                        expected = 2 if enabled else 5
                        self.assertEqual(trainer.n_epochs_run, expected)
                        self.assertEqual(trainer.stopped_early, enabled)
                        self.assertTrue(all(len(values) == expected for values in history.values()))
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        self.assertEqual(len(logs), expected if adaptive else 0)
                        raw = sum(np.asarray(history[loss.name]) * (1 if adaptive else weight)
                                  for loss, weight in zip(trainer.losses, [1, 1, 1] if adaptive else trainer.lossesW))
                        np.testing.assert_allclose(trainer.monitor_history, raw)
                        with tempfile.TemporaryDirectory() as directory:
                            metadata = save_training_controls(trainer, config, directory, 8 if hybrid else 0)
                            metadata.update(epochs=5, adam_epochs=5)
                            (Path(directory) / 'metadata.json').write_text(json.dumps(metadata))
                            schedule = run_schedule(dict(run=directory, method='pinn-de' if hybrid else 'pinn', budget=999))
                            self.assertEqual(schedule['adam_updates'], expected)
                            self.assertEqual(schedule['actual_evaluations'], expected + (8 if hybrid else 0))
                            with np.load(Path(directory) / 'adam_training_controls.npz') as saved:
                                self.assertEqual(saved['learning_rates'].shape, (expected, 1))
                                self.assertLess(saved['learning_rates'][-1, 0], saved['learning_rates'][0, 0])


if __name__ == '__main__':
    unittest.main()
