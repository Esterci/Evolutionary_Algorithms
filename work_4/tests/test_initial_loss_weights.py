"""Verify shared initial coefficients, precision and DE-to-Adam continuity."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from methods.loss_weight_config import initial_loss_coefficients, loss_log_vars
from methods.pinn_model import build_trainer
from methods.pinn_de_adam import PINNFitness, make_adam_trainer
from work_4.tests.test_pinn_de_adam import tiny_trainer
from runners.train_pinn_de_adam import parse_args


class InitialLossWeightsTests(unittest.TestCase):
    def test_defaults_split_and_overrides(self):
        self.assertEqual(initial_loss_coefficients(), dict(Initial=1000, Boundary=1000, PDE=1000))
        self.assertEqual(initial_loss_coefficients(split_pde=True),
                         dict(Initial=1000, Boundary=1000, PDE_u=500, PDE_v=500))
        args = parse_args(['--initial-loss-weights', '20', '30', '40'])
        self.assertEqual(args.initial_loss_weights, dict(Initial=20, Boundary=30, PDE=40))
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                initial_loss_coefficients(dict(Initial=value, Boundary=1, PDE=1))

    def test_adam_and_de_start_from_registered_coefficients(self):
        for dtype in (torch.float32, torch.float64):
            for split in (False, True):
                trainer = tiny_trainer(dtype, adaptive=True, split_pde=split)
                expected = torch.tensor(list(initial_loss_coefficients(split_pde=split).values()), dtype=dtype)
                torch.testing.assert_close(torch.exp(-trainer.adaptive_weights.log_vars), expected)
                self.assertEqual(len(trainer.losses), len(expected))
            registered = tiny_trainer(dtype)
            fitness = PINNFitness(registered, evolve_loss_weights=True)
            torch.testing.assert_close(torch.exp(-fitness.loss_log_vars), torch.full((3,), 1000., dtype=dtype))
            low, high = fitness.bounds(-2, 2)
            torch.testing.assert_close((low[-3:] + high[-3:]) / 2, fitness.loss_log_vars)
            # The population box remains anchored even after loading evolved genes.
            vector = fitness.vector()
            vector[-3:] += .2
            fitness.load(vector)
            new_low, new_high = fitness.bounds(-2, 2)
            torch.testing.assert_close(low, new_low)
            torch.testing.assert_close(high, new_high)
            adam = make_adam_trainer(registered, epochs=1, loss_log_vars=fitness.loss_log_vars)
            torch.testing.assert_close(adam.adaptive_weights.log_vars, vector[-3:])
            fixed = make_adam_trainer(registered, epochs=1)
            self.assertEqual(fixed.lossesW, [1000., 1000., 1000.])
