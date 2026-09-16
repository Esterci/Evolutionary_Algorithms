"""Layer configuration, checkpoint reconstruction and derivative smoke checks."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from methods.pinn_model import (BurgersProblem, NormalizedFisiocomPINN,
                                build_trainer, resolve_activations)
from runners.train_pinn_de_adam import parse_args
from fisiocomPinn.Net import FullyConnectedNetwork


class ActivationConfigurationTests(unittest.TestCase):
    def test_cli_canonical_names_and_invalid_options(self):
        args = parse_args(['--hidden-sizes', '4', '5', '4', '--activation-functions',
                           'Tanh', 'ReLu', 'Tanh', '--adaptation-probability', '0.3'])
        self.assertEqual(args.activation_functions, ['Tanh', 'ReLU', 'Tanh'])
        self.assertEqual(args.adaptation_probability, .3)
        self.assertEqual(resolve_activations([4, 5]), ['Tanh', 'Tanh'])
        for flags in [ ['--activation-functions', 'Tanh'],
                       ['--activation-functions', 'bad', 'Tanh', 'Tanh'],
                       ['--adaptation-probability', '1.1'] ]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_mixed_network_checkpoint_and_loss_backward(self):
        problem = BurgersProblem(
            dict(h=.5, k=.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, .02]),
            dict(nu=.01, boundary='zero_gradient', mean_velocity_u=.5,
                 mean_velocity_v=.25, perturbation_amplitude=1.), torch.float64)
        names = ['Tanh', 'ReLU', 'Tanh']
        model, trainer = build_trainer(problem, [4, 5, 4], epochs=1, adaptive=False,
                                       activation_functions=names)
        self.assertIsInstance(model.backbone, FullyConnectedNetwork)
        self.assertEqual([type(layer).__name__ for layer in model.backbone.layers],
                         ['Linear', 'Tanh', 'Linear', 'ReLU', 'Linear', 'Tanh', 'Linear'])
        coords = torch.tensor([[.01, .3, .7], [.015, .8, .4]], dtype=torch.float64)
        expected = model(coords)
        self.assertEqual(expected.shape, (2, 2))
        self.assertEqual(expected.dtype, torch.float64)
        restored = NormalizedFisiocomPINN(problem, [4, 5, 4], names).to(dtype=torch.float64)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(coords), expected)
        trainer.optimizer.zero_grad()
        losses = [loss.forward(model) for loss in trainer.losses]
        self.assertTrue(all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        trainer.optimizer.step()
        # The default keeps the old framework state keys and seeded initialization.
        torch.manual_seed(123)
        original = FullyConnectedNetwork(3, 2, [4, 5, 4], dtype=problem.dtype)
        torch.manual_seed(123)
        default = NormalizedFisiocomPINN(problem, [4, 5, 4]).backbone
        for key, value in original.state_dict().items():
            torch.testing.assert_close(default.state_dict()[key], value)
