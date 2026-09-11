"""Reproduce a counterexample for the shared adaptive objective, without training."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from pinn_model import BurgersProblem, build_trainer

torch.set_num_threads(1)
mesh = json.loads((ROOT / 'config/mesh_properties.json').read_text())
constants = json.loads((ROOT / 'config/constant_properties.json').read_text())
problem = BurgersProblem(mesh, constants, torch.float32)
model, trainer = build_trainer(problem, hidden_sizes=[32, 32, 32], device='cpu', epochs=1)
outputs = [constants['mean_velocity_u'], constants['mean_velocity_v']]
with torch.no_grad():
    for parameter in model.parameters():
        parameter.zero_()
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    layers[-1].bias.copy_(torch.tensor(outputs))
losses = [loss.forward(model) for loss in trainer.losses]
raw = {loss.name: float(value.detach()) for loss, value in zip(trainer.losses, losses)}
rows = []
for k in (0., 5., 10., 20.):
    with torch.no_grad():
        trainer.adaptive_weights.log_vars.copy_(torch.tensor([0., -k, -k]))
    objective, _ = trainer.adaptive_weights(losses)
    rows.append(dict(log_vars=[0., -k, -k], adaptive_objective=float(objective.detach())))
result = dict(
    purpose='Mathematical counterexample for interpreting the shared objective; not a trained run or an experimental outcome',
    constant_outputs=outputs, raw_losses=raw, cases=rows,
    inference='Constant fields satisfy homogeneous Neumann and conservative PDE residuals exactly while missing the sinusoidal initial condition. For s_Initial=0 and s_Boundary=s_PDE=-k, the objective equals Initial MSE-2k; it has no finite lower bound along this path. This does not prove a trained network reaches a constant field.',
)
(ROOT / 'constant_field_objective_check.json').write_text(
    json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
