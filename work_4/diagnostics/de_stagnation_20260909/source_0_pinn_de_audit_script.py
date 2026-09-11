"""Reproduce initialization/adaptation diagnostics on a reduced fixed PINN objective.

These runs inspect implementation behavior, not physical solution accuracy or
algorithm superiority. They deliberately retain the original fixed 10:1:1 loss
formula to isolate DE initialization and control adaptation from loss evolution.
"""

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path("/home/esterci/Repositorios/ppgmc/Evolutionary_Algorithms")
sys.path.insert(0, str(ROOT / "work_4"))
import differential_evolution as de_module
import pinn_model


def main():
    output = Path(tempfile.mkdtemp(prefix="pinn_de_stagnation_audit_", dir="/tmp"))
    source_paths = [
        Path(__file__).resolve(),
        Path(de_module.__file__),
        Path(pinn_model.__file__),
    ]
    source_paths.extend(
        Path(inspect.getfile(cls))
        for cls in (
            pinn_model.Trainer,
            pinn_model.LOSS,
            pinn_model.FullyConnectedNetwork,
        )
    )
    hashes = {}
    for index, source in enumerate(source_paths):
        content = source.read_bytes()
        snapshot = f"source_{index}_{source.name}"
        (output / snapshot).write_bytes(content)
        hashes[str(source)] = dict(
            sha256=hashlib.sha256(content).hexdigest(), snapshot=snapshot
        )
    torch.set_num_threads(1)
    mesh = dict(h=0.5, k=0.01, x_dom=[0, 1.5], y_dom=[0, 1.5], t_dom=[0, 0.02])
    constants = dict(
        nu=0.01,
        boundary="zero_gradient",
        mean_velocity_u=0.5,
        mean_velocity_v=0.25,
        perturbation_amplitude=1.0,
    )
    metadata = dict(
        scope="reduced diagnostic, not a final scientific comparison",
        mesh=mesh,
        constants=constants,
        architecture=[32] * 5,
        dtype="float64",
        device="cpu",
        torch_version=torch.__version__,
        torch_num_threads=1,
        network_seeds=[2026, 2028],
        de_seed=2027,
        initial_condition_weight=10.0,
        boundary_weight=1.0,
        pde_weight=1.0,
        adaptive_framework_loss=False,
        evolve_loss_weights=False,
        collocation="fixed full deterministic 3x3 spatial grid, 2 time intervals",
        fitness="10*MSE_initial + MSE_boundary + MSE_PDE",
        source_files=hashes,
    )
    rows = []
    for seed in metadata["network_seeds"]:
        for initialization, adaptation in [
            ("uniform", "legacy"),
            ("uniform", "jde"),
            ("local", "legacy"),
            ("local", "jde"),
        ]:
            torch.manual_seed(seed)
            problem = pinn_model.BurgersProblem(mesh, constants, torch.float64)
            model, trainer = pinn_model.build_trainer(
                problem, [32] * 5, "cpu", epochs=2, adaptive=False, print_steps=1000
            )
            assert tuple(loss.name for loss in trainer.losses) == (
                "Initial",
                "Boundary",
                "PDE",
            )
            assert trainer.lossesW == [10.0, 1.0, 1.0]
            parameters = tuple(model.parameters())
            initial = torch.cat([p.detach().reshape(-1) for p in parameters])
            evaluated = []

            def objective(vector):
                with torch.no_grad():
                    offset = 0
                    for parameter in parameters:
                        parameter.copy_(
                            vector[offset : offset + parameter.numel()].reshape_as(
                                parameter
                            )
                        )
                        offset += parameter.numel()
                value = sum(
                    weight * float(loss.forward(model).detach())
                    for weight, loss in zip(trainer.lossesW, trainer.losses)
                )
                evaluated.append(value)
                return value

            config = de_module.DEConfig(
                population_size=12,
                generations=8,
                lower_bound=-2,
                upper_bound=2,
                seed=2027,
                initialization=initialization,
                control_adaptation=adaptation,
            )
            result = de_module.differential_evolution(objective, initial, config)
            row = dict(
                network_seed=seed,
                de_config=asdict(config),
                dimension=initial.numel(),
                initialized_network_fitness=evaluated[0],
                best_other_initial_fitness=min(evaluated[1 : config.population_size]),
                initial_fitnesses=evaluated[: config.population_size],
                all_evaluated_fitness=evaluated,
                history=result.history,
                best_final=result.best_fitness,
                fitness_evaluations=result.fitness_evaluations,
                nonfinite_evaluations=result.nonfinite_evaluations,
            )
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        initialization=initialization,
                        adaptation=adaptation,
                        best_start=result.history[0]["best_fitness"],
                        best_final=result.best_fitness,
                        mean_start=result.history[0]["mean_fitness"],
                        mean_final=result.history[-1]["mean_fitness"],
                        accepted=sum(r["accepted_trials"] for r in result.history),
                        control_attempts=sum(
                            r["control_adaptation_attempts"] for r in result.history
                        ),
                        control_accepted=sum(
                            r["control_adaptation_acceptances"] for r in result.history
                        ),
                    )
                ),
                flush=True,
            )
    (output / "diagnostic.json").write_text(
        json.dumps(dict(metadata=metadata, runs=rows), indent=2) + "\n"
    )
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
