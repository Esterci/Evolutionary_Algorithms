"""Train the notebook PINN and store a complete FVM comparison.

Examples (from the repository root)::

    python work_4/train_pinn.py --epochs 5000
    python work_4/train_pinn.py --epochs 2 --hidden-sizes 8 8 --device cpu
    python work_4/train_pinn.py --checkpoint work_4/pinn_sim/pinn_model_TIMESTAMP.pt

Checkpoint mode uses the accompanying notebook metadata instead of current JSON
settings. The existing serial FVM solver is run on exactly the same configuration.
FVM cell values are compared with PINN point predictions at cell centers; this is
an approximate numerical reference, not an analytical ground truth.
"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from pinn_model import (
    BASE_DIRECTORY,
    BurgersProblem,
    Trainer,
    LOSS,
    FullyConnectedNetwork,
    build_trainer,
    burgers_residual,
    predict_in_batches,
    train_with_loss_weight_history,
    save_loss_weight_artifacts,
)
from fvm_model_serial import solve_pde


def physics_validation(problem, model):
    """Evaluate residuals at fixed quarter-cell offsets, outside training points."""
    parameter = next(model.parameters())
    axes = [
        torch.tensor(
            [domain[0] + 0.25 * spacing], device=parameter.device, dtype=parameter.dtype
        )
        + torch.arange(count, device=parameter.device, dtype=parameter.dtype) * spacing
        for domain, spacing, count in (
            (problem.t_domain, problem.k, min(21, problem.number_of_time_intervals)),
            (problem.x_domain, problem.h, min(21, problem.number_of_x_cells)),
            (problem.y_domain, problem.h, min(21, problem.number_of_y_cells)),
        )
    ]
    coordinates = torch.stack(
        [a.flatten() for a in torch.meshgrid(*axes, indexing="ij")], 1
    )
    residual = burgers_residual(
        coordinates.requires_grad_(True), model, problem.nu
    ).detach()
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Non-finite validation residual")
    return {
        "rmse": residual.square().mean(0).sqrt().cpu().tolist(),
        "maximum_absolute": residual.abs().amax(0).cpu().tolist(),
        "points": len(coordinates),
        "grid": "quarter-cell offsets; up to first 21 cells per axis",
    }


def compare_fvm(problem, model, batch_size):
    """Run the unchanged FVM and evaluate PINN at every reference time level."""
    shape = (problem.number_of_x_cells, problem.number_of_y_cells)
    # Use float64 analytical targets for the float64 FVM solver.
    reference_problem = BurgersProblem(problem.mesh, problem.constants, torch.float64)
    _, initial = reference_problem.generate_full_initial_grid(
        problem.number_of_initial_points, "cpu"
    )
    initial = initial.detach().numpy().reshape(*shape, 2)
    start = time.perf_counter()
    u, v = solve_pde(
        problem.number_of_time_levels,
        *shape,
        problem.h,
        problem.k,
        problem.nu,
        initial[:, :, 0],
        initial[:, :, 1],
        problem.boundary,
    )
    fvm_seconds = time.perf_counter() - start
    x, y = [a.numpy() for a in reference_problem.fvm_cell_center_axes("cpu")]
    t = problem.t_domain[0] + np.arange(problem.number_of_time_levels) * problem.k
    xx, yy = np.meshgrid(x, y, indexing="ij")
    predictions = []
    start = time.perf_counter()
    for instant in t:
        inputs = np.column_stack((np.full(xx.size, instant), xx.ravel(), yy.ravel()))
        predictions.append(
            predict_in_batches(model, inputs, batch_size).reshape(*shape, 2)
        )
    prediction = np.stack(predictions)
    inference_seconds = time.perf_counter() - start
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite PINN predictions")
    arrays = dict(
        x=x, y=y, t=t, u=prediction[..., 0], v=prediction[..., 1], fvm_u=u, fvm_v=v
    )
    metrics = {}
    for name, reference in [("u", u), ("v", v)]:
        difference = arrays[name] - reference
        norm = float(np.linalg.norm(reference.ravel()))
        metrics[name] = dict(
            rmse=float(np.sqrt(np.mean(difference**2))),
            mae=float(np.mean(np.abs(difference))),
            maximum_absolute=float(np.max(np.abs(difference))),
            relative_l2=(
                float(np.linalg.norm(difference.ravel()) / norm) if norm else None
            ),
        )
        arrays[name + "_rmse_by_time"] = np.sqrt(np.mean(difference**2, axis=(1, 2)))
    return arrays, metrics, fvm_seconds, inference_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-directory", type=Path, default=BASE_DIRECTORY / "control_dicts"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=BASE_DIRECTORY / "pinn_sim"
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64] * 5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--adaptive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Inference batch size; training remains full-batch",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate a saved notebook/script model without training",
    )
    args = parser.parse_args(argv)
    if (
        args.epochs < 1
        or min(args.hidden_sizes) < 1
        or args.lr <= 0
        or args.batch_size < 1
    ):
        parser.error(
            "epochs, hidden sizes, learning rate and batch size must be positive"
        )
    metadata = {}
    if args.checkpoint:
        metadata_path = args.checkpoint.with_name(
            args.checkpoint.name.replace("pinn_model_", "pinn_metadata_")
        ).with_suffix(".json")
        if args.checkpoint.name == "pinn_model.pt":
            metadata_path = args.checkpoint.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        mesh, constants = metadata["mesh"], metadata["constants"]
        args.hidden_sizes = metadata["architecture"]
        args.dtype = metadata["dtype"].removeprefix("torch.")
    else:
        mesh = json.loads((args.config_directory / "mesh_properties.json").read_text())
        constants = json.loads(
            (args.config_directory / "constant_properties.json").read_text()
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    problem = BurgersProblem(mesh, constants, getattr(torch, args.dtype))
    run = args.output_directory / str(time.time_ns())
    run.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {run}", flush=True)
    model, trainer = build_trainer(
        problem, args.hidden_sizes, args.device, args.epochs, args.lr, args.adaptive
    )
    history = {}
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        )
        loss_path = args.checkpoint.with_name(
            args.checkpoint.name.replace("pinn_model_", "pinn_losses_")
        ).with_suffix(".npz")
        if args.checkpoint.name == "pinn_model.pt":
            loss_path = args.checkpoint.parent / "losses.npz"
        if loss_path.exists():
            with np.load(loss_path) as stored:
                history = {name: stored[name] for name in stored.files}
        metadata["source_checkpoint"] = str(args.checkpoint.resolve())
    else:
        # A copied model makes the smoke check independent of actual training.
        smoke = copy.deepcopy(model)
        optimizer = torch.optim.Adam(smoke.parameters(), lr=args.lr)
        losses = [loss.forward(smoke) for loss in trainer.losses]
        if any(value.ndim != 0 or not torch.isfinite(value) for value in losses):
            raise FloatingPointError("Smoke-test losses must be finite scalars")
        sum(losses).backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in smoke.parameters()
        ):
            raise FloatingPointError("Non-finite smoke-test gradients")
        optimizer.step()
        print(
            "Smoke-test losses:",
            {loss.name: value.item() for loss, value in zip(trainer.losses, losses)},
        )
        del smoke, optimizer, losses
        start = time.perf_counter()
        model, history, log_history = train_with_loss_weight_history(trainer)
        elapsed = time.perf_counter() - start
        save_loss_weight_artifacts(trainer, log_history, run)
        torch.save(trainer.optimizer.state_dict(), run / 'adam_optimizer.pt')
        if not all(np.isfinite(values).all() for values in history.values()):
            raise FloatingPointError("Non-finite training history")
        metadata = dict(
            seed=args.seed,
            architecture=args.hidden_sizes,
            dtype=str(problem.dtype),
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.lr,
            optimizer=('external torch.optim.Adam with framework AdaptiveLossWeights in Trainer'
                       if args.adaptive else 'Adam created internally by FisiocomPINN Trainer'),
            betas=[0.9, 0.999],
            adaptive_loss=args.adaptive,
            loss_weights=None if args.adaptive else dict(Initial=10, Boundary=1, PDE=1),
            adaptive_formula="sum(exp(-s_i)*MSE_i + s_i)" if args.adaptive else None,
            loss_weight_names=[loss.name for loss in trainer.losses],
            adaptive_weight_dtype=str(problem.dtype) if args.adaptive else None,
            adam_loss_weighting='adaptive from zero log variances' if args.adaptive else 'fixed 10:1:1',
            loss_log_variance_bounds=None,
            initialization="Xavier uniform weights; zero biases; Tanh activations",
            training_seconds=elapsed,
            smoke_test=True,
            collocation=dict(
                method="full deterministic FVM grid every iteration",
                random_sampling=False,
                initial_points=problem.number_of_initial_points,
                boundary_points=problem.number_of_boundary_points,
                pde_points=problem.number_of_pde_points,
            ),
        )
        if args.adaptive:
            if not torch.isfinite(log_history).all():
                raise FloatingPointError('Non-finite adaptive loss weights')
            log_vars = trainer.adaptive_weights.log_vars.detach()
            coefficients = torch.exp(-log_vars)
            if not torch.isfinite(coefficients).all():
                raise FloatingPointError('Non-finite adaptive loss coefficients')
            names = metadata['loss_weight_names']
            metadata.update(
                initial_loss_weights=dict.fromkeys(names, 1.0),
                initial_loss_log_vars=dict.fromkeys(names, 0.0),
                final_loss_weights=dict(zip(names, coefficients.cpu().tolist())),
                final_loss_log_vars=dict(zip(names, log_vars.cpu().tolist())),
                adam_weight_history_timing='after each Adam update',
            )
    source_paths = sorted(
        {
            Path(inspect.getfile(cls)).resolve()
            for cls in (Trainer, LOSS, FullyConnectedNetwork)
        }
    )
    framework_root = source_paths[0].parent.parent
    commit = subprocess.run(
        ["git", "-C", str(framework_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    metadata.update(
        mesh=mesh,
        constants=constants,
        torch_version=torch.__version__,
        evaluation_device=args.device,
        input_order=["t", "x", "y"],
        output_order=["u", "v"],
        framework_evaluation_commit=commit.stdout.strip() or None,
        framework_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    )
    torch.save(model.state_dict(), run / "pinn_model.pt")
    np.savez_compressed(run / "losses.npz", **history)
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metadata["physics_validation"] = physics_validation(problem, model)
    arrays, metrics, fvm_seconds, inference_seconds = compare_fvm(
        problem, model, args.batch_size
    )
    np.savez_compressed(run / "comparison.npz", **arrays)
    metadata.update(
        fvm_comparison=metrics,
        fvm_seconds=fvm_seconds,
        inference_seconds=inference_seconds,
        fvm_dtype="float64",
        reference="serial donor-cell FVM; cell-center point comparison",
    )
    (run / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(
        f'Plot with: python {BASE_DIRECTORY / "plot_pinn_results.py"} --run-directory {run}'
    )


if __name__ == "__main__":
    main()
