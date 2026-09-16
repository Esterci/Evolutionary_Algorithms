r"""Plot saved Adam/DE-to-Adam PINNs, FVM fields, errors and training histories.

    python -m utils.plot_pinn_results
    python -m utils.plot_pinn_results --run-directory pinn_sim/ADAM_RUN \
        --run-directory pinn_sim/DE_ADAM_RUN --frames 60 --fps 12

Only stored NumPy outputs are needed: this script does not import PyTorch or
retrain the PINN. Field colors are shared between methods and fixed over time.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import numpy as np

BASE_DIRECTORY = Path(__file__).resolve().parents[1]


def load_comparison(path):
    """Validate the shared grid and [time, x, y] field ordering."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name] for name in ("x", "y", "t", "u", "v", "fvm_u", "fvm_v")
        }
    for name in ("x", "y", "t"):
        axis = arrays[name]
        if (
            axis.ndim != 1
            or not len(axis)
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
        ):
            raise ValueError(f"{name} must be a finite increasing one-dimensional axis")
    shape = tuple(len(arrays[name]) for name in ("t", "x", "y"))
    for name in ("u", "v", "fvm_u", "fvm_v"):
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
    return arrays


def method_name(metadata):
    """Identify the optimizer from saved provenance, never from a directory name."""
    if metadata.get("de_generations", 0) > 0 or metadata.get("de"):
        return "PINN DE → Adam"
    if "adam" in str(metadata.get("optimizer", "")).lower():
        return "PINN Adam"
    return "PINN (optimizer unknown)"


def load_run(directory):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return dict(
        directory=directory,
        metadata=metadata,
        label=f"{method_name(metadata)}",
        arrays=load_comparison(directory / "comparison.npz"),
    )


def select_runs(root):
    """Select the latest saved comparison for each optimizer variant."""
    latest = {}
    for path in sorted(
        Path(root).glob("*/comparison.npz"), key=lambda p: p.stat().st_mtime_ns
    ):
        metadata_path = path.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        if metadata.get("status", "complete") != "complete":
            continue
        latest[method_name(metadata)] = path.parent
    if not latest:
        raise ValueError(
            "No completed comparisons found. Run train_pinn.py or train_pinn_de_adam.py first."
        )
    return list(latest.values())


def validate_runs(runs):
    """Reject incompatible physical problems, grids or numerical references."""
    reference = runs[0]
    for run in runs[1:]:
        for key in ("mesh", "constants"):
            if key not in reference["metadata"] or key not in run["metadata"]:
                raise ValueError(
                    f"Multiple-run comparison requires {key} in every metadata.json"
                )
            if reference["metadata"][key] != run["metadata"][key]:
                raise ValueError(
                    f'Incompatible {key}: {reference["directory"]} and {run["directory"]}. '
                    "Select runs with the same physical configuration."
                )
        for key in ("x", "y", "t", "fvm_u", "fvm_v"):
            left, right = reference["arrays"][key], run["arrays"][key]
            if left.shape != right.shape or not np.allclose(
                left, right, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f'Incompatible {key} arrays in {run["directory"]}; '
                    "select matching grids and FVM references."
                )


def comparison_panel(arrays, predictions=None, variable=None):
    """Use shared field and error scales across every method and time level."""
    predictions = predictions or {"PINN": arrays}
    separated = variable is not None
    variables = (variable,) if separated else ("u", "v")
    rows = 1 + len(predictions) if separated else 2
    columns = 2 if separated else 1 + 2 * len(predictions)
    figure, axes = plt.subplots(
        rows, columns, figsize=(10, 3.6 * rows) if separated else (4 * columns, 7),
        constrained_layout=True
    )
    images, fields = [], []
    for row, variable in enumerate(variables):
        reference = arrays["fvm_" + variable]
        errors = {
            label: np.abs(data[variable] - reference)
            for label, data in predictions.items()
        }
        values = [reference] + [data[variable] for data in predictions.values()]
        low, high = min(float(v.min()) for v in values), max(
            float(v.max()) for v in values
        )
        if low == high:
            low, high = low - 1e-8, high + 1e-8
        field_norm = Normalize(low, high)
        error_norm = Normalize(
            0, max(1e-12, max(float(v.max()) for v in errors.values()))
        )
        panels = [("FVM", reference, False)]
        panels += [
            (label, data[variable], False) for label, data in predictions.items()
        ]
        panels += [
            (f"Absolute error: {label}", error, True) for label, error in errors.items()
        ]
        for column, (label, field, is_error) in enumerate(panels):
            if separated:
                panel_row = column - len(predictions) if is_error else column
                axis = axes[panel_row, 1 if is_error else 0]
            else:
                axis = axes[row, column]
            im = axis.pcolormesh(
                arrays["x"],
                arrays["y"],
                field[0].T,
                shading="auto",
                cmap="magma" if is_error else "viridis",
                norm=error_norm if is_error else field_norm,
            )
            axis.set(
                title=f"{label}\n{variable}", xlabel="x", ylabel="y", aspect="equal"
            )
            figure.colorbar(im, ax=axis)
            images.append(im)
            fields.append(field)
    if separated:
        axes[0, 1].set_axis_off()
    title = figure.suptitle("")

    def update(index):
        for im, field in zip(images, fields):
            im.set_array(field[index].T.ravel())
        title.set_text(
            f'Conservative Burgers: FVM and PINNs | t = {arrays["t"][index]:.5g}'
        )
        return images + [title]

    return figure, update


def comparison_report(runs):
    """Recompute metrics from stored fields and expose experimental differences."""
    rows = []
    for run in runs:
        data, metadata = run["arrays"], run["metadata"]
        for variable in ("u", "v"):
            reference = data["fvm_" + variable]
            difference = data[variable] - reference
            norm = float(np.linalg.norm(reference.ravel()))
            rows.append(
                dict(
                    method=method_name(metadata),
                    run=str(run["directory"].resolve()),
                    variable=variable,
                    rmse=float(np.sqrt(np.mean(difference**2))),
                    mae=float(np.mean(np.abs(difference))),
                    maximum_absolute=float(np.max(np.abs(difference))),
                    relative_l2=(
                        float(np.linalg.norm(difference.ravel()) / norm)
                        if norm
                        else None
                    ),
                    adaptive_loss=metadata.get("adaptive_loss"),
                    loss_weights=metadata.get("loss_weights"),
                    de_adaptive_loss_weights=metadata.get(
                        "de_adaptive_loss_weights", False
                    ),
                    de_adaptive_pde_weights=metadata.get(
                        "de_adaptive_pde_weights", False
                    ),
                    loss_weight_names=metadata.get("loss_weight_names"),
                    de_best_loss_weights=metadata.get("de_best_loss_weights"),
                    final_loss_weights=metadata.get("final_loss_weights"),
                    adam_loss_weighting=metadata.get("adam_loss_weighting"),
                    adaptive_formula=metadata.get("adaptive_formula"),
                    fitness_formula=metadata.get("fitness_formula"),
                    reference_fitness_formula=metadata.get("reference_fitness_formula"),
                    epochs=metadata.get("epochs"),
                    de_generations=metadata.get("de_generations", 0),
                    adam_epochs=metadata.get("adam_epochs", metadata.get("epochs")),
                    fitness_evaluations=metadata.get("de", {}).get(
                        "fitness_evaluations"
                    ),
                    training_seconds=metadata.get("training_seconds"),
                    de_seconds=metadata.get("de_seconds"),
                    adam_seconds=metadata.get("adam_seconds"),
                    seed=metadata.get("seed"),
                    de_seed=metadata.get("de_seed"),
                    architecture=metadata.get("architecture"),
                    dtype=metadata.get("dtype"),
                    device=metadata.get("device"),
                )
            )
    return rows


def plot_histories(runs, output):
    """Keep DE fitness evaluations distinct from Adam gradient iterations."""
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plotted = False
    for run in runs:
        path = run["directory"] / "losses.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as losses:
            for name in losses.files:
                values = losses[name]
                if values.ndim != 1 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid Adam history {name} in {path}")
                if len(values):
                    axis.semilogy(
                        np.arange(1, len(values) + 1),
                        values,
                        label=f'{run["label"]}: {name}',
                    )
                    plotted = True
    if plotted:
        axis.set(
            xlabel="Adam iteration (after DE for hybrid runs)",
            ylabel="Unweighted MSE",
            title="Training losses before each Adam update",
        )
        axis.legend(fontsize="small")
        axis.grid(alpha=0.3)
        figure.savefig(output / "training_losses.png", dpi=150)
    plt.close(figure)
    for run in runs:
        weights_path = run["directory"] / "adam_loss_weights.npz"
        if weights_path.exists():
            with np.load(weights_path, allow_pickle=False) as history:
                names, weights = history["loss_names"], history["weights"]
                if (
                    names.ndim != 1
                    or weights.ndim != 2
                    or weights.shape[1] != len(names)
                    or not np.isfinite(weights).all()
                    or np.any(weights <= 0)
                ):
                    raise ValueError(f"Invalid adaptive Adam weights in {weights_path}")
                figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
                for index, name in enumerate(names):
                    axis.semilogy(
                        np.arange(1, len(weights) + 1),
                        weights[:, index],
                        label=str(name),
                    )
                axis.set(
                    xlabel="Adam iteration",
                    ylabel="Adaptive loss coefficient",
                    title=f'{run["label"]}: weights after each Adam update',
                )
                axis.legend()
                axis.grid(alpha=0.3)
                figure.savefig(
                    output / f'adam_loss_weights_{run["directory"].name}.png', dpi=150
                )
                plt.close(figure)
        path = run["directory"] / "de_history.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as history:
            figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for name, label in (
                ("best_fitness", "Best"),
                ("mean_fitness", "Population mean"),
            ):
                axis.plot(history["fitness_evaluations"], history[name], label=label)
            # Log-variance regularization permits zero and negative fitness.
            if any(
                np.any(history[name] <= 0) for name in ("best_fitness", "mean_fitness")
            ):
                axis.set_yscale("symlog", linthresh=1.0)
            else:
                axis.set_yscale("log")
            axis.set(
                xlabel="DE fitness evaluations (including initialization)",
                ylabel=(
                    "Regularized PINN fitness"
                    if run["metadata"].get("de_adaptive_loss_weights")
                    or run["metadata"].get("de_adaptive_pde_weights")
                    else "Weighted PINN fitness"
                ),
                title=run["label"],
            )
            axis.legend()
            axis.grid(alpha=0.3)
            figure.savefig(
                output / f'de_convergence_{run["directory"].name}.png', dpi=150
            )
            plt.close(figure)
            if "selected_reference_fitness" in history.files:
                figure, axes = plt.subplots(
                    1, 2, figsize=(12, 4), constrained_layout=True
                )
                evaluations = history["fitness_evaluations"]
                axes[0].semilogy(
                    evaluations,
                    history["selected_reference_fitness"],
                    label="Selected by DE fitness",
                )
                axes[0].semilogy(
                    evaluations,
                    history["best_reference_fitness"],
                    label="Best fixed reference among all evaluated candidates",
                )
                axes[0].set(
                    ylabel="Fixed-weight reference objective",
                    xlabel="DE fitness evaluations",
                )
                for name in history.files:
                    if name.startswith("best_weight_"):
                        axes[1].semilogy(
                            evaluations,
                            history[name],
                            label=name.removeprefix("best_weight_"),
                        )
                axes[1].set(
                    ylabel="Selected loss coefficient",
                    xlabel="DE fitness evaluations",
                )
                for axis in axes:
                    axis.legend(fontsize="small")
                    axis.grid(alpha=0.3)
                figure.savefig(
                    output / f'de_diagnostics_{run["directory"].name}.png', dpi=150
                )
                plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        action="append",
        help="Saved run; repeat to compare Adam and DE-to-Adam together",
    )
    parser.add_argument(
        "--runs-directory",
        type=Path,
        default=BASE_DIRECTORY / "pinn_sim",
        help="Without explicit runs, select the latest run of each optimizer",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args(argv)
    if args.frames < 2 or args.fps < 1:
        parser.error("frames must be at least 2 and fps must be positive")
    try:
        directories = args.run_directory or select_runs(args.runs_directory)
        if len({p.resolve() for p in directories}) != len(directories):
            raise ValueError("Select each run only once")
        runs = [load_run(directory) for directory in directories]
        validate_runs(runs)
        if len({run["label"] for run in runs}) != len(runs):
            for run in runs:
                run["label"] += f' ({run["directory"].resolve()})'
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    output = args.output_directory or (
        runs[0]["directory"] / "plots"
        if len(runs) == 1
        else BASE_DIRECTORY / "plots" / f"pinn_comparison_{time.time_ns()}"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = comparison_report(runs)
    report = dict(
        reference="Serial donor-cell FVM; cell-center point comparison",
        interpretation="Descriptive single-run results; unequal training budgets or "
        "loss weighting do not establish optimizer superiority.",
        runs=[
            dict(directory=str(r["directory"].resolve()), metadata=r["metadata"])
            for r in runs
        ],
        metrics=rows,
    )
    (output / "comparison_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (output / "comparison_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for run in runs:
        print(
            f'{run["label"]}: adaptive_loss={run["metadata"].get("adaptive_loss", "unknown")}, '
            f'epochs={run["metadata"].get("epochs", "unknown")}'
        )
    arrays = runs[0]["arrays"]
    figure, update = comparison_panel(arrays, {r["label"]: r["arrays"] for r in runs})
    for index in np.unique(np.linspace(0, len(arrays["t"]) - 1, 3, dtype=int)):
        update(index)
        figure.savefig(output / f"comparison_time_{index:05d}.png", dpi=150)
    indices = np.unique(
        np.linspace(
            0, len(arrays["t"]) - 1, min(args.frames, len(arrays["t"])), dtype=int
        )
    )
    animation = FuncAnimation(
        figure, update, frames=indices, interval=1000 / args.fps, blit=False
    )
    animation.save(
        output / "pinn_fvm_comparison.gif", writer=PillowWriter(fps=args.fps), dpi=90
    )
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for run in runs:
        for variable in ("u", "v"):
            data = run["arrays"]
            rmse = np.sqrt(
                np.mean((data[variable] - data["fvm_" + variable]) ** 2, axis=(1, 2))
            )
            axis.plot(data["t"], rmse, label=f'{run["label"]}: {variable}')
    axis.set(
        xlabel="Time",
        ylabel="RMSE against FVM",
        title="PINN discrepancy from the numerical reference",
    )
    axis.legend(fontsize="small")
    axis.grid(alpha=0.3)
    figure.savefig(output / "error_evolution.png", dpi=150)
    plt.close(figure)
    plot_histories(runs, output)
    print(f"Plots, GIF and metrics: {output.resolve()}")


if __name__ == "__main__":
    main()
