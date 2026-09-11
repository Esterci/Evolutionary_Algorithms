"""Train the existing conservative Burgers PINN with DE followed by Adam.

From the repository root (using an environment with the local Pinn-Torch)::

    python work_4/train_pinn_de_adam.py --epochs 1000 --de-generations 100

One DE generation consumes one schedule slot, then Adam uses the remaining
slots. A generation costs population_size fitness evaluations; it is not
computationally equivalent to one Adam epoch. See README_de_adam.md.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from differential_evolution import DEConfig, differential_evolution
from pinn_de_adam import PINNFitness, make_adam_trainer
from pinn_model import (
    BASE_DIRECTORY,
    BurgersProblem,
    FullyConnectedNetwork,
    LOSS,
    Trainer,
    build_trainer,
    burgers_residual,
    train_with_loss_weight_history,
    save_loss_weight_artifacts,
)
from train_pinn import compare_fvm


def physics_validation(problem, model, maximum_cells=21):
    """Check PDE residual on fixed, unused quarter-cell points across the domain.

    At most 21 evenly selected cells per axis keep the independent check bounded.
    Neither DE selection nor Adam sees these points.
    """
    parameter = next(model.parameters())
    axes = []
    for domain, spacing, count in (
        (problem.t_domain, problem.k, problem.number_of_time_intervals),
        (problem.x_domain, problem.h, problem.number_of_x_cells),
        (problem.y_domain, problem.h, problem.number_of_y_cells),
    ):
        indices = torch.linspace(
            0,
            count - 1,
            min(count, maximum_cells),
            device=parameter.device,
            dtype=parameter.dtype,
        ).round()
        axes.append(domain[0] + (indices + 0.25) * spacing)
    coordinates = torch.stack(
        [axis.reshape(-1) for axis in torch.meshgrid(*axes, indexing="ij")], dim=1
    )
    with torch.enable_grad():
        residual = burgers_residual(
            coordinates.requires_grad_(True), model, problem.nu
        ).detach()
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Non-finite held-out PDE residual")
    return dict(
        rmse=residual.square().mean(0).sqrt().cpu().tolist(),
        maximum_absolute=residual.abs().amax(0).cpu().tolist(),
        points=len(coordinates),
        grid="quarter-cell offsets; up to 21 evenly selected cells per axis",
    )


def source_metadata():
    """Fingerprint only the relevant scientific implementation files."""
    sources = {
        Path(inspect.getfile(cls)).resolve()
        for cls in (Trainer, LOSS, FullyConnectedNetwork)
    }
    framework_root = Path(inspect.getfile(Trainer)).resolve().parent.parent
    commit = subprocess.run(
        ["git", "-C", str(framework_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    local_sources = [
        BASE_DIRECTORY / name
        for name in (
            "differential_evolution.py",
            "pinn_de_adam.py",
            "train_pinn_de_adam.py",
            "pinn_model.py",
            "train_pinn.py",
            "fvm_model_serial.py",
        )
    ]
    local_sources.append(
        BASE_DIRECTORY.parent / "work_3/rastrigin/de_best_1_bin_adpt.py"
    )
    return dict(
        framework_commit=commit.stdout.strip() or None,
        framework_directory=str(framework_root),
        framework_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)
        },
        experiment_source_sha256={
            str(p.relative_to(BASE_DIRECTORY.parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in local_sources
        },
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-directory", type=Path, default=BASE_DIRECTORY / "control_dicts"
    )
    parser.add_argument(
        "--output-directory", type=Path, default=BASE_DIRECTORY / "pinn_sim"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4852,
        help="Total schedule slots: DE generations + Adam epochs",
    )
    parser.add_argument("--de-generations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=-2.0,
        help="DE search lower bound for every weight and bias",
    )
    parser.add_argument("--upper-bound", type=float, default=2.0)
    parser.add_argument(
        "--control-adaptation",
        choices=["jde", "legacy"],
        default="jde",
        help="Dimension-independent proposals, or the original all-coordinate rule",
    )
    parser.add_argument("--adaptation-probability", type=float, default=0.1)
    parser.add_argument(
        "--initialization",
        choices=["local", "uniform"],
        default="local",
        help="Population near the initialized network, or across all bounds",
    )
    parser.add_argument(
        "--initial-spread",
        type=float,
        default=0.05,
        help="Local initialization half-width as a fraction of each bound width",
    )
    parser.add_argument(
        "--evolve-loss-weights",
        "--evolve-pde-weights",
        dest="evolve_loss_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evolve Initial, Boundary and combined PDE log variances (old PDE flag is a deprecated alias)",
    )
    parser.add_argument(
        "--log-weight-init-bound",
        type=float,
        default=4.0,
        help="Log-variance initialization box only; no clipping during DE or Adam",
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32] * 3)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--seed", type=int, default=2026, help="Network initialization seed"
    )
    parser.add_argument(
        "--de-seed", type=int, default=2027, help="Independent DE random stream"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="FVM comparison inference batch size",
    )
    parser.add_argument(
        "--print-steps", type=int, default=250, help="Adam log interval"
    )
    parser.add_argument(
        "--de-print-steps", type=int, default=10, help="DE generation log interval"
    )
    arguments = sys.argv[1:] if argv is None else argv
    if any(argument.split('=')[0] == '--log-weight-bound' for argument in arguments):
        parser.error('--log-weight-bound was removed: use --log-weight-init-bound for '
                     'initialization only; adaptive log variances are now unconstrained')
    args = parser.parse_args(arguments)
    if any(argument in ('--evolve-pde-weights', '--no-evolve-pde-weights')
           for argument in arguments):
        print('Deprecated PDE flag: use --evolve-loss-weights or --no-evolve-loss-weights. '
              'Adaptive mode now evolves Initial, Boundary and combined PDE weights.', file=sys.stderr)
    if not 1 <= args.de_generations < args.epochs:
        parser.error("Require 1 <= de-generations < epochs so both stages run")
    if args.population_size < 4:
        parser.error("population-size must be at least 4 for DE/rand/1/bin")
    if (
        not math.isfinite(args.lower_bound)
        or not math.isfinite(args.upper_bound)
        or args.lower_bound >= args.upper_bound
    ):
        parser.error("DE bounds must be finite and increasing")
    if (
        min(args.hidden_sizes) < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or args.batch_size < 1
        or args.print_steps < 1
        or args.de_print_steps < 1
    ):
        parser.error(
            "hidden sizes, learning rate, batch size and print interval must be positive"
        )
    if not 0 <= args.seed < 2**32 or not 0 <= args.de_seed < 2**63:
        parser.error("Require 0 <= seed < 2**32 and 0 <= de-seed < 2**63")
    if (
        not math.isfinite(args.adaptation_probability)
        or not 0 <= args.adaptation_probability <= 1
    ):
        parser.error("adaptation-probability must lie in [0, 1]")
    if not math.isfinite(args.initial_spread) or not 0 < args.initial_spread <= 1:
        parser.error("initial-spread must lie in (0, 1]")
    if not math.isfinite(args.log_weight_init_bound) or not 0 < args.log_weight_init_bound <= 20:
        parser.error("log-weight-init-bound must lie in (0, 20]")
    return args


def main(argv=None):
    args = parse_args(argv)
    mesh = json.loads((args.config_directory / "mesh_properties.json").read_text())
    constants = json.loads(
        (args.config_directory / "constant_properties.json").read_text()
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    problem = BurgersProblem(mesh, constants, getattr(torch, args.dtype))
    adam_epochs = args.epochs - args.de_generations
    model, registered_trainer = build_trainer(
        problem,
        args.hidden_sizes,
        args.device,
        adam_epochs,
        args.lr,
        adaptive=False,
        print_steps=args.print_steps,
    )
    objective = PINNFitness(
        registered_trainer, evolve_loss_weights=args.evolve_loss_weights
    )
    initial_vector = objective.vector()
    bounds = objective.bounds(args.lower_bound, args.upper_bound, args.log_weight_init_bound)
    bound_mask = torch.ones_like(initial_vector, dtype=torch.bool)
    if args.evolve_loss_weights:
        bound_mask[objective.network_dimension:] = False
    if (
        not torch.isfinite(initial_vector).all()
        or not ((initial_vector >= bounds[0]) & (initial_vector <= bounds[1])).all()
    ):
        raise ValueError("DE bounds must contain the unmodified initialized network")
    config = DEConfig(
        population_size=args.population_size,
        generations=args.de_generations,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        seed=args.de_seed,
        control_adaptation=args.control_adaptation,
        adaptation_probability=args.adaptation_probability,
        initialization=args.initialization,
        initial_spread=args.initial_spread,
    )
    metadata = dict(
        status="running",
        seed=args.seed,
        de_seed=args.de_seed,
        architecture=args.hidden_sizes,
        activation="Tanh",
        dtype=str(problem.dtype),
        device=args.device,
        torch_version=torch.__version__,
        hardware=(
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda"
            else "CPU"
        ),
        torch_num_threads=torch.get_num_threads(),
        epochs=args.epochs,
        de_generations=args.de_generations,
        adam_epochs=adam_epochs,
        schedule_unit="one DE generation or one Adam epoch; unequal computation per slot",
        optimizer="self-adaptive DE/rand/1/bin -> external torch.optim.Adam in Trainer",
        learning_rate=args.lr,
        betas=[0.9, 0.999],
        adaptive_loss=args.evolve_loss_weights,
        de_adaptive_loss_weights=args.evolve_loss_weights,
        adaptive_formula='sum(exp(-s_i)*MSE_i + s_i)' if args.evolve_loss_weights else None,
        loss_weight_names=[loss.name for loss in objective.losses],
        adaptive_weight_dtype=str(problem.dtype) if args.evolve_loss_weights else None,
        adam_loss_weighting=('adaptive from selected DE log variances' if args.evolve_loss_weights
                             else 'fixed 10:1:1'),
        loss_log_variance_bounds=None,
        loss_log_variance_initialization_box=(
            [-args.log_weight_init_bound, args.log_weight_init_bound]
            if args.evolve_loss_weights
            else None
        ),
        initial_loss_weights={
            loss.name: weight
            for loss, weight in zip(objective.losses, objective.weights)
        },
        loss_weights=None if args.evolve_loss_weights else dict(Initial=10, Boundary=1, PDE=1),
        reference_loss_weights={
            loss.name: weight
            for loss, weight in zip(objective.losses, objective.reference_weights)
        },
        fitness_formula=(
            "sum(exp(-s_i)*MSE_i+s_i), i=Initial,Boundary,PDE"
            if args.evolve_loss_weights
            else "10*MSE_initial + MSE_boundary + MSE_PDE"
        ),
        reference_fitness_formula=(
            "MSE_initial + MSE_boundary + MSE_PDE"
            if args.evolve_loss_weights
            else "10*MSE_initial + MSE_boundary + MSE_PDE"
        ),
        fitness_role="direct weight optimization; no inner PINN training per individual",
        initialization="Xavier uniform weights; zero biases; preserved as population[0]",
        de=dict(
            asdict(config),
            strategy="rand/1/bin",
            synchronous=True,
            initialization_description=(
                "population[0] is unchanged network; others uniform "
                "in initial +/- initial_spread*bound_width, clipped"
                if args.initialization == "local"
                else "population[0] is network; others uniform within bounds"
            ),
            mutation_factor_range=[0.3, 0.9],
            crossover_rate_range=[0.9, 1.0],
            control_adaptation_description=(
                "independent F/CR uniform proposals before mutation; "
                "inherit only on strict fitness improvement"
                if args.control_adaptation == "jde"
                else "donor difference only when every coordinate crosses"
            ),
            replacement="strict greedy <; target retained on ties",
            constraint_handling="clip network parameters only; adaptive log variances unconstrained",
            stopping="fixed complete generation count",
            expected_fitness_evaluations=args.population_size
            * (args.de_generations + 1),
        ),
        genome_dimension=objective.dimension,
        network_dimension=objective.network_dimension,
        genome_layout=objective.layout,
        collocation=dict(
            method="full deterministic FVM-aligned grid",
            random_sampling=False,
            resampling_frequency=0,
            graph_refresh="fresh coordinates at every objective or Adam evaluation",
            initial_points=problem.number_of_initial_points,
            boundary_points=problem.number_of_boundary_points,
            pde_points=problem.number_of_pde_points,
            batching="full batch per loss",
        ),
        mesh=mesh,
        constants=constants,
        input_order=["t", "x", "y"],
        output_order=["u", "v"],
        physical_parameters_inferred=[],
        observational_training_data=False,
        normalization="network coordinates to [-1,1]; physical PDE unchanged",
        independent_runs=1,
        **source_metadata(),
    )
    metadata["de"]["crossover_rate_range"] = (
        [0.0, 1.0] if args.control_adaptation == "jde" else [0.9, 1.0]
    )
    run = args.output_directory / str(time.time_ns())
    run.mkdir(parents=True, exist_ok=False)

    def save_metadata():
        (run / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )

    save_metadata()
    print(f"Artifacts: {run}", flush=True)
    print(
        f"DE: {args.de_generations} generations, {args.population_size} individuals, "
        f'{metadata["de"]["expected_fitness_evaluations"]} fitness evaluations, '
        f"{objective.dimension} parameters; then {adam_epochs} Adam epochs.",
        flush=True,
    )
    try:
        start = time.perf_counter()
        survivor_diagnostics = [None] * args.population_size
        survivor_scores = [math.inf] * args.population_size
        evaluation_index = 0
        best_reference = math.inf
        best_reference_vector = None
        diagnostic_history = []

        def evaluate_candidate(vector):
            nonlocal evaluation_index, best_reference, best_reference_vector
            value = objective(vector)
            # Match the precision used for population ranking, including ties.
            ranked = torch.as_tensor(value, dtype=vector.dtype).item()
            # DE evaluates initialization in index order, then one trial per
            # target per synchronous generation. Preserve diagnostics with the
            # same strict survivor rule, including equal global-best scores.
            target = evaluation_index % args.population_size
            if math.isfinite(ranked) and ranked < survivor_scores[target]:
                survivor_scores[target] = ranked
                survivor_diagnostics[target] = objective.last_diagnostics.copy()
            evaluation_index += 1
            reference = objective.last_diagnostics["reference_fitness"]
            if math.isfinite(reference) and reference < best_reference:
                best_reference = reference
                best_reference_vector = vector.detach().clone()
            return value

        with (run / "de_history.csv").open("w", newline="") as stream:
            writer = None

            def record_generation(row):
                nonlocal writer
                best_diagnostics = survivor_diagnostics[row["best_index"]]
                row.update(
                    {
                        f"best_{key}": value
                        for key, value in best_diagnostics.items()
                        if key not in ("fitness", "reference_fitness")
                    }
                )
                row.update(
                    selected_reference_fitness=best_diagnostics["reference_fitness"],
                    best_reference_fitness=best_reference,
                )
                diagnostic_history.append(row.copy())
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                if (
                    row["generation"] % args.de_print_steps == 0
                    or row["generation"] == config.generations
                ):
                    print(
                        f'DE generation {row["generation"]}: '
                        f'best fitness {row["best_fitness"]:.8g}, '
                        f'fixed reference {row["selected_reference_fitness"]:.8g}, '
                        f'accepted {row["accepted_trials"]}/{args.population_size}, '
                        f'evaluations {row["fitness_evaluations"]}',
                        flush=True,
                    )

            result = differential_evolution(
                evaluate_candidate,
                initial_vector,
                config,
                bounds=bounds,
                bound_mask=bound_mask,
                on_generation=record_generation,
            )
        metadata["de_seconds"] = time.perf_counter() - start
        # Evaluating the final trial leaves that trial in the model, even if it
        # lost. Always load the returned global best before starting Adam.
        objective.load(result.best_vector)
        torch.save(model.state_dict(), run / "de_best_model.pt")
        torch.save(
            dict(
                population=result.population,
                fitness=result.fitness,
                mutation_factors=result.mutation_factors,
                crossover_rates=result.crossover_rates,
                best_vector=result.best_vector,
                initial_vector=initial_vector,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                bound_mask=bound_mask,
                best_reference_vector=best_reference_vector,
                best_reference_fitness=best_reference,
            ),
            run / "de_population.pt",
        )
        np.savez_compressed(
            run / "de_history.npz",
            **{
                name: np.asarray([row[name] for row in diagnostic_history])
                for name in diagnostic_history[0]
            },
        )
        metadata["de"].update(
            fitness_evaluations=result.fitness_evaluations,
            nonfinite_evaluations=result.nonfinite_evaluations,
            best_fitness=result.best_fitness,
        )
        metadata["de_best_losses"] = objective.terms()
        metadata["de_best_reference_fitness"] = objective.reference_fitness(
            metadata["de_best_losses"]
        )
        metadata["de_best_loss_weights"] = {
            loss.name: weight
            for loss, weight in zip(objective.losses, objective.weights)
        }
        metadata["de_best_loss_log_vars"] = dict(zip(
            metadata['loss_weight_names'], objective.loss_log_vars.cpu().tolist()))
        metadata["de_physics_validation"] = physics_validation(problem, model)
        metadata["status"] = "de_complete"
        save_metadata()
        print(f"DE best fitness: {result.best_fitness:.8g}; starting Adam.", flush=True)

        adam_trainer = make_adam_trainer(
            registered_trainer,
            adam_epochs,
            args.lr,
            args.print_steps,
            loss_log_vars=objective.loss_log_vars if args.evolve_loss_weights else None,
        )
        start = time.perf_counter()
        model, history, log_history = train_with_loss_weight_history(adam_trainer)
        metadata["adam_seconds"] = time.perf_counter() - start
        # Preserve the completed trajectory even when finite-value checks fail.
        np.savez_compressed(run / "losses.npz", **history)
        save_loss_weight_artifacts(adam_trainer, log_history, run)
        if args.evolve_loss_weights:
            final_vector = objective.vector()
            final_vector[objective.network_dimension:] = adam_trainer.adaptive_weights.log_vars.detach()
            if not torch.isfinite(final_vector).all() or not torch.isfinite(log_history).all():
                raise FloatingPointError('Non-finite adaptive weights or model parameters')
            objective.load(final_vector)
        if (
            not all(np.isfinite(values).all() for values in history.values())
            or not torch.isfinite(objective.vector()).all()
        ):
            raise FloatingPointError("Non-finite Adam training history or parameters")
        torch.save(model.state_dict(), run / "pinn_model.pt")
        torch.save(adam_trainer.optimizer.state_dict(), run / "adam_optimizer.pt")
        final_losses = objective.terms()
        if not all(math.isfinite(value) for value in final_losses.values()):
            raise FloatingPointError("Non-finite final PINN losses")
        final_weights = objective.weights
        final_reference = objective.reference_fitness(final_losses)
        final_score = objective.score(final_losses)
        # Keep metadata JSON-serializable even when finite log variances make
        # exp(-s) overflow. The exception handler must still record the failure.
        if not all(math.isfinite(value) for value in
                   (*final_weights, final_reference, final_score)):
            raise FloatingPointError('Non-finite final adaptive coefficients or fitness')
        metadata["final_losses"] = final_losses
        metadata["final_loss_log_vars"] = dict(zip(
            metadata['loss_weight_names'], objective.loss_log_vars.cpu().tolist()))
        metadata["final_loss_weights"] = dict(zip(metadata['loss_weight_names'], final_weights))
        metadata["adam_weight_history_timing"] = 'after each Adam update'
        metadata["final_reference_fitness"] = final_reference
        metadata["final_regularized_fitness"] = final_score
        metadata["status"] = "training_complete"
        save_metadata()
        metadata["physics_validation"] = physics_validation(problem, model)
        arrays, metrics, fvm_seconds, inference_seconds = compare_fvm(
            problem, model, args.batch_size
        )
        np.savez_compressed(run / "comparison.npz", **arrays)
        metadata.update(
            status="complete",
            fvm_comparison=metrics,
            fvm_seconds=fvm_seconds,
            inference_seconds=inference_seconds,
            fvm_dtype="float64",
            reference="serial donor-cell FVM; cell-center point comparison",
            diagnostic_objective_evaluations=2,
        )
        save_metadata()
        print(json.dumps(metrics, indent=2))
        print(f"Plot with: python work_4/plot_pinn_results.py --run-directory {run}")
    except Exception as error:
        metadata.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        save_metadata()
        raise


if __name__ == "__main__":
    main()
