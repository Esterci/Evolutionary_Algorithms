"""Self-adaptive DE/rand/1/bin with a static penalty for welded-beam design."""

import csv
from pathlib import Path

import torch as tc

POPULATION_SIZE = 30
DIMENSION = 4
MAX_FITNESS_EVALUATIONS = 320_000
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 2
NUMBER_OF_CONSTRAINTS = 5
FEASIBILITY_TOLERANCE = 1.0e-8
PENALTY_WEIGHT = 1.0e4

# Decision vector: x = [h, l, t, b].
LOWER_BOUNDS = tc.tensor([0.125, 0.1, 0.1, 0.1], dtype=tc.float64)
UPPER_BOUNDS = tc.tensor([10.0, 10.0, 10.0, 10.0], dtype=tc.float64)


def welded_beam_objective(x):
    """Return the welded-beam cost for one solution or a population."""
    h = x[..., 0]
    length = x[..., 1]
    t = x[..., 2]
    b = x[..., 3]
    return 1.10471 * h**2 * length + 0.04811 * t * b * (14.0 + length)


def welded_beam_constraints(x):
    """Return constraints and mechanical quantities.

    The constraints use the convention g_j(x) >= 0.
    """
    h = x[..., 0]
    length = x[..., 1]
    t = x[..., 2]
    b = x[..., 3]

    alpha = tc.sqrt(0.25 * (length**2 + (h + t) ** 2))

    # The classical formulation uses h*l in the denominator.
    tau_prime = 6000.0 / ((2.0**0.5) * h * length)
    tau_double_prime = (
        6000.0
        * (14.0 + 0.5 * length)
        * alpha
        / (2.0 * (0.707 * h * length * (length**2 / 12.0 + 0.25 * (h + t) ** 2)))
    )
    tau = tc.sqrt(
        tau_prime**2
        + tau_double_prime**2
        + length * tau_prime * tau_double_prime / alpha
    )
    sigma = 504000.0 / (t**2 * b)
    critical_load = 64746.022 * (1.0 - 0.0282346 * t) * t * b**3
    deflection = 2.1952 / (t**3 * b)

    constraints = tc.stack(
        (
            1.0 - tau / 13600.0,
            1.0 - sigma / 30000.0,
            b / h - 1.0,
            critical_load / 6000.0 - 1.0,
            1.0 - deflection / 0.25,
        ),
        dim=-1,
    )
    quantities = tc.stack((tau, sigma, critical_load, deflection), dim=-1)
    return constraints, quantities


def evaluate_welded_beam(x, penalty_weight=PENALTY_WEIGHT):
    """Return penalized fitness, cost, violations, constraints and quantities."""
    if penalty_weight < 0:
        raise ValueError("penalty_weight must be nonnegative")

    objective = welded_beam_objective(x)
    constraints, quantities = welded_beam_constraints(x)

    # Because feasibility is g_j(x) >= 0, only negative values are violations.
    violations = tc.clamp(-constraints, min=0.0)
    penalty = penalty_weight * tc.sum(violations, dim=-1)
    fitness = objective + penalty
    return fitness, objective, violations, constraints, quantities


def population_statistics(fitness, violations):
    """Return penalized-fitness and constraint-violation statistics."""
    statistics = {
        "fit_mean": fitness.mean().item(),
        "fit_std": fitness.std(unbiased=False).item(),
    }
    for index in range(NUMBER_OF_CONSTRAINTS):
        values = violations[:, index]
        prefix = f"constraint_{index + 1}_violation"
        statistics[f"{prefix}_mean"] = values.mean().item()
        statistics[f"{prefix}_std"] = values.std(unbiased=False).item()
    return statistics


def select_best_index(fitness, objectives, violations):
    """Prefer the best feasible cost; use penalized fitness as fallback."""
    feasible_mask = tc.all(violations <= FEASIBILITY_TOLERANCE, dim=-1)
    feasible_indices = tc.nonzero(feasible_mask, as_tuple=False).flatten()

    if feasible_indices.numel() > 0:
        local_best = tc.argmin(objectives[feasible_indices])
        return feasible_indices[local_best], feasible_mask

    return tc.argmin(fitness), feasible_mask


def save_evolution_curve(
    curve,
    output_directory,
    population_size,
    seed,
    max_fitness_evaluations,
    evaluation_step,
):
    """Save one run using the complete configuration in its filename."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"seed-{seed}--pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)

    return output_path


def save_final_result(
    result,
    output_directory,
    population_size,
    max_fitness_evaluations,
    evaluation_step,
):
    """Append one run to the CSV identified by the experiment configuration."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    if not write_header:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_fields = csv.DictReader(csv_file).fieldnames or []
        if existing_fields != list(result.keys()):
            raise ValueError(f"incompatible CSV schema in {output_path}")

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    return output_path


def differential_evolution(
    seed,
    population_size=POPULATION_SIZE,
    max_fitness_evaluations=MAX_FITNESS_EVALUATIONS,
    evaluation_step=EVALUATION_STEP,
    lower_bound=LOWER_BOUNDS,
    upper_bound=UPPER_BOUNDS,
    penalty_weight=PENALTY_WEIGHT,
    output_directory=None,
    results_directory=None,
):
    """Minimize welded-beam cost with a quadratic static exterior penalty."""
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if max_fitness_evaluations < population_size:
        raise ValueError("maximum evaluations must be at least population size")
    if evaluation_step < 1:
        raise ValueError("evaluation_step must be positive")
    if lower_bound.shape != upper_bound.shape:
        raise ValueError("bounds must have the same shape")
    if tc.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than its upper bound")

    dimension = len(lower_bound)
    generator = tc.Generator().manual_seed(seed)
    population = lower_bound + (upper_bound - lower_bound) * tc.rand(
        population_size,
        dimension,
        dtype=tc.float64,
        generator=generator,
    )
    (
        fitness,
        objectives,
        violations,
        constraints,
        quantities,
    ) = evaluate_welded_beam(population, penalty_weight)
    fitness_evaluations = population_size

    differential_weights = 0.3 + 0.6 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    crossover_rates = 0.9 + 0.1 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )

    evolution_curve = [
        {
            "seed": seed,
            "fitness_evaluations": fitness_evaluations,
            **population_statistics(fitness, violations),
        }
    ]
    next_curve_evaluation = fitness_evaluations + evaluation_step
    completed_generations = 0
    partial_generation_evaluations = 0

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size,
            max_fitness_evaluations - fitness_evaluations,
        )
        new_population = population.clone()
        new_fitness = fitness.clone()
        new_objectives = objectives.clone()
        new_violations = violations.clone()
        new_constraints = constraints.clone()
        new_quantities = quantities.clone()
        new_differential_weights = differential_weights.clone()
        new_crossover_rates = crossover_rates.clone()

        if evaluations_this_generation < population_size:
            target_indices = tc.randperm(population_size, generator=generator)[
                :evaluations_this_generation
            ].tolist()
        else:
            target_indices = range(population_size)

        for i in target_indices:
            available = tc.arange(population_size)
            available = available[available != i]
            donor_indices = available[
                tc.randperm(population_size - 1, generator=generator)[:3]
            ]

            differential_weight = differential_weights[i]
            crossover_rate = crossover_rates[i]

            mutant = population[donor_indices[0]] + differential_weight * (
                population[donor_indices[1]] - population[donor_indices[2]]
            )
            mutant = tc.maximum(tc.minimum(mutant, upper_bound), lower_bound)

            crossover_mask = tc.rand(dimension, generator=generator) < crossover_rate
            forced_dimension = tc.randint(dimension, (1,), generator=generator)
            crossover_mask[forced_dimension] = True
            trial = tc.where(crossover_mask, mutant, population[i])

            # Mutate encoded parameters only when all decision variables come
            # from the mutant; otherwise the target parameters are inherited.
            if crossover_mask.all():
                child_differential_weight = (
                    differential_weights[donor_indices[0]]
                    + differential_weight
                    * (
                        differential_weights[donor_indices[1]]
                        - differential_weights[donor_indices[2]]
                    )
                ).clamp(0.3, 0.9)
                child_crossover_rate = (
                    crossover_rates[donor_indices[0]]
                    + differential_weight
                    * (
                        crossover_rates[donor_indices[1]]
                        - crossover_rates[donor_indices[2]]
                    )
                ).clamp(0.9, 1.0)
            else:
                child_differential_weight = differential_weight
                child_crossover_rate = crossover_rate

            (
                trial_fitness,
                trial_objective,
                trial_violations,
                trial_constraints,
                trial_quantities,
            ) = evaluate_welded_beam(trial, penalty_weight)
            fitness_evaluations += 1

            # Static-penalty selection: only penalized fitness is compared.
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                new_objectives[i] = trial_objective
                new_violations[i] = trial_violations
                new_constraints[i] = trial_constraints
                new_quantities[i] = trial_quantities
                new_differential_weights[i] = child_differential_weight
                new_crossover_rates[i] = child_crossover_rate

            if fitness_evaluations == next_curve_evaluation:
                evolution_curve.append(
                    {
                        "seed": seed,
                        "fitness_evaluations": fitness_evaluations,
                        **population_statistics(new_fitness, new_violations),
                    }
                )
                next_curve_evaluation += evaluation_step

        population = new_population
        fitness = new_fitness
        objectives = new_objectives
        violations = new_violations
        constraints = new_constraints
        quantities = new_quantities
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

        if evaluations_this_generation == population_size:
            completed_generations += 1
        else:
            partial_generation_evaluations = evaluations_this_generation

    best_index, feasible_mask = select_best_index(fitness, objectives, violations)
    feasible_percentage = 100.0 * feasible_mask.to(tc.float64).mean().item()

    if evolution_curve[-1]["fitness_evaluations"] != fitness_evaluations:
        evolution_curve.append(
            {
                "seed": seed,
                "fitness_evaluations": fitness_evaluations,
                **population_statistics(fitness, violations),
            }
        )

    base_directory = Path(__file__).resolve().parent
    if output_directory is None:
        output_directory = base_directory / "evolution_curves"
    curve_path = save_evolution_curve(
        evolution_curve,
        output_directory,
        population_size,
        seed,
        max_fitness_evaluations,
        evaluation_step,
    )

    best_x = population[best_index]
    result = {
        "algorithm": ("Self-adaptive DE/rand/1/bin with quadratic static penalty"),
        "best_f": fitness[best_index].item(),
        "best_objective": objectives[best_index].item(),
        "penalty": (fitness[best_index] - objectives[best_index]).item(),
        "max_constraint_violation": violations[best_index].max().item(),
        "is_feasible": bool(feasible_mask[best_index].item()),
        "final_population_feasible_percentage": feasible_percentage,
        "population_size": population_size,
        "dimension": dimension,
        "max_fitness_evaluations": max_fitness_evaluations,
        "evaluation_step": evaluation_step,
        "completed_generations": completed_generations,
        "partial_generation_evaluations": partial_generation_evaluations,
        "seed": seed,
        "mean_differential_weight": differential_weights.mean().item(),
        "std_differential_weight": (differential_weights.std(unbiased=False).item()),
        "mean_crossover_rate": crossover_rates.mean().item(),
        "std_crossover_rate": crossover_rates.std(unbiased=False).item(),
        "fitness_evaluations": fitness_evaluations,
        "dtype": "float64",
        "device": "cpu",
        "evolution_curve_file": str(curve_path),
        "penalty_weight": penalty_weight,
        "best_x_0": best_x[0].item(),
        "best_x_1": best_x[1].item(),
        "best_x_2": best_x[2].item(),
        "best_x_3": best_x[3].item(),
        "g1": constraints[best_index, 0].item(),
        "g2": constraints[best_index, 1].item(),
        "g3": constraints[best_index, 2].item(),
        "g4": constraints[best_index, 3].item(),
        "g5": constraints[best_index, 4].item(),
        "tau": quantities[best_index, 0].item(),
        "sigma": quantities[best_index, 1].item(),
        "critical_load": quantities[best_index, 2].item(),
        "deflection": quantities[best_index, 3].item(),
    }

    if results_directory is None:
        results_directory = base_directory / "results"
    result_path = save_final_result(
        result,
        results_directory,
        population_size,
        max_fitness_evaluations,
        evaluation_step,
    )

    print(f"Evolution curve saved to: {curve_path}")
    print(f"Final result saved to: {result_path}")
    return result


def main():
    """Execute all independent runs."""
    for run in range(1, NUMBER_OF_RUNS + 1):
        print(f"Run {run}/{NUMBER_OF_RUNS} with seed {run}")
        differential_evolution(seed=run)


if __name__ == "__main__":
    main()
