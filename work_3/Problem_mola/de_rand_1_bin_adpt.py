"""Self-adaptive DE/rand/1/bin with a static penalty for the spring problem."""

import csv
from pathlib import Path

import torch as tc


POPULATION_SIZE = 30
DIMENSION = 3
MAX_FITNESS_EVALUATIONS = 36_000
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 2
NUMBER_OF_CONSTRAINTS = 4
FEASIBILITY_TOLERANCE = 1.0e-12
PENALTY_WEIGHT = 1.0e5

# Decision vector: x = [N, D, d].
LOWER_BOUNDS = tc.tensor([2.0, 0.25, 0.05], dtype=tc.float64)
UPPER_BOUNDS = tc.tensor([15.0, 1.30, 2.00], dtype=tc.float64)


def spring_objective(x):
    """Return the spring volume for one solution or a population."""
    number_of_coils = x[..., 0]
    coil_diameter = x[..., 1]
    wire_diameter = x[..., 2]
    return (number_of_coils + 2.0) * coil_diameter * wire_diameter**2


def constraint_values(x):
    """Return g(x); a solution is feasible when every g_j(x) <= 0."""
    number_of_coils = x[..., 0]
    coil_diameter = x[..., 1]
    wire_diameter = x[..., 2]
    return tc.stack(
        (
            1.0
            - coil_diameter**3 * number_of_coils
            / (71785.0 * wire_diameter**4),
            (4.0 * coil_diameter**2 - wire_diameter * coil_diameter)
            / (
                12566.0
                * (coil_diameter * wire_diameter**3 - wire_diameter**4)
            )
            + 1.0 / (5108.0 * wire_diameter**2)
            - 1.0,
            1.0
            - 140.45 * wire_diameter
            / (coil_diameter**2 * number_of_coils),
            (coil_diameter + wire_diameter) / 1.5 - 1.0,
        ),
        dim=-1,
    )


def evaluate_spring(x, penalty_weight=PENALTY_WEIGHT):
    """Return penalized fitness, objective and nonnegative violations."""
    if penalty_weight < 0:
        raise ValueError("penalty_weight must be nonnegative")
    objective = spring_objective(x)
    violations = tc.clamp(constraint_values(x), min=0.0)
    fitness = objective + penalty_weight * tc.sum(violations, dim=-1)
    return fitness, objective, violations


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
    """Prefer the best feasible objective; use penalized fitness as fallback."""
    feasible_mask = tc.all(violations <= FEASIBILITY_TOLERANCE, dim=-1)
    feasible_indices = tc.nonzero(feasible_mask, as_tuple=False).flatten()
    if feasible_indices.numel() > 0:
        local_best = tc.argmin(objectives[feasible_indices])
        return feasible_indices[local_best], feasible_mask
    return tc.argmin(fitness), feasible_mask


def save_evolution_curve(
    curve, output_directory, population_size, seed,
    max_fitness_evaluations, evaluation_step,
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
    result, output_directory, population_size,
    max_fitness_evaluations, evaluation_step,
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
    """Minimize spring volume with a quadratic static exterior penalty."""
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

    generator = tc.Generator().manual_seed(seed)
    population = lower_bound + (upper_bound - lower_bound) * tc.rand(
        population_size, DIMENSION, dtype=tc.float64, generator=generator
    )
    fitness, objectives, violations = evaluate_spring(population, penalty_weight)
    fitness_evaluations = population_size

    differential_weights = 0.3 + 0.6 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    crossover_rates = 0.9 + 0.1 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )

    evolution_curve = [{
        "seed": seed,
        "fitness_evaluations": fitness_evaluations,
        **population_statistics(fitness, violations),
    }]
    next_curve_evaluation = fitness_evaluations + evaluation_step
    completed_generations = 0
    partial_generation_evaluations = 0

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size, max_fitness_evaluations - fitness_evaluations
        )
        new_population = population.clone()
        new_fitness = fitness.clone()
        new_objectives = objectives.clone()
        new_violations = violations.clone()
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
            donors = available[
                tc.randperm(population_size - 1, generator=generator)[:3]
            ]
            differential_weight = differential_weights[i]
            crossover_rate = crossover_rates[i]

            mutant = population[donors[0]] + differential_weight * (
                population[donors[1]] - population[donors[2]]
            )
            mutant = tc.maximum(tc.minimum(mutant, upper_bound), lower_bound)

            crossover_mask = tc.rand(DIMENSION, generator=generator) < crossover_rate
            forced_dimension = tc.randint(DIMENSION, (1,), generator=generator)
            crossover_mask[forced_dimension] = True
            trial = tc.where(crossover_mask, mutant, population[i])

            if crossover_mask.all():
                child_weight = (
                    differential_weights[donors[0]]
                    + differential_weight
                    * (differential_weights[donors[1]] - differential_weights[donors[2]])
                ).clamp(0.3, 0.9)
                child_crossover = (
                    crossover_rates[donors[0]]
                    + differential_weight
                    * (crossover_rates[donors[1]] - crossover_rates[donors[2]])
                ).clamp(0.9, 1.0)
            else:
                child_weight = differential_weight
                child_crossover = crossover_rate

            trial_fitness, trial_objective, trial_violations = evaluate_spring(
                trial, penalty_weight
            )
            fitness_evaluations += 1

            # Static-penalty selection: only penalized fitness is compared.
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                new_objectives[i] = trial_objective
                new_violations[i] = trial_violations
                new_differential_weights[i] = child_weight
                new_crossover_rates[i] = child_crossover

            if fitness_evaluations == next_curve_evaluation:
                evolution_curve.append({
                    "seed": seed,
                    "fitness_evaluations": fitness_evaluations,
                    **population_statistics(new_fitness, new_violations),
                })
                next_curve_evaluation += evaluation_step

        population = new_population
        fitness = new_fitness
        objectives = new_objectives
        violations = new_violations
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

        if evaluations_this_generation == population_size:
            completed_generations += 1
        else:
            partial_generation_evaluations = evaluations_this_generation

    best_index, feasible_mask = select_best_index(fitness, objectives, violations)
    feasible_percentage = 100.0 * feasible_mask.to(tc.float64).mean().item()
    if evolution_curve[-1]["fitness_evaluations"] != fitness_evaluations:
        evolution_curve.append({
            "seed": seed,
            "fitness_evaluations": fitness_evaluations,
            **population_statistics(fitness, violations),
        })

    base_directory = Path(__file__).resolve().parent
    if output_directory is None:
        output_directory = base_directory / "evolution_curves"
    curve_path = save_evolution_curve(
        evolution_curve, output_directory, population_size, seed,
        max_fitness_evaluations, evaluation_step,
    )

    best_x = population[best_index]
    result = {
        "algorithm": "Self-adaptive DE/rand/1/bin with quadratic static penalty",
        "best_f": fitness[best_index].item(),
        "best_objective": objectives[best_index].item(),
        "penalty": (fitness[best_index] - objectives[best_index]).item(),
        "max_constraint_violation": violations[best_index].max().item(),
        "is_feasible": bool(feasible_mask[best_index].item()),
        "final_population_feasible_percentage": feasible_percentage,
        "population_size": population_size,
        "dimension": DIMENSION,
        "max_fitness_evaluations": max_fitness_evaluations,
        "evaluation_step": evaluation_step,
        "completed_generations": completed_generations,
        "partial_generation_evaluations": partial_generation_evaluations,
        "seed": seed,
        "mean_differential_weight": differential_weights.mean().item(),
        "std_differential_weight": differential_weights.std(unbiased=False).item(),
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
    }

    if results_directory is None:
        results_directory = base_directory / "results"
    result_path = save_final_result(
        result, results_directory, population_size,
        max_fitness_evaluations, evaluation_step,
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
