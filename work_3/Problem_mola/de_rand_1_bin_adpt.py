"""Self-adaptive DE/rand/1/bin for the tension/compression spring problem."""

import csv
from pathlib import Path

import torch as tc


POPULATION_SIZE = 30
DIMENSION = 3
MAX_FITNESS_EVALUATIONS = 36_000
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 35

# Decision vector: x = [N, D, d].
LOWER_BOUNDS = tc.tensor([2.0, 0.25, 0.05], dtype=tc.float64)
UPPER_BOUNDS = tc.tensor([15.0, 1.30, 2.00], dtype=tc.float64)


def spring_problem(x):
    """Return spring volume, constraints and total constraint violation."""

    number_of_coils = x[..., 0]
    coil_diameter = x[..., 1]
    wire_diameter = x[..., 2]

    volume = (
        (number_of_coils + 2.0)
        * coil_diameter
        * wire_diameter**2
    )

    constraints = tc.stack(
        (
            # Corrected numerator: D**3 * N (x_2**3 * x_1).
            1.0
            - coil_diameter**3
            * number_of_coils
            / (71785.0 * wire_diameter**4),
            (
                (4.0 * coil_diameter**2 - wire_diameter * coil_diameter)
                / (
                    12566.0
                    * (
                        coil_diameter * wire_diameter**3
                        - wire_diameter**4
                    )
                )
                + 1.0 / (5108.0 * wire_diameter**2)
                - 1.0
            ),
            1.0
            - 140.45
            * wire_diameter
            / (coil_diameter**2 * number_of_coils),
            # Corrected constraint: the original statement omitted -1.
            (coil_diameter + wire_diameter) / 1.5 - 1.0,
        ),
        dim=-1,
    )
    violation = tc.clamp(constraints, min=0.0).sum(dim=-1)
    return volume, constraints, violation


def candidate_is_better(trial_f, trial_violation, target_f, target_violation):
    """Compare candidates using feasibility rules for constrained problems."""

    trial_feasible = trial_violation <= 1e-12
    target_feasible = target_violation <= 1e-12

    if trial_feasible and not target_feasible:
        return True
    if target_feasible and not trial_feasible:
        return False
    if trial_feasible and target_feasible:
        return bool(trial_f < target_f)
    return bool(trial_violation < target_violation)


def population_statistics(objective, violation):
    """Return statistics without mixing objective and constraint violation."""

    feasible = violation <= 1e-12
    feasible_objective = objective[feasible]
    best_feasible = (
        feasible_objective.min().item()
        if feasible_objective.numel()
        else float("nan")
    )
    return {
        "objective_mean": objective.mean().item(),
        "objective_std": objective.std(unbiased=False).item(),
        "best_feasible_objective": best_feasible,
        "feasible_percentage": 100.0 * feasible.double().mean().item(),
        "mean_violation": violation.mean().item(),
    }


def save_csv(rows, output_path):
    """Write dictionaries to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def differential_evolution(
    population_size=POPULATION_SIZE,
    max_fitness_evaluations=MAX_FITNESS_EVALUATIONS,
    evaluation_step=EVALUATION_STEP,
    seed=1,
    output_directory=None,
):
    """Minimize the spring volume with self-adaptive DE/rand/1/bin."""

    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if max_fitness_evaluations < population_size:
        raise ValueError("maximum evaluations must be at least population size")

    generator = tc.Generator().manual_seed(seed)
    population = LOWER_BOUNDS + (UPPER_BOUNDS - LOWER_BOUNDS) * tc.rand(
        population_size, DIMENSION, dtype=tc.float64, generator=generator
    )
    objective, constraints, violation = spring_problem(population)
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
            **population_statistics(objective, violation),
        }
    ]
    next_curve_evaluation = fitness_evaluations + evaluation_step

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size,
            max_fitness_evaluations - fitness_evaluations,
        )
        new_population = population.clone()
        new_objective = objective.clone()
        new_constraints = constraints.clone()
        new_violation = violation.clone()
        new_differential_weights = differential_weights.clone()
        new_crossover_rates = crossover_rates.clone()

        if evaluations_this_generation < population_size:
            target_indices = tc.randperm(
                population_size, generator=generator
            )[:evaluations_this_generation].tolist()
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
            mutant = tc.maximum(tc.minimum(mutant, UPPER_BOUNDS), LOWER_BOUNDS)

            crossover_mask = (
                tc.rand(DIMENSION, generator=generator) < crossover_rate
            )
            forced_dimension = tc.randint(DIMENSION, (1,), generator=generator)
            crossover_mask[forced_dimension] = True
            trial = tc.where(crossover_mask, mutant, population[i])

            # Preserve the self-adaptation rule used in the original code.
            if crossover_mask.all():
                child_weight = (
                    differential_weights[donors[0]]
                    + differential_weight
                    * (
                        differential_weights[donors[1]]
                        - differential_weights[donors[2]]
                    )
                ).clamp(0.3, 0.9)
                child_crossover = (
                    crossover_rates[donors[0]]
                    + differential_weight
                    * (
                        crossover_rates[donors[1]]
                        - crossover_rates[donors[2]]
                    )
                ).clamp(0.9, 1.0)
            else:
                child_weight = differential_weight
                child_crossover = crossover_rate

            trial_f, trial_constraints, trial_violation = spring_problem(trial)
            fitness_evaluations += 1

            if candidate_is_better(
                trial_f,
                trial_violation,
                objective[i],
                violation[i],
            ):
                new_population[i] = trial
                new_objective[i] = trial_f
                new_constraints[i] = trial_constraints
                new_violation[i] = trial_violation
                new_differential_weights[i] = child_weight
                new_crossover_rates[i] = child_crossover

            if fitness_evaluations >= next_curve_evaluation:
                evolution_curve.append(
                    {
                        "seed": seed,
                        "fitness_evaluations": fitness_evaluations,
                        **population_statistics(new_objective, new_violation),
                    }
                )
                next_curve_evaluation += evaluation_step

        population = new_population
        objective = new_objective
        constraints = new_constraints
        violation = new_violation
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

    feasible = violation <= 1e-12
    if feasible.any():
        feasible_indices = tc.where(feasible)[0]
        best_index = feasible_indices[tc.argmin(objective[feasible])]
    else:
        best_index = tc.argmin(violation)

    best_x = population[best_index]
    result = {
        "algorithm": "Self-adaptive DE/rand/1/bin with feasibility rules",
        "seed": seed,
        "best_volume": objective[best_index].item(),
        "N": best_x[0].item(),
        "D": best_x[1].item(),
        "d": best_x[2].item(),
        "g1": constraints[best_index, 0].item(),
        "g2": constraints[best_index, 1].item(),
        "g3": constraints[best_index, 2].item(),
        "g4": constraints[best_index, 3].item(),
        "total_violation": violation[best_index].item(),
        "feasible": bool(feasible[best_index]),
        "mean_F": differential_weights.mean().item(),
        "mean_CR": crossover_rates.mean().item(),
        "fitness_evaluations": fitness_evaluations,
    }

    if output_directory is not None:
        output_directory = Path(output_directory)
        save_csv(
            evolution_curve,
            output_directory / "curves" / f"seed-{seed}.csv",
        )
    return result


def main():
    """Execute the 35 independent runs used in the dissertation."""

    output_directory = Path(__file__).resolve().parent / "spring_results"
    results = []
    for run in range(1, NUMBER_OF_RUNS + 1):
        result = differential_evolution(seed=run, output_directory=output_directory)
        results.append(result)
        print(
            f"Run {run:02d}: V={result['best_volume']:.8f}, "
            f"N={result['N']:.6f}, D={result['D']:.6f}, "
            f"d={result['d']:.6f}, feasible={result['feasible']}"
        )

    result_path = save_csv(results, output_directory / "final_results.csv")
    best = min(
        (result for result in results if result["feasible"]),
        key=lambda result: result["best_volume"],
    )
    print(f"Best result: {best}")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()