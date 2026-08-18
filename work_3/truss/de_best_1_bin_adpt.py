"""Self-adaptive DE/rand/1/bin runner for the 2D Rastrigin function."""

import csv
from pathlib import Path

import torch as tc


POPULATION_SIZE = 30
DIMENSION = 2
MAX_FITNESS_EVALUATIONS = 3030
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 20


def rastrigin(x):
    """Evaluate the Rastrigin function along the last tensor dimension."""

    dimension = x.shape[-1]
    return 10.0 * dimension + tc.sum(
        x**2 - 10.0 * tc.cos(2.0 * tc.pi * x), dim=-1
    )


def population_statistics(fitness):
    """Return population mean and population standard deviation."""

    return {
        "fit_mean": fitness.mean().item(),
        "fit_std": fitness.std(unbiased=False).item(),
    }


def save_evolution_curve(curve, output_directory, population_size, seed,
                         max_fitness_evaluations, evaluation_step):
    """Save one execution's convergence curve as CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"seed-{seed}--pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "seed",
                "fitness_evaluations",
                "fit_mean",
                "fit_std",
            ),
        )
        writer.writeheader()
        writer.writerows(curve)

    return output_path


def save_final_result(result, output_directory, population_size,
                      max_fitness_evaluations, evaluation_step):
    """Append one execution's final result to the configuration CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    return output_path


def differential_evolution(
    population_size,
    dimension,
    max_fitness_evaluations,
    evaluation_step,
    seed,
    lower_bound=-5.12,
    upper_bound=5.12,
    output_directory=None,
    results_directory=None,
):
    """Minimize Rastrigin with self-adaptive DE/rand/1/bin."""

    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if max_fitness_evaluations < population_size:
        raise ValueError(
            "max_fitness_evaluations must be at least population_size"
        )
    if evaluation_step < 1:
        raise ValueError("evaluation_step must be positive")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    generator = tc.Generator()
    generator.manual_seed(seed)

    population = lower_bound + (upper_bound - lower_bound) * tc.rand(
        population_size,
        dimension,
        dtype=tc.float64,
        generator=generator,
    )
    fitness = rastrigin(population)
    fitness_evaluations = population_size
    initial_statistics = population_statistics(fitness)
    evolution_curve = [
        {
            "seed": seed,
            "fitness_evaluations": fitness_evaluations,
            **initial_statistics,
        }
    ]
    next_curve_evaluation = fitness_evaluations + evaluation_step

    # Each population member carries its own F and CR values.
    differential_weights = 0.3 + 0.6 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    crossover_rates = 0.9 + 0.1 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    completed_generations = 0
    partial_generation_evaluations = 0

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size,
            max_fitness_evaluations - fitness_evaluations,
        )
        new_population = population.clone()
        new_fitness = fitness.clone()
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
            donor_indices = available[
                tc.randperm(population_size - 1, generator=generator)[:3]
            ]

            differential_weight = differential_weights[i]
            crossover_rate = crossover_rates[i]

            mutant = population[donor_indices[0]] + differential_weight * (
                population[donor_indices[1]]
                - population[donor_indices[2]]
            )
            mutant.clamp_(lower_bound, upper_bound)

            crossover_mask = (
                tc.rand(dimension, generator=generator) < crossover_rate
            )
            forced_dimension = tc.randint(
                dimension, (1,), generator=generator
            )
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

            trial_fitness = rastrigin(trial)
            fitness_evaluations += 1

            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                new_differential_weights[i] = child_differential_weight
                new_crossover_rates[i] = child_crossover_rate
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
                new_differential_weights[i] = differential_weight
                new_crossover_rates[i] = crossover_rate

            if fitness_evaluations == next_curve_evaluation:
                statistics = population_statistics(new_fitness)
                evolution_curve.append(
                    {
                        "seed": seed,
                        "fitness_evaluations": fitness_evaluations,
                        **statistics,
                    }
                )
                next_curve_evaluation += evaluation_step

        population = new_population
        fitness = new_fitness
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

        if evaluations_this_generation == population_size:
            completed_generations += 1
        else:
            partial_generation_evaluations = evaluations_this_generation

    best_index = tc.argmin(fitness)

    if evolution_curve[-1]["fitness_evaluations"] != fitness_evaluations:
        final_statistics = population_statistics(fitness)
        evolution_curve.append(
            {
                "seed": seed,
                "fitness_evaluations": fitness_evaluations,
                **final_statistics,
            }
        )

    if output_directory is None:
        output_directory = Path(__file__).resolve().parent / "evolution_curves"
    curve_path = save_evolution_curve(
        evolution_curve,
        output_directory,
        population_size,
        seed,
        max_fitness_evaluations,
        evaluation_step,
    )

    result = {
        "algorithm": "Self-adaptive DE/rand/1/bin",
        "best_f": fitness[best_index].item(),
        "population_size": population_size,
        "dimension": dimension,
        "max_fitness_evaluations": max_fitness_evaluations,
        "evaluation_step": evaluation_step,
        "completed_generations": completed_generations,
        "partial_generation_evaluations": partial_generation_evaluations,
        "seed": seed,
        "mean_differential_weight": differential_weights.mean().item(),
        "std_differential_weight": differential_weights.std(
            unbiased=False
        ).item(),
        "mean_crossover_rate": crossover_rates.mean().item(),
        "std_crossover_rate": crossover_rates.std(unbiased=False).item(),
        "fitness_evaluations": fitness_evaluations,
        "dtype": "float64",
        "device": "cpu",
        "evolution_curve_file": str(curve_path),
    }
    for index, value in enumerate(population[best_index].tolist()):
        result[f"best_x_{index}"] = value

    if results_directory is None:
        results_directory = Path(__file__).resolve().parent / "results"
    result_path = save_final_result(
        result,
        results_directory,
        population_size,
        max_fitness_evaluations,
        evaluation_step,
    )

    print(f"Evolution curve saved to: {curve_path}")
    print(f"Final result saved to: {result_path}")


def main():
    for run in range(1, NUMBER_OF_RUNS + 1):
        print(f"Run {run}/{NUMBER_OF_RUNS} with seed {run}")
        differential_evolution(
            population_size=POPULATION_SIZE,
            dimension=DIMENSION,
            max_fitness_evaluations=MAX_FITNESS_EVALUATIONS,
            evaluation_step=EVALUATION_STEP,
            seed=run,
        )


if __name__ == "__main__":
    main()
