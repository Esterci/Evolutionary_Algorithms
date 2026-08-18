"""Self-adaptive DE/rand/1/bin runner for the 2D Rastrigin function."""

import argparse
import json

import torch as tc


def rastrigin(x):
    """Evaluate the Rastrigin function along the last tensor dimension."""

    dimension = x.shape[-1]
    return 10.0 * dimension + tc.sum(
        x**2 - 10.0 * tc.cos(2.0 * tc.pi * x), dim=-1
    )


def differential_evolution(
    population_size,
    dimension,
    generations,
    seed,
    lower_bound=-5.12,
    upper_bound=5.12,
):
    """Minimize Rastrigin with self-adaptive DE/rand/1/bin."""

    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if generations < 1:
        raise ValueError("generations must be positive")
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

    # Each population member carries its own F and CR values.
    differential_weights = 0.3 + 0.6 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    crossover_rates = 0.9 + 0.1 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    for _ in range(generations):
        new_population = tc.empty_like(population)
        new_fitness = tc.empty_like(fitness)
        new_differential_weights = tc.empty_like(differential_weights)
        new_crossover_rates = tc.empty_like(crossover_rates)

        for i in range(population_size):
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

        population = new_population
        fitness = new_fitness
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

    best_index = tc.argmin(fitness)

    return {
        "algorithm": "Self-adaptive DE/rand/1/bin",
        "best_x": population[best_index].tolist(),
        "best_f": fitness[best_index].item(),
        "population_size": population_size,
        "dimension": dimension,
        "generations": generations,
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
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run self-adaptive DE/rand/1/bin on 2D Rastrigin."
    )
    parser.add_argument("population_size", type=int)
    parser.add_argument("generations", type=int)
    parser.add_argument("seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    result = differential_evolution(
        population_size=args.population_size,
        dimension=2,
        generations=args.generations,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
