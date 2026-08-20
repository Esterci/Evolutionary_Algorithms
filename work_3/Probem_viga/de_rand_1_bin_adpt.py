"""Self-adaptive DE/rand/1/bin for the welded beam design problem."""

import csv
from pathlib import Path

import torch as tc


POPULATION_SIZE = 30
DIMENSION = 4
MAX_FITNESS_EVALUATIONS = 320_000
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 35
FEASIBILITY_TOLERANCE = 1e-10

# Decision vector: x = [h, l, t, b].
LOWER_BOUNDS = tc.tensor([0.125, 0.1, 0.1, 0.1], dtype=tc.float64)
UPPER_BOUNDS = tc.tensor([10.0, 10.0, 10.0, 10.0], dtype=tc.float64)


def welded_beam_problem(x):
    """Return cost, constraints, violation and mechanical quantities."""

    h = x[..., 0]
    length = x[..., 1]
    t = x[..., 2]
    b = x[..., 3]

    cost = 1.10471 * h**2 * length + 0.04811 * t * b * (14.0 + length)

    alpha = tc.sqrt(0.25 * (length**2 + (h + t) ** 2))

    # Correction: the classical formulation uses h*l in the denominator.
    # Using h*t, as printed in the supplied PDF, does not reproduce Table 5.16.
    tau_prime = 6000.0 / ((2.0**0.5) * h * length)
    tau_double_prime = (
        6000.0
        * (14.0 + 0.5 * length)
        * alpha
        / (
            2.0
            * (
                0.707
                * h
                * length
                * (length**2 / 12.0 + 0.25 * (h + t) ** 2)
            )
        )
    )
    tau = tc.sqrt(
        tau_prime**2
        + tau_double_prime**2
        + length * tau_prime * tau_double_prime / alpha
    )
    sigma = 504000.0 / (t**2 * b)
    critical_load = 64746.022 * (1.0 - 0.0282346 * t) * t * b**3
    deflection = 2.1952 / (t**3 * b)

    # All constraints follow the PDF convention g_i(x) >= 0.
    constraints = tc.stack(
        (
            13600.0 - tau,
            30000.0 - sigma,
            b - h,
            critical_load - 6000.0,
            0.25 - deflection,
        ),
        dim=-1,
    )
    violation = tc.clamp(-constraints, min=0.0).sum(dim=-1)
    quantities = tc.stack((tau, sigma, critical_load, deflection), dim=-1)
    return cost, constraints, violation, quantities


def candidate_is_better(trial_f, trial_v, target_f, target_v):
    """Compare two candidates using feasibility rules."""

    trial_feasible = trial_v <= FEASIBILITY_TOLERANCE
    target_feasible = target_v <= FEASIBILITY_TOLERANCE
    if trial_feasible and not target_feasible:
        return True
    if target_feasible and not trial_feasible:
        return False
    if trial_feasible and target_feasible:
        return bool(trial_f < target_f)
    return bool(trial_v < target_v)


def population_statistics(objective, violation):
    """Return objective and feasibility statistics for one population."""

    feasible = violation <= FEASIBILITY_TOLERANCE
    feasible_values = objective[feasible]
    return {
        "objective_mean": objective.mean().item(),
        "objective_std": objective.std(unbiased=False).item(),
        "best_feasible_objective": (
            feasible_values.min().item()
            if feasible_values.numel()
            else float("nan")
        ),
        "feasible_percentage": 100.0 * feasible.double().mean().item(),
        "mean_violation": violation.mean().item(),
    }


def save_csv(rows, output_path):
    """Write a list of dictionaries to CSV."""

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
    """Minimize welded-beam cost with self-adaptive DE/rand/1/bin."""

    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if max_fitness_evaluations < population_size:
        raise ValueError("maximum evaluations must be at least population size")

    generator = tc.Generator().manual_seed(seed)
    population = LOWER_BOUNDS + (UPPER_BOUNDS - LOWER_BOUNDS) * tc.rand(
        population_size, DIMENSION, dtype=tc.float64, generator=generator
    )
    objective, constraints, violation, quantities = welded_beam_problem(population)
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
        **population_statistics(objective, violation),
    }]
    next_curve_evaluation = fitness_evaluations + evaluation_step

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size, max_fitness_evaluations - fitness_evaluations
        )
        new_population = population.clone()
        new_objective = objective.clone()
        new_constraints = constraints.clone()
        new_violation = violation.clone()
        new_quantities = quantities.clone()
        new_weights = differential_weights.clone()
        new_rates = crossover_rates.clone()

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
            weight = differential_weights[i]
            crossover_rate = crossover_rates[i]

            mutant = population[donors[0]] + weight * (
                population[donors[1]] - population[donors[2]]
            )
            mutant = tc.maximum(tc.minimum(mutant, UPPER_BOUNDS), LOWER_BOUNDS)

            crossover_mask = tc.rand(
                DIMENSION, generator=generator
            ) < crossover_rate
            forced_dimension = tc.randint(DIMENSION, (1,), generator=generator)
            crossover_mask[forced_dimension] = True
            trial = tc.where(crossover_mask, mutant, population[i])

            # Preserve the self-adaptation mechanism of the original code.
            if crossover_mask.all():
                child_weight = (
                    differential_weights[donors[0]]
                    + weight
                    * (
                        differential_weights[donors[1]]
                        - differential_weights[donors[2]]
                    )
                ).clamp(0.3, 0.9)
                child_rate = (
                    crossover_rates[donors[0]]
                    + weight
                    * (crossover_rates[donors[1]] - crossover_rates[donors[2]])
                ).clamp(0.9, 1.0)
            else:
                child_weight = weight
                child_rate = crossover_rate

            trial_f, trial_g, trial_v, trial_q = welded_beam_problem(trial)
            fitness_evaluations += 1

            if candidate_is_better(
                trial_f, trial_v, objective[i], violation[i]
            ):
                new_population[i] = trial
                new_objective[i] = trial_f
                new_constraints[i] = trial_g
                new_violation[i] = trial_v
                new_quantities[i] = trial_q
                new_weights[i] = child_weight
                new_rates[i] = child_rate

            if fitness_evaluations >= next_curve_evaluation:
                evolution_curve.append({
                    "seed": seed,
                    "fitness_evaluations": fitness_evaluations,
                    **population_statistics(new_objective, new_violation),
                })
                next_curve_evaluation += evaluation_step

        population = new_population
        objective = new_objective
        constraints = new_constraints
        violation = new_violation
        quantities = new_quantities
        differential_weights = new_weights
        crossover_rates = new_rates

    feasible = violation <= FEASIBILITY_TOLERANCE
    if feasible.any():
        feasible_indices = tc.where(feasible)[0]
        best_index = feasible_indices[tc.argmin(objective[feasible])]
    else:
        best_index = tc.argmin(violation)

    best_x = population[best_index]
    result = {
        "algorithm": "Self-adaptive DE/rand/1/bin with feasibility rules",
        "seed": seed,
        "best_cost": objective[best_index].item(),
        "h": best_x[0].item(),
        "l": best_x[1].item(),
        "t": best_x[2].item(),
        "b": best_x[3].item(),
        "g1": constraints[best_index, 0].item(),
        "g2": constraints[best_index, 1].item(),
        "g3": constraints[best_index, 2].item(),
        "g4": constraints[best_index, 3].item(),
        "g5": constraints[best_index, 4].item(),
        "tau": quantities[best_index, 0].item(),
        "sigma": quantities[best_index, 1].item(),
        "critical_load": quantities[best_index, 2].item(),
        "deflection": quantities[best_index, 3].item(),
        "total_violation": violation[best_index].item(),
        "feasible": bool(feasible[best_index]),
        "mean_F": differential_weights.mean().item(),
        "mean_CR": crossover_rates.mean().item(),
        "fitness_evaluations": fitness_evaluations,
    }

    if output_directory is not None:
        save_csv(
            evolution_curve,
            Path(output_directory) / "curves" / f"seed-{seed}.csv",
        )
    return result


def main():
    """Execute the 35 independent runs used in the dissertation."""

    output_directory = Path(__file__).resolve().parent / "welded_beam_results"
    results = []
    for run in range(1, NUMBER_OF_RUNS + 1):
        result = differential_evolution(seed=run, output_directory=output_directory)
        results.append(result)
        print(
            f"Run {run:02d}: C={result['best_cost']:.8f}, "
            f"h={result['h']:.6f}, l={result['l']:.6f}, "
            f"t={result['t']:.6f}, b={result['b']:.6f}, "
            f"feasible={result['feasible']}"
        )

    result_path = save_csv(results, output_directory / "final_results.csv")
    feasible_results = [result for result in results if result["feasible"]]
    if not feasible_results:
        raise RuntimeError("No feasible solution was found in the 35 runs")
    best = min(feasible_results, key=lambda result: result["best_cost"])
    print(f"Best result: {best}")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()