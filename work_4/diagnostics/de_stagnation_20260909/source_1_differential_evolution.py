"""Self-adaptive DE/rand/1/bin for a vector of PINN parameters.

The optional legacy control adaptation follows
``work_3/rastrigin/de_best_1_bin_adpt.py``. Despite that reference filename,
its mutation strategy is DE/rand/1/bin. This module only searches vectors;
the caller constructs the PINN objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class DEConfig:
    """Configuration for synchronous generations with clipped bounds.

    ``generations`` excludes initialization: the evaluation budget is
    ``population_size * (generations + 1)``. Bounds apply to every genome
    coordinate unless per-coordinate bounds are supplied. The first
    individual is the caller's initial vector. Other individuals are
    uniform within the bounds, or uniform perturbations of that vector
    within +/- ``initial_spread`` times each coordinate's bound width,
    clipped to the bounds when ``initialization='local'``.
    """

    population_size: int = 20
    generations: int = 100
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    seed: int = 2027
    control_adaptation: str = "jde"
    adaptation_probability: float = 0.1
    initialization: str = "uniform"
    initial_spread: float = 0.05

    def __post_init__(self) -> None:
        for name, minimum in (("population_size", 4), ("generations", 1), ("seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**63:
            raise ValueError("seed must be smaller than 2**63")
        for name in ("lower_bound", "upper_bound"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")
        if self.control_adaptation not in ("jde", "legacy"):
            raise ValueError("control_adaptation must be 'jde' or 'legacy'")
        if self.initialization not in ("uniform", "local"):
            raise ValueError("initialization must be 'uniform' or 'local'")
        for name in ("adaptation_probability", "initial_spread"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if not 0.0 <= self.adaptation_probability <= 1.0:
            raise ValueError("adaptation_probability must be between zero and one")
        if self.initial_spread <= 0.0:
            raise ValueError("initial_spread must be positive")


@dataclass
class DEResult:
    """Detached final population, best genome, and evaluation history."""

    best_vector: Tensor
    best_fitness: float
    population: Tensor
    fitness: Tensor
    mutation_factors: Tensor
    crossover_rates: Tensor
    history: list[dict[str, int | float]]
    fitness_evaluations: int
    nonfinite_evaluations: int


def differential_evolution(
    objective: Callable[[Tensor], Tensor | float],
    initial_vector: Tensor,
    config: DEConfig,
    *,
    bounds: tuple[Tensor, Tensor] | None = None,
    on_generation: Callable[[dict[str, int | float]], None] | None = None,
) -> DEResult:
    """Minimize a fixed scalar objective over the initial vector's shape.

    Three distinct donors exclude the target. Binomial crossover forces
    one mutant coordinate; strict improvement selects the survivor. All
    donors come from the previous generation, so updates are synchronous.
    Each individual carries F in [0.3, 0.9]. By default, F and CR are each
    independently resampled with ``adaptation_probability`` before trial
    construction, with CR in [0, 1]. Proposed controls survive only when
    the trial improves its target. This jDE-style update does not require
    all genome coordinates to cross. The ``legacy`` mode keeps CR in
    [0.9, 1.0] and work_3's donor-difference control update gated by all
    coordinates crossing; that event is rare for large network genomes.

    ``bounds`` can supply two [D] tensors matching the genome's dtype and
    device; each coordinate then has its own interval. No broadcasting of
    supplied bounds is allowed. Scalar config bounds are used otherwise.

    Candidate vectors are detached copies on the initial vector's device
    and dtype. Autograd remains enabled while calling the objective, so
    PINN residuals may differentiate with respect to their input points.
    The objective must use fixed data and a fixed scalar formula for cached
    parent fitness to remain comparable. Any gene-dependent loss weights
    must be decoded deterministically from the evaluated genome; mutable
    external weights must not change a cached genome's fitness.
    Nonfinite scalar values count as failed evaluations and receive +inf;
    programming exceptions propagate.
    ``on_generation`` receives a copy of each history row, including
    initialization (generation zero), for streaming progress or artifacts.
    History also reports accepted trials, control-update proposal and
    acceptance counts (individuals with at least one proposed control),
    mean F/CR, and population_spread: the mean coordinate standard
    deviation divided by that coordinate's bound width. Generation-zero
    acceptance/proposal counts are zero. Spread is dimensionless, so
    different gene units do not distort its scale.
    """
    if not isinstance(config, DEConfig):
        raise TypeError("config must be a DEConfig")
    if not isinstance(initial_vector, Tensor):
        raise TypeError("initial_vector must be a torch.Tensor")
    if initial_vector.ndim != 1 or initial_vector.numel() == 0:
        raise ValueError("initial_vector must be a nonempty one-dimensional tensor")
    if not initial_vector.is_floating_point():
        raise ValueError("initial_vector must use a floating-point dtype")
    if not torch.isfinite(initial_vector).all():
        raise ValueError("initial_vector must contain only finite values")

    device, dtype = initial_vector.device, initial_vector.dtype
    if bounds is None:
        lower = (
            torch.tensor(config.lower_bound, device=device, dtype=dtype)
            .expand_as(initial_vector)
            .clone()
        )
        upper = (
            torch.tensor(config.upper_bound, device=device, dtype=dtype)
            .expand_as(initial_vector)
            .clone()
        )
    else:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("bounds must contain lower and upper [D] tensors")
        if any(not isinstance(bound, Tensor) for bound in bounds):
            raise ValueError("bounds must contain lower and upper [D] tensors")
        if any(bound.shape != initial_vector.shape for bound in bounds):
            raise ValueError("bounds must have exactly the initial_vector shape")
        if any(bound.device != device or bound.dtype != dtype for bound in bounds):
            raise ValueError("bounds must match initial_vector device and dtype")
        lower, upper = (bound.detach().clone() for bound in bounds)
    if (
        not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
        or not (lower < upper).all()
    ):
        raise ValueError("bounds must be finite and distinct in initial_vector's dtype")
    if not torch.isfinite(upper - lower).all():
        raise ValueError("bound width must be finite in initial_vector's dtype")
    if torch.any(initial_vector < lower) or torch.any(initial_vector > upper):
        raise ValueError("initial_vector must lie inside the DE bounds")

    generator = torch.Generator(device=device).manual_seed(int(config.seed))
    population_size, dimension = config.population_size, initial_vector.numel()
    samples = torch.rand(
        population_size, dimension, device=device, dtype=dtype, generator=generator
    )
    if config.initialization == "local":
        population = (
            initial_vector.detach()
            + (2.0 * samples - 1.0) * (upper - lower) * config.initial_spread
        )
        population = torch.maximum(torch.minimum(population, upper), lower)
    else:
        population = lower + (upper - lower) * samples
    population[0].copy_(initial_vector.detach())
    fitness = torch.empty(population_size, device=device, dtype=dtype)
    fitness_evaluations = 0
    nonfinite_evaluations = 0

    def evaluate(vector: Tensor) -> float:
        nonlocal fitness_evaluations, nonfinite_evaluations
        with torch.enable_grad():
            value = objective(vector.detach().clone())
        if isinstance(value, Tensor):
            if value.ndim != 0 or value.is_complex():
                raise ValueError("objective must return a real scalar")
            value = value.detach().item()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("objective must return a real scalar")
        fitness_evaluations += 1
        # Conversion to the genome precision must not turn finite fitness
        # into an uncounted failure when ranking the population.
        value = float(value)
        if not math.isfinite(value) or abs(value) > torch.finfo(dtype).max:
            nonfinite_evaluations += 1
            return math.inf
        return value

    for index in range(population_size):
        fitness[index] = evaluate(population[index])
    if not torch.isfinite(fitness).any():
        raise ValueError("initial DE population has no finite objective value")

    mutation_factors = 0.3 + 0.6 * torch.rand(
        population_size, device=device, dtype=dtype, generator=generator
    )
    crossover_rates = torch.rand(
        population_size, device=device, dtype=dtype, generator=generator
    )
    if config.control_adaptation == "legacy":
        crossover_rates = 0.9 + 0.1 * crossover_rates
    history: list[dict[str, int | float]] = []

    def record_generation(
        generation: int,
        accepted_trials: int = 0,
        adaptation_attempts: int = 0,
        adaptation_acceptances: int = 0,
    ) -> None:
        # Infinite failed fitness has no finite population standard
        # deviation; report +inf explicitly instead of generating NaN.
        all_finite = bool(torch.isfinite(fitness).all())
        history.append(
            {
                "generation": generation,
                "fitness_evaluations": fitness_evaluations,
                "best_fitness": fitness.min().item(),
                "mean_fitness": fitness.mean().item() if all_finite else math.inf,
                "std_fitness": (
                    fitness.std(unbiased=False).item() if all_finite else math.inf
                ),
                "accepted_trials": accepted_trials,
                "control_adaptation_attempts": adaptation_attempts,
                "control_adaptation_acceptances": adaptation_acceptances,
                "population_spread": (
                    population.std(dim=0, unbiased=False) / (upper - lower)
                )
                .mean()
                .item(),
                "mean_mutation_factor": mutation_factors.mean().item(),
                "mean_crossover_rate": crossover_rates.mean().item(),
            }
        )
        if on_generation is not None:
            on_generation(history[-1].copy())

    record_generation(0)
    for generation in range(1, config.generations + 1):
        new_population = population.clone()
        new_fitness = fitness.clone()
        new_mutation_factors = mutation_factors.clone()
        new_crossover_rates = crossover_rates.clone()
        accepted_trials = adaptation_attempts = adaptation_acceptances = 0

        for index in range(population_size):
            available = torch.arange(population_size, device=device)
            available = available[available != index]
            donors = available[
                torch.randperm(population_size - 1, device=device, generator=generator)[
                    :3
                ]
            ]
            factor, rate = mutation_factors[index], crossover_rates[index]
            proposed_control = False
            if config.control_adaptation == "jde":
                if (
                    torch.rand((), device=device, dtype=dtype, generator=generator)
                    < config.adaptation_probability
                ):
                    factor = 0.3 + 0.6 * torch.rand(
                        (), device=device, dtype=dtype, generator=generator
                    )
                    proposed_control = True
                if (
                    torch.rand((), device=device, dtype=dtype, generator=generator)
                    < config.adaptation_probability
                ):
                    rate = torch.rand(
                        (), device=device, dtype=dtype, generator=generator
                    )
                    proposed_control = True
            mutant = population[donors[0]] + factor * (
                population[donors[1]] - population[donors[2]]
            )
            mutant = torch.maximum(torch.minimum(mutant, upper), lower)

            crossover_mask = (
                torch.rand(dimension, device=device, dtype=dtype, generator=generator)
                < rate
            )
            forced_dimension = torch.randint(
                dimension, (1,), device=device, generator=generator
            )
            crossover_mask[forced_dimension] = True
            trial = torch.where(crossover_mask, mutant, population[index])

            if config.control_adaptation == "legacy" and crossover_mask.all():
                proposed_control = True
                child_factor = (
                    mutation_factors[donors[0]]
                    + factor
                    * (mutation_factors[donors[1]] - mutation_factors[donors[2]])
                ).clamp(0.3, 0.9)
                child_rate = (
                    crossover_rates[donors[0]]
                    + factor * (crossover_rates[donors[1]] - crossover_rates[donors[2]])
                ).clamp(0.9, 1.0)
            else:
                child_factor, child_rate = factor, rate

            adaptation_attempts += int(proposed_control)
            trial_fitness = evaluate(trial)
            if trial_fitness < fitness[index]:
                accepted_trials += 1
                adaptation_acceptances += int(proposed_control)
                new_population[index] = trial
                new_fitness[index] = trial_fitness
                new_mutation_factors[index] = child_factor
                new_crossover_rates[index] = child_rate

        population, fitness = new_population, new_fitness
        mutation_factors, crossover_rates = new_mutation_factors, new_crossover_rates
        record_generation(
            generation, accepted_trials, adaptation_attempts, adaptation_acceptances
        )

    best_index = fitness.argmin()
    return DEResult(
        best_vector=population[best_index].detach().clone(),
        best_fitness=fitness[best_index].item(),
        population=population.detach().clone(),
        fitness=fitness.detach().clone(),
        mutation_factors=mutation_factors.detach().clone(),
        crossover_rates=crossover_rates.detach().clone(),
        history=history,
        fitness_evaluations=fitness_evaluations,
        nonfinite_evaluations=nonfinite_evaluations,
    )
