"""Shared initial loss coefficients for fixed, Adam-adaptive and DE training."""
import math

import torch

# Change these three values to change the defaults in every training entry point.
DEFAULT_LOSS_WEIGHTS = {'Initial': 1e3, 'Boundary': 1e3, 'PDE': 1e3}


def initial_loss_coefficients(values=None, split_pde=False):
    """Return named coefficients; split PDE terms each receive half the PDE total."""
    weights = dict(DEFAULT_LOSS_WEIGHTS) if values is None else dict(values)
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError('Initial weights require exactly Initial, Boundary and PDE')
    weights = {name: float(weights[name]) for name in DEFAULT_LOSS_WEIGHTS}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError('Initial loss weights must be finite and positive')
    if split_pde:
        pde = weights.pop('PDE')
        weights.update(PDE_u=pde / 2, PDE_v=pde / 2)
    return weights


def loss_log_vars(weights, *, device, dtype):
    """Convert effective coefficients w to framework parameters s = -log(w)."""
    coefficients = torch.as_tensor(list(weights), device=device, dtype=dtype)
    if (coefficients.ndim != 1 or not coefficients.numel()
            or not torch.isfinite(coefficients).all() or (coefficients <= 0).any()):
        raise ValueError('Initial coefficients must remain finite and positive in the model dtype')
    return -coefficients.log()


def add_loss_weight_arguments(parser):
    parser.add_argument('--initial-loss-weights', type=float, nargs=3,
                        metavar=('INITIAL', 'BOUNDARY', 'PDE'),
                        help='Initial coefficients shared by fixed and adaptive training; default 1000 1000 1000')


def weights_from_arguments(args, parser):
    try:
        return initial_loss_coefficients(
            dict(zip(DEFAULT_LOSS_WEIGHTS, args.initial_loss_weights))
            if args.initial_loss_weights is not None else None)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
