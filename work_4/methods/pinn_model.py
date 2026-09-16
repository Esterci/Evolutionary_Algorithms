"""Data-free conservative Burgers PINN extracted from pinn_fvm_collocation.ipynb.

Inputs [N, 3] are physical (t, x, y); outputs [N, 2] are (u, v).
Equations: u_t + (u²/2)_x + (uv)_y = nu*(u_xx + u_yy),
           v_t + (uv)_x + (v²/2)_y = nu*(v_xx + v_yy).
The JSON domain and analytical initial condition are used without changing units
or nondimensionalizing the PDE. Only network inputs are scaled to [-1, 1].
All four faces have homogeneous Neumann conditions. No parameters are inferred
and no observational/FVM targets enter training. Full deterministic collocation
is regenerated as fresh autograd leaves each iteration, without subsampling.
"""

import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

BASE_DIRECTORY = Path(__file__).resolve().parents[1]
FRAMEWORK_DIRECTORY = (
    Path(
        os.environ.get("FISIOCOMPINN_PATH", BASE_DIRECTORY.parent.parent / "Pinn-Torch")
    )
    .expanduser()
    .resolve()
)
if (FRAMEWORK_DIRECTORY / "fisiocomPinn").is_dir():
    sys.path.insert(0, str(FRAMEWORK_DIRECTORY))
from methods.loss_weight_config import initial_loss_coefficients, loss_log_vars
from fisiocomPinn.Loss import LOSS
from fisiocomPinn.Net import FullyConnectedNetwork, activation_map
from fisiocomPinn.Trainer import AdaptiveLossWeights, Trainer


def number_of_intervals(domain, spacing, name):
    if len(domain) != 2 or domain[1] <= domain[0] or spacing <= 0:
        raise ValueError(f"{name} requires increasing bounds and positive spacing")
    intervals = (domain[1] - domain[0]) / spacing
    rounded = round(intervals)
    if not np.isclose(intervals, rounded, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} length must be an integer multiple of its spacing")
    return int(rounded)


def require_full_batch(requested_size, expected_size, loss_name):
    if int(requested_size) != expected_size:
        raise ValueError(
            f"{loss_name} requires all {expected_size} deterministic points; "
            f"received batch_size={requested_size}."
        )


def resolve_activations(hidden_sizes, names=None):
    """Validate deterministic, shape-preserving hidden activations from the framework."""
    supported = {name.lower(): name for name in activation_map
                 if name not in ('Linear', 'GLU', 'RReLU')}
    names = ['Tanh'] * len(hidden_sizes) if names is None else list(names)
    if len(names) != len(hidden_sizes):
        raise ValueError('Provide exactly one activation function per hidden layer')
    if any(name.lower() not in supported for name in names):
        raise ValueError('Supported activation functions: ' + ', '.join(supported.values()))
    return [supported[name.lower()] for name in names]


class ConfigurableFullyConnectedNetwork(FullyConnectedNetwork):
    """Local framework extension: hidden activations with the existing linear output.

    FullyConnectedNetworkMod adds an output Tanh, which changes the Burgers
    velocity range. Override only layer construction to preserve that range.
    """
    def __init__(self, hidden_sizes, activation_functions, dtype):
        self.activation_functions = resolve_activations(hidden_sizes, activation_functions)
        super().__init__(3, 2, hidden_sizes, dtype=dtype)

    def initFullyConnected(self, input_shape, output_shape, hidden_sizes):
        for size, name in zip(hidden_sizes, self.activation_functions):
            self.layers.append(nn.Linear(input_shape, size))
            self.layers.append(activation_map[name]())
            input_shape = size
        self.layers.append(nn.Linear(input_shape, output_shape))


class NormalizedFisiocomPINN(nn.Module):
    def __init__(self, problem, hidden_sizes, activation_functions=None):
        super().__init__()
        self.backbone = ConfigurableFullyConnectedNetwork(hidden_sizes, activation_functions, problem.dtype)
        self.register_buffer(
            "lower_bounds",
            torch.tensor(
                [problem.t_domain[0], problem.x_domain[0], problem.y_domain[0]],
                dtype=problem.dtype,
            ),
        )
        self.register_buffer(
            "upper_bounds",
            torch.tensor(
                [problem.t_domain[1], problem.x_domain[1], problem.y_domain[1]],
                dtype=problem.dtype,
            ),
        )

    def forward(self, coordinates):
        normalized = (
            2.0
            * (coordinates - self.lower_bounds)
            / (self.upper_bounds - self.lower_bounds)
            - 1.0
        )
        return self.backbone(normalized)


def initialize_linear_layer(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def derivative(field, coordinates):
    return torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def evaluate_model(coordinates, model):
    return model(coordinates)


def burgers_residual(coordinates, model, viscosity):
    prediction = model(coordinates)
    u = prediction[:, 0:1]
    v = prediction[:, 1:2]

    gradient_u = derivative(u, coordinates)
    gradient_v = derivative(v, coordinates)
    u_t, u_x, u_y = gradient_u.split(1, dim=1)
    v_t, v_x, v_y = gradient_v.split(1, dim=1)

    u_xx = derivative(u_x, coordinates)[:, 1:2]
    u_yy = derivative(u_y, coordinates)[:, 2:3]
    v_xx = derivative(v_x, coordinates)[:, 1:2]
    v_yy = derivative(v_y, coordinates)[:, 2:3]

    flux_x_u_x = derivative(0.5 * u.square(), coordinates)[:, 1:2]
    flux_y_u_y = derivative(u * v, coordinates)[:, 2:3]
    flux_x_v_x = derivative(u * v, coordinates)[:, 1:2]
    flux_y_v_y = derivative(0.5 * v.square(), coordinates)[:, 2:3]

    residual_u = u_t + flux_x_u_x + flux_y_u_y - viscosity * (u_xx + u_yy)
    residual_v = v_t + flux_x_v_x + flux_y_v_y - viscosity * (v_xx + v_yy)
    return torch.cat([residual_u, residual_v], dim=1)


def burgers_component_residual(coordinates, model, viscosity, component):
    """Return one conservative PDE residual as [N, 1], without broadcasting."""
    if component not in (0, 1):
        raise ValueError("The PDE component must be 0 (u) or 1 (v)")
    return burgers_residual(coordinates, model, viscosity)[:, component : component + 1]


def zero_gradient_boundary_residual(batch, model):
    coordinates, derivative_columns = batch
    prediction = model(coordinates)
    gradient_u = derivative(prediction[:, 0:1], coordinates)
    gradient_v = derivative(prediction[:, 1:2], coordinates)

    # Select the normal derivatives in each boundary
    columns = derivative_columns.reshape(-1, 1)
    normal_derivative_u = torch.gather(gradient_u, dim=1, index=columns)
    normal_derivative_v = torch.gather(gradient_v, dim=1, index=columns)

    return torch.cat([normal_derivative_u, normal_derivative_v], dim=1)


class BurgersProblem:
    """Physical configuration and deterministic full-grid loss generators."""

    def __init__(self, mesh, constants, dtype=torch.float32):
        self.mesh = dict(mesh)
        self.constants = dict(constants)
        self.dtype = dtype
        self.h = float(mesh["h"])
        self.k = float(mesh["k"])
        self.x_domain = tuple(map(float, mesh["x_dom"]))
        self.y_domain = tuple(map(float, mesh["y_dom"]))
        self.t_domain = tuple(map(float, mesh["t_dom"]))
        self.nu = float(constants["nu"])
        self.boundary = constants.get("boundary", "periodic")
        self.mean_velocity_u = float(constants.get("mean_velocity_u", 0.0))
        self.mean_velocity_v = float(constants.get("mean_velocity_v", 0.0))
        self.perturbation_amplitude = float(
            constants.get("perturbation_amplitude", 1.0)
        )

        if not np.isfinite(
            [
                self.h,
                self.k,
                self.nu,
                self.mean_velocity_u,
                self.mean_velocity_v,
                self.perturbation_amplitude,
            ]
        ).all():
            raise ValueError("Physical parameters must be finite")
        if self.perturbation_amplitude < 0:
            raise ValueError("perturbation_amplitude must be non-negative")
        if self.h <= 0.0 or self.k <= 0.0 or self.nu < 0.0:
            raise ValueError("h and k must be positive and nu must be non-negative")
        if self.boundary != "zero_gradient":
            raise NotImplementedError(
                "This module currently implements the zero_gradient configuration. "
                "Periodic PINN constraints require matching values and derivatives on opposite faces."
            )

        self.number_of_x_cells = number_of_intervals(self.x_domain, self.h, "x_dom")
        self.number_of_y_cells = number_of_intervals(self.y_domain, self.h, "y_dom")
        self.number_of_time_intervals = number_of_intervals(
            self.t_domain, self.k, "t_dom"
        )
        self.number_of_time_levels = self.number_of_time_intervals + 1

        self.number_of_pde_points = (
            self.number_of_time_intervals
            * self.number_of_x_cells
            * self.number_of_y_cells
        )
        self.number_of_initial_points = self.number_of_x_cells * self.number_of_y_cells
        self.number_of_boundary_points = (
            self.number_of_time_levels
            * 2
            * (self.number_of_x_cells + self.number_of_y_cells)
        )

    def fvm_cell_center_axes(self, device):
        """Generate spatial fvm grid colocation points"""
        x = (
            self.x_domain[0]
            + (
                torch.arange(self.number_of_x_cells, device=device, dtype=self.dtype)
                + 0.5
            )
            * self.h
        )
        y = (
            self.y_domain[0]
            + (
                torch.arange(self.number_of_y_cells, device=device, dtype=self.dtype)
                + 0.5
            )
            * self.h
        )
        return x, y

    def generate_full_pde_grid(self, batch_size, device):
        """Generate all fvm grid colocation points"""
        require_full_batch(batch_size, self.number_of_pde_points, "PDE loss")
        t = (
            self.t_domain[0]
            + torch.arange(
                1, self.number_of_time_levels, device=device, dtype=self.dtype
            )
            * self.k
        )
        x, y = self.fvm_cell_center_axes(device)
        t_grid, x_grid, y_grid = torch.meshgrid(t, x, y, indexing="ij")
        coordinates = torch.stack(
            [t_grid.reshape(-1), x_grid.reshape(-1), y_grid.reshape(-1)], dim=1
        ).requires_grad_(True)
        target = torch.zeros(
            (self.number_of_pde_points, 2), dtype=self.dtype, device=device
        )
        return coordinates, target

    def generate_full_pde_component_grid(self, batch_size, device):
        """Use the same full collocation grid with a scalar zero target [N, 1]."""
        coordinates, target = self.generate_full_pde_grid(batch_size, device)
        return coordinates, target[:, 0:1]

    def generate_full_initial_grid(self, batch_size, device):
        """Generate fvm initial condition grid colocation points"""
        require_full_batch(batch_size, self.number_of_initial_points, "Initial loss")
        x, y = self.fvm_cell_center_axes(device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        t_flat = torch.full_like(x_flat, self.t_domain[0])
        coordinates = torch.stack([t_flat, x_flat, y_flat], dim=1).requires_grad_(True)

        phase_x = (
            2.0
            * torch.pi
            * (x_flat - self.x_domain[0])
            / (self.x_domain[1] - self.x_domain[0])
        )
        phase_y = (
            2.0
            * torch.pi
            * (y_flat - self.y_domain[0])
            / (self.y_domain[1] - self.y_domain[0])
        )
        initial_u = self.mean_velocity_u + self.perturbation_amplitude * torch.sin(
            phase_x
        ) * torch.cos(phase_y)
        initial_v = self.mean_velocity_v - self.perturbation_amplitude * torch.cos(
            phase_x
        ) * torch.sin(phase_y)
        target = torch.stack([initial_u, initial_v], dim=1)
        return coordinates, target

    def generate_full_boundary_grid(self, batch_size, device):
        require_full_batch(batch_size, self.number_of_boundary_points, "Boundary loss")
        t = (
            self.t_domain[0]
            + torch.arange(self.number_of_time_levels, device=device, dtype=self.dtype)
            * self.k
        )
        x, y = self.fvm_cell_center_axes(device)
        t_vertical, y_vertical = torch.meshgrid(t, y, indexing="ij")
        t_horizontal, x_horizontal = torch.meshgrid(t, x, indexing="ij")

        vertical_count = t_vertical.numel()
        horizontal_count = t_horizontal.numel()
        left = torch.stack(
            [
                t_vertical.reshape(-1),
                torch.full(
                    (vertical_count,), self.x_domain[0], dtype=self.dtype, device=device
                ),
                y_vertical.reshape(-1),
            ],
            dim=1,
        )
        right = left.clone()
        right[:, 1] = self.x_domain[1]
        bottom = torch.stack(
            [
                t_horizontal.reshape(-1),
                x_horizontal.reshape(-1),
                torch.full(
                    (horizontal_count,),
                    self.y_domain[0],
                    dtype=self.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        top = bottom.clone()
        top[:, 2] = self.y_domain[1]

        coordinates = torch.cat([left, right, bottom, top], dim=0).requires_grad_(True)
        derivative_columns = torch.cat(
            [
                torch.ones(2 * vertical_count, dtype=torch.int64, device=device),
                torch.full(
                    (2 * horizontal_count,), 2, dtype=torch.int64, device=device
                ),
            ]
        )
        target = torch.zeros(
            (self.number_of_boundary_points, 2), dtype=self.dtype, device=device
        )
        return (coordinates, derivative_columns), target


def build_trainer(
    problem,
    hidden_sizes=(32, 32, 32),
    device="cpu",
    epochs=5000,
    lr=1e-3,
    adaptive=True,
    print_steps=250,
    split_pde=False,
    training_controls=None,
    activation_functions=None,
    initial_loss_weights=None,
):
    """Register IC, Neumann and conservative PDE terms in FisiocomPINN.

    Fixed and adaptive modes start from the same configured coefficients.
    Adaptive mode explicitly owns the framework weight module and Adam so log
    variances match the network precision and can be saved after training.
    With split_pde=True, the two PDE terms have scalar targets [N, 1] and
    coefficients half the combined PDE coefficient, preserving its MSE scaling.
    Separate LOSS evaluations regenerate their own derivative graphs.
    """
    coefficients = initial_loss_coefficients(initial_loss_weights, split_pde)
    model = NormalizedFisiocomPINN(problem, hidden_sizes, activation_functions).to(
        device=device, dtype=problem.dtype
    )
    model.apply(initialize_linear_layer)
    adaptive_weights = None
    optimizer = None

    if adaptive:
        adaptive_weights = AdaptiveLossWeights(4 if split_pde else 3).to(
            device=device, dtype=problem.dtype
        )
        with torch.no_grad():
            adaptive_weights.log_vars.copy_(loss_log_vars(coefficients.values(), device=device, dtype=problem.dtype))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(adaptive_weights.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
        )
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    from methods.training_controls import trainer_control_kwargs

    control_kwargs = trainer_control_kwargs(
        optimizer, epochs, training_controls
    )  # Initialize kwargs by default if none is given

    trainer = Trainer(
        n_epochs=epochs,
        model=model,
        device=device,
        adaptive=adaptive,
        # lr=lr,
        # betas=(0.9, 0.999),
        print_steps=print_steps,
        **control_kwargs,
        optimizer=optimizer,
    )
    
    if adaptive:
        # The local Trainer reuses a supplied weight module with an external
        # optimizer; its internal-Adam path recreates float32 log variances.
        trainer.adaptive_weights = adaptive_weights
        
    terms = [
        (
            "Initial",
            problem.number_of_initial_points,
            problem.generate_full_initial_grid,
            evaluate_model,
            (),
            coefficients["Initial"],
        ),
        (
            "Boundary",
            problem.number_of_boundary_points,
            problem.generate_full_boundary_grid,
            zero_gradient_boundary_residual,
            (),
            coefficients["Boundary"],
        ),
    ]
    if split_pde:
        terms.extend(
            (
                f"PDE_{name}",
                problem.number_of_pde_points,
                problem.generate_full_pde_component_grid,
                burgers_component_residual,
                (problem.nu, component),
                coefficients[f"PDE_{name}"],
            )
            for component, name in enumerate(("u", "v"))
        )
    else:
        terms.append(
            (
                "PDE",
                problem.number_of_pde_points,
                problem.generate_full_pde_grid,
                burgers_residual,
                (problem.nu,),
                coefficients["PDE"],
            )
        )
    for name, count, generator, evaluator, args, weight in terms:
        loss = LOSS(device=device, criterium="MSE", name=name, batch_size=count)
        loss.setBatchGenerator(generator)
        loss.setEvalFunction(evaluator, *args)
        trainer.add_loss(loss, weight)
    return model, trainer


def train_with_loss_weight_history(trainer):
    """Run the framework Trainer and record adaptive log variances after updates.

    The optimizer hook only observes the three weights; it does not alter the
    training loop or add loss evaluations. Raw MSE histories remain pre-update.
    """
    snapshots = []
    learning_rates = []
    lr_handle = trainer.optimizer.register_step_pre_hook(
        lambda optimizer, args, kwargs: learning_rates.append(
            [group["lr"] for group in optimizer.param_groups]
        )
    )
    handle = None
    if trainer.adaptive:
        if trainer.adaptive_weights is None or trainer.optimizer is None:
            raise ValueError(
                "Adaptive weights and optimizer must be initialized before tracking"
            )

        def record_weights(optimizer, args, kwargs):
            snapshots.append(trainer.adaptive_weights.log_vars.detach().cpu().clone())

        handle = trainer.optimizer.register_step_post_hook(record_weights)
    try:
        model, history = trainer.train()
    finally:
        trainer.learning_rate_history = learning_rates
        lr_handle.remove()
        if handle is not None:
            handle.remove()
    parameter = next(trainer.model.parameters())
    logs = (
        torch.stack(snapshots)
        if snapshots
        else torch.empty((0, len(trainer.losses)), dtype=parameter.dtype)
    )
    return model, history, logs


def save_loss_weight_artifacts(trainer, log_history, directory):
    """Store framework adaptive state and post-update weights for reproducibility."""
    if not trainer.adaptive:
        return
    directory = Path(directory)
    torch.save(trainer.adaptive_weights.state_dict(), directory / "adaptive_weights.pt")
    np.savez_compressed(
        directory / "adam_loss_weights.npz",
        loss_names=np.asarray([loss.name for loss in trainer.losses]),
        log_vars=log_history.numpy(),
        weights=torch.exp(-log_history).numpy(),
    )


def predict_in_batches(model, coordinates, batch_size=65536):
    """Evaluate physical coordinates without retaining derivative graphs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parameter = next(model.parameters())
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(
                    torch.as_tensor(
                        coordinates[start : start + batch_size],
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                .cpu()
                .numpy()
                for start in range(0, len(coordinates), batch_size)
            ]
        )
