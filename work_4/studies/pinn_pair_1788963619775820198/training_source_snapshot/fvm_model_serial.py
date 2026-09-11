"""Serial finite-volume solver for the two-dimensional viscous Burgers system.

The conservative equations solved are

    u_t + (u**2 / 2)_x + (u * v)_y = nu * Laplacian(u),
    v_t + (u * v)_x + (v**2 / 2)_y = nu * Laplacian(v).

The convective terms use a first-order donor-cell upwind flux and the diffusive
terms use centered face fluxes. Time integration is forward Euler.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import time as py_time

Array = np.ndarray


def _shift(field: Array, offset: int, axis: int, boundary: str) -> Array:
    """Return neighboring cell values for a supported boundary condition."""
    if boundary == "periodic":
        return np.roll(field, shift=offset, axis=axis)

    shifted = np.empty_like(field)
    if axis == 0:
        if offset == -1:
            shifted[:-1, :] = field[1:, :]
            shifted[-1, :] = field[-1, :]
        else:
            shifted[1:, :] = field[:-1, :]
            shifted[0, :] = field[0, :]
    else:
        if offset == -1:
            shifted[:, :-1] = field[:, 1:]
            shifted[:, -1] = field[:, -1]
        else:
            shifted[:, 1:] = field[:, :-1]
            shifted[:, 0] = field[:, 0]
    return shifted


def _physical_flux(u: Array, v: Array, axis: int) -> tuple[Array, Array]:
    """Evaluate the x or y physical flux of the conservative system."""
    if axis == 0:
        return 0.5 * u * u, u * v
    return u * v, 0.5 * v * v


def _upwind_flux(
    u_left: Array,
    v_left: Array,
    u_right: Array,
    v_right: Array,
    axis: int,
) -> tuple[Array, Array]:
    """Evaluate a first-order donor-cell flux at each right/up face."""
    flux_u_left, flux_v_left = _physical_flux(u_left, v_left, axis)
    flux_u_right, flux_v_right = _physical_flux(u_right, v_right, axis)

    normal_velocity_left = u_left if axis == 0 else v_left
    normal_velocity_right = u_right if axis == 0 else v_right
    face_velocity = 0.5 * (normal_velocity_left + normal_velocity_right)
    use_left_state = face_velocity >= 0.0
    flux_u = np.where(use_left_state, flux_u_left, flux_u_right)
    flux_v = np.where(use_left_state, flux_v_left, flux_v_right)
    return flux_u, flux_v


def _incoming_face_flux(
    face_flux: Array, physical_flux: Array, axis: int, boundary: str
) -> Array:
    """Return the left/down face flux associated with every cell."""
    if boundary == "periodic":
        return np.roll(face_flux, shift=1, axis=axis)

    incoming = np.empty_like(face_flux)
    if axis == 0:
        incoming[1:, :] = face_flux[:-1, :]
        incoming[0, :] = physical_flux[0, :]
    else:
        incoming[:, 1:] = face_flux[:, :-1]
        incoming[:, 0] = physical_flux[:, 0]
    return incoming


def solve_pde(
    size_t: int,
    size_x: int,
    size_y: int,
    h: float,
    k: float,
    nu: float,
    u_initial: Array,
    v_initial: Array,
    boundary: str = "periodic",
    verbose: bool = False,
) -> tuple[Array, Array]:
    """Solve the 2D viscous Burgers system on a uniform square-cell mesh.

    ``u_initial`` and ``v_initial`` are optional arrays with shape
    ``(size_x, size_y)``. Both must be supplied together. If omitted, a smooth
    periodic field is generated. The histories have shape
    ``(size_t, size_x, size_y)``.

    ``boundary`` can be ``"periodic"`` or ``"zero_gradient"`` (homogeneous
    Neumann). A ``ValueError`` is raised if the explicit stability estimate is
    violated, rather than returning a partly filled history.
    """
    if size_t < 1 or size_x < 1 or size_y < 1:
        raise ValueError("size_t, size_x, and size_y must be positive integers")
    if h <= 0.0 or k <= 0.0:
        raise ValueError("h and k must be positive")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")
    if boundary not in {"periodic", "zero_gradient"}:
        raise ValueError("boundary must be 'periodic' or 'zero_gradient'")

    # Calculating domain values for de mesh
    x_coordinates = (np.arange(size_x, dtype=float) + 0.5) * h
    y_coordinates = (np.arange(size_y, dtype=float) + 0.5) * h
    x, y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

    # Checking initial condition sizes
    expected_shape = (size_x, size_y)
    if u_initial.shape != expected_shape or v_initial.shape != expected_shape:
        raise ValueError(f"u_initial and v_initial must have shape {expected_shape}")
    if not np.all(np.isfinite(u_initial)) or not np.all(np.isfinite(v_initial)):
        raise ValueError("initial arrays contain NaN or infinite values")

    # Creating results matrices
    u_history = np.empty((size_t, size_x, size_y), dtype=float)
    v_history = np.empty_like(u_history)

    # Initializaing values
    u_history[0] = u_initial
    v_history[0] = v_initial

    # Initializaing time iteration vector for variables
    u = u_initial
    v = v_initial

    for time_index in range(1, size_t):
        # Sufficient explicit estimate for advection plus two-dimensional diffusion.
        max_directional_speed = float(np.max(np.abs(u)) + np.max(np.abs(v)))
        cfl_advection = max_directional_speed * k / h
        cfl_diffusion = 4.0 * nu * k / (h * h)
        stability_number = cfl_advection + cfl_diffusion
        if stability_number > 1.0 + 1.0e-12:
            raise ValueError(
                "explicit stability estimate violated at time index "
                f"{time_index}: advective={cfl_advection:.6g}, "
                f"diffusive={cfl_diffusion:.6g}, "
                f"total={stability_number:.6g} > 1"
            )

        # Creating adjancet matrices
        u_right = _shift(u, -1, axis=0, boundary=boundary)
        v_right = _shift(v, -1, axis=0, boundary=boundary)
        u_left = _shift(u, 1, axis=0, boundary=boundary)
        v_left = _shift(v, 1, axis=0, boundary=boundary)

        u_up = _shift(u, -1, axis=1, boundary=boundary)
        v_up = _shift(v, -1, axis=1, boundary=boundary)
        u_down = _shift(u, 1, axis=1, boundary=boundary)
        v_down = _shift(v, 1, axis=1, boundary=boundary)

        # Computing outcome upwind flux
        flux_x_u, flux_x_v = _upwind_flux(u, v, u_right, v_right, axis=0)
        flux_y_u, flux_y_v = _upwind_flux(u, v, u_up, v_up, axis=1)

        # Computing 'in cell' flux used on homegeneus Newman
        physical_x_u, physical_x_v = _physical_flux(u, v, axis=0)
        physical_y_u, physical_y_v = _physical_flux(u, v, axis=1)

        # Shift to obtain income fluxes
        flux_x_u_left = _incoming_face_flux(
            flux_x_u, physical_x_u, axis=0, boundary=boundary
        )
        flux_x_v_left = _incoming_face_flux(
            flux_x_v, physical_x_v, axis=0, boundary=boundary
        )
        flux_y_u_down = _incoming_face_flux(
            flux_y_u, physical_y_u, axis=1, boundary=boundary
        )
        flux_y_v_down = _incoming_face_flux(
            flux_y_v, physical_y_v, axis=1, boundary=boundary
        )

        # Computing diffusion term
        laplacian_u = (u_right + u_left + u_up + u_down - 4.0 * u) / (h * h)
        laplacian_v = (v_right + v_left + v_up + v_down - 4.0 * v) / (h * h)

        # Computing solution
        u = (
            u
            - (k / h) * (flux_x_u - flux_x_u_left + flux_y_u - flux_y_u_down)
            + k * nu * laplacian_u
        )
        v = (
            v
            - (k / h) * (flux_x_v - flux_x_v_left + flux_y_v - flux_y_v_down)
            + k * nu * laplacian_v
        )

        u_history[time_index] = u
        v_history[time_index] = v

        if verbose and time_index % max(1, (size_t - 1) // 10) == 0:
            print(f"step={time_index}, stability_number={stability_number:.6g}")

    return u_history, v_history


def _number_of_intervals(domain: list[float], spacing: float, name: str) -> int:
    """Return the integer number of uniform intervals in a configured domain."""
    if len(domain) != 2 or domain[1] <= domain[0]:
        raise ValueError(f"{name} must contain two increasing limits")
    intervals = (domain[1] - domain[0]) / spacing
    rounded_intervals = round(intervals)
    if not np.isclose(intervals, rounded_intervals, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} length must be an integer multiple of its spacing")
    return int(rounded_intervals)


def run_from_json(config_directory: Path, output_directory: Path) -> Path:
    """Run the configured simulation and save arrays plus reproducibility data."""

    timestamp = py_time.time()

    # Define mesh and constant config paths
    mesh_path = config_directory / "mesh_properties.json"
    constants_path = config_directory / "constant_properties.json"

    # Open files
    with mesh_path.open(encoding="utf-8") as mesh_file:
        mesh = json.load(mesh_file)
    with constants_path.open(encoding="utf-8") as constants_file:
        constants = json.load(constants_file)

    # Check config files
    required_mesh_keys = {"h", "k", "x_dom", "y_dom", "t_dom"}
    missing_mesh_keys = required_mesh_keys.difference(mesh)
    if missing_mesh_keys:
        raise ValueError(f"missing mesh properties: {sorted(missing_mesh_keys)}")
    if "nu" not in constants:
        raise ValueError("constant_properties.json must define 'nu'")

    # Define mesh parameters
    h = float(mesh["h"])
    k = float(mesh["k"])
    nu = float(constants["nu"])
    mean_velocity_u = float(constants.get("mean_velocity_u", 0.0))
    mean_velocity_v = float(constants.get("mean_velocity_v", 0.0))
    perturbation_amplitude = float(constants.get("perturbation_amplitude", 1.0))
    initial_condition_parameters = np.array(
        [mean_velocity_u, mean_velocity_v, perturbation_amplitude]
    )
    if not np.all(np.isfinite(initial_condition_parameters)):
        raise ValueError("initial-condition parameters must be finite")
    if perturbation_amplitude < 0.0:
        raise ValueError("perturbation_amplitude must be non-negative")

    size_x = _number_of_intervals(mesh["x_dom"], h, "x_dom")
    size_y = _number_of_intervals(mesh["y_dom"], h, "y_dom")
    time_steps = _number_of_intervals(mesh["t_dom"], k, "t_dom")
    if size_x < 2 or size_y < 2:
        raise ValueError("a 2D simulation requires at least two cells per direction")

    # Define mesh
    x = mesh["x_dom"][0] + (np.arange(size_x, dtype=float) + 0.5) * h
    y = mesh["y_dom"][0] + (np.arange(size_y, dtype=float) + 0.5) * h
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    # Compute de initial conditions
    phase_x = (
        2.0
        * np.pi
        * (x_grid - mesh["x_dom"][0])
        / (mesh["x_dom"][1] - mesh["x_dom"][0])
    )
    phase_y = (
        2.0
        * np.pi
        * (y_grid - mesh["y_dom"][0])
        / (mesh["y_dom"][1] - mesh["y_dom"][0])
    )
    u_initial = mean_velocity_u + perturbation_amplitude * np.sin(phase_x) * np.cos(
        phase_y
    )
    v_initial = mean_velocity_v - perturbation_amplitude * np.cos(phase_x) * np.sin(
        phase_y
    )

    # Solve PDE system
    u_history, v_history = solve_pde(
        size_t=time_steps + 1,
        size_x=size_x,
        size_y=size_y,
        h=h,
        k=k,
        nu=nu,
        u_initial=u_initial,
        v_initial=v_initial,
        boundary=constants.get("boundary", "periodic"),
        verbose=True,
    )

    # Defining time mesh
    time = mesh["t_dom"][0] + np.arange(time_steps + 1, dtype=float) * k

    # Saving simulation data
    output_directory.mkdir(parents=True, exist_ok=True)
    solution_path = output_directory / "solution_{}.npz".format(timestamp)
    np.savez_compressed(
        solution_path,
        u=u_history,
        v=v_history,
        x=x,
        y=y,
        t=time,
    )

    # Saving run metadata
    metadata = {
        "equations": [
            "u_t + (u^2/2)_x + (u*v)_y = nu*Laplacian(u)",
            "v_t + (u*v)_x + (v^2/2)_y = nu*Laplacian(v)",
        ],
        "numerical_flux": "first-order donor-cell upwind",
        "time_integrator": "forward Euler",
        "initial_condition": {
            "u": (
                "mean_velocity_u + perturbation_amplitude" "*sin(phase_x)*cos(phase_y)"
            ),
            "v": (
                "mean_velocity_v - perturbation_amplitude" "*cos(phase_x)*sin(phase_y)"
            ),
        },
        "mesh": mesh,
        "constants": constants,
        "array_shape": [time_steps + 1, size_x, size_y],
    }
    with (output_directory / "metadata_{}.json".format(timestamp)).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)
        file.write("\n")
    return solution_path


def main() -> None:
    """Read command-line paths and run the configured Burgers simulation."""
    work_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=work_directory / "control_dicts",
        help="directory containing mesh_properties.json and constant_properties.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=work_directory / "fvm_sim",
        help="directory in which solution.npz and metadata.json are saved",
    )
    arguments = parser.parse_args()
    solution_path = run_from_json(arguments.config_dir, arguments.output_dir)
    print(f"solution saved to {solution_path}")


if __name__ == "__main__":
    main()
