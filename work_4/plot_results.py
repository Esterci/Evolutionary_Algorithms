"""Plot the stored two-dimensional viscous Burgers solution.

The script creates two complementary visualizations:

1. a static 3-by-2 panel with the ``u`` and ``v`` fields at three times;
2. a lightweight GIF with ``u`` and ``v`` side by side.

The animation samples a limited number of uniformly spaced time indices instead
of rendering every time step of the numerical simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import numpy as np


BASE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SOLUTIONS_DIRECTORY = BASE_DIRECTORY / "fvm_sim"
DEFAULT_PLOTS_DIRECTORY = BASE_DIRECTORY / "plots"
DEFAULT_GIF_FRAMES = 60
DEFAULT_GIF_FPS = 12
TIMESTAMPED_SOLUTION_PATTERN = re.compile(
    r"^solution_(?P<timestamp>[0-9]+(?:\.[0-9]+)?)\.npz$"
)


def solution_timestamp(solution_path: Path) -> str | None:
    """Extract the Unix timestamp from a timestamped solution filename."""
    match = TIMESTAMPED_SOLUTION_PATTERN.fullmatch(Path(solution_path).name)
    return match.group("timestamp") if match is not None else None


def resolve_solution_path(
    solutions_directory: Path = DEFAULT_SOLUTIONS_DIRECTORY,
    timestamp: str | None = None,
    explicit_solution: Path | None = None,
) -> Path:
    """Select an explicit, timestamped, or latest timestamped solution file."""
    if explicit_solution is not None:
        solution_path = Path(explicit_solution)
        if not solution_path.is_file():
            raise FileNotFoundError(f"solution file not found: {solution_path}")
        return solution_path

    solutions_directory = Path(solutions_directory)
    if timestamp is not None:
        timestamp_match = re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", timestamp)
        if timestamp_match is None:
            raise ValueError(
                "timestamp must contain only a Unix timestamp, for example "
                "1756223456.123"
            )
        solution_path = solutions_directory / f"solution_{timestamp}.npz"
        if not solution_path.is_file():
            raise FileNotFoundError(
                f"no solution found for Unix timestamp {timestamp}: {solution_path}"
            )
        return solution_path

    if not solutions_directory.is_dir():
        raise FileNotFoundError(f"solutions directory not found: {solutions_directory}")

    timestamped_solutions = []
    for solution_path in solutions_directory.glob("solution_*.npz"):
        match = TIMESTAMPED_SOLUTION_PATTERN.fullmatch(solution_path.name)
        if match is not None:
            timestamped_solutions.append(
                (float(match.group("timestamp")), solution_path)
            )

    if not timestamped_solutions:
        raise FileNotFoundError(
            "no timestamped solution files matching "
            f"'solution_<unix_timestamp>.npz' were found in {solutions_directory}"
        )

    return max(timestamped_solutions, key=lambda item: item[0])[1]


def load_solution(solution_path: Path) -> tuple[np.ndarray, ...]:
    """Load and validate the coordinates and velocity histories."""
    solution_path = Path(solution_path)
    with np.load(solution_path) as solution:
        required_arrays = {"u", "v", "x", "y", "t"}
        missing_arrays = required_arrays.difference(solution.files)
        if missing_arrays:
            raise ValueError(
                f"{solution_path} is missing arrays: {sorted(missing_arrays)}"
            )

        u_history = np.asarray(solution["u"])
        v_history = np.asarray(solution["v"])
        x_coordinates = np.asarray(solution["x"])
        y_coordinates = np.asarray(solution["y"])
        time = np.asarray(solution["t"])

    if u_history.ndim != 3 or v_history.ndim != 3:
        raise ValueError("u and v histories must have shape [time, x, y]")
    if u_history.shape != v_history.shape:
        raise ValueError("u and v histories must have the same shape")

    expected_shape = (time.size, x_coordinates.size, y_coordinates.size)
    if u_history.shape != expected_shape:
        raise ValueError(
            f"history shape {u_history.shape} does not match coordinates "
            f"{expected_shape}"
        )
    if time.size < 3:
        raise ValueError("at least three time steps are required for the static plot")
    if x_coordinates.size < 2 or y_coordinates.size < 2:
        raise ValueError("at least two coordinates are required in each direction")

    arrays = (u_history, v_history, x_coordinates, y_coordinates, time)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("the stored solution contains NaN or infinite values")
    if np.any(np.diff(x_coordinates) <= 0.0):
        raise ValueError("x coordinates must be strictly increasing")
    if np.any(np.diff(y_coordinates) <= 0.0):
        raise ValueError("y coordinates must be strictly increasing")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time coordinates must be strictly increasing")

    return arrays


def coordinate_edges(coordinates: np.ndarray) -> np.ndarray:
    """Compute cell edges from a one-dimensional array of cell centers."""
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    return edges


def shared_color_normalization(
    u_history: np.ndarray, v_history: np.ndarray
) -> Normalize:
    """Return one symmetric color scale for both velocity components."""
    maximum_magnitude = max(
        float(np.max(np.abs(u_history))),
        float(np.max(np.abs(v_history))),
    )
    if maximum_magnitude == 0.0:
        maximum_magnitude = 1.0
    return Normalize(vmin=-maximum_magnitude, vmax=maximum_magnitude)


def select_static_indices(
    time: np.ndarray, requested_times: list[float] | None = None
) -> np.ndarray:
    """Select three distinct time indices for the static visualization."""
    if requested_times is None:
        return np.linspace(0, time.size - 1, 3, dtype=int)

    requested_times_array = np.asarray(requested_times, dtype=float)
    if not np.all(np.isfinite(requested_times_array)):
        raise ValueError("requested static times must be finite")
    if np.any(requested_times_array < time[0]) or np.any(
        requested_times_array > time[-1]
    ):
        raise ValueError(f"static times must be within [{time[0]:.6g}, {time[-1]:.6g}]")

    indices = np.array(
        [
            int(np.argmin(np.abs(time - requested_time)))
            for requested_time in requested_times
        ]
    )
    if np.unique(indices).size != 3:
        raise ValueError("static times must map to three distinct time steps")
    return indices


def create_static_plot(
    u_history: np.ndarray,
    v_history: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: np.ndarray,
    output_path: Path,
    requested_times: list[float] | None = None,
) -> Path:
    """Create six heatmaps for both fields at three selected times."""
    time_indices = select_static_indices(time, requested_times)
    x_edges = coordinate_edges(x_coordinates)
    y_edges = coordinate_edges(y_coordinates)
    color_normalization = shared_color_normalization(u_history, v_history)

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(11, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fields = (("u", u_history), ("v", v_history))
    heatmaps = []

    for row, time_index in enumerate(time_indices):
        for column, (field_name, field_history) in enumerate(fields):
            axis = axes[row, column]
            heatmap = axis.pcolormesh(
                x_edges,
                y_edges,
                field_history[time_index].T,
                cmap="coolwarm",
                norm=color_normalization,
                shading="flat",
                rasterized=True,
            )
            heatmaps.append(heatmap)
            axis.set_title(rf"${field_name}(x,y,t)$ at $t={time[time_index]:.4f}$")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_aspect("equal", adjustable="box")

    colorbar = figure.colorbar(
        heatmaps[0],
        ax=axes,
        orientation="vertical",
        shrink=0.82,
        pad=0.03,
    )
    colorbar.set_label("Velocity component")
    figure.suptitle("Two-dimensional viscous Burgers solution", fontsize=16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def select_animation_indices(
    number_of_time_steps: int, maximum_frames: int
) -> np.ndarray:
    """Select at most ``maximum_frames`` uniformly spaced time indices."""
    if maximum_frames < 2:
        raise ValueError("maximum_frames must be at least 2")
    number_of_frames = min(number_of_time_steps, maximum_frames)
    return np.unique(
        np.linspace(0, number_of_time_steps - 1, number_of_frames, dtype=int)
    )


def create_animation(
    u_history: np.ndarray,
    v_history: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    time: np.ndarray,
    output_path: Path,
    maximum_frames: int = DEFAULT_GIF_FRAMES,
    frames_per_second: int = DEFAULT_GIF_FPS,
) -> tuple[Path, int]:
    """Create a sampled side-by-side GIF of the two velocity components."""
    if frames_per_second < 1:
        raise ValueError("frames_per_second must be positive")

    frame_indices = select_animation_indices(time.size, maximum_frames)
    x_edges = coordinate_edges(x_coordinates)
    y_edges = coordinate_edges(y_coordinates)
    color_normalization = shared_color_normalization(u_history, v_history)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fields = (("u", u_history), ("v", v_history))
    heatmaps = []

    initial_time_index = int(frame_indices[0])
    for axis, (field_name, field_history) in zip(axes, fields):
        heatmap = axis.pcolormesh(
            x_edges,
            y_edges,
            field_history[initial_time_index].T,
            cmap="coolwarm",
            norm=color_normalization,
            shading="flat",
        )
        heatmaps.append(heatmap)
        axis.set_title(rf"${field_name}(x,y,t)$")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="box")

    colorbar = figure.colorbar(
        heatmaps[0], ax=axes, orientation="vertical", shrink=0.84, pad=0.03
    )
    colorbar.set_label("Velocity component")
    time_title = figure.suptitle(rf"$t={time[initial_time_index]:.4f}$", fontsize=15)

    def update(frame_index: int):
        """Update both heatmaps for one sampled time index."""
        time_index = int(frame_indices[frame_index])
        heatmaps[0].set_array(u_history[time_index].T.ravel())
        heatmaps[1].set_array(v_history[time_index].T.ravel())
        time_title.set_text(rf"$t={time[time_index]:.4f}$")
        return (*heatmaps, time_title)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices.size,
        interval=1000 / frames_per_second,
        # A full redraw keeps the figure-level time label in every GIF frame.
        blit=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=frames_per_second), dpi=100)
    plt.close(figure)
    return output_path, int(frame_indices.size)


def create_plots(
    solution_path: Path,
    plots_directory: Path = DEFAULT_PLOTS_DIRECTORY,
    static_times: list[float] | None = None,
    gif_frames: int = DEFAULT_GIF_FRAMES,
    gif_fps: int = DEFAULT_GIF_FPS,
) -> tuple[Path, Path]:
    """Load one solution and create the static figure and sampled GIF."""
    solution_path = Path(solution_path)
    solution = load_solution(solution_path)
    plots_directory = Path(plots_directory)
    timestamp = solution_timestamp(solution_path)
    output_suffix = f"_{timestamp}" if timestamp is not None else ""

    print(f"Solution loaded from: {solution_path}")

    static_path = create_static_plot(
        *solution,
        output_path=plots_directory / f"velocity_heatmaps{output_suffix}.png",
        requested_times=static_times,
    )
    animation_path, rendered_frames = create_animation(
        *solution,
        output_path=plots_directory / f"velocity_evolution{output_suffix}.gif",
        maximum_frames=gif_frames,
        frames_per_second=gif_fps,
    )

    print(f"Static plot saved to: {static_path}")
    print(
        f"Animation saved to: {animation_path} "
        f"({rendered_frames} of {solution[-1].size} time steps)"
    )
    return static_path, animation_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for both visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    solution_selection = parser.add_mutually_exclusive_group()
    solution_selection.add_argument(
        "--solution",
        type=Path,
        help="explicit path to a solution file",
    )
    solution_selection.add_argument(
        "--timestamp",
        type=str,
        metavar="UNIX_TIMESTAMP",
        help=(
            "Unix timestamp in a solution_<timestamp>.npz filename "
            "(default: select the latest timestamp)"
        ),
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=DEFAULT_SOLUTIONS_DIRECTORY,
        help=(
            "directory containing timestamped solutions "
            f"(default: {DEFAULT_SOLUTIONS_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIRECTORY,
        help="directory in which the PNG and GIF are saved",
    )
    parser.add_argument(
        "--static-times",
        type=float,
        nargs=3,
        metavar=("TIME_1", "TIME_2", "TIME_3"),
        help="three physical times for the static plot (default: start, middle, end)",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=DEFAULT_GIF_FRAMES,
        help=(
            "maximum number of uniformly sampled GIF frames "
            f"(default: {DEFAULT_GIF_FRAMES})"
        ),
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help=f"GIF playback speed in frames per second (default: {DEFAULT_GIF_FPS})",
    )
    return parser.parse_args()


def main() -> None:
    """Create visualizations using command-line paths and sampling options."""
    arguments = parse_arguments()
    solution_path = resolve_solution_path(
        solutions_directory=arguments.solutions_dir,
        timestamp=arguments.timestamp,
        explicit_solution=arguments.solution,
    )
    create_plots(
        solution_path=solution_path,
        plots_directory=arguments.output_dir,
        static_times=arguments.static_times,
        gif_frames=arguments.gif_frames,
        gif_fps=arguments.gif_fps,
    )


if __name__ == "__main__":
    main()
