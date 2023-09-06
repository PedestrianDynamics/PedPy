# type: ignore
"""Top level imports, for easier usage."""
from . import _version
from .data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from .data.trajectory_data import TrajectoryData
from .io.trajectory_loader import TrajectoryUnit, load_trajectory
from .methods.density_calculator import (
    compute_classic_density,
    compute_passing_density,
    compute_voronoi_density,
)
from .methods.flow_calculator import compute_flow, compute_n_t
from .methods.method_utils import (
    Cutoff,
    compute_frame_range_in_area,
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
    compute_neighbors,
    compute_time_distance_line,
    get_invalid_trajectory,
    is_trajectory_valid,
)
from .methods.profile_calculator import SpeedMethod, compute_profiles
from .methods.speed_calculator import (
    SpeedCalculation,
    compute_individual_speed,
    compute_mean_speed_per_frame,
    compute_passing_speed,
    compute_voronoi_speed,
)
from .plotting.plotting import (
    PEDPY_BLUE,
    PEDPY_GREEN,
    PEDPY_GREY,
    PEDPY_ORANGE,
    PEDPY_PETROL,
    PEDPY_RED,
    plot_density,
    plot_density_distribution,
    plot_flow,
    plot_measurement_setup,
    plot_neighborhood,
    plot_nt,
    plot_profiles,
    plot_speed,
    plot_speed_distribution,
    plot_time_distance,
    plot_trajectories,
    plot_voronoi_cells,
    plot_walkable_area,
)

__version__ = _version.get_versions()["version"]
