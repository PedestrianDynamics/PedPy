# type: ignore
"""Top level imports, for easier usage."""
try:
    from . import _version  # pylint: disable=import-self

    __version__ = _version.__version__
    __commit_hash__ = _version.__commit_hash__

except ImportError:
    __version__ = "unknown"
    __commit_hash__ = "unknown"

from .data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from .data.trajectory_data import TrajectoryData
from .io.trajectory_loader import (
    LoadTrajectoryError,
    TrajectoryUnit,
    load_trajectory,
    load_trajectory_from_jupedsim_sqlite,
    load_trajectory_from_ped_data_archive_hdf5,
    load_trajectory_from_txt,
    load_trajectory_from_viswalk,
    load_walkable_area_from_jupedsim_sqlite,
    load_walkable_area_from_ped_data_archive_hdf5,
)
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
from .methods.profile_calculator import (
    DensityMethod,
    SpeedMethod,
    compute_density_profile,
    compute_grid_cell_polygon_intersection_area,
    compute_profiles,
    compute_speed_profile,
    get_grid_cells,
)
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

__all__ = [
    "MeasurementArea",
    "MeasurementLine",
    "WalkableArea",
    "TrajectoryData",
    "LoadTrajectoryError",
    "TrajectoryUnit",
    "load_trajectory",
    "load_trajectory_from_jupedsim_sqlite",
    "load_trajectory_from_ped_data_archive_hdf5",
    "load_trajectory_from_txt",
    "load_trajectory_from_viswalk",
    "load_walkable_area_from_jupedsim_sqlite",
    "load_walkable_area_from_ped_data_archive_hdf5",
    "compute_classic_density",
    "compute_passing_density",
    "compute_voronoi_density",
    "compute_flow",
    "compute_n_t",
    "Cutoff",
    "compute_frame_range_in_area",
    "compute_individual_voronoi_polygons",
    "compute_intersecting_polygons",
    "compute_neighbors",
    "compute_time_distance_line",
    "get_invalid_trajectory",
    "is_trajectory_valid",
    "DensityMethod",
    "SpeedMethod",
    "compute_density_profile",
    "compute_grid_cell_polygon_intersection_area",
    "compute_profiles",
    "compute_speed_profile",
    "get_grid_cells",
    "SpeedCalculation",
    "compute_individual_speed",
    "compute_mean_speed_per_frame",
    "compute_passing_speed",
    "compute_voronoi_speed",
    "PEDPY_BLUE",
    "PEDPY_GREEN",
    "PEDPY_GREY",
    "PEDPY_ORANGE",
    "PEDPY_PETROL",
    "PEDPY_RED",
    "plot_density",
    "plot_density_distribution",
    "plot_flow",
    "plot_measurement_setup",
    "plot_neighborhood",
    "plot_nt",
    "plot_profiles",
    "plot_speed",
    "plot_speed_distribution",
    "plot_time_distance",
    "plot_trajectories",
    "plot_voronoi_cells",
    "plot_walkable_area",
    "__version__",
]
