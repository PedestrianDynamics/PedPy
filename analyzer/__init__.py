from . import _version
from .data.geometry import Geometry
from .data.trajectory_data import TrajectoryData, TrajectoryUnit
from .io.geometry_parser import parse_geometry
from .io.trajectory_parser import load_trajectory
from .methods.density_calculator import (
    compute_classic_density,
    compute_passing_density,
    compute_voronoi_density,
)
from .methods.flow_calculator import compute_flow, compute_n_t
from .methods.method_utils import (
    compute_frame_range_in_area,
    get_peds_in_area,
    get_peds_in_frame_range,
)
from .methods.profile_calculator import compute_profiles
from .methods.velocity_calculator import (
    compute_individual_velocity,
    compute_mean_velocity_per_frame,
    compute_passing_speed,
    compute_voronoi_velocity,
)

__version__ = _version.get_versions()["version"]
