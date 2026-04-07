"""Tests that public API functions reject non-TrajectoryData inputs."""

import pandas as pd
import pytest

from pedpy.errors import PedPyTypeError
from pedpy.methods.acceleration_calculator import (
    compute_individual_acceleration,
    compute_mean_acceleration_per_frame,
    compute_voronoi_acceleration,
)
from pedpy.methods.density_calculator import compute_classic_density
from pedpy.methods.flow_calculator import compute_n_t
from pedpy.methods.method_utils import (
    compute_crossing_frames,
    compute_frame_range_in_area,
    compute_individual_voronoi_polygons,
    compute_neighbor_distance,
    compute_time_distance_line,
    get_invalid_trajectory,
    is_trajectory_valid,
)
from pedpy.methods.spatial_analysis import compute_pair_distribution_function
from pedpy.methods.speed_calculator import (
    compute_individual_speed,
    compute_mean_speed_per_frame,
    compute_species,
    compute_voronoi_speed,
)

_BAD = pd.DataFrame({"x": [1], "y": [2]})
_DF = pd.DataFrame()


@pytest.mark.parametrize(
    "func,kwargs",
    [
        pytest.param(compute_individual_speed, dict(traj_data=_BAD, frame_step=5), id="compute_individual_speed"),
        pytest.param(compute_mean_speed_per_frame, dict(traj_data=_BAD, individual_speed=_DF, measurement_area=None), id="compute_mean_speed_per_frame"),
        pytest.param(compute_voronoi_speed, dict(traj_data=_BAD, individual_speed=_DF, individual_voronoi_intersection=_DF, measurement_area=None), id="compute_voronoi_speed"),
        pytest.param(compute_species, dict(trajectory_data=_BAD, individual_voronoi_polygons=_DF, measurement_line=None, frame_step=5), id="compute_species"),
        pytest.param(compute_individual_acceleration, dict(traj_data=_BAD, frame_step=5), id="compute_individual_acceleration"),
        pytest.param(compute_mean_acceleration_per_frame, dict(traj_data=_BAD, individual_acceleration=_DF, measurement_area=None), id="compute_mean_acceleration_per_frame"),
        pytest.param(compute_voronoi_acceleration, dict(traj_data=_BAD, individual_acceleration=_DF, individual_voronoi_intersection=_DF, measurement_area=None), id="compute_voronoi_acceleration"),
        pytest.param(compute_classic_density, dict(traj_data=_BAD, measurement_area=None), id="compute_classic_density"),
        pytest.param(compute_n_t, dict(traj_data=_BAD, measurement_line=None), id="compute_n_t"),
        pytest.param(is_trajectory_valid, dict(traj_data=_BAD, walkable_area=None), id="is_trajectory_valid"),
        pytest.param(get_invalid_trajectory, dict(traj_data=_BAD, walkable_area=None), id="get_invalid_trajectory"),
        pytest.param(compute_frame_range_in_area, dict(traj_data=_BAD, measurement_line=None, width=1.0), id="compute_frame_range_in_area"),
        pytest.param(compute_neighbor_distance, dict(traj_data=_BAD, neighborhood=_DF), id="compute_neighbor_distance"),
        pytest.param(compute_time_distance_line, dict(traj_data=_BAD, measurement_line=None), id="compute_time_distance_line"),
        pytest.param(compute_individual_voronoi_polygons, dict(traj_data=_BAD, walkable_area=None), id="compute_individual_voronoi_polygons"),
        pytest.param(compute_crossing_frames, dict(traj_data=_BAD, measurement_line=None), id="compute_crossing_frames"),
        pytest.param(compute_pair_distribution_function, dict(traj_data=_BAD, radius_bin_size=0.1), id="compute_pair_distribution_function"),
    ],
)
def test_raises_type_error_for_non_trajectory_data(func, kwargs):
    with pytest.raises(PedPyTypeError, match="Expected .* to be a TrajectoryData"):
        func(**kwargs)
