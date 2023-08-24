import numpy as np
import pandas as pd
import pytest
import shapely

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementArea, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    SpeedBorderMethod,
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
)
from pedpy.methods.speed_calculator import (
    SpeedError,
    compute_individual_speed,
    compute_mean_speed_per_frame,
    compute_voronoi_speed,
)
from tests.utils.utils import get_trajectory


def test_mean_speed_needs_same_length_speed_and_polygon_data():
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[5, 5],
            number_frames=100,
            start_position=np.array([-1, -1]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )

    walkable_area = WalkableArea(
        [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    )
    measurement_area = MeasurementArea([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])

    speed = compute_individual_speed(
        traj_data=traj_data,
        frame_step=5,
        speed_border_method=SpeedBorderMethod.EXCLUDE,
    )

    assert len(speed.index) != len(traj_data.data.index)
    with pytest.raises(SpeedError, match=".*mean speed.*"):
        compute_mean_speed_per_frame(
            traj_data=traj_data,
            individual_speed=speed,
            measurement_area=measurement_area,
        )


def test_voronoi_speed_needs_same_length_speed_and_polygon_data():
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[5, 5],
            number_frames=100,
            start_position=np.array([-1, -1]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )

    walkable_area = WalkableArea(
        [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    )
    measurement_area = MeasurementArea([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])

    speed = compute_individual_speed(
        traj_data=traj_data,
        frame_step=5,
        speed_border_method=SpeedBorderMethod.EXCLUDE,
    )

    polygons = compute_individual_voronoi_polygons(
        traj_data=traj_data,
        walkable_area=walkable_area,
    )
    intersection = compute_intersecting_polygons(
        individual_voronoi_data=polygons, measurement_area=measurement_area
    )

    assert len(speed.index) != len(intersection.index)
    with pytest.raises(SpeedError, match=".*Voronoi speed.*"):
        compute_voronoi_speed(
            traj_data=traj_data,
            individual_speed=speed,
            individual_voronoi_intersection=intersection,
            measurement_area=measurement_area,
        )
