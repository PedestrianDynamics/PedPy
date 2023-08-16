import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.density_calculator import (
    _get_num_peds_per_frame,
    compute_classic_density,
)
from tests.utils.utils import get_trajectory, get_trajectory_data


@pytest.mark.parametrize(
    "measurement_area, ped_distance, num_ped_col, num_ped_row",
    [
        (MeasurementArea([(-5, -5), (-5, 5), (5, 5), (5, -5)]), 1.0, 5, 5),
        (
            MeasurementArea(
                [(0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)]
            ),
            0.5,
            5,
            5,
        ),
    ],
)
def test_compute_classic_density(
    measurement_area, ped_distance, num_ped_col, num_ped_row
):
    velocity = 1
    movement_direction = np.array([velocity, 0])
    num_frames = 50

    trajectory_data = get_trajectory_data(
        grid_shape=[num_ped_col, num_ped_row],
        number_frames=num_frames,
        movement_direction=movement_direction,
        start_position=np.array([-10, 2 * ped_distance]),
        ped_distance=ped_distance,
        fps=25,
    )

    computed_density = compute_classic_density(
        traj_data=trajectory_data, measurement_area=measurement_area
    )

    num_peds_in_area_per_frame = {frame: 0 for frame in range(0, num_frames)}

    for index, row in trajectory_data.data.iterrows():
        point = Point([row.x, row.y])
        if point.within(measurement_area.polygon):
            num_peds_in_area_per_frame[row.frame] += 1

    expected_density = pd.DataFrame.from_dict(
        {
            frame: [num_peds / measurement_area.area]
            for frame, num_peds in num_peds_in_area_per_frame.items()
        },
        orient="index",
        columns=[DENSITY_COL],
    )
    expected_density.index.name = FRAME_COL

    assert computed_density.index.min() == 0
    assert computed_density.index.max() == num_frames - 1
    assert expected_density.equals(computed_density)


@pytest.mark.parametrize(
    "num_peds_row, num_peds_col, num_frames",
    (
        [
            (4, 5, 100),
            (1, 1, 200),
        ]
    ),
)
def test_get_num_peds_per_frame(num_peds_row, num_peds_col, num_frames):
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[num_peds_col, num_peds_row],
            number_frames=num_frames,
            start_position=np.array([0, 0]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )
    num_peds = num_peds_col * num_peds_row
    num_peds_per_frame = _get_num_peds_per_frame(traj_data)

    assert (num_peds_per_frame[COUNT_COL] == num_peds).all()
