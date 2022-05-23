import numpy as np
import pandas as pd
import pygeos
import pytest
from shapely.geometry import Point

from report.methods.density_calculator import compute_classic_density
from tests.utils.utils import get_trajectory_data


@pytest.mark.parametrize(
    "measurement_area, ped_distance, num_ped_col, num_ped_row",
    [
        (pygeos.polygons([(-5, -5), (-5, 5), (5, 5), (5, -5)]), 1.0, 5, 5),
        (
            pygeos.polygons([(0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)]),
            0.5,
            5,
            5,
        ),
    ],
)
def test_compute_classic_density(measurement_area, ped_distance, num_ped_col, num_ped_row):
    velocity = 1
    movement_direction = np.array([velocity, 0])
    num_frames = 50

    trajectory_data = get_trajectory_data(
        grid_shape=[num_ped_col, num_ped_row],
        number_frames=num_frames,
        movement_direction=movement_direction,
        start_position=[-10, 2 * ped_distance],
        ped_distance=ped_distance,
        fps=25,
    )

    computed_density = compute_classic_density(trajectory_data, measurement_area)

    num_peds_in_area_per_frame = {frame: 0 for frame in range(0, num_frames)}

    for index, row in trajectory_data.data.iterrows():
        point = Point([row.X, row.Y])
        if point.within(pygeos.to_shapely(measurement_area)):
            num_peds_in_area_per_frame[row.frame] += 1

    expected_density = pd.DataFrame.from_dict(
        {
            frame: [num_peds / pygeos.area(measurement_area)]
            for frame, num_peds in num_peds_in_area_per_frame.items()
        },
        orient="index",
        columns=["classic density"],
    )
    expected_density.index.name = "frame"

    assert computed_density.index.min() == 0
    assert computed_density.index.max() == num_frames - 1
    assert expected_density.equals(computed_density)
