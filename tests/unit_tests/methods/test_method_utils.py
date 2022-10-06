import numpy as np
import pygeos
import pytest

from analyzer.methods.method_utils import (
    get_peds_in_area,
    get_peds_in_frame_range,
)
from tests.utils.utils import get_trajectory


@pytest.mark.parametrize(
    "area",
    [
        pygeos.polygons([[10, 10], [10, 10], [10, -10], [10, -10]]),
        pygeos.polygons([[0, 0], [2.5, 2.5], [5, 0]]),
    ],
)
def test_get_peds_in_area(area):
    traj_data = get_trajectory(
        shape=[5, 5],
        number_frames=400,
        start_position=np.array([2.5, 0]),
        movement_direction=np.array([0.1, 0.1]),
        ped_distance=1.0,
    )
    filtered_data = get_peds_in_area(traj_data, area)
    assert pygeos.contains(area, filtered_data["points"]).all()


@pytest.mark.parametrize(
    "min_frame, max_frame",
    [(50, 200), (200, 200), (None, 40), (40, None), (None, None)],
)
def test_peds_in_frame_range(min_frame, max_frame):
    traj_data = get_trajectory(
        shape=[5, 5],
        number_frames=400,
        start_position=np.array([0, 0]),
        movement_direction=np.array([0, 0.1]),
        ped_distance=1.0,
    )

    filtered_data = get_peds_in_frame_range(traj_data, min_frame, max_frame)

    if min_frame is not None:
        assert filtered_data["frame"].min() == min_frame

    if max_frame is not None:
        assert filtered_data["frame"].max() == max_frame

    if min_frame is None and max_frame is None:
        assert filtered_data.equals(traj_data)
