import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point

from tests.utils.utils import get_trajectory_data


@pytest.mark.parametrize(
    "num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance",
    [
        (10, 100, 20, np.array([0, 1]), np.array([10.0, 5]), 1.0),
        (20, 30, 30, np.array([1, 1]), np.array([0.0, 0]), 2.0),
    ],
)
def test_get_pedestrian_positions_all_included(
    num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance
):
    trajectory_data = get_trajectory_data(
        grid_shape=[num_pedestrian, 1],
        number_frames=frame + 1.5 * window_size,
        movement_direction=movement_direction,
        start_position=start_position,
        ped_distance=ped_distance,
    )

    min_frame = int(frame - window_size / 2)
    max_frame = int(frame + window_size / 2)

    for pedestrian_id in range(num_pedestrian):
        start = start_position + pedestrian_id * np.array([0, 1]) * ped_distance
        expected_positions = [
            Point(start + i * movement_direction) for i in range(min_frame, max_frame + 1)
        ]

        positions = trajectory_data.get_pedestrian_positions(frame, pedestrian_id, window_size)
        assert len(positions) == window_size + 1
        assert MultiPoint(positions).equals(MultiPoint(expected_positions))


@pytest.mark.parametrize(
    "num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance",
    [
        (10, 100, 20, np.array([0, 1]), np.array([10.0, 5]), 1.0),
        (20, 30, 30, np.array([1, 1]), np.array([0.0, 0]), 2.0),
    ],
)
def test_get_pedestrian_positions_window_start_to_frame_included(
    num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance
):
    trajectory_data = get_trajectory_data(
        grid_shape=[num_pedestrian, 1],
        number_frames=frame + 0.5 * window_size,
        movement_direction=movement_direction,
        start_position=start_position,
        ped_distance=ped_distance,
    )

    min_frame = int(frame - window_size / 2)
    max_frame = frame

    for pedestrian_id in range(num_pedestrian):
        start = start_position + pedestrian_id * np.array([0, 1]) * ped_distance
        expected_positions = [
            Point(start + i * movement_direction) for i in range(min_frame, max_frame + 1)
        ]

        positions = trajectory_data.get_pedestrian_positions(frame, pedestrian_id, window_size)
        assert len(positions) == window_size / 2 + 1
        assert MultiPoint(positions).equals(MultiPoint(expected_positions))


@pytest.mark.parametrize(
    "num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance",
    [
        (10, 0, 20, np.array([0, 1]), np.array([10.0, 5]), 1.0),
        (20, 14, 30, np.array([1, 1]), np.array([0.0, 0]), 2.0),
    ],
)
def test_get_pedestrian_positions_frame_to_window_end_included(
    num_pedestrian, frame, window_size, movement_direction, start_position, ped_distance
):
    trajectory_data = get_trajectory_data(
        grid_shape=[num_pedestrian, 1],
        number_frames=frame + window_size,
        movement_direction=movement_direction,
        start_position=start_position,
        ped_distance=ped_distance,
    )

    min_frame = frame
    max_frame = int(frame + window_size / 2)

    for pedestrian_id in range(num_pedestrian):
        start = start_position + pedestrian_id * np.array([0, 1]) * ped_distance
        expected_positions = [
            Point(start + i * movement_direction) for i in range(min_frame, max_frame + 1)
        ]

        positions = trajectory_data.get_pedestrian_positions(frame, pedestrian_id, window_size)
        assert len(positions) == window_size / 2 + 1
        assert MultiPoint(positions).equals(MultiPoint(expected_positions))


@pytest.mark.parametrize(
    "num_pedestrian, window_size, movement_direction, start_position, ped_distance",
    [
        (10, 20, np.array([0, 1]), np.array([10.0, 5]), 1.0),
        (20, 30, np.array([1, 1]), np.array([0.0, 0]), 2.0),
    ],
)
def test_get_pedestrian_position_only_frame_included(
    num_pedestrian, window_size, movement_direction, start_position, ped_distance
):
    num_frames = window_size - 1
    frame = int(num_frames / 2)
    trajectory_data = get_trajectory_data(
        grid_shape=[num_pedestrian, 1],
        number_frames=num_frames,
        movement_direction=movement_direction,
        start_position=start_position,
        ped_distance=ped_distance,
    )

    min_frame = frame
    max_frame = frame

    for pedestrian_id in range(num_pedestrian):
        start = start_position + pedestrian_id * np.array([0, 1]) * ped_distance
        expected_positions = [
            Point(start + i * movement_direction) for i in range(min_frame, max_frame + 1)
        ]

        positions = trajectory_data.get_pedestrian_positions(frame, pedestrian_id, window_size)
        assert len(positions) == 1
        assert MultiPoint(positions).equals(MultiPoint(expected_positions))
