from dataclasses import dataclass

import pytest
from shapely.geometry import Point

from report.methods.velocity_calculator import VelocityCalculator


@dataclass
class TrajectoryDataMock:
    frame_rate: float

    def get_pedestrian_positions(self):
        pass


@pytest.mark.parametrize(
    "agent_positions, fps, expected_speed",
    [
        ([Point(0, 0), Point(0, 0)], 1, 0.0),
        ([Point(0, 0), Point(0, 0)], 10, 0.0),
        ([Point(0, 0), Point(0, 1)], 1, 1.0),
        ([Point(0, 0), Point(1, 1)], 1, 2 ** 0.5),
        ([Point(-10, 10), Point(10, -10)], 10, 800 ** 0.5 * 10),
        ([Point(10, -10), Point(-10, 10)], 10, 800 ** 0.5 * 10),
    ],
)
def test_compute_instantaneous_velocity_no_movement_direction_not_ignore_backwards(
    mocker, agent_positions, fps, expected_speed
):
    traj = TrajectoryDataMock(fps)

    vc = VelocityCalculator(10, "will_be_ignored", False)
    mocker.patch.object(traj, "get_pedestrian_positions", return_value=agent_positions)
    velocity = vc.compute_instantaneous_velocity(traj, 0, 1)
    assert velocity == expected_speed
