import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point

from pedpy.methods.method_utils import _compute_individual_movement
from tests.utils.utils import get_trajectory, get_trajectory_data

@pytest.mark.parametrize(
    "num_peds_row, num_peds_col, num_frames",
    (
        [
            (4, 5, 100),
            (2, 1, 100),
        ]
    ),
)
def test_indidividual_movment_only_contains_data_from_ped(num_peds_row, num_peds_col, num_frames):
    traj_data = get_trajectory(
        shape=[num_peds_col, num_peds_row],
        number_frames=num_frames,
        start_position=np.array([0, 0]),
        movement_direction=np.array([0, 0.1]),
        ped_distance=1.0,
    )

    movement = _compute_individual_movement(traj_data=traj_data, frame_step=5)

    for ped_id, ped_data in movement.groupby('ID'):
        ped_traj = traj_data.groupby("ID").get_group(ped_id)
        movement_line = shapely.LineString(ped_traj.points.values)

        assert shapely.dwithin(ped_data.start, movement_line, 1e-10).all()
        assert shapely.dwithin(ped_data.end, movement_line, 1e-10).all()


