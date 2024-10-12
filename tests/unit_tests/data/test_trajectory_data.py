from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from pedpy.data.trajectory_data import TrajectoryData


def test_changing_trajectory_data_fails():
    trajectory_data = TrajectoryData(
        data=pd.DataFrame.from_dict(
            {
                "id": [0, 1],
                "frame": [0, 0],
                "x": [0, 1],
                "y": [0, 1],
            }
        ),
        frame_rate=25,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        trajectory_data.frame_rate = 10
