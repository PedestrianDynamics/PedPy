import pathlib
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from pedpy.data.trajectory_data import TrajectoryData


def test_changing_trajectory_data_fails():
    with pytest.raises(
        FrozenInstanceError,
    ):
        trajectory_data = TrajectoryData(
            data=pd.DataFrame.from_dict(
                {
                    "ID": [0, 1],
                    "frame": [0, 0],
                    "X": [0, 1],
                    "Y": [0, 1],
                }
            ),
            frame_rate=25,
        )
        trajectory_data.frame_rate = 10
