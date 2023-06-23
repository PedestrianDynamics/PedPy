import pandas as pd
import pytest

from pedpy.data.trajectory_data import TrajectoryData


def test_changing_trajectory_data_fails():
    with pytest.raises(
        AttributeError,
        match=fr"Trajectory data can not be changed after construction!",
    ):
        trajectory_data = TrajectoryData(
            data=pd.DataFrame.from_dict(
                {
                    "ID": [0, 1],
                    "frame": [0, 0],
                    "X": [0, 1],
                    "Y": [0, 1],
                    "Z": [0, 1],
                }
            ),
            frame_rate=25,
            file=None,
        )
        trajectory_data.frame_rate = 10
