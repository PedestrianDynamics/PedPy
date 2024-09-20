from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from helper.create_trajectories import get_grid_trajectory

from pedpy.column_identifier import X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData


def get_trajectory(
    *,
    shape: List[int],
    number_frames: int,
    start_position: npt.NDArray[np.float64],
    movement_direction: npt.NDArray[np.float64],
    ped_distance: float,
) -> pd.DataFrame:
    grid = get_grid_trajectory(
        shape=shape,
        start_position=start_position,
        movement_direction=movement_direction,
        ped_distance=ped_distance,
        random_ids=False,
        number_frames=number_frames,
    )
    grid = grid.rename(columns={"FR": "frame"})
    grid["point"] = shapely.points(grid[X_COL], grid[Y_COL])
    return grid


def get_trajectory_data(
    *,
    grid_shape: List[int],
    number_frames: int,
    start_position: npt.NDArray[np.float64],
    movement_direction: npt.NDArray[np.float64],
    ped_distance: float,
    fps: int = 10,
) -> TrajectoryData:
    grid = get_grid_trajectory(
        shape=grid_shape,
        start_position=start_position,
        movement_direction=movement_direction,
        ped_distance=ped_distance,
        random_ids=False,
        number_frames=number_frames,
    )
    grid = grid.rename(columns={"FR": "frame"})
    return TrajectoryData(data=grid, frame_rate=fps)
