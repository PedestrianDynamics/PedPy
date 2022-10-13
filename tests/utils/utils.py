import pathlib

import shapely

from helper.create_trajectories import get_grid_trajectory
from pedpy.data.trajectory_data import TrajectoryData


def get_trajectory(
    *, shape, number_frames, start_position, movement_direction, ped_distance
):
    grid = get_grid_trajectory(
        shape=shape,
        start_position=start_position,
        movement_direction=movement_direction,
        ped_distance=ped_distance,
        random_ids=False,
        number_frames=number_frames,
    )
    grid = grid.rename(columns={"FR": "frame"})
    grid["points"] = shapely.points(grid["X"], grid["Y"])
    return grid


def get_trajectory_data(
    *,
    grid_shape,
    number_frames,
    start_position,
    movement_direction,
    ped_distance,
    fps=10,
):
    grid = get_grid_trajectory(
        shape=grid_shape,
        start_position=start_position,
        movement_direction=movement_direction,
        ped_distance=ped_distance,
        random_ids=False,
        number_frames=number_frames,
    )
    grid = grid.rename(columns={"FR": "frame"})
    return TrajectoryData(grid, fps, pathlib.Path("not_relevant"))
