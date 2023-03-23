"""Module containing functions to compute velocities."""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from shapely import Polygon

from pedpy.defintitons import VelocityBorderMethod
from pedpy.methods.method_utils import _compute_individual_movement


def compute_individual_velocity(
    *,
    traj_data: pd.DataFrame,
    frame_rate: float,
    frame_step: int,
    border_method: VelocityBorderMethod = VelocityBorderMethod.SINGLE_SIDED,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    x_y_components: bool = True,
) -> pd.DataFrame:
    """Compute the individual velocity for each pedestrian.

    For computing the velocity at a certain a frame range is used to smooth the
    results:
                  frame
        x--x--x--x--X--x--x-x--x---x
           |________|_______|
        frame_step    frame_step

    When not enough frames on both sides are available, the given
    border_method decides how it is dealt with.

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the
            velocity.
        border_method (VelocityBorderMethod): how to deal with edge cases,
            when not enough frames are on one side
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        x_y_components (bool): compute the x and y components of the speed
    Returns:
        DataFrame containing the columns 'ID', 'frame', and 'speed' in m/s,
        'v_x' and 'v_y' with the speed components in x and y direction if
        x_y_components is True
    """
    df_movement = _compute_individual_movement(
        traj_data=traj_data, frame_step=frame_step, border_method=border_method
    )
    df_speed = _compute_individual_speed(
        movement_data=df_movement,
        frame_rate=frame_rate,
        movement_direction=movement_direction,
        x_y_components=x_y_components,
    )

    return df_speed


def compute_mean_velocity_per_frame(
    *,
    traj_data: pd.DataFrame,
    individual_velocity: pd.DataFrame,
    measurement_area: Polygon,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean velocity per frame.

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (pd.DataFrame): trajectory data
        individual_velocity (pd.DataFrame): individual velocity data
        measurement_area (shapely.Polygon): measurement area for which the
            velocity is computed

    Returns:
        DataFrame containing the columns 'frame' and 'speed'
    """
    combined = traj_data.merge(individual_velocity, on=["ID", "frame"])
    df_mean = (
        combined[shapely.within(combined["points"], measurement_area)]
        .groupby("frame")["speed"]
        .mean()
    )
    df_mean = df_mean.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_mean


def compute_voronoi_velocity(
    *,
    traj_data: pd.DataFrame,
    individual_velocity: pd.DataFrame,
    individual_voronoi_intersection: pd.DataFrame,
    measurement_area: Polygon,
) -> pd.Series:
    """Compute the voronoi velocity per frame.

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (pd.DataFrame): trajectory data
        individual_velocity (pd.DataFrame): individual velocity data
        individual_voronoi_intersection (pd.DataFrame): intersections of the
            individual with the measurement area of each pedestrian
        measurement_area (shapely.Polygon): area in which the voronoi velocity
            should be computed

    Returns:
        DataFrame containing the columns 'frame' and 'voronoi speed'
    """
    df_voronoi = pd.merge(
        individual_voronoi_intersection, individual_velocity, on=["ID", "frame"]
    )
    df_voronoi["voronoi speed"] = (
        shapely.area(df_voronoi["intersection voronoi"])
        * df_voronoi["speed"]
        / measurement_area.area
    )
    df_voronoi_speed = df_voronoi.groupby("frame")["voronoi speed"].sum()
    df_voronoi_speed = df_voronoi_speed.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )
    return pd.Series(df_voronoi_speed)


def compute_passing_speed(
    *, frames_in_area: pd.DataFrame, frame_rate: float, distance: float
) -> pd.DataFrame:
    """Compute the individual speed of the pedestrian who pass the area.

    Args:
        frames_in_area (pd.DataFrame): information for each pedestrian in the
            area, need to contain the following columns: 'ID', 'start', 'end',
            'frame_start', 'frame_end'
        frame_rate (float): frame rate of the trajectory
        distance (float): distance between the two measurement lines
    Returns:
        DataFrame containing the columns: 'ID', 'speed' which is the speed
        in m/s

    """
    speed = pd.DataFrame(frames_in_area["ID"], columns=["ID", "speed"])
    speed["speed"] = (
        frame_rate
        * distance
        / (np.abs(frames_in_area.frame_end - frames_in_area.frame_start))
    )
    return speed


def _compute_individual_speed(
    *,
    movement_data: pd.DataFrame,
    frame_rate: float,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    x_y_components: bool = True,
) -> pd.DataFrame:
    """Compute the instantaneous velocity of each pedestrian.

    Args:
        movement_data (pd.DataFrame): movement data
            (see compute_individual_movement)
        frame_rate (float): frame rate of the trajectory data
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        x_y_components (bool): compute the x and y components of the speed

    Returns:
        DataFrame containing the columns: 'ID', 'frame', 'speed' with the
        speed in m/s, 'v_x' and 'v_y' with the speed components in x and y
        direction if x_y_components is True
    """
    columns = ["ID", "frame", "speed"]
    time_interval = (
        movement_data["end_frame"] - movement_data["start_frame"]
    ) / frame_rate

    # Compute displacements in x and y direction
    movement_data[["d_x", "d_y"]] = shapely.get_coordinates(
        movement_data["end"]
    ) - shapely.get_coordinates(movement_data["start"])

    movement_data["speed"] = (
        np.linalg.norm(movement_data[["d_x", "d_y"]], axis=1) / time_interval
    )

    if movement_direction is not None:
        # Projection of the displacement onto the movement direction
        norm_movement_direction = np.dot(movement_direction, movement_direction)
        movement_data[["d_x", "d_y"]] = (
            np.dot(movement_data[["d_x", "d_y"]].values, movement_direction)[
                :, None
            ]
            * movement_direction
            * norm_movement_direction
        )
        movement_data["speed"] = (
            np.dot(movement_data[["d_x", "d_y"]].values, movement_direction)
            / np.linalg.norm(movement_direction)
            / time_interval
        )

    if x_y_components:
        movement_data["v_x"] = movement_data["d_x"].values / time_interval
        movement_data["v_y"] = movement_data["d_y"].values / time_interval
        columns.append("v_x")
        columns.append("v_y")

    return movement_data[columns]
