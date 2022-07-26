"""Module containing functions to compute velocities"""

from typing import Tuple

import numpy as np
import pandas as pd
import pygeos

from analyzer.methods.method_utils import _compute_individual_movement


def compute_individual_velocity(
    traj_data: pd.DataFrame,
    frame_rate: float,
    frame_step: int,
    movement_direction: np.ndarray = None,
) -> pd.DataFrame:
    """Compute the individual velocity for each pedestrian

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the velocity
        movement_direction (np.ndarray): main movement direction on which the actual movement is
            projected (default: None, when the un-projected movement should be used)
    Returns:
        DataFrame containing the columns 'ID', 'frame', and 'speed'
    """
    df_movement = _compute_individual_movement(traj_data, frame_step)
    df_speed = _compute_individual_speed(df_movement, frame_rate, movement_direction)

    return df_speed


def compute_mean_velocity_per_frame(
    traj_data: pd.DataFrame,
    measurement_area: pygeos.Geometry,
    frame_rate: float,
    frame_step: int,
    movement_direction: np.ndarray = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean velocity per frame

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_area (pygeos.Geometry): measurement area for which the velocity is computed
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the velocity
        movement_direction (np.ndarray): main movement direction on which the actual movement is
            projected (default: None, when the un-projected movement should be used)

    Returns:
        DataFrame containing the columns 'frame' and 'speed'
        DataFrame containing the columns 'ID', 'frame', and 'speed'
    """
    df_speed = compute_individual_velocity(traj_data, frame_rate, frame_step, movement_direction)
    combined = traj_data.merge(df_speed, on=["ID", "frame"])
    df_mean = (
        combined[pygeos.within(combined["points"], measurement_area)]
        .groupby("frame")["speed"]
        .mean()
    )
    df_mean = df_mean.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_mean, df_speed


def compute_voronoi_velocity(
    traj_data: pd.DataFrame,
    individual_voronoi_intersection: pd.DataFrame,
    frame_rate: float,
    frame_step: int,
    measurement_area: pygeos.Geometry,
    movement_direction: np.ndarray = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the voronoi velocity per frame

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_voronoi_intersection (pd.DataFrame): intersections of the individual with the
            measurement area of each pedestrian
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the velocity
        measurement_area (pygeos.Geometry): area in which the voronoi velocity should be computed
        movement_direction (np.ndarray): main movement direction on which the actual movement is
            projected (default: None, when the un-projected movement should be used)

    Returns:
        DataFrame containing the columns 'frame' and 'voronoi speed'
        DataFrame containing the columns 'ID', 'frame', and 'speed'
    """
    df_speed = compute_individual_velocity(traj_data, frame_rate, frame_step, movement_direction)
    df_voronoi = pd.merge(individual_voronoi_intersection, df_speed, on=["ID", "frame"])
    df_voronoi["voronoi speed"] = (
        pygeos.area(df_voronoi["intersection voronoi"])
        * df_voronoi["speed"]
        / pygeos.area(measurement_area)
    )
    df_voronoi_speed = df_voronoi.groupby("frame")["voronoi speed"].sum()
    df_voronoi_speed = df_voronoi_speed.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_voronoi_speed, df_speed


def _compute_individual_speed(
    movement_data: pd.DataFrame, frame_rate: float, movement_direction=None
) -> pd.DataFrame:
    """Compute the instantaneous velocity of each pedestrian.

    Args:
        movement_data (pd.DataFrame): movement data (see compute_individual_movement)
        frame_rate (float): frame rate of the trajectory data
        movement_direction (np.ndarray): main movement direction on which the actual movement is
            projected (default: None, when the un-projected movement should be used)

    Returns:
        DataFrame containing the columns: 'ID', 'frame', and 'speed' with the speed in m/s
    """
    if movement_direction is None:
        movement_data["distance"] = pygeos.distance(movement_data["start"], movement_data["end"])
    else:
        # get the length of the projection of the movement in the frame range onto the
        # movement_direction
        movement_data["distance"] = (
            np.dot(
                pygeos.get_coordinates(movement_data["end"])
                - pygeos.get_coordinates(movement_data["start"]),
                movement_direction,
            )
            / np.linalg.norm(movement_direction)
        )

    movement_data["speed"] = movement_data["distance"] / (
        (movement_data["end_frame"] - movement_data["start_frame"]) / frame_rate
    )

    return movement_data


def compute_passing_speed(frames_in_area: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Compute the individual speed of the pedestrian who pass the area.

    Args:
        frames_in_area (pd.DataFrame): information for each pedestrian in the area, need to contain
                the following columns: 'ID', 'start', 'end', 'frame_start', 'frame_end'
        frame_rate (float): frame rate of the trajectory

    Returns:
        DataFrame containing the columns: 'ID", 'speed' which is the speed in m/s

    """
    speed = pd.DataFrame(frames_in_area["ID"], columns=["ID", "speed"])
    speed["speed"] = pygeos.distance(frames_in_area.start, frames_in_area.end) / (
        np.abs(frames_in_area.frame_end - frames_in_area.frame_start) / frame_rate
    )
    return speed
