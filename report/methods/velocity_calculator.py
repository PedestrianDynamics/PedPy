"""Module containing functions to compute velocities"""

from typing import Tuple

import numpy as np
import pandas as pd
import pygeos


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
    frame_rate: float,
    frame_step: int,
    movement_direction: np.ndarray = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean velocity per frame

    Note: when using a movement direction the velocity may be negative!

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the velocity
        movement_direction (np.ndarray): main movement direction on which the actual movement is
            projected (default: None, when the un-projected movement should be used)

    Returns:
        DataFrame containing the columns 'frame' and 'speed'
        DataFrame containing the columns 'ID', 'frame', and 'speed'
    """
    df_speed = compute_individual_velocity(traj_data, frame_rate, frame_step, movement_direction)
    df_mean = df_speed.groupby("frame")["speed"].mean()
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


def _compute_individual_movement(traj_data: pd.DataFrame, frame_step: int) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step

    The movement is computed for the interval [frame - frame_step: frame + frame_step], if one of
    the boundaries is not contained in the trajectory frame will be used as boundary. Hence, the
    intervals become [frame, frame + frame_step], or [frame - frame_step, frame] respectively.

    Args:
        traj_data (pd.DataFrame): trajectory data
        frame_step (int): how frames back and forwards are used to compute the movement

    Returns:
        DataFrame containing the columns: 'ID', 'frame', 'start', 'end', 'start_frame, and
        'end_frame'. Where 'start'/'end' are the points where the movement start/ends, and
        'start_frame'/'end_frame' are the corresponding frames.
    """
    df_movement = traj_data.copy(deep=True)

    df_movement["start"] = (
        df_movement.groupby("ID")["points"].shift(frame_step).fillna(df_movement["points"])
    )
    df_movement["end"] = (
        df_movement.groupby("ID")["points"].shift(-frame_step).fillna(df_movement["points"])
    )
    df_movement["start_frame"] = (
        df_movement.groupby("ID")["frame"].shift(frame_step).fillna(df_movement["frame"])
    )
    df_movement["end_frame"] = (
        df_movement.groupby("ID")["frame"].shift(-frame_step).fillna(df_movement["frame"])
    )

    return df_movement[["ID", "frame", "start", "end", "start_frame", "end_frame"]]


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

    return movement_data[["ID", "frame", "speed"]]


def compute_individual_speed_main_movement(
    traj_data: pd.DataFrame,
    measurement_line: pygeos.Geometry,
    frame_rate: float,
    frame_step: int,
):
    """Compute the individual velocity in direction of the main movement direction for each
    pedestrian.

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (pygeos.Geometry): for which the speed is computed
        frame_rate (float): frame rate of the trajectory
        frame_step (int): gives the size of time interval for calculating the velocity

    Returns:
        DataFrame containing the columns: 'ID', 'frame', and 'speed' with the speed in m/s
    """
    movement_direction, normal_vector = _compute_main_movement_direction(
        traj_data, measurement_line
    )
    movement_data = _compute_individual_movement(traj_data, frame_step)

    movement_data = pd.merge(movement_data, movement_direction, on="ID")

    movement_data["distance"] = movement_data["main movement direction"] * (
        np.dot(
            pygeos.get_coordinates(movement_data["end"])
            - pygeos.get_coordinates(movement_data["start"]),
            normal_vector,
        )
        / np.linalg.norm(normal_vector)
    )

    movement_data["speed"] = movement_data["distance"] / (
        (movement_data["end_frame"] - movement_data["start_frame"]) / frame_rate
    )

    return movement_data[["ID", "frame", "speed"]]


def _compute_main_movement_direction(traj_data: pd.DataFrame, measurement_line: pygeos.Geometry):
    """Detect the main movement direction of each pedestrian

    The main movement direction is always orthogonal to the given measurement line. For detection
    a rectangle with width of 1m is created with the measurement line in the middle. Then the first
    and last point inside the area is detected, from which the main movement direction can be
    computed.

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (pygeos.Geometry): line at which the main movement direction is computed

    Returns:
        DataFrame containing the columns: 'ID', 'main movement direction' (which is either +1 or -1)
        normalized normal vector of the measurement line
    """
    line_width = 0.5
    buffered = pygeos.buffer(measurement_line, line_width, cap_style="flat")

    inside = traj_data[pygeos.within(traj_data["points"], buffered)]

    first_frame = inside.loc[inside.groupby("ID")["frame"].idxmin()][["ID", "points"]]
    last_frame = inside.loc[inside.groupby("ID")["frame"].idxmax()][["ID", "points"]]

    movement_direction = pd.merge(first_frame, last_frame, on="ID")

    # Compute normal vector of measurement_line
    coordinates = pygeos.get_coordinates(measurement_line)
    normal_vector = np.array(
        [-1 * (coordinates[1, 1] - coordinates[0, 1]), coordinates[1, 0] - coordinates[0, 0]]
    )
    normal_vector /= np.linalg.norm(normal_vector)

    movement_direction["main movement direction"] = np.sign(
        np.dot(
            pygeos.get_coordinates(movement_direction["points_y"])
            - pygeos.get_coordinates(movement_direction["points_x"]),
            normal_vector,
        )
    )

    return movement_direction[["ID", "main movement direction"]], normal_vector
