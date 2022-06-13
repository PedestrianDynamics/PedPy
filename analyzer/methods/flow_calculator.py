"""Module containing functions to compute flows"""
from typing import Tuple

import numpy as np
import pandas as pd
import pygeos

from analyzer.methods.method_utils import _compute_individual_movement


def compute_n_t(
    traj_data: pd.DataFrame, measurement_line: pygeos.Geometry, frame_rate: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the cumulative number of pedestrians who passed the line for each frame.

    Note: for each pedestrian only the first passing of the line is considered!
    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (pygeos.Geometry): line for which n-t is computed
        frame_rate (float): frame rate of the trajectory data

    Returns:
        DataFrame containing the columns 'frame', 'Cumulative pedestrians', and 'Time [s]' and
        DataFrame containing the columns 'ID', and 'frame'
    """
    df_movement = _compute_individual_movement(traj_data, 1, False)

    # stack is used to get the coordinates in the correct order, as pygeos does not support
    # creating linestring from points directly. The resulting array looks as follows:
    # [[[x_0_start, y_0_start], [x_0_end, y_0_end]],
    #  [[x_1_start, y_1_start], [x_1_end, y_1_end]], ... ]
    df_movement["movement"] = pygeos.linestrings(
        np.stack(
            [
                pygeos.get_coordinates(df_movement["start"]),
                pygeos.get_coordinates(df_movement["end"]),
            ],
            axis=1,
        )
    )

    # crossing means, the current movement crosses the line and the end point of the movement is
    # not on the line. The result is sorted by frame number
    crossing_frames = (
        df_movement.loc[
            (pygeos.intersects(df_movement["movement"], measurement_line))
            & (~pygeos.intersects(df_movement["end"], measurement_line))
        ]
        .groupby("ID")["frame"]
        .min()
        .sort_values()
    )

    nt = (
        crossing_frames.reset_index()
        .groupby("frame")["frame"]
        .size()
        .cumsum()
        .rename("Cumulative pedestrians")
    )

    # add missing values, to get values for each frame. First fill everything with the previous
    # valid value (fillna('ffill')). When this is done only the frame at the beginning where no one
    # has passed the line yet area missing (fillna(0)).
    nt = (
        nt.reindex(list(range(traj_data.frame.min(), traj_data.frame.max() + 1)))
        .fillna(method="ffill")
        .fillna(0)
    )

    nt = nt.to_frame()

    # frame number is the index
    nt["Time [s]"] = nt.index / frame_rate
    return nt, crossing_frames


def compute_flow(
    nt: pd.DataFrame,
    crossing_frames: pd.DataFrame,
    individual_speed: pd.DataFrame,
    delta_t: int,
    frame_rate: float,
) -> pd.DataFrame:
    """Compute the flow for the given crossing_frames and nt.

    Args:
        nt (pd.DataFrame): DataFrame containing the columns 'frame', 'Cumulative pedestrians', and 'Time [s]' (see result from compute_nt)
        crossing_frames (pd.DataFrame): DataFrame containing the columns 'ID', and 'frame' (see result from compute_nt)
        individual_speed (pd.DataFrame): DataFrame containing the columns 'ID', 'frame', and 'speed'
        delta_t (int): size of the time interval to compute the flow
        frame_rate (float): frame rate of the trajectories

    Returns:
        DataFrame containing the columns 'Flow rate(1/s)', and 'Mean velocity(m/s)'
    """
    crossing_speeds = pd.merge(crossing_frames, individual_speed, on=["ID", "frame"])

    # Get frame where the first person passes the line
    num_passed_before = 0
    passed_frame_before = nt[nt["Cumulative pedestrians"] > 0].index.min()

    flow = pd.DataFrame(columns=["Flow rate(1/s)", "Mean velocity(m/s)"])

    for frame in range(passed_frame_before + delta_t, nt.index.max(), delta_t):
        passed_num_peds = nt.loc[frame]["Cumulative pedestrians"]
        passed_frame = nt[nt["Cumulative pedestrians"] == passed_num_peds].index.min() + 1

        if passed_num_peds != num_passed_before:
            n = passed_num_peds - num_passed_before
            t = passed_frame - passed_frame_before

            flow_rate = n / t * frame_rate
            v = crossing_speeds[
                crossing_speeds.frame.between(passed_frame_before, passed_frame, inclusive="both")
            ]["speed"].mean()

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame

            flow = flow.append(
                {"Flow rate(1/s)": flow_rate, "Mean velocity(m/s)": v}, ignore_index=True
            )

    return flow
