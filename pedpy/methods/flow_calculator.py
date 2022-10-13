"""Module containing functions to compute flows"""
from typing import Tuple

import pandas as pd
from shapely import LineString

from pedpy.methods.method_utils import _compute_crossing_frames


def compute_n_t(
    *,
    traj_data: pd.DataFrame,
    measurement_line: LineString,
    frame_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the cumulative number of pedestrians who passed the line for
    each frame.

    Note: for each pedestrian only the first passing of the line is considered!
    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (shapely.LineString): line for which n-t is computed
        frame_rate (float): frame rate of the trajectory data

    Returns:
        DataFrame containing the columns 'frame', 'Cumulative pedestrians',
        and 'Time [s]' and
        DataFrame containing the columns 'ID', and 'frame'
    """
    crossing_frames = _compute_crossing_frames(traj_data, measurement_line)
    crossing_frames = (
        crossing_frames.groupby("ID")["frame"].min().sort_values().reset_index()
    )

    nt = (
        crossing_frames.groupby("frame")["frame"]
        .size()
        .cumsum()
        .rename("Cumulative pedestrians")
    )

    # add missing values, to get values for each frame. First fill everything
    # with the previous valid value (fillna('ffill')). When this is done only
    # the frame at the beginning where no one has passed the line yet area
    # missing (fillna(0)).
    nt = (
        nt.reindex(
            list(range(traj_data.frame.min(), traj_data.frame.max() + 1))
        )
        .fillna(method="ffill")
        .fillna(0)
    )

    nt = nt.to_frame()
    nt["Cumulative pedestrians"] = nt["Cumulative pedestrians"].astype(int)

    # frame number is the index
    nt["Time [s]"] = nt.index / frame_rate
    return nt, crossing_frames


def compute_flow(
    *,
    nt: pd.DataFrame,
    crossing_frames: pd.DataFrame,
    individual_speed: pd.DataFrame,
    delta_t: int,
    frame_rate: float,
) -> pd.DataFrame:
    """Compute the flow for the given crossing_frames and nt.

    Args:
        nt (pd.DataFrame): DataFrame containing the columns 'frame',
            'Cumulative pedestrians', and 'Time [s]' (see result from
            compute_nt)
        crossing_frames (pd.DataFrame): DataFrame containing the columns
            'ID',  and 'frame' (see result from compute_nt)
        individual_speed (pd.DataFrame): DataFrame containing the columns
            'ID', 'frame', and 'speed'
        delta_t (int): size of the time interval to compute the flow
        frame_rate (float): frame rate of the trajectories

    Returns:
        DataFrame containing the columns 'Flow rate(1/s)', and 'Mean
        velocity(m/s)'
    """
    crossing_speeds = pd.merge(
        crossing_frames, individual_speed, on=["ID", "frame"]
    )

    # Get frame where the first person passes the line
    num_passed_before = 0
    passed_frame_before = nt[nt["Cumulative pedestrians"] > 0].index.min()

    flow = pd.DataFrame(columns=["Flow rate(1/s)", "Mean velocity(m/s)"])

    for frame in range(passed_frame_before + delta_t, nt.index.max(), delta_t):
        passed_num_peds = nt.loc[frame]["Cumulative pedestrians"]
        passed_frame = (
            nt[nt["Cumulative pedestrians"] == passed_num_peds].index.min() + 1
        )

        if passed_num_peds != num_passed_before:
            num_passing_peds = passed_num_peds - num_passed_before
            t = passed_frame - passed_frame_before

            flow_rate = num_passing_peds / t * frame_rate
            v = crossing_speeds[
                crossing_speeds.frame.between(
                    passed_frame_before, passed_frame, inclusive="both"
                )
            ]["speed"].mean()

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame

            flow = flow.append(
                {"Flow rate(1/s)": flow_rate, "Mean velocity(m/s)": v},
                ignore_index=True,
            )

    return flow
