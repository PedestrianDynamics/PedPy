"""Module containing functions to compute flows."""
from typing import Tuple

import pandas as pd

from pedpy.data.geometry import MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import compute_crossing_frames
from pedpy.types import (
    CUMULATED_COL,
    FLOW_COL,
    FRAME_COL,
    ID_COL,
    MEAN_SPEED_COL,
    SPEED_COL,
    TIME_COL,
)


def compute_n_t(
    *,
    traj_data: TrajectoryData,
    measurement_line: MeasurementLine,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the frame-wise cumulative number of pedestrians passing the line.

    Warnings:
        For each pedestrian only the first passing of the line is considered!

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (MeasurementLine): line for which n-t is computed

    Returns:
        DataFrame containing the columns 'frame', 'cumulative_pedestrians',
        and 'time' and DataFrame containing the columns 'ID', and 'frame'.

    """
    crossing_frames = compute_crossing_frames(
        traj_data=traj_data, measurement_line=measurement_line
    )
    crossing_frames = (
        crossing_frames.groupby(by=ID_COL)[FRAME_COL]
        .min()
        .sort_values()
        .reset_index()
    )

    n_t = (
        crossing_frames.groupby(by=FRAME_COL)[FRAME_COL]
        .size()
        .cumsum()
        .rename(CUMULATED_COL)
    )

    # add missing values, to get values for each frame. First fill everything
    # with the previous valid value (fillna('ffill')). When this is done only
    # the frame at the beginning where no one has passed the line yet area
    # missing (fillna(0)).
    n_t = (
        n_t.reindex(
            list(
                range(
                    traj_data.data.frame.min(), traj_data.data.frame.max() + 1
                )
            )
        )
        .fillna(method="ffill")
        .fillna(0)
    )

    n_t = n_t.to_frame()
    n_t.cumulative_pedestrians = n_t.cumulative_pedestrians.astype(int)

    # frame number is the index
    n_t[TIME_COL] = n_t.index / traj_data.frame_rate
    return n_t, crossing_frames


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
            'cumulative_pedestrians', and 'time' (see result from
            :func:`compute_nt`)
        crossing_frames (pd.DataFrame): DataFrame containing the columns
            'ID',  and 'frame' (see result from compute_nt)
        individual_speed (pd.DataFrame): DataFrame containing the columns
            'ID', 'frame', and 'speed'
        delta_t (int): size of the time interval to compute the flow
        frame_rate (float): frame rate of the trajectories

    Returns:
        DataFrame containing the columns 'flow' in 1/s, and 'mean_speed' in m/s
    """
    crossing_speeds = pd.merge(
        crossing_frames, individual_speed, on=[ID_COL, FRAME_COL]
    )

    # Get frame where the first person passes the line
    num_passed_before = 0
    passed_frame_before = nt[nt[CUMULATED_COL] > 0].index.min()

    rows = []

    for frame in range(passed_frame_before + delta_t, nt.index.max(), delta_t):
        passed_num_peds = nt.loc[frame][CUMULATED_COL]
        passed_frame = nt[nt[CUMULATED_COL] == passed_num_peds].index.min() + 1

        if passed_num_peds != num_passed_before:
            num_passing_peds = passed_num_peds - num_passed_before
            time_range = passed_frame - passed_frame_before

            flow_rate = num_passing_peds / time_range * frame_rate
            velocity = crossing_speeds[
                crossing_speeds.frame.between(
                    passed_frame_before, passed_frame, inclusive="both"
                )
            ][SPEED_COL].mean()

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame

            rows.append(
                {FLOW_COL: flow_rate, MEAN_SPEED_COL: velocity},
            )

    return pd.DataFrame(rows)
