"""Module containing functions to compute flows"""

import numpy as np
import pandas as pd
import pygeos

from analyzer.methods.method_utils import _compute_individual_movement


def compute_n_t(
    traj_data: pd.DataFrame, measurement_line: pygeos.Geometry, frame_rate: float
) -> pd.DataFrame:
    """Compute the cumulative number of pedestrians who passed the line for each frame.

    Note: for each pedestrian only the first passing of the line is considered!
    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (pygeos.Geometry): line for which n-t is computed
        frame_rate (float): frame rate of the trajectory data

    Returns:
        DataFrame containing the columns 'frame', 'Cumulative pedestrians', and 'Time [s]'

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
    return nt
