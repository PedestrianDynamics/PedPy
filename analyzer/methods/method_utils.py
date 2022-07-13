"""Helper functions for the analysis methods"""

import pandas as pd
import pygeos


def get_peds_in_area(traj_data: pd.DataFrame, measurement_area: pygeos.Geometry) -> pd.DataFrame:
    """Filters the trajectory date to pedestrians which are inside the given area.

    Args:
        traj_data (pd.DataFrame): trajectory data to filter
        measurement_area (pygeos.Geometry):

    Returns:
         Filtered data set, only containing data of pedestrians inside the measurement_area
    """
    return traj_data[pygeos.contains(measurement_area, traj_data["points"])]


def get_peds_in_frame_range(
    traj_data: pd.DataFrame, min_frame: int = None, max_frame: int = None
) -> pd.DataFrame:
    """Filters the trajectory data by the given min and max frames. If only one of them is given
    (not None) then the other is taken as filter.

    Note:
        min_frame >= max_frame is assumed!

    Args:
        traj_data (pd.DataFrame): trajectory data to filter
        min_frame (int): min frame number still in the filtered data set
        max_frame (int): max frame number still in the filtered data set

    Returns:
        Filtered data set, only containing data within the given frame range

    """
    if min_frame is None and max_frame is not None:
        return traj_data[traj_data["frame"].le(max_frame)]

    if max_frame is None and min_frame is not None:
        return traj_data[traj_data["frame"].ge(min_frame)]

    if min_frame is not None and max_frame is not None:
        return traj_data[traj_data["frame"].between(min_frame, max_frame, inclusive="both")]

    return traj_data


def compute_frame_range_in_area(traj: pd.DataFrame, measurement_area: pygeos.Geometry):
    """Compute the frame ranges for each pedestrian inside the measurement area.

    Note:
        It is assumed that the pedestrians cross the measurement area ones, hence the frame range
        inside the area is the first frame they enter the area to the last frame they are inside.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_area (pygeos.Geometry): rectangular area

    Returns:
        DataFrame containing the columns: 'ID', 'frame_start', 'frame_end', 'start' (first position
        inside the area), 'end' (last position inside the area)
    """
    assert pygeos.equals(
        measurement_area, pygeos.oriented_envelope(measurement_area)
    ), f"The measurement area needs to be a rectangle, please check your input {measurement_area}"

    inside = traj[pygeos.within(traj.points, measurement_area)]
    frames_in_area = (
        inside.groupby("ID")
        .agg(
            ID=("ID", "first"),
            frame_start=("frame", "first"),
            frame_end=("frame", "last"),
            start=("points", "first"),
            end=("points", "last"),
        )
        .reset_index(drop=True)
    )
    return frames_in_area


def _compute_individual_movement(
    traj_data: pd.DataFrame, frame_step: int, bidirectional: bool = True
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step

    The movement is computed for the interval [frame - frame_step: frame + frame_step], if one of
    the boundaries is not contained in the trajectory frame will be used as boundary. Hence, the
    intervals become [frame, frame + frame_step], or [frame - frame_step, frame] respectively.

    Args:
        traj_data (pd.DataFrame): trajectory data
        frame_step (int): how frames back and forwards are used to compute the movement
        bidirectional (bool): if True also the prev. frame_step points will be used to determine
            the movement

    Returns:
        DataFrame containing the columns: 'ID', 'frame', 'start', 'end', 'start_frame, and
        'end_frame'. Where 'start'/'end' are the points where the movement start/ends, and
        'start_frame'/'end_frame' are the corresponding frames.
    """
    df_movement = traj_data.copy(deep=True)

    if bidirectional:
        df_movement["start"] = (
            df_movement.groupby("ID")["points"].shift(frame_step).fillna(df_movement["points"])
        )
        df_movement["start_frame"] = (
            df_movement.groupby("ID")["frame"].shift(frame_step).fillna(df_movement["frame"])
        )
    else:
        df_movement["start"] = df_movement["points"]
        df_movement["start_frame"] = df_movement["frame"]

    df_movement["end"] = (
        df_movement.groupby("ID")["points"].shift(-frame_step).fillna(df_movement["points"])
    )
    df_movement["end_frame"] = (
        df_movement.groupby("ID")["frame"].shift(-frame_step).fillna(df_movement["frame"])
    )

    return df_movement[["ID", "frame", "start", "end", "start_frame", "end_frame"]]
