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
