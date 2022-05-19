"""Module containing functions to compute densities"""

import pandas as pd
import pygeos

from report.data.trajectory_data import TrajectoryData
from report.methods.method_utils import (
    get_num_peds_per_frame,
    get_peds_in_area,
    get_peds_in_frame_range,
)


def compute_classic_density(
    traj_data: TrajectoryData,
    measurement_area: pygeos.Geometry,
    frame_min: int = None,
    frame_max: int = None,
) -> pd.DataFrame:
    """Compute the classic density of the trajectory per frame inside the given measurement area

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        measurement_area (pygeos.Geometry): area for which the density is computed
        frame_min (int): first frame to include density computation (default: None, means start at
                         first frame of traj_data)
        frame_max (int): last frame to include in density computation (default: None, means end at
                         last frame of traj_data)

    Returns:
        DataFrame containing the columns: 'frame' and 'density'
    """
    peds_in_frame_range = get_peds_in_frame_range(traj_data.data, frame_min, frame_max)
    peds_in_area = get_peds_in_area(peds_in_frame_range, measurement_area)
    peds_in_area_per_frame = get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / pygeos.area(measurement_area)
    density.columns = ["density"]
    density = density.reindex(
        list(range(peds_in_frame_range.frame.min(), peds_in_frame_range.frame.max() + 1)),
        fill_value=0.0,
    )

    return density
