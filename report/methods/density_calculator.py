"""Module containing functions to compute densities"""

import pandas as pd
import pygeos

from report.data.trajectory_data import TrajectoryData
from report.methods.method_utils import get_num_peds_per_frame, get_peds_in_area


def compute_classic_density(
    traj_data: TrajectoryData,
    measurement_area: pygeos.Geometry,
) -> pd.DataFrame:
    """Compute the classic density of the trajectory per frame inside the given measurement area

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        measurement_area (pygeos.Geometry): area for which the density is computed

    Returns:
        DataFrame containing the columns: 'frame' and 'classic density'
    """
    peds_in_area = get_peds_in_area(traj_data.data, measurement_area)
    peds_in_area_per_frame = get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / pygeos.area(measurement_area)
    density.columns = ["classic density"]
    density = density.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density
