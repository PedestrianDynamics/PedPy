"""Module containing functions to compute densities"""

import pandas as pd
import pygeos

from report.data.geometry import Geometry
from report.data.trajectory_data import TrajectoryData
from report.methods.method_utils import (
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
    get_num_peds_per_frame,
    get_peds_in_area,
)


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

    # Rename column and add missing zero values
    density.columns = ["classic density"]
    density = density.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density


def compute_voronoi_density(
    traj_data: TrajectoryData, measurement_area: pygeos.Geometry, geometry: Geometry
) -> pd.DataFrame:
    """Compute the voronoi density of the trajectory per frame inside the given measurement area

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        measurement_area (pygeos.Geometry): area for which the density is computed
        geometry (Geometry): bounding area, where pedestrian are supposed to be

    Returns:
          DataFrame containing the columns: 'frame' and 'voronoi density',
          DataFrame containing the columns: 'ID', 'frame', 'individual voronoi'
    """
    df_individual = compute_individual_voronoi_polygons(traj_data.data, geometry)
    df_intersecting = compute_intersecting_polygons(df_individual, measurement_area)

    df_combined = pd.merge(df_individual, df_intersecting, on=["ID", "frame"], how="outer")
    df_combined["relation"] = pygeos.area(df_combined["intersection voronoi"]) / pygeos.area(
        df_combined["individual voronoi"]
    )

    df_voronoi_density = (
        df_combined.groupby("frame")["relation"].sum() / pygeos.area(measurement_area)
    ).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density.columns = ["voronoi density"]
    df_voronoi_density = df_voronoi_density.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )

    return df_voronoi_density, df_individual
