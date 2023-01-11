"""Module containing functions to compute densities."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from shapely import Polygon

from pedpy.data.geometry import Geometry
from pedpy.methods.method_utils import compute_individual_voronoi_polygons


def compute_classic_density(
    *,
    traj_data: pd.DataFrame,
    measurement_area: Polygon,
) -> pd.DataFrame:
    """Compute the classic density per frame inside the given measurement area.

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (shapely.Polygon): area for which the density is
            computed

    Returns:
        DataFrame containing the columns: 'frame' and 'classic density'
    """
    peds_in_area = traj_data[
        shapely.contains(measurement_area, traj_data["points"])
    ]
    peds_in_area_per_frame = _get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / measurement_area.area

    # Rename column and add missing zero values
    density.columns = ["classic density"]
    density = density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density


def compute_voronoi_density(
    *,
    traj_data: pd.DataFrame,
    measurement_area: shapely.Polygon,
    geometry: Geometry,
    cut_off: Optional[Tuple[float, int]] = None,
    use_blind_points: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the voronoi density per frame inside the given measurement area.

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (shapely.Polygon): area for which the density is
            computed
        geometry (Geometry): bounding area, where pedestrian are supposed to be
        cut_off (Tuple[float, int): radius of max extended voronoi cell (in m),
            number of linear segments in the approximation of circular arcs,
            needs to be divisible by 4!
        use_blind_points (bool): adds extra 4 points outside the geometry to
            also compute voronoi cells when less than 4 peds are in the
            geometry
    Returns:
          DataFrame containing the columns: 'frame' and 'voronoi density',
          DataFrame containing the columns: 'ID', 'frame', 'individual
            voronoi', 'intersecting voronoi'
    """
    df_individual = compute_individual_voronoi_polygons(
        traj_data=traj_data,
        geometry=geometry,
        cut_off=cut_off,
        use_blind_points=use_blind_points,
    )
    df_intersecting = _compute_intersecting_polygons(
        df_individual, measurement_area
    )

    df_combined = pd.merge(
        df_individual, df_intersecting, on=["ID", "frame"], how="outer"
    )
    df_combined["relation"] = shapely.area(
        df_combined["intersection voronoi"]
    ) / shapely.area(df_combined["individual voronoi"])

    df_voronoi_density = (
        df_combined.groupby("frame")["relation"].sum() / measurement_area.area
    ).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density.columns = ["voronoi density"]
    df_voronoi_density = df_voronoi_density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return (
        df_voronoi_density,
        df_combined.loc[:, df_combined.columns != "relation"],
    )


def compute_passing_density(
    *, density_per_frame: pd.DataFrame, frames: pd.DataFrame
) -> pd.DataFrame:
    """Compute the individual density of the pedestrian who pass the area.

    Args:
        density_per_frame (pd.DataFrame): density per frame, DataFrame
                containing the columns: 'frame' (as index) and 'density'
        frames (pd.DataFrame): information for each pedestrian in the area,
                need to contain the following columns: 'ID','frame_start',
                'frame_end'

    Returns:
          DataFrame containing the columns: 'ID' and 'density' in 1/m
    """
    density = pd.DataFrame(frames["ID"], columns=["ID", "density"])

    densities = []
    for _, row in frames.iterrows():
        densities.append(
            density_per_frame[
                density_per_frame.index.to_series().between(
                    int(row.frame_start), int(row.frame_end), inclusive="left"
                )
            ].mean()
        )
    density["density"] = np.array(densities)
    return density


def _get_num_peds_per_frame(traj_data: pd.DataFrame) -> pd.DataFrame:
    """Returns the number of pedestrians in each frame as DataFrame.

    Args:
        traj_data (pd.DataFrame): trajectory data

    Returns:
        DataFrame containing the columns: 'frame' (as index) and 'num_peds'.

    """
    num_peds_per_frame = traj_data.groupby("frame").agg(
        num_peds=("ID", "count")
    )

    return num_peds_per_frame


def _compute_intersecting_polygons(
    individual_voronoi_data: pd.DataFrame, measurement_area: Polygon
) -> pd.DataFrame:
    """Compute the intersection of the voronoi cells with the measurement area.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data, needs
                to contain a column 'individual voronoi' which holds
                shapely.Polygon information
        measurement_area (shapely.Polygon):

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and
        'intersection voronoi'.
    """
    df_intersection = individual_voronoi_data[["ID", "frame"]].copy()
    df_intersection["intersection voronoi"] = shapely.intersection(
        individual_voronoi_data["individual voronoi"], measurement_area
    )
    return df_intersection
