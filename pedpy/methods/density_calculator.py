"""Module containing functions to compute densities."""
from typing import Tuple

import numpy as np
import pandas as pd
import shapely

from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import compute_intersecting_polygons
from pedpy.types import COUNT_COL, DENSITY_COL, FRAME_COL, ID_COL


def compute_classic_density(
    *,
    traj_data: TrajectoryData,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    """Compute the classic density per frame inside the given measurement area.

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        measurement_area (MeasurementArea): area for which the density is computed


    Returns:
        DataFrame containing the columns: 'frame' and 'density'
    """
    peds_in_area = TrajectoryData(
        traj_data.data[
            shapely.contains(measurement_area.polygon, traj_data.data.points)
        ],
        traj_data.frame_rate,
    )
    peds_in_area_per_frame = _get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / measurement_area.area

    # Rename column and add missing zero values
    density.columns = [DENSITY_COL]
    density = density.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density


def compute_voronoi_density(
    *,
    individual_voronoi_data: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the voronoi density per frame inside the given measurement area.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data per
            frame needs to contain the columns: 'ID', 'frame',
            'voronoi_polygon', which holds a shapely.Polygon
        measurement_area (MeasurementArea): area for which the density is
            computed
    Returns:
        DataFrame containing the columns: 'frame' and 'density',
        DataFrame containing the columns: 'ID', 'frame', 'voronoi_polygon',
        'voronoi_ma_intersection'.

    """
    df_intersecting = compute_intersecting_polygons(
        individual_voronoi_data=individual_voronoi_data,
        measurement_area=measurement_area,
    )

    df_combined = pd.merge(
        individual_voronoi_data,
        df_intersecting,
        on=[ID_COL, FRAME_COL],
        how="outer",
    )

    relation_col = "relation"
    df_combined[relation_col] = shapely.area(
        df_combined.voronoi_ma_intersection
    ) / shapely.area(df_combined.voronoi_polygon)

    df_voronoi_density = (
        df_combined.groupby(df_combined.frame).relation.sum()
        / measurement_area.area
    ).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density.columns = [DENSITY_COL]
    df_voronoi_density = df_voronoi_density.reindex(
        list(
            range(
                individual_voronoi_data.frame.min(),
                individual_voronoi_data.frame.max() + 1,
            )
        ),
        fill_value=0.0,
    )

    return (
        df_voronoi_density,
        df_combined.loc[:, df_combined.columns != relation_col],
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
                'frame_end'.

    Returns:
          DataFrame containing the columns: 'ID' and 'density' in 1/m^2

    """
    density = pd.DataFrame(frames.ID, columns=[ID_COL, DENSITY_COL])

    densities = []
    for _, row in frames.iterrows():
        densities.append(
            density_per_frame[
                density_per_frame.index.to_series().between(
                    int(row.frame_start), int(row.frame_end), inclusive="left"
                )
            ].mean()
        )
    density.density = np.array(densities)
    return density


def _get_num_peds_per_frame(traj_data: TrajectoryData) -> pd.DataFrame:
    """Returns the number of pedestrians in each frame as DataFrame.

    Args:
        traj_data (TrajectoryData): trajectory data

    Returns:
        DataFrame containing the columns: 'frame' (as index) and 'num_peds'.

    """
    num_peds_per_frame = (
        traj_data.data.groupby(traj_data.data.frame)
        .agg({ID_COL: "count"})
        .rename(columns={ID_COL: COUNT_COL})
    )

    return num_peds_per_frame
