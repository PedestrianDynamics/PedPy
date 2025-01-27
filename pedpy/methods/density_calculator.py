"""Module containing functions to compute densities."""

from typing import Tuple

import numpy as np
import pandas as pd
import shapely

from pedpy.column_identifier import (
    COUNT_COL,
    DENSITY_COL,
    DENSITY_SP1_COL,
    DENSITY_SP2_COL,
    FRAME_COL,
    ID_COL,
    POLYGON_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    InputError,
    _apply_lambda_for_intersecting_frames,
    _compute_partial_line_length,
    compute_intersecting_polygons,
    is_species_valid,
)


def compute_classic_density(
    *,
    traj_data: TrajectoryData,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    r"""Compute the classic density per frame inside the given measurement area.

    The classic density :math:`\rho_{classic}(t)` is the number of pedestrians
    inside the given measurement area :math:`M` at the time :math:`t`, divided
    by the area of that space :math:`A(M)`:

    .. math::

        \rho_{classic} = {N \over A(M)},

    where :math:`N` is the number of pedestrians inside the measurement area
    :math:`M`.

    .. image:: /images/classic_density.svg
        :width: 60 %
        :align: center

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        measurement_area (MeasurementArea): area for which the density is
            computed

    Returns:
        DataFrame containing the columns 'frame' and 'density' in :math:`1/m^2`
    """
    peds_in_area = TrajectoryData(
        traj_data.data[
            shapely.contains(measurement_area.polygon, traj_data.data.point)
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
    r"""Compute the Voronoi density per frame inside the given measurement area.

    The Voronoi density :math:`\rho_{voronoi}(t)` is computed based on the
    individual Voronoi polygons :math:`V_i(t)` from
    :func:`~method_utils.compute_individual_voronoi_polygons`.
    Pedestrians whose Voronoi polygon have an intersection with the measurement
    area are taken into account.

    .. image:: /images/voronoi_density.svg
        :width: 60 %
        :align: center

    The Voronoi density :math:`\rho_{voronoi}(t)` is defined as

    .. math::

        \rho_{voronoi}(t) = { \int\int \rho_{xy}(t) dxdy \over A(M)},

    where :math:`\rho_{xy}(t) = 1 / A(V_i(t))` is the individual density of
    each pedestrian, whose :math:`V_i(t) \cap M` and :math:`A(M)` the area of
    the measurement area.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data per
            frame, result from
            :func:`method_utils.compute_individual_voronoi_polygon`
        measurement_area (MeasurementArea): area for which the density is
            computed

    Returns:
        DataFrame containing the columns 'id' and 'density' in :math:`1/m^2`,
        DataFrame containing the columns: 'id', 'frame', 'polygon' which
        contains the Voronoi polygon of the pedestrian, 'intersection' which
        contains the intersection area of the Voronoi polygon and the given
        measurement area.

    """
    df_intersecting = compute_intersecting_polygons(
        individual_voronoi_data=individual_voronoi_data,
        measurement_area=measurement_area,
    )

    df_combined = individual_voronoi_data.merge(
        df_intersecting,
        on=[ID_COL, FRAME_COL],
        how="outer",
    )

    relation_col = "relation"
    df_combined[relation_col] = shapely.area(
        df_combined.intersection
    ) / shapely.area(df_combined.polygon)

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
    r"""Compute the individual density of the pedestrian who pass the area.

    The passing density for each pedestrian :math:`\rho_{passing}(i)` is the
    average number of pedestrian who are in the same measurement area :math:`M`
    in the same time interval (:math:`[t_{in}(i), t_{out}(i)]`) as the
    pedestrian :math:`i` divided by the area of that measurement area
    :math:`A(M)`.
    Then the computation becomes:

    .. math::

        \rho_{passing}(i) = {1 \over {t_{out}(i)-t_{in}(i)}}
        \int^{t_{out}(i)}_{t_{in}(i)} {{N(t)} \over A(M)} dt,

    where

    * :math:`t_{in}(i) = f_{in}(i) / fps` is the time pedestrian :math:`i`
      crosses the first line.
    * :math:`t_{out}(i) = f_{out}(i) / fps` when pedestrian :math:`i` crosses
    the second line.
    * :math:`f_{in}` and :math:`f_{out}` are the frames at which pedestrian
    :math:`i` crosses the first and second lines, respectively.
    * :math:`fps` is the frame rate of the trajectory data, defined by
    :attr:`~trajectory_data.TrajectoryData.frame_rate` of the trajectory data.

    Args:
        density_per_frame (pd.DataFrame): density per frame, result from
            :func:`~density_calculator.compute_classic_density`
        frames (pd.DataFrame): information for each pedestrian in the area,
            result from :func:`~method_utils.compute_frame_range_in_area`

    Returns:
        DataFrame containing the columns 'id' and 'density' in :math:`1/m^2`

    """
    density = pd.DataFrame(frames.id, columns=[ID_COL, DENSITY_COL])

    densities = []
    for _, row in frames.iterrows():
        densities.append(
            density_per_frame[
                density_per_frame.index.to_series().between(
                    int(row.entering_frame),
                    int(row.leaving_frame),
                    inclusive="left",
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


def compute_line_density(
    *,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    species: pd.DataFrame,
) -> pd.DataFrame:
    r"""Calculates density for both species and total density at line.

    The density of each frame is accumulated from

    .. math::
          \frac{1}{A_i(t)} \cdot \frac{w_i(t)}{w},

    for each pedestrian :math:`i` whose Voronoi cell intersects the line.

    * :math:`A_i(t)` is the area of the Voronoi Cell.
    * :math:`w` is the length of the measurement line.
    * :math:`w_i(t)` is the length of the intersecting line of the Voronoi cell
      in frame :math:`t`.

    Results are computed for both species
    (see :func:`~speed_calculator.compute_species`)

    Args:
        individual_voronoi_polygons (pd.DataFrame): individual Voronoi data per
            frame, result
            from :func:`~method_utils.compute_individual_voronoi_polygons`

        measurement_line (MeasurementLine): line at which the density
            is calculated

        species (pd.DataFrame): dataframe containing information about
            the species of every pedestrian intersecting the line,
            result from :func:`~speed_calculator.compute_species`
    Returns:
        Dataframe containing columns 'frame', 'p_sp+1', 'p_sp-1', 'density'
    """
    if not is_species_valid(
        species=species,
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
    ):
        raise InputError(
            "the species data does not contain all information "
            "required to calculate the line density.\n"
            "Perhaps the species was computed with different"
            " Voronoi data or a different measurement line."
        )

    result = _apply_lambda_for_intersecting_frames(
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
        species=species,
        lambda_for_group=lambda group, line: (
            group[DENSITY_COL]
            * (_compute_partial_line_length(group[POLYGON_COL], line))
        ).sum(),
        column_id_sp1=DENSITY_SP1_COL,
        column_id_sp2=DENSITY_SP2_COL,
    )

    result[DENSITY_COL] = result[DENSITY_SP1_COL].fillna(0) + result[
        DENSITY_SP2_COL
    ].fillna(0)
    return result
