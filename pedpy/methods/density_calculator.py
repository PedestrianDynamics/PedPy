"""Module containing functions to compute densities."""

from enum import Enum, auto
from typing import Final, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from pedpy.column_identifier import (
    COUNT_COL,
    DENSITY_COL,
    DENSITY_SP1_COL,
    DENSITY_SP2_COL,
    FRAME_COL,
    ID_COL,
    POINT_COL,
    POLYGON_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import InputError, PedPyTypeError, PedPyValueError
from pedpy.methods.method_utils import (
    _apply_lambda_for_intersecting_frames,
    _check_trajectory_data,
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
    _check_trajectory_data(traj_data)
    peds_in_area = TrajectoryData(
        traj_data.data[shapely.contains(measurement_area.polygon, traj_data.data.point)],
        traj_data.frame_rate,
    )
    peds_in_area_per_frame = _get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / measurement_area.area

    # Rename column and add missing zero values
    density = density.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    density = density.reset_index()
    density.columns = [FRAME_COL, DENSITY_COL]

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
        DataFrame containing the columns 'frame' and 'density' in :math:`1/m^2`,
        DataFrame containing the columns: 'id', 'frame', 'polygon' which
        contains the Voronoi polygon of the pedestrian, 'density' in
        :math:`1/m^2` which contains the individual density of the pedestrian,
        'intersection' which contains the intersection area of the Voronoi
        polygon and the given measurement area.

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
    df_combined[relation_col] = shapely.area(df_combined.intersection) / shapely.area(df_combined.polygon)

    df_voronoi_density = (df_combined.groupby(df_combined.frame).relation.sum() / measurement_area.area).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density = df_voronoi_density.reindex(
        list(
            range(
                individual_voronoi_data.frame.min(),
                individual_voronoi_data.frame.max() + 1,
            )
        ),
        fill_value=0.0,
    )
    df_voronoi_density = df_voronoi_density.reset_index()
    df_voronoi_density.columns = [FRAME_COL, DENSITY_COL]

    return (
        df_voronoi_density,
        df_combined.loc[:, df_combined.columns != relation_col],
    )


def compute_passing_density(*, density_per_frame: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
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
                density_per_frame.frame.between(
                    int(row.entering_frame),
                    int(row.leaving_frame),
                    inclusive="left",
                )
            ].density.mean()
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
        traj_data.data.groupby(traj_data.data.frame).agg({ID_COL: "count"}).rename(columns={ID_COL: COUNT_COL})
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
        Dataframe containing columns 'frame', 'p_sp+1' which contains the
        density in :math:`1/m^2` for species +1, 'p_sp-1' which contains the
        density in :math:`1/m^2` for species -1, 'density' which contains the
        density at the line in :math:`1/m^2`.
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
            group[DENSITY_COL] * (_compute_partial_line_length(group[POLYGON_COL], line))
        ).sum(),
        column_id_sp1=DENSITY_SP1_COL,
        column_id_sp2=DENSITY_SP2_COL,
    )

    result[DENSITY_COL] = result[DENSITY_SP1_COL].fillna(0) + result[DENSITY_SP2_COL].fillna(0)
    return result


_MIN_SPACING_THRESHOLD: Final = 1e-12
_HALF_FACTOR: Final = 0.5


class SingleFileMethod(Enum):  # pylint: disable=too-few-public-methods
    r"""1D individual density definition for single-file trajectories.

    Attributes:
        VORONOI: 1D Voronoi cell width :math:`d_k = (s_{k+1} - s_{k-1}) / 2`,
            yielding individual density :math:`\rho_k = 1 / d_k`.
        HEADWAY: Forward headway :math:`h_k = s_{k+1} - s_{k}`, yielding
            individual density :math:`\rho_k = 1 / h_k`.
    """

    VORONOI = auto()
    HEADWAY = auto()


def _project_single_file_positions(
    df: pd.DataFrame,
    centerline: Optional[shapely.LineString],
    axis: str,
    closed: bool,
    track_length: Optional[float],
) -> Tuple[npt.NDArray[np.float64], Optional[float]]:
    """Project pedestrian positions onto a 1D coordinate and resolve loop length."""
    if centerline is not None:
        if not isinstance(centerline, shapely.LineString):
            raise PedPyTypeError(
                f"Expected 'centerline' to be a shapely.LineString, got {type(centerline).__name__!r} instead."
            )
        s_coords = np.asarray(shapely.line_locate_point(centerline, df[POINT_COL].to_numpy()), dtype=np.float64)
        resolved_length = track_length if track_length is not None else (float(centerline.length) if closed else None)
    else:
        if axis not in (X_COL, Y_COL):
            raise PedPyValueError(f"'axis' must be '{X_COL}' or '{Y_COL}', got {axis!r}.")
        s_coords = df[axis].to_numpy(dtype=np.float64)
        resolved_length = track_length

    if closed:
        if resolved_length is None or resolved_length <= 0.0:
            raise PedPyValueError("A positive 'track_length' (or non-empty 'centerline') is required when closed=True.")
        s_coords = np.mod(s_coords, resolved_length)

    return s_coords, resolved_length


def _compute_single_file_frame_cells(
    s_sorted: npt.NDArray[np.float64],
    closed: bool,
    loop_length: Optional[float],
    method: SingleFileMethod,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute ``(spacing, left_edge, right_edge)`` for sorted 1D coordinates."""
    n_peds = len(s_sorted)
    spacing = np.full(n_peds, np.nan, dtype=np.float64)
    left_edge = np.full(n_peds, np.nan, dtype=np.float64)
    right_edge = np.full(n_peds, np.nan, dtype=np.float64)

    if n_peds < 2:
        return spacing, left_edge, right_edge

    if closed:
        assert loop_length is not None
        forward_gaps = np.empty(n_peds, dtype=np.float64)
        forward_gaps[:-1] = s_sorted[1:] - s_sorted[:-1]
        forward_gaps[-1] = (s_sorted[0] + loop_length) - s_sorted[-1]
        backward_gaps = np.roll(forward_gaps, 1)

        if method == SingleFileMethod.HEADWAY:
            spacing = forward_gaps
            left_edge = s_sorted
            right_edge = s_sorted + forward_gaps
        else:
            spacing = _HALF_FACTOR * (backward_gaps + forward_gaps)
            left_edge = s_sorted - _HALF_FACTOR * backward_gaps
            right_edge = s_sorted + _HALF_FACTOR * forward_gaps
        return spacing, left_edge, right_edge

    if method == SingleFileMethod.HEADWAY:
        spacing[:-1] = s_sorted[1:] - s_sorted[:-1]
        left_edge[:-1] = s_sorted[:-1]
        right_edge[:-1] = s_sorted[1:]
    elif n_peds >= 3:
        midpoints = _HALF_FACTOR * (s_sorted[:-1] + s_sorted[1:])
        left_edge[1:-1] = midpoints[:-1]
        right_edge[1:-1] = midpoints[1:]
        spacing[1:-1] = right_edge[1:-1] - left_edge[1:-1]

    return spacing, left_edge, right_edge


def _interval_overlap_length(
    left_edge: npt.NDArray[np.float64],
    right_edge: npt.NDArray[np.float64],
    s_min: float,
    s_max: float,
    closed: bool,
    loop_length: Optional[float],
) -> npt.NDArray[np.float64]:
    """Compute 1D overlap of ``[left_edge, right_edge]`` with ``[s_min, s_max]``."""
    valid = np.isfinite(left_edge) & np.isfinite(right_edge)
    overlap = np.zeros_like(left_edge, dtype=np.float64)
    if not np.any(valid):
        return overlap

    l_v = left_edge[valid]
    r_v = right_edge[valid]
    seg_overlap = np.maximum(0.0, np.minimum(r_v, s_max) - np.maximum(l_v, s_min))
    if closed and loop_length is not None:
        seg_overlap += np.maximum(0.0, np.minimum(r_v + loop_length, s_max) - np.maximum(l_v + loop_length, s_min))
        seg_overlap += np.maximum(0.0, np.minimum(r_v - loop_length, s_max) - np.maximum(l_v - loop_length, s_min))
    overlap[valid] = seg_overlap
    return overlap


def compute_single_file_density(
    *,
    traj_data: TrajectoryData,
    centerline: Optional[shapely.LineString] = None,
    axis: str = X_COL,
    closed: bool = False,
    track_length: Optional[float] = None,
    method: SingleFileMethod = SingleFileMethod.VORONOI,
    measurement_interval: Optional[Tuple[float, float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    r"""Compute 1D individual (and optional interval) densities for single-file trajectories.

    Projects pedestrian positions onto a 1D coordinate :math:`s_k(t)` (either
    arc length along ``centerline`` or coordinate ``axis``) and evaluates
    individual 1D densities in :math:`1/\mathrm{m}`:

    - :attr:`SingleFileMethod.VORONOI`: 1D Voronoi cell width
      :math:`d_k = (s_{k+1} - s_{k-1}) / 2` and :math:`\rho_k = 1 / d_k`.
    - :attr:`SingleFileMethod.HEADWAY`: forward headway
      :math:`h_k = s_{k+1} - s_k` and :math:`\rho_k = 1 / h_k`.

    When ``closed=True`` (e.g. oval or circular single-file tracks of
    circumference ``track_length``), wrap-around neighbors are used so every
    pedestrian receives a finite cell width summing to ``track_length``.

    Args:
        traj_data (TrajectoryData): Trajectory data to analyze.
        centerline (shapely.LineString, optional): Optional 1D reference path
            onto which pedestrian positions are projected via arc length.
        axis (str): Coordinate column (``"x"`` or ``"y"``) used when
            ``centerline`` is ``None`` (default: ``"x"``).
        closed (bool): Treat the 1D domain as periodic with circumference
            ``track_length`` (default: ``False``).
        track_length (float, optional): Periodic track circumference in meters
            when ``closed=True`` (defaults to ``centerline.length`` when
            ``centerline`` is provided).
        method (SingleFileMethod): 1D cell definition (default:
            ``SingleFileMethod.VORONOI``).
        measurement_interval (Tuple[float, float], optional): Optional 1D
            measurement section ``(s_min, s_max)``. When provided, returns a
            2-tuple ``(frame_density_df, individual_df)`` analogous to
            :func:`compute_voronoi_density`.

    Returns:
        DataFrame with columns ``id``, ``frame``, ``arc_length``, ``spacing``,
        and ``density`` (in :math:`1/\mathrm{m}`) when ``measurement_interval``
        is ``None``; otherwise ``(frame_density_df, individual_df)``.
    """
    _check_trajectory_data(traj_data)
    if not isinstance(method, SingleFileMethod):
        raise PedPyValueError(f"Unsupported single-file density method: {method!r}.")
    if measurement_interval is not None and measurement_interval[1] <= measurement_interval[0]:
        raise PedPyValueError("'measurement_interval' must satisfy s_max > s_min.")

    work_df = traj_data.data[[ID_COL, FRAME_COL, X_COL, Y_COL, POINT_COL]].copy()
    work_df["_orig_order"] = np.arange(len(work_df))
    s_coords, resolved_length = _project_single_file_positions(work_df, centerline, axis, closed, track_length)
    work_df["arc_length"] = s_coords

    per_frame_chunks: list[pd.DataFrame] = []
    frame_densities: dict[int, float] = {}

    for frame_val, group in work_df.groupby(FRAME_COL):
        sorted_group = group.sort_values(by=["arc_length", ID_COL]).copy()
        s_sorted = sorted_group["arc_length"].to_numpy(dtype=np.float64)
        spacing, left_e, right_e = _compute_single_file_frame_cells(s_sorted, closed, resolved_length, method)
        density = np.where(
            np.isfinite(spacing) & (spacing > _MIN_SPACING_THRESHOLD),
            np.divide(1.0, spacing, out=np.full_like(spacing, np.nan), where=spacing > _MIN_SPACING_THRESHOLD),
            np.nan,
        )
        sorted_group["spacing"] = spacing
        sorted_group[DENSITY_COL] = density
        per_frame_chunks.append(sorted_group)

        if measurement_interval is not None:
            s_min, s_max = float(measurement_interval[0]), float(measurement_interval[1])
            interval_len = s_max - s_min
            if method == SingleFileMethod.VORONOI:
                overlaps = _interval_overlap_length(left_e, right_e, s_min, s_max, closed, resolved_length)
                valid = (overlaps > 0.0) & np.isfinite(spacing) & (spacing > _MIN_SPACING_THRESHOLD)
                frame_densities[int(frame_val)] = float(np.sum(overlaps[valid] / spacing[valid]) / interval_len)
            else:
                in_sec = (s_sorted >= s_min) & (s_sorted <= s_max)
                frame_densities[int(frame_val)] = float(np.count_nonzero(in_sec) / interval_len)

    if per_frame_chunks:
        combined = pd.concat(per_frame_chunks, ignore_index=True).sort_values("_orig_order")
    else:
        combined = pd.DataFrame(columns=[ID_COL, FRAME_COL, "arc_length", "spacing", DENSITY_COL])
    individual_df = combined[[ID_COL, FRAME_COL, "arc_length", "spacing", DENSITY_COL]].reset_index(drop=True)

    if measurement_interval is None:
        return individual_df

    if traj_data.data.empty:
        return pd.DataFrame(columns=[FRAME_COL, DENSITY_COL]), individual_df

    full_frames = list(range(int(traj_data.data[FRAME_COL].min()), int(traj_data.data[FRAME_COL].max()) + 1))
    frame_series = pd.Series(frame_densities, dtype=float).reindex(full_frames, fill_value=0.0)
    frame_df = frame_series.reset_index()
    frame_df.columns = [FRAME_COL, DENSITY_COL]
    return frame_df, individual_df
