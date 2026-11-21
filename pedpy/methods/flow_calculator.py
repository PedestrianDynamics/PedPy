"""Module containing functions to compute flows."""

from typing import Any, Optional, Tuple

import pandas as pd

from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    FIRST_FRAME_COL,
    FLOW_COL,
    FLOW_SP1_COL,
    FLOW_SP2_COL,
    FRAME_COL,
    ID_COL,
    LAST_FRAME_COL,
    MEAN_SPEED_COL,
    SPEED_COL,
    TIME_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import InputError
from pedpy.methods.density_calculator import compute_classic_density
from pedpy.methods.method_utils import (
    DataValidationStatus,
    _apply_lambda_for_intersecting_frames,
    _check_trajectory_data,
    _compute_orthogonal_speed_in_relation_to_proportion,
    compute_crossing_frames,
    is_individual_speed_valid,
    is_species_valid,
)


def compute_n_t(
    *,
    traj_data: TrajectoryData,
    measurement_line: MeasurementLine,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the frame-wise cumulative number of pedestrians passing the line.

    Records the frames, when a pedestrian crossed the given measurement line.
    A frame counts as crossed when the movement is across the line, but does
    not end on it. Then the next frame when the movement starts on the line
    is counted as crossing frame.

    .. warning::

        For each pedestrian only the first passing of the line is considered!

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (MeasurementLine): line for which n-t is computed

    Returns:
        DataFrame containing the columns 'frame', 'cumulative_pedestrians',
        and 'time' since frame 0, and DataFrame containing the columns 'ID',
        and 'frame' which gives the frame the pedestrian crossed the
        measurement line.

    """
    _check_trajectory_data(traj_data)
    crossing_frames = compute_crossing_frames(traj_data=traj_data, measurement_line=measurement_line)
    crossing_frames = crossing_frames.groupby(by=ID_COL)[FRAME_COL].min().sort_values().reset_index()

    n_t = crossing_frames.groupby(by=FRAME_COL)[FRAME_COL].size().cumsum().rename(CUMULATED_COL)

    # add missing values, to get values for each frame. First fill everything
    # with the previous valid value (fillna('ffill')). When this is done only
    # the frame at the beginning where no one has passed the line yet area
    # missing (fillna(0)).
    n_t = n_t.reindex(list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1))).ffill().fillna(0)

    n_t = n_t.to_frame()
    n_t[CUMULATED_COL] = n_t.cumulative_pedestrians.astype(int)

    # frame number is the index
    n_t[TIME_COL] = n_t.index / traj_data.frame_rate
    n_t = n_t.reset_index()
    n_t.columns = [FRAME_COL, CUMULATED_COL, TIME_COL]
    return n_t, crossing_frames


def compute_flow(
    *,
    nt: pd.DataFrame,
    crossing_frames: pd.DataFrame,
    individual_speed: Optional[pd.DataFrame] = None,
    delta_frame: int,
    frame_rate: float,
) -> pd.DataFrame:
    r"""Compute the flow for the given frame window from the nt information.

    Computes the flow :math:`J` in a frame interval of length
    :data:`delta_frame` (:math:`\Delta frame`). The first intervals starts,
    when the first person crossed the measurement, given by
    :data:`crossing_frames`. The next interval always starts at the time when
    the last person in the previous frame interval crossed the line.

    .. image:: /images/flow.svg
        :align: center
        :width: 80 %

    In each of the time interval it is checked, if any person has crossed the
    line, if yes, a flow :math:`J` can be computed. From the first frame the
    line was crossed :math:`f^{\Delta frame}_1`, the last frame someone
    crossed the line :math:`f^{\Delta frame}_N` the length of the frame
    interval :math:`\Delta f` can be computed:

    .. math::

        \Delta f = f^{\Delta frame}_N - f^{\Delta frame}_1

    This directly together with the frame rate with :data:`frame_rate`
    (:math:`fps`) gives the time interval :math:`\Delta t`:

    .. math::

        \Delta t = \Delta f / fps

    Given the number of pedestrian crossing the line is given by
    :math:`N^{\Delta frame}`, the flow :math:`J` becomes:

    .. math::

        J = \frac{N^{\Delta frame}}{\Delta t}

    .. image:: /images/flow_zoom.svg
        :align: center
        :width: 60 %

    When :data:`individual_speed` is provided, the mean speed of the
    pedestrians when crossing the line is also computed:

    .. math::

        v_{crossing} = {1 \over N^{\Delta t} } \sum^{N^{\Delta t}}_{i=1} v_i(t)

    Args:
        nt (pd.DataFrame): DataFrame containing the columns 'frame',
            'cumulative_pedestrians', and 'time' (see result from
            :func:`~flow_calculator.compute_n_t`)
        crossing_frames (pd.DataFrame): DataFrame containing the columns
            'ID', and 'frame' (see result from
            :func:`~flow_calculator.compute_n_t`)
        individual_speed (Optional[pd.DataFrame]): Optional DataFrame
            containing the columns 'ID', 'frame', and 'speed'. When ``None``,
            only ``flow`` is computed and ``mean_speed`` is omitted.
        delta_frame (int): size of the frame interval to compute the flow
        frame_rate (float): frame rate of the trajectories

    Returns:
        DataFrame containing the column 'flow' in 1/s (and 'mean_speed' in
        m/s when ``individual_speed`` is provided).
    """
    crossing_speeds: Optional[pd.DataFrame] = None
    if individual_speed is not None:
        crossing_speeds = crossing_frames.merge(individual_speed, on=[ID_COL, FRAME_COL])

    if nt[nt[CUMULATED_COL] > 0].empty:
        cols = [FLOW_COL, MEAN_SPEED_COL] if individual_speed is not None else [FLOW_COL]
        return pd.DataFrame(columns=cols)

    # Get frame where the first person passes the line
    num_passed_before = 0
    passed_frame_before = int(nt[nt[CUMULATED_COL] > 0].frame.min())

    rows: list[dict[str, Any]] = []

    for frame in range(passed_frame_before + delta_frame, int(nt.frame.max()), delta_frame):
        passed_num_peds = int(nt[CUMULATED_COL][nt[FRAME_COL] == frame].to_numpy()[0])
        passed_frame = int(nt[nt[CUMULATED_COL] == passed_num_peds].frame.min()) + 1

        if passed_num_peds != num_passed_before:
            num_passing_peds = passed_num_peds - num_passed_before
            time_range = passed_frame - passed_frame_before

            flow_rate = num_passing_peds / time_range * frame_rate
            row: dict[str, Any] = {FLOW_COL: flow_rate}
            if crossing_speeds is not None:
                velocity = crossing_speeds[
                    crossing_speeds.frame.between(passed_frame_before, passed_frame, inclusive="both")
                ][SPEED_COL].mean()
                row[MEAN_SPEED_COL] = velocity

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame

            rows.append(row)

    if not rows:
        cols = [FLOW_COL, MEAN_SPEED_COL] if individual_speed is not None else [FLOW_COL]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def compute_classic_fundamental_diagram(
    *,
    traj_data: TrajectoryData,
    measurement_area: MeasurementArea,
    measurement_line: MeasurementLine,
    delta_frame: int,
    individual_speed: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    r"""Compute aligned classical density and flow intervals for a Fundamental Diagram.

    Pairs the flow :math:`J` across ``measurement_line`` (in ``1/s``) over each
    crossing interval ``[first_frame, last_frame]`` of width ``delta_frame``
    with the mean classical density :math:`\rho_{classic}` inside
    ``measurement_area`` (in ``1/m^2``) over that exact same frame interval.

    Args:
        traj_data: Trajectory data to analyze.
        measurement_area: Measurement area used for :func:`compute_classic_density`.
        measurement_line: Measurement line used for :func:`compute_n_t` and flow.
        delta_frame: Frame window size for flow aggregation.
        individual_speed: Optional DataFrame with columns ``id``, ``frame``,
            and ``speed``; if provided, ``mean_speed`` (in ``m/s``) at the line
            is included as well.

    Returns:
        DataFrame with columns ``first_frame``, ``last_frame``, ``density``
        (in ``1/m^2``), and ``flow`` (in ``1/s``), plus ``mean_speed`` (in
        ``m/s``) when ``individual_speed`` is given.
    """
    classic_density = compute_classic_density(
        traj_data=traj_data,
        measurement_area=measurement_area,
    )
    nt, crossing_frames = compute_n_t(
        traj_data=traj_data,
        measurement_line=measurement_line,
    )
    crossing_speeds: Optional[pd.DataFrame] = None
    if individual_speed is not None:
        crossing_speeds = crossing_frames.merge(individual_speed, on=[ID_COL, FRAME_COL])

    base_cols = [FIRST_FRAME_COL, LAST_FRAME_COL, DENSITY_COL, FLOW_COL]
    if individual_speed is not None:
        base_cols.append(MEAN_SPEED_COL)

    if nt[nt[CUMULATED_COL] > 0].empty:
        return pd.DataFrame(columns=base_cols)

    num_passed_before = 0
    passed_frame_before = int(nt[nt[CUMULATED_COL] > 0].frame.min())
    rows: list[dict[str, Any]] = []

    for frame in range(passed_frame_before + delta_frame, int(nt.frame.max()), delta_frame):
        passed_num_peds = int(nt[CUMULATED_COL][nt[FRAME_COL] == frame].to_numpy()[0])
        passed_frame = int(nt[nt[CUMULATED_COL] == passed_num_peds].frame.min()) + 1

        if passed_num_peds != num_passed_before:
            num_passing_peds = passed_num_peds - num_passed_before
            time_range = passed_frame - passed_frame_before
            flow_rate = num_passing_peds / time_range * traj_data.frame_rate
            mean_density = float(
                classic_density[
                    classic_density[FRAME_COL].between(passed_frame_before, passed_frame, inclusive="both")
                ][DENSITY_COL].mean()
            )
            row: dict[str, Any] = {
                FIRST_FRAME_COL: passed_frame_before,
                LAST_FRAME_COL: passed_frame,
                DENSITY_COL: mean_density,
                FLOW_COL: flow_rate,
            }
            if crossing_speeds is not None:
                row[MEAN_SPEED_COL] = crossing_speeds[
                    crossing_speeds.frame.between(passed_frame_before, passed_frame, inclusive="both")
                ][SPEED_COL].mean()

            num_passed_before = passed_num_peds
            passed_frame_before = passed_frame
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_cols)
    return pd.DataFrame(rows)


def _validate_inputs(
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    individual_speed: pd.DataFrame,
    species: pd.DataFrame,
) -> None:
    """Validate input data for consistency and correctness before computation.

    This function performs checks on the input data to ensure that:

    1. **Species Data Consistency:**
       Verifies that the `species` DataFrame aligns with the
       `individual_voronoi_polygons` and the `measurement_line`.
       This ensures that species data is correctly associated with
       the individual Voronoi polygons and the measurement setup.

    2. **Individual Speed Data Integrity:**
       Validates the `individual_speed` DataFrame by checking for:
         - Missing entries in trajectory data.
         - Required velocity columns to ensure proper speed calculation.
         - General data completeness necessary for line speed computation.

    Args:
        individual_voronoi_polygons (pd.DataFrame):
            DataFrame containing Voronoi polygons for each individual.

        measurement_line (MeasurementLine):
            Object representing the measurement line used in calculations.

        individual_speed (pd.DataFrame):
            DataFrame containing individual speed data, including velocity
            components.

        species (pd.DataFrame):
            DataFrame mapping individuals to their respective species or
            categories.
    """
    if not is_species_valid(
        species=species,
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
    ):
        raise InputError(
            "Species data validation failed. Ensure species data matches "
            "the Voronoi data and measurement line used for calculation."
        )

    speed_status = is_individual_speed_valid(
        individual_speed=individual_speed,
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
    )
    error_messages = {
        DataValidationStatus.ENTRY_MISSING: """
        Missing speed data entries. Check for gaps in trajectory data.""",
        DataValidationStatus.COLUMN_MISSING: """
        Required velocity columns missing.
        Ensure speed was calculated with compute_velocity option.""",
        DataValidationStatus.DATA_CORRECT: None,
    }
    if speed_status != DataValidationStatus.DATA_CORRECT:
        error_msg = error_messages.get(
            speed_status,
            "Individual speed doesn't contain all data required to calculate the line speed.",
        )
        raise InputError(error_msg)


def compute_line_flow(
    *,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    individual_speed: pd.DataFrame,
    species: pd.DataFrame,
) -> pd.DataFrame:
    r"""Calculates flow for both species and total flow orthogonal to line.

    The flow of each frame is accumulated from

    .. math::
           v_{i} \cdot n_{l} \cdot \frac{1}{A_i(t)} \cdot  \frac{w_i(t)}{w},

    for each pedestrian :math:`i` whose Voronoi cell intersects the line.

    Here:

    * :math:`v_{i} \cdot n_{l}` is the speed of pedestrian :math:`i` orthogonal
      to the line :math:`l`.
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

        measurement_line (MeasurementLine): line at which the flow is calculated

        individual_speed (pd.DataFrame): individual speed data per frame, result
            from :func:`~methods.speed_calculator.compute_individual_speed`
            using :code:`compute_velocity`

        species (pd.DataFrame): dataframe containing information about the
            species of every pedestrian intersecting the line,
            result from :func:`~speed_calculator.compute_species`
    Returns:
        Dataframe containing columns 'frame', 'j_sp+1' which contains the
        specific flow in :math:`1/(m \cdot s)` for species +1, 'j_sp-1' which
        contains the specific flow in :math:`1/(m \cdot s)` for species -1, and
        'flow' which contains the total specific flow at the line in
        :math:`1/(m \cdot s)`.
    """
    _validate_inputs(individual_voronoi_polygons, measurement_line, individual_speed, species)
    result = _apply_lambda_for_intersecting_frames(
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
        species=species,
        lambda_for_group=lambda group, line: (
            _compute_orthogonal_speed_in_relation_to_proportion(group, line) * group[DENSITY_COL]
        ).sum(),
        column_id_sp1=FLOW_SP1_COL,
        column_id_sp2=FLOW_SP2_COL,
        individual_speed=individual_speed,
    )
    result[FLOW_SP2_COL] *= -1
    result[FLOW_COL] = result[FLOW_SP1_COL].fillna(0) + result[FLOW_SP2_COL].fillna(0)
    return result
