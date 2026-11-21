"""Module containing functions to compute velocities."""

from typing import Final, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from pedpy.column_identifier import (
    ALIGNMENT_COL,
    FRAME_COL,
    ID_COL,
    INTERSECTION_COL,
    POINT_COL,
    POLYGON_COL,
    SPECIES_COL,
    SPEED_COL,
    SPEED_SP1_COL,
    SPEED_SP2_COL,
    V_X_COL,
    V_Y_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import InputError, PedPyValueError, SpeedError
from pedpy.methods.method_utils import (
    DataValidationStatus,
    SpeedCalculation,
    _apply_lambda_for_intersecting_frames,
    _check_trajectory_data,
    _compute_individual_movement,
    _compute_orthogonal_speed_in_relation_to_proportion,
    is_individual_speed_valid,
    is_species_valid,
)

_INTERSECTION_AREA_THRESHOLD: Final = 1e-16
_VECTOR_NORM_THRESHOLD: Final = 1e-12


def compute_individual_speed(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_velocity: bool = False,
    speed_calculation: SpeedCalculation = SpeedCalculation.BORDER_EXCLUDE,
) -> pd.DataFrame:
    r"""Compute the individual speed for each pedestrian.

    For computing the individuals speed at a specific frame :math:`v_i(t)`,
    a specific frame step (:math:`n`) is needed.
    Together with the
    :attr:`~trajectory_data.TrajectoryData.frame_rate` of
    the trajectory data :math:`fps` the time frame :math:`\Delta t` for
    computing the speed becomes:

    .. math::

        \Delta t = 2 n / fps.

    This time step describes how many frames before and after the current
    position :math:`X_{current}` are used to compute the movement.
    These positions are called :math:`X_{future}`, :math:`X_{past}`
    respectively.

    |

    .. image:: /images/speed_both.svg
        :width: 80 %
        :align: center

    |

    First computing the displacement between these positions :math:`\bar{X}`.
    This then can be used to compute the speed with:

    .. math::

        \bar{X} = X_{future} - X_{past}.


    When getting closer to the start, or end of the trajectory data, it is not
    possible to use the full range of the frame interval for computing the
    speed. For these cases *PedPy* offers three different methods to compute
    the speed:

    #. exclude these parts.
    #. adaptively shrink the window in which the speed is computed.
    #. switch to one-sided window.

    **Exclude border:**

    When not enough frames available to compute the speed at the borders, for
    these parts no speed can be computed and they are ignored. Use
    :code:`speed_calculation=SpeedCalculation.BORDER_EXCLUDE`.

    **Adaptive border window:**

    In the adaptive approach, it is checked how many frames :math:`n` are
    available to from :math:`X_{current}` to the end of the trajectory. This
    number is then used on both sides to create a smaller symmetric window,
    which yields :math:`X_{past}` and :math:`X_{future}`. Now with the same
    principles as before the individual speed :math:`v_i(t)` can be computed.

    .. image:: /images/speed_border_adaptive_future.svg
        :width: 46 %

    .. image:: /images/speed_border_adaptive_past.svg
        :width: 46 %

    Use :code:`speed_calculation=SpeedCalculation.BORDER_ADAPTIVE`.

    .. important::

        As the time interval gets smaller to the ends of the individual
        trajectories, the oscillations in the speed increase here.


    **Single sided border window:**

    In these cases, one of the end points to compute the movement becomes the
    current position :math:`X_{current}`. When getting too close to the start
    of the trajectory, the movement is computed from :math:`X_{current}` to
    :math:`X_{future}`. In the other case the movement is from :math:`X_{past}`
    to :math:`X_{current}`.

    .. math::

        v_i(t) = {|{X_{future} - X_{current}|}\over{ \frac{1}{2} \Delta t}},
        \text{, or }
        v_i(t) = {|{X_{current} - X_{past}|}\over{ \frac{1}{2} \Delta t}}.

    .. image:: /images/speed_border_single_sided_future.svg
        :width: 46 %

    .. image:: /images/speed_border_single_sided_past.svg
        :width: 46 %

    |

    Use :code:`speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED`.

    .. important::

        As at the edges of the trajectories the time interval gets halved,
        there may occur some jumps computed speeds at this point.


    **With movement direction:**

    It is also possible to compute the individual speed in a specific direction
    :math:`d`, for this the movement :math:`\bar{X}` is projected onto the
    desired movement direction. :math:`\bar{X}` and :math:`\Delta t` are
    computed as described above. Hence, the speed then becomes:

    .. math::

        v_i(t) = {{|\boldsymbol{proj}_d\; \bar{X}|} \over {\Delta t}}.

    |

    .. image:: /images/speed_movement_direction.svg
        :width: 80 %
        :align: center

    |

    .. important::

        Using a movement direction may lead to negative speeds!

    If :code:`compute_velocity` is `True` also :math:`\bar{X}` is returned.

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_step (int): gives the size of time interval for calculating the
            velocity.
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        compute_velocity (bool): compute the x and y components of the velocity
        speed_calculation (method_utils.SpeedCalculation): method used to
            compute the speed at the borders of the individual trajectories

    Returns:
        DataFrame containing the columns 'id', 'frame', and 'speed' in m/s,
        'v_x' and 'v_y' with the speed components in x and y direction if
        :code:`compute_velocity` is True
    """
    _check_trajectory_data(traj_data)
    df_movement = _compute_individual_movement(
        traj_data=traj_data,
        frame_step=frame_step,
        speed_border_method=speed_calculation,
    )
    df_speed = _compute_individual_speed(
        movement_data=df_movement,
        frame_rate=traj_data.frame_rate,
        movement_direction=movement_direction,
        compute_velocity=compute_velocity,
    )

    return df_speed


def compute_mean_speed_per_frame(
    *,
    traj_data: TrajectoryData,
    individual_speed: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    r"""Compute mean speed per frame inside a given measurement area.

    Computes the mean speed :math:`v_{mean}(t)` inside the measurement area
    from the given individual speed data :math:`v_i(t)` (see
    :func:`~speed_calculator.compute_individual_speed` for
    details of the computation). The mean speed :math:`v_{mean}` is defined
    as

    .. math::

        v_{mean}(t) = {{1} \over {N}} \sum_{i \in P_M} v_i(t),

    where :math:`P_M` are all pedestrians inside the measurement area, and
    :math:`N` the number of pedestrians inside the measurement area (
    :math:`|P_M|`).

    .. image:: /images/classic_density.svg
        :width: 60 %
        :align: center

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_speed (pandas.DataFrame): individual speed data from
            :func:`~speed_calculator.compute_individual_speed`
        measurement_area (MeasurementArea): measurement area for which the
            speed is computed

    Returns:
        DataFrame containing the columns 'frame' and 'speed' in m/s
    """
    _check_trajectory_data(traj_data)
    if len(individual_speed.index) < len(traj_data.data.index):
        raise SpeedError(
            f"Can not compute the mean speed, as the there are less speed "
            f"data (rows={len(individual_speed)}) than trajectory data "
            f"(rows={len(traj_data.data.index)}). This means a person "
            f"occupies space but has no speed at some frames."
            f"To resolve this either edit your trajectory data, s.th. it only "
            f"contains the data that is also contained in the speed data. Or "
            f"use a different speed border method when computing the "
            f"individual speed."
        )

    combined = traj_data.data.merge(individual_speed, on=[ID_COL, FRAME_COL])
    df_mean = combined[shapely.within(combined.point, measurement_area.polygon)].groupby(by=FRAME_COL).speed.mean()
    df_mean = df_mean.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    df_mean = df_mean.reset_index()
    df_mean.columns = [FRAME_COL, SPEED_COL]

    return df_mean


def compute_voronoi_speed(
    *,
    traj_data: TrajectoryData,
    individual_speed: pd.DataFrame,
    individual_voronoi_intersection: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    r"""Compute the Voronoi speed per frame inside the measurement area.

    This function calculates the Voronoi speed, :math:`v_{voronoi}(t)`, within
    the measurement area :math:`M`. It uses the individual speed data,
    :math:`v_i(t)`
    (computed via :func:`~speed_calculator.compute_individual_speed`), and the
    Voronoi intersection data
    (obtained from :func:`~density_calculator.compute_voronoi_density`).

    The individual speeds are weighted by the proportion of their Voronoi cell,
    :math:`V_i`, that intersects with the measurement area, :math:`V_i \cap M`.

    The Voronoi speed, :math:`v_{voronoi}(t)`, is defined as:

    .. math::

        v_{voronoi}(t) = \frac{\int\int v_{xy}(t) \, dx \, dy}{A(M)},

    where:

    * :math:`v_{xy}(t) = v_i(t)` represents the individual speed of each
      pedestrian whose Voronoi cell intersects with the measurement area,
    * :math:`V_i(t) \cap M` is the overlapping region between a pedestrian's
      Voronoi cell and the measurement area,
    * and :math:`A(M)` is the area of the measurement region.

    .. image:: /images/voronoi_density.svg
        :width: 60 %
        :align: center

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_speed (pandas.DataFrame): individual speed data from
            :func:`~speed_calculator.compute_individual_speed`
        individual_voronoi_intersection (pandas.DataFrame): intersections of the
            individual with the measurement area of each pedestrian from
            :func:`~method_utils.compute_intersecting_polygons`
        measurement_area (MeasurementArea): area in which the voronoi speed
            should be computed

    Returns:
        DataFrame containing the columns 'frame' and 'speed' in m/s
    """
    _check_trajectory_data(traj_data)
    if len(individual_speed.index) < len(individual_voronoi_intersection.index):
        raise SpeedError(
            f"Can not compute the Voronoi speed, as the there are less speed "
            f"data (rows={len(individual_speed)}) than Voronoi intersection "
            f"data (rows={len(individual_voronoi_intersection.index)}). This "
            f"means a person occupies space but has no speed at some frames."
            f"To resolve this either edit your Voronoi intersection data, "
            f"s.th. it only contains the data that is also contained in the "
            f"speed data. Or use a different speed border method when "
            f"computing the individual speed."
        )

    df_voronoi = individual_voronoi_intersection.merge(
        individual_speed,
        on=[ID_COL, FRAME_COL],
    )
    df_voronoi[SPEED_COL] = shapely.area(df_voronoi.intersection) * df_voronoi.speed / measurement_area.area
    df_voronoi_speed = df_voronoi.groupby(by=df_voronoi.frame).speed.sum()
    df_voronoi_speed = df_voronoi_speed.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    df_voronoi_speed = df_voronoi_speed.reset_index()
    df_voronoi_speed.columns = [FRAME_COL, SPEED_COL]
    return pd.DataFrame(df_voronoi_speed)


def compute_arithmetic_voronoi_speed(
    *,
    traj_data: TrajectoryData,
    individual_speed: pd.DataFrame,
    individual_voronoi_intersection: pd.DataFrame,
    measurement_area: Optional[MeasurementArea] = None,
) -> pd.DataFrame:
    r"""Compute the arithmetic Voronoi speed per frame in a measurement area.

    Calculates the unweighted arithmetic mean speed :math:`v_{arithmetic}(t)`
    over all pedestrians :math:`P_M(t)` whose Voronoi cell :math:`V_i(t)`
    intersects the measurement area :math:`M` with positive area
    (:math:`A(V_i(t) \cap M) > 0`):

    .. math::

        v_{arithmetic}(t) = \frac{1}{|P_M(t)|} \sum_{i \in P_M(t)} v_i(t).

    Frames in which no Voronoi cell intersects the measurement area receive a
    speed of ``0.0``.

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_speed (pandas.DataFrame): individual speed data from
            :func:`~speed_calculator.compute_individual_speed`
        individual_voronoi_intersection (pandas.DataFrame): intersections of
            individual Voronoi cells with the measurement area from
            :func:`~method_utils.compute_intersecting_polygons` (or Voronoi
            polygon data when ``measurement_area`` is also passed)
        measurement_area (MeasurementArea, optional): measurement area used to
            compute intersections when ``individual_voronoi_intersection``
            contains ``polygon`` instead of ``intersection``

    Returns:
        DataFrame containing the columns ``frame`` and ``speed`` in m/s
    """
    _check_trajectory_data(traj_data)
    if len(individual_speed.index) < len(individual_voronoi_intersection.index):
        raise SpeedError(
            f"Can not compute the arithmetic Voronoi speed, as there are less speed "
            f"data (rows={len(individual_speed)}) than Voronoi intersection "
            f"data (rows={len(individual_voronoi_intersection.index)}). This "
            f"means a person occupies space but has no speed at some frames. "
            f"To resolve this either edit your Voronoi intersection data, "
            f"s.th. it only contains the data that is also contained in the "
            f"speed data. Or use a different speed border method when "
            f"computing the individual speed."
        )

    df_voronoi = individual_voronoi_intersection.merge(
        individual_speed,
        on=[ID_COL, FRAME_COL],
    )

    if INTERSECTION_COL in df_voronoi.columns:
        intersections = df_voronoi[INTERSECTION_COL]
    elif POLYGON_COL in df_voronoi.columns and measurement_area is not None:
        intersections = shapely.intersection(df_voronoi[POLYGON_COL], measurement_area.polygon)
    else:
        raise InputError(
            f"Expected column '{INTERSECTION_COL}' in 'individual_voronoi_intersection' "
            f"(or '{POLYGON_COL}' together with 'measurement_area')."
        )

    intersecting = df_voronoi[shapely.area(intersections) > _INTERSECTION_AREA_THRESHOLD]
    df_arithmetic_speed = intersecting.groupby(by=FRAME_COL)[SPEED_COL].mean()
    df_arithmetic_speed = df_arithmetic_speed.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    df_arithmetic_speed = df_arithmetic_speed.reset_index()
    df_arithmetic_speed.columns = [FRAME_COL, SPEED_COL]
    return pd.DataFrame(df_arithmetic_speed)


def _resolve_desired_unit_vectors(
    combined: pd.DataFrame,
    target_point: Optional[npt.ArrayLike],
    desired_direction: Optional[Union[npt.ArrayLike, pd.DataFrame]],
) -> npt.NDArray[np.float64]:
    """Compute per-row unit desired direction vectors ``(u_x, u_y)``."""
    if (target_point is None) == (desired_direction is None):
        raise PedPyValueError("Provide exactly one of 'target_point' or 'desired_direction'.")

    if target_point is not None:
        target = np.asarray(target_point, dtype=np.float64)
        if target.shape != (2,):
            raise PedPyValueError(f"'target_point' must have shape (2,), got {target.shape}.")
        raw_u = target[None, :] - combined[[X_COL, Y_COL]].to_numpy(dtype=np.float64)
    elif isinstance(desired_direction, pd.DataFrame):
        if not {"u_x", "u_y"}.issubset(desired_direction.columns):
            raise PedPyValueError("DataFrame 'desired_direction' must contain 'u_x' and 'u_y' columns.")
        join_cols = [col for col in (ID_COL, FRAME_COL) if col in desired_direction.columns]
        if not join_cols:
            raise PedPyValueError(
                f"DataFrame 'desired_direction' must contain '{ID_COL}' (and optionally '{FRAME_COL}')."
            )
        aligned = combined[[ID_COL, FRAME_COL]].merge(
            desired_direction[[*join_cols, "u_x", "u_y"]],
            on=join_cols,
            how="left",
        )
        raw_u = aligned[["u_x", "u_y"]].to_numpy(dtype=np.float64)
    else:
        vec = np.asarray(desired_direction, dtype=np.float64)
        if vec.shape != (2,):
            raise PedPyValueError(f"'desired_direction' must have shape (2,), got {vec.shape}.")
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm <= _VECTOR_NORM_THRESHOLD:
            raise PedPyValueError("'desired_direction' must be a non-zero 2D vector.")
        return np.broadcast_to(vec / vec_norm, (len(combined), 2)).copy()

    norms = np.linalg.norm(raw_u, axis=1, keepdims=True)
    return np.divide(raw_u, norms, out=np.zeros_like(raw_u), where=norms > _VECTOR_NORM_THRESHOLD)


def compute_alignment_parameter(
    *,
    traj_data: TrajectoryData,
    individual_speed: pd.DataFrame,
    target_point: Optional[npt.ArrayLike] = None,
    desired_direction: Optional[Union[npt.ArrayLike, pd.DataFrame]] = None,
    measurement_area: Optional[MeasurementArea] = None,
) -> pd.DataFrame:
    r"""Compute the per-frame alignment parameter :math:`\phi_d(t)`.

    Quantifies how closely pedestrian velocities align with their desired
    direction of motion (e.g. toward an exit door or along a corridor),
    following Feliciani et al. (2020), *Scientific Reports*, 10, 15947
    (``doi:10.1038/s41598-020-72733-w``):

    .. math::

        \phi_d(t) = \frac{1}{N(t)} \sum_{n=1}^{N(t)}
        \frac{\vec{v}_t^n \cdot \vec{u}_t^n}{\|\vec{v}_t^n\|},

    where :math:`\vec{v}_t^n` is the velocity vector of pedestrian :math:`n` at
    frame :math:`t`, :math:`\vec{u}_t^n` is their unit desired direction of
    motion, and :math:`N(t)` is the number of pedestrians considered at frame
    :math:`t`. Values near ``+1`` indicate organized movement toward the
    desired target/direction, ``0`` indicates perpendicular/azimuthal motion
    (or stationary pedestrians), and ``-1`` indicates motion in the opposite
    direction.

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_speed (pandas.DataFrame): individual velocity data from
            :func:`~speed_calculator.compute_individual_speed` computed with
            ``compute_velocity=True``
        target_point (array-like, optional): 2D coordinates ``(x, y)`` of a
            target point (such as an exit center) toward which each
            pedestrian's desired unit vector points. Mutually exclusive with
            ``desired_direction``.
        desired_direction (array-like or pandas.DataFrame, optional): either a
            constant 2D direction vector ``(u_x, u_y)`` or a DataFrame mapping
            ``id`` (and optionally ``frame``) to ``u_x`` and ``u_y``. Mutually
            exclusive with ``target_point``.
        measurement_area (MeasurementArea, optional): if provided, restricts
            the evaluation to pedestrians inside ``measurement_area``

    Returns:
        DataFrame containing the columns ``frame`` and ``alignment``
    """
    _check_trajectory_data(traj_data)
    if not {ID_COL, FRAME_COL, V_X_COL, V_Y_COL}.issubset(individual_speed.columns):
        raise SpeedError(
            "Required velocity columns ('v_x', 'v_y') are missing from 'individual_speed'. "
            "Compute individual speed with `compute_velocity=True`."
        )

    if measurement_area is not None and len(individual_speed.index) < len(traj_data.data.index):
        raise SpeedError(
            f"Can not compute the alignment parameter inside a measurement area, as "
            f"there are less speed data (rows={len(individual_speed)}) than trajectory "
            f"data (rows={len(traj_data.data.index)})."
        )

    combined = traj_data.data.merge(
        individual_speed[[ID_COL, FRAME_COL, V_X_COL, V_Y_COL]],
        on=[ID_COL, FRAME_COL],
    )
    if measurement_area is not None:
        combined = combined[shapely.within(combined[POINT_COL], measurement_area.polygon)].copy()

    unit_u = _resolve_desired_unit_vectors(
        combined=combined,
        target_point=target_point,
        desired_direction=desired_direction,
    )
    velocities = combined[[V_X_COL, V_Y_COL]].to_numpy(dtype=np.float64)
    v_norms = np.linalg.norm(velocities, axis=1)
    dots = np.sum(velocities * unit_u, axis=1)
    combined = combined.copy()
    combined[ALIGNMENT_COL] = np.divide(
        dots,
        v_norms,
        out=np.zeros_like(dots),
        where=v_norms > _VECTOR_NORM_THRESHOLD,
    )

    df_alignment = combined.groupby(by=FRAME_COL)[ALIGNMENT_COL].mean()
    df_alignment = df_alignment.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    df_alignment = df_alignment.reset_index()
    df_alignment.columns = [FRAME_COL, ALIGNMENT_COL]
    return pd.DataFrame(df_alignment)


def compute_passing_speed(*, frames_in_area: pd.DataFrame, frame_rate: float, distance: float) -> pd.DataFrame:
    r"""Compute the individual speed of the pedestrian who pass the area.

    The individual speed, :math:`v^i_{passing}`, is calculated as the speed at
    which a pedestrian travels a given distance :math:`d`. It is defined by
    the formula:

    .. math::
        v^i_{passing} = \frac{d}{\Delta t},

    where:

    * :math:`\Delta t = \frac{(f_{out} - f_{in})}{\text{fps}}` is the time
      required for the pedestrian to cross the area.
    * :math:`f_{in}` and :math:`f_{out}` are the frames where the pedestrian
      crossed the first and second lines, respectively,
    * and :math:`\text{fps}` is the frame rate of the trajectory data, given
      by :attr:`~trajectory_data.TrajectoryData.frame_rate`.

    For details on how the crossing frames (:math:`f_{in}` and :math:`f_{out}`)
    are computed, see :func:`~method_utils.compute_frame_range_in_area`.

    Args:
        frames_in_area (pandas.DataFrame): information for each pedestrian when
            they were in the area, result from
            :func:`~method_utils.compute_frame_range_in_area`
        frame_rate (float): frame rate of the trajectory
        distance (float): distance between the two measurement lines

    Returns:
        DataFrame containing the columns 'id' and 'speed' in m/s
    """
    speed = pd.DataFrame(frames_in_area.id, columns=[ID_COL, SPEED_COL])
    speed[SPEED_COL] = frame_rate * distance / (np.abs(frames_in_area.leaving_frame - frames_in_area.entering_frame))
    return speed


def _compute_individual_speed(
    *,
    movement_data: pd.DataFrame,
    frame_rate: float,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_velocity: bool = True,
) -> pd.DataFrame:
    """Compute the instantaneous speed of each pedestrian.

    Args:
        movement_data (pandas.DataFrame): movement data
            (see compute_individual_movement)
        frame_rate (float): frame rate of the trajectory data
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        compute_velocity (bool): compute the x and y components of the velocity

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'speed' with the
        speed in m/s, 'v_x' and 'v_y' with the speed components in x and y
        direction if compute_velocity is True
    """
    columns = [ID_COL, FRAME_COL, SPEED_COL]
    time_interval = movement_data.window_size / frame_rate

    # Compute displacements in x and y direction
    movement_data[["d_x", "d_y"]] = shapely.get_coordinates(movement_data.end_position) - shapely.get_coordinates(
        movement_data.start_position
    )

    movement_data[SPEED_COL] = np.linalg.norm(movement_data[["d_x", "d_y"]], axis=1) / time_interval

    if movement_direction is not None:
        # Projection of the displacement onto the movement direction
        norm_movement_direction = np.dot(movement_direction, movement_direction)
        movement_data[["d_x", "d_y"]] = (
            np.dot(movement_data[["d_x", "d_y"]].to_numpy(), movement_direction)[:, None]
            * movement_direction
            * norm_movement_direction
        )
        movement_data[SPEED_COL] = (
            np.dot(movement_data[["d_x", "d_y"]].to_numpy(), movement_direction)
            / np.linalg.norm(movement_direction)
            / time_interval
        )

    if compute_velocity:
        movement_data[V_X_COL] = movement_data["d_x"].to_numpy() / time_interval
        movement_data[V_Y_COL] = movement_data["d_y"].to_numpy() / time_interval
        columns.append(V_X_COL)
        columns.append(V_Y_COL)

    return movement_data[columns]


def _validate_inputs(
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    individual_speed: pd.DataFrame,
    species: pd.DataFrame,
) -> None:
    """Validate the consistency and completeness of input data.

    This function performs two validation checks to ensure that the input data
    is suitable for further processing:

    1. **Species Data Validation:**
       Confirms that the `species` DataFrame is consistent with the
       `individual_voronoi_polygons` and the `measurement_line`.
       This ensures that all species are correctly associated with individuals
       in the Voronoi data and that the measurement line is correctly defined
       for analysis.

    2. **Individual Speed Data Validation:**
       Checks the `individual_speed` DataFrame for:
         - **Missing Trajectory Entries:** Verifies that all individuals have
           corresponding speed data with no missing entries.
         - **Missing Velocity Columns:** Ensures that necessary velocity
           components are present.
           This typically requires enabling the `compute_velocity` option.
         - **Overall Data Completeness:** Confirms that all required
           data is available for accurate line speed computation.

    Args:
        individual_voronoi_polygons (pd.DataFrame):
            DataFrame containing Voronoi polygons for each individual
            in the system.

        measurement_line (MeasurementLine):
            Object defining the measurement line used for analysis
            and calculations.

        individual_speed (pd.DataFrame):
            DataFrame holding individual speed data, typically including
            velocity components (`vx`, `vy`).

        species (pd.DataFrame):
            DataFrame mapping individuals to species or categories, used for
            classification in the analysis.
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


def compute_line_speed(
    *,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    individual_speed: pd.DataFrame,
    species: pd.DataFrame,
) -> pd.DataFrame:
    r"""Calculates speed of both species and total speed orthogonal to line.

    The speed of each frame is accumulated from

    .. math::
         v_{i} \cdot n_{l} \cdot  \frac{w_i(t)}{w},

    for each pedestrian :math:`i` whose Voronoi cell intersects the line
    :math:`l`.

    Here:

    * :math:`v_{i} \cdot n_{l}` is the speed of pedestrian :math:`i` orthogonal
      to the line :math:`l`.
    * :math:`w` is the length of the measurement line.
    * :math:`w_i(t)` is the length of the intersecting line of the Voronoi cell
      in frame :math:`t`.

    Results are computed for both species
    (see :func:`~speed_calculator.compute_species`)

    Args:
        individual_voronoi_polygons (pandas.DataFrame): individual Voronoi data
            per frame, result
            from :func:`~method_utils.compute_individual_voronoi_polygons`.

        measurement_line (MeasurementLine): line at which the speed
            is calculated

        individual_speed (pandas.DataFrame): individual speed data per frame
            , result from :func:`~speed_calculator.compute_individual_speed`
            using :code:`compute_velocity`

        species (pandas.DataFrame): dataframe containing information about
            the species of every pedestrian intersecting the line,
            result from :func:`~speed_calculator.compute_species`
    Returns:
        Dataframe containing columns 'frame', 's_sp+1' which contains the
        speed in :math:`m/s` for species +1, 's_sp-1' which contains the
        speed in :math:`m/s` for species -1, 'speed' which contains the
        density at the line in :math:`1/m^2`.
    """
    _validate_inputs(individual_voronoi_polygons, measurement_line, individual_speed, species)

    result = _apply_lambda_for_intersecting_frames(
        individual_voronoi_polygons=individual_voronoi_polygons,
        measurement_line=measurement_line,
        species=species,
        lambda_for_group=lambda group, line: (_compute_orthogonal_speed_in_relation_to_proportion(group, line)).sum(),
        column_id_sp1=SPEED_SP1_COL,
        column_id_sp2=SPEED_SP2_COL,
        individual_speed=individual_speed,
    )
    result[SPEED_SP2_COL] *= -1
    result[SPEED_COL] = result[SPEED_SP1_COL].fillna(0) + result[SPEED_SP2_COL].fillna(0)
    return result


def compute_species(
    *,
    trajectory_data: TrajectoryData,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    frame_step: int,
) -> pd.DataFrame:
    r"""Creates a Dataframe containing the species for each pedestrian.

    The species indicate from which side a pedestrian
    encounters the measurement line.
    The species of a pedestrian :math:`i` is calculated
    by

    .. math::
           sign(n \cdot v_i(t_{i,l})),

    where:

    * :math:`n` the normal vector of the measurement line,
    * :math:`v_i` is the velocity  of pedestrian :math:`i`
      at the time when their Voronoi cell intersects the measurement line
      :math:`t_{i,l}` for the first time.

    Pedestrians whose Voronoi polygons never intersect the measurement line
    are excluded from the resulting DataFrame.

    .. image:: /images/species_determination.svg
        :width: 60 %
        :align: center

    This image illustrates the frame in which the decision is made regarding
    the species classification of a pedestrian. The decision is based on the
    current velocity at the first frame where the pedestrian's Voronoi cell
    intersects the measurement line.

    It is important to note that the decision does not depend on whether
    the pedestrian actually crosses the measurement line afterward.

    Args:
        trajectory_data (TrajectoryData): trajectory data

        individual_voronoi_polygons (pd.DataFrame): individual Voronoi data per
            frame, result
            from :func:`~method_utils.compute_individual_voronoi_polygons`

        measurement_line (MeasurementLine): measurement line

        frame_step (int): gives the size of time interval for calculating the
            velocity.

    Returns:
        Dataframe containing columns 'id' and 'species'
    """
    _check_trajectory_data(trajectory_data, "trajectory_data")
    # create dataframe with id and first frame
    # where Voronoi polygon intersects measurement line
    intersecting_polys = individual_voronoi_polygons[
        shapely.intersects(individual_voronoi_polygons[POLYGON_COL], measurement_line.line)
    ]
    first_frames = intersecting_polys.groupby(ID_COL)[FRAME_COL].min().reset_index()

    normal_vector = measurement_line.normal_vector()

    initial_speed = compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=frame_step,
        compute_velocity=True,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
        movement_direction=normal_vector,
    )
    # create dataframe with 'id' and 'species'
    result = first_frames.merge(initial_speed, left_on=[ID_COL, FRAME_COL], right_on=[ID_COL, FRAME_COL])
    result[SPECIES_COL] = np.sign(result[SPEED_COL])
    return result[[ID_COL, SPECIES_COL]]
