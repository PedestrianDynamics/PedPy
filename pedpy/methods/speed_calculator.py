"""Module containing functions to compute velocities."""
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas
import shapely

from pedpy.column_identifier import (
    FRAME_COL,
    ID_COL,
    SPEED_COL,
    V_X_COL,
    V_Y_COL,
)
from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    SpeedCalculation,
    _compute_individual_movement,
)


class SpeedError(Exception):
    """Class reflecting errors when computing speeds with PedPy."""

    def __init__(self, message):
        """Create SpeedError with the given message.

        Args:
            message: Error message
        """
        self.message = message


def compute_individual_speed(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_velocity: bool = False,
    speed_calculation: SpeedCalculation = SpeedCalculation.BORDER_EXCLUDE,
) -> pandas.DataFrame:
    r"""Compute the individual speed for each pedestrian.

    For computing the individuals speed at a specific frame :math:`v_i(t)`,
    a specific frame step (:math:`n`) is needed.
    Together with the
    :attr:`~trajectory_data.TrajectoryData.frame_rate` of
    the trajectory data :math:`fps` the time frame :math:`\Delta t` for
    computing the speed becomes:

    .. math::

        \Delta t = 2 n / fps

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
        \bar{X} = X_{future} - X_{past}


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

        v_i(t) = {|{X_{future} - X_{current}|}\over{ \frac{1}{2} \Delta t}}
        \text{, or }
        v_i(t) = {|{X_{current} - X_{past}|}\over{ \frac{1}{2} \Delta t}}

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

        v_i(t) = {{|\boldsymbol{proj}_d\; \bar{X}|} \over {\Delta t}}

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
    individual_speed: pandas.DataFrame,
    measurement_area: MeasurementArea,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    r"""Compute mean speed per frame inside a given measurement area.

    Computes the mean speed :math:`v_{mean}(t)` inside the measurement area from
    the given individual speed data :math:`v_i(t)` (see
    :func:`~speed_calculator.compute_individual_speed` for
    details of the computation). The mean speed :math:`v_{mean}` is defined as

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
    if len(individual_speed.index) < len(traj_data.data.index):
        raise SpeedError(
            f"Can not compute the mean speed, as the there are less speed "
            f"data (rows={len(individual_speed)}) than trajectory data "
            f"(rows={len(traj_data.data.index)}). This means a person occupies "
            f"space but has no speed at some frames."
            f"To resolve this either edit your trajectory data, s.th. it only "
            f"contains the data that is also contained in the speed data. Or "
            f"use a different speed border method when computing the individual "
            f"speed."
        )

    combined = traj_data.data.merge(individual_speed, on=[ID_COL, FRAME_COL])
    df_mean = (
        combined[shapely.within(combined.point, measurement_area.polygon)]
        .groupby(by=FRAME_COL)
        .speed.mean()
    )
    df_mean = df_mean.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_mean


def compute_voronoi_speed(
    *,
    traj_data: TrajectoryData,
    individual_speed: pandas.DataFrame,
    individual_voronoi_intersection: pandas.DataFrame,
    measurement_area: MeasurementArea,
) -> pandas.DataFrame:
    r"""Compute the Voronoi speed per frame inside the measurement area.

    Computes the Voronoi speed :math:`v_{voronoi}(t)` inside the measurement
    area :math:`M` from the given individual speed data :math:`v_i(t)` (see
    :func:`~speed_calculator.compute_individual_speed` for
    details of the computation) and their individual Voronoi intersection data
    (from :func:`~density_calculator.compute_voronoi_density`).
    The individuals speed are weighted by the proportion of their Voronoi cell
    :math:`V_i` and the intersection with the measurement area
    :math:`V_i \cap M`.

    The Voronoi speed :math:`v_{voronoi}(t)` is defined as

    .. math::

        v_{voronoi}(t) = { \int\int v_{xy}(t) dxdy \over A(M)},

    where :math:`v_{xy}(t) = v_i(t)` is the individual speed of
    each pedestrian, whose :math:`V_i(t) \cap M` and :math:`A(M)` the area of
    the measurement area.

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
    if len(individual_speed.index) < len(individual_voronoi_intersection.index):
        raise SpeedError(
            f"Can not compute the Voronoi speed, as the there are less speed "
            f"data (rows={len(individual_speed)}) than Voronoi intersection "
            f"data (rows={len(individual_voronoi_intersection.index)}). This "
            f"means a person occupies space but has no speed at some frames."
            f"To resolve this either edit your Voronoi intersection data, s.th. "
            f"it only contains the data that is also contained in the speed "
            f"data. Or use a different speed border method when computing the "
            f"individual speed."
        )

    df_voronoi = pandas.merge(
        individual_voronoi_intersection,
        individual_speed,
        on=[ID_COL, FRAME_COL],
    )
    df_voronoi[SPEED_COL] = (
        shapely.area(df_voronoi.intersection)
        * df_voronoi.speed
        / measurement_area.area
    )
    df_voronoi_speed = df_voronoi.groupby(by=df_voronoi.frame).speed.sum()
    df_voronoi_speed = df_voronoi_speed.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    return pandas.DataFrame(df_voronoi_speed)


def compute_passing_speed(
    *, frames_in_area: pandas.DataFrame, frame_rate: float, distance: float
) -> pandas.DataFrame:
    r"""Compute the individual speed of the pedestrian who pass the area.

    Compute the individual speed :math:`v^i_{passing}` at which the pedestrian
    traveled the given distance :math:`d`, which is defined as:

    .. math::

        v^i_{passing} = {{d} \over{ \Delta t}},

    where :math:`\Delta t = (f_{out} - f_{in}) / fps` is the time the
    pedestrian needed to cross the area, where :math:`f_{in}` and
    :math:`f_{out}` are the frames where the pedestrian crossed the first line,
    and the second line respectively. For details on the computation of the
    crossing frames, see :func:`~method_utils.compute_frame_range_in_area`.
    And :math:`fps` is the :attr:`~trajectory_data.TrajectoryData.frame_rate`
    of the trajectory data.

    Args:
        frames_in_area (pandas.DataFrame): information for each pedestrian when
            they were in the area, result from
            :func:`~method_utils.compute_frame_range_in_area`
        frame_rate (float): frame rate of the trajectory
        distance (float): distance between the two measurement lines

    Returns:
        DataFrame containing the columns 'id' and 'speed' in m/s
    """
    speed = pandas.DataFrame(frames_in_area.id, columns=[ID_COL, SPEED_COL])
    speed[SPEED_COL] = (
        frame_rate
        * distance
        / (np.abs(frames_in_area.leaving_frame - frames_in_area.entering_frame))
    )
    return speed


def _compute_individual_speed(
    *,
    movement_data: pandas.DataFrame,
    frame_rate: float,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_velocity: bool = True,
) -> pandas.DataFrame:
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
    movement_data[["d_x", "d_y"]] = shapely.get_coordinates(
        movement_data.end_position
    ) - shapely.get_coordinates(movement_data.start_position)

    movement_data[SPEED_COL] = (
        np.linalg.norm(movement_data[["d_x", "d_y"]], axis=1) / time_interval
    )

    if movement_direction is not None:
        # Projection of the displacement onto the movement direction
        norm_movement_direction = np.dot(movement_direction, movement_direction)
        movement_data[["d_x", "d_y"]] = (
            np.dot(movement_data[["d_x", "d_y"]].values, movement_direction)[
                :, None
            ]
            * movement_direction
            * norm_movement_direction
        )
        movement_data[SPEED_COL] = (
            np.dot(movement_data[["d_x", "d_y"]].values, movement_direction)
            / np.linalg.norm(movement_direction)
            / time_interval
        )

    if compute_velocity:
        movement_data[V_X_COL] = movement_data["d_x"].values / time_interval
        movement_data[V_Y_COL] = movement_data["d_y"].values / time_interval
        columns.append(V_X_COL)
        columns.append(V_Y_COL)

    return movement_data[columns]
