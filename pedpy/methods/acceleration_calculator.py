"""Module containing functions to compute accelerations."""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from pedpy.column_identifier import ACC_COL, A_X_COL, A_Y_COL, FRAME_COL, ID_COL
from pedpy.data.geometry import MeasurementArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import AccelerationError
from pedpy.methods.method_utils import (
    AccelerationCalculation,
    _compute_individual_movement_acceleration,
)


def compute_individual_acceleration(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_acceleration_components: bool = False,
    acceleration_calculation: AccelerationCalculation = (
        AccelerationCalculation.BORDER_EXCLUDE
    ),
) -> pd.DataFrame:
    r"""Compute the individual acceleration for each pedestrian.

    For computing the individuals' acceleration at a specific frame
    :math:`a_i(t_k)`, a specific frame step (:math:`n`) is needed.
    Together with the
    :attr:`~trajectory_data.TrajectoryData.frame_rate` of the trajectory data
    :math:`fps` the time frame :math:`\Delta t` for computing the speed
    becomes:

    .. math::

        \Delta t = 2 n / fps

    This time step describes how many frames before and after the current
    position :math:`X(t_k)` are used to compute the movement.
    These positions are called :math:`X(t_{k+n})`, :math:`X(t_{k-n})`
    respectively.

    In order to compute the acceleration at time 't_k', we first calculate the
    displacements :math:`\bar{X}` around 't_{k+n}' and 't_{k-n}':

    .. math::

        \bar{X}(t_{k+n}) = X(t_{k+2n}) - X(t_{k})

    .. math::

        \bar{X}(t_{k-n}) = X(t_{k}) - X(t_{k-2n})

    The acceleration is then calculated from the difference of the
    displacements

    .. math::

        \Delta\bar{X}(t_k) = \bar{X}(t_{k+n}) - \bar{X}(t_{k-n})

    divided by the square of the time interval '\Delta t':

    .. math::

        a_i(t_k) = \Delta\bar{X}(t_k) / \Delta t^{2}


    When getting closer to the start, or end of the trajectory data, it is not
    possible to use the full range of the frame interval for computing the
    acceleration. For these cases *PedPy* offers a method to compute
    the acceleration:

    **Exclude border:**

    When not enough frames available to compute the speed at the borders, for
    these parts no acceleration can be computed and they are ignored. Use
    :code:`acceleration_calculation=AccelerationCalculation.BORDER_EXCLUDE`.

    **With movement direction:**

    It is also possible to compute the individual acceleration in a specific
    direction :math:`d`, for this the movement :math:`\Delta\bar{X}` is
    projected onto the desired movement direction. :math:`\Delta\bar{X}` and
    :math:`\Delta t` are computed as described above. Hence, the acceleration
    then becomes:

    .. math::

        a_i(t) = {{|\boldsymbol{proj}_d\; \Delta\bar{X}|} \over {\Delta t^{2}}}


    If :code:`compute_acceleration_components` is `True` also
    :math:`\Delta\bar{X}` is returned.

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_step (int): gives the size of time interval for calculating the
            acceleration.
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        compute_acceleration_components (bool): compute the x and y components
            of the acceleration
        acceleration_calculation (method_utils.AccelerationCalculation): method
            used to compute the acceleration at the borders of the individual
            trajectories

    Returns:
        DataFrame containing the columns 'id', 'frame', and 'acceleration' in
        :math:`m/s^2`, 'a_x' and 'a_y' with the acceleration components
        in x and y direction if :code:`compute_acceleration_components` is True
    """
    df_movement = _compute_individual_movement_acceleration(
        traj_data=traj_data,
        frame_step=frame_step,
        acceleration_border_method=acceleration_calculation,
    )

    df_acceleration = _compute_individual_acceleration(
        movement_data=df_movement,
        frame_rate=traj_data.frame_rate,
        movement_direction=movement_direction,
        compute_acceleration_components=compute_acceleration_components,
    )

    return df_acceleration


def compute_mean_acceleration_per_frame(
    *,
    traj_data: TrajectoryData,
    individual_acceleration: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    r"""Compute mean acceleration per frame inside a given measurement area.

    Computes the mean acceleration :math:`a_{mean}(t)` inside the measurement
    area from the given individual acceleration data :math:`a_i(t)` (see
    :func:`~acceleration_calculator.compute_individual_acceleration` for
    details of the computation). The mean acceleration :math:`a_{mean}` is
    defined as

    .. math::

        a_{mean}(t) = {{1} \over {N}} \sum_{i \in P_M} a_i(t),

    where :math:`P_M` are all pedestrians inside the measurement area, and
    :math:`N` the number of pedestrians inside the measurement area (
    :math:`|P_M|`).

    .. image:: /images/classic_density.svg
        :width: 60 %
        :align: center

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_acceleration (pandas.DataFrame): individual acceleration
            data from
            :func:`~acceleration_calculator.compute_individual_acceleration`
        measurement_area (MeasurementArea): measurement area for which the
            acceleration is computed

    Returns:
        DataFrame containing the columns 'frame' and 'acceleration' in
        :math:`m/s^2`
    """
    if len(individual_acceleration.index) < len(traj_data.data.index):
        raise AccelerationError(
            f"Can not compute the mean acceleration, as the there are less "
            f"acceleration data (rows={len(individual_acceleration)}) than "
            f"trajectory data (rows={len(traj_data.data.index)}). This means a "
            f"person occupies space but has no acceleration at some frames."
            f"To resolve this either edit your trajectory data, s.th. it only "
            f"contains the data that is also contained in the acceleration "
            f"data. Or use a different acceleration border method when "
            f"computing the individual acceleration."
        )

    combined = traj_data.data.merge(
        individual_acceleration, on=[ID_COL, FRAME_COL]
    )
    df_mean = (
        combined[shapely.within(combined.point, measurement_area.polygon)]
        .groupby(by=FRAME_COL)
        .acceleration.mean()
    )
    df_mean = df_mean.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_mean


def compute_voronoi_acceleration(
    *,
    traj_data: TrajectoryData,
    individual_acceleration: pd.DataFrame,
    individual_voronoi_intersection: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    r"""Computes the Voronoi acceleration.

    Computes the Voronoi acceleration :math:`a_{voronoi}(t)` inside the
    measurement area :math:`M` from the given individual acceleration data
    :math:`a_i(t)` (see
    :func:`~acceleration_calculator.compute_individual_acceleration` for
    details of the computation) and their individual Voronoi intersection data
    (from :func:`~density_calculator.compute_voronoi_density`).
    The individuals' accelerations are weighted by the proportion of their
    Voronoi cell :math:`V_i` and the intersection with the measurement area
    :math:`V_i \cap M`.

    The Voronoi acceleration :math:`a_{voronoi}(t)` is defined as

    .. math::

        a_{voronoi}(t) = { \int\int a_{xy}(t) dxdy \over A(M)},

    where :math:`a_{xy}(t) = a_i(t)` is the individual acceleration of
    each pedestrian, whose :math:`V_i(t) \cap M` and :math:`A(M)` the area of
    the measurement area.

    .. image:: /images/voronoi_density.svg
        :width: 60 %
        :align: center

    Args:
        traj_data (TrajectoryData): trajectory data
        individual_acceleration (pandas.DataFrame): individual acceleration
            data from
            :func:`~acceleration_calculator.compute_individual_acceleration`
        individual_voronoi_intersection (pandas.DataFrame): intersections of
            the individual with the measurement area of each pedestrian from
            :func:`~method_utils.compute_intersecting_polygons`
        measurement_area (MeasurementArea): area in which the voronoi
            acceleration should be computed

    Returns:
        DataFrame containing the columns 'frame' and 'acceleration' in
        :math:`m/s^2`
    """
    if len(individual_acceleration.index) < len(
        individual_voronoi_intersection.index
    ):
        raise AccelerationError(
            f"Can not compute the Voronoi acceleration, as the there are less "
            f"acceleration data (rows={len(individual_acceleration)}) than "
            f"Voronoi intersection data ("
            f"rows={len(individual_voronoi_intersection.index)}). This "
            f"means a person occupies space but has no acceleration at some "
            f"frames. To resolve this either edit your Voronoi intersection "
            f"data, s.th. it only contains the data that is also contained "
            f"in the acceleration data. Or use a different acceleration border "
            f"method when computing the individual acceleration."
        )

    df_voronoi = individual_voronoi_intersection.merge(
        individual_acceleration,
        on=[ID_COL, FRAME_COL],
    )
    df_voronoi[ACC_COL] = (
        shapely.area(df_voronoi.intersection)
        * df_voronoi.acceleration
        / measurement_area.area
    )
    df_voronoi_acceleration = df_voronoi.groupby(
        by=df_voronoi.frame
    ).acceleration.sum()
    df_voronoi_acceleration = df_voronoi_acceleration.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    return pd.DataFrame(df_voronoi_acceleration)


def _compute_individual_acceleration(
    *,
    movement_data: pd.DataFrame,
    frame_rate: float,
    movement_direction: Optional[npt.NDArray[np.float64]] = None,
    compute_acceleration_components: bool = True,
) -> pd.DataFrame:
    """Compute the instantaneous acceleration of each pedestrian.

    Args:
        movement_data (pandas.DataFrame): movement data
            (see compute_individual_movement)
        frame_rate (float): frame rate of the trajectory data
        movement_direction (np.ndarray): main movement direction on which the
            actual movement is projected (default: None, when the un-projected
            movement should be used)
        compute_acceleration_components (bool): compute the x and y components
            of the acceleration

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'acceleration' with
        the acceleration in m/s^2, 'a_x' and 'a_y' with the acceleration
        components in x and y direction if compute_acceleration is True
    """
    columns = [ID_COL, FRAME_COL, ACC_COL]
    time_interval = movement_data.window_size / frame_rate

    # Compute displacements in x and y direction
    movement_data[["dd_x", "dd_y"]] = (
        shapely.get_coordinates(movement_data.end_position)
        - shapely.get_coordinates(movement_data.mid_position)
    ) - (
        shapely.get_coordinates(movement_data.mid_position)
        - shapely.get_coordinates(movement_data.start_position)
    )

    movement_data[ACC_COL] = (
        np.linalg.norm(movement_data[["dd_x", "dd_y"]], axis=1)
        / time_interval**2
    )

    if movement_direction is not None:
        # Projection of the displacement onto the movement direction
        norm_movement_direction = np.dot(movement_direction, movement_direction)
        movement_data[["dd_x", "dd_y"]] = (
            np.dot(
                movement_data[["dd_x", "dd_y"]].to_numpy(), movement_direction
            )[:, None]
            * movement_direction
            * norm_movement_direction
        )
        movement_data[ACC_COL] = (
            np.dot(
                movement_data[["dd_x", "dd_y"]].to_numpy(), movement_direction
            )
            / np.linalg.norm(movement_direction)
            / time_interval**2
        )

    if compute_acceleration_components:
        movement_data[A_X_COL] = (
            movement_data["dd_x"].to_numpy() / time_interval**2
        )
        movement_data[A_Y_COL] = (
            movement_data["dd_y"].to_numpy() / time_interval**2
        )
        columns.append(A_X_COL)
        columns.append(A_Y_COL)

    return movement_data[columns]
