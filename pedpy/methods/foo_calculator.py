"""Foo."""


from enum import Enum

import numpy as np
import pandas
import shapely

from pedpy.column_identifier import (
    AVOIDANCE_COL,
    FRAME_COL,
    ID_COL,
    INTRUSION_COL,
)
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    SpeedCalculation,
    _compute_individual_distances,
)
from pedpy.methods.speed_calculator import compute_individual_speed


class IntrusionMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of method to compute the intrusion."""

    MAX = "max"
    """Use the max value in neighborhood as intrusion."""
    SUM = "sum"
    """Use the sum of all values in neighborhood as intrusion."""


def compute_intrusion(
    *,
    traj_data: TrajectoryData,
    r_soc: float = 0.8,
    l_min: float = 0.2,
    method: IntrusionMethod = IntrusionMethod.SUM,
) -> pandas.DataFrame:
    """_summary_.

    TODO add documentation here
    Args:
        traj_data (TrajectoryData): _description_
        r_soc (float): _description_
        l_min (float): _description_
        method (IntrusionMethod, optional): _description_. Defaults to IntrusionMethod.SUM.

    Returns:
        pandas.DataFrame: _description_
    """
    intrusion = _compute_individual_distances(traj_data=traj_data)
    intrusion[INTRUSION_COL] = (
        (r_soc - l_min) / (intrusion.distance - l_min)
    ) ** 2
    intrusion = (
        intrusion.groupby(by=[ID_COL, FRAME_COL])
        .agg(
            intrusion=(INTRUSION_COL, method.value),
        )
        .reset_index()
    )

    return intrusion


def compute_avoidance(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    radius: float = 0.2,
    tau_0: float,
) -> pandas.DataFrame:
    """_summary_.

    TODO add documentation here

    Args:
        traj_data (TrajectoryData): _description_
        frame_step (int): _description_
        radius (float): _description_
        tau_0 (float): _description_

    Returns:
        pandas.DataFrame: _description_
    """
    velocity = compute_individual_speed(
        traj_data=traj_data,
        frame_step=frame_step,
        compute_velocity=True,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    data = pandas.merge(traj_data.data, velocity, on=[ID_COL, FRAME_COL])
    data["velocity"] = shapely.points(data.v_x, data.v_y)

    matrix = pandas.merge(
        data, data, how="outer", on=FRAME_COL, suffixes=("", "_neighbor")
    )
    matrix = matrix[matrix.id != matrix.id_neighbor]

    distance = np.linalg.norm(
        shapely.get_coordinates(matrix.point)
        - shapely.get_coordinates(matrix.point_neighbor),
        axis=1,
    )

    e_v = (
        shapely.get_coordinates(matrix.point)
        - shapely.get_coordinates(matrix.point_neighbor)
    ) / distance[:, np.newaxis]

    v_rel = shapely.get_coordinates(matrix.velocity) - shapely.get_coordinates(
        matrix.velocity_neighbor
    ) / np.linalg.norm(
        shapely.get_coordinates(matrix.velocity)
        - shapely.get_coordinates(matrix.velocity_neighbor)
    )

    v_rel_norm = np.linalg.norm(v_rel, axis=1)

    dot_product = np.sum(
        np.array(np.array(e_v.tolist())) * np.array(np.array(v_rel.tolist())),
        axis=1,
    )

    cos_alpha = dot_product / (distance * v_rel_norm)

    ttc = np.full(matrix.shape[0], np.inf)

    capital_a = (cos_alpha**2 - 1) * distance**2 + radius**2

    # (0.5 * l_a + 0.5 * l_b)**2 in paper
    sqrt_a_safe = np.where(
        capital_a >= 0, np.sqrt(np.where(capital_a >= 0, capital_a, 0)), np.nan
    )

    valid_conditions = (
        (capital_a >= 0)
        & (-cos_alpha * distance - sqrt_a_safe >= 0)
        & (v_rel_norm != 0)
    )

    ttc[valid_conditions] = (
        -cos_alpha[valid_conditions] * distance[valid_conditions]
        - np.sqrt(capital_a[valid_conditions])
    ) / v_rel_norm[valid_conditions]

    matrix[AVOIDANCE_COL] = tau_0 / ttc

    avoidance = matrix.groupby(by=[ID_COL, FRAME_COL], as_index=False).agg(
        avoidance=(AVOIDANCE_COL, "max")
    )
    return avoidance
