"""Dimensionless numbers for pedestrian crowd classification.

Implements the Intrusion number and Avoidance number defined in
Cordes et al., PNAS Nexus 2024 (https://doi.org/10.1093/pnasnexus/pgae120).
"""


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
    r"""Compute the intrusion number for each pedestrian per frame.

    The intrusion variable :math:`\mathcal{I}n_i` quantifies how much
    other agents encroach on pedestrian *i*'s personal space
    (Cordes et al. 2024, Eq. 1):

    .. math::

        \mathcal{I}n_i = \sum_{j \in \mathcal{N}_i}
        \left(\frac{r_\text{soc} - \ell_\text{min}}
        {r_{ij} - \ell_\text{min}}\right)^{k_I},

    with :math:`k_I = 2`. The neighbor set :math:`\mathcal{N}_i` contains
    all agents *j* with :math:`r_{ij} \leq 3\,r_\text{soc}`.

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        r_soc (float): social radius in m (default 0.8)
        l_min (float): pedestrian diameter in m (default 0.2)
        method (IntrusionMethod): aggregation over neighbors,
            ``SUM`` (default, as in the paper) or ``MAX``

    Returns:
        DataFrame with columns 'id', 'frame', and 'intrusion'
    """
    intrusion = _compute_individual_distances(traj_data=traj_data)
    intrusion = intrusion[intrusion.distance <= 3 * r_soc]
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
    r"""Compute the avoidance number for each pedestrian per frame.

    The avoidance variable :math:`\mathcal{A}v_i` quantifies the
    imminence of collisions that pedestrian *i* faces, based on
    the time-to-collision (TTC) with neighbors
    (Cordes et al. 2024, Eq. 2):

    .. math::

        \mathcal{A}v_i = \sum_{j \in \mathcal{N}'_i}
        \left(\frac{\tau_0}{\tau_{ij}}\right)^{k_A},

    with :math:`k_A = 1`. The neighbor set :math:`\mathcal{N}'_i` is
    restricted to the agent with the shortest TTC :math:`\tau_{ij}`,
    implemented by taking the ``max`` over :math:`\tau_0 / \tau_{ij}`.

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        frame_step (int): number of frames used for velocity computation
        radius (float): combined disc radius for TTC in m (default 0.2)
        tau_0 (float): reference timescale in s (paper uses 3.0)

    Returns:
        DataFrame with columns 'id', 'frame', and 'avoidance'
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

    delta_v = (
        shapely.get_coordinates(matrix.velocity)
        - shapely.get_coordinates(matrix.velocity_neighbor)
    )
    delta_v_norm = np.linalg.norm(delta_v, axis=1)
    v_rel_hat = delta_v / delta_v_norm[:, np.newaxis]

    cos_alpha = np.sum(
        np.array(e_v.tolist()) * np.array(v_rel_hat.tolist()),
        axis=1,
    )

    ttc = np.full(matrix.shape[0], np.inf)

    capital_a = (cos_alpha**2 - 1) * distance**2 + radius**2

    # (0.5 * l_a + 0.5 * l_b)**2 in paper
    sqrt_a_safe = np.where(
        capital_a >= 0, np.sqrt(np.where(capital_a >= 0, capital_a, 0)), np.nan
    )

    valid_conditions = (
        (capital_a >= 0)
        & (-cos_alpha * distance - sqrt_a_safe >= 0)
        & (delta_v_norm != 0)
    )

    ttc[valid_conditions] = (
        -cos_alpha[valid_conditions] * distance[valid_conditions]
        - np.sqrt(capital_a[valid_conditions])
    ) / delta_v_norm[valid_conditions]

    matrix[AVOIDANCE_COL] = tau_0 / ttc

    avoidance = matrix.groupby(by=[ID_COL, FRAME_COL], as_index=False).agg(
        avoidance=(AVOIDANCE_COL, "max")
    )
    return avoidance
