"""Foo."""


from enum import Enum

from pedpy.data.trajectory_data import TrajectoryData
from .method_utils import _compute_individual_distances
from pedpy.column_identifier import FRAME_COL, ID_COL, INTRUSION_COL
import pandas


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
