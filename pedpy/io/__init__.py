"""IO module for loading trajectories from various formats."""

from enum import Enum


class TrajectoryUnit(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of the unit of the trajectory coordinates."""

    METER = 1
    """meter (m)"""
    CENTIMETER = 100
    """centimeter (cm)"""
