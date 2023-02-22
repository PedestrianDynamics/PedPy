"""Module handling the trajectory data of the analysis."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pandas as pd
import shapely
from aenum import Enum


class TrajectoryUnit(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of the unit of the trajectory coordinates."""

    _init_ = "value __doc__"
    METER = 1, "meter (m)"
    CENTIMETER = 100, "centimeter (cm)"


@dataclass
class TrajectoryData:
    """Trajectory Data.

    Wrapper around the trajectory data, holds the data as a data frame.

    Note:
        The coordinate data is stored in meter ('m')!

    Attributes:
        data (pd.DataFrame): data frame containing the actual data in the form:
            "ID", "frame", "X", "Y", "Z"

        frame_rate (float): frame rate of the trajectory file

        file (pathlib.Path): file from which is trajectories was read

    """

    data: pd.DataFrame
    frame_rate: float
    file: pathlib.Path

    def __init__(
        self,
        data: pd.DataFrame,
        frame_rate: float,
        file: pathlib.Path,
    ):
        """Create a trajectory.

        Args:
            data (pd.DataFrame): data frame containing the actual data in the
                form: "ID", "frame", "X", "Y", "Z"
            frame_rate (float): frame rate of the trajectory file
            file (pathlib.Path): file from which is trajectories was read
        """
        self.frame_rate = frame_rate
        self.file = file

        self.data = data
        self.data["points"] = shapely.points(self.data["X"], self.data["Y"])
