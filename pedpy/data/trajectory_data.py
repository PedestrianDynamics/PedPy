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


@dataclass(frozen=True)
class TrajectoryData:
    """Trajectory Data.

    Wrapper around the trajectory data, holds the data as a data frame.

    Note:
        The coordinate data is stored in meter ('m')!

    Args:
        data (pd.DataFrame): data frame containing the data in the form:
            "ID", "frame", "X", "Y", "Z"
        frame_rate (float): frame rate of the trajectory file
        file (pathlib.Path): file from which is trajectories was read

    Attributes:
        data (pd.DataFrame): data frame containing the trajectory data with the
            columns: "ID", "frame", "X", "Y", "Z", "points"
        frame_rate (float): frame rate of the trajectory data
        file (pathlib.Path): file from which is trajectories was read
    """

    data: pd.DataFrame
    frame_rate: float
    file: pathlib.Path

    def __post_init__(self):
        """Adds a column with the position to :py:attr:`data`.

        The current position of the pedestrian in the row is added as a
        shapely Point to the dataframe, allowing easier geometrical
        computations directly.
        """
        data = self.data
        data["points"] = shapely.points(data["X"], data["Y"])
        object.__setattr__(self, "data", data)

    def __repr__(self):
        """String representation for TrajectoryData object.

        Returns: string representation for TrajectoryData object
        """
        message = f"""
        TrajectoryData:
        file: {self.file}
        frame rate: {self.frame_rate}
        frames: [{self.data.frame.min(), self.data.frame.max()}]
        number pedestrians: {self.data.ID.unique().size}
        bounding box: {shapely.MultiPoint(self.data.points).bounds}
        data: 
        {self.data.head(10)}
        """
        return message
