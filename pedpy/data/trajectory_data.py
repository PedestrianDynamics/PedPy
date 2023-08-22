"""Module handling the trajectory data of the analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas
import shapely

from pedpy.column_identifier import POINT_COL


@dataclass(frozen=True)
class TrajectoryData:
    """Trajectory Data.

    Wrapper around the trajectory data, holds the data as a data frame.

    Note:
        The coordinate data is stored in meter ('m')!

    Args:
        data (pandas.DataFrame): data frame containing the data in the form:
            "id", "frame", "x", "y"
        frame_rate (float): frame rate of the trajectory file

    Attributes:
        data (pandas.DataFrame): data frame containing the trajectory data with the
            columns: "id", "frame", "x", "y", "point"
        frame_rate (float): frame rate of the trajectory data
    """

    data: pandas.DataFrame
    frame_rate: float

    def __post_init__(self):
        """Adds a column with the position to :attr:`data`.

        The current position of the pedestrian in the row is added as a
        :class:`shapely.Point` to :attr:`data`, allowing easier geometrical
        computations directly.
        """
        data = self.data.copy(deep=True)
        data.loc[:, POINT_COL] = shapely.points(data.x, data.y)
        object.__setattr__(self, "data", data)

    def __repr__(self):
        """String representation for TrajectoryData object.

        Returns:
            string representation for TrajectoryData object
        """
        message = f"""TrajectoryData:
        frame rate: {self.frame_rate}
        frames: [{self.data.frame.min(), self.data.frame.max()}]
        number pedestrians: {self.data.id.unique().size}
        bounding box: {shapely.MultiPoint(self.data.point).bounds}
        data: 
        {self.data.head(10)}
        """
        return message
