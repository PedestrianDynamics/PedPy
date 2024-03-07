"""Module handling the trajectory data of the analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas
import shapely

from pedpy.column_identifier import FRAME_COL, POINT_COL


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

    def __getitem__(self, key):
        """Enables slicing the TrajectoryData based on frame range.

        Args:
            key (slice): A slice object indicating the frame range.

        Returns:
            TrajectoryData: A new instance of TrajectoryData containing only the rows
                            within the specified frame range.
        """
        if isinstance(key, slice):
            # Handle cases where start or stop might be None
            start = (
                key.start
                if key.start is not None
                else self.data[FRAME_COL].min()
            )
            stop = (
                key.stop if key.stop is not None else self.data[FRAME_COL].max()
            )

            # Ensure the slice does not go beyond the data bounds
            start = max(start, self.data[FRAME_COL].min())
            stop = min(stop, self.data[FRAME_COL].max())

            # Filter the dataframe for the specified frame range
            filtered_data = self.data[
                self.data.frame.between(start, stop, inclusive="left")
            ]

            # Return a new TrajectoryData instance with the filtered data
            return TrajectoryData(
                filtered_data.reset_index(drop=True), self.frame_rate
            )

        raise TypeError("Slicing requires a 'slice' object.")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Minimum bounding region of all points in the trajectory.

        Returns:
            Minimum bounding region (minx, miny, maxx, maxy)
        """
        return shapely.MultiPoint(self.data.point).bounds

    @property
    def number_pedestrians(self) -> int:
        """Number of pedestrians in the trajectory data.

        Returns:
            Number of pedestrians in the trajectory data.
        """
        return self.data.id.unique().size

    @property
    def frame_range(self) -> tuple[int, int]:
        """Min and max frame of the trajectory data.

        Returns:
            Min and max frame of the trajectory data (min, max)
        """
        return self.data.frame.min(), self.data.frame.max()

    def __repr__(self):
        """String representation for TrajectoryData object.

        Returns:
            string representation for TrajectoryData object
        """
        message = f"""TrajectoryData:
        frame rate: {self.frame_rate}
        frames: [{self.frame_range}]
        number pedestrians: {self.number_pedestrians}
        bounding box: {self.bounds}
        data: 
        {self.data.head(10)}
        """
        return message
