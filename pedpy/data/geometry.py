"""Module handling the geometrical environment of the analysis."""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import shapely
from shapely import Polygon

log = logging.getLogger(__name__)


@dataclass
class Geometry:
    """Class holding the geometry information of the analysis.

    Attributes:
        walkable_area (shapely.Polygon): area in which the pedestrian walk,
            they are only considered for the analysis when inside this area.
        obstacles (List[shapely.Polygon]): areas which are excluded from the
            analysis, pedestrians inside these areas will be ignored.
    """

    walkable_area: Polygon
    obstacles: List[Polygon]

    def __init__(
        self,
        *,
        walkable_area: Polygon,
        obstacles: Optional[List[Polygon]] = None,
    ):
        """Create a geometry object.

        Args:
            walkable_area (Polygon): area in pedestrian can walk
            obstacles (Optional[List[Polygon]]): list of obstacles, which will
                be excluded from the walkable area
        """
        self.obstacles = []
        self.walkable_area = walkable_area

        if obstacles is None:
            obstacles = []

        for obstacle in obstacles:
            self.add_obstacle(obstacle)

        shapely.prepare(self.walkable_area)

    def add_obstacle(self, obstacle: Polygon) -> None:
        """Adds an obstacle to the geometry.

        Args:
            obstacle (Polygon): area which will be excluded from the
                analysis.
        """
        if obstacle.within(self.walkable_area):
            self.walkable_area = shapely.difference(
                self.walkable_area, obstacle
            )
            self.obstacles.append(obstacle)
            shapely.prepare(self.walkable_area)
        else:
            log.warning(
                f"The obstacle {obstacle} is not inside the walkable area of "
                "the geometry and thus will be ignored!"
            )


###############################################################################
# Measurement Area
###############################################################################
class MeasurementArea:
    """Areas to study pedestrian dynamics.

    A measurement area is defined as an area, which is convex, simple, and
    covers a non-zero area.
    """

    _polygon: shapely.Polygon
    _frozen = False

    def __init__(self, coordinates: Any):
        """Create a measurement area from the given input.

        The measurement area may be a convex, simple area which covers a
        non-zero area.

        Args:
            coordinates: A sequence of (x, y [,z]) numeric coordinate pairs, or
                an array-like with shape (N, 2). Also, can be a sequence of
                shapely.Point objects. Passing a wkt representation of a polygon
                is also allowed.
        """
        try:
            if isinstance(coordinates, shapely.Polygon):
                self._polygon = coordinates
            elif isinstance(coordinates, str):
                self._polygon = shapely.from_wkt(coordinates)
            else:
                self._polygon = shapely.polygons(
                    shapely.LinearRing(coordinates)
                )
        except Exception as exc:
            raise ValueError(
                f"Could not create measurement area from the given coordinates: {exc}."
            ) from exc

        if not isinstance(self._polygon, shapely.Polygon):
            raise ValueError("Could not create a polygon from the given input.")

        if self._polygon.interiors:
            raise ValueError(
                "Measurement area can not be created from polygon with holes."
            )

        if not self._polygon.is_simple or self._polygon.area == 0:
            raise ValueError(
                "Only simple measurement areas with non-zero area are allowed, "
                "self-intersections area only allowed at boundary points."
            )

        if (
            not shapely.difference(
                self._polygon.convex_hull, self._polygon
            ).area
            == 0
        ):
            raise ValueError("Measurement areas needs to be convex.")

        self._frozen = True

    def __setattr__(self, attr, value):
        """Overwritten to mimic the behavior const object.

        Args:
            attr: attribute to set
            value: value to be set to attribute
        """
        if getattr(self, "_frozen"):
            raise AttributeError(
                "Measurement area can not be changed after construction!"
            )
        return super().__setattr__(attr, value)

    @property
    def coords(self):
        """Coordinates of the measurement area's points.

        Returns:
            Coordinates of the points on the measurement area
        """
        return self._polygon.exterior.coords

    @property
    def area(self):
        """Area of the measurement area.

        Returns:
            Areas of the measurement area
        """
        return self._polygon.area

    @property
    def polygon(self):
        """Measurement area as shapely Polygon.

        Returns:
            Measurement area as shapely Polygon.
        """
        return self._polygon


###############################################################################
# Measurement Line
###############################################################################
class MeasurementLine:
    """Line segments, which are used to analyze pedestrian dynamics.

    A measurement line is defined as line segment between two given points with
    a non-zero distance.
    """

    _line: shapely.LineString
    _frozen = False

    def __init__(self, coordinates: Any):
        """Create a measurement line from the given input.

        The measurement line may only consist of two points with a non-zero
        distance.

        Args:
            coordinates: A sequence of (x, y [,z]) numeric coordinate pairs, or
                an array-like with shape (N, 2). Also, can be a sequence of
                shapely.Point objects. Passing a wkt representation of a polygon
                is also allowed.
        """
        try:
            if isinstance(coordinates, shapely.LineString):
                self._line = coordinates
            elif isinstance(coordinates, str):
                self._line = shapely.from_wkt(coordinates)
            else:
                if hasattr(coordinates, "__array__"):
                    coordinates = np.asarray(coordinates)
                if isinstance(coordinates, np.ndarray) and np.issubdtype(
                    coordinates.dtype, np.number
                ):
                    pass
                else:
                    # check coordinates on points
                    def _coords(obj):
                        if isinstance(obj, shapely.Point):
                            return obj.coords[0]

                        return [float(c) for c in obj]

                coordinates = [_coords(o) for o in coordinates]
                self._line = shapely.LineString(coordinates)
        except Exception as exc:
            raise ValueError(
                f"Could not create measurement line from the given coordinates: {exc}."
            ) from exc

        if not isinstance(self._line, shapely.LineString):
            raise ValueError(
                "Could not create a line string from the given input."
            )

        if len(self._line.coords) != 2:
            raise ValueError(
                f"Measurement line may only consists of 2 points, but "
                f"{len(self._line.coords)} points given."
            )
        if self._line.length == 0:
            raise ValueError(
                "Start and end point of measurement line need to be different."
            )

        self._frozen = True

    def __setattr__(self, attr, value):
        """Overwritten to mimic the behavior const object.

        Args:
            attr: attribute to set
            value: value to be set to attribute
        """
        if getattr(self, "_frozen"):
            raise AttributeError(
                "Measurement line can not be changed after construction!"
            )
        return super().__setattr__(attr, value)

    @property
    def coords(self):
        """Coordinates of the measurement line's points.

        Returns:
            Coordinates of the points on the measurement line
        """
        return self._line.coords

    @property
    def length(self):
        """Length of the measurement line.

        Returns:
            Length of the measurement line
        """
        return self._line.length

    @property
    def xy(self):  # pylint: disable=invalid-name
        """Separate arrays of X and Y coordinate values.

        Returns:
            Separate arrays of X and Y coordinate values
        """
        return self._line.xy

    @property
    def line(self):
        """Measurement line as shapely LineString.

        Returns:
            Measurement line as shapely LineString.
        """
        return self._line
