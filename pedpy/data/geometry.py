"""Module handling the geometrical environment of the analysis."""

from dataclasses import dataclass
from typing import Any, List, Optional

import shapely


class GeometryError(Exception):
    """Class reflecting errors when creating PedPy geometry objects."""

    def __init__(self, message):
        """Create GeometryError with the given message.

        Args:
            message: Error message
        """
        self.message = message


@dataclass
class WalkableArea:
    """Class holding the geometry information of the analysis.

    The walkable area is the area in which the pedestrians can walk, only
    pedestrians inside this area are considered in the analysis. Parts which
    are obstructed and/or can not be reached by the pedestrians can be excluded.

    Walkable area need to be simple and cover a non-zero area.
    """

    _polygon: shapely.Polygon
    _frozen = False

    def __init__(
        self,
        polygon: Any,
        obstacles: Optional[Any] = None,
    ):
        """Creates a measurement area.

        Args:
            polygon: A sequence of (x, y) numeric coordinate pairs, or
                an array-like with shape (N, 2). Also, can be a sequence of
                :class:`shapely.Point` objects. Passing a wkt representation of
                a polygon is also allowed.
            obstacles (Optional): list of sequences of (x, y) numeric
                coordinate pairs, or an array-like with shape (N, 2). Also, can
                be a sequence of :class:`shapely.Point` objects.
        """
        self._polygon = polygon
        try:
            self._polygon = _create_polygon_from_input(polygon, obstacles)
        except Exception as exc:
            raise GeometryError(
                f"Could not create walkable area from the given "
                f"coordinates: {polygon}. Following exception was raised: {exc}"
            ) from exc

        for hole in self._polygon.interiors:
            if not self._polygon.covers(hole):
                raise GeometryError(
                    "Holes need to be inside the walkable area."
                )
        shapely.prepare(self._polygon)
        self._frozen = True

    def __setattr__(self, attr, value):
        """Overwritten to mimic the behavior const object.

        Args:
            attr: attribute to set
            value: value to be set to attribute
        """
        if getattr(self, "_frozen"):
            raise AttributeError(
                "Walkable area can not be changed after construction!"
            )
        return super().__setattr__(attr, value)

    @property
    def coords(self):
        """Coordinates of the walkable area's points.

        Returns:
            Coordinates of the points on the walkable area
        """
        return self._polygon.exterior.coords

    @property
    def area(self):
        """Area of the walkable area.

        Returns:
            Areas of the walkable area
        """
        return self._polygon.area

    @property
    def polygon(self):
        """Walkable area as :class:`shapely.Polygon`.

        Returns:
            Walkable area as :class:`shapely.Polygon`.
        """
        return self._polygon

    @property
    def bounds(self):
        """Minimum bounding region (minx, miny, maxx, maxy).

        Returns:
            Minimum bounding region (minx, miny, maxx, maxy)
        """
        return self._polygon.bounds


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
            coordinates: A sequence of (x, y) numeric coordinate pairs, or
                an array-like with shape (N, 2). Also, can be a sequence of
                shapely.Point objects. Passing a wkt representation of a polygon
                is also allowed.
        """
        try:
            self._polygon = _create_polygon_from_input(coordinates)
        except Exception as exc:
            raise GeometryError(
                f"Could not create measurement area from the given "
                f"coordinates: {exc}."
            ) from exc

        if self._polygon.interiors:
            raise GeometryError(
                "Measurement area can not be created from polygon with holes."
            )

        if (
            not shapely.difference(
                self._polygon.convex_hull, self._polygon
            ).area
            == 0
        ):
            raise GeometryError("Measurement areas needs to be convex.")

        shapely.prepare(self._polygon)
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
        """Measurement area as :class:`shapely.Polygon`.

        Returns:
            Measurement area as :class:`shapely.Polygon`.
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
            coordinates: A sequence of (x, y) numeric coordinate pairs, or
                an array-like with shape (N, 2). Also, can be a sequence of
                shapely.Point objects. Passing a wkt representation of a
                LineString is also allowed.
        """
        try:
            if isinstance(coordinates, shapely.LineString):
                self._line = coordinates
            elif isinstance(coordinates, str):
                self._line = shapely.from_wkt(coordinates)
            else:
                self._line = shapely.LineString(coordinates)
        except Exception as exc:
            raise GeometryError(
                f"Could not create measurement line from the given coordinates: {exc}."
            ) from exc

        if not isinstance(self._line, shapely.LineString):
            raise GeometryError(
                "Could not create a line string from the given input."
            )

        if len(self._line.coords) != 2:
            raise GeometryError(
                f"Measurement line may only consists of 2 points, but "
                f"{len(self._line.coords)} points given."
            )
        if self._line.length == 0:
            raise GeometryError(
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
        """Measurement line as :class:`shapely.LineString`.

        Returns:
            Measurement line as :class:`shapely.LineString`.
        """
        return self._line


###############################################################################
# Helper functions
###############################################################################
def _create_polygon_from_input(
    polygon_input: Any, holes: Optional[List[Any]] = None
) -> shapely.Polygon:
    """Convince function to create a shapely.Polygon from different input types.

    Can create a shapely.Polygon from:
        - list of coordinates
        - list of shapely.Points
        - shapely.Polygon
        - WKT-string

    Args:
        polygon_input: input which can used to assemble a shapely.Polygon
        holes (optional): List of objects which represent holes in the
            shapely.Polygon. May not be used, when polygon_input is a
            shapely.Polygon or WKT.

    Returns:
        shapely.Polygon constructed by the input
    """
    return_poly = None
    if isinstance(polygon_input, shapely.Polygon):
        if holes is not None:
            raise GeometryError(
                "If polygon is of type shapely.Polygon additional holes are not allowed"
            )
        return_poly = polygon_input
    elif isinstance(polygon_input, str):
        if holes is not None:
            raise GeometryError(
                "If polygon is of type WKT additional holes are not allowed"
            )
        try:
            wkt_type = shapely.from_wkt(polygon_input)
        except Exception as exc:
            raise GeometryError(
                f"Could not create polygon from the given WKT: {polygon_input}."
                f" See following error message:\n{exc}"
            ) from exc

        if isinstance(wkt_type, shapely.Polygon):
            return_poly = wkt_type
        else:
            raise GeometryError(
                f"Could not create a polygon from the given WKT: {polygon_input}"
            )
    else:
        try:
            return_poly = shapely.Polygon(polygon_input, holes)
        except Exception as exc:
            raise GeometryError(
                f"Could not create polygon from the given input: {polygon_input}."
            ) from exc

    if not return_poly.is_simple or return_poly.area == 0:
        raise GeometryError(
            "Only simple polygons with non-zero area are allowed, "
            "self-intersections area only allowed at boundary points."
        )

    return return_poly
