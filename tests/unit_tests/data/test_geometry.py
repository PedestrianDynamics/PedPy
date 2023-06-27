import numpy as np
import pytest
import shapely
from shapely import Point, Polygon

from pedpy.data.geometry import MeasurementArea, MeasurementLine


###############################################################################
# Measurement Area
###############################################################################
@pytest.mark.parametrize(
    "coordinates",
    [
        [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
        [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
        [
            (2.4, -1.3),
            (4.1, -5),
            (3, -7.2),
            (-1.3, -10.1),
            (-2, -3.1),
            (2.4, -1.3),
        ],
    ],
)
def test_create_measurement_area_from_coordinates(coordinates):
    measurement_area = MeasurementArea(coordinates)
    assert measurement_area.coords[:] == coordinates
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple


@pytest.mark.parametrize(
    "points",
    [
        [
            Point(0, 3.1),
            Point(1.2, 5.4),
            Point(4.1, 7.9),
            Point(7.1, 3.0),
            Point(0, 3.1),
        ],
        [
            Point(-5.1, -3.2),
            Point(-3.1, 2.4),
            Point(-1.1, -2.5),
            Point(-5.1, -3.2),
        ],
        [
            Point(2.4, -1.3),
            Point(4.1, -5),
            Point(3, -7.2),
            Point(-1.3, -10.1),
            Point(-2, -3.1),
            Point(2.4, -1.3),
        ],
    ],
)
def test_create_measurement_area_from_points(points):
    reference_polygon = Polygon(points)
    measurement_area = MeasurementArea(points)
    assert measurement_area.coords[:] == reference_polygon.exterior.coords[:]
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple


@pytest.mark.parametrize(
    "polygon",
    [
        Polygon(
            [
                Point(0, 3.1),
                Point(1.2, 5.4),
                Point(4.1, 7.9),
                Point(7.1, 3.0),
                Point(0, 3.1),
            ]
        ),
        Polygon(
            [
                Point(-5.1, -3.2),
                Point(-3.1, 2.4),
                Point(-1.1, -2.5),
                Point(-5.1, -3.2),
            ]
        ),
        Polygon(
            [
                Point(2.4, -1.3),
                Point(4.1, -5),
                Point(3, -7.2),
                Point(-1.3, -10.1),
                Point(-2, -3.1),
                Point(2.4, -1.3),
            ]
        ),
    ],
)
def test_create_measurement_area_from_polygon(polygon):
    measurement_area = MeasurementArea(polygon)
    assert measurement_area.coords[:] == polygon.exterior.coords[:]
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple


@pytest.mark.parametrize(
    "wkt",
    [
        shapely.to_wkt(
            Polygon(
                [
                    Point(0, 3.1),
                    Point(1.2, 5.4),
                    Point(4.1, 7.9),
                    Point(7.1, 3.0),
                    Point(0, 3.1),
                ]
            )
        ),
        shapely.to_wkt(
            Polygon(
                [
                    Point(-5.1, -3.2),
                    Point(-3.1, 2.4),
                    Point(-1.1, -2.5),
                    Point(-5.1, -3.2),
                ]
            )
        ),
        shapely.to_wkt(
            Polygon(
                [
                    Point(2.4, -1.3),
                    Point(4.1, -5),
                    Point(3, -7.2),
                    Point(-1.3, -10.1),
                    Point(-2, -3.1),
                    Point(2.4, -1.3),
                ]
            )
        ),
    ],
)
def test_create_measurement_area_from_wkt(wkt):
    reference_polygon = shapely.from_wkt(wkt)

    measurement_area = MeasurementArea(wkt)
    assert measurement_area.coords[:] == reference_polygon.exterior.coords[:]
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple


@pytest.mark.parametrize(
    "area_input, message",
    [
        (
            [
                Point(2.4, -1.3),
            ],
            "Could not create measurement area from the given coordinates:",
        ),
        (
            Polygon(
                [
                    Point(-5.1, -3.2),
                    Point(-3.1, 2.4),
                    Point(-1.1, -2.5),
                    Point(-5.1, -3.2),
                ],
                [
                    [
                        Point(-3, -1),
                        Point(-2.5, -1.5),
                        Point(-3, 2.5),
                    ]
                ],
            ),
            "Measurement area can not be created from polygon with holes",
        ),
        (
            shapely.to_wkt(
                Polygon(
                    [
                        Point(-5.1, -3.2),
                        Point(-3.1, 2.4),
                        Point(-1.1, -2.5),
                        Point(-5.1, -3.2),
                    ],
                    [
                        [
                            Point(-3, -1),
                            Point(-2.5, -1.5),
                            Point(-3, 2.5),
                        ]
                    ],
                )
            ),
            "Measurement area can not be created from polygon with holes",
        ),
        (
            Polygon(
                [
                    Point(1, 2),
                    Point(1, 4),
                    Point(1, 10),
                    Point(1, 2),
                ]
            ),
            "Only simple measurement areas with non-zero area are allowed",
        ),
        (
            Polygon(
                [
                    Point(-2, 2),
                    Point(2, 5),
                    Point(2, 2),
                    Point(-2, 5),
                ]
            ),
            "Only simple measurement areas with non-zero area are allowed",
        ),
        (
            Polygon(
                [
                    Point(1, -3),
                    Point(4, -1),
                    Point(8, -5),
                    Point(6, -7),
                    Point(4, -3),
                ]
            ),
            "Measurement areas needs to be convex",
        ),
        (
            shapely.to_wkt(shapely.LineString([(0.0, 3.0), (1.0, -10.0)])),
            "Could not create a polygon from the given input",
        ),
    ],
)
def test_create_measurement_area_error(area_input, message):
    with pytest.raises(ValueError, match=fr".*{message}.*"):
        measurement_area = MeasurementArea(area_input)


def test_changing_measurement_area_fails():
    with pytest.raises(
        AttributeError,
        match=fr"Measurement area can not be changed after construction!",
    ):
        measurement_area = MeasurementArea(((0, 0), (0, 1), (1, 1), (1, 0)))
        measurement_area._polygon = shapely.LinearRing(
            ((0, 0), (0, 1), (1, 1), (1, 0))
        )


###############################################################################
# Measurement Line
###############################################################################
@pytest.mark.parametrize(
    "coordinates",
    [
        [(0.0, 3.0), (1.0, -10.0)],
        [(20, 1), (1, 0)],
        [(-1.0, -1.0), (-4.0, 4.0)],
    ],
)
def test_create_measurement_line_from_coordinates(coordinates):
    measurement_line = MeasurementLine(coordinates)
    assert measurement_line.coords[:] == coordinates
    assert len(measurement_line.coords) == 2
    assert measurement_line.length != 0


@pytest.mark.parametrize(
    "linestring",
    [
        shapely.LineString([(0.0, 3.0), (1.0, -10.0)]),
        shapely.LineString([(20, 1), (1, 0)]),
        shapely.LineString([(-1.0, -1.0), (-4.0, 4.0)]),
    ],
)
def test_create_measurement_line_from_linestring(linestring):
    measurement_line = MeasurementLine(linestring)
    assert measurement_line.coords[:] == linestring.coords[:]
    assert len(measurement_line.coords) == 2
    assert measurement_line.length != 0


@pytest.mark.parametrize(
    "points",
    [
        [shapely.Point((0.0, 3.0)), shapely.Point((1.0, -10.0))],
        np.array([shapely.Point((20, 1)), shapely.Point((1, 0))]),
    ],
)
def test_create_measurement_line_from_points(points):
    reference_line = shapely.LineString(points)
    measurement_line = MeasurementLine(points)
    assert measurement_line.coords[:] == reference_line.coords[:]
    assert len(measurement_line.coords) == 2
    assert measurement_line.length != 0


@pytest.mark.parametrize(
    "linestring",
    [
        shapely.LineString([(0.0, 3.0), (1.0, -10.0)]),
        shapely.LineString([(20, 1), (1, 0)]),
        shapely.LineString([(-1.0, -1.0), (-4.0, 4.0)]),
    ],
)
def test_create_measurement_line_from_wkt(linestring):
    measurement_line = MeasurementLine(shapely.to_wkt(linestring))
    assert measurement_line.coords[:] == linestring.coords[:]
    assert len(measurement_line.coords) == 2
    assert measurement_line.length != 0


@pytest.mark.parametrize(
    "line_input, message",
    [
        (
            ([0, 1], [1, 0], [2, 0]),
            "Measurement line may only consists of 2 points",
        ),
        (
            ([0, 1], [1, 0], [2, 0], [-2, 0]),
            "Measurement line may only consists of 2 points",
        ),
        (
            ([0, 1]),
            "Could not create measurement line from the given coordinates",
        ),
        (
            ([0, 1], [0, 1]),
            "Start and end point of measurement line need to be different.",
        ),
        (
            shapely.LineString([[0, 1], [1, 0], [2, 0], [-2, 0]]),
            "Measurement line may only consists of 2 points",
        ),
        (
            [shapely.Point((0.0, 3.0)), shapely.Point((0.0, 3.0))],
            "Start and end point of measurement line need to be different.",
        ),
        (
            np.array([shapely.Point((20, 1))]),
            "Could not create measurement line from the given coordinates",
        ),
        (
            shapely.to_wkt(shapely.Point((20, 1))),
            "Could not create a line string from the given input",
        ),
    ],
)
def test_create_measurement_line_error(line_input, message):
    with pytest.raises(ValueError, match=fr".*{message}.*"):
        measurement_line = MeasurementLine(line_input)


def test_changing_measurement_line_fails():
    with pytest.raises(
        AttributeError,
        match=fr"Measurement line can not be changed after construction!",
    ):
        measurement_line = MeasurementLine(
            [shapely.Point((0.0, 3.0)), shapely.Point((1.0, -10.0))]
        )
        measurement_line._line = shapely.LineString(
            [shapely.Point((0.0, 3.0)), shapely.Point((1.0, -10.0))]
        )
