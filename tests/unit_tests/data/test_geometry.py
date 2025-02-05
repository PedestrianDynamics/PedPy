import numpy as np
import pytest
import shapely

from pedpy.data.geometry import (
    GeometryError,
    MeasurementArea,
    MeasurementLine,
    WalkableArea,
    _create_polygon_from_input,
)


# ###############################################################################
# # Walkable Area
# ###############################################################################
@pytest.mark.parametrize(
    "point_input",
    [
        ([(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],),
        (
            [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
            [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
        ),
        (
            [
                (2.4, -1.3),
                (4.1, -5),
                (3, -7.2),
                (-1.3, -10.1),
                (-2, -3.1),
                (2.4, -1.3),
            ],
            [
                [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
            ],
        ),
        (
            [
                (-1.1, -2.1),
                (-2.3, 3.1),
                (2.5, 4.2),
                (1.2, -3.4),
                (0.1, 1.1),
            ],
        ),
    ],
)
def test_create_walkable_area_from_coordinates(point_input):
    reference_polygon = shapely.Polygon(*point_input)
    walkable_area = WalkableArea(*point_input)

    assert walkable_area.polygon.equals_exact(
        reference_polygon, tolerance=1e-20
    )
    assert walkable_area.area != 0
    assert walkable_area.polygon.is_simple
    assert shapely.is_prepared(walkable_area.polygon)


@pytest.mark.parametrize(
    "point_input",
    [
        (
            [
                shapely.Point(0, 3.1),
                shapely.Point(1.2, 5.4),
                shapely.Point(4.1, 7.9),
                shapely.Point(7.1, 3.0),
                shapely.Point(0, 3.1),
            ],
        ),
        (
            [
                shapely.Point(-5.1, -3.2),
                shapely.Point(-3.1, 2.4),
                shapely.Point(-1.1, -2.5),
                shapely.Point(-5.1, -3.2),
            ],
            [
                [
                    shapely.Point(-4.1, -2.3),
                    shapely.Point(-2.1, -2.3),
                    shapely.Point(-2.1, -1.1),
                    shapely.Point(-4.1, -1.1),
                ]
            ],
        ),
        (
            [
                shapely.Point(2.4, -1.3),
                shapely.Point(4.1, -5),
                shapely.Point(3, -7.2),
                shapely.Point(-1.3, -10.1),
                shapely.Point(-2, -3.1),
                shapely.Point(2.4, -1.3),
            ],
            [
                [
                    shapely.Point(3.1, -3.5),
                    shapely.Point(1.7, -2.3),
                    shapely.Point(0.9, -4.1),
                ],
                [
                    shapely.Point(-1.2, -3.9),
                    shapely.Point(-0.8, -6.1),
                    shapely.Point(0.1, -5.9),
                    shapely.Point(0.2, -4.3),
                ],
                [
                    shapely.Point(0.7, -7.5),
                    shapely.Point(2.1, -6.5),
                    shapely.Point(2.5, -4.7),
                ],
            ],
        ),
        (
            [
                shapely.Point(-1.1, -2.1),
                shapely.Point(-2.3, 3.1),
                shapely.Point(2.5, 4.2),
                shapely.Point(1.2, -3.4),
                shapely.Point(0.1, 1.1),
            ],
        ),
    ],
)
def test_create_walkable_area_from_points(point_input):
    reference_polygon = shapely.Polygon(*point_input)
    walkable_area = WalkableArea(*point_input)

    assert walkable_area.polygon.equals_exact(
        reference_polygon, tolerance=1e-20
    )
    assert walkable_area.area != 0
    assert walkable_area.polygon.is_simple
    assert shapely.is_prepared(walkable_area.polygon)


@pytest.mark.parametrize(
    "reference_polygon",
    [
        shapely.Polygon(
            [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
        ),
        shapely.Polygon(
            [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
            [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
        ),
        shapely.Polygon(
            [
                (2.4, -1.3),
                (4.1, -5),
                (3, -7.2),
                (-1.3, -10.1),
                (-2, -3.1),
                (2.4, -1.3),
            ],
            [
                [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
            ],
        ),
        shapely.Polygon(
            [(-1.1, -2.1), (-2.3, 3.1), (2.5, 4.2), (1.2, -3.4), (0.1, 1.1)]
        ),
    ],
)
def test_create_walkable_area_from_polygon(reference_polygon):
    walkable_area = WalkableArea(reference_polygon)

    assert walkable_area.polygon.equals_exact(
        reference_polygon, tolerance=1e-20
    )
    assert walkable_area.area != 0
    assert walkable_area.polygon.is_simple
    assert shapely.is_prepared(walkable_area.polygon)


@pytest.mark.parametrize(
    "geometry_collection",
    [
        shapely.GeometryCollection(
            shapely.Polygon(
                [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
            )
        ),
        shapely.GeometryCollection(
            [
                shapely.Polygon(
                    [(-1, 3), (-1, -7), (-13, -2), (-5, 3), (-1, 3)],
                    [[(-6, 0), (-8, -2), (-5, -2)]],
                ),
                shapely.Polygon([(-1, 3), (11, 3), (11, -5), (-1, -7)]),
            ]
        ),
        shapely.GeometryCollection(
            shapely.GeometryCollection(
                [
                    shapely.Polygon(
                        [(-1, 3), (-1, -7), (-13, -2), (-5, 3), (-1, 3)],
                        [[(-6, 0), (-8, -2), (-5, -2)]],
                    ),
                    shapely.Polygon([(-1, 3), (11, 3), (11, -5), (-1, -7)]),
                ]
            )
        ),
    ],
)
def test_create_walkable_area_from_geometry_collection(geometry_collection):
    reference_polygon = shapely.union_all(geometry_collection)

    walkable_area = WalkableArea(geometry_collection)

    assert walkable_area.polygon.equals_exact(
        reference_polygon, tolerance=1e-20
    )
    assert walkable_area.area != 0
    assert walkable_area.polygon.is_simple
    assert shapely.is_prepared(walkable_area.polygon)


@pytest.mark.parametrize(
    "wkt",
    [
        shapely.to_wkt(
            shapely.Polygon(
                [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
                [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [
                    (2.4, -1.3),
                    (4.1, -5),
                    (3, -7.2),
                    (-1.3, -10.1),
                    (-2, -3.1),
                    (2.4, -1.3),
                ],
                [
                    [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                    [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                    [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
                ],
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [(-1.1, -2.1), (-2.3, 3.1), (2.5, 4.2), (1.2, -3.4), (0.1, 1.1)]
            ),
        ),
        shapely.to_wkt(
            shapely.GeometryCollection(
                shapely.Polygon(
                    [
                        (2.4, -1.3),
                        (4.1, -5),
                        (3, -7.2),
                        (-1.3, -10.1),
                        (-2, -3.1),
                        (2.4, -1.3),
                    ],
                    [
                        [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                        [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                        [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
                    ],
                )
            )
        ),
    ],
)
def test_create_walkable_area_from_wkt(wkt):
    reference_polygon = shapely.from_wkt(wkt)
    walkable_area = WalkableArea(wkt)

    if isinstance(reference_polygon, shapely.GeometryCollection):
        reference_polygon = shapely.union_all(reference_polygon)

    assert walkable_area.polygon.equals_exact(
        reference_polygon, tolerance=1e-20
    )
    assert walkable_area.area != 0
    assert walkable_area.polygon.is_simple
    assert shapely.is_prepared(walkable_area.polygon)


@pytest.mark.parametrize(
    "area_input, message",
    [
        (
            shapely.Polygon(
                [(-1, -1), (-1, 1), (1, 1), (1, -1)], [[(5, 5), (4, 5), (4, 4)]]
            ),
            "Holes need to be inside the walkable area",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(2, 5),
                    shapely.Point(2, 2),
                    shapely.Point(-2, 5),
                ]
            ),
            "Only simple polygons with non-zero area are allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                ]
            ),
            "Only simple polygons with non-zero area are allowed",
        ),
    ],
)
def test_create_walkable_area_error(area_input, message):
    with pytest.raises(GeometryError, match=rf".*{message}.*"):
        walkable_area = WalkableArea(area_input)


def test_changing_walkable_area_fails():
    with pytest.raises(
        AttributeError,
        match=r"Walkable area can not be changed after construction!",
    ):
        walkable_area = WalkableArea(
            shapely.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
        )

        walkable_area._polygon = shapely.LinearRing(
            ((0, 0), (0, 1), (1, 1), (1, 0))
        )


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
            shapely.Point(0, 3.1),
            shapely.Point(1.2, 5.4),
            shapely.Point(4.1, 7.9),
            shapely.Point(7.1, 3.0),
            shapely.Point(0, 3.1),
        ],
        [
            shapely.Point(-5.1, -3.2),
            shapely.Point(-3.1, 2.4),
            shapely.Point(-1.1, -2.5),
            shapely.Point(-5.1, -3.2),
        ],
        [
            shapely.Point(2.4, -1.3),
            shapely.Point(4.1, -5),
            shapely.Point(3, -7.2),
            shapely.Point(-1.3, -10.1),
            shapely.Point(-2, -3.1),
            shapely.Point(2.4, -1.3),
        ],
    ],
)
def test_create_measurement_area_from_points(points):
    reference_polygon = shapely.Polygon(points)
    measurement_area = MeasurementArea(points)
    assert measurement_area.coords[:] == reference_polygon.exterior.coords[:]
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple
    assert shapely.is_prepared(measurement_area.polygon)


@pytest.mark.parametrize(
    "polygon",
    [
        shapely.Polygon(
            [
                shapely.Point(0, 3.1),
                shapely.Point(1.2, 5.4),
                shapely.Point(4.1, 7.9),
                shapely.Point(7.1, 3.0),
                shapely.Point(0, 3.1),
            ]
        ),
        shapely.Polygon(
            [
                shapely.Point(-5.1, -3.2),
                shapely.Point(-3.1, 2.4),
                shapely.Point(-1.1, -2.5),
                shapely.Point(-5.1, -3.2),
            ]
        ),
        shapely.Polygon(
            [
                shapely.Point(2.4, -1.3),
                shapely.Point(4.1, -5),
                shapely.Point(3, -7.2),
                shapely.Point(-1.3, -10.1),
                shapely.Point(-2, -3.1),
                shapely.Point(2.4, -1.3),
            ]
        ),
    ],
)
def test_create_measurement_area_from_polygon(polygon):
    measurement_area = MeasurementArea(polygon)
    assert measurement_area.coords[:] == polygon.exterior.coords[:]
    assert measurement_area.area != 0
    assert measurement_area._polygon.is_simple
    assert shapely.is_prepared(measurement_area.polygon)


@pytest.mark.parametrize(
    "wkt",
    [
        shapely.to_wkt(
            shapely.Polygon(
                [
                    shapely.Point(0, 3.1),
                    shapely.Point(1.2, 5.4),
                    shapely.Point(4.1, 7.9),
                    shapely.Point(7.1, 3.0),
                    shapely.Point(0, 3.1),
                ]
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [
                    shapely.Point(-5.1, -3.2),
                    shapely.Point(-3.1, 2.4),
                    shapely.Point(-1.1, -2.5),
                    shapely.Point(-5.1, -3.2),
                ]
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [
                    shapely.Point(2.4, -1.3),
                    shapely.Point(4.1, -5),
                    shapely.Point(3, -7.2),
                    shapely.Point(-1.3, -10.1),
                    shapely.Point(-2, -3.1),
                    shapely.Point(2.4, -1.3),
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
    assert shapely.is_prepared(measurement_area.polygon)


@pytest.mark.parametrize(
    "area_input, message",
    [
        (
            [
                shapely.Point(2.4, -1.3),
            ],
            "Could not create measurement area from the given input:",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-5.1, -3.2),
                    shapely.Point(-3.1, 2.4),
                    shapely.Point(-1.1, -2.5),
                    shapely.Point(-5.1, -3.2),
                ],
                [
                    [
                        shapely.Point(-3, -1),
                        shapely.Point(-2.5, -1.5),
                        shapely.Point(-3, 2.5),
                    ]
                ],
            ),
            "Measurement area can not be created from polygon with holes",
        ),
        (
            shapely.to_wkt(
                shapely.Polygon(
                    [
                        shapely.Point(-5.1, -3.2),
                        shapely.Point(-3.1, 2.4),
                        shapely.Point(-1.1, -2.5),
                        shapely.Point(-5.1, -3.2),
                    ],
                    [
                        [
                            shapely.Point(-3, -1),
                            shapely.Point(-2.5, -1.5),
                            shapely.Point(-3, 2.5),
                        ]
                    ],
                )
            ),
            "Measurement area can not be created from polygon with holes",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(1, 2),
                    shapely.Point(1, 4),
                    shapely.Point(1, 10),
                    shapely.Point(1, 2),
                ]
            ),
            "Only simple polygons with non-zero area are allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(2, 5),
                    shapely.Point(2, 2),
                    shapely.Point(-2, 5),
                ]
            ),
            "Only simple polygons with non-zero area are allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(1, -3),
                    shapely.Point(4, -1),
                    shapely.Point(8, -5),
                    shapely.Point(6, -7),
                    shapely.Point(4, -3),
                ]
            ),
            "Measurement areas needs to be convex",
        ),
        (
            shapely.to_wkt(shapely.LineString([(0.0, 3.0), (1.0, -10.0)])),
            "Unexpected geometry type found in GeometryCollection",
        ),
    ],
)
def test_create_measurement_area_error(area_input, message):
    with pytest.raises(GeometryError, match=rf".*{message}.*"):
        measurement_area = MeasurementArea(area_input)


def test_changing_measurement_area_fails():
    with pytest.raises(
        AttributeError,
        match=r"Measurement area can not be changed after construction!",
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
        [
            shapely.Point((0.0, 3.0)),
            shapely.Point((1.0, -10.0)),
        ],
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
            [
                shapely.Point((0.0, 3.0)),
                shapely.Point((0.0, 3.0)),
            ],
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
    with pytest.raises(GeometryError, match=rf".*{message}.*"):
        measurement_line = MeasurementLine(line_input)


def test_changing_measurement_line_fails():
    with pytest.raises(
        AttributeError,
        match=r"Measurement line can not be changed after construction!",
    ):
        measurement_line = MeasurementLine(
            [
                shapely.Point((0.0, 3.0)),
                shapely.Point((1.0, -10.0)),
            ]
        )
        measurement_line._line = shapely.LineString(
            [
                shapely.Point((0.0, 3.0)),
                shapely.Point((1.0, -10.0)),
            ]
        )


###############################################################################
# Helper functions
###############################################################################
@pytest.mark.parametrize(
    "coordinate_input",
    [
        ([(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],),
        (
            [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
            [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
        ),
        (
            [
                (2.4, -1.3),
                (4.1, -5),
                (3, -7.2),
                (-1.3, -10.1),
                (-2, -3.1),
                (2.4, -1.3),
            ],
            [
                [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
            ],
        ),
    ],
)
def test_create_polygon_from_coordinates(coordinate_input):
    reference_polygon = shapely.Polygon(*coordinate_input)
    polygon = _create_polygon_from_input(*coordinate_input)

    assert isinstance(polygon, shapely.Polygon)
    assert polygon.equals_exact(reference_polygon, tolerance=1e-20)
    assert polygon.area != 0
    assert polygon.is_simple


@pytest.mark.parametrize(
    "point_input",
    [
        (
            [
                shapely.Point(0, 3.1),
                shapely.Point(1.2, 5.4),
                shapely.Point(4.1, 7.9),
                shapely.Point(7.1, 3.0),
                shapely.Point(0, 3.1),
            ],
        ),
        (
            [
                shapely.Point(-5.1, -3.2),
                shapely.Point(-3.1, 2.4),
                shapely.Point(-1.1, -2.5),
                shapely.Point(-5.1, -3.2),
            ],
            [
                [
                    shapely.Point(-4.1, -2.3),
                    shapely.Point(-2.1, -2.3),
                    shapely.Point(-2.1, -1.1),
                    shapely.Point(-4.1, -1.1),
                ]
            ],
        ),
        (
            [
                shapely.Point(2.4, -1.3),
                shapely.Point(4.1, -5),
                shapely.Point(3, -7.2),
                shapely.Point(-1.3, -10.1),
                shapely.Point(-2, -3.1),
                shapely.Point(2.4, -1.3),
            ],
            [
                [
                    shapely.Point(3.1, -3.5),
                    shapely.Point(1.7, -2.3),
                    shapely.Point(0.9, -4.1),
                ],
                [
                    shapely.Point(-1.2, -3.9),
                    shapely.Point(-0.8, -6.1),
                    shapely.Point(0.1, -5.9),
                    shapely.Point(0.2, -4.3),
                ],
                [
                    shapely.Point(0.7, -7.5),
                    shapely.Point(2.1, -6.5),
                    shapely.Point(2.5, -4.7),
                ],
            ],
        ),
    ],
)
def test_create_polygon_from_points(point_input):
    reference_polygon = shapely.Polygon(*point_input)
    polygon = _create_polygon_from_input(*point_input)

    assert isinstance(polygon, shapely.Polygon)
    assert polygon.equals_exact(reference_polygon, tolerance=1e-20)
    assert polygon.area != 0
    assert polygon.is_simple


@pytest.mark.parametrize(
    "reference_polygon",
    [
        shapely.Polygon(
            [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
        ),
        shapely.Polygon(
            [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
            [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
        ),
        shapely.Polygon(
            [
                (2.4, -1.3),
                (4.1, -5),
                (3, -7.2),
                (-1.3, -10.1),
                (-2, -3.1),
                (2.4, -1.3),
            ],
            [
                [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
            ],
        ),
    ],
)
def test_create_polygon_from_polygon(reference_polygon):
    polygon = _create_polygon_from_input(reference_polygon)

    assert isinstance(polygon, shapely.Polygon)
    assert polygon.equals_exact(reference_polygon, tolerance=1e-20)
    assert polygon.area != 0
    assert polygon.is_simple


@pytest.mark.parametrize(
    "wkt",
    [
        shapely.to_wkt(
            shapely.Polygon(
                [(0, 3.1), (1.2, 5.4), (4.1, 7.9), (7.1, 3.0), (0, 3.1)],
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [(-5.1, -3.2), (-3.1, 2.4), (-1.1, -2.5), (-5.1, -3.2)],
                [[(-4.1, -2.3), (-2.1, -2.3), (-2.1, -1.1), (-4.1, -1.1)]],
            )
        ),
        shapely.to_wkt(
            shapely.Polygon(
                [
                    (2.4, -1.3),
                    (4.1, -5),
                    (3, -7.2),
                    (-1.3, -10.1),
                    (-2, -3.1),
                    (2.4, -1.3),
                ],
                [
                    [(3.1, -3.5), (1.7, -2.3), (0.9, -4.1)],
                    [(-1.2, -3.9), (-0.8, -6.1), (0.1, -5.9), (0.2, -4.3)],
                    [(0.7, -7.5), (2.1, -6.5), (2.5, -4.7)],
                ],
            )
        ),
    ],
)
def test_create_polygon_from_wkt(wkt):
    reference_polygon = shapely.from_wkt(wkt)
    polygon = _create_polygon_from_input(wkt)

    assert isinstance(polygon, shapely.Polygon)
    assert polygon.equals_exact(reference_polygon, tolerance=1e-20)
    assert polygon.area != 0
    assert polygon.is_simple


@pytest.mark.parametrize(
    "polygon_input, hole_input, message",
    [
        (
            shapely.Point(0, 0),
            None,
            "Could not create polygon from the given input",
        ),
        (
            shapely.LineString([(0, 0), (1, 1)]),
            None,
            "Could not create polygon from the given input",
        ),
        ("foo", None, "Could not create polygon from the given WKT"),
        (
            shapely.LineString([(0, 0), (1, 1)]).wkt,
            None,
            "Could not create polygon from the given WKT",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(2, 5),
                    shapely.Point(2, 2),
                    shapely.Point(-2, 5),
                ]
            ),
            None,
            "Only simple polygons with non-zero area are allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                ]
            ),
            None,
            "Only simple polygons with non-zero area are allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                ]
            ),
            [
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
            ],
            "If polygon is of type shapely.Polygon additional holes are not allowed",
        ),
        (
            shapely.Polygon(
                [
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                    shapely.Point(-2, 2),
                ]
            ).wkt,
            [
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
                shapely.Point(-2, 2),
            ],
            "If polygon is of type WKT additional holes are not allowed",
        ),
    ],
)
def test_create_polygon_error(polygon_input, hole_input, message):
    with pytest.raises(Exception, match=rf".*{message}.*"):
        if hole_input is None:
            polygon = _create_polygon_from_input(polygon_input)
        else:
            polygon = _create_polygon_from_input(polygon_input, hole_input)
