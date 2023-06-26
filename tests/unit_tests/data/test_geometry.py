import numpy as np
import pytest
import shapely
from numpy.testing import assert_equal
from shapely import LineString

from pedpy.data.geometry import MeasurementLine


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
    "coordinates, message",
    [
        (
            ([0, 1], [1, 0], [2, 0]),
            "measurement line may only consists of 2 points",
        ),
        (
            ([0, 1], [1, 0], [2, 0], [-2, 0]),
            "measurement line may only consists of 2 points",
        ),
        (
            ([0, 1]),
            "could not create measurement line from the given coordinates",
        ),
        (
            ([0, 1], [0, 1]),
            "start and end point of measurement line need to be different.",
        ),
        (
            shapely.LineString([[0, 1], [1, 0], [2, 0], [-2, 0]]),
            "measurement line may only consists of 2 points",
        ),
        (
            [shapely.Point((0.0, 3.0)), shapely.Point((0.0, 3.0))],
            "start and end point of measurement line need to be different.",
        ),
        (
            np.array([shapely.Point((20, 1))]),
            "could not create measurement line from the given coordinates",
        ),
        (
            shapely.to_wkt(shapely.Point((20, 1))),
            "could not create a line string from the given input",
        ),
    ],
)
def test_create_measurement_line_from_coordinates_error(coordinates, message):
    with pytest.raises(ValueError, match=fr".*{message}.*"):
        measurement_line = MeasurementLine(coordinates)
        assert len(measurement_line.coords) == 2


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
