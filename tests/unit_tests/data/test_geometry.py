import numpy as np
import pytest
from shapely import LineString
from numpy.testing import assert_equal
from pedpy.data.geometry import MeasurementLine

@pytest.mark.parametrize(
    "coordinates",
    [
        [(0., 3.), (1., -10.)],
        [(20, 1), (1, 0)],
        [(-1., -1.), (-4., 4.)],
    ]
)
def test_create_measurement_line_from_coordinates(coordinates):
    measurement_line = MeasurementLine(coordinates)
    # assert measurement_line.coords()[:] == coordinates
    assert measurement_line.coords[:] == coordinates
    assert len(measurement_line.coords) == 2
    assert measurement_line.length != 0

@pytest.mark.parametrize(
    "coordinates, message",
    [
        (([0, 1], [1, 0], [2, 0]), "measurement line may only consists of 2 points"),
        (([0, 1], [1, 0], [2, 0], [-2, 0]), "measurement line may only consists of 2 points"),
        (([0, 1]), "Could not create measurement line from the given coordinates"),
        (([0, 1], [0, 1]), "start and end point of measurement line need to be different."),
    ]
)
def test_create_measurement_line_from_coordinates_error(coordinates, message):
    with pytest.raises(ValueError, match=fr".*{message}.*"):
        measurement_line = MeasurementLine(coordinates)
        assert len(measurement_line.coords) == 2
        assert measurement_line.length
