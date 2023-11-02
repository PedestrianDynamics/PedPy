import pytest
import numpy as np
from shapely import Polygon

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementLine
from pedpy.methods.method_utils import _compute_partial_line_length, _compute_orthogonal_speed_in_relation_to_proprotion


def test_calc_n_correct_result():
    line = MeasurementLine([(0, 0), (1, 1)])
    expected_n = np.array([0.5 ** 0.5, -0.5 ** 0.5])
    actual_n = line.normal_vector()
    tolerance = 1e-8
    assert (np.allclose(expected_n, actual_n, atol=tolerance))


@pytest.mark.parametrize(
    "line, polygon, expected",
    [
        (MeasurementLine([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)]), 0.5),
        (MeasurementLine([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, -0.5), (0, -0.5)]), 0)
    ]
)
def test_partial_line_length_correct(line, polygon, expected):
    actual = _compute_partial_line_length(polygon, line)
    assert (expected == actual)


def test_compute_orthogonal_speed_in_relation_to_proportion():
    v_x = 1
    v_y = 5
    poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    group = {V_X_COL: v_x, V_Y_COL: v_y, POLYGON_COL: poly}
    n = [0.5 ** 0.5, -0.5 ** 0.5]
    line = MeasurementLine([(0, 0), (1, 1)])
    assert (np.allclose(n, line.normal_vector()))
    actual = _compute_orthogonal_speed_in_relation_to_proprotion(group=group, measurement_line=line)
    assert _compute_partial_line_length(poly, line) == 0.5
    expected = (v_x * n[0] + v_y * n[1]) * 0.5
    assert np.isclose(actual, expected)
