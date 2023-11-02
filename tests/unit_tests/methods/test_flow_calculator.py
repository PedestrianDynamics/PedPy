from pedpy.data.geometry import MeasurementLine
from pedpy.methods.flow_calculator import compute_line_flow


import pytest
from shapely import Polygon
import numpy as np
import pandas as pd

from pedpy.column_identifier import *

@pytest.fixture
def example_data():
    species = pd.DataFrame({
        ID_COL: [1, 2, 3, 4],
        SPECIES_COL: [1, -1, np.nan, 1]
    })
    speed = pd.DataFrame({
        ID_COL: [1, 2, 3, 4],
        FRAME_COL: [3, 3, 3, 3],
        SPEED_COL: [0, 0, 0, 0],
        V_X_COL: [1, -1, 5, -5],
        V_Y_COL: [1, -1, 5, -5]
    })
    line = MeasurementLine([(2, 0), (0, 2)])
    matching_poly1 = Polygon([(0, 0), (2, 2), (1, 3), (-1, 1)])
    matching_poly2 = Polygon([(1, -1), (3, 1), (2, 2), (0, 0)])
    non_matching_poly = Polygon()
    voronoi = pd.DataFrame({
        ID_COL: [1, 2, 3, 4],
        FRAME_COL: [3, 3, 3, 3],
        POLYGON_COL: [matching_poly1, matching_poly2, matching_poly1, non_matching_poly],
        DENSITY_COL: [3, 3, 3, 3]
    })
    return species, speed, voronoi, line


def test_compute_line_flow(example_data):
    species, speed, voronoi, line = example_data

    flow_on_line = compute_line_flow(individual_voronoi_polygons=voronoi, measurement_line=line,
                                     species=species, individual_speed=speed)

    n = line.normal_vector()
    assert flow_on_line.shape[0] == 1
    assert flow_on_line[FLOW_SP1_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 0.5 * 3)
    assert flow_on_line[FLOW_SP2_COL].values[0] == pytest.approx((n[0] * -1 + n[1] * -1) * 0.5 * -1 * 3)

    assert flow_on_line[FLOW_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 3)
