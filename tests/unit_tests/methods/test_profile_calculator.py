import pytest
import shapely
from shapely.geometry import box

from pedpy.data.geometry import (
    AxisAlignedMeasurementArea,
    MeasurementArea,
    WalkableArea,
)
from pedpy.errors import PedPyTypeError, PedPyValueError
from pedpy.methods.profile_calculator import get_grid_cells


def test_get_grid_cells_walkable_area():
    # Simple 2x2 grid over a 2x2 area with grid_size=1
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    grid_cells, rows, cols = get_grid_cells(
        walkable_area=walkable_area,
        grid_size=1.0,
    )
    assert rows == 2
    assert cols == 2
    assert grid_cells.shape == (4,)
    # Check that the grid cells are correct shapely boxes
    expected_boxes = [
        box(0, 2, 1, 1),
        box(1, 2, 2, 1),
        box(0, 1, 1, 0),
        box(1, 1, 2, 0),
    ]
    for cell, expected in zip(grid_cells, expected_boxes, strict=False):
        assert cell.equals(expected)


def test_get_grid_cells_axis_aligned_measurement_area():
    area = AxisAlignedMeasurementArea(0, 0, 3, 2)
    grid_cells, rows, cols = get_grid_cells(
        axis_aligned_measurement_area=area,
        grid_size=1.0,
    )
    assert rows == 2
    assert cols == 3
    assert grid_cells.shape == (6,)
    # Check that the grid cells cover the expected area
    expected_boxes = [
        box(0, 2, 1, 1),
        box(1, 2, 2, 1),
        box(2, 2, 3, 1),
        box(0, 1, 1, 0),
        box(1, 1, 2, 0),
        box(2, 1, 3, 0),
    ]
    for cell, expected in zip(grid_cells, expected_boxes, strict=False):
        assert cell.equals(expected)


def test_get_grid_cells_both_none():
    with pytest.raises(
        PedPyValueError,
        match="Either `walkable_area` or `axis_aligned_measurement_area`",
    ):
        get_grid_cells(grid_size=1.0)


def test_get_grid_cells_axis_aligned_measurement_area_wrong_type():
    area = MeasurementArea(shapely.box(0, 0, 1, 1))
    with pytest.raises(
        PedPyTypeError,
        match="AxisAlignedMeasurementArea.from_measurement_area()",
    ):
        get_grid_cells(axis_aligned_measurement_area=area, grid_size=1.0)


def test_get_grid_cells_axis_aligned_measurement_area_not_instance():
    with pytest.raises(
        PedPyTypeError,
        match="`axis_aligned_measurement_area` must be an instance of AxisAlignedMeasurementArea",
    ):
        get_grid_cells(axis_aligned_measurement_area="not_an_area", grid_size=1.0)
