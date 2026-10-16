import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

import pedpy.methods.profile_calculator as _profile_calc_module
from pedpy.column_identifier import FRAME_COL, ID_COL
from pedpy.data.geometry import (
    AxisAlignedMeasurementArea,
    MeasurementArea,
    WalkableArea,
)
from pedpy.errors import PedPyTypeError, PedPyValueError
from pedpy.methods.profile_calculator import (
    DensityMethod,
    SpeedMethod,
    compute_density_profile,
    compute_grid_cell_polygon_intersection_area,
    compute_profiles,
    compute_speed_profile,
    get_grid_cells,
)


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


def _make_profile_test_data():
    """Create test data with Voronoi polygons across two frames."""
    walkable_area = WalkableArea(shapely.box(0, 0, 2, 2))
    grid_size = 1.0
    grid_cells, rows, cols = get_grid_cells(walkable_area=walkable_area, grid_size=grid_size)

    # Two frames, each with 2 pedestrians and their Voronoi polygons
    data = pd.DataFrame(
        {
            ID_COL: [1, 2, 1, 2],
            FRAME_COL: [0, 0, 1, 1],
            "x": [0.5, 1.5, 0.5, 1.5],
            "y": [0.5, 1.5, 1.5, 0.5],
            "speed": [1.0, 2.0, 1.5, 2.5],
            "polygon": [
                shapely.box(0, 0, 1, 1),
                shapely.box(1, 1, 2, 2),
                shapely.box(0, 1, 1, 2),
                shapely.box(1, 0, 2, 1),
            ],
        }
    )
    return data, walkable_area, grid_size, grid_cells, rows, cols


# ── on-the-fly vs precomputed parity ─────────────────────────────────────────
# The three Voronoi/Arithmetic methods all support passing a pre-computed
# intersection matrix.  One parametrized test covers all three paths.


@pytest.mark.parametrize(
    "method_kwargs",
    [
        {"density_method": DensityMethod.VORONOI},
        {"speed_method": SpeedMethod.VORONOI},
        {"speed_method": SpeedMethod.ARITHMETIC},
    ],
    ids=["density-voronoi", "speed-voronoi", "speed-arithmetic"],
)
def test_precomputed_intersection_matches_on_the_fly(method_kwargs):
    """Profiles computed with a pre-computed intersection matrix equal those computed on-the-fly."""
    data, walkable_area, grid_size, grid_cells, _, _ = _make_profile_test_data()
    intersection_area, resorted_data = compute_grid_cell_polygon_intersection_area(data=data, grid_cells=grid_cells)

    is_density = "density_method" in method_kwargs
    compute_fn = compute_density_profile if is_density else compute_speed_profile

    profiles_precomputed = compute_fn(
        data=resorted_data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        grid_intersections_area=intersection_area,
        **method_kwargs,
    )
    profiles_on_the_fly = compute_fn(
        data=data,
        walkable_area=walkable_area,
        grid_size=grid_size,
        **method_kwargs,
    )

    assert len(profiles_precomputed) == len(profiles_on_the_fly)
    for pre, fly in zip(profiles_precomputed, profiles_on_the_fly, strict=True):
        np.testing.assert_allclose(pre, fly)


def test_grid_cell_polygon_intersection_frame_chunked():
    """Verify _compute_grid_polygon_intersection produces correct results with frame-by-frame processing."""
    data, _, _, grid_cells, _, _ = _make_profile_test_data()

    intersection_area, _resorted_data = compute_grid_cell_polygon_intersection_area(data=data, grid_cells=grid_cells)

    # 4 grid cells x 4 total rows
    assert intersection_area.shape == (4, 4)

    # Each polygon exactly covers one grid cell, so each column should have
    # exactly one non-zero entry equal to 1.0 (area of the 1x1 grid cell)
    for col_idx in range(4):
        col = intersection_area[:, col_idx]
        assert np.isclose(np.sum(col), 1.0)
        assert np.count_nonzero(col > 0) == 1


# ── helpers ───────────────────────────────────────────────────────────────────


def _single_cell_area() -> tuple[WalkableArea, float]:
    """1×1 m walkable area with grid_size=1 → single 1 m² grid cell."""
    return WalkableArea(shapely.box(0, 0, 1, 1)), 1.0


def _single_ped_data(n_frames: int = 1, speed: float = 1.0) -> pd.DataFrame:
    """N frames, each with one ped whose Voronoi polygon covers the full 1×1 cell."""
    return pd.DataFrame(
        {
            ID_COL: list(range(n_frames)),
            FRAME_COL: list(range(n_frames)),
            "x": [0.5] * n_frames,
            "y": [0.5] * n_frames,
            "speed": [speed] * n_frames,
            "polygon": [shapely.box(0, 0, 1, 1)] * n_frames,
        }
    )


# ── compute_profiles ─────────────────────────────────────────────────────────


def test_compute_profiles_voronoi_returns_one_profile_per_frame():
    wa, gs = _single_cell_area()
    n_frames = 3
    density_profiles, speed_profiles = compute_profiles(
        data=_single_ped_data(n_frames=n_frames),
        walkable_area=wa,
        grid_size=gs,
        speed_method=SpeedMethod.VORONOI,
    )
    assert len(density_profiles) == n_frames
    assert len(speed_profiles) == n_frames
    # Each profile must be reshaped to (rows, cols) — i.e. (1, 1) for a single cell.
    assert density_profiles[0].shape == (1, 1)
    assert speed_profiles[0].shape == (1, 1)


def test_compute_profiles_classic_density_mean_speed_walkable_area():
    """Exercise bounds computation path via walkable_area."""
    wa, gs = _single_cell_area()
    density_profiles, speed_profiles = compute_profiles(
        data=_single_ped_data(n_frames=2),
        walkable_area=wa,
        grid_size=gs,
        density_method=DensityMethod.CLASSIC,
        speed_method=SpeedMethod.MEAN,
    )
    assert len(density_profiles) == 2
    assert len(speed_profiles) == 2


def test_compute_profiles_classic_density_mean_speed_axis_aligned():
    """Exercise bounds computation path via axis_aligned_measurement_area."""
    aa = AxisAlignedMeasurementArea(0, 0, 1, 1)
    density_profiles, speed_profiles = compute_profiles(
        data=_single_ped_data(n_frames=2),
        axis_aligned_measurement_area=aa,
        grid_size=1.0,
        density_method=DensityMethod.CLASSIC,
        speed_method=SpeedMethod.MEAN,
    )
    assert len(density_profiles) == 2
    assert len(speed_profiles) == 2


def test_compute_profiles_gaussian_density_and_speed():
    """Exercise x_center/y_center path via both GAUSSIAN methods."""
    wa = WalkableArea(shapely.box(0, 0, 2, 2))
    data = pd.DataFrame(
        {
            ID_COL: [1],
            FRAME_COL: [0],
            "x": [1.0],
            "y": [1.0],
            "speed": [1.0],
            "polygon": [shapely.box(0, 0, 2, 2)],
        }
    )
    density_profiles, speed_profiles = compute_profiles(
        data=data,
        walkable_area=wa,
        grid_size=1.0,
        density_method=DensityMethod.GAUSSIAN,
        speed_method=SpeedMethod.GAUSSIAN,
        gaussian_width=1.0,
    )
    assert len(density_profiles) == 1
    assert len(speed_profiles) == 1


def test_compute_profiles_gaussian_density_raises_without_gaussian_width():
    wa, gs = _single_cell_area()
    with pytest.raises(PedPyValueError, match="gaussian_width"):
        compute_profiles(
            data=_single_ped_data(),
            walkable_area=wa,
            grid_size=gs,
            density_method=DensityMethod.GAUSSIAN,
            speed_method=SpeedMethod.VORONOI,
        )


def test_compute_profiles_gaussian_speed_raises_without_gaussian_width():
    wa, gs = _single_cell_area()
    with pytest.raises(PedPyValueError, match="gaussian_width"):
        compute_profiles(
            data=_single_ped_data(),
            walkable_area=wa,
            grid_size=gs,
            density_method=DensityMethod.VORONOI,
            speed_method=SpeedMethod.GAUSSIAN,
        )


# ── compute_density_profile missing branches ──────────────────────────────────


def test_classic_density_with_axis_aligned_measurement_area():
    aa = AxisAlignedMeasurementArea(0, 0, 1, 1)
    profiles = compute_density_profile(
        data=_single_ped_data(),
        axis_aligned_measurement_area=aa,
        grid_size=1.0,
        density_method=DensityMethod.CLASSIC,
    )
    np.testing.assert_allclose(profiles[0], [[1.0]])


def test_gaussian_density_raises_without_gaussian_width():
    wa, gs = _single_cell_area()
    with pytest.raises(PedPyValueError, match="gaussian_width"):
        compute_density_profile(
            data=_single_ped_data(),
            walkable_area=wa,
            grid_size=gs,
            density_method=DensityMethod.GAUSSIAN,
        )


# ── _compute_density_for_frame else-branch ────────────────────────────────────


def test_density_for_frame_invalid_method_raises():
    """Defensive else-branch: passing a non-DensityMethod value raises PedPyValueError."""
    wa, gs = _single_cell_area()
    grid_cells, _, _ = get_grid_cells(walkable_area=wa, grid_size=gs)
    with pytest.raises(PedPyValueError, match="density method not accepted"):
        _profile_calc_module._compute_density_for_frame(
            frame_data=_single_ped_data(n_frames=1),
            density_method=None,  # not a DensityMethod member → hits else branch
            grid_intersections_area_frame=None,
            grid_cells=grid_cells,
            bounds=None,
            x_center=None,
            y_center=None,
            gaussian_width=None,
            grid_size=gs,
        )


# ── compute_speed_profile missing branches ────────────────────────────────────


def test_gaussian_speed_raises_without_gaussian_width():
    wa, gs = _single_cell_area()
    with pytest.raises(PedPyValueError, match="gaussian_width"):
        compute_speed_profile(
            data=_single_ped_data(),
            walkable_area=wa,
            grid_size=gs,
            speed_method=SpeedMethod.GAUSSIAN,
            gaussian_width=None,
        )


def test_mean_speed_with_axis_aligned_measurement_area():
    aa = AxisAlignedMeasurementArea(0, 0, 1, 1)
    profiles = compute_speed_profile(
        data=_single_ped_data(speed=2.0),
        axis_aligned_measurement_area=aa,
        grid_size=1.0,
        speed_method=SpeedMethod.MEAN,
    )
    np.testing.assert_allclose(profiles[0], [[2.0]])


# ── _compute_speed_for_frame else-branch ──────────────────────────────────────


def test_speed_for_frame_invalid_method_raises():
    """Defensive else-branch: passing a non-SpeedMethod value raises PedPyValueError."""
    wa, gs = _single_cell_area()
    grid_cells, _, _ = get_grid_cells(walkable_area=wa, grid_size=gs)
    with pytest.raises(PedPyValueError, match="Speed method not accepted"):
        _profile_calc_module._compute_speed_for_frame(
            frame_data=_single_ped_data(n_frames=1),
            speed_method=None,  # not a SpeedMethod member → hits else branch
            grid_intersections_area_frame=None,
            grid_cells=grid_cells,
            bounds=None,
            x_center=None,
            y_center=None,
            gaussian_width=None,
            fill_value=np.nan,
            grid_size=gs,
        )


# ── compute_grid_cell_polygon_intersection_area: empty data ───────────────────


def test_grid_cell_polygon_intersection_empty_data_returns_zero_columns():
    """Empty input → result shape (n_cells, 0) without concatenation error."""
    wa, gs = _single_cell_area()
    grid_cells, _, _ = get_grid_cells(walkable_area=wa, grid_size=gs)
    empty_data = pd.DataFrame(
        {
            ID_COL: pd.Series([], dtype=int),
            FRAME_COL: pd.Series([], dtype=int),
            "polygon": pd.Series([], dtype=object),
        }
    )
    intersection, _ = compute_grid_cell_polygon_intersection_area(data=empty_data, grid_cells=grid_cells)
    assert intersection.shape == (len(grid_cells), 0)


# ── analytical correctness tests ──────────────────────────────────────────────


def test_voronoi_density_single_ped_covering_full_cell():
    wa, gs = _single_cell_area()
    profiles = compute_density_profile(
        data=_single_ped_data(),
        walkable_area=wa,
        grid_size=gs,
        density_method=DensityMethod.VORONOI,
    )
    assert len(profiles) == 1
    np.testing.assert_allclose(profiles[0], [[1.0]])


def test_classic_density_single_ped_in_only_cell():
    wa, gs = _single_cell_area()
    profiles = compute_density_profile(
        data=_single_ped_data(),
        walkable_area=wa,
        grid_size=gs,
        density_method=DensityMethod.CLASSIC,
    )
    np.testing.assert_allclose(profiles[0], [[1.0]])


def test_voronoi_speed_single_ped_covering_full_cell():
    wa, gs = _single_cell_area()
    profiles = compute_speed_profile(
        data=_single_ped_data(speed=2.5),
        walkable_area=wa,
        grid_size=gs,
        speed_method=SpeedMethod.VORONOI,
    )
    np.testing.assert_allclose(profiles[0], [[2.5]])


def test_arithmetic_speed_two_peds_in_same_cell():
    """2 peds each covering half the cell → arithmetic mean of their speeds."""
    wa, gs = _single_cell_area()
    data = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            "x": [0.25, 0.75],
            "y": [0.5, 0.5],
            "speed": [2.0, 4.0],
            "polygon": [shapely.box(0, 0, 0.5, 1), shapely.box(0.5, 0, 1, 1)],
        }
    )
    profiles = compute_speed_profile(
        data=data,
        walkable_area=wa,
        grid_size=gs,
        speed_method=SpeedMethod.ARITHMETIC,
    )
    np.testing.assert_allclose(profiles[0], [[3.0]])


def test_mean_speed_two_peds_in_same_cell():
    """2 peds in the same cell → histogram-weighted mean of their speeds."""
    wa, gs = _single_cell_area()
    data = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            "x": [0.3, 0.7],
            "y": [0.4, 0.6],
            "speed": [2.0, 4.0],
            "polygon": [shapely.box(0, 0, 0.5, 1), shapely.box(0.5, 0, 1, 1)],
        }
    )
    profiles = compute_speed_profile(
        data=data,
        walkable_area=wa,
        grid_size=gs,
        speed_method=SpeedMethod.MEAN,
    )
    np.testing.assert_allclose(profiles[0], [[3.0]])


def test_gaussian_density_ped_equidistant_from_all_cell_centers():
    """Ped placed equidistant from all 4 cell centers → all 4 cells identical (symmetry)."""
    wa = WalkableArea(shapely.box(0, 0, 2, 2))
    # Cell centers: (0.5,0.5), (1.5,0.5), (0.5,1.5), (1.5,1.5).
    # Ped at (1,1) has |Δx|=|Δy|=0.5 to every center → identical Gaussian weight.
    data = pd.DataFrame(
        {
            ID_COL: [1],
            FRAME_COL: [0],
            "x": [1.0],
            "y": [1.0],
            "polygon": [shapely.box(0, 0, 2, 2)],
        }
    )
    profiles = compute_density_profile(
        data=data,
        walkable_area=wa,
        grid_size=1.0,
        density_method=DensityMethod.GAUSSIAN,
        gaussian_width=1.0,
    )
    assert profiles[0].shape == (2, 2)
    np.testing.assert_allclose(profiles[0], np.full((2, 2), profiles[0][0, 0]), rtol=1e-10)


# ── dispatch / program-flow tests ─────────────────────────────────────────────
# Strategy: spy on the private low-level implementations and assert they are
# called exactly once per frame.  This verifies the dispatch logic in
# _compute_density_for_frame / _compute_speed_for_frame without needing to
# replicate expected output values.


@pytest.mark.parametrize(
    "density_method,spy_target",
    [
        (DensityMethod.VORONOI, "_compute_voronoi_density_profile"),
        (DensityMethod.CLASSIC, "_compute_classic_density_profile"),
        (DensityMethod.GAUSSIAN, "_compute_gaussian_density_profile"),
    ],
)
def test_density_dispatch_calls_correct_impl_per_frame(mocker, density_method, spy_target):
    """compute_density_profile routes to the correct implementation once per frame."""
    spy = mocker.spy(_profile_calc_module, spy_target)
    wa, gs = _single_cell_area()
    n_frames = 3
    compute_density_profile(
        data=_single_ped_data(n_frames=n_frames),
        walkable_area=wa,
        grid_size=gs,
        density_method=density_method,
        gaussian_width=1.0,  # ignored unless GAUSSIAN
    )
    assert spy.call_count == n_frames


@pytest.mark.parametrize(
    "speed_method,spy_target",
    [
        (SpeedMethod.VORONOI, "_compute_voronoi_speed_profile"),
        (SpeedMethod.ARITHMETIC, "_compute_arithmetic_voronoi_speed_profile"),
        (SpeedMethod.MEAN, "_compute_mean_speed_profile"),
        (SpeedMethod.GAUSSIAN, "_compute_gaussian_speed_profile"),
    ],
)
def test_speed_dispatch_calls_correct_impl_per_frame(mocker, speed_method, spy_target):
    """compute_speed_profile routes to the correct implementation once per frame."""
    spy = mocker.spy(_profile_calc_module, spy_target)
    wa, gs = _single_cell_area()
    n_frames = 3
    compute_speed_profile(
        data=_single_ped_data(n_frames=n_frames),
        walkable_area=wa,
        grid_size=gs,
        speed_method=speed_method,
        gaussian_width=1.0,  # ignored unless GAUSSIAN
    )
    assert spy.call_count == n_frames
