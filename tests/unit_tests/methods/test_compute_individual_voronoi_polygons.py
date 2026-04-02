"""Unit tests for compute_individual_voronoi_polygons."""

import warnings

import numpy as np
import pandas as pd
import pytest
import shapely

from pedpy.column_identifier import (
    DENSITY_COL,
    FRAME_COL,
    ID_COL,
    POLYGON_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import PedPyValueError
from pedpy.methods.method_utils import (
    Cutoff,
    _resolve_multipolygons,
    compute_individual_voronoi_polygons,
)

_OVERLAP_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Helper to build TrajectoryData from simple lists
# ---------------------------------------------------------------------------
def _make_traj(ids, frames, xs, ys, frame_rate=1.0):
    """Build a TrajectoryData from plain lists / arrays."""
    data = pd.DataFrame(
        {
            ID_COL: ids,
            FRAME_COL: frames,
            X_COL: xs,
            Y_COL: ys,
        }
    )
    return TrajectoryData(data=data, frame_rate=frame_rate)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def square_walkable_area():
    """A 10x10 square walkable area centered at (5,5)."""
    return WalkableArea([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def four_peds_one_frame(square_walkable_area):
    """Four pedestrians in a single frame, evenly placed in a 10x10 area."""
    traj = _make_traj(
        ids=[0, 1, 2, 3],
        frames=[0, 0, 0, 0],
        xs=[2.5, 7.5, 2.5, 7.5],
        ys=[2.5, 2.5, 7.5, 7.5],
    )
    return traj, square_walkable_area


@pytest.fixture
def two_peds_one_frame(square_walkable_area):
    """Two pedestrians in a single frame."""
    traj = _make_traj(
        ids=[0, 1],
        frames=[0, 0],
        xs=[3.0, 7.0],
        ys=[5.0, 5.0],
    )
    return traj, square_walkable_area


@pytest.fixture
def single_ped_one_frame(square_walkable_area):
    """One pedestrian in a single frame."""
    traj = _make_traj(
        ids=[0],
        frames=[0],
        xs=[5.0],
        ys=[5.0],
    )
    return traj, square_walkable_area


# ===========================================================================
# 1. Output schema / columns
# ===========================================================================
class TestOutputSchema:
    """Verify the returned DataFrame has expected columns and types."""

    def test_output_has_expected_columns(self, four_peds_one_frame):
        """Verify result contains id, frame, polygon, and density columns."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        assert set(result.columns) == {ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL}

    def test_polygon_column_contains_shapely_polygons(self, four_peds_one_frame):
        """Verify polygon column entries are shapely Polygon instances."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            assert isinstance(poly, shapely.Polygon)

    def test_one_row_per_ped_per_frame(self, four_peds_one_frame):
        """Verify one result row per pedestrian per frame."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        assert len(result) == 4
        assert set(result[ID_COL]) == {0, 1, 2, 3}
        assert (result[FRAME_COL] == 0).all()

    def test_multiple_frames_one_row_per_ped_per_frame(self, square_walkable_area):
        """Two frames with 4 peds each produce 8 rows."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 0, 1, 2, 3],
            frames=[0, 0, 0, 0, 1, 1, 1, 1],
            xs=[2.5, 7.5, 2.5, 7.5, 2.5, 7.5, 2.5, 7.5],
            ys=[2.5, 2.5, 7.5, 7.5, 2.5, 2.5, 7.5, 7.5],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert len(result) == 8
        assert set(result[FRAME_COL]) == {0, 1}
        for frame in [0, 1]:
            frame_result = result[result[FRAME_COL] == frame]
            assert len(frame_result) == 4
            assert set(frame_result[ID_COL]) == {0, 1, 2, 3}
        # No duplicate (id, frame) pairs
        assert result.duplicated(subset=[ID_COL, FRAME_COL]).sum() == 0


# ===========================================================================
# 2. Density correctness
# ===========================================================================
class TestDensity:
    """Density should equal 1 / polygon_area."""

    def test_density_equals_inverse_area(self, four_peds_one_frame):
        """Verify density equals 1 / polygon area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        expected_density = 1.0 / shapely.area(result[POLYGON_COL].values)
        np.testing.assert_allclose(result[DENSITY_COL].values, expected_density)

    def test_density_positive(self, four_peds_one_frame):
        """Verify all density values are positive."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        assert (result[DENSITY_COL] > 0).all()


# ===========================================================================
# 3. Polygons lie within the walkable area
# ===========================================================================
class TestPolygonsWithinWalkableArea:
    """Each polygon must be fully within the walkable area (or very close)."""

    def test_polygons_inside_walkable_area(self, four_peds_one_frame):
        """Verify all polygons lie within the walkable area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            # Allow tiny floating-point tolerance
            assert wa.polygon.buffer(1e-8).contains(poly)

    def test_polygons_inside_walkable_area_with_cutoff(self, four_peds_one_frame):
        """Verify polygons with cutoff still lie within the walkable area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=2.0))
        for poly in result[POLYGON_COL]:
            assert wa.polygon.buffer(1e-8).contains(poly)


# ===========================================================================
# 4. Pedestrian position inside their own polygon
# ===========================================================================
class TestPositionInsidePolygon:
    """Each pedestrian's position should lie inside their own Voronoi polygon."""

    def test_positions_inside_polygons(self, four_peds_one_frame):
        """Verify each pedestrian's position is inside their polygon."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        merged = result.merge(traj.data[[ID_COL, FRAME_COL, X_COL, Y_COL]], on=[ID_COL, FRAME_COL])
        for _, row in merged.iterrows():
            pt = shapely.Point(row[X_COL], row[Y_COL])
            assert row[POLYGON_COL].contains(pt) or row[POLYGON_COL].touches(pt)

    def test_positions_inside_polygons_with_cutoff(self, four_peds_one_frame):
        """Verify positions are inside polygons when cutoff is applied."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=3.0))
        merged = result.merge(traj.data[[ID_COL, FRAME_COL, X_COL, Y_COL]], on=[ID_COL, FRAME_COL])
        for _, row in merged.iterrows():
            pt = shapely.Point(row[X_COL], row[Y_COL])
            assert row[POLYGON_COL].contains(pt) or row[POLYGON_COL].touches(pt)


# ===========================================================================
# 5. Symmetric pedestrian placement -> equal polygon areas
# ===========================================================================
class TestSymmetricPlacement:
    """Symmetrically placed pedestrians should have equal-area polygons."""

    def test_four_symmetric_peds_equal_area(self, four_peds_one_frame):
        """Verify four symmetric peds each get a quarter of the area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        areas = shapely.area(result[POLYGON_COL].values)
        # All four should be equal (quarter of 10x10 = 25)
        np.testing.assert_allclose(areas, 25.0, atol=1e-6)

    def test_two_symmetric_peds_equal_area(self, square_walkable_area):
        """Two peds symmetric about x=5 in a 10x10 square each get 50."""
        traj = _make_traj(
            ids=[0, 1],
            frames=[0, 0],
            xs=[2.5, 7.5],
            ys=[5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        areas = shapely.area(result[POLYGON_COL].values)
        np.testing.assert_allclose(areas, 50.0, atol=1e-6)


# ===========================================================================
# 6. Polygons cover the entire walkable area (no gaps)
# ===========================================================================
class TestCoverage:
    """Union of all polygons should cover the entire walkable area."""

    def test_union_covers_walkable_area(self, four_peds_one_frame):
        """Verify the union of polygons equals the walkable area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        union = shapely.union_all(result[POLYGON_COL].values)
        # The union area should equal the walkable area (100)
        np.testing.assert_allclose(union.area, wa.polygon.area, atol=1e-6)

    def test_total_area_equals_walkable_area_single_ped(self, single_ped_one_frame):
        """Verify single ped's polygon covers the entire walkable area."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        total_area = sum(shapely.area(result[POLYGON_COL].values))
        np.testing.assert_allclose(total_area, wa.polygon.area, atol=1e-6)


# ===========================================================================
# 7. Cutoff reduces polygon size
# ===========================================================================
class TestCutoff:
    """Cutoff should limit the maximum extent of a Voronoi polygon."""

    def test_cutoff_reduces_polygon_area(self, four_peds_one_frame):
        """Verify cutoff produces smaller or equal polygon areas."""
        traj, wa = four_peds_one_frame
        result_no_cutoff = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        result_with_cutoff = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=1.0)
        )
        areas_no = shapely.area(result_no_cutoff[POLYGON_COL].values)
        areas_with = shapely.area(result_with_cutoff[POLYGON_COL].values)
        assert (areas_with <= areas_no + 1e-10).all()

    def test_cutoff_polygon_within_circle(self, four_peds_one_frame):
        """Each polygon should be within the buffered region used by cutoff."""
        traj, wa = four_peds_one_frame
        radius = 2.0
        quad_segments = 3
        result = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=radius, quad_segments=quad_segments)
        )
        merged = result.merge(traj.data[[ID_COL, FRAME_COL, X_COL, Y_COL]], on=[ID_COL, FRAME_COL])
        for _, row in merged.iterrows():
            pt = shapely.Point(row[X_COL], row[Y_COL])
            # Use the same buffer parameters as the cutoff to match behavior
            circle = pt.buffer(radius, quad_segs=quad_segments).buffer(1e-8)
            assert circle.contains(row[POLYGON_COL])

    def test_large_cutoff_same_as_no_cutoff(self, four_peds_one_frame):
        """A cutoff larger than the walkable area should give the same result."""
        traj, wa = four_peds_one_frame
        result_no_cutoff = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        result_large_cutoff = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=1000.0)
        )
        areas_no = shapely.area(result_no_cutoff[POLYGON_COL].values)
        areas_large = shapely.area(result_large_cutoff[POLYGON_COL].values)
        np.testing.assert_allclose(areas_no, areas_large, atol=1e-6)

    def test_cutoff_quad_segments_affects_shape(self, four_peds_one_frame):
        """Different quad_segments should produce different polygon shapes."""
        traj, wa = four_peds_one_frame
        radius = 2.0
        result_low = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=radius, quad_segments=1)
        )
        result_high = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=radius, quad_segments=16)
        )
        areas_low = shapely.area(result_low[POLYGON_COL].values)
        areas_high = shapely.area(result_high[POLYGON_COL].values)
        # Higher quad_segments -> closer to a perfect circle -> larger area
        assert (areas_high >= areas_low - 1e-10).all()

    def test_cutoff_with_non_convex_walkable_area(self):
        """Verify cutoff works correctly with a non-convex walkable area."""
        l_shape = WalkableArea([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[2.5, 7.5, 2.5, 2.5],
            ys=[2.5, 2.5, 7.5, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=l_shape, cut_off=Cutoff(radius=2.0))
        assert set(result[ID_COL]) == {0, 1, 2, 3}
        for poly in result[POLYGON_COL]:
            assert isinstance(poly, shapely.Polygon)
            assert poly.is_valid
            assert l_shape.polygon.buffer(1e-8).contains(poly)


# ===========================================================================
# 8. Blind points behavior (use_blind_points) – deprecated parameter
# ===========================================================================
class TestBlindPoints:
    """Test that use_blind_points is deprecated and has no effect.

    The parameter is kept for backwards compatibility only. All behavioral
    correctness for various pedestrian counts is covered by the other test
    classes (TestSinglePedestrian, TestSymmetricPlacement, etc.).
    """

    def test_passing_true_emits_deprecation_warning(self, single_ped_one_frame):
        """Passing use_blind_points=True must emit a DeprecationWarning."""
        traj, wa = single_ped_one_frame
        with pytest.warns(DeprecationWarning, match="use_blind_points"):
            compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=True)

    def test_passing_false_emits_deprecation_warning(self, single_ped_one_frame):
        """Passing use_blind_points=False must emit a DeprecationWarning."""
        traj, wa = single_ped_one_frame
        with pytest.warns(DeprecationWarning, match="use_blind_points"):
            compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=False)

    def test_omitting_parameter_emits_no_warning(self, single_ped_one_frame):
        """Not passing use_blind_points must not emit any DeprecationWarning."""
        traj, wa = single_ped_one_frame
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)

    def test_parameter_is_noop_true_vs_omitted(self, single_ped_one_frame):
        """use_blind_points=True must produce the same result as omitting it."""
        traj, wa = single_ped_one_frame
        result_default = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        with pytest.warns(DeprecationWarning):
            result_explicit = compute_individual_voronoi_polygons(
                traj_data=traj, walkable_area=wa, use_blind_points=True
            )
        pd.testing.assert_frame_equal(
            result_default.reset_index(drop=True),
            result_explicit.reset_index(drop=True),
        )

    def test_parameter_is_noop_false_vs_omitted(self, four_peds_one_frame):
        """use_blind_points=False must produce the same result as omitting it."""
        traj, wa = four_peds_one_frame
        result_default = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        with pytest.warns(DeprecationWarning):
            result_explicit = compute_individual_voronoi_polygons(
                traj_data=traj, walkable_area=wa, use_blind_points=False
            )
        pd.testing.assert_frame_equal(
            result_default.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True),
            result_explicit.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True),
        )


# ===========================================================================
# 9. Multiple frames
# ===========================================================================
class TestMultipleFrames:
    """Test correct behavior across multiple frames."""

    def test_ids_preserved_across_frames(self, square_walkable_area):
        """Verify pedestrian IDs are preserved across frames."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 0, 1, 2, 3],
            frames=[0, 0, 0, 0, 1, 1, 1, 1],
            xs=[2.5, 7.5, 2.5, 7.5, 3.0, 7.0, 3.0, 7.0],
            ys=[2.5, 2.5, 7.5, 7.5, 3.0, 3.0, 7.0, 7.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        for frame in [0, 1]:
            frame_ids = set(result[result[FRAME_COL] == frame][ID_COL])
            assert frame_ids == {0, 1, 2, 3}

    def test_different_positions_give_different_polygons(self, square_walkable_area):
        """Moving pedestrians should produce different polygons in different frames."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 0, 1, 2, 3],
            frames=[0, 0, 0, 0, 1, 1, 1, 1],
            xs=[2.5, 7.5, 2.5, 7.5, 3.0, 7.0, 2.0, 8.0],
            ys=[2.5, 2.5, 7.5, 7.5, 3.0, 3.0, 8.0, 8.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        # Ped 0 in frame 0 vs frame 1 should have different polygon area
        poly_f0 = result[(result[ID_COL] == 0) & (result[FRAME_COL] == 0)][POLYGON_COL].iloc[0]
        poly_f1 = result[(result[ID_COL] == 0) & (result[FRAME_COL] == 1)][POLYGON_COL].iloc[0]
        assert not np.isclose(poly_f0.area, poly_f1.area)

    def test_non_sequential_frame_numbers(self, square_walkable_area):
        """Verify function works with non-contiguous frame numbers."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 0, 1, 2, 3],
            frames=[5, 5, 5, 5, 20, 20, 20, 20],
            xs=[2.5, 7.5, 2.5, 7.5, 3.0, 7.0, 3.0, 7.0],
            ys=[2.5, 2.5, 7.5, 7.5, 3.0, 3.0, 7.0, 7.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert set(result[FRAME_COL]) == {5, 20}
        for frame in [5, 20]:
            assert set(result[result[FRAME_COL] == frame][ID_COL]) == {0, 1, 2, 3}

    def test_non_sequential_ped_ids(self, square_walkable_area):
        """Verify function works with arbitrary pedestrian IDs."""
        traj = _make_traj(
            ids=[42, 99, 7, 256],
            frames=[0, 0, 0, 0],
            xs=[2.5, 7.5, 2.5, 7.5],
            ys=[2.5, 2.5, 7.5, 7.5],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert set(result[ID_COL]) == {42, 99, 7, 256}
        assert (result[FRAME_COL] == 0).all()


# ===========================================================================
# 10. Single pedestrian gets (approximately) full walkable area
# ===========================================================================
class TestSinglePedestrian:
    """With one ped, the polygon should cover the full walkable area."""

    def test_single_ped_polygon_covers_walkable_area(self, single_ped_one_frame):
        """Verify single ped's polygon covers the full walkable area."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        poly = result.iloc[0][POLYGON_COL]
        np.testing.assert_allclose(poly.area, wa.polygon.area, atol=1e-6)

    def test_single_ped_density(self, single_ped_one_frame):
        """Verify single ped density equals 1 / walkable area."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        expected_density = 1.0 / wa.polygon.area
        np.testing.assert_allclose(result.iloc[0][DENSITY_COL], expected_density, atol=1e-10)


# ===========================================================================
# 11. Non-convex walkable area
# ===========================================================================
class TestNonConvexWalkableArea:
    """Polygons should still be clipped to a non-convex walkable area."""

    def test_polygons_within_l_shaped_area(self):
        """Verify polygons are clipped to an L-shaped walkable area."""
        # L-shaped walkable area
        l_shape = WalkableArea([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[2.5, 7.5, 2.5, 2.5],
            ys=[2.5, 2.5, 7.5, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=l_shape)
        for poly in result[POLYGON_COL]:
            assert l_shape.polygon.buffer(1e-8).contains(poly)

    def test_total_area_equals_l_shaped_area(self):
        """Verify polygon union equals the L-shaped area."""
        l_shape = WalkableArea([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[2.5, 7.5, 2.5, 2.5],
            ys=[2.5, 2.5, 7.5, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=l_shape)
        union = shapely.union_all(result[POLYGON_COL].values)
        np.testing.assert_allclose(union.area, l_shape.polygon.area, atol=1e-6)


# ===========================================================================
# 12. Walkable area with obstacles (holes)
# ===========================================================================
class TestWalkableAreaWithObstacles:
    """Test with a walkable area that has a hole/obstacle."""

    def test_polygons_do_not_overlap_obstacle(self):
        """Verify polygons do not extend into obstacle areas."""
        wa = WalkableArea(
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            obstacles=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
        )
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[2.0, 8.0, 2.0, 8.0],
            ys=[2.0, 2.0, 8.0, 8.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            assert wa.polygon.buffer(1e-8).contains(poly)


# ===========================================================================
# 13. Collinear pedestrians (edge case for Voronoi)
# ===========================================================================
class TestCollinearPedestrians:
    """Collinear pedestrians can cause degenerate Voronoi diagrams.

    shapely.voronoi_polygons handles this natively.
    """

    def test_collinear_peds_produce_valid_polygons(self, square_walkable_area):
        """Verify collinear peds produce valid polygons."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 4],
            frames=[0, 0, 0, 0, 0],
            xs=[1.0, 3.0, 5.0, 7.0, 9.0],
            ys=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert set(result[ID_COL]) == {0, 1, 2, 3, 4}
        assert (result[FRAME_COL] == 0).all()
        for poly in result[POLYGON_COL]:
            assert poly.is_valid
            assert poly.area > 0


# ===========================================================================
# 14. Pedestrians on the boundary of the walkable area
# ===========================================================================
class TestBoundaryPedestrians:
    """Pedestrians on the edge of the walkable area."""

    def test_peds_on_boundary(self, square_walkable_area):
        """Verify boundary peds produce valid polygons."""
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[0.0, 10.0, 5.0, 5.0],
            ys=[5.0, 5.0, 0.0, 10.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert set(result[ID_COL]) == {0, 1, 2, 3}
        assert (result[FRAME_COL] == 0).all()
        for poly in result[POLYGON_COL]:
            assert poly.is_valid
            assert poly.area > 0


# ===========================================================================
# 15. Consistency: calling twice with same input -> same result
# ===========================================================================
class TestDeterminism:
    """Function should be deterministic."""

    def test_same_input_same_output(self, four_peds_one_frame):
        """Verify identical inputs produce identical outputs."""
        traj, wa = four_peds_one_frame
        r1 = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        r2 = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        r1 = r1.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True)
        r2 = r2.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True)
        pd.testing.assert_frame_equal(r1[[ID_COL, FRAME_COL, DENSITY_COL]], r2[[ID_COL, FRAME_COL, DENSITY_COL]])
        for p1, p2 in zip(r1[POLYGON_COL], r2[POLYGON_COL], strict=True):
            assert p1.equals(p2)


# ===========================================================================
# 16. Cutoff with single pedestrian
# ===========================================================================
class TestCutoffSinglePed:
    """With cutoff and a single ped, polygon should be the circle-area intersection.

    The polygon should approximate the intersection of the cutoff circle
    and the walkable area.
    """

    def test_cutoff_limits_single_ped_polygon(self, single_ped_one_frame):
        """Verify cutoff limits single ped polygon to the circle area."""
        traj, wa = single_ped_one_frame
        radius = 2.0
        result = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=radius, quad_segments=16)
        )
        poly = result.iloc[0][POLYGON_COL]
        # The polygon area should be approximately pi*r^2 (circle fully inside 10x10 square)
        expected_area = np.pi * radius**2
        np.testing.assert_allclose(poly.area, expected_area, rtol=0.01)


# ===========================================================================
# 17. Varying number of peds across frames
# ===========================================================================
class TestVaryingPedsPerFrame:
    """Different number of pedestrians in different frames."""

    def test_different_ped_counts_per_frame(self, square_walkable_area):
        """Verify frames with different ped counts are handled correctly."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 0, 1],
            frames=[0, 0, 0, 0, 1, 1],
            xs=[2.5, 7.5, 2.5, 7.5, 3.0, 7.0],
            ys=[2.5, 2.5, 7.5, 7.5, 5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        frame0 = result[result[FRAME_COL] == 0]
        frame1 = result[result[FRAME_COL] == 1]
        assert set(frame0[ID_COL]) == {0, 1, 2, 3}
        assert set(frame1[ID_COL]) == {0, 1}


# ===========================================================================
# 18. Polygons are valid (no self-intersections)
# ===========================================================================
class TestPolygonValidity:
    """All returned polygons should be valid shapely Polygons."""

    def test_all_polygons_valid(self, four_peds_one_frame):
        """Verify all polygons are valid."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            assert poly.is_valid

    def test_all_polygons_valid_with_cutoff(self, four_peds_one_frame):
        """Verify all polygons are valid when cutoff is applied."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, cut_off=Cutoff(radius=2.0))
        for poly in result[POLYGON_COL]:
            assert poly.is_valid

    def test_no_empty_polygons(self, four_peds_one_frame):
        """Verify no polygons are empty."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            assert not poly.is_empty

    def test_all_polygons_have_positive_area(self, four_peds_one_frame):
        """Verify all polygons have positive area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        for poly in result[POLYGON_COL]:
            assert poly.area > 0


# ===========================================================================
# 19. Close pedestrians
# ===========================================================================
class TestClosePedestrians:
    """Pedestrians very close together should still get valid polygons."""

    def test_close_peds_valid_polygons(self, square_walkable_area):
        """Verify very close pedestrians produce valid polygons."""
        traj = _make_traj(
            ids=[0, 1, 2, 3],
            frames=[0, 0, 0, 0],
            xs=[5.0, 5.01, 5.0, 5.01],
            ys=[5.0, 5.0, 5.01, 5.01],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=square_walkable_area)
        assert set(result[ID_COL]) == {0, 1, 2, 3}
        assert (result[FRAME_COL] == 0).all()
        for poly in result[POLYGON_COL]:
            assert poly.is_valid
            assert poly.area > 0


# ===========================================================================
# 20. Non-overlapping polygons
# ===========================================================================
class TestNonOverlapping:
    """Voronoi polygons should not overlap (ignoring boundary touches)."""

    def test_polygons_do_not_overlap(self, four_peds_one_frame):
        """Verify pairwise polygon intersections have negligible area."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa)
        polys = list(result[POLYGON_COL])
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                intersection = polys[i].intersection(polys[j])
                # Intersection should be at most a line or point (dim <= 1)
                assert intersection.area < _OVERLAP_TOLERANCE


# ===========================================================================
# 21. Multipolygon resolution (non-convex geometry producing GeometryCollections)
# ===========================================================================
class TestMultipolygonResolution:
    """Test _resolve_multipolygons and the integration path.

    Tests geometries that produce non-Polygon results after intersection
    with the walkable area.
    """

    def test_two_rooms_geometry_produces_valid_polygons(self):
        """Two rooms connected by a thin corridor produce valid polygons."""
        # Two rooms connected by a narrow corridor at y=4..6
        two_rooms = WalkableArea(
            [
                (0, 0),
                (4, 0),
                (4, 4),
                (5, 4),
                (5, 0),
                (10, 0),
                (10, 10),
                (5, 10),
                (5, 6),
                (4, 6),
                (4, 10),
                (0, 10),
            ]
        )
        traj = _make_traj(
            ids=[0, 1],
            frames=[0, 0],
            xs=[2.0, 8.0],
            ys=[5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=two_rooms)
        assert set(result[ID_COL]) == {0, 1}
        assert (result[FRAME_COL] == 0).all()
        for poly in result[POLYGON_COL]:
            assert isinstance(poly, shapely.Polygon)
            assert poly.is_valid
            assert poly.area > 0

    def test_two_rooms_positions_inside_polygons(self):
        """Each pedestrian's position should lie inside their resolved polygon."""
        two_rooms = WalkableArea(
            [
                (0, 0),
                (4, 0),
                (4, 4),
                (5, 4),
                (5, 0),
                (10, 0),
                (10, 10),
                (5, 10),
                (5, 6),
                (4, 6),
                (4, 10),
                (0, 10),
            ]
        )
        traj = _make_traj(
            ids=[0, 1],
            frames=[0, 0],
            xs=[2.0, 8.0],
            ys=[5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=two_rooms)
        merged = result.merge(
            traj.data[[ID_COL, FRAME_COL, X_COL, Y_COL]],
            on=[ID_COL, FRAME_COL],
        )
        for _, row in merged.iterrows():
            pt = shapely.Point(row[X_COL], row[Y_COL])
            assert row[POLYGON_COL].contains(pt) or row[POLYGON_COL].touches(pt)


# ===========================================================================
# 22. _resolve_multipolygons unit tests (direct)
# ===========================================================================
class TestResolveMultipolygons:
    """Direct unit tests for the _resolve_multipolygons helper."""

    def test_all_polygons_returned_unchanged(self):
        """When all geometries are already Polygons, return them as-is."""
        polys = np.array(
            [
                shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ]
        )
        points = np.array(
            [
                shapely.Point(0.5, 0.5),
                shapely.Point(1.5, 0.5),
            ]
        )
        result = _resolve_multipolygons(polys, points)
        assert len(result) == 2
        for r, p in zip(result, polys, strict=True):
            assert r.equals(p)

    def test_multipolygon_resolved_to_containing_part(self):
        """A MultiPolygon should be resolved to the part containing the point."""
        part_a = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_b = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        multi = shapely.MultiPolygon([part_a, part_b])
        point_in_b = shapely.Point(5.5, 5.5)

        polys = np.array([multi])
        points = np.array([point_in_b])
        result = _resolve_multipolygons(polys, points)

        assert len(result) == 1
        assert isinstance(result[0], shapely.Polygon)
        assert result[0].equals(part_b)

    def test_point_outside_all_parts_raises(self):
        """A MultiPolygon where the point is outside all parts must raise.

        There is no safe 'nearest' guess: picking the wrong part would silently
        assign the Voronoi cell to the wrong region (e.g. the wrong side of a
        barrier), producing physically meaningless density values.
        """
        part_a = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_b = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        multi = shapely.MultiPolygon([part_a, part_b])
        # Point at (2, 0.5) is outside both parts
        point_outside = shapely.Point(2.0, 0.5)

        polys = np.array([multi])
        points = np.array([point_outside])

        with pytest.raises(PedPyValueError, match="does not lie within"):
            _resolve_multipolygons(polys, points)

    def test_geometry_collection_resolved(self):
        """A GeometryCollection containing polygons should be resolved."""
        part_a = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_b = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        gc = shapely.GeometryCollection([part_a, part_b])
        point_in_a = shapely.Point(0.5, 0.5)

        polys = np.array([gc])
        points = np.array([point_in_a])
        result = _resolve_multipolygons(polys, points)

        assert len(result) == 1
        assert isinstance(result[0], shapely.Polygon)
        assert result[0].equals(part_a)

    def test_mixed_array_only_non_polygons_resolved(self):
        """In a mixed array, only non-Polygon entries should be resolved.

        Polygon entries should remain unchanged.
        """
        simple_poly = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_a = shapely.Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        part_b = shapely.Polygon([(20, 20), (21, 20), (21, 21), (20, 21)])
        multi = shapely.MultiPolygon([part_a, part_b])

        polys = np.array([simple_poly, multi])
        points = np.array(
            [
                shapely.Point(0.5, 0.5),
                shapely.Point(20.5, 20.5),
            ]
        )
        result = _resolve_multipolygons(polys, points)

        assert result[0].equals(simple_poly)
        assert result[1].equals(part_b)

    def test_input_array_not_mutated(self):
        """Verify _resolve_multipolygons does not modify the input array."""
        part_a = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_b = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        multi = shapely.MultiPolygon([part_a, part_b])
        point_in_a = shapely.Point(0.5, 0.5)

        polys = np.array([multi])
        original = polys.copy()
        points = np.array([point_in_a])
        _resolve_multipolygons(polys, points)

        # Input should be unchanged
        assert polys[0].equals(original[0])

    def test_geometry_collection_without_polygon_parts_raises(self):
        """A GeometryCollection with no polygon parts must raise PedPyValueError.

        This indicates the trajectory position is outside the walkable area or
        inside an obstacle — silent fallback would hide a data integrity problem.
        """
        gc = shapely.GeometryCollection([shapely.LineString([(0, 0), (1, 1)]), shapely.Point(0.5, 0.5)])
        polys = np.array([gc])
        points = np.array([shapely.Point(0.5, 0.5)])

        with pytest.raises(PedPyValueError, match="no polygonal parts"):
            _resolve_multipolygons(polys, points)

    def test_point_outside_all_polygon_parts_raises(self):
        """A point not within any polygon part must raise PedPyValueError.

        Silently picking the nearest part would assign the Voronoi cell to the
        wrong side of a barrier, producing physically meaningless results.
        For example: a pedestrian leaning over an obstacle such that their
        recorded (x, y) position lands inside the obstacle, leaving their
        point outside all polygon parts of the split Voronoi cell.
        """
        part_a = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        part_b = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        multi = shapely.MultiPolygon([part_a, part_b])
        # Point is outside both parts
        point_outside = shapely.Point(10.0, 10.0)

        polys = np.array([multi])
        points = np.array([point_outside])

        with pytest.raises(PedPyValueError, match="does not lie within"):
            _resolve_multipolygons(polys, points)

    def test_geometry_collection_with_mixed_parts_ignores_non_polygon_parts(self):
        """Non-polygon parts in a GeometryCollection must not be selected.

        Only the polygon part should be a candidate even if the point lies
        exactly on a line part.
        """
        poly_part = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        line_part = shapely.LineString([(3, 0), (3, 2)])
        gc = shapely.GeometryCollection([line_part, poly_part])
        # Point is inside the polygon part (not on the line)
        point_in_poly = shapely.Point(1.0, 1.0)

        polys = np.array([gc])
        points = np.array([point_in_poly])
        result = _resolve_multipolygons(polys, points)

        assert len(result) == 1
        assert isinstance(result[0], shapely.Polygon)
        assert result[0].equals(poly_part)
