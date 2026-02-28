"""Unit tests for compute_individual_voronoi_polygons (old implementation).

These tests are compatible with the original scipy-based implementation
and can be backported to verify behavior before the refactoring.

Excluded from the main test file:
- Tests for _resolve_multipolygons (function did not exist)
- Tests for empty-result handling (old code crashed with ValueError)
- Tests for cutoff + non-convex (old multipolygon handling was fragile)
"""

import logging

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
from pedpy.methods.method_utils import (
    Cutoff,
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
        union = shapely.unary_union(result[POLYGON_COL].values)
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


# ===========================================================================
# 8. Blind points behavior (use_blind_points)
# ===========================================================================
class TestBlindPoints:
    """Test behavior with use_blind_points on/off."""

    def test_single_ped_with_blind_points_returns_result(self, single_ped_one_frame):
        """With blind points enabled, a single ped should produce a result."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=True)
        assert set(result[ID_COL]) == {0}
        assert set(result[FRAME_COL]) == {0}

    def test_single_ped_without_blind_points_raises(self, single_ped_one_frame):
        """Without blind points, a single ped crashes (no frames processed)."""
        traj, wa = single_ped_one_frame
        with pytest.raises(ValueError, match="No objects to concatenate"):
            compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=False)

    def test_two_peds_without_blind_points_raises(self, two_peds_one_frame):
        """Without blind points, two peds crashes (no frames processed)."""
        traj, wa = two_peds_one_frame
        with pytest.raises(ValueError, match="No objects to concatenate"):
            compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=False)

    def test_three_peds_without_blind_points_raises(self, square_walkable_area):
        """Without blind points, three peds crashes (no frames processed)."""
        traj = _make_traj(
            ids=[0, 1, 2],
            frames=[0, 0, 0],
            xs=[2.0, 8.0, 5.0],
            ys=[2.0, 2.0, 8.0],
        )
        with pytest.raises(ValueError, match="No objects to concatenate"):
            compute_individual_voronoi_polygons(
                traj_data=traj, walkable_area=square_walkable_area, use_blind_points=False
            )

    def test_four_peds_without_blind_points_returns_result(self, four_peds_one_frame):
        """With 4+ peds, blind_points=False should still produce results."""
        traj, wa = four_peds_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=False)
        assert set(result[ID_COL]) == {0, 1, 2, 3}
        assert (result[FRAME_COL] == 0).all()

    def test_blind_points_do_not_affect_results_with_many_peds(self, four_peds_one_frame):
        """With 4+ peds, results should be the same with or without blind points."""
        traj, wa = four_peds_one_frame
        result_bp_on = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=True)
        result_bp_off = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=False)
        areas_on = shapely.area(result_bp_on.sort_values(ID_COL).reset_index(drop=True)[POLYGON_COL].values)
        areas_off = shapely.area(result_bp_off.sort_values(ID_COL).reset_index(drop=True)[POLYGON_COL].values)
        np.testing.assert_allclose(areas_on, areas_off, atol=1e-6)

    def test_without_blind_points_skips_frame_with_few_peds_keeps_other_frames(self, square_walkable_area):
        """Frames with <4 peds are skipped, frames with >=4 are kept."""
        traj = _make_traj(
            ids=[0, 1, 0, 1, 2, 3],
            frames=[0, 0, 1, 1, 1, 1],
            xs=[3.0, 7.0, 2.5, 7.5, 2.5, 7.5],
            ys=[5.0, 5.0, 2.5, 2.5, 7.5, 7.5],
        )
        result = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=square_walkable_area, use_blind_points=False
        )
        # Frame 0 has 2 peds -> skipped; Frame 1 has 4 peds -> kept
        assert set(result[FRAME_COL]) == {1}
        assert set(result[ID_COL]) == {0, 1, 2, 3}

    def test_skipped_frames_emit_warning(self, square_walkable_area, caplog):
        """Verify a warning is logged when frames are skipped.

        Uses a scenario where only some frames are skipped (not all),
        so the old implementation does not crash on pd.concat([]).
        """
        traj = _make_traj(
            ids=[0, 1, 0, 1, 2, 3],
            frames=[0, 0, 1, 1, 1, 1],
            xs=[3.0, 7.0, 2.5, 7.5, 2.5, 7.5],
            ys=[5.0, 5.0, 2.5, 2.5, 7.5, 7.5],
        )
        with caplog.at_level(logging.WARNING):
            compute_individual_voronoi_polygons(
                traj_data=traj, walkable_area=square_walkable_area, use_blind_points=False
            )
        assert any("Not enough pedestrians" in msg for msg in caplog.messages)


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
    """With blind points and one ped, the polygon should cover the walkable area."""

    def test_single_ped_polygon_covers_walkable_area(self, single_ped_one_frame):
        """Verify single ped's polygon covers the full walkable area."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=True)
        poly = result.iloc[0][POLYGON_COL]
        np.testing.assert_allclose(poly.area, wa.polygon.area, atol=1e-6)

    def test_single_ped_density(self, single_ped_one_frame):
        """Verify single ped density equals 1 / walkable area."""
        traj, wa = single_ped_one_frame
        result = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=wa, use_blind_points=True)
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
        union = shapely.unary_union(result[POLYGON_COL].values)
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

    Blind points should handle this gracefully.
    """

    def test_collinear_peds_with_blind_points(self, square_walkable_area):
        """Verify collinear peds produce valid polygons with blind points."""
        traj = _make_traj(
            ids=[0, 1, 2, 3, 4],
            frames=[0, 0, 0, 0, 0],
            xs=[1.0, 3.0, 5.0, 7.0, 9.0],
            ys=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        result = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=square_walkable_area, use_blind_points=True
        )
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
        result = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=square_walkable_area, use_blind_points=True
        )
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
# 21. Multipolygon resolution (integration test only)
# ===========================================================================
class TestMultipolygonResolution:
    """Test geometries that produce non-Polygon results after clipping.

    Only integration tests are included here; direct _resolve_multipolygons
    tests are not applicable to the old implementation.
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
