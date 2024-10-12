import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    DataValidationStatus,
    _compute_orthogonal_speed_in_relation_to_proprotion,
    _compute_partial_line_length,
    compute_crossing_frames,
    is_individual_speed_valid,
    is_species_valid,
)


def test_calc_n_correct_result():
    line = MeasurementLine([(0, 0), (1, 1)])
    expected_n = np.array([0.5**0.5, -(0.5**0.5)])
    actual_n = line.normal_vector()
    tolerance = 1e-8
    assert np.allclose(expected_n, actual_n, atol=tolerance)


@pytest.mark.parametrize(
    "line, polygon, expected",
    [
        (
            MeasurementLine([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)]),
            0.5,
        ),
        (
            MeasurementLine([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, -0.5), (0, -0.5)]),
            0,
        ),
    ],
)
def test_partial_line_length_correct(line, polygon, expected):
    actual = _compute_partial_line_length(polygon, line)
    assert expected == actual


def test_compute_orthogonal_speed_in_relation_to_proportion():
    v_x = 1
    v_y = 5
    poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    group = {V_X_COL: v_x, V_Y_COL: v_y, POLYGON_COL: poly}
    n = [0.5**0.5, -(0.5**0.5)]
    line = MeasurementLine([(0, 0), (1, 1)])
    assert np.allclose(n, line.normal_vector())
    actual = _compute_orthogonal_speed_in_relation_to_proprotion(
        group=group, measurement_line=line
    )
    assert _compute_partial_line_length(poly, line) == 0.5
    expected = (v_x * n[0] + v_y * n[1]) * 0.5
    assert np.isclose(actual, expected)


def test_is_species_valid_for_correct_species():
    species = pd.DataFrame(
        data={ID_COL: [1, 2, 3, 4, 5, 6], SPECIES_COL: [1, 1, 1, -1, -1, -1]}
    )
    intersecting_poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    non_intersecting_poly = Polygon([(0, -1), (1, -1), (1, -0.5), (0, -0.5)])
    voronoi_polys = pd.DataFrame(
        data={
            ID_COL: [0, 1, 2, 3, 7, 6, 5, 4],
            FRAME_COL: [1, 1, 1, 1, 2, 2, 2, 2],
            POLYGON_COL: [non_intersecting_poly] * 1
            + [intersecting_poly] * 3
            + [non_intersecting_poly] * 1
            + [intersecting_poly] * 3,
        }
    )
    measurement_line = MeasurementLine([(0, 0), (1, 1)])
    assert is_species_valid(
        species=species,
        individual_voronoi_polygons=voronoi_polys,
        measurement_line=measurement_line,
    )


def test_is_species_valid_for_incorrect_species():
    species = pd.DataFrame(
        data={ID_COL: [1, 2, 3, 4, 5, 6], SPECIES_COL: [1, 1, 1, -1, -1, -1]}
    )
    intersecting_poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    voronoi_polys = pd.DataFrame(
        data={
            ID_COL: [0, 1, 2, 3, 7, 6, 5, 4],
            FRAME_COL: [1, 1, 1, 1, 2, 2, 2, 2],
            POLYGON_COL: [intersecting_poly] * 8,
        }
    )
    measurement_line = MeasurementLine([(0, 0), (1, 1)])
    assert not is_species_valid(
        species=species,
        individual_voronoi_polygons=voronoi_polys,
        measurement_line=measurement_line,
    )


def test_is_speed_valid_for_correct_speed():
    speed = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2, 1, 2],
            FRAME_COL: [1, 1, 2, 2, 3, 3],
            SPEED_COL: [1, 2, 3, 4, 5, 6],
            V_X_COL: [1, 2, 3, 4, 5, 6],
            V_Y_COL: [1, 2, 3, 4, 5, 6],
        }
    )
    intersecting_poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    non_intersecting_poly = Polygon([(0, -1), (1, -1), (1, -0.5), (0, -0.5)])
    voronoi_polys = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2, 1, 2],
            FRAME_COL: [1, 1, 2, 2, 3, 3],
            POLYGON_COL: [non_intersecting_poly] * 1
            + [intersecting_poly] * 3
            + [non_intersecting_poly] * 1
            + [intersecting_poly] * 1,
        }
    )
    measurement_line = MeasurementLine([(0, 0), (1, 1)])
    assert (
        is_individual_speed_valid(
            individual_speed=speed,
            individual_voronoi_polygons=voronoi_polys,
            measurement_line=measurement_line,
        )
        == DataValidationStatus.DATA_CORRECT
    )


def test_is_speed_valid_for_missing_speed():
    speed = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2],
            FRAME_COL: [2, 2, 3, 3],
            SPEED_COL: [3, 4, 5, 6],
            V_X_COL: [3, 4, 5, 6],
            V_Y_COL: [3, 4, 5, 6],
        }
    )
    intersecting_poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    non_intersecting_poly = Polygon([(0, -1), (1, -1), (1, -0.5), (0, -0.5)])
    voronoi_polys = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2, 1, 2],
            FRAME_COL: [1, 1, 2, 2, 3, 3],
            POLYGON_COL: [non_intersecting_poly] * 1
            + [intersecting_poly] * 3
            + [non_intersecting_poly] * 1
            + [intersecting_poly] * 1,
        }
    )
    measurement_line = MeasurementLine([(0, 0), (1, 1)])
    assert (
        is_individual_speed_valid(
            individual_speed=speed,
            individual_voronoi_polygons=voronoi_polys,
            measurement_line=measurement_line,
        )
        == DataValidationStatus.ENTRY_MISSING
    )


def test_is_speed_valid_for_missing_velocity():
    speed = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2],
            FRAME_COL: [2, 2, 3, 3],
            SPEED_COL: [3, 4, 5, 6],
        }
    )
    intersecting_poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    non_intersecting_poly = Polygon([(0, -1), (1, -1), (1, -0.5), (0, -0.5)])
    voronoi_polys = pd.DataFrame(
        data={
            ID_COL: [1, 2, 1, 2, 1, 2],
            FRAME_COL: [1, 1, 2, 2, 3, 3],
            POLYGON_COL: [non_intersecting_poly] * 1
            + [intersecting_poly] * 3
            + [non_intersecting_poly] * 1
            + [intersecting_poly] * 1,
        }
    )
    measurement_line = MeasurementLine([(0, 0), (1, 1)])
    assert (
        is_individual_speed_valid(
            individual_speed=speed,
            individual_voronoi_polygons=voronoi_polys,
            measurement_line=measurement_line,
        )
        == DataValidationStatus.COLUMN_MISSING
    )


def create_single_straight_trajectory_crossing_zero_line(
    *, start_x=-10, frames_before, frames_on_line=0, end_x=10, frames_after
):
    x = np.concatenate(
        [
            np.linspace(
                start_x, -0.1 if frames_on_line == 0 else 0, frames_before
            ),
            np.linspace(0, 0, frames_on_line - 1 if frames_on_line > 0 else 0),
            np.linspace(0.1 if frames_on_line == 0 else 0, end_x, frames_after),
        ]
    )
    y = np.zeros_like(x)

    frames = np.arange(0, len(x))
    id = np.ones_like(x)

    df_traj_simple = pd.DataFrame({"id": id, "frame": frames, "x": x, "y": y})
    return TrajectoryData(data=df_traj_simple, frame_rate=1)


@pytest.mark.parametrize(
    "frames_before, frames_after",
    [
        (10, 5),
        (1, 10),
    ],
)
def test_compute_crossing_frame_movement_across_line(
    frames_before, frames_after
):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=frames_before, frames_after=frames_after
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])
    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 1
    assert crossing_frames.iloc[0].id == 1
    assert crossing_frames.iloc[0].frame == frames_before


@pytest.mark.parametrize(
    "frames_before, frames_on_line",
    [
        (3, 1),
        (10, 5),
        (7, 10),
        (0, 1),
        (0, 10),
    ],
)
def test_compute_crossing_frame_movement_stops_on_line(
    frames_before, frames_on_line
):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=frames_before,
        frames_after=5,
        frames_on_line=frames_on_line,
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])

    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 1
    assert crossing_frames.iloc[0].id == 1
    assert crossing_frames.iloc[0].frame == frames_before + frames_on_line


@pytest.mark.parametrize(
    "frames_on_line",
    [
        1,
        5,
        10,
    ],
)
def test_compute_crossing_frame_trajectory_ends_on_line(frames_on_line):
    traj = create_single_straight_trajectory_crossing_zero_line(
        frames_before=10,
        frames_after=0,
        frames_on_line=frames_on_line,
    )
    ml = MeasurementLine([(0, -0.5), (0, 0.5)])
    crossing_frames = compute_crossing_frames(
        traj_data=traj, measurement_line=ml
    )

    assert len(crossing_frames) == 0
