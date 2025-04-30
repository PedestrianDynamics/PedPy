from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from shapely import Point, Polygon

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    DataValidationStatus,
    _compute_orthogonal_speed_in_relation_to_proportion,
    _compute_partial_line_length,
    compute_crossing_frames,
    compute_neighbor_distance,
    compute_neighbors,
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
    actual = _compute_orthogonal_speed_in_relation_to_proportion(
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


@pytest.fixture
def sample_voronoi_data():
    return pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [0, 0, 0, 0],  # All in the same frame
            POLYGON_COL: [
                Polygon([(0, 0), (1, 0), (0.5, 1)]),  # ID 1
                Polygon([(1, 0), (2, 0), (1.5, 1)]),  # ID 2
                Polygon([(0.5, 1), (1.5, 1), (1, 2)]),  # ID 3
                Polygon([(3, 3), (4, 3), (3.5, 4)]),  # ID 4 (isolated)
            ],
        }
    )


@pytest.mark.filterwarnings(
    "ignore:The parameter 'as_list=True' is deprecated and may change in a future version.*:DeprecationWarning"
)
def test_compute_neighbors_as_list(sample_voronoi_data: pd.DataFrame):
    result = compute_neighbors(sample_voronoi_data, as_list=True)

    expected = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [0, 0, 0, 0],
            NEIGHBORS_COL: [
                [2, 3],
                [1, 3],
                [1, 2],
                [],
            ],  # ID 4 has no neighbors
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(by=ID_COL).reset_index(drop=True), expected
    )


def test_compute_neighbors_single(sample_voronoi_data):
    result = compute_neighbors(sample_voronoi_data, as_list=False)

    expected = pd.DataFrame(
        {
            ID_COL: [1, 1, 2, 2, 3, 3],
            FRAME_COL: [0, 0, 0, 0, 0, 0],
            NEIGHBOR_ID_COL: [2, 3, 1, 3, 1, 2],
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(by=[ID_COL, NEIGHBOR_ID_COL]).reset_index(drop=True),
        expected,
    )


@pytest.mark.filterwarnings(
    "ignore:The parameter 'as_list=True' is deprecated and may change in a future version.*:DeprecationWarning"
)
def test_compute_neighbors_empty():
    empty_df = pd.DataFrame(columns=[ID_COL, FRAME_COL, POLYGON_COL])

    result_list = compute_neighbors(empty_df, as_list=True)
    result_single = compute_neighbors(empty_df, as_list=False)

    assert result_list.empty
    assert result_single.empty


@pytest.mark.filterwarnings(
    "ignore:The parameter 'as_list=True' is deprecated and may change in a future version.*:DeprecationWarning"
)
def test_compute_neighbors_single_pedestrian():
    df = pd.DataFrame(
        {
            ID_COL: [1],
            FRAME_COL: [0],
            POLYGON_COL: [Polygon([(0, 0), (1, 0), (0.5, 1)])],
        }
    )

    result_list = compute_neighbors(df, as_list=True)
    result_single = compute_neighbors(df, as_list=False)

    expected_list = pd.DataFrame(
        {ID_COL: [1], FRAME_COL: [0], NEIGHBORS_COL: [[]]}
    )
    expected_single = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, NEIGHBOR_ID_COL]
    ).astype({ID_COL: "int64", FRAME_COL: "int64", NEIGHBOR_ID_COL: "int64"})
    pd.testing.assert_frame_equal(result_list, expected_list)
    pd.testing.assert_frame_equal(result_single, expected_single)


@pytest.mark.filterwarnings(
    "ignore:The parameter 'as_list=True' is deprecated and may change in a future version.*:DeprecationWarning"
)
def test_compute_neighbors_multiple_frames():
    df = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [0, 0, 1, 1],  # Two frames
            POLYGON_COL: [
                Polygon([(0, 0), (1, 0), (0.5, 1)]),  # Frame 0
                Polygon([(1, 0), (2, 0), (1.5, 1)]),  # Frame 0
                Polygon([(3, 3), (4, 3), (3.5, 4)]),  # Frame 1
                Polygon([(4, 3), (5, 3), (4.5, 4)]),  # Frame 1
            ],
        }
    )

    result_list = compute_neighbors(df, as_list=True)

    expected_list = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [0, 0, 1, 1],
            NEIGHBORS_COL: [[2], [1], [4], [3]],
        }
    )

    pd.testing.assert_frame_equal(
        result_list.sort_values(by=ID_COL).reset_index(drop=True), expected_list
    )


@pytest.mark.filterwarnings(
    "ignore:The parameter 'as_list=True' is deprecated and may change in a future version.*:DeprecationWarning"
)
def test_compute_neighbors_no_touching():
    df = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            POLYGON_COL: [
                Polygon([(0, 0), (1, 0), (0.5, 1)]),
                Polygon([(2, 0), (3, 0), (2.5, 1)]),  # Not touching ID 1
            ],
        }
    )

    result_list = compute_neighbors(df, as_list=True)
    expected_list = pd.DataFrame(
        {ID_COL: [1, 2], FRAME_COL: [0, 0], NEIGHBORS_COL: [[], []]}
    )

    pd.testing.assert_frame_equal(result_list, expected_list)


def test_compute_neighbors_deprecation_warning():
    dummy_data = pd.DataFrame(columns=[ID_COL, FRAME_COL, POLYGON_COL])

    with pytest.warns(
        DeprecationWarning, match="The parameter 'as_list=True' is deprecated"
    ):
        compute_neighbors(dummy_data, as_list=True)


def test_compute_neighbor_distance():
    traj_data = TrajectoryData(
        data=pd.DataFrame(
            {
                ID_COL: [1, 2, 3],
                FRAME_COL: [0, 0, 0],
                X_COL: [0, 3, 6],
                Y_COL: [0, 4, 8],
            }
        ),
        frame_rate=1,
    )

    neighborhood = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            NEIGHBOR_ID_COL: [2, 3],
        }
    )

    result = compute_neighbor_distance(
        traj_data=traj_data, neighborhood=neighborhood
    )

    expected_result = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            NEIGHBOR_ID_COL: [2, 3],
            DISTANCE_COL: [
                5.0,
                5.0,
            ],  # Euclidean distances: sqrt(3^2 + 4^2) = 5
        }
    )

    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_compute_neighbor_distance_invalid_list_input():
    traj_data = MagicMock(spec=TrajectoryData)
    neighborhood = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            NEIGHBORS_COL: [[2], [3]],  # as_list=True adds this column
        }
    )

    with pytest.raises(
        ValueError,
        match="Cannot compute distance between neighbors with list-format data.",
    ):
        compute_neighbor_distance(
            traj_data=traj_data, neighborhood=neighborhood
        )


def test_compute_neighbor_distance_empty_input():
    traj_data = TrajectoryData(
        data=pd.DataFrame(columns=[ID_COL, FRAME_COL, X_COL, Y_COL]).astype(
            {X_COL: "float64", Y_COL: "float64"}
        ),
        frame_rate=1,
    )
    neighborhood = pd.DataFrame(columns=[ID_COL, FRAME_COL, NEIGHBOR_ID_COL])

    result = compute_neighbor_distance(
        traj_data=traj_data, neighborhood=neighborhood
    )

    expected_result = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, NEIGHBOR_ID_COL, DISTANCE_COL]
    )

    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


def test_compute_neighbor_distance_different_distances():
    traj_data = TrajectoryData(
        data=pd.DataFrame(
            {
                ID_COL: [1, 2, 3],
                FRAME_COL: [0, 0, 0],
                X_COL: [0, 3, 10],
                Y_COL: [0, 4, 10],
            }
        ),
        frame_rate=1,
    )

    neighborhood = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            NEIGHBOR_ID_COL: [2, 3],
        }
    )

    result = compute_neighbor_distance(
        traj_data=traj_data, neighborhood=neighborhood
    )

    expected_result = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            NEIGHBOR_ID_COL: [2, 3],
            DISTANCE_COL: [5.0, 9.21954445729],  # sqrt(7^2 + 6^2) = 9.21
        }
    )

    pd.testing.assert_frame_equal(
        result, expected_result, check_dtype=False, atol=1e-6
    )
