import numpy as np
import pandas as pd
import pytest
from shapely import Polygon
from tests.utils.utils import get_trajectory

from pedpy.column_identifier import *
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    Cutoff,
    InputError,
    SpeedCalculation,
    compute_individual_voronoi_polygons,
    compute_intersecting_polygons,
)
from pedpy.methods.speed_calculator import (
    SpeedError,
    compute_individual_speed,
    compute_line_speed,
    compute_mean_speed_per_frame,
    compute_species,
    compute_voronoi_speed,
)


def test_mean_speed_needs_same_length_speed_and_polygon_data():
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[5, 5],
            number_frames=100,
            start_position=np.array([-1, -1]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )

    walkable_area = WalkableArea(
        [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    )
    measurement_area = MeasurementArea([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])

    speed = compute_individual_speed(
        traj_data=traj_data,
        frame_step=5,
        speed_calculation=SpeedCalculation.BORDER_EXCLUDE,
    )

    assert len(speed.index) != len(traj_data.data.index)
    with pytest.raises(SpeedError, match=".*mean speed.*"):
        compute_mean_speed_per_frame(
            traj_data=traj_data,
            individual_speed=speed,
            measurement_area=measurement_area,
        )


def test_voronoi_speed_needs_same_length_speed_and_polygon_data():
    traj_data = TrajectoryData(
        data=get_trajectory(
            shape=[5, 5],
            number_frames=100,
            start_position=np.array([-1, -1]),
            movement_direction=np.array([0, 0.1]),
            ped_distance=1.0,
        ),
        frame_rate=25.0,
    )

    walkable_area = WalkableArea(
        [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    )
    measurement_area = MeasurementArea([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)])

    speed = compute_individual_speed(
        traj_data=traj_data,
        frame_step=5,
        speed_calculation=SpeedCalculation.BORDER_EXCLUDE,
    )

    polygons = compute_individual_voronoi_polygons(
        traj_data=traj_data,
        walkable_area=walkable_area,
    )
    intersection = compute_intersecting_polygons(
        individual_voronoi_data=polygons, measurement_area=measurement_area
    )

    assert len(speed.index) != len(intersection.index)
    with pytest.raises(SpeedError, match=".*Voronoi speed.*"):
        compute_voronoi_speed(
            traj_data=traj_data,
            individual_speed=speed,
            individual_voronoi_intersection=intersection,
            measurement_area=measurement_area,
        )


@pytest.fixture
def example_data():
    species = pd.DataFrame(
        {ID_COL: [1, 2, 3, 4], SPECIES_COL: [1, -1, np.nan, 1]}
    )
    speed = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [3, 3, 3, 3],
            SPEED_COL: [0, 0, 0, 0],
            V_X_COL: [1, -1, 5, -5],
            V_Y_COL: [1, -1, 5, -5],
        }
    )
    line = MeasurementLine([(2, 0), (0, 2)])
    matching_poly1 = Polygon([(0, 0), (2, 2), (1, 3), (-1, 1)])
    matching_poly2 = Polygon([(1, -1), (3, 1), (2, 2), (0, 0)])
    non_matching_poly = Polygon()
    voronoi = pd.DataFrame(
        {
            ID_COL: [1, 2, 3, 4],
            FRAME_COL: [3, 3, 3, 3],
            POLYGON_COL: [
                matching_poly1,
                matching_poly2,
                non_matching_poly,
                non_matching_poly,
            ],
            DENSITY_COL: [3, 3, 3, 3],
        }
    )
    return species, speed, voronoi, line


def test_compute_line_speed(example_data):
    species, speed, voronoi, line = example_data

    speed_on_line = compute_line_speed(
        individual_speed=speed,
        species=species,
        individual_voronoi_polygons=voronoi,
        measurement_line=line,
    )

    n = line.normal_vector()
    assert speed_on_line.shape[0] == 1
    assert speed_on_line[SPEED_SP1_COL].values[0] == pytest.approx(
        (n[0] * 1 + n[1] * 1) * 0.5
    )
    assert speed_on_line[SPEED_SP2_COL].values[0] == pytest.approx(
        (n[0] * -1 + n[1] * -1) * 0.5 * -1
    )

    assert speed_on_line[SPEED_COL].values[0] == pytest.approx(
        ((n[0] * 1 + n[1] * 1) * 0.5) + (n[0] * -1 + n[1] * -1) * 0.5 * -1
    )


@pytest.fixture
def bidirectional_setup(
    bidirectional_traj,
    bidirectional_walkable_area,
    bidirectional_measurement_line,
):
    return (
        bidirectional_traj,
        bidirectional_walkable_area,
        bidirectional_measurement_line,
    )


@pytest.fixture
def bidirectional_traj():
    traj_up = get_trajectory(
        shape=[5, 5],
        number_frames=200,
        start_position=np.array([0, 0]),
        movement_direction=np.array([0, 0.2]),
        ped_distance=2.0,
    )
    traj_down = get_trajectory(
        shape=[5, 5],
        number_frames=200,
        start_position=np.array([1, 40]),
        movement_direction=np.array([0, -0.2]),
        ped_distance=2.0,
    )
    traj_down[ID_COL] += 25
    return TrajectoryData(
        data=pd.concat([traj_up, traj_down], ignore_index=True), frame_rate=10.0
    )


@pytest.fixture
def bidirectional_measurement_line():
    return MeasurementLine([(-0.5, 25), (10.5, 25)])


@pytest.fixture
def bidirectional_walkable_area():
    return WalkableArea(
        [(-0.5, -0.5), (10.5, -0.5), (10.5, 50.5), (-0.5, 50.5)]
    )


def test_compute_species_correct_with_cutoff(bidirectional_setup):
    traj, walkable_area, measurement_line = bidirectional_setup

    min_idx = traj.data.groupby(ID_COL)[FRAME_COL].idxmin()
    expected_species = traj.data.loc[min_idx, [ID_COL]]
    expected_species[SPECIES_COL] = np.where(
        expected_species[ID_COL] < 25, -1, 1
    )

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=0.8, quad_segments=3),
    )
    actual_species = compute_species(
        individual_voronoi_polygons=individual_cutoff,
        measurement_line=measurement_line,
        frame_step=10,
        trajectory_data=traj,
    )

    suffixes = ("_expected", "_species")
    non_matching = expected_species.merge(
        actual_species, on="id", suffixes=suffixes
    )
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[
        non_matching[columnnames[0]] != non_matching[columnnames[1]]
    ]
    assert non_matching.shape[0] == 0


def test_compute_species_correct_without_cutoff(bidirectional_setup):
    traj, walkable_area, measurement_line = bidirectional_setup

    min_idx = traj.data.groupby(ID_COL)[FRAME_COL].idxmin()
    expected_species = traj.data.loc[min_idx, [ID_COL]]
    expected_species[SPECIES_COL] = np.where(
        expected_species[ID_COL] < 25, -1, 1
    )

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    actual_species = compute_species(
        individual_voronoi_polygons=individual_cutoff,
        measurement_line=measurement_line,
        frame_step=10,
        trajectory_data=traj,
    )

    suffixes = ("_expected", "_species")
    non_matching = expected_species.merge(
        actual_species, on="id", suffixes=suffixes, how="outer"
    )
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[
        non_matching[columnnames[0]] != non_matching[columnnames[1]]
    ]
    assert non_matching.shape[0] == 0


def test_compute_species_correct_amount_without_cutoff(bidirectional_setup):
    traj, walkable_area, measurement_line = bidirectional_setup

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    actual_species = compute_species(
        individual_voronoi_polygons=individual_cutoff,
        measurement_line=measurement_line,
        frame_step=10,
        trajectory_data=traj,
    )
    assert actual_species.shape[0] == 50


def test_compute_species_correct_amount_with_cutoff(bidirectional_setup):
    traj, walkable_area, measurement_line = bidirectional_setup

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=0.8, quad_segments=3),
    )
    actual_species = compute_species(
        individual_voronoi_polygons=individual_cutoff,
        measurement_line=measurement_line,
        frame_step=10,
        trajectory_data=traj,
    )
    assert actual_species.shape[0] == 50


@pytest.fixture
def non_intersecting_setup(
    non_intersecting_traj,
    non_intersecting_walkable_area,
    non_intersecting_measurement_line,
):
    return (
        non_intersecting_traj,
        non_intersecting_walkable_area,
        non_intersecting_measurement_line,
    )


@pytest.fixture
def non_intersecting_traj():
    data_t1 = get_trajectory(
        shape=[3, 1],
        number_frames=60,
        start_position=np.array([9.5, 1.5]),
        movement_direction=np.array([-0.1, 0.0]),
        ped_distance=1.0,
    )
    data_t1.loc[data_t1[ID_COL] == 1, X_COL] -= 0.7
    data_t1[ID_COL] += 1
    data_2_t1 = get_trajectory(
        shape=[1, 1],
        number_frames=24,
        start_position=np.array([9.5, 2.5]),
        movement_direction=np.array([-0.1, 0.0]),
        ped_distance=1.0,
    )
    data_3_t1 = get_trajectory(
        shape=[1, 1],
        number_frames=30,
        start_position=np.array([7.3, 2.5]),
        movement_direction=np.array([+0.1, 0.0]),
        ped_distance=1.0,
    )
    data_3_t1[FRAME_COL] += 24

    final_data = pd.concat([data_t1, data_2_t1, data_3_t1], ignore_index=True)
    return TrajectoryData(data=final_data, frame_rate=10)


@pytest.fixture
def non_intersecting_measurement_line():
    return MeasurementLine([(7.0, 0.0), (7.0, 5.0)])


@pytest.fixture
def non_intersecting_walkable_area():
    return WalkableArea([(2, 0), (12, 0), (12, 5), (2, 5)])


def test_compute_species_correct_for_not_intersecting(non_intersecting_setup):
    traj, walkable_area, measurement_line = non_intersecting_setup

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=1.0, quad_segments=3),
    )
    expected = pd.DataFrame(
        data=[[0, -1], [1, -1], [2, -1], [3, -1]], columns=[ID_COL, SPECIES_COL]
    )
    actual = compute_species(
        individual_voronoi_polygons=individual_cutoff,
        measurement_line=measurement_line,
        frame_step=4,
        trajectory_data=traj,
    )

    suffixes = ("expected", "actual")
    non_matching = expected.merge(
        actual, on="id", suffixes=suffixes, how="outer"
    )
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[
        non_matching[columnnames[0]] != non_matching[columnnames[1]]
    ]
    assert non_matching.shape[0] == 0


def test_compute_line_speed_invalid_species(example_data):
    species, speed, voronoi, line = example_data
    # Modify species to make it invalid
    invalid_species = species.copy()
    invalid_species.iloc[0, 0] = 999  # Add invalid pedestrian ID

    with pytest.raises(InputError, match="Species data validation failed."):
        compute_line_speed(
            individual_speed=speed,
            species=invalid_species,
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
        )


def test_compute_line_speed_missing_speed_entries(example_data):
    species, speed, voronoi, line = example_data
    # Remove some speed entries
    invalid_speed = speed.copy()
    invalid_speed = invalid_speed.iloc[1:]  # Remove first row

    with pytest.raises(InputError, match="Missing speed data entries"):
        compute_line_speed(
            individual_speed=invalid_speed,
            species=species,
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
        )


def test_compute_line_speed_missing_velocity_columns(example_data):
    species, speed, voronoi, line = example_data
    # Remove velocity columns
    invalid_speed = speed.copy()
    invalid_speed = invalid_speed.drop(
        [V_X_COL, V_Y_COL], axis=1
    )  # Assuming these are your velocity columns

    with pytest.raises(InputError, match="Required velocity columns missing"):
        compute_line_speed(
            individual_speed=invalid_speed,
            species=species,
            individual_voronoi_polygons=voronoi,
            measurement_line=line,
        )
