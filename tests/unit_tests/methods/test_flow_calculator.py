import pathlib

from pedpy.data.geometry import WalkableArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.flow_calculator import partial_line_length, weight_value, separate_species, \
    calc_speed_on_line, calc_density_on_line, calc_flow_on_line

import pytest
from shapely import LineString, Polygon
import numpy as np
import pandas as pd

from pedpy.column_identifier import *
from pedpy.methods.method_utils import compute_individual_voronoi_polygons, Cutoff
from pedpy.methods.speed_calculator import compute_individual_speed
from tests.utils.utils import get_trajectory


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
    actual = partial_line_length(polygon, line)
    assert (expected == actual)


def test_weight_value():
    v_x = 1
    v_y = 5
    poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    group = {V_X_COL: v_x, V_Y_COL: v_y, POLYGON_COL: poly}
    n = [0.5 ** 0.5, -0.5 ** 0.5]
    line = MeasurementLine([(0, 0), (1, 1)])
    assert (np.allclose(n, line.normal_vector()))
    actual = weight_value(group=group, measurement_line=line)
    assert partial_line_length(poly, line) == 0.5
    expected = (v_x * n[0] + v_y * n[1]) * 0.5
    assert np.isclose(actual, expected)


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
    return {"species": species, "speed": speed, "voronoi": voronoi, "line": line}


def test_calc_speed_on_line(example_data):
    species = example_data["species"]
    speed = example_data["speed"]
    voronoi = example_data["voronoi"]
    line = example_data["line"]

    speed_on_line = calc_speed_on_line(individual_speed=speed, species=species,
                                       individual_voronoi_polygons=voronoi, measurement_line=line)

    n = line.normal_vector()
    assert speed_on_line.shape[0] == 1
    assert speed_on_line[V_SP1_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 0.5)
    assert speed_on_line[V_SP2_COL].values[0] == pytest.approx((n[0] * -1 + n[1] * -1) * 0.5 * -1)

    assert speed_on_line[VELOCITY_COL].values[0] == pytest.approx(((n[0] * 1 + n[1] * 1) * 0.5) +
                                                                  (n[0] * -1 + n[1] * -1) * 0.5 * -1)


def test_calc_density_on_line(example_data):
    species = example_data["species"]
    voronoi = example_data["voronoi"]
    line = example_data["line"]

    desity_on_line = calc_density_on_line(individual_voronoi_polygons=voronoi, measurement_line=line, species=species)

    assert desity_on_line.shape[0] == 1
    assert desity_on_line[DENSITY_SP1_COL].values[0] == pytest.approx(3 * 0.5)
    assert desity_on_line[DENSITY_SP2_COL].values[0] == pytest.approx(3 * 0.5)

    assert desity_on_line[DENSITY_COL].values[0] == pytest.approx(3)


def test_calc_flow_on_line(example_data):
    species = example_data["species"]
    speed = example_data["speed"]
    voronoi = example_data["voronoi"]
    line = example_data["line"]

    flow_on_line = calc_flow_on_line(individual_voronoi_polygons=voronoi, measurement_line=line,
                                     species=species, individual_speed=speed)

    n = line.normal_vector()
    assert flow_on_line.shape[0] == 1
    assert flow_on_line[FLOW_SP1_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 0.5 * 3)
    assert flow_on_line[FLOW_SP2_COL].values[0] == pytest.approx((n[0] * -1 + n[1] * -1) * 0.5 * -1 * 3)

    assert flow_on_line[FLOW_COL].values[0] == pytest.approx((n[0] * 1 + n[1] * 1) * 3)


@pytest.fixture
def bidirectional_setup(bidirectional_traj, bidirectional_walkable_area, bidirectional_measurement_line):
    return {
        "traj": bidirectional_traj,
        "walkable_area": bidirectional_walkable_area,
        "measurement_line": bidirectional_measurement_line,
    }


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
        ped_distance=2.0
    )
    traj_down[ID_COL] += 25
    return TrajectoryData(
        data=pd.concat([traj_up, traj_down], ignore_index=True),
        frame_rate=10.0
    )


@pytest.fixture
def bidirectional_measurement_line():
    return MeasurementLine([(-0.5, 25), (10.5, 25)])


@pytest.fixture
def bidirectional_walkable_area():
    return WalkableArea([(-0.5, -0.5), (10.5, -0.5), (10.5, 50.5), (-0.5, 50.5)])


def test_separate_species_correct_with_cutoff(bidirectional_setup):
    traj = bidirectional_setup["traj"]
    walkable_area = bidirectional_setup["walkable_area"]
    measurement_line = bidirectional_setup["measurement_line"]

    min_idx = traj.data.groupby(ID_COL)[FRAME_COL].idxmin()
    expected_species = traj.data.loc[min_idx, [ID_COL]]
    expected_species[SPECIES_COL] = np.where(expected_species[ID_COL] < 25, -1, 1)

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=0.8, quad_segments=3)
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      frame_step=10,
                                      traj=traj)

    suffixes = ('_expected', '_species')
    non_matching = expected_species.merge(actual_species, on="id", suffixes=suffixes)
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[non_matching[columnnames[0]] != non_matching[columnnames[1]]]
    assert non_matching.shape[0] == 0


def test_separate_species_correct_without_cutoff(bidirectional_setup):
    traj = bidirectional_setup["traj"]
    walkable_area = bidirectional_setup["walkable_area"]
    measurement_line = bidirectional_setup["measurement_line"]

    min_idx = traj.data.groupby(ID_COL)[FRAME_COL].idxmin()
    expected_species = traj.data.loc[min_idx, [ID_COL]]
    expected_species[SPECIES_COL] = np.where(expected_species[ID_COL] < 25, -1, 1)

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      frame_step=10,
                                      traj=traj)

    suffixes = ('_expected', '_species')
    non_matching = expected_species.merge(actual_species, on="id", suffixes=suffixes, how="outer")
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[non_matching[columnnames[0]] != non_matching[columnnames[1]]]
    assert non_matching.shape[0] == 0


def test_separate_species_correct_amount_without_cutoff(bidirectional_setup):
    traj = bidirectional_setup["traj"]
    walkable_area = bidirectional_setup["walkable_area"]
    measurement_line = bidirectional_setup["measurement_line"]

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      frame_step=10,
                                      traj=traj)
    assert actual_species.shape[0] == 50


def test_separate_species_correct_amount_with_cutoff(bidirectional_setup):
    traj = bidirectional_setup["traj"]
    walkable_area = bidirectional_setup["walkable_area"]
    measurement_line = bidirectional_setup["measurement_line"]

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
        cut_off=Cutoff(radius=0.8, quad_segments=3)
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      frame_step=10,
                                      traj=traj)
    assert actual_species.shape[0] == 50


@pytest.fixture
def non_intersecting_setup(non_intersecting_traj, non_intersecting_walkable_area, non_intersecting_measurement_line):
    return {
        "traj": non_intersecting_traj,
        "walkable_area": non_intersecting_walkable_area,
        "measurement_line": non_intersecting_measurement_line,
    }


@pytest.fixture
def non_intersecting_traj():
    data_t1 = get_trajectory(shape=[3, 1], number_frames=60, start_position=np.array([9.5, 1.5]),
                             movement_direction=np.array([-0.1, 0.0]), ped_distance=1.0)
    data_t1.loc[data_t1[ID_COL] == 1, X_COL] -= 0.7
    data_t1[ID_COL] += 1
    data_2_t1 = get_trajectory(shape=[1, 1], number_frames=24, start_position=np.array([9.5, 2.5]),
                               movement_direction=np.array([-0.1, 0.0]), ped_distance=1.0)
    data_3_t1 = get_trajectory(shape=[1, 1], number_frames=30, start_position=np.array([7.3, 2.5]),
                               movement_direction=np.array([+0.1, 0.0]), ped_distance=1.0)
    data_3_t1[FRAME_COL] += 24

    final_data = pd.concat([data_t1, data_2_t1, data_3_t1], ignore_index=True)
    return TrajectoryData(data=final_data, frame_rate=10)


@pytest.fixture
def non_intersecting_measurement_line():
    return MeasurementLine([(7.0, 0.0), (7.0, 5.0)])


@pytest.fixture
def non_intersecting_walkable_area():
    return WalkableArea([(2, 0), (12, 0), (12, 5), (2, 5)])


def test_separate_species_correct_for_not_intersecting(non_intersecting_setup):
    traj = non_intersecting_setup["traj"]
    walkable_area = non_intersecting_setup["walkable_area"]
    measurement_line = non_intersecting_setup["measurement_line"]
    individual_cutoff = compute_individual_voronoi_polygons(traj_data=traj,
                                                            walkable_area=walkable_area,
                                                            cut_off=Cutoff(radius=1.0, quad_segments=3))
    expected = pd.DataFrame(data=[[0, -1], [1, -1], [2, -1], [3, -1]], columns=[ID_COL, SPECIES_COL])
    actual = separate_species(individual_voronoi_polygons=individual_cutoff,
                              measurement_line=measurement_line,
                              frame_step=4,
                              traj=traj)

    suffixes = ("expected", "actual")
    non_matching = expected.merge(actual, on="id", suffixes=suffixes, how="outer")
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[non_matching[columnnames[0]] != non_matching[columnnames[1]]]
    assert non_matching.shape[0] == 0
