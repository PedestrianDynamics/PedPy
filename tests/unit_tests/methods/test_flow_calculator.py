import pathlib

from pedpy.data.geometry import WalkableArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.io.trajectory_loader import load_trajectory
from pedpy.methods.flow_calculator import calc_n, partial_line_length, weight_value, merge_table, separate_species, \
    calc_speed_on_line, calc_density_on_line, calc_flow_on_line

import pytest
from shapely import LineString, Polygon
import numpy as np
import pandas as pd

from pedpy.column_identifier import V_X_COL, V_Y_COL, POLYGON_COL, ID_COL, FRAME_COL, DENSITY_COL, SPECIES_COL, \
    SPEED_COL, V_SP1_COL, V_SP2_COL, VELOCITY_COL, DENSITY_SP1_COL, DENSITY_SP2_COL, FLOW_SP1_COL, \
    FLOW_SP2_COL, FLOW_COL
from pedpy.methods.method_utils import compute_individual_voronoi_polygons, Cutoff
from pedpy.methods.speed_calculator import compute_individual_speed
from tests.utils.utils import get_trajectory


def test_calc_n_correct_result():
    line = LineString([(0, 0), (1, 1)])
    expected_n = np.array([0.5 ** 0.5, -0.5 ** 0.5])
    actual_n = calc_n(line)
    tolerance = 1e-8
    assert (np.allclose(expected_n, actual_n, atol=tolerance))


def test_calc_n_with_line_length_zero():
    line = LineString([(0, 0), (0, 0)])
    actual_n = calc_n(line)
    assert (np.all(np.isnan(actual_n)))


@pytest.mark.parametrize(
    "line, polygon, expected",
    [
        (LineString([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)]), 0.5),
        (LineString([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, -0.5), (0, -0.5)]), 0)
    ]
)
def test_partial_line_length_correct(line, polygon, expected):
    actual = partial_line_length(polygon, line)
    assert (expected == actual)


def test_partial_line_length_with_line_length_zero():
    line = LineString([(0, 0), (0, 0)])
    polygon = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    actual = partial_line_length(polygon, line)
    assert (np.isnan(actual))


def test_weight_value():
    v_x = 1
    v_y = 5
    poly = Polygon([(0, 0), (1, 0), (1, 0.5), (0, 0.5)])
    group = {V_X_COL: v_x, V_Y_COL: v_y, POLYGON_COL: poly}
    n = [0.5 ** 0.5, -0.5 ** 0.5]
    line = LineString([(0, 0), (1, 1)])
    actual = weight_value(group=group, n=n, line=line)
    assert partial_line_length(poly, line) == 0.5
    expected = (v_x * n[0] + v_y * n[1]) * 0.5
    assert actual == expected


def test_merge_table_all_columns_included_with_speed():
    individual_voronoi = pd.DataFrame(columns=[ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL])
    species = pd.DataFrame(columns=[ID_COL, SPECIES_COL])
    line = LineString()
    individual_speed = pd.DataFrame(columns=[ID_COL, FRAME_COL, SPEED_COL, V_X_COL, V_Y_COL])
    merged = merge_table(individual_voronoi_polygons=individual_voronoi,
                         species=species, line=line,
                         individual_speed=individual_speed)
    for column in [ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL, SPECIES_COL, V_X_COL, V_Y_COL]:
        assert column in merged.columns


def test_merge_table_all_columns_included_without_speed():
    individual_voronoi = pd.DataFrame(columns=[ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL])
    species = pd.DataFrame(columns=[ID_COL, SPECIES_COL])
    line = LineString()
    merged = merge_table(individual_voronoi_polygons=individual_voronoi,
                         species=species, line=line, individual_speed=None)
    for column in [ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL, SPECIES_COL]:
        assert column in merged.columns


def test_merge_table_correct():
    species = pd.DataFrame({
        ID_COL: [1, 3, 4, 5],
        SPECIES_COL: [1, -1, 1, 2]
    })
    line = LineString([(0, 0), (1, 1)])
    matching_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    non_matching_poly = Polygon()
    individual_voronoi = pd.DataFrame({
        ID_COL: [1, 2, 3, 4],
        FRAME_COL: [5, 6, 7, 8],
        POLYGON_COL: [matching_poly, matching_poly, matching_poly, non_matching_poly],
        DENSITY_COL: [1, 2, 3, 4]
    })
    individual_speed = pd.DataFrame({
        ID_COL: [1, 2, 4, 6],
        FRAME_COL: [5, 6, 8, 10],
        SPEED_COL: [1, 2, 4, 6],
        V_X_COL: [1, 2, 4, 6],
        V_Y_COL: [1, 2, 4, 6]
    })
    merged = merge_table(individual_voronoi_polygons=individual_voronoi,
                         species=species, line=line, individual_speed=individual_speed)
    for elem in merged[ID_COL]:  # assert cases to not be included in merged table
        assert elem not in [3,  # not included in speed
                            4,  # non matching polygon
                            5,  # only in species included
                            6]  # only in speed included
    elem = merged[merged[ID_COL] == 1]  # default case
    assert elem.shape[0] == 1
    assert elem[SPECIES_COL].values[0] == 1 and elem[FRAME_COL].values[0] == 5 and elem[V_X_COL].values[0] == 1

    elem = merged[merged[ID_COL] == 2]  # is not included in species but should be merged
    assert np.isnan(elem[SPECIES_COL].values[0])


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

    n = calc_n(line.line)
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

    n = calc_n(line.line)
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
    individual_speed = compute_individual_speed(
        traj_data=traj,
        frame_step=10,
        compute_velocity=True,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      individual_speed=individual_speed)

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
    individual_speed = compute_individual_speed(
        traj_data=traj,
        frame_step=10,
        compute_velocity=True,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      individual_speed=individual_speed)

    suffixes = ('_expected', '_species')
    non_matching = expected_species.merge(actual_species, on="id", suffixes=suffixes, how="outer")
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[non_matching[columnnames[0]] != non_matching[columnnames[1]]]
    assert non_matching.shape[0] == 0


def test_separate_species_correct_amount_without_cutoff(bidirectional_setup):
    traj = bidirectional_setup["traj"]
    walkable_area = bidirectional_setup["walkable_area"]
    measurement_line = bidirectional_setup["measurement_line"]
    number_of_agents = bidirectional_setup["n_o_a"]

    individual_cutoff = compute_individual_voronoi_polygons(
        traj_data=traj,
        walkable_area=walkable_area,
    )
    individual_speed = compute_individual_speed(
        traj_data=traj,
        frame_step=10,
        compute_velocity=True,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      individual_speed=individual_speed)
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
    individual_speed = compute_individual_speed(
        traj_data=traj,
        frame_step=10,
        compute_velocity=True,
    )
    actual_species = separate_species(individual_voronoi_polygons=individual_cutoff,
                                      measurement_line=measurement_line,
                                      individual_speed=individual_speed)
    assert actual_species.shape[0] == 50
