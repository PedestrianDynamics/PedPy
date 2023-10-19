import pathlib

from pedpy.data.geometry import WalkableArea, MeasurementLine
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.io.trajectory_loader import load_trajectory
from pedpy.methods.flow_calculator import calc_n, partial_line_length, weight_value, merge_table, separate_species

import pytest
from shapely import LineString, Polygon
import numpy as np
import pandas as pd

from pedpy.column_identifier import V_X_COL, V_Y_COL, POLYGON_COL, ID_COL, FRAME_COL, DENSITY_COL, SPECIES_COL, \
    SPEED_COL, POINT_COL
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



def test_separate_species_with_cutoff():
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
    traj = TrajectoryData(
        data=pd.concat([traj_up, traj_down]),
        frame_rate=10.0
    )
    walkable_area = WalkableArea([(-0.5, -0.5), (10.5, -0.5), (10.5, 50.5), (-0.5, 50.5)])
    measurement_line = MeasurementLine([(-0.5, 25), (10.5, 25)])

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


def test_separate_species_without_cutoff():
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
    traj = TrajectoryData(
        data=pd.concat([traj_up, traj_down]),
        frame_rate=10.0
    )
    walkable_area = WalkableArea([(-0.5, -0.5), (10.5, -0.5), (10.5, 50.5), (-0.5, 50.5)])
    measurement_line = MeasurementLine([(-0.5, 25), (10.5, 25)])

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
    non_matching = expected_species.merge(actual_species, on="id", suffixes=suffixes)
    columnnames = (SPECIES_COL + suffixes[0], SPECIES_COL + suffixes[1])
    non_matching = non_matching[non_matching[columnnames[0]] != non_matching[columnnames[1]]]
    assert non_matching.shape[0] == 0
