import pathlib

import numpy as np
import pytest
import shapely

from pedpy import InputError
from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import TrajectoryUnit, load_trajectory
from pedpy.methods.method_utils import is_trajectory_valid
from pedpy.preprocessing.trajectory_projector import (
    WallType,
    _calculate_movement_wall,
    _move_from_wall,
    _point_valid_test,
    _wall_facing_zero_point,
    correct_invalid_trajectories,
)


@pytest.fixture
def setup_walkable_area():
    walk_area = WalkableArea(
        [
            (3.5, -2),
            (3.5, 8),
            (-3.5, 8),
            (-3.5, -2),
        ],
        obstacles=[
            [
                (-0.7, -1.1),
                (-0.25, -1.1),
                (-0.25, -0.15),
                (-0.4, 0.0),
                (-2.8, 0.0),
                (-2.8, 6.7),
                (-3.05, 6.7),
                (-3.05, -0.3),
                (-0.7, -0.3),
                (-0.7, -1.0),
            ],
            [
                (0.25, -1.1),
                (0.7, -1.1),
                (0.7, -0.3),
                (3.05, -0.3),
                (3.05, 6.7),
                (2.8, 6.7),
                (2.8, 0.0),
                (0.4, 0.0),
                (0.25, -0.15),
                (0.25, -1.1),
            ],
        ],
    )
    return walk_area


@pytest.fixture
def setup_trajectory_data():
    trajectory_data = load_trajectory(
        trajectory_file=pathlib.Path(pathlib.Path(__file__).parent.parent / "io/test-data/030_c_56_h0.txt"),
        default_unit=TrajectoryUnit.METER,
    )
    return trajectory_data


def test_correct_invalid_trajectories(setup_trajectory_data, setup_walkable_area):
    assert not (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area))
    corrected_trajectory_data, invalid_person_ids = correct_invalid_trajectories(
        trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area
    )
    assert is_trajectory_valid(traj_data=corrected_trajectory_data, walkable_area=setup_walkable_area) == True
    assert np.array_equal(
        invalid_person_ids,
        np.array(
            [
                4,
                5,
                6,
                7,
                9,
                10,
                13,
                15,
                23,
                25,
                27,
                28,
                29,
                34,
                36,
                37,
                40,
                41,
                42,
                43,
                48,
                49,
                52,
                53,
                56,
                63,
                65,
                66,
                67,
                70,
                71,
                73,
            ]
        ),
    )


def test_correct_invalid_trajectories_min_parameter(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False
    corrected_trajectory_data, invalid_person_ids = correct_invalid_trajectories(
        trajectory_data=setup_trajectory_data,
        walkable_area=setup_walkable_area,
        back_distance_wall=-1.5,
        min_distance_wall=0.01,
        max_distance_wall=0.01,
        back_distance_obst=-1.5,
        min_distance_obst=0.01,
        max_distance_obst=0.01,
    )
    assert is_trajectory_valid(traj_data=corrected_trajectory_data, walkable_area=setup_walkable_area) == True
    assert np.array_equal(
        invalid_person_ids,
        np.array([4, 5, 6, 7, 9, 10, 13, 23, 25, 27, 28, 34, 37, 41, 42, 43, 48, 49, 52, 53, 56, 65, 71, 73]),
    )


def test_correct_invalid_trajectories_invalid_back_distance_wall(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False

    with pytest.raises(InputError, match=r"back_distance_wall has to be <0, is currently 0"):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area, back_distance_wall=0
        )


def test_correct_invalid_trajectories_invalid_back_distance_obst(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False

    with pytest.raises(InputError, match=r"back_distance_obst has to be <0, is currently 1"):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area, back_distance_obst=1
        )


def test_correct_invalid_trajectories_invalid_min_distance_obst(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False

    with pytest.raises(
        InputError,
        match=r"min_distance_obst and max_distance_obst have to be > 0 and "
        "max_distance_obst has to be >=  min_distance_obst",
    ):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data,
            walkable_area=setup_walkable_area,
            back_distance_wall=-0.5,
            min_distance_wall=0.01,
            max_distance_wall=0.04,
            back_distance_obst=-0.5,
            min_distance_obst=0.04,
            max_distance_obst=0.01,
        )


def test_correct_invalid_trajectories_invalid_min_distance_wall(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False

    with pytest.raises(
        InputError,
        match=r"min_distance_wall and max_distance_wall have to be > 0 and "
        "max_distance_wall has to be >=  min_distance_wall",
    ):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data,
            walkable_area=setup_walkable_area,
            back_distance_wall=-0.5,
            min_distance_wall=0.08,
            max_distance_wall=0.04,
        )


def test_correct_invalid_trajectories_invalid_max_distance(setup_trajectory_data, setup_walkable_area):
    assert (is_trajectory_valid(traj_data=setup_trajectory_data, walkable_area=setup_walkable_area)) == False

    with pytest.raises(
        InputError,
        match=r"max_distance is too large: shifting points away from one wall brings them too close to the opposite wall.",
    ):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data,
            walkable_area=setup_walkable_area,
            back_distance_wall=-0.5,
            min_distance_wall=0.01,
            max_distance_wall=0.04,
            back_distance_obst=-0.5,
            min_distance_obst=0.04,
            max_distance_obst=0.5,
        )


def test_correct_invalid_trajectories_valid_data(setup_walkable_area):
    trajectory_data = load_trajectory(
        trajectory_file=pathlib.Path(pathlib.Path(__file__).parent.parent / "io/test-data/ped_data_archive_text.txt"),
        default_unit=TrajectoryUnit.METER,
    )
    corrected_trajectory_data, invalid_persons = correct_invalid_trajectories(
        trajectory_data=trajectory_data,
        walkable_area=setup_walkable_area,
        back_distance_wall=-0.5,
        min_distance_wall=0.01,
        max_distance_wall=0.04,
        back_distance_obst=-0.5,
        min_distance_obst=0.04,
        max_distance_obst=0.5,
    )
    assert is_trajectory_valid(traj_data=corrected_trajectory_data, walkable_area=setup_walkable_area) == True
    assert invalid_persons == []


def test_calculate_movement_wall_failsafe_true():
    points = np.array([shapely.Point(4, 2), shapely.Point(5, 0), shapely.Point(6, 2)])
    wall_line = shapely.LineString([[1, 1], [9, 1]])
    distances = np.array([-1, 1, -1])
    corr_points = _calculate_movement_wall(
        wall_line,
        distances,
        back_distance_wall=-2,
        max_distance_wall=0.5,
        min_distance_wall=0.1,
        points=points,
        failsafe=True,
    )
    assert np.array_equal(corr_points, np.array([shapely.Point(4, 0.74), shapely.Point(5, 0), shapely.Point(6, 0.74)]))


def test_move_from_wall_base_near_zero():
    point = shapely.Point(5, 1)
    wall_line = shapely.LineString([[1, 1], [9, 1]])
    new_distance = 0.15
    distance = 0
    point_moved = _move_from_wall(point, wall_line, new_distance, distance)
    assert point_moved.x == 5
    assert point_moved.y == 0.85


def test_point_valid_test_invalid_wall_point(setup_walkable_area):
    point = np.array([shapely.Point(0, -2.1)])
    assert shapely.within(point, setup_walkable_area.polygon) == False
    x, y = _point_valid_test(
        point,
        setup_walkable_area.polygon,
        back_distance_wall=-1,
        min_distance_wall=0.01,
        max_distance_wall=0.05,
        back_distance_obst=-1,
        min_distance_obst=0.01,
        max_distance_obst=0.05,
    )
    assert shapely.within(shapely.Point([x[0], y[0]]), setup_walkable_area.polygon) == True


def test_wall_facing_zero_point_wall_type_wall():
    assert _wall_facing_zero_point(shapely.LineString([[1, 1], [1, 1]]), walltype=WallType.walls) == 1
