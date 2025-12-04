import pathlib
import pytest
import numpy as np

from pedpy import InputError
from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import load_trajectory, TrajectoryUnit
from pedpy.methods.method_utils import is_trajectory_valid
from pedpy.preprocessing.trajectory_projector import correct_invalid_trajectories


@pytest.fixture
def setup_walkable_area():
    walk_area = WalkableArea(
        [(3.5, -2), (3.5, 8), (-3.5, 8), (-3.5, -2), ],

         obstacles=[[(-0.7, -1.1), (-0.25, -1.1), (-0.25, -0.15),
                     (-0.4, 0.0), (-2.8, 0.0),
                     (-2.8, 6.7), (-3.05, 6.7), (-3.05, -0.3),
                     (-0.7, -0.3), (-0.7, -1.0), ],

                    [(0.25, -1.1), (0.7, -1.1), (0.7, -0.3),
                     (3.05, -0.3), (3.05, 6.7),
                     (2.8, 6.7), (2.8, 0.0), (0.4, 0.0),
                     (0.25, -0.15), (0.25, -1.1), ],
                    ], )
    return walk_area

@pytest.fixture
def setup_trajectory_data():
    trajectory_data = load_trajectory(
        trajectory_file=pathlib.Path("unit_tests/io/test-data/030_c_56_h0.txt"),
        default_unit=TrajectoryUnit.METER)

    return trajectory_data

def test_correct_invalid_trajectories(setup_trajectory_data, setup_walkable_area):
    assert not  (is_trajectory_valid(traj_data=setup_trajectory_data,
                                     walkable_area=setup_walkable_area))
    corrected_trajectory_data = correct_invalid_trajectories(
        trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area)
    assert is_trajectory_valid(traj_data= corrected_trajectory_data, walkable_area= setup_walkable_area) == True

def test_correct_invalid_trajectories_min_paramter(setup_trajectory_data, setup_walkable_area):

    assert (is_trajectory_valid(traj_data=setup_trajectory_data,
                                walkable_area=setup_walkable_area)) == False
    corrected_trajectory_data = correct_invalid_trajectories(
        trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area,
        back_distance_wall=-1, start_distance_wall=0.01, end_distance_wall=0.01,
        back_distance_obst=-1, start_distance_obst=0.01, end_distance_obst=0.01 )
    assert is_trajectory_valid(traj_data= corrected_trajectory_data, walkable_area= setup_walkable_area) == True

def test_correct_invalid_trajectories_invalid_back_distance(setup_trajectory_data, setup_walkable_area):

    assert (is_trajectory_valid(traj_data=setup_trajectory_data,
                                walkable_area=setup_walkable_area)) == False

    with pytest.raises(InputError, match = r"back_distance_wall has to be <0, is currently 0"):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area,
            back_distance_wall=0, start_distance_wall=0.01, end_distance_wall=0.04,
            back_distance_obst=1, start_distance_obst=0.01, end_distance_obst=0.04)

def test_correct_invalid_trajectories_invalid_start_distance(setup_trajectory_data, setup_walkable_area):

    assert (is_trajectory_valid(traj_data=setup_trajectory_data,
                                walkable_area=setup_walkable_area)) == False

    with pytest.raises(InputError, match = r"start_distance_obst and end_distance_obst have to be > 0 and "
            "end_distance_obst has to be >=  start_distance_obst"):
        correct_invalid_trajectories(
            trajectory_data=setup_trajectory_data, walkable_area=setup_walkable_area,
            back_distance_wall=-0.5, start_distance_wall=0.01, end_distance_wall=0.04,
            back_distance_obst=-0.5, start_distance_obst=0.04, end_distance_obst=0.01)