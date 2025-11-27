import pathlib

from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import load_trajectory, TrajectoryUnit
from pedpy.methods.method_utils import is_trajectory_valid
from pedpy.preprocessing.trajectory_projector import correct_invalid_trajectories


def test_correct_invalid_trajectories():
    trajectory_data= load_trajectory(trajectory_file=pathlib.Path("unit_tests/io/test-data/030_c_56_h0.txt"), default_unit=TrajectoryUnit.METER)
    walk_area = WalkableArea([ (3.5, -2), (3.5, 8), (-3.5, 8),(-3.5, -2), ],

    obstacles=[ [ (-0.7, -1.1), (-0.25, -1.1), (-0.25, -0.15), (-0.4, 0.0), (-2.8, 0.0),
                (-2.8, 6.7), (-3.05, 6.7), (-3.05, -0.3), (-0.7, -0.3), (-0.7, -1.0), ],

                [ (0.25, -1.1), (0.7, -1.1), (0.7, -0.3), (3.05, -0.3), (3.05, 6.7),
                (2.8, 6.7), (2.8, 0.0), (0.4, 0.0), (0.25, -0.15), (0.25, -1.1), ],
              ], )

    assert (is_trajectory_valid(traj_data=trajectory_data, walkable_area=walk_area)) == False
    corrected_trajectory_data = correct_invalid_trajectories(
        trajectory_data=trajectory_data, walkable_area=walk_area)
    assert is_trajectory_valid(traj_data= corrected_trajectory_data, walkable_area= walk_area) == True

def test_correct_invalid_trajectories_min_paramters():
    trajectory_data = load_trajectory(
        trajectory_file=pathlib.Path("unit_tests/io/test-data/030_c_56_h0.txt"),
        default_unit=TrajectoryUnit.METER)
    walk_area = WalkableArea([(3.5, -2), (3.5, 8), (-3.5, 8), (-3.5, -2), ],

                             obstacles=[[(-0.7, -1.1), (-0.25, -1.1), (-0.25, -0.15),
                                         (-0.4, 0.0), (-2.8, 0.0),
                                         (-2.8, 6.7), (-3.05, 6.7), (-3.05, -0.3),
                                         (-0.7, -0.3), (-0.7, -1.0), ],

                                        [(0.25, -1.1), (0.7, -1.1), (0.7, -0.3),
                                         (3.05, -0.3), (3.05, 6.7),
                                         (2.8, 6.7), (2.8, 0.0), (0.4, 0.0),
                                         (0.25, -0.15), (0.25, -1.1), ],
                                        ], )

    assert (is_trajectory_valid(traj_data=trajectory_data,
                                walkable_area=walk_area)) == False
    corrected_trajectory_data = correct_invalid_trajectories(
        trajectory_data=trajectory_data, walkable_area=walk_area, back_distance_wall=-1,
        start_distance_wall=0.01, end_distance_wall=0.01, back_distance_obst=-1,
        start_distance_obst=0.01, end_distance_obst=0.01, )
    assert is_trajectory_valid(traj_data= corrected_trajectory_data, walkable_area= walk_area) == True



