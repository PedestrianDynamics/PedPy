import pandas as pd
from shapely.geometry import LineString, Point

from report.data.trajectory_data import TrajectoryData
from report.methods.method_utils import compute_individual_movement, compute_individual_speed


class VelocityCalculator:
    """Calculator for the instantaneous velocities of the pedestrians

    Attributes:
         frame_step (int): gives the size of time interval for calculating the velocity
         set_movement_direction (str): indicates in which direction the velocity will be projected
         ignore_backward_movement (bool):  indicates whether you want to ignore the movement opposite to
                                           the direction from `set_movement_direction`
    """

    frame_step: int
    set_movement_direction: str
    ignore_backward_movement: bool

    def __init__(self, frame_step: int, movement_direction: str, ignore_backward_movement: bool):
        self.frame_step = frame_step
        self.set_movement_direction = movement_direction
        self.ignore_backward_movement = ignore_backward_movement

    def compute_instantaneous_velocity(
        self,
        trajectory: TrajectoryData,
        agent_id: int,
        frame: int,
    ):
        """Compute the instantaneous velocity of a pedestrian at a specific frame

        Args:
            trajectory (TrajectoryData): trajectory data
            agent_id (int): id of the agent
            frame: frame for which the velocity is calculated

        Returns:
            the instantaneous [in meter/second]
        """
        positions = trajectory.get_pedestrian_positions(frame, agent_id, self.frame_step)
        line = LineString(positions)
        length = Point(line.coords[0]).distance(Point(line.coords[-1]))

        time_movement = (len(line.coords) - 1) * 1.0 / trajectory.frame_rate

        speed = length / time_movement

        return speed


def compute_individual_velocity(traj_data: TrajectoryData, frame_step: int) -> pd.DataFrame:
    """Compute the individual velocity for each pedestrian

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_step (int): gives the size of time interval for calculating the velocity

    Returns:
        DataFrame containing the columns 'ID', 'frame', 'speed'
    """
    df_movement = compute_individual_movement(traj_data.data, frame_step)
    df_speed = compute_individual_speed(df_movement, traj_data.frame_rate)

    return df_speed


def compute_mean_velocity_per_frame(traj_data: TrajectoryData, frame_step: int) -> pd.DataFrame:
    """Compute mean velocity per frame

    Args:
        traj_data (TrajectoryData): trajectory data
        frame_step (int): gives the size of time interval for calculating the velocity

    Returns:
        DataFrame containing the columns 'frame' and 'speed' and
        DataFrame containing the columns 'ID', 'frame', 'speed' and
    """
    df_speed = compute_individual_velocity(traj_data, frame_step)
    df_mean = df_speed.groupby("frame")["speed"].mean()
    df_mean = df_mean.reindex(
        list(range(traj_data.data.frame.min(), traj_data.data.frame.max() + 1)),
        fill_value=0.0,
    )
    return df_mean, df_speed
