from shapely.geometry import LineString, Point

from report.data.trajectory_data import TrajectoryData


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
