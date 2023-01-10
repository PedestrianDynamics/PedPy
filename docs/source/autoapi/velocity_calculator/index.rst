:py:mod:`velocity_calculator`
=============================

.. py:module:: velocity_calculator

.. autoapi-nested-parse::

   Module containing functions to compute velocities.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   velocity_calculator.compute_individual_velocity
   velocity_calculator.compute_mean_velocity_per_frame
   velocity_calculator.compute_voronoi_velocity
   velocity_calculator.compute_passing_speed



.. py:function:: compute_individual_velocity(*, traj_data: pandas.DataFrame, frame_rate: float, frame_step: int, movement_direction: Optional[numpy.typing.NDArray[numpy.float64]] = None) -> pandas.DataFrame

   Compute the individual velocity for each pedestrian.

   Note: when using a movement direction the velocity may be negative!

   Args:
       traj_data (TrajectoryData): trajectory data
       frame_rate (float): frame rate of the trajectory
       frame_step (int): gives the size of time interval for calculating the
           velocity.
       movement_direction (np.ndarray): main movement direction on which the
           actual movement is projected (default: None, when the un-projected
           movement should be used)

   Returns:
       DataFrame containing the columns 'ID', 'frame', and 'speed'


.. py:function:: compute_mean_velocity_per_frame(*, traj_data: pandas.DataFrame, measurement_area: shapely.Polygon, frame_rate: float, frame_step: int, movement_direction: Optional[numpy.typing.NDArray[numpy.float64]] = None) -> Tuple[pandas.DataFrame, pandas.DataFrame]

   Compute mean velocity per frame.

   Note: when using a movement direction the velocity may be negative!

   Args:
       traj_data (TrajectoryData): trajectory data
       measurement_area (shapely.Polygon): measurement area for which the
           velocity is computed
       frame_rate (float): frame rate of the trajectory
       frame_step (int): gives the size of time interval for calculating the
           velocity
       movement_direction (np.ndarray): main movement direction on which the
           actual movement is projected (default: None, when the un-projected
           movement should be used)

   Returns:
       DataFrame containing the columns 'frame' and 'speed'
       DataFrame containing the columns 'ID', 'frame', and 'speed'


.. py:function:: compute_voronoi_velocity(*, traj_data: pandas.DataFrame, individual_voronoi_intersection: pandas.DataFrame, frame_rate: float, frame_step: int, measurement_area: shapely.Polygon, movement_direction: Optional[numpy.typing.NDArray[numpy.float64]] = None) -> Tuple[pandas.Series, pandas.DataFrame]

   Compute the voronoi velocity per frame.

   Note: when using a movement direction the velocity may be negative!

   Args:
       traj_data (TrajectoryData): trajectory data
       individual_voronoi_intersection (pd.DataFrame): intersections of the
           individual with the measurement area of each pedestrian
       frame_rate (float): frame rate of the trajectory
       frame_step (int): gives the size of time interval for calculating the
           velocity
       measurement_area (shapely.Polygon): area in which the voronoi velocity
           should be computed
       movement_direction (np.ndarray): main movement direction on which the
           actual movement is projected (default: None, when the un-projected
           movement should be used)

   Returns:
       DataFrame containing the columns 'frame' and 'voronoi speed'
       DataFrame containing the columns 'ID', 'frame', and 'speed'


.. py:function:: compute_passing_speed(*, frames_in_area: pandas.DataFrame, frame_rate: float, distance: float) -> pandas.DataFrame

   Compute the individual speed of the pedestrian who pass the area.

   Args:
       frames_in_area (pd.DataFrame): information for each pedestrian in the
           area, need to contain the following columns: 'ID', 'start', 'end',
           'frame_start', 'frame_end'
       frame_rate (float): frame rate of the trajectory
       distance (float): distance between the two measurement lines
   Returns:
       DataFrame containing the columns: 'ID', 'speed' which is the speed
       in m/s



