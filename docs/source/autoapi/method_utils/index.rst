:py:mod:`method_utils`
======================

.. py:module:: method_utils

.. autoapi-nested-parse::

   Helper functions for the analysis methods.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   method_utils.is_trajectory_valid
   method_utils.get_invalid_trajectory
   method_utils.compute_frame_range_in_area



.. py:function:: is_trajectory_valid(*, traj: pedpy.data.trajectory_data.TrajectoryData, geometry: pedpy.data.geometry.Geometry) -> bool

   Checks if all trajectory data points lie within the given geometry.

   Args:
       traj (TrajectoryData): trajectory data
       geometry (Geometry): geometry

   Returns:
       All points lie within geometry


.. py:function:: get_invalid_trajectory(*, traj: pedpy.data.trajectory_data.TrajectoryData, geometry: pedpy.data.geometry.Geometry) -> pandas.DataFrame

   Returns all trajectory data points outside the given geometry.

   Args:
       traj (TrajectoryData): trajectory data
       geometry (Geometry): geometry

   Returns:
       DataFrame showing all data points outside the given geometry


.. py:function:: compute_frame_range_in_area(*, traj_data: pandas.DataFrame, measurement_line: shapely.LineString, width: float) -> Tuple[pandas.DataFrame, shapely.Polygon]

   Compute the frame ranges for each pedestrian inside the measurement area.

   Note:
       Only pedestrians passing the complete measurement area will be
       considered. Meaning they need to cross measurement_line and the line
       with the given offset in one go. If leaving the area between two lines
        through the same line will be ignored.

       As passing we define the frame the pedestrians enter the area and
       then moves through the complete area without leaving it. Hence,
       doing a closed analysis of the movement area with several measuring
       ranges underestimates the actual movement time.

   Args:
       traj_data (pd.DataFrame): trajectory data
       measurement_line (shapely.LineString): measurement line
       width (float): distance to the second measurement line

   Returns:
       DataFrame containing the columns: 'ID', 'frame_start', 'frame_end' and
       the created measurement area


