:py:mod:`trajectory_loader`
===========================

.. py:module:: trajectory_loader

.. autoapi-nested-parse::

   Load trajectories to the internal trajectory data format.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   trajectory_loader.load_trajectory



.. py:function:: load_trajectory(*, trajectory_file: pathlib.Path, default_frame_rate: Optional[float] = None, default_unit: Optional[pedpy.data.trajectory_data.TrajectoryUnit] = None) -> pedpy.data.trajectory_data.TrajectoryData

   Loads the trajectory file in the internal trajectory data format.

   Loads the relevant data: trajectory data, frame rate, and type of
   trajectory from the given trajectory file. If the file does not contain
   some data, defaults can be submitted.

   Args:
       trajectory_file (pathlib.Path): file containing the trajectory
       default_frame_rate (float): frame rate of the file, None if frame rate
           from file is used
       default_unit (TrajectoryUnit): unit in which the coordinates are stored
               in the file, None if unit should be parsed from the file

   Returns:
       Tuple containing: trajectory data, frame rate, and type of trajectory.


