:py:mod:`flow_calculator`
=========================

.. py:module:: flow_calculator

.. autoapi-nested-parse::

   Module containing functions to compute flows.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   flow_calculator.compute_n_t
   flow_calculator.compute_flow



.. py:function:: compute_n_t(*, traj_data: pandas.DataFrame, measurement_line: shapely.LineString, frame_rate: float) -> Tuple[pandas.DataFrame, pandas.DataFrame]

   Compute the framewise cumulative number of pedestrians passing the line.

   Note: for each pedestrian only the first passing of the line is considered!
   Args:
       traj_data (TrajectoryData): trajectory data
       measurement_line (shapely.LineString): line for which n-t is computed
       frame_rate (float): frame rate of the trajectory data

   Returns:
       DataFrame containing the columns 'frame', 'Cumulative pedestrians',
       and 'Time [s]' and
       DataFrame containing the columns 'ID', and 'frame'


.. py:function:: compute_flow(*, nt: pandas.DataFrame, crossing_frames: pandas.DataFrame, individual_speed: pandas.DataFrame, delta_t: int, frame_rate: float) -> pandas.DataFrame

   Compute the flow for the given crossing_frames and nt.

   Args:
       nt (pd.DataFrame): DataFrame containing the columns 'frame',
           'Cumulative pedestrians', and 'Time [s]' (see result from
           compute_nt)
       crossing_frames (pd.DataFrame): DataFrame containing the columns
           'ID',  and 'frame' (see result from compute_nt)
       individual_speed (pd.DataFrame): DataFrame containing the columns
           'ID', 'frame', and 'speed'
       delta_t (int): size of the time interval to compute the flow
       frame_rate (float): frame rate of the trajectories

   Returns:
       DataFrame containing the columns 'Flow rate(1/s)', and 'Mean
       velocity(m/s)'


