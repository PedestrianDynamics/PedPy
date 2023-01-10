:py:mod:`trajectory_data`
=========================

.. py:module:: trajectory_data

.. autoapi-nested-parse::

   Module handling the trajectory data of the analysis.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   trajectory_data.TrajectoryUnit
   trajectory_data.TrajectoryData




.. py:class:: TrajectoryUnit

   Bases: :py:obj:`aenum.Enum`

   Identifier of the unit of the trajectory coordinates.

   .. py:attribute:: METER
      :annotation: = [1, 'meter (m)']

      

   .. py:attribute:: CENTIMETER
      :annotation: = [100, 'centimeter (cm)']

      


.. py:class:: TrajectoryData(data: pandas.DataFrame, frame_rate: float, file: pathlib.Path)

   Trajectory Data.

   Wrapper around the trajectory data, holds the data as a data frame.

   Note:
       The coordinate data is stored in meter ('m')!

   Attributes:
       data (pd.DataFrame): data frame containing the actual data in the form:
           "ID", "frame", "X", "Y", "Z"

       frame_rate (float): frame rate of the trajectory file

       file (pothlib.Path): file from which is trajectories was read


   .. py:attribute:: data
      :annotation: :pandas.DataFrame

      

   .. py:attribute:: frame_rate
      :annotation: :float

      

   .. py:attribute:: file
      :annotation: :pathlib.Path

      


