:py:mod:`profile_calculator`
============================

.. py:module:: profile_calculator

.. autoapi-nested-parse::

   Module containing functions to compute profiles.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   profile_calculator.VelocityMethod



Functions
~~~~~~~~~

.. autoapisummary::

   profile_calculator.compute_profiles



.. py:class:: VelocityMethod

   Bases: :py:obj:`aenum.Enum`

   Identifier for the method used to compute the mean velocity.

   .. py:attribute:: ARITHMETIC
      :annotation: = [0, 'arithmetic mean velocity']

      

   .. py:attribute:: VORONOI
      :annotation: = [1, 'voronoi velocity']

      


.. py:function:: compute_profiles(*, individual_voronoi_velocity_data: pandas.DataFrame, walkable_area: shapely.Polygon, grid_size: float, velocity_method: VelocityMethod) -> Tuple[List[numpy.typing.NDArray[numpy.float64]], List[numpy.typing.NDArray[numpy.float64]]]

   Computes the density and velocity profiles.

   Note: As this is a quite compute heavy operation, it is suggested to
   reduce the geometry to the important areas.

   Args:
       individual_voronoi_velocity_data (pd.DataFrame): individual voronoi
           and velocity data, needs to contain a column 'individual voronoi'
           which holds shapely.Polygon information and a column 'speed'
           which holds a floating point value
       walkable_area (shapely.Polygon): geometry for which the profiles are
           computed
       grid_size (float): resolution of the grid used for computing the
           profiles
       velocity_method (VelocityMethod): velocity method used to compute the
           velocity
   Returns:
       (List of density profiles, List of velocity profiles)


