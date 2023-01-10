:py:mod:`density_calculator`
============================

.. py:module:: density_calculator

.. autoapi-nested-parse::

   Module containing functions to compute densities.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   density_calculator.compute_classic_density
   density_calculator.compute_voronoi_density
   density_calculator.compute_passing_density
   density_calculator.compute_individual_voronoi_polygons



.. py:function:: compute_classic_density(*, traj_data: pandas.DataFrame, measurement_area: shapely.Polygon) -> pandas.DataFrame

   Compute the classic density per frame inside the given measurement area.

   Args:
       traj_data (pd.DataFrame): trajectory data to analyze
       measurement_area (shapely.Polygon): area for which the density is
           computed

   Returns:
       DataFrame containing the columns: 'frame' and 'classic density'


.. py:function:: compute_voronoi_density(*, traj_data: pandas.DataFrame, measurement_area: shapely.Polygon, geometry: pedpy.data.geometry.Geometry, cut_off: Optional[Tuple[float, int]] = None, use_blind_points: bool = False) -> Tuple[pandas.DataFrame, pandas.DataFrame]

   Compute the voronoi density per frame inside the given measurement area.

   Args:
       traj_data (pd.DataFrame): trajectory data to analyze
       measurement_area (shapely.Polygon): area for which the density is
           computed
       geometry (Geometry): bounding area, where pedestrian are supposed to be
       cut_off (Tuple[float, int): radius of max extended voronoi cell (in m),
           number of linear segments in the approximation of circular arcs,
           needs to be divisible by 4!
       use_blind_points (bool): adds extra 4 points outside the geometry to
           also compute voronoi cells when less than 4 peds are in the
           geometry
   Returns:
         DataFrame containing the columns: 'frame' and 'voronoi density',
         DataFrame containing the columns: 'ID', 'frame', 'individual
           voronoi', 'intersecting voronoi'


.. py:function:: compute_passing_density(*, density_per_frame: pandas.DataFrame, frames: pandas.DataFrame) -> pandas.DataFrame

   Compute the individual density of the pedestrian who pass the area.

   Args:
       density_per_frame (pd.DataFrame): density per frame, DataFrame
               containing the columns: 'frame' (as index) and 'density'
       frames (pd.DataFrame): information for each pedestrian in the area,
               need to contain the following columns: 'ID','frame_start',
               'frame_end'

   Returns:
         DataFrame containing the columns: 'ID' and 'density' in 1/m


.. py:function:: compute_individual_voronoi_polygons(*, traj_data: pandas.DataFrame, geometry: pedpy.data.geometry.Geometry, cut_off: Optional[Tuple[float, int]] = None, use_blind_points: bool = False) -> pandas.DataFrame

   Compute the individual voronoi cells for each person and frame.

   Args:
       traj_data (pd.DataFrame): trajectory data
       geometry (Geometry): bounding area, where pedestrian are supposed to be
       cut_off (Tuple[float, int]): radius of max extended voronoi cell (in
               m), number of linear segments in the approximation of circular
               arcs, needs to be divisible by 4!
       use_blind_points (bool): adds extra 4 points outside the geometry to
               also compute voronoi cells when less than 4 peds are in the
               geometry

   Returns:
       DataFrame containing the columns: 'ID', 'frame' and 'individual
       voronoi'.


