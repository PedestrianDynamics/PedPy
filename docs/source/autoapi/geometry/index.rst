:py:mod:`geometry`
==================

.. py:module:: geometry

.. autoapi-nested-parse::

   Module handling the geometrical environment of the analysis.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geometry.Geometry




Attributes
~~~~~~~~~~

.. autoapisummary::

   geometry.log


.. py:data:: log
   

   

.. py:class:: Geometry(*, walkable_area: shapely.Polygon, obstacles: Optional[List[shapely.Polygon]] = None)

   Class holding the geometry information of the analysis.

   Attributes:
       walkable_area (shapely.Polygon): area in which the pedestrian walk,
           they are only considered for the analysis when inside this area.
       obstacles (List[shapely.Polygon]): areas which are excluded from the
           analysis, pedestrians inside these areas will be ignored.

   .. py:attribute:: walkable_area
      :annotation: :shapely.Polygon

      

   .. py:attribute:: obstacles
      :annotation: :List[shapely.Polygon]

      

   .. py:method:: add_obstacle(obstacle: shapely.Polygon) -> None

      Adds an obstacle to the geometry.

      Args:
          obstacle (Polygon): area which will be excluded from the
          analysis.



