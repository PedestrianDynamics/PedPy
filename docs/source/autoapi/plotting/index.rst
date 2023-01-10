:py:mod:`plotting`
==================

.. py:module:: plotting

.. autoapi-nested-parse::

   Module containing plotting functionalities.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   plotting.plot_geometry
   plotting.plot_trajectories
   plotting.plot_measurement_setup
   plotting.plot_voronoi_cells



Attributes
~~~~~~~~~~

.. autoapisummary::

   plotting.log


.. py:data:: log
   

   

.. py:function:: plot_geometry(*, geometry: pedpy.data.geometry.Geometry, ax: Optional[matplotlib.axes.Axes] = None, **kwargs: Any) -> matplotlib.axes.Axes

   Plot the given geometry in 2-D.

   Args:
       geometry (Geometry): Geometry object to plot
       ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
       line_color (optional): color of the borders
       line_width (optional): line width of the borders
       hole_color (optional): background color of holes
       hole_alpha (optional): alpha of background color for holes

   Returns:
       matplotlib.axes.Axes instance where the geometry is plotted


.. py:function:: plot_trajectories(*, traj: pedpy.data.trajectory_data.TrajectoryData, geometry: Optional[pedpy.data.geometry.Geometry] = None, ax: Optional[matplotlib.axes.Axes] = None, **kwargs: Any) -> matplotlib.axes.Axes

   Plot the given trajectory and geometry in 2-D.

   Args:
       traj (TrajectoryData): Trajectory object to plot
       geometry (Geometry, optional): Geometry object to plot
       ax (matplotlib.axes.Axes, optional): Axes to plot on,
           if None new will be created
       traj_color (optional): color of the trajectories
       traj_width (optional): width of the trajectories
       traj_alpha (optional): alpha of the trajectories
       traj_start_marker (optional): marker to indicate the start of the
           trajectory
       traj_end_marker (optional): marker to indicate the end of the trajectory
       line_color (optional): color of the borders
       line_width (optional): line width of the borders
       hole_color (optional): background color of holes
       hole_alpha (optional): alpha of background color for holes

   Returns:
       matplotlib.axes.Axes instance where the geometry is plotted


.. py:function:: plot_measurement_setup(*, traj: Optional[pedpy.data.trajectory_data.TrajectoryData] = None, geometry: Optional[pedpy.data.geometry.Geometry] = None, measurement_areas: Optional[List[shapely.Polygon]] = None, measurement_lines: Optional[List[shapely.LineString]] = None, ax: Optional[matplotlib.axes.Axes] = None, **kwargs: Any) -> matplotlib.axes.Axes

   Plot the given measurement setup in 2D.

   Args:
       traj (TrajectoryData, optional): Trajectory object to plot
       geometry (Geometry, optional): Geometry object to plot
       measurement_areas (List[Polygon], optional): List of measurement areas
           to plot
       measurement_lines (List[LineString], optional): List of measurement
           lines to plot
       ax (matplotlib.axes.Axes, optional): Axes to plot on,
           if None new will be created
       ma_line_color (optional): color of the measurement areas borders
       ma_line_width (optional): line width of the measurement areas borders
       ma_color (optional): fill color of the measurement areas
       ma_alpha (optional): alpha of measurement area fill color
       ml_color (optional): color of the measurement lines
       ml_width (optional): line width of the measurement lines
       traj_color (optional): color of the trajectories
       traj_width (optional): width of the trajectories
       traj_alpha (optional): alpha of the trajectories
       traj_start_marker (optional): marker to indicate the start of the
           trajectory
       traj_end_marker (optional): marker to indicate the end of the trajectory
       line_color (optional): color of the borders
       line_width (optional): line width of the borders
       hole_color (optional): background color of holes
       hole_alpha (optional): alpha of background color for holes

   Returns:
       matplotlib.axes.Axes instance where the geometry is plotted


.. py:function:: plot_voronoi_cells(*, data: pandas.DataFrame, geometry: Optional[pedpy.data.geometry.Geometry] = None, measurement_area: Optional[shapely.Polygon] = None, ax: Optional[matplotlib.axes.Axes] = None, **kwargs: Any) -> matplotlib.axes.Axes

   Plot the Voronoi cells, geometry, and measurement area in 2D.

   Args:
       data (pd.DataFrame): Voronoi data to plot, should only contain data
           from one frame!
       geometry (Geometry, optional): Geometry object to plot
       measurement_area (List[Polygon], optional): measurement area used to
           compute the Voronoi cells
       ax (matplotlib.axes.Axes, optional): Axes to plot on,
           if None new will be created
       show_ped_positions (optional): show the current positions of the
           pedestrians, data needs to contain columns "X", and "Y"!
       ped_color (optional): color used to display current ped positions
       voronoi_border_color (optional): border color of Voronoi cells
       voronoi_inside_ma_alpha (optional): alpha of part of Voronoi cell
           inside the measurement area, data needs to contain column
           "intersection voronoi"!
       voronoi_outside_ma_alpha (optional): alpha of part of Voronoi cell
           outside the measurement area
       color_mode (optional): color mode to color the Voronoi cells, "density",
           "velocity", and "id". For 'velocity' data needs to contain a
           column 'speed'
       vmin (optional): vmin of colormap, only used when color_mode != "id"
       vmax (optional): vmax of colormap, only used when color_mode != "id"
       show_colorbar (optional): colorbar is displayed, only used when
           color_mode != "id"
       cb_location (optional): location of the colorbar, only used when
           color_mode != "id"
       ma_line_color (optional): color of the measurement areas borders
       ma_line_width (optional): line width of the measurement areas borders
       ma_color (optional): fill color of the measurement areas
       ma_alpha (optional): alpha of measurement area fill color
       ml_color (optional): color of the measurement lines
       ml_width (optional): line width of the measurement lines
       line_color (optional): color of the borders
       line_width (optional): line width of the borders
       hole_color (optional): background color of holes
       hole_alpha (optional): alpha of background color for holes

   Returns:
       matplotlib.axes.Axes instance where the geometry is plotted


