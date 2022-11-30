"""Module containing plotting functionalities"""
import logging
from typing import Any, List, Optional

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import shapely

log = logging.getLogger(__name__)

from pedpy.data.geometry import Geometry
from pedpy.data.trajectory_data import TrajectoryData


def plot_geometry(
    *,
    geometry: Geometry,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Plot the given geometry in 2-D

    Parameters:
        geometry (Geometry): Geometry object to plot
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        line_color (optional): color of the borders
        line_width (optional): line width of the borders
        walkable_area_color (optional): background color of walkable areas
        walkable_area_alpha (optional): alpha of background color for walkable
            areas
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the geometry is plotted
    """
    if ax is None:
        ax = plt.gca()

    line_color = kwargs.get("line_color", "k")
    line_width = kwargs.get("line_width", 1.0)

    walkable_area_color = kwargs.get("walkable_area_color", "w")
    walkable_area_alpha = kwargs.get("walkable_area_alpha", 1.0)

    hole_color = kwargs.get("hole_color", "w")
    hole_alpha = kwargs.get("hole_alpha", 1.0)

    ax.plot(
        *geometry.walkable_area.exterior.xy,
        color=line_color,
        linewidth=line_width,
    )
    ax.fill(
        *geometry.walkable_area.exterior.xy,
        color=walkable_area_color,
        alpha=walkable_area_alpha,
    )

    for hole in geometry.walkable_area.interiors:
        ax.plot(*hole.xy, color=line_color, linewidth=line_width)
        # Paint all holes first white, then with the desired color
        ax.fill(*hole.xy, color="w", alpha=1)
        ax.fill(*hole.xy, color=hole_color, alpha=hole_alpha)
    return ax


def plot_trajectories(
    *,
    traj: TrajectoryData,
    geometry: Optional[Geometry] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Plot the given trajectory and geometry in 2-D

    Parameters:
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
        walkable_area_color (optional): background color of walkable areas
        walkable_area_alpha (optional): alpha of background color for walkable
            areas
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the geometry is plotted
    """

    traj_color = kwargs.get("traj_color", "r")
    traj_width = kwargs.get("traj_width", 1.0)
    traj_alpha = kwargs.get("traj_alpha", 1.0)

    traj_start_marker = kwargs.get("traj_start_marker", "")
    traj_end_marker = kwargs.get("traj_end_marker", "")

    if ax is None:
        ax = plt.gca()

    if geometry is not None:
        ax = plot_geometry(geometry=geometry, ax=ax, **kwargs)

    for id, ped in traj.data.groupby("ID"):
        p = ax.plot(
            ped["X"],
            ped["Y"],
            alpha=traj_alpha,
            color=traj_color,
            linewidth=traj_width,
        )
        ax.scatter(
            ped[ped.frame == ped.frame.min()]["X"],
            ped[ped.frame == ped.frame.min()]["Y"],
            c=p[-1].get_color(),
            marker=traj_start_marker,
        )
        ax.scatter(
            ped[ped.frame == ped.frame.max()]["X"],
            ped[ped.frame == ped.frame.max()]["Y"],
            c=p[-1].get_color(),
            marker=traj_end_marker,
        )

    return ax


def plot_measurement_setup(
    *,
    traj: Optional[TrajectoryData] = None,
    geometry: Optional[Geometry] = None,
    measurement_areas: Optional[List[shapely.Polygon]] = None,
    measurement_lines: Optional[List[shapely.LineString]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Plot the given measurement setup, including trajectories, geometry,
    measurement areas, and measurement lines in 2-D

    Parameters:
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
        walkable_area_color (optional): background color of walkable areas
        walkable_area_alpha (optional): alpha of background color for walkable
            areas
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the geometry is plotted
    """

    ma_line_color = kwargs.get("ma_line_color", "k")
    ma_line_width = kwargs.get("ma_line_width", 1.0)
    ma_color = kwargs.get("ma_color", "w")
    ma_alpha = kwargs.get("ma_alpha", 1.0)

    ml_color = kwargs.get("ml_color", "k")
    ml_width = kwargs.get("ml_width", 1.0)

    if ax is None:
        ax = plt.gca()

    if measurement_areas is not None:
        for measurement_area in measurement_areas:
            ax.plot(
                *measurement_area.exterior.xy,
                color=ma_line_color,
                linewidth=ma_line_width,
            )
            ax.fill(
                *measurement_area.exterior.xy,
                color=ma_color,
                alpha=ma_alpha,
            )

    if measurement_lines is not None:
        for measurement_line in measurement_lines:
            ax.plot(*measurement_line.xy, color=ml_color, linewidth=ml_width)

    if geometry is not None:
        plot_geometry(geometry=geometry, ax=ax, **kwargs)

    if traj is not None:
        plot_trajectories(traj=traj, geometry=None, ax=ax, **kwargs)

    return ax


def plot_voronoi_cells(
    *,
    data: pd.DataFrame,
    geometry: Optional[Geometry] = None,
    measurement_area: Optional[shapely.Polygon] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Plot the Voronoi cells, geometry, and measurement area in 2D.

    Parameters:
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
    """

    show_ped_positions = kwargs.get("show_ped_positions", False)
    ped_color = kwargs.get("ped_color", "w")
    ped_size = kwargs.get("ped_size", 1)
    voronoi_border_color = kwargs.get("voronoi_border_color", "w")
    voronoi_inside_ma_alpha = kwargs.get("voronoi_inside_ma_alpha", 1)
    voronoi_outside_ma_alpha = kwargs.get("voronoi_outside_ma_alpha", 1)

    color_mode = kwargs.get("color_mode", "density")
    color_mode = color_mode.lower()
    if color_mode not in ("density", "velocity", "id"):
        log.warning(
            "'density', 'velocity', and 'id' are the only supported color modes. Use "
            "default 'density'"
        )
        color_mode = "density"

    vmin = kwargs.get("vmin", 0)
    vmax = kwargs.get("vmax", 5 if color_mode == "velocity" else 10)
    cb_location = kwargs.get("cb_location", "right")
    show_colorbar = kwargs.get("show_colorbar", True)

    if ax is None:
        ax = plt.gca()

    # Create color mapping for velocity/density to color
    if color_mode != "id":
        voronoi_colormap = plt.get_cmap("jet")
        norm = mpl.colors.Normalize(vmin, vmax)
        scalar_mappable = mpl.cm.ScalarMappable(
            norm=norm, cmap=voronoi_colormap
        )

    else:
        voronoi_colormap = plt.get_cmap("tab20c")

    if measurement_area is not None:
        plot_measurement_setup(
            measurement_areas=[measurement_area], ax=ax, **kwargs
        )

    for _, row in data.iterrows():
        poly = row["individual voronoi"]

        if color_mode != "id":
            color = (
                scalar_mappable.to_rgba(row["speed"])
                if color_mode == "velocity"
                else scalar_mappable.to_rgba(1 / poly.area)
            )
        else:
            color = voronoi_colormap(row["ID"] % 20)

        ax.plot(*poly.exterior.xy, alpha=1, color=voronoi_border_color)
        ax.fill(*poly.exterior.xy, fc=color, alpha=voronoi_outside_ma_alpha)

        if not shapely.is_empty(row["intersection voronoi"]):
            intersection_poly = row["intersection voronoi"]
            ax.fill(
                *intersection_poly.exterior.xy,
                fc=color,
                alpha=voronoi_inside_ma_alpha,
            )

        if show_ped_positions:
            ax.scatter(row["X"], row["Y"], color=ped_color, s=ped_size)

    if show_colorbar and color_mode != "id":
        plt.colorbar(
            scalar_mappable,
            ax=ax,
            label=r"v / $\frac{m}{s}$"
            if color_mode == "velocity"
            else r" $\rho$ / $\frac{1}{m^2}$",
            shrink=0.4,
            location=cb_location,
        )

    if geometry is not None:
        plot_geometry(
            ax=ax, geometry=geometry, walkable_area_alpha=0.0, **kwargs
        )
    ax.set_xlabel(r"x/m")
    ax.set_ylabel(r"y/m")
    return ax
