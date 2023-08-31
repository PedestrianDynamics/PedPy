"""Module containing plotting functionalities."""
import logging
from typing import Any, List, Optional

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import shapely

from pedpy.column_identifier import (
    ID_COL,
    INTERSECTION_COL,
    POLYGON_COL,
    SPEED_COL,
    X_COL,
    Y_COL,
    TIME_COL,
    CUMULATED_COL,
    DENSITY_COL
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData

_log = logging.getLogger(__name__)


def _plot_series(ax: matplotlib.axes.Axes, title: str, x, y, color: str, x_label: str, y_label: str):
    ax.set_title(title)
    ax.plot(
        x,
        y,
        c=color
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


def plot_nt(
        *,
        nt: pd.DataFrame,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the number of pedestrians over time.

    Args:
        nt (pd.DataFrame): List of density profiles
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot

    Returns:
        matplotlib.axes.Axes instance where the number of pedestrians is plotted
    """
    if ax is None:
        ax = plt.gca()

    color = kwargs.get("color", "k")
    return _plot_series(ax=ax,
                        title="N-t",
                        x=nt[TIME_COL],
                        y=nt[CUMULATED_COL],
                        color=color,
                        x_label="t / s",
                        y_label="# pedestrians")


def plot_density(
        *,
        classic_density: pd.DataFrame,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the density over time.

    Args:
        classic_density(pd.DataFrame) : density per frame
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if ax is None:
        ax = plt.gca()

    color = kwargs.get("color", "k")
    return _plot_series(ax=ax,
                        title="Classic density over time",
                        x=classic_density.index,
                        y=classic_density[DENSITY_COL],
                        color=color,
                        x_label="frame",
                        y_label="# pedestrians")


def plot_speed(
        *,
        speed: pd.Series,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any
) -> matplotlib.axes.Axes:
    """Plot the speed over time.

    Args:
        speed(pd.Series): speed per frame
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if ax is None:
        ax = plt.gca()

    color = kwargs.get("color", "k")
    return _plot_series(ax=ax,
                        title="speed over time",
                        x=speed.index,
                        y=speed,
                        color=color,
                        x_label="frame",
                        y_label="v / m/s")


def plot_passing_density(
        *,
        passing_density: pd.DataFrame,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the passing density.

    Args:
        passing_density(pd.DataFrame): DataFrame containing the columns: 'ID' and 'density' in 1/m^2.
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        facecolor (optional): color of the plot body
        edgecolor (optional): color of the edges of the plot

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if ax is None:
        ax = plt.gca()

    facecolor = kwargs.get("facecolor", "r")
    edgecolor = kwargs.get("edgecolor", "k")

    ax.set_title("Individual density")
    violin_parts = plt.violinplot(
        passing_density.density, showmeans=True, showextrema=True, showmedians=True
    )
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor(edgecolor)

    ax.set_xlabel("")
    ax.set_ylabel("$\\rho$ / 1/$m^2$")
    ax.set_xticks([])
    return ax


def plot_walkable_area(
        *,
        walkable_area: WalkableArea,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the given walkable area in 2-D.

    Args:
        walkable_area (WalkableArea): WalkableArea object to plot
        ax (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        line_color (optional): color of the borders
        line_color (optional): line width of the borders
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the walkable area is plotted
    """
    if ax is None:
        ax = plt.gca()

    line_color = kwargs.get("line_color", "k")
    line_width = kwargs.get("line_width", 1.0)

    hole_color = kwargs.get("hole_color", "w")
    hole_alpha = kwargs.get("hole_alpha", 1.0)

    ax.plot(
        *walkable_area.polygon.exterior.xy,
        color=line_color,
        linewidth=line_width,
    )

    for hole in walkable_area.polygon.interiors:
        ax.plot(*hole.xy, color=line_color, linewidth=line_width)
        # Paint all holes first white, then with the desired color
        ax.fill(*hole.xy, color="w", alpha=1)
        ax.fill(*hole.xy, color=hole_color, alpha=hole_alpha)

    ax.set_xlabel(r"x/m")
    ax.set_ylabel(r"y/m")

    return ax


def plot_trajectories(
        *,
        traj: TrajectoryData,
        walkable_area: Optional[WalkableArea] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the given trajectory and walkable area in 2-D.

    Args:
        traj (TrajectoryData): Trajectory object to plot
        walkable_area (WalkableArea, optional): WalkableArea object to plot
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
        matplotlib.axes.Axes instance where the trajectories are plotted
    """
    traj_color = kwargs.get("traj_color", "r")
    traj_width = kwargs.get("traj_width", 1.0)
    traj_alpha = kwargs.get("traj_alpha", 1.0)

    traj_start_marker = kwargs.get("traj_start_marker", "")
    traj_end_marker = kwargs.get("traj_end_marker", "")

    if ax is None:
        ax = plt.gca()

    if walkable_area is not None:
        ax = plot_walkable_area(walkable_area=walkable_area, ax=ax, **kwargs)

    for _, ped in traj.data.groupby(ID_COL):
        plot = ax.plot(
            ped[X_COL],
            ped[Y_COL],
            alpha=traj_alpha,
            color=traj_color,
            linewidth=traj_width,
        )
        ax.scatter(
            ped[ped.frame == ped.frame.min()][X_COL],
            ped[ped.frame == ped.frame.min()][Y_COL],
            c=plot[-1].get_color(),
            marker=traj_start_marker,
        )
        ax.scatter(
            ped[ped.frame == ped.frame.max()][X_COL],
            ped[ped.frame == ped.frame.max()][Y_COL],
            c=plot[-1].get_color(),
            marker=traj_end_marker,
        )

    ax.set_xlabel(r"x/m")
    ax.set_ylabel(r"y/m")

    return ax


def plot_measurement_setup(
        *,
        traj: Optional[TrajectoryData] = None,
        walkable_area: Optional[WalkableArea] = None,
        measurement_areas: Optional[List[MeasurementArea]] = None,
        measurement_lines: Optional[List[MeasurementLine]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the given measurement setup in 2D.

    Args:
        traj (TrajectoryData, optional): Trajectory object to plot
        walkable_area (WalkableArea, optional): WalkableArea object to plot
        measurement_areas (List[MeasurementArea], optional): List of measurement areas
            to plot
        measurement_lines (List[MeasurementLine], optional): List of measurement
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
        matplotlib.axes.Axes instance where the measurement setup is plotted
    """
    ma_line_color = kwargs.get("ma_line_color", "k")
    ma_line_width = kwargs.get("ma_line_width", 1.0)
    ma_color = kwargs.get("ma_color", "w")
    ma_alpha = kwargs.get("ma_alpha", 1.0)

    ml_color = kwargs.get("ml_color", "k")
    ml_width = kwargs.get("ml_width", 1.0)

    if ax is None:
        ax = plt.gca()

    if walkable_area is not None:
        plot_walkable_area(walkable_area=walkable_area, ax=ax, **kwargs)

    if traj is not None:
        plot_trajectories(traj=traj, walkable_area=None, ax=ax, **kwargs)

    if measurement_areas is not None:
        for measurement_area in measurement_areas:
            ax.plot(
                *measurement_area.polygon.exterior.xy,
                color=ma_line_color,
                linewidth=ma_line_width,
            )
            ax.fill(
                *measurement_area.polygon.exterior.xy,
                color=ma_color,
                alpha=ma_alpha,
            )

    if measurement_lines is not None:
        for measurement_line in measurement_lines:
            ax.plot(*measurement_line.xy, color=ml_color, linewidth=ml_width)

    ax.set_xlabel(r"x/m")
    ax.set_ylabel(r"y/m")

    return ax


def plot_voronoi_cells(  # pylint: disable=too-many-locals
        *,
        data: pd.DataFrame,
        walkable_area: Optional[WalkableArea] = None,
        measurement_area: Optional[MeasurementArea] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the Voronoi cells, walkable able, and measurement area in 2D.

    Args:
        data (pd.DataFrame): Voronoi data to plot, should only contain data
            from one frame!
        walkable_area (WalkableArea, optional): WalkableArea object to plot
        measurement_area (MeasurementArea, optional): measurement area used to
            compute the Voronoi cells
        ax (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        show_ped_positions (optional): show the current positions of the
            pedestrians, data needs to contain columns "x", and "y"!
        ped_color (optional): color used to display current ped positions
        voronoi_border_color (optional): border color of Voronoi cells
        voronoi_inside_ma_alpha (optional): alpha of part of Voronoi cell
            inside the measurement area, data needs to contain column
            "intersection"!
        voronoi_outside_ma_alpha (optional): alpha of part of Voronoi cell
            outside the measurement area
        color_mode (optional): color mode to color the Voronoi cells, "density",
            "speed", and "id". For 'speed' data needs to contain a
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
        matplotlib.axes.Axes instance where the Voronoi cells are plotted
    """
    show_ped_positions = kwargs.get("show_ped_positions", False)
    ped_color = kwargs.get("ped_color", "w")
    ped_size = kwargs.get("ped_size", 1)
    voronoi_border_color = kwargs.get("voronoi_border_color", "w")
    voronoi_inside_ma_alpha = kwargs.get("voronoi_inside_ma_alpha", 1)
    voronoi_outside_ma_alpha = kwargs.get("voronoi_outside_ma_alpha", 1)

    color_mode = kwargs.get("color_mode", "density")
    color_mode = color_mode.lower()
    if color_mode not in ("density", "speed", "id"):
        _log.warning(
            "'density', 'speed', and 'id' are the only supported color modes. Use "
            "default 'density'"
        )
        color_mode = "density"

    vmin = kwargs.get("vmin", 0)
    vmax = kwargs.get("vmax", 5 if color_mode == "speed" else 10)
    cb_location = kwargs.get("cb_location", "right")
    show_colorbar = kwargs.get("show_colorbar", True)

    if ax is None:
        ax = plt.gca()

    # Create color mapping for speed/density to color
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
        poly = row[POLYGON_COL]

        if color_mode != "id":
            color = (
                scalar_mappable.to_rgba(row[SPEED_COL])
                if color_mode == "speed"
                else scalar_mappable.to_rgba(1 / poly.area)
            )
        else:
            color = voronoi_colormap(row[ID_COL] % 20)

        ax.plot(*poly.exterior.xy, alpha=1, color=voronoi_border_color)
        ax.fill(*poly.exterior.xy, fc=color, alpha=voronoi_outside_ma_alpha)

        if INTERSECTION_COL in data.columns:
            if not shapely.is_empty(row[INTERSECTION_COL]):
                intersection_poly = row[INTERSECTION_COL]
                ax.fill(
                    *intersection_poly.exterior.xy,
                    fc=color,
                    alpha=voronoi_inside_ma_alpha,
                )

        if show_ped_positions:
            ax.scatter(row[X_COL], row[Y_COL], color=ped_color, s=ped_size)

    if show_colorbar and color_mode != "id":
        plt.colorbar(
            scalar_mappable,
            ax=ax,
            label=r"v / $\frac{m}{s}$"
            if color_mode == "speed"
            else r" $\rho$ / $\frac{1}{m^2}$",
            shrink=0.4,
            location=cb_location,
        )

    if walkable_area is not None:
        plot_walkable_area(ax=ax, walkable_area=walkable_area, **kwargs)
    ax.set_xlabel(r"x/m")
    ax.set_ylabel(r"y/m")
    return ax
