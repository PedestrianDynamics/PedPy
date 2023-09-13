"""Module containing plotting functionalities."""
import logging
from typing import Any, List, Optional

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    FLOW_COL,
    FRAME_COL,
    ID_COL,
    INTERSECTION_COL,
    MEAN_SPEED_COL,
    POLYGON_COL,
    SPEED_COL,
    TIME_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData

_log = logging.getLogger(__name__)

PEDPY_BLUE = (89 / 255, 178 / 255, 216 / 255)
PEDPY_ORANGE = (220 / 255, 160 / 255, 73 / 255)
PEDPY_GREEN = (108 / 255, 190 / 255, 167 / 255)
PEDPY_PETROL = (98 / 255, 190 / 255, 190 / 255)
PEDPY_GREY = (114 / 255, 125 / 255, 139 / 255)
PEDPY_RED = (233 / 255, 117 / 255, 134 / 255)


def _plot_series(  # pylint: disable=too-many-arguments
    axes: matplotlib.axes.Axes,
    title: str,
    x: pd.Series,
    y: pd.Series,
    color: str,
    x_label: str,
    y_label: str,
) -> matplotlib.axes.Axes:
    axes.set_title(title)
    axes.plot(x, y, color=color)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return axes


def plot_nt(
    *,
    nt: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the number of pedestrians over time.

    Args:
        nt (pd.DataFrame): cumulative number of pedestrians over time
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the number of pedestrians is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.get("color", PEDPY_BLUE)
    title = kwargs.get("title", "N-t")
    x_label = kwargs.get("x_label", "t / s")
    y_label = kwargs.get("y_label", "# pedestrians")
    return _plot_series(
        axes=axes,
        title=title,
        x=nt[TIME_COL],
        y=nt[CUMULATED_COL],
        color=color,
        x_label=x_label,
        y_label=y_label,
    )


def plot_density(
    *,
    density: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the density over time.

    Args:
        density(pd.DataFrame) : density per frame
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.get("color", PEDPY_BLUE)
    title = kwargs.get("title", "density over time")
    x_label = kwargs.get("x_label", "frame")
    y_label = kwargs.get("y_label", "$\\rho$ / 1/$m^2$")

    return _plot_series(
        axes=axes,
        title=title,
        x=density.index,
        y=density[DENSITY_COL],
        color=color,
        x_label=x_label,
        y_label=y_label,
    )


def plot_speed(
    *,
    speed: pd.Series,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the speed over time.

    Args:
        speed(pd.Series): speed per frame
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.get("color", PEDPY_BLUE)
    title = kwargs.get("title", "speed over time")
    x_label = kwargs.get("x_label", "frame")
    y_label = kwargs.get("y_label", "v / m/s")

    return _plot_series(
        axes=axes,
        title=title,
        x=speed.index,
        y=speed,
        color=color,
        x_label=x_label,
        y_label=y_label,
    )


def _plot_violin_xy(
    *,
    data: pd.Series,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    if axes is None:
        axes = plt.gca()

    facecolor = kwargs.get("facecolor", PEDPY_BLUE)
    edgecolor = kwargs.get("edgecolor", PEDPY_RED)
    title = kwargs.get("title", "")
    x_label = kwargs.get("x_label", "")
    y_label = kwargs.get("y_label", "")

    axes.set_title(title)
    violin_parts = plt.violinplot(
        data,
        showmeans=True,
        showextrema=True,
        showmedians=True,
    )
    for parts in violin_parts["bodies"]:
        parts.set_facecolor(facecolor)
        parts.set_edgecolor(edgecolor)

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_xticks([])
    return axes


def plot_speed_distribution(
    *,
    speed: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the speed distribution as violin plot.

    Args:
        speed(pd.DataFrame): speed of the pedestrians
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        facecolor (optional): color of the plot body
        edgecolor (optional): color of the edges of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis


    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if "title" not in kwargs:
        kwargs["title"] = "Individual speed"
    if "y_label" not in kwargs:
        kwargs["y_label"] = "m/s"
    return _plot_violin_xy(data=speed.speed, axes=axes, **kwargs)


def plot_density_distribution(
    *,
    density: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the density distribution as violin plot.

    Args:
        density(pd.DataFrame): individual density of the pedestrian
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        facecolor (optional): color of the plot body
        edgecolor (optional): color of the edges of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis


    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if "title" not in kwargs:
        kwargs["title"] = "Individual density"
    if "y_label" not in kwargs:
        kwargs["y_label"] = "$\\rho$ / 1/$m^2$"
    return _plot_violin_xy(data=density.density, axes=axes, **kwargs)


def plot_flow(
    *,
    flow: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the flow.

    Args:
        flow(pd.DataFrame): flow for some given crossing_frames and nt
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the flow is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.get("color", PEDPY_BLUE)
    title = kwargs.get("title", "flow")
    x_label = kwargs.get("x_label", "J / 1/s")
    y_label = kwargs.get("y_label", "v / m/s")
    axes.set_title(title)
    axes.scatter(flow[FLOW_COL], flow[MEAN_SPEED_COL], color=color)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return axes


def plot_neighborhood(
    *,
    pedestrian_id: int,
    neighbors: pd.DataFrame,
    frame: int,
    voronoi_data: pd.DataFrame,
    walkable_area: WalkableArea,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the neighborhood.

    Args:
        pedestrian_id(int): id of pedestrian to plot neighbors for
        neighbors(pd.DataFrame): neighborhood data based on the Voronoi cells
        frame(int): frame for which the plot is created
        voronoi_data (pd.DataFrame): individual Voronoi polygon for each person and frame
        walkable_area(WalkableArea): WalkableArea object of plot
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        hole_color (optional): color of the holes in the walkable area
        base_color (optional): color of the base pedestrians
        neighbor_color (optional): color of neighbor pedestrians
        default_color (optional): color of default pedestrians
    Returns:
        matplotlib.axes.Axes: instances where the neighborhood is plotted
    """
    hole_color = kwargs.get("hole_color", "w")
    base_color = kwargs.get("base_color", PEDPY_RED)
    neighbor_color = kwargs.get("neighbor_color", PEDPY_GREEN)
    default_color = kwargs.get("default_color", PEDPY_GREY)
    voronoi_neighbors = pd.merge(
        voronoi_data[voronoi_data.frame == frame],
        neighbors[neighbors.frame == frame],
        on=[ID_COL, FRAME_COL],
    )

    base_neighbors = voronoi_neighbors[
        voronoi_neighbors[ID_COL] == pedestrian_id
    ]["neighbors"].values[0]
    if axes is None:
        axes = plt.gca()
    axes.set_title(f"Neighbors of pedestrian {pedestrian_id}")

    plot_walkable_area(
        axes=axes, walkable_area=walkable_area, hole_color=hole_color
    )

    for _, row in voronoi_neighbors.iterrows():
        poly = row[POLYGON_COL]
        ped_id = row[ID_COL]

        are_neighbors = ped_id in base_neighbors

        color = default_color
        alpha = 0.2
        if ped_id == pedestrian_id:
            color = base_color
            alpha = 0.5

        if are_neighbors:
            color = neighbor_color
            alpha = 0.5

        axes.plot(*poly.exterior.xy, alpha=1, color=color)
        axes.fill(*poly.exterior.xy, alpha=alpha, color=color)
        axes.set_aspect("equal")

    return axes


def plot_time_distance(
    *,
    time_distance: pd.DataFrame,
    frame_rate: float,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plots the time to reach a target over distance.

    Args:
        time_distance(pd.DataFrame): DataFrame containing information on time and
            distance to some target
        frame_rate(float): frame_rate of the trajectory
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        marker_color (optional): color of the markers on the plot
        line_color (optional): color of the lines on the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the distance is plotted
    """
    if axes is None:
        axes = plt.gcf().add_subplot(111)

    line_color = kwargs.get("line_color", PEDPY_GREY)
    marker_color = kwargs.get("marker_color", PEDPY_GREY)
    title = kwargs.get("title", "distance plot")
    x_label = kwargs.get("x_label", "distance / m")
    y_label = kwargs.get("y_label", "time / s")

    axes.set_title(title)
    for _, ped_data in time_distance.groupby(by=ID_COL):
        axes.plot(
            ped_data.distance,
            ped_data.time / frame_rate,
            color=line_color,
            alpha=0.7,
            lw=0.25,
        )
        min_data = ped_data[ped_data.frame == ped_data.frame.min()]
        axes.scatter(
            min_data.distance,
            min_data.time / frame_rate,
            color=marker_color,
            s=5,
            marker="o",
        )

    axes.grid()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    axes.set_xlim([0, None])
    axes.set_ylim([0, None])

    return axes


def plot_profiles(
    *,
    walkable_area: WalkableArea,
    profiles: list[NDArray[np.float64]],
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the mean of the profiles.

    Args:
        walkable_area(WalkableArea): walkable area of the plot
        profiles(list): List of profiles like speed or density profiles
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        title (optional): title of the plot
        walkable_color (optional): color of the walkable area in the plot
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
         matplotlib.axes.Axes instance where the profiles are plotted
    """
    title = kwargs.get("title", "")
    walkable_color = kwargs.get("walkable_color", "w")
    hole_color = kwargs.get("hole_color", "w")
    hole_alpha = kwargs.get("hole_alpha", 1.0)
    bounds = walkable_area.bounds
    vmin = kwargs.get("vmin", np.min(profiles))
    vmax = kwargs.get("vmax", np.max(profiles))
    label = kwargs.get("label", None)

    if axes is None:
        axes = plt.gca()

    axes.set_title(title)
    imshow = axes.imshow(
        np.mean(profiles, axis=0),
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
        interpolation="None",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig = plt.gcf()

    fig.colorbar(imshow, cax=cax, orientation="vertical", label=label)

    axes.plot(*walkable_area.polygon.exterior.xy, color=walkable_color)
    plot_walkable_area(
        walkable_area=walkable_area,
        axes=axes,
        hole_color=hole_color,
        hole_alpha=hole_alpha,
    )

    return axes


def plot_walkable_area(
    *,
    walkable_area: WalkableArea,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the given walkable area in 2-D.

    Args:
        walkable_area (WalkableArea): WalkableArea object to plot
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be created
        line_color (optional): color of the borders
        line_color (optional): line width of the borders
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the walkable area is plotted
    """
    if axes is None:
        axes = plt.gca()

    line_color = kwargs.get("line_color", PEDPY_GREY)
    line_width = kwargs.get("line_width", 1.0)

    hole_color = kwargs.get("hole_color", "lightgrey")
    hole_alpha = kwargs.get("hole_alpha", 1.0)

    axes.plot(
        *walkable_area.polygon.exterior.xy,
        color=line_color,
        linewidth=line_width,
    )

    for hole in walkable_area.polygon.interiors:
        axes.plot(*hole.xy, color=line_color, linewidth=line_width)
        # Paint all holes first white, then with the desired color
        axes.fill(*hole.xy, color="w", alpha=1)
        axes.fill(*hole.xy, color=hole_color, alpha=hole_alpha)

    axes.set_xlabel(r"x/m")
    axes.set_ylabel(r"y/m")

    return axes


def plot_trajectories(
    *,
    traj: TrajectoryData,
    walkable_area: Optional[WalkableArea] = None,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the given trajectory and walkable area in 2-D.

    Args:
        traj (TrajectoryData): Trajectory object to plot
        walkable_area (WalkableArea, optional): WalkableArea object to plot
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
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
    traj_color = kwargs.get("traj_color", PEDPY_RED)
    traj_width = kwargs.get("traj_width", 1.0)
    traj_alpha = kwargs.get("traj_alpha", 1.0)

    traj_start_marker = kwargs.get("traj_start_marker", "")
    traj_end_marker = kwargs.get("traj_end_marker", "")

    if axes is None:
        axes = plt.gca()

    if walkable_area is not None:
        axes = plot_walkable_area(
            walkable_area=walkable_area, axes=axes, **kwargs
        )

    for _, ped in traj.data.groupby(ID_COL):
        axes.plot(
            ped[X_COL],
            ped[Y_COL],
            alpha=traj_alpha,
            color=traj_color,
            linewidth=traj_width,
        )
        axes.scatter(
            ped[ped.frame == ped.frame.min()][X_COL],
            ped[ped.frame == ped.frame.min()][Y_COL],
            color=traj_color,
            marker=traj_start_marker,
        )
        axes.scatter(
            ped[ped.frame == ped.frame.max()][X_COL],
            ped[ped.frame == ped.frame.max()][Y_COL],
            color=traj_color,
            marker=traj_end_marker,
        )

    axes.set_xlabel(r"x/m")
    axes.set_ylabel(r"y/m")

    return axes


def plot_measurement_setup(
    *,
    traj: Optional[TrajectoryData] = None,
    walkable_area: Optional[WalkableArea] = None,
    measurement_areas: Optional[List[MeasurementArea]] = None,
    measurement_lines: Optional[List[MeasurementLine]] = None,
    axes: Optional[matplotlib.axes.Axes] = None,
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
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
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
    ma_line_color = kwargs.get("ma_line_color", PEDPY_BLUE)
    ma_line_width = kwargs.get("ma_line_width", 1.0)
    ma_color = kwargs.get("ma_color", PEDPY_BLUE)
    ma_alpha = kwargs.get("ma_alpha", 0.2)

    ml_color = kwargs.get("ml_color", PEDPY_BLUE)
    ml_width = kwargs.get("ml_width", 1.0)

    if axes is None:
        axes = plt.gca()

    if walkable_area is not None:
        plot_walkable_area(walkable_area=walkable_area, axes=axes, **kwargs)

    if traj is not None:
        plot_trajectories(traj=traj, walkable_area=None, axes=axes, **kwargs)

    if measurement_areas is not None:
        for measurement_area in measurement_areas:
            axes.plot(
                *measurement_area.polygon.exterior.xy,
                color=ma_line_color,
                linewidth=ma_line_width,
            )
            axes.fill(
                *measurement_area.polygon.exterior.xy,
                color=ma_color,
                alpha=ma_alpha,
            )

    if measurement_lines is not None:
        for measurement_line in measurement_lines:
            axes.plot(*measurement_line.xy, color=ml_color, linewidth=ml_width)

    axes.set_xlabel(r"x / m")
    axes.set_ylabel(r"y / m")

    return axes


def plot_voronoi_cells(  # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    *,
    voronoi_data: pd.DataFrame,
    frame: int,
    traj_data: Optional[TrajectoryData] = None,
    walkable_area: Optional[WalkableArea] = None,
    measurement_area: Optional[MeasurementArea] = None,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the Voronoi cells, walkable able, and measurement area in 2D.

    Args:
        voronoi_polygons (pd.DataFrame): voronoi polygon data as returned by
            :func:`~density_calculator.compute_voronoi_density`
        frame (int): frame index
        walkable_area (WalkableArea, optional): WalkableArea object to plot
        measurement_area (MeasurementArea, optional): measurement area used to
            compute the Voronoi cells
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        traj_data (TrajectoryData, optional): Will add pedestrian positions to the plot
            if provided.
        ped_color (optional): color used to display current ped positions
        voronoi_border_color (optional): border color of Voronoi cells
        voronoi_inside_ma_alpha (optional): alpha of part of Voronoi cell
            inside the measurement area, data needs to contain column
            "intersection"!
        voronoi_outside_ma_alpha (optional): alpha of part of Voronoi cell
            outside the measurement area
        color_by_column (str, optional): Optioanlly provide a column name to specify
            the data to color the cell. Only supports Integer and Float data types.
            E.g. color_by_column `DENSITY_COL`
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
    ped_color = kwargs.get("ped_color", PEDPY_BLUE)
    ped_size = kwargs.get("ped_size", 5)
    voronoi_border_color = kwargs.get("voronoi_border_color", PEDPY_BLUE)
    voronoi_inside_ma_alpha = kwargs.get("voronoi_inside_ma_alpha", 1)
    voronoi_outside_ma_alpha = kwargs.get("voronoi_outside_ma_alpha", 1)

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    cb_location = kwargs.get("cb_location", "right")
    show_colorbar = kwargs.get("show_colorbar", True)
    color_by_column = kwargs.get("color_by_column", None)

    if axes is None:
        axes = plt.gca()

    if measurement_area is not None:
        plot_measurement_setup(
            measurement_areas=[measurement_area], axes=axes, **kwargs
        )

    if traj_data:
        data = pd.merge(
            traj_data.data,
            voronoi_data[voronoi_data.frame == frame],
            on=[ID_COL, FRAME_COL],
        )
    else:
        data = voronoi_data[voronoi_data.frame == frame]

    typ = ""
    if color_by_column:
        typ = data.dtypes[color_by_column]
        if typ == "float64":
            if not vmin:
                vmin = voronoi_data[color_by_column].min()
            if not vmax:
                vmax = voronoi_data[color_by_column].max()
            voronoi_colormap = plt.get_cmap("YlGn")
            norm = mpl.colors.Normalize(vmin, vmax)
            scalar_mappable = mpl.cm.ScalarMappable(
                norm=norm, cmap=voronoi_colormap
            )
            color_mapper = scalar_mappable.to_rgba
        elif typ == "int64":
            voronoi_colormap = plt.get_cmap("tab20c")

            def _map(x):
                return voronoi_colormap(x % 20)

            color_mapper = _map
        else:
            pass

    for _, row in data.iterrows():
        poly = row[POLYGON_COL]

        if color_by_column:
            color = color_mapper(row[color_by_column])
        else:
            color = "w"

        axes.plot(*poly.exterior.xy, alpha=1, color=voronoi_border_color)
        axes.fill(
            *poly.exterior.xy, facecolor=color, alpha=voronoi_outside_ma_alpha
        )

        if INTERSECTION_COL in data.columns:
            if not shapely.is_empty(row[INTERSECTION_COL]):
                intersection_poly = row[INTERSECTION_COL]
                axes.fill(
                    *intersection_poly.exterior.xy,
                    facecolor=color,
                    alpha=voronoi_inside_ma_alpha,
                )

        if traj_data:
            axes.scatter(row[X_COL], row[Y_COL], color=ped_color, s=ped_size)

    if show_colorbar and color_by_column and typ == "float64":
        if color_by_column == DENSITY_COL:
            label = "$\\rho$ / $\\frac{1}{m^2}$"
        elif color_by_column == SPEED_COL:
            label = r"v / $\frac{m}{s}$"
        elif color_by_column == ID_COL:
            label = "Id"
        else:
            label = ""
        plt.colorbar(
            scalar_mappable,
            ax=axes,
            label=label,
            shrink=0.4,
            location=cb_location,
        )

    if walkable_area is not None:
        plot_walkable_area(axes=axes, walkable_area=walkable_area, **kwargs)
    axes.set_xlabel(r"x / m")
    axes.set_ylabel(r"y / m")
    return axes
