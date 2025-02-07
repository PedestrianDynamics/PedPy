"""Module containing plotting functionalities."""

# pylint: disable=C0302

import logging
import warnings
from typing import Any, List, Optional

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from pedpy.column_identifier import (
    CUMULATED_COL,
    DENSITY_COL,
    DENSITY_SP1_COL,
    DENSITY_SP2_COL,
    FLOW_COL,
    FLOW_SP1_COL,
    FLOW_SP2_COL,
    FRAME_COL,
    ID_COL,
    INTERSECTION_COL,
    MEAN_SPEED_COL,
    NEIGHBORS_COL,
    NEIGHBOR_ID_COL,
    POLYGON_COL,
    SPEED_COL,
    SPEED_SP1_COL,
    SPEED_SP2_COL,
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


def _plot_polygon(
    *,
    axes: matplotlib.axes.Axes,
    polygon: shapely.Polygon,
    polygon_color: Any,
    polygon_alpha: float = 1,
    line_color: Any = PEDPY_GREY,
    line_width: float = 1,
    hole_color: Any = "lightgrey",
    hole_alpha: float = 1,
    zorder: float = 1000,
) -> matplotlib.axes.Axes:
    """Plot the shapely polygon (including holes).

    Args:
        polygon (shapely.Polygon): polygon to plot
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        polygon_color (Any): background color of the polygon
        polygon_alpha (float): alpha of the background for the polygon
        line_color (Any): color of the borders
        line_width (float): line width of the borders
        hole_color (Any): background color of holes
        hole_alpha (float): alpha of background color for holes
        zorder (float): Specifies the drawing order of the polygon,
            lower values are drawn first

    Returns:
        matplotlib.axes.Axes instance where the polygon is plotted

    """
    # Plot the boundary of the polygon/holes separately to get the same color
    # as the outside as alpha modifies all colors

    # Plot the exterior of the polygon
    exterior_coords = list(polygon.exterior.coords)
    exterior_polygon_border = Polygon(
        exterior_coords,
        edgecolor=line_color,
        facecolor="none",
        linewidth=line_width,
        closed=True,
        zorder=zorder,
    )
    axes.add_patch(exterior_polygon_border)

    exterior_polygon_fill = Polygon(
        exterior_coords,
        edgecolor="none",
        facecolor=polygon_color,
        linewidth=line_width,
        alpha=polygon_alpha,
        closed=True,
        zorder=zorder,
    )
    axes.add_patch(exterior_polygon_fill)

    # Plot the interiors (holes) of the polygon
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        interior_polygon_border = Polygon(
            interior_coords,
            edgecolor=line_color,
            facecolor="none",
            linewidth=line_width,
            alpha=1,
            closed=True,
            zorder=zorder,
        )
        axes.add_patch(interior_polygon_border)

        interior_polygon_fill = Polygon(
            interior_coords,
            edgecolor="none",
            facecolor=hole_color,
            linewidth=line_width,
            alpha=hole_alpha,
            closed=True,
            zorder=zorder,
        )
        axes.add_patch(interior_polygon_fill)

    return axes


def _plot_series(  # pylint: disable=too-many-arguments
    axes: matplotlib.axes.Axes,
    title: str,
    x: pd.Series,
    y: pd.Series,
    color: str,
    x_label: str,
    y_label: str,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    axes.set_title(title)
    axes.plot(x, y, color=color, **kwargs)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return axes


def _plot_multiple_series(  # pylint: disable=too-many-arguments
    axes: matplotlib.axes.Axes,
    title: str,
    x: pd.Series,
    y_s: list[pd.Series],
    colors: list[str],
    labels: list[str],
    line_width: float,
    x_label: str,
    y_label: str,
) -> matplotlib.axes.Axes:
    axes.set_title(title)
    for y, color, label in zip(y_s, colors, labels, strict=False):
        axes.plot(x, y, color=color, label=label, linewidth=line_width)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()
    return axes


def plot_speed_at_line(
    *,
    speed_at_line: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the speed of both species and the total speed at the line.

    Args:
        speed_at_line(pd.DataFrame): DataFrame containing information on
            speed at the line
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        title (optional): title of the plot
        color_species1 (optional): color of the speed of species 1 in the plot
        color_species2 (optional): color of the speed of species 2 in the plot
        color_total (optional): color of the total speed in the plot
        label_species1 (optional): tag of species 1 in the legend
        label_species2 (optional): tag of species 2 in the legend
        label_total (optional): tag of total speed in the legend

    Returns:
         matplotlib.axes.Axes instance where the speeds are plotted
    """
    if axes is None:
        axes = plt.gca()

    color_sp1 = kwargs.get("color_species1", PEDPY_BLUE)
    color_sp2 = kwargs.get("color_species2", PEDPY_ORANGE)
    color_total = kwargs.get("color_total", PEDPY_GREEN)
    title = kwargs.get("title", "Speed at Line")
    x_label = kwargs.get("x_label", "Frame")
    y_label = kwargs.get("y_label", "v / m/s")
    label_sp1 = kwargs.get("lable_species1", "species 1")
    label_sp2 = kwargs.get("lable_species2", "species 2")
    label_total = kwargs.get("lable_total", "total")
    line_width = kwargs.get("line_width", 0.5)

    return _plot_multiple_series(
        axes=axes,
        title=title,
        x=speed_at_line[FRAME_COL],
        y_s=[
            speed_at_line[SPEED_SP1_COL],
            speed_at_line[SPEED_SP2_COL],
            speed_at_line[SPEED_COL],
        ],
        colors=[color_sp1, color_sp2, color_total],
        labels=[label_sp1, label_sp2, label_total],
        x_label=x_label,
        y_label=y_label,
        line_width=line_width,
    )


def plot_density_at_line(
    *,
    density_at_line: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the density of both species and the total density at the line.

    Args:
        density_at_line(pd.DataFrame): DataFrame containing information on
            density at the line
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        title (optional): title of the plot
        color_species1 (optional): color of the density of species 1 in the plot
        color_species2 (optional): color of the density of species 2 in the plot
        color_total (optional): color of the total density in the plot
        label_species1 (optional): tag of species 1 in the legend
        label_species2 (optional): tag of species 2 in the legend
        label_total (optional): tag of total speed in the legend

    Returns:
         matplotlib.axes.Axes instance where the densities are plotted
    """
    if axes is None:
        axes = plt.gca()

    color_sp1 = kwargs.get("color_species1", PEDPY_BLUE)
    color_sp2 = kwargs.get("color_species2", PEDPY_ORANGE)
    color_total = kwargs.get("color_total", PEDPY_GREEN)
    title = kwargs.get("title", "Density at Line")
    x_label = kwargs.get("x_label", "Frame")
    y_label = kwargs.get("y_label", "$\\rho$ / 1/$m^2$")
    label_sp1 = kwargs.get("lable_species1", "species 1")
    label_sp2 = kwargs.get("lable_species2", "species 2")
    label_total = kwargs.get("lable_total", "total")
    line_width = kwargs.get("line_width", 0.5)

    return _plot_multiple_series(
        axes=axes,
        title=title,
        x=density_at_line[FRAME_COL],
        y_s=[
            density_at_line[DENSITY_SP1_COL],
            density_at_line[DENSITY_SP2_COL],
            density_at_line[DENSITY_COL],
        ],
        colors=[color_sp1, color_sp2, color_total],
        labels=[label_sp1, label_sp2, label_total],
        x_label=x_label,
        y_label=y_label,
        line_width=line_width,
    )


def plot_flow_at_line(
    *,
    flow_at_line: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the flow of both species and the total flow at the line.

    Args:
        flow_at_line(pd.DataFrame): DataFrame containing information on
            flow at the line
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        title (optional): title of the plot
        color_species1 (optional): color of the flow of species 1 in the plot
        color_species2 (optional): color of the flow of species 2 in the plot
        color_total (optional): color of the total flow in the plot
        label_species1 (optional): tag of species 1 in the legend
        label_species2 (optional): tag of species 2 in the legend
        label_total (optional): tag of total speed in the legend

    Returns:
         matplotlib.axes.Axes instance where the profiles are plotted
    """
    if axes is None:
        axes = plt.gca()

    color_sp1 = kwargs.get("color_species1", PEDPY_BLUE)
    color_sp2 = kwargs.get("color_species2", PEDPY_ORANGE)
    color_total = kwargs.get("color_total", PEDPY_GREEN)
    title = kwargs.get("title", "Flow at Line")
    x_label = kwargs.get("x_label", "Frame")
    y_label = kwargs.get("y_label", "J / 1/s")
    label_sp1 = kwargs.get("lable_species1", "species 1")
    label_sp2 = kwargs.get("lable_species2", "species 2")
    label_total = kwargs.get("lable_total", "total")
    line_width = kwargs.get("line_width", 0.5)

    return _plot_multiple_series(
        axes=axes,
        title=title,
        x=flow_at_line[FRAME_COL],
        y_s=[
            flow_at_line[FLOW_SP1_COL],
            flow_at_line[FLOW_SP2_COL],
            flow_at_line[FLOW_COL],
        ],
        colors=[color_sp1, color_sp2, color_total],
        labels=[label_sp1, label_sp2, label_total],
        x_label=x_label,
        y_label=y_label,
        line_width=line_width,
    )


def plot_nt(
    *,
    nt: pd.DataFrame,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the number of pedestrians over time.

    Args:
        nt (pd.DataFrame): cumulative number of pedestrians over time
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the number of pedestrians is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.pop("color", PEDPY_BLUE)
    title = kwargs.pop("title", "N-t")
    x_label = kwargs.pop("x_label", "t / s")
    y_label = kwargs.pop(
        "y_label",
        r"\# pedestrians" if plt.rcParams["text.usetex"] else "# pedestrians",
    )
    return _plot_series(
        axes=axes,
        title=title,
        x=nt[TIME_COL],
        y=nt[CUMULATED_COL],
        color=color,
        x_label=x_label,
        y_label=y_label,
        **kwargs,
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.pop("color", PEDPY_BLUE)
    title = kwargs.pop("title", "density over time")
    x_label = kwargs.pop("x_label", "frame")
    y_label = kwargs.pop("y_label", "$\\rho$ / 1/$m^2$")

    return _plot_series(
        axes=axes,
        title=title,
        x=density.index,
        y=density[DENSITY_COL],
        color=color,
        x_label=x_label,
        y_label=y_label,
        **kwargs,
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.pop("color", PEDPY_BLUE)
    title = kwargs.pop("title", "speed over time")
    x_label = kwargs.pop("x_label", "frame")
    y_label = kwargs.pop("y_label", "v / m/s")

    return _plot_series(
        axes=axes,
        title=title,
        x=speed.index,
        y=speed,
        color=color,
        x_label=x_label,
        y_label=y_label,
        **kwargs,
    )


def _plot_violin_xy(
    *,
    data: pd.Series,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    if axes is None:
        axes = plt.gca()

    facecolor = kwargs.pop("facecolor", PEDPY_BLUE)
    edgecolor = kwargs.pop("edgecolor", PEDPY_RED)
    title = kwargs.pop("title", "")
    x_label = kwargs.pop("x_label", "")
    y_label = kwargs.pop("y_label", "")

    axes.set_title(title)
    violin_parts = axes.violinplot(
        data,
        showmeans=True,
        showextrema=True,
        showmedians=True,
        **kwargs,
    )
    for parts in violin_parts["bodies"]:  # type: ignore[attr-defined]
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the flow is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.pop("color", PEDPY_BLUE)
    title = kwargs.pop("title", "flow")
    x_label = kwargs.pop("x_label", "J / 1/s")
    y_label = kwargs.pop("y_label", "v / m/s")
    axes.set_title(title)
    axes.scatter(flow[FLOW_COL], flow[MEAN_SPEED_COL], color=color, **kwargs)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    return axes


def plot_acceleration(
    *,
    acceleration: pd.Series,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the acceleration over time.

    Args:
        acceleration(pd.Series): acceleration per frame
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        color (optional): color of the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the density is plotted
    """
    if axes is None:
        axes = plt.gca()

    color = kwargs.pop("color", PEDPY_BLUE)
    title = kwargs.pop("title", "acceleration over time")
    x_label = kwargs.pop("x_label", "frame")
    y_label = kwargs.pop("y_label", "a / $m/s^2$")

    return _plot_series(
        axes=axes,
        title=title,
        x=acceleration.index,
        y=acceleration,
        color=color,
        x_label=x_label,
        y_label=y_label,
        **kwargs,
    )


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
    """Plots the neighborhood of a specified pedestrian.

    This function visualizes the neighborhood for a given pedestrian at a
    specified frame using Voronoi polygons for each pedestrian. It colors
    the specified pedestrian, their neighbors, and other pedestrians
    distinctly based on the neighborhood data.

    Args:
        pedestrian_id(int): id of pedestrian to plot neighbors for
        neighbors(pd.DataFrame): neighborhood data based on the Voronoi cells
        frame(int): frame for which the plot is created
        voronoi_data (pd.DataFrame): individual Voronoi polygon for each person
            and frame
        walkable_area(WalkableArea): WalkableArea object of plot
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        hole_color (optional): color of the holes in the walkable area
        base_color (optional): color of the base pedestrians
        neighbor_color (optional): color of neighbor pedestrians
        default_color (optional): color of default pedestrians
    Returns:
        matplotlib.axes.Axes: instances where the neighborhood is plotted
    """
    if NEIGHBORS_COL in neighbors.columns:
        # Extract neighbors from when they are stored as list in a column
        neighbors_in_frame = neighbors[neighbors[FRAME_COL] == frame].set_index(
            ID_COL
        )
        neighbor_ids = neighbors_in_frame[NEIGHBORS_COL].to_dict()
    elif NEIGHBOR_ID_COL in neighbors.columns:
        # Extract neighbors from when they are stored as one neighbor per row
        neighbors_in_frame = neighbors[neighbors[FRAME_COL] == frame]
        neighbor_ids = (
            neighbors_in_frame.groupby(ID_COL)[NEIGHBOR_ID_COL]
            .apply(set)
            .to_dict()
        )
    else:
        raise RuntimeError("Unknown neighbor data format")

    return _plot_neighborhood(
        pedestrian_id=pedestrian_id,
        neighbor_ids=neighbor_ids,
        frame=frame,
        voronoi_data=voronoi_data,
        walkable_area=walkable_area,
        axes=axes,
        **kwargs,
    )


def _plot_neighborhood(
    *,
    pedestrian_id: int,
    neighbor_ids: dict[int, List[int]],
    frame: int,
    voronoi_data: pd.DataFrame,
    walkable_area: WalkableArea,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plot the neighborhood of a pedestrian.

    Args:
        pedestrian_id(int): id of pedestrian to plot neighbors for
        neighbor_ids(dict[int, List[int]]): neighborhood with base id as key
        frame(int): frame for which the plot is created
        voronoi_data (pd.DataFrame): individual Voronoi polygon for each person
            and frame
        walkable_area(WalkableArea): WalkableArea object of plot
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        hole_color (optional): color of the holes in the walkable area
        base_color (optional): color of the base pedestrians
        neighbor_color (optional): color of neighbor pedestrians
        default_color (optional): color of default pedestrians
    Returns:
        matplotlib.axes.Axes: instances where the neighborhood is plotted
    """
    # Extract color settings from kwargs
    hole_color = kwargs.pop("hole_color", "w")
    base_color = kwargs.pop("base_color", PEDPY_RED)
    neighbor_color = kwargs.pop("neighbor_color", PEDPY_GREEN)
    default_color = kwargs.pop("default_color", PEDPY_GREY)

    # Filter voronoi_data for polygons in the same frame
    voronoi_neighbors = voronoi_data[voronoi_data[FRAME_COL] == frame]

    # Prepare arrays for colors and alphas to avoid conditionals in the loop
    ped_ids = voronoi_neighbors[ID_COL].to_numpy()
    polygons = voronoi_neighbors[POLYGON_COL].to_numpy()
    colors = np.full((len(ped_ids), 3), default_color, dtype=float)
    alphas = np.full(len(ped_ids), 0.2)

    # Set base pedestrian color and neighbors colors
    for idx, ped_id in enumerate(ped_ids):
        if ped_id == pedestrian_id:
            colors[idx] = base_color
            alphas[idx] = 0.5
        elif ped_id in neighbor_ids.get(pedestrian_id, []):
            colors[idx] = neighbor_color
            alphas[idx] = 0.5

    # Set up the plot
    if axes is None:
        axes = plt.gca()
    axes.set_title(f"Neighbors of pedestrian {pedestrian_id}")

    # Plot the walkable area
    plot_walkable_area(
        axes=axes,
        walkable_area=walkable_area,
        hole_color=hole_color,
    )

    # Plot each polygon with precomputed colors and alphas
    for poly, color, alpha in zip(polygons, colors, alphas, strict=False):
        _plot_polygon(
            axes=axes,
            polygon=poly,
            line_color=color,
            polygon_color=color,
            polygon_alpha=alpha,
        )

    # Set aspect ratio
    axes.set_aspect("equal")

    return axes


def plot_time_distance(  # noqa: PLR0915
    *,
    time_distance: pd.DataFrame,
    speed: Optional[pd.DataFrame] = None,
    axes: Optional[matplotlib.axes.Axes] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Plots the time to reach a target over distance.

    If a speed DataFrame is provided,
    lines will be colored according to speed values.

    Args:
        time_distance (pd.DataFrame): DataFrame containing information on time
            and distance to some target
        speed (pd.DataFrame): DataFrame containing speed calculation.
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        marker_color (optional): color of the markers on the plot
        line_color (optional): color of the lines on the plot
        title (optional): title of the plot
        x_label (optional): label on the x-axis
        y_label (optional): label on the y-axis

    Returns:
        matplotlib.axes.Axes instance where the distance is plotted
    """

    def _setup_plot(axes: matplotlib.axes.Axes, **kwargs: Any) -> None:
        """Configures the initial settings of the plot, incl. title and labels.

        Args:
        axes: The matplotlib axes to configure.
        **kwargs: Keyword arguments containing
        'title', 'x_label' and 'y_label'.
        """
        title = kwargs.get("title", "Distance Plot")
        x_label = kwargs.get("x_label", "Distance / m")
        y_label = kwargs.get("y_label", "Time / s")

        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)

    def _scatter_min_data(
        axes: matplotlib.axes.Axes,
        ped_data: pd.DataFrame,
        color: str,
    ) -> None:
        """Adds a scatter plot marker at the start of a pedestrian's line.

        Args:
        axes: The matplotlib axes to plot on.
        ped_data: DataFrame containing a single pedestrian's data.
        color: Color of the scatter plot marker.
        """
        min_data = ped_data[ped_data.frame == ped_data.frame.min()]
        axes.scatter(
            min_data.distance,
            min_data.time,
            color=color,
            s=5,
            marker="o",
        )

    def _scatter_min_data_with_color(
        axes: matplotlib.axes.Axes,
        ped_data: pd.DataFrame,
        norm: Normalize,
    ) -> None:
        """Adds a scatter plot marker at the start of a pedestrian's line.

        Args:
        axes: The matplotlib axes to plot on.
        ped_data: DataFrame containing a single pedestrian's data.
        norm: Normalization for the colormap based on speed.
        cmap: The colormap to use for coloring the line based on speed.
        frame_rate: Frame rate used to adjust time values.
        color: Color of the scatter plot marker.
        """
        min_data = ped_data[ped_data.frame == ped_data.frame.min()]
        axes.scatter(
            min_data.distance,
            min_data.time,
            c=min_data.speed,
            cmap="jet",
            norm=norm,
            s=5,
            marker="o",
        )

    def _plot_line(
        axes: matplotlib.axes.Axes,
        ped_data: pd.DataFrame,
        color: str,
    ) -> None:
        """Plots a line for a single pedestrian's data.

        Args:
        axes: The matplotlib axes to plot on.
        ped_data: DataFrame containing a single pedestrian's distance and time
            data.
        color: Color of the line.
        """
        axes.plot(
            ped_data.distance,
            ped_data.time,
            color=color,
            alpha=0.7,
            lw=0.25,
        )

    def _plot_colored_line(
        axes: matplotlib.axes.Axes,
        ped_data: pd.DataFrame,
        norm: Normalize,
    ) -> None:
        """Plots a line for a single pedestrian's data.

        The color of the line segment indicated the pedestrian speed at
        this time.

        Args:
        axes: The matplotlib axes to plot on.
        ped_data: DataFrame containing a single pedestrian's distance and time
            data.
        norm: Normalization for the colormap based on speed.
        cmap: The colormap to use for coloring the line based on speed.
        """
        points = ped_data[["distance", "time"]].to_numpy()
        speed_id = ped_data.speed.to_numpy()
        segments = [
            [
                (points[i, 0], points[i, 1]),
                (points[i + 1, 0], points[i + 1, 1]),
            ]
            for i in range(len(points) - 1)
        ]
        line_collection = LineCollection(
            segments, cmap="jet", alpha=0.7, norm=norm
        )
        line_collection.set_array(speed_id)
        line_collection.set_linewidth(0.5)
        axes.add_collection(line_collection)

    def _add_colorbar(axes: matplotlib.axes.Axes, norm: Normalize) -> None:
        """Adds a colorbar to the plot, mapping the speed values to colors.

        Args:
        axes: The matplotlib axes to plot on.
        cmap: The colormap used for the plot.
        norm: Normalization used for the colormap.
        """
        scalar_map = plt.cm.ScalarMappable(cmap="jet", norm=norm)
        scalar_map.set_array([])
        cbar = plt.colorbar(scalar_map, ax=axes)
        cbar.set_label("Speed / m/s")

    def _finalize_plot(axes: matplotlib.axes.Axes) -> None:
        """Applies final adjustments to the plot.

        These final adjustments include autoscaling and setting margins.

        Args:
        axes: The matplotlib axes to adjust.
        """
        axes.autoscale()
        axes.margins(0.1)
        axes.grid(alpha=0.3)
        axes.set_xlim(0, None)
        axes.set_ylim(0, None)

    def _plot_with_speed_colors(
        axes: matplotlib.axes.Axes,
        time_distance: pd.DataFrame,
        speed: pd.DataFrame,
    ) -> None:
        """Plots pedestrian data with lines colored according to speed.

        Args:
        axes: The matplotlib axes to plot on.
        time_distance: DataFrame containing the pedestrian data.
        speed: DataFrame containing speed calculations.
        frame_rate: Frame rate used to adjust time values.
        """
        time_distance = time_distance.merge(speed, on=[ID_COL, FRAME_COL])
        norm = Normalize(
            vmin=time_distance.speed.min(), vmax=time_distance.speed.max()
        )

        for _, ped_data in time_distance.groupby(ID_COL):
            _plot_colored_line(axes, ped_data, norm)

        _scatter_min_data_with_color(axes, time_distance, norm)
        _add_colorbar(axes, norm)

    def _plot_without_colors(
        axes: matplotlib.axes.Axes,
        time_distance: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Plots pedestrian data without using speed colors.

        Args:
        axes: The matplotlib axes to plot on.
        time_distance: DataFrame containing the pedestrian data.
        frame_rate: Frame rate used to adjust time values.
        **kwargs: Additional customization options (line_color, marker_color).
        """
        line_color = kwargs.pop("line_color", PEDPY_GREY)
        marker_color = kwargs.pop("marker_color", PEDPY_GREY)
        for _, ped_data in time_distance.groupby(ID_COL):
            _plot_line(axes, ped_data, line_color)

        _scatter_min_data(axes, time_distance, marker_color)

    axes = axes or plt.gca()
    _setup_plot(axes, **kwargs)
    if speed is not None:
        _plot_with_speed_colors(axes, time_distance, speed)
    else:
        _plot_without_colors(axes, time_distance, **kwargs)

    _finalize_plot(axes)

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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        title (optional): title of the plot
        walkable_color (optional): color of the walkable area in the plot
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
         matplotlib.axes.Axes instance where the profiles are plotted
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_profiles = np.nanmean(profiles, axis=0)

    bounds = walkable_area.bounds

    title = kwargs.pop("title", "")
    walkable_color = kwargs.pop("walkable_color", "w")
    hole_color = kwargs.pop("hole_color", "w")
    hole_alpha = kwargs.pop("hole_alpha", 1.0)
    vmin = kwargs.pop("vmin", np.nanmin(mean_profiles))
    vmax = kwargs.pop("vmax", np.nanmax(mean_profiles))
    label = kwargs.pop("label", None)

    if axes is None:
        axes = plt.gca()

    axes.set_title(title)
    imshow = axes.imshow(
        mean_profiles,
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        cmap=kwargs.pop("cmap", "jet"),
        vmin=vmin,
        vmax=vmax,
        **kwargs,
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
        axes (matplotlib.axes.Axes): Axes to plot on, if None new will be
            created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        line_color (optional): color of the borders
        line_width (optional): line width of the borders
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the walkable area is plotted
    """
    if axes is None:
        axes = plt.gca()

    line_color = kwargs.pop("line_color", PEDPY_GREY)
    line_width = kwargs.pop("line_width", 1.0)

    hole_color = kwargs.pop("hole_color", "lightgrey")
    hole_alpha = kwargs.pop("hole_alpha", 1.0)

    axes = _plot_polygon(
        polygon=walkable_area.polygon,
        polygon_color="none",
        line_color=line_color,
        line_width=line_width,
        hole_color=hole_color,
        hole_alpha=hole_alpha,
        axes=axes,
    )

    axes.set_xlabel(r"x/m")
    axes.set_ylabel(r"y/m")

    axes.autoscale_view()

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
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
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
    traj_color = kwargs.pop("traj_color", PEDPY_RED)
    traj_width = kwargs.pop("traj_width", 1.0)
    traj_alpha = kwargs.pop("traj_alpha", 1.0)

    traj_start_marker = kwargs.pop("traj_start_marker", "")
    traj_end_marker = kwargs.pop("traj_end_marker", "")

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
        measurement_areas (List[MeasurementArea], optional): List of
            measurement areas to plot
        measurement_lines (List[MeasurementLine], optional): List of
            measurement lines to plot
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
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
        traj_end_marker (optional): marker to indicate the end of the
            trajectory
        line_color (optional): color of the borders
        line_width (optional): line width of the borders
        hole_color (optional): background color of holes
        hole_alpha (optional): alpha of background color for holes

    Returns:
        matplotlib.axes.Axes instance where the measurement setup is plotted
    """
    ma_line_color = kwargs.pop("ma_line_color", PEDPY_BLUE)
    ma_line_width = kwargs.pop("ma_line_width", 1.0)
    ma_color = kwargs.pop("ma_color", PEDPY_BLUE)
    ma_alpha = kwargs.pop("ma_alpha", 0.2)

    ml_color = kwargs.pop("ml_color", PEDPY_BLUE)
    ml_width = kwargs.pop("ml_width", 1.0)

    if axes is None:
        axes = plt.gca()

    if measurement_areas is not None:
        for measurement_area in measurement_areas:
            _plot_polygon(
                axes=axes,
                polygon=measurement_area.polygon,
                line_color=ma_line_color,
                line_width=ma_line_width,
                polygon_alpha=ma_alpha,
                polygon_color=ma_color,
            )

    if walkable_area is not None:
        plot_walkable_area(walkable_area=walkable_area, axes=axes, **kwargs)

    if traj is not None:
        plot_trajectories(traj=traj, walkable_area=None, axes=axes, **kwargs)

    if measurement_lines is not None:
        for measurement_line in measurement_lines:
            axes.plot(*measurement_line.xy, color=ml_color, linewidth=ml_width)

    axes.set_xlabel(r"x / m")
    axes.set_ylabel(r"y / m")

    return axes


def plot_voronoi_cells(  # noqa: PLR0912,PLR0915
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
        voronoi_data (pd.DataFrame): voronoi polygon data as returned by
            :func:`~density_calculator.compute_voronoi_density`
        frame (int): frame index
        traj_data (TrajectoryData, optional): Will add pedestrian positions
            to the plot if provided.
        walkable_area (WalkableArea, optional): WalkableArea object to plot
        measurement_area (MeasurementArea, optional): measurement area used to
            compute the Voronoi cells
        axes (matplotlib.axes.Axes, optional): Axes to plot on,
            if None new will be created
        kwargs: Additional parameters to change the plot appearance, see
            below for list of usable keywords

    Keyword Args:
        ped_color (optional): color used to display current ped positions
        voronoi_border_color (optional): border color of Voronoi cells
        voronoi_inside_ma_alpha (optional): alpha of part of Voronoi cell
            inside the measurement area, data needs to contain column
            "intersection"!
        voronoi_outside_ma_alpha (optional): alpha of part of Voronoi cell
            outside the measurement area
        color_by_column (str, optional): Optioanlly provide a column name to
            specify the data to color the cell. Only supports Integer and
            Float data types. E.g. color_by_column `DENSITY_COL`
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
        cmap (optional): colormap used for
    Returns:
        matplotlib.axes.Axes instance where the Voronoi cells are plotted
    """
    ped_color = kwargs.pop("ped_color", PEDPY_BLUE)
    ped_size = kwargs.pop("ped_size", 5)
    voronoi_border_color = kwargs.pop("voronoi_border_color", PEDPY_BLUE)
    voronoi_inside_ma_alpha = kwargs.pop("voronoi_inside_ma_alpha", 1)
    voronoi_outside_ma_alpha = kwargs.pop("voronoi_outside_ma_alpha", 1)

    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    cb_location = kwargs.pop("cb_location", "right")
    show_colorbar = kwargs.pop("show_colorbar", True)
    color_by_column = kwargs.pop("color_by_column", None)
    voronoi_colormap = plt.get_cmap(kwargs.pop("cmap", "YlGn"))

    if axes is None:
        axes = plt.gca()

    if measurement_area is not None:
        plot_measurement_setup(
            measurement_areas=[measurement_area], axes=axes, **kwargs
        )

    if traj_data:
        data = traj_data.data.merge(
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
            norm = mpl.colors.Normalize(vmin, vmax)
            scalar_mappable = mpl.cm.ScalarMappable(
                norm=norm, cmap=voronoi_colormap
            )
            color_mapper = scalar_mappable.to_rgba
        elif typ == "int64":
            voronoi_colormap = plt.get_cmap("tab20c")

            def forward(values):
                return values % 20

            def inverse(values):
                return forward(values)

            norm = mpl.colors.FuncNorm((forward, inverse), 0, 19)
            scalar_mappable = mpl.cm.ScalarMappable(
                norm=norm, cmap=voronoi_colormap
            )
            color_mapper = scalar_mappable.to_rgba
        else:
            pass

    for _, row in data.iterrows():
        poly = row[POLYGON_COL]

        if color_by_column:
            color = color_mapper(row[color_by_column])
        else:
            color = np.array([1, 1, 1])

        _plot_polygon(
            axes=axes,
            polygon=poly,
            line_color=voronoi_border_color,
            polygon_color=color,
            polygon_alpha=voronoi_outside_ma_alpha,
            zorder=1,
        )

        if INTERSECTION_COL in data.columns:
            if not shapely.is_empty(row[INTERSECTION_COL]):
                intersection_poly = row[INTERSECTION_COL]
                _plot_polygon(
                    axes=axes,
                    polygon=intersection_poly,
                    line_color="none",
                    polygon_color=color,
                    polygon_alpha=voronoi_inside_ma_alpha,
                    zorder=1,
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
