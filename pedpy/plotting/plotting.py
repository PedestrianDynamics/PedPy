"""Module containing plotting functionalities"""
from typing import Any, Optional

import matplotlib.axes
import matplotlib.pyplot as plt

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
        ax (matplotlib.axes.Axes):  Axes to plot on, if None new will be created
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
        ax (matplotlib.axes.Axes, optional):  Axes to plot on, if None new will be created
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

    if geometry is None and ax is None:
        ax = plt.gca()
    elif geometry is not None:
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
