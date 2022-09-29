"""Module containing functions to compute profiles"""

import numpy as np
import pandas as pd
import pygeos


def compute_profiles(
    individuaL_voronoi_velocity_data: pd.DataFrame, geometry: pygeos.Geometry, grid_size: float
):
    """Computes the density and velocity profiles of the given trajectory within the geometry

    Note: As this is a quite compute heavy operation, it is suggested to reduce the geometry to
        the important areas.

    Args:
        individuaL_voronoi_velocity_data (pd.DataFrame): individual voronoi and velocity data,
            needs to contain a column 'individual voronoi' which holds pygeos.Polygon information
            and a column 'speed' which holds a floating point value
        geometry (pygeos.Geometry): geometry for which the profiles are computed
        grid_size (float): resolution of the grid used for computing the profiles

    Returns:
        (List of density profiles, List of velocity profiles)

    """
    grid_cells, rows, cols = _get_grid_cells(geometry, grid_size)
    density_profiles = []
    velocity_profiles = []

    for _, frame_data in individuaL_voronoi_velocity_data.groupby("frame"):
        grid_intersections_area = pygeos.area(
            pygeos.intersection(
                grid_cells[:, np.newaxis], frame_data["individual voronoi"][np.newaxis, :]
            )
        )

        density = (
            np.sum(
                grid_intersections_area
                * (1 / pygeos.area(frame_data["individual voronoi"].values)),
                axis=1,
            )
            / pygeos.area(grid_cells[0])
        )
        velocity = np.sum(
            grid_intersections_area * frame_data["speed"].values, axis=1
        ) / pygeos.area(grid_cells[0])

        density_profiles.append(density.reshape(rows, cols))
        velocity_profiles.append(velocity.reshape(rows, cols))

    return density_profiles, velocity_profiles


def _get_grid_cells(geometry: pygeos.Geometry, grid_size: float):
    """Creates a list of square grid cells which cover the space used by geometry.

    Args:
        geometry (pygeos.Geometry): geometry for which the profiles are computed
        grid_size (float): resolution of the grid used for computing the profiles

    Returns:
        (List of grid cells, number of grid rows, number of grid columns)
    """
    bounds = pygeos.bounds(geometry)
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    xs = np.arange(min_x, max_x + grid_size, grid_size)
    ys = np.arange(max_y, min_y - grid_size, -grid_size)

    grid_cells = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            grid_cell = pygeos.box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            grid_cells.append(grid_cell)

    return np.array(grid_cells), len(ys) - 1, len(xs) - 1
