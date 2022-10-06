"""Module containing functions to compute profiles"""

import numpy as np
import pandas as pd
import pygeos


def compute_profiles(
    individual_voronoi_velocity_data: pd.DataFrame,
    geometry: pygeos.Geometry,
    grid_size: float,
):
    """Computes the density and velocity profiles of the given trajectory
    within the geometry

    Note: As this is a quite compute heavy operation, it is suggested to
    reduce the geometry to the important areas.

    Args:
        individual_voronoi_velocity_data (pd.DataFrame): individual voronoi
            and velocity data, needs to contain a column 'individual voronoi'
            which holds pygeos.Polygon information and a column 'speed'
            which holds a floating point value
        geometry (pygeos.Geometry): geometry for which the profiles are
            computed
        grid_size (float): resolution of the grid used for computing the
            profiles

    Returns:
        (List of density profiles, List of velocity profiles)

    """
    grid_cells, rows, cols = _get_grid_cells(geometry, grid_size)
    density_profiles = []
    velocity_profiles = []

    for _, frame_data in individual_voronoi_velocity_data.groupby("frame"):
        grid_intersections_area = pygeos.area(
            pygeos.intersection(
                np.array(grid_cells)[:, np.newaxis],
                np.array(frame_data["individual voronoi"])[np.newaxis, :],
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

        grid_intersections_area[grid_intersections_area > 0] = 1
        accumulated_velocity = np.sum(
            grid_intersections_area * frame_data["speed"].values, axis=1
        )
        num_peds = np.count_nonzero(grid_intersections_area, axis=1)

        velocity = np.divide(
            accumulated_velocity,
            num_peds,
            out=np.zeros_like(accumulated_velocity),
            where=num_peds != 0,
        )

        density_profiles.append(density.reshape(rows, cols))
        velocity_profiles.append(velocity.reshape(rows, cols))

    return density_profiles, velocity_profiles


def _get_grid_cells(geometry: pygeos.Geometry, grid_size: float):
    """Creates a list of square grid cells which cover the space used by
    geometry.

    Args:
        geometry (pygeos.Geometry): geometry for which the profiles are
            computed.
        grid_size (float): resolution of the grid used for computing the
            profiles.

    Returns:
        (List of grid cells, number of grid rows, number of grid columns)
    """
    bounds = pygeos.bounds(geometry)
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    x = np.arange(min_x, max_x + grid_size, grid_size)
    y = np.arange(max_y, min_y - grid_size, -grid_size)

    grid_cells = []
    for j in range(len(y) - 1):
        for i in range(len(x) - 1):
            grid_cell = pygeos.box(x[i], y[j], x[i + 1], y[j + 1])
            grid_cells.append(grid_cell)

    return np.array(grid_cells), len(y) - 1, len(x) - 1
