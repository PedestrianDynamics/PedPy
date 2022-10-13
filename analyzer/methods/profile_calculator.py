"""Module containing functions to compute profiles"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import shapely
from aenum import Enum
from shapely import Polygon


class VelocityMethod(Enum):
    """Identifier of the method used to compute the mean velocity per grid
    cell
    """

    _init_ = "value __doc__"
    ARITHMETIC = 0, "arithmetic mean velocity"
    VORONOI = 1, "voronoi velocity"


def compute_profiles(
    *,
    individual_voronoi_velocity_data: pd.DataFrame,
    walkable_area: Polygon,
    grid_size: float,
    velocity_method: VelocityMethod,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Computes the density and velocity profiles of the given trajectory
    within the geometry

    Note: As this is a quite compute heavy operation, it is suggested to
    reduce the geometry to the important areas.

    Args:
        individual_voronoi_velocity_data (pd.DataFrame): individual voronoi
            and velocity data, needs to contain a column 'individual voronoi'
            which holds shapely.Polygon information and a column 'speed'
            which holds a floating point value
        walkable_area (shapely.Polygon): geometry for which the profiles are
            computed
        grid_size (float): resolution of the grid used for computing the
            profiles
        velocity_method (VelocityMethod): velocity method used to compute the
            velocity
    Returns:
        (List of density profiles, List of velocity profiles)
    """
    grid_cells, rows, cols = _get_grid_cells(walkable_area, grid_size)
    density_profiles = []
    velocity_profiles = []

    for _, frame_data in individual_voronoi_velocity_data.groupby("frame"):
        grid_intersections_area = shapely.area(
            shapely.intersection(
                np.array(grid_cells)[:, np.newaxis],
                np.array(frame_data["individual voronoi"])[np.newaxis, :],
            )
        )

        # Compute density
        density = (
            np.sum(
                grid_intersections_area
                * (1 / shapely.area(frame_data["individual voronoi"].values)),
                axis=1,
            )
            / grid_cells[0].area
        )

        # Compute velocity
        if velocity_method == VelocityMethod.VORONOI:
            velocity = _compute_voronoi_velocity(
                frame_data, grid_intersections_area, grid_cells[0].area
            )
        elif velocity_method == VelocityMethod.ARITHMETIC:
            velocity = _compute_arithmetic_velocity(
                frame_data, grid_intersections_area
            )
        else:
            raise ValueError("velocity method not accepted")

        density_profiles.append(density.reshape(rows, cols))
        velocity_profiles.append(velocity.reshape(rows, cols))

    return density_profiles, velocity_profiles


def _compute_arithmetic_velocity(
    frame_data: np.ndarray, grid_intersections_area: np.ndarray
) -> np.ndarray:
    """Compute the arithmetic mean velocity per grid cell

    Args:
        frame_data (np.ndarray): all relevant data in a specific frame
        grid_intersections_area (np.ndarray): intersection areas for each
                pedestrian with each grid cells
    Returns:
        Arithmetic mean velocity per grid cell
    """
    cells_with_peds = np.where(grid_intersections_area > 1e-16, 1, 0)

    accumulated_velocity = np.sum(
        cells_with_peds * frame_data["speed"].values, axis=1
    )
    num_peds = np.count_nonzero(cells_with_peds, axis=1)

    velocity = np.where(
        num_peds > 0,
        accumulated_velocity / num_peds,
        0,
    )
    return velocity


def _compute_voronoi_velocity(
    frame_data: np.ndarray,
    grid_intersections_area: np.ndarray,
    grid_area: float,
) -> np.ndarray:
    """Compute the Voronoi velocity per grid cell

    Args:
        frame_data (np.ndarray): all relevant data in a specific frame
        grid_intersections_area (np.ndarray): intersection areas for each
                pedestrian with each grid cells
        grid_area (float): area of one grid cell
    Returns:
        Voronoi velocity per grid cell
    """
    velocity = (
        np.sum(grid_intersections_area * frame_data["speed"].values, axis=1)
    ) / grid_area

    return velocity


def _get_grid_cells(
    walkable_area: Polygon, grid_size: float
) -> Tuple[np.ndarray, int, int]:
    """Creates a list of square grid cells which cover the space used by
    geometry.

    Args:
        walkable_area (shapely.Polygon): geometry for which the profiles are
            computed.
        grid_size (float): resolution of the grid used for computing the
            profiles.

    Returns:
        (List of grid cells, number of grid rows, number of grid columns)
    """
    bounds = walkable_area.bounds
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    x = np.arange(min_x, max_x + grid_size, grid_size)
    y = np.arange(max_y, min_y - grid_size, -grid_size)

    grid_cells = []
    for j in range(len(y) - 1):
        for i in range(len(x) - 1):
            grid_cell = shapely.box(x[i], y[j], x[i + 1], y[j + 1])
            grid_cells.append(grid_cell)

    return np.array(grid_cells), len(y) - 1, len(x) - 1
