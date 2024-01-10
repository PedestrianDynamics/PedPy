"""Module containing functions to compute profiles."""
from enum import Enum, auto
from typing import List, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas
import shapely

from pedpy.column_identifier import FRAME_COL
from pedpy.data.geometry import WalkableArea
from pedpy.internal.utils import alias


class SpeedMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier for the method used to compute the speed profile."""

    ARITHMETIC = auto()
    """arithmetic mean speed profile"""
    VORONOI = auto()
    """Voronoi speed profile"""


class DensityMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier for the method used to compute density profile."""

    VORONOI = auto()
    """Voronoi density profile"""


@alias({"data": "individual_voronoi_speed_data"})
def compute_profiles(
    *,
    data: pandas.DataFrame = None,
    walkable_area: WalkableArea,
    grid_size: float,
    speed_method: SpeedMethod,
    density_method: DensityMethod = DensityMethod.VORONOI,
    # pylint: disable=unused-argument
    **kwargs: Any,
) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]:
    """Computes the density and speed profiles.

    For the computation of the profiles the given
    :class:`~geometry.WalkableArea` is divided into square grid cells with an
    edge length of :code:`grid_size`.

    .. image:: /images/profile_grid.svg
        :width: 60 %
        :align: center

    Each of these grid cells is then used as a
    :class:`~geometry.MeasurementArea`  to compute the Voronoi density as in
    :func:`~density_calculator.compute_voronoi_density`.

    The computation of the speed in each cell is either done with the Voronoi
    speed computation as in :func:`~speed_calculator.compute_voronoi_speed`
    when using :data:`SpeedMethod.VORONOI`. When using
    :data:`SpeedMethod.ARITHMETIC` the mean speed of all pedestrians whose
    Voronoi cells intersects with the grid cell is computed.

    .. note::

        As this is a quite compute heavy operation, it is suggested to reduce
        the geometry to the important areas and limit the :code:`data` to the
        most relevant frame interval.

    Args:
        data (pandas.DataFrame): Data from which the profiles are computes.
            The DataFrame must contain a `frame` and a `speed` (result from
            :func:`~speed_calculator.compute_individual_speed`) column.
            For computing density profiles, it must contain a `polygon` column
            (from :func:`~method_utils.compute_individual_voronoi_polygons`)
            when using the `DensityMethod.VORONOI`. Computing the speed
            profiles needs a `polygon` column (from
            :func:`~method_utils.compute_individual_voronoi_polygons`) when
            using the `SpeedMethod.VORONOI` or `SpeedMethod.ARITHMETIC`.
            For getting a DataFrame containing all the needed data, you can
            merge the results of the different function on the 'id' and
            'frame' columns (see :func:`pandas.DataFrame.merge` and
            :func:`pandas.merge`).
        walkable_area (WalkableArea): geometry for which the profiles are
            computed
        grid_size (float): resolution of the grid used for computing the
            profiles
        speed_method (SpeedMethod): speed method used to compute the
            speed profile
        density_method (DensityMethod): density method to compute the density
            profile (default: DensityMethod.VORONOI)
        individual_voronoi_speed_data (pandas.DataFrame): deprecated alias for
            :code:`data`. Please use :code:`data` in the future.

    Returns:
        List of density profiles, List of speed profiles
    """
    grid_cells, rows, cols = _get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )
    density_profiles = []
    speed_profiles = []

    for _, frame_data in data.groupby(FRAME_COL):
        grid_intersections_area = shapely.area(
            shapely.intersection(
                np.array(grid_cells)[:, np.newaxis],
                np.array(frame_data.polygon)[np.newaxis, :],
            )
        )

        # Compute density
        if density_method == DensityMethod.VORONOI:
            density = _compute_voronoi_density_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area,
                grid_area=grid_cells[0].area,
            )
        else:
            raise ValueError("density method not accepted")

        # Compute speed
        if speed_method == SpeedMethod.VORONOI:
            speed = _compute_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area,
                grid_area=grid_cells[0].area,
            )
        elif speed_method == SpeedMethod.ARITHMETIC:
            speed = _compute_arithmetic_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area,
            )
        else:
            raise ValueError("speed method not accepted")

        density_profiles.append(density.reshape(rows, cols))
        speed_profiles.append(speed.reshape(rows, cols))

    return density_profiles, speed_profiles


def _compute_voronoi_density_profile(
    *,
    frame_data: pandas.DataFrame,
    grid_intersections_area: npt.NDArray[np.float64],
    grid_area: float,
) -> npt.NDArray[np.float64]:
    return (
        np.sum(
            grid_intersections_area
            * (1 / shapely.area(frame_data.polygon.values)),
            axis=1,
        )
        / grid_area
    )


def _compute_arithmetic_voronoi_speed_profile(
    *,
    frame_data: pandas.DataFrame,
    grid_intersections_area: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the arithmetic mean speed per grid cell.

    Args:
        frame_data (npt.NDArray[np.float64]): all relevant data in a specific
            frame
        grid_intersections_area (npt.NDArray[np.float64]): intersection areas
            for each pedestrian with each grid cells

    Returns:
        Arithmetic mean speed per grid cell
    """
    cells_with_peds = np.where(grid_intersections_area > 1e-16, 1, 0)

    accumulated_speed = np.sum(
        cells_with_peds * frame_data.speed.values, axis=1
    )
    num_peds = np.count_nonzero(cells_with_peds, axis=1)
    speed = np.divide(
        accumulated_speed,
        num_peds,
        out=np.zeros_like(accumulated_speed),
        where=num_peds != 0,
    )

    return speed


def _compute_voronoi_speed_profile(
    *,
    frame_data: pandas.DataFrame,
    grid_intersections_area: npt.NDArray[np.float64],
    grid_area: float,
) -> npt.NDArray[np.float64]:
    """Compute the Voronoi speed per grid cell.

    Args:
        frame_data (npt.NDArray[np.float64]): all relevant data in a specific frame
        grid_intersections_area (npt.NDArray[np.float64]): intersection areas for each
                pedestrian with each grid cells
        grid_area (float): area of one grid cell
    Returns:
        Voronoi speed per grid cell
    """
    speed = (
        np.sum(grid_intersections_area * frame_data.speed.values, axis=1)
    ) / grid_area

    return speed


def _get_grid_cells(
    *, walkable_area: WalkableArea, grid_size: float
) -> Tuple[npt.NDArray[shapely.Polygon], int, int]:
    """Creates a list of square grid cells covering the geometry.

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

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(max_y, min_y - grid_size, -grid_size)

    grid_cells = []
    for j in range(len(y_coords) - 1):
        for i in range(len(x_coords) - 1):
            grid_cell = shapely.box(
                x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1]
            )
            grid_cells.append(grid_cell)

    return np.array(grid_cells), len(y_coords) - 1, len(x_coords) - 1
