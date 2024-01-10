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
    CLASSIC = auto()
    """Classic density profile"""


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
            when using the `DensityMethod.VORONOI`. When computing the classic
            density profile (`DensityMethod.CLASSIC`) the DataFrame needs to
            contain the columns 'x' and 'y'. Computing the speed
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
    grid_cells, _, _ = _get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    grid_intersections_area, internal_data = _compute_grid_polygon_intersection(
        data=data, grid_cells=grid_cells
    )

    density_profiles = _compute_density_profile(
        data=internal_data,
        grid_intersections_area=grid_intersections_area,
        density_method=density_method,
        walkable_area=walkable_area,
        grid_size=grid_size,
    )

    speed_profiles = _compute_speed_profile(
        data=internal_data,
        grid_intersections_area=grid_intersections_area,
        speed_method=speed_method,
        walkable_area=walkable_area,
        grid_size=grid_size,
    )

    return density_profiles, speed_profiles


def _compute_density_profile(
    *,
    data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    density_method: DensityMethod,
    grid_intersections_area: npt.NDArray[np.float64],
) -> List[npt.NDArray[np.float64]]:
    grid_cells, rows, cols = _get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    data_grouped_by_frame = data.groupby(FRAME_COL)

    density_profiles = []
    for frame, frame_data in data_grouped_by_frame:
        if density_method == DensityMethod.VORONOI:
            grid_intersections_area_frame = grid_intersections_area[
                :, data_grouped_by_frame.indices[frame]
            ]

            density = _compute_voronoi_density_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area_frame,
                grid_area=grid_cells[0].area,
            )
        elif density_method == DensityMethod.CLASSIC:
            density = _compute_classic_density_profile(
                frame_data=frame_data,
                walkable_area=walkable_area,
                grid_size=grid_size,
            )
        else:
            raise ValueError("density method not accepted")

        density_profiles.append(density.reshape(rows, cols))

    return density_profiles


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


def _compute_classic_density_profile(
    *,
    frame_data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
) -> npt.NDArray[np.float64]:
    min_x, min_y, max_x, max_y = walkable_area.bounds

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    hist, _, _ = np.histogram2d(
        x=frame_data.x, y=frame_data.y, bins=[x_coords, y_coords]
    )
    hist = hist / (grid_size**2)

    # rotate the result, such that is displayed with imshow correctly and
    # has the same orientation as the other results
    hist = np.rot90(hist)

    return hist


def _compute_speed_profile(
    *,
    data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    speed_method: SpeedMethod,
    grid_intersections_area: npt.NDArray[np.float64],
) -> List[npt.NDArray[np.float64]]:
    grid_cells, rows, cols = _get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    data_grouped_by_frame = data.groupby(FRAME_COL)

    speed_profiles = []

    for frame, frame_data in data_grouped_by_frame:
        grid_intersections_area_frame = grid_intersections_area[
            :, data_grouped_by_frame.indices[frame]
        ]
        if speed_method == SpeedMethod.VORONOI:
            speed = _compute_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area_frame,
                grid_area=grid_cells[0].area,
            )
        elif speed_method == SpeedMethod.ARITHMETIC:
            speed = _compute_arithmetic_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area_frame,
            )
        else:
            raise ValueError("speed method not accepted")

        speed_profiles.append(speed.reshape(rows, cols))

    return speed_profiles


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


def _compute_grid_polygon_intersection(
    *,
    data,
    grid_cells,
):
    internal_data = data.copy(deep=True)
    internal_data = internal_data.sort_values(by=FRAME_COL)
    internal_data = internal_data.reset_index(drop=True)

    grid_intersections_area = shapely.area(
        shapely.intersection(
            np.array(grid_cells)[:, np.newaxis],
            np.array(internal_data.polygon)[np.newaxis, :],
        )
    )
    return grid_intersections_area, internal_data


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
