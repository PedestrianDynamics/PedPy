"""Module containing functions to compute profiles.

For the computation of the profiles the given :class:`~geometry.WalkableArea`
is divided into square grid cells.

.. image:: /images/profile_grid.svg
    :width: 60 %
    :align: center

Each of these grid cells is then used as a :class:`~geometry.MeasurementArea`
in which the mean speed and density can be computed with different methods.
"""
from enum import Enum, auto
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas
import shapely

from pedpy.column_identifier import FRAME_COL
from pedpy.data.geometry import WalkableArea
from pedpy.internal.utils import alias


class SpeedMethod(Enum):  # pylint: disable=too-few-public-methods
    """Method used to compute the speed profile."""

    ARITHMETIC = auto()
    r"""Compute arithmetic Voronoi speed profile.
    
    In each cell :math:`M` the arithmetic Voronoi speed :math:`v_{arithmetic}` is defined as

    .. math::

        v_{arithmetic} = \frac{1}{N} \sum_{i \in P_M} v_i,

    
    where :math:`P_M` are the pedestrians, whose Voronoi cell :math:`V_i` 
    intersects with the grid cell :math:`M` (:math:`V_i \cap M`). Then
    :math:`N` is the number of pedestrians in :math:`P_M` (:math:`|P_M|`).
    """

    VORONOI = auto()
    r"""Compute Voronoi speed profile.
    
    In each cell :math:`M` the Voronoi speed :math:`v_{voronoi}` is defined as

    .. math::

        v_{voronoi} = { \int\int v_{xy} dxdy \over A(M)},
        
    where :math:`v_{xy} = v_i` is the individual speed of
    each pedestrian, whose :math:`V_i \cap M` and :math:`A(M)` the area 
    the grid cell.
    """

    MEAN = auto()
    r"""Compute mean speed profile.
    
    In each cell :math:`M` the mean speed :math:`v_{mean}` is defined as
    
    .. math::
    
        v_{mean} = \frac{1}{N} \sum_{i \in P_M} v_i,
        
    where :math:`P_M` are the pedestrians inside the grid cell. Then :math:`N`
    is the number of pedestrians inside :math:`P_M` (:math:`|P_M|`).
    """


class DensityMethod(Enum):  # pylint: disable=too-few-public-methods
    """Method used to compute the density profile."""

    VORONOI = auto()
    r"""Voronoi density profile.
    
    In each cell the density :math:`\rho_{voronoi}` is defined by 
    
    .. math::

            \rho_{voronoi} = { \int\int \rho_{xy} dxdy \over A(M)},

    where :math:`\rho_{xy} = 1 / A(V_i)` is the individual density of
    each pedestrian, with the individual Voronoi polygons :math:`V_i` where 
    :math:`V_i \cap M` and :math:`A(M)` the area of the grid cell.
    """

    CLASSIC = auto()
    r"""Classic density profile.
    
    In each cell the density  :math:`\rho_{classic}` is defined by 
    
    .. math::
     
        \rho_{classic} = {N \over A(M)},
    
    where :math:`N` is the number of pedestrians inside the grid cell :math:`M`
    and the area of that grid cell (:math:`A(M)`). 
    """

    GAUSSIAN = auto()
    r"""Gaussian density profile.
    
    In each cell the density :math:`\rho_{gaussian}` is defined by 
    
    .. math::
     
        \rho_{gaussian} = \sum_{i=1}^{N}{\delta (\boldsymbol{r}_i - \boldsymbol{c})},

    where :math:`\boldsymbol{r}_i` is the position of a pedestrian and :math:`\boldsymbol{c}`
    is the center of the grid cell. Finally :math:`\delta(x)` is approximated by a Gaussian
    
    .. math::
        
        \delta(x) = \frac{1}{\sqrt{\pi}a}\exp[-x^2/a^2]
    """


@alias({"data": "individual_voronoi_speed_data"})
def compute_profiles(
    *,
    data: pandas.DataFrame = None,
    walkable_area: WalkableArea,
    grid_size: float,
    speed_method: SpeedMethod,
    density_method: DensityMethod = DensityMethod.VORONOI,
    gaussian_width: Optional[float] = None,
    # pylint: disable=unused-argument
    **kwargs: Any,
) -> Tuple[List[npt.NDArray[np.float64]], List[npt.NDArray[np.float64]]]:
    """Computes the density and speed profiles.

    .. note::

        As this is a quite compute heavy operation, it is suggested to reduce
        the geometry to the important areas and limit the :code:`data` to the
        most relevant frame interval.

    Args:
        data: Data from which the profiles are computes.
            The DataFrame must contain a `frame` and a `speed` (result from
            :func:`~speed_calculator.compute_individual_speed`) column.
            For computing density profiles, it must contain a `polygon` column
            (from :func:`~method_utils.compute_individual_voronoi_polygons`)
            when using the `DensityMethod.VORONOI`. When computing the classic
            density profile (`DensityMethod.CLASSIC`) the DataFrame needs to
            contain the columns 'x' and 'y'. Computing the speed
            profiles needs a `polygon` column (from
            :func:`~method_utils.compute_individual_voronoi_polygons`) when
            using the :attr:`SpeedMethod.VORONOI` or
            :attr:`SpeedMethod.ARITHMETIC`. For getting a DataFrame containing
            all the needed data, you can merge the results of the different
            function on the 'id' and 'frame' columns (see
            :func:`pandas.DataFrame.merge` and :func:`pandas.merge`).
        walkable_area: geometry for which the profiles are computed
        grid_size: resolution of the grid used for computing the
            profiles
        speed_method: speed method used to compute the
            speed profile
        density_method: density method to compute the density
            profile (default: :attr:`DensityMethod.VORONOI`)
        gaussian_width: full width at half maximum for Gaussian
            approximation of the density, only needed when using
            :attr:`DensityMethod.GAUSSIAN`.
        individual_voronoi_speed_data: deprecated alias for
            :code:`data`. Please use :code:`data` in the future.

    Returns:
        List of density profiles, List of speed profiles
    """
    grid_cells, _, _ = get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    grid_intersections_area, internal_data = _compute_grid_polygon_intersection(
        data=data, grid_cells=grid_cells
    )

    density_profiles = compute_density_profile(
        data=internal_data,
        grid_intersections_area=grid_intersections_area,
        density_method=density_method,
        walkable_area=walkable_area,
        grid_size=grid_size,
        gaussian_width=gaussian_width,
    )

    speed_profiles = compute_speed_profile(
        data=internal_data,
        grid_intersections_area=grid_intersections_area,
        speed_method=speed_method,
        walkable_area=walkable_area,
        grid_size=grid_size,
    )

    return density_profiles, speed_profiles


def compute_density_profile(
    *,
    data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    density_method: DensityMethod,
    grid_intersections_area: Optional[npt.NDArray[np.float64]] = None,
    gaussian_width: Optional[float] = None,
) -> List[npt.NDArray[np.float64]]:
    """Compute the density profile.

    Args:
        data: Data from which the profiles are computes.
            The DataFrame must contain a `frame` column. It must contain
            a `polygon` column (from
            :func:`~method_utils.compute_individual_voronoi_polygons`)
            when using the :attr:`DensityMethod.VORONOI`. When computing
            the classic density profile (:attr:`DensityMethod.CLASSIC`) or
            Gaussian density profile (:attr:`DensityMethod.GAUSSIAN`) the
            DataFrame needs to contain the columns 'x' and 'y'. For getting
            a DataFrame containing all the needed data, you can merge the
            results of the different function on the 'id' and 'frame'
            columns (see :func:`pandas.DataFrame.merge` and
            :func:`pandas.merge`).
        walkable_area (WalkableArea): geometry for which the profiles are
            computed
        grid_size (float): resolution of the grid used for computing the
            profiles
        density_method: density method to compute the density
            profile
        grid_intersections_area: intersection of grid cells with the Voronoi
            polygons (result from
            :func:`compute_grid_cell_polygon_intersection_area`)
        gaussian_width: full width at half maximum for Gaussian
            approximation of the density, only needed when using
            :attr:`DensityMethod.GAUSSIAN`.

    Returns:
        List of density profiles
    """
    grid_cells, rows, cols = get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    grid_center = np.vectorize(shapely.centroid)(grid_cells)
    x_center = shapely.get_x(grid_center[:cols])
    y_center = shapely.get_y(grid_center[::cols])

    data_grouped_by_frame = data.groupby(FRAME_COL)

    density_profiles = []
    for frame, frame_data in data_grouped_by_frame:
        if density_method == DensityMethod.VORONOI:
            if grid_intersections_area is None:
                raise RuntimeError(
                    "Computing a Voronoi density profile needs the parameter "
                    "`grid_intersections_area`."
                )

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
        elif density_method == DensityMethod.GAUSSIAN:
            if gaussian_width is None:
                raise ValueError(
                    "Computing a Gaussian density profile needs a parameter 'width'."
                )

            density = _compute_gaussian_density_profile(
                frame_data=frame_data,
                center_x=x_center,
                center_y=y_center,
                width=gaussian_width,
            )
        else:
            raise ValueError("density method not accepted.")

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


def _compute_gaussian_density_profile(
    *,
    frame_data: pandas.DataFrame,
    center_x: npt.NDArray[np.float64],
    center_y: npt.NDArray[np.float64],
    width: float,
) -> npt.NDArray[np.float64]:
    def _gaussian_full_width_half_maximum(width: float) -> float:
        """Computes the full width at half maximum.

        Fast lookup for:
            width * np.sqrt(2) / (2 * np.sqrt(2 * np.log(2)))

        Args:
            width: width for which the half maximum should be computed

        Returns:
            Full width at half maximum of a gaussian.
        """
        return width * 0.6005612

    def _compute_gaussian_density(
        x: npt.NDArray[np.float64], fwhm: float
    ) -> npt.NDArray[np.float64]:
        """Computes the Gaussian density.

        Gaussian density p(x, a) is defined as:
            p(x,a) = 1 / (sqrt(pi) * a) * e^(-x^2 / a^2)
        Args:
            x: value(s) for which the Gaussian should be computed
            fwhm: full width at half maximum

        Returns:
            Gaussian corresponding to the given values
        """
        #
        return 1 / (1.7724538 * fwhm) * np.e ** (-(x**2) / fwhm**2)

    positions_x = frame_data.x.values
    positions_y = frame_data.y.values

    # distance from each grid center x/y coordinates to the pedestrian positions
    distance_x = np.add.outer(-center_x, positions_x)
    distance_y = np.add.outer(-center_y, positions_y)

    fwhm = _gaussian_full_width_half_maximum(width)

    gauss_density_x = _compute_gaussian_density(distance_x, fwhm)
    gauss_density_y = _compute_gaussian_density(distance_y, fwhm)

    gauss_density = np.matmul(gauss_density_x, np.transpose(gauss_density_y))
    return np.array(gauss_density.T)


def compute_speed_profile(
    *,
    data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    speed_method: SpeedMethod,
    grid_intersections_area: Optional[npt.NDArray[np.float64]] = None,
    fill_value: float = np.nan,
) -> List[npt.NDArray[np.float64]]:
    """Compute the speed profile.

    Args:
        data: Data from which the profiles are computes.
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
        walkable_area: geometry for which the profiles are
            computed
        grid_size: resolution of the grid used for computing the
            profiles
        speed_method: speed method used to compute the
            speed profile
        grid_intersections_area: intersection of grid cells with the Voronoi
            polygons (result from
            :func:`compute_grid_cell_polygon_intersection_area`)
        fill_value: fill value for cells with no pedestrians inside when using
            `SpeedMethod.MEAN` (default = `np.nan`)

    Returns:
        List of speed profiles
    """
    grid_cells, rows, cols = get_grid_cells(
        walkable_area=walkable_area, grid_size=grid_size
    )

    data_grouped_by_frame = data.groupby(FRAME_COL)

    speed_profiles = []

    for frame, frame_data in data_grouped_by_frame:
        if speed_method == SpeedMethod.VORONOI:
            if grid_intersections_area is None:
                raise RuntimeError(
                    "Computing a Arithmetic speed profile needs the parameter "
                    "`grid_intersections_area`."
                )
            grid_intersections_area_frame = grid_intersections_area[
                :, data_grouped_by_frame.indices[frame]
            ]

            speed = _compute_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area_frame,
                grid_area=grid_cells[0].area,
            )
        elif speed_method == SpeedMethod.ARITHMETIC:
            if grid_intersections_area is None:
                raise RuntimeError(
                    "Computing a Arithmetic speed profile needs the parameter "
                    "`grid_intersections_area`."
                )
            grid_intersections_area_frame = grid_intersections_area[
                :, data_grouped_by_frame.indices[frame]
            ]

            speed = _compute_arithmetic_voronoi_speed_profile(
                frame_data=frame_data,
                grid_intersections_area=grid_intersections_area_frame,
            )
        elif speed_method == SpeedMethod.MEAN:
            speed = _compute_mean_speed_profile(
                frame_data=frame_data,
                walkable_area=walkable_area,
                grid_size=grid_size,
                fill_value=fill_value,
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


def _compute_mean_speed_profile(
    *,
    frame_data: pandas.DataFrame,
    walkable_area: WalkableArea,
    grid_size: float,
    fill_value: float,
) -> npt.NDArray[np.float64]:
    min_x, min_y, max_x, max_y = walkable_area.bounds

    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)

    hist, _, _ = np.histogram2d(
        x=frame_data.x, y=frame_data.y, bins=[x_coords, y_coords]
    )
    hist_speed, _, _ = np.histogram2d(
        x=frame_data.x,
        y=frame_data.y,
        bins=[x_coords, y_coords],
        weights=frame_data.speed,
    )
    speed = np.divide(
        hist_speed,
        hist,
        out=np.full(shape=hist.shape, fill_value=float(fill_value)),
        where=hist != 0,
    )

    # rotate the result, such that is displayed with imshow correctly and
    # has the same orientation as the other results
    speed = np.rot90(speed)

    return speed


def compute_grid_cell_polygon_intersection_area(
    *, data: pandas.DataFrame, grid_cells: npt.NDArray[shapely.Polygon]
) -> Tuple[npt.NDArray[np.float64], pandas.DataFrame]:
    """Computes the intersection area of the grid cells with the Voronoi polygons.

    .. note::

        As this is a quite compute heavy operation, it is suggested to reduce
        limit the :code:`data` to the most relevant frame interval.

    .. note::

        If computing the speed/density profiles multiple times, e.g., with different methods it
        is of advantage to compute the grid cell polygon intersections before and then pass the
        result to the other functions.

    .. important::

        When passing the grid cell-polygon intersection, make sure to also pass the returned
        DataFrame as data, as it has the same ordering of rows as used for the grid cell-polygon
        intersection. Changing the order afterward will return wrong results!

    Args:
        data: DataFrame containing at least the columns 'frame' and 'polygon' (which should hold
            the result from :func:`~method_utils.compute_individual_voronoi_polygons`)
        grid_cells: Grid cells used for computing the profiles, e.g., result from
            :func:`get_grid_cells`

    Returns:
        Tuple containing first the grid cell-polygon intersection areas, and second the reordered
        data by 'frame', which needs to be used in the next steps.
    """
    grid_intersections_area, used_data = _compute_grid_polygon_intersection(
        data=data, grid_cells=grid_cells
    )
    return grid_intersections_area, used_data


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


def get_grid_cells(
    *, walkable_area: WalkableArea, grid_size: float
) -> Tuple[npt.NDArray[shapely.Polygon], int, int]:
    """Creates a list of square grid cells covering the geometry.

    .. image:: /images/profile_grid.svg
        :width: 60 %
        :align: center

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
