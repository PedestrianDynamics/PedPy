"""Helper functions for the analysis methods."""
# pylint: disable=C0302

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Final, List, Optional, Tuple, TypeAlias

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import Voronoi

from pedpy.column_identifier import (
    CROSSING_FRAME_COL,
    DENSITY_COL,
    DISTANCE_COL,
    END_POSITION_COL,
    FIRST_FRAME_COL,
    FRAME_COL,
    ID_COL,
    INTERSECTION_COL,
    LAST_FRAME_COL,
    MID_POSITION_COL,
    NEIGHBORS_COL,
    NEIGHBOR_ID_COL,
    POINT_COL,
    POLYGON_COL,
    SPECIES_COL,
    START_POSITION_COL,
    TIME_COL,
    V_X_COL,
    V_Y_COL,
    WINDOW_SIZE_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData

_log = logging.getLogger(__name__)

LambdaGroupFunction: TypeAlias = Callable[
    [pd.DataFrame, MeasurementLine], pd.DataFrame
]


class SpeedCalculation(Enum):  # pylint: disable=too-few-public-methods
    """Method-identifier used to compute the movement at traj borders."""

    BORDER_EXCLUDE = auto()
    BORDER_ADAPTIVE = auto()
    BORDER_SINGLE_SIDED = auto()


class AccelerationCalculation(Enum):  # pylint: disable=too-few-public-methods
    """Method-identifier used to compute the movement at traj borders."""

    BORDER_EXCLUDE = auto()


class DataValidationStatus(Enum):  # pylint: disable=too-few-public-methods
    """Identifies the result of a return value."""

    DATA_CORRECT = auto()
    COLUMN_MISSING = auto()
    ENTRY_MISSING = auto()


@dataclass(
    kw_only=True,
)
class Cutoff:
    """Maximal extend of a Voronoi polygon.

    The maximal extend is an approximated circle with the given radius and
    number of line segments used to approximate a quarter circle.

    Attributes:
        radius (float): radius of the approximated circle
        quad_segments (int): number of line elements used to approximate a
                quarter circle
    """

    radius: float
    quad_segments: int = 3


class InputError(Exception):
    """Class reflecting errors when incorrect input was given."""

    def __init__(self, message):
        """Create InputError with the given message.

        Args:
            message: Error message
        """
        self.message = message


def is_trajectory_valid(
    *, traj_data: TrajectoryData, walkable_area: WalkableArea
) -> bool:
    """Checks if all trajectory data points lie within the given walkable area.

    Args:
        traj_data (TrajectoryData): trajectory data
        walkable_area (WalkableArea): walkable area in which the pedestrians
            should be

    Returns:
        All points lie within walkable area
    """
    return get_invalid_trajectory(
        traj_data=traj_data, walkable_area=walkable_area
    ).empty


def get_invalid_trajectory(
    *, traj_data: TrajectoryData, walkable_area: WalkableArea
) -> pd.DataFrame:
    """Returns all trajectory data points outside the given walkable area.

    Args:
        traj_data (TrajectoryData): trajectory data
        walkable_area (WalkableArea): walkable area in which the pedestrians
            should be

    Returns:
        DataFrame showing all data points outside the given walkable area
    """
    return traj_data.data.loc[
        ~shapely.within(traj_data.data.point, walkable_area.polygon)
    ]


def compute_frame_range_in_area(
    *,
    traj_data: TrajectoryData,
    measurement_line: MeasurementLine,
    width: float,
) -> Tuple[pd.DataFrame, MeasurementArea]:
    """Compute the frame ranges for each pedestrian inside the measurement area.

    The measurement area is virtually created by creating a second measurement
    line parallel to the given one offsetting by the given width. The area
    between these line is the used measurement area.

    .. image:: /images/passing_area_from_lines.svg
        :width: 80 %
        :align: center

    For each pedestrians now the frames when they enter and leave the virtual
    measurement area is computed. In this frame interval they have to be inside
    the measurement area continuously. They also need to enter and leave the
    measurement area via different measurement lines. If leaving the area
    between the two lines, crossing the same line twice they will be ignored.
    For a better understanding, see the image below, where red parts of the
    trajectories are the detected ones inside the area. These frame intervals
    will be returned.

    .. image:: /images/frames_in_area.svg
        :width: 80 %
        :align: center


    .. note::

        As passing we define the frame, the pedestrians enter the area and then
        move through the complete area without leaving it. Hence, doing a
        closed analysis of the movement area with several measuring ranges
        underestimates the actual movement time.

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (MeasurementLine): measurement line
        width (float): distance to the second measurement line

    Returns:
        Tuple[pandas.DataFrame, MeasurementArea]: DataFrame containing the
        columns 'id', 'entering_frame' describing the frame the pedestrian
        crossed the first or second line, 'leaving_frame' describing the frame
        the pedestrian crossed the second or first line, and the created
        measurement area
    """
    # Create the second measurement line with the given offset
    second_line = MeasurementLine(
        shapely.offset_curve(measurement_line.line, distance=width)
    )

    # Reverse the order of the coordinates for the second line string to
    # create a rectangular area between the lines
    measurement_area = MeasurementArea(
        [
            *measurement_line.coords,
            *second_line.coords[::-1],
        ]
    )

    inside_range = _get_continuous_parts_in_area(
        traj_data=traj_data, measurement_area=measurement_area
    )

    crossing_frames_first = _compute_crossing_frames(
        traj_data=traj_data,
        measurement_line=measurement_line,
        count_on_line=True,
    )
    crossing_frames_second = _compute_crossing_frames(
        traj_data=traj_data, measurement_line=second_line, count_on_line=True
    )

    start_crossed_1 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_first,
        check_column=FIRST_FRAME_COL,
        column_name="start_crossed_1",
    )
    end_crossed_1 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_first,
        check_column=LAST_FRAME_COL,
        column_name="end_crossed_1",
    )
    start_crossed_2 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_second,
        check_column=FIRST_FRAME_COL,
        column_name="start_crossed_2",
    )
    end_crossed_2 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_second,
        check_column=LAST_FRAME_COL,
        column_name="end_crossed_2",
    )

    frame_range_between_lines = (
        start_crossed_1.merge(
            start_crossed_2,
            how="outer",
            on=[ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL],
        )
        .merge(
            end_crossed_1,
            how="outer",
            on=[ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL],
        )
        .merge(
            end_crossed_2,
            how="outer",
            on=[ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL],
        )
    )

    frame_range_between_lines = frame_range_between_lines[
        (
            frame_range_between_lines.start_crossed_1
            & frame_range_between_lines.end_crossed_2
        )
        | (
            frame_range_between_lines.start_crossed_2
            & frame_range_between_lines.end_crossed_1
        )
    ]

    return (
        frame_range_between_lines.loc[
            :, (ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL)
        ],
        measurement_area,
    )


def compute_neighbors(
    individual_voronoi_data: pd.DataFrame,
    as_list: bool = True,
) -> pd.DataFrame:
    """Compute the neighbors of each pedestrian based on the Voronoi cells.

    Computation of the neighborhood of each pedestrian per frame. Every other
    pedestrian is a neighbor if the Voronoi cells of both pedestrian touch
    and some point. The threshold for touching is set to 1mm.

    Args:
        individual_voronoi_data (pandas.DataFrame): individual voronoi data,
            needs to contain a column 'polygon', which holds a
            :class:`shapely.Polygon` (result from
            :func:`~method_utils.compute_individual_voronoi_polygons`)
        as_list (bool): Return the neighbors as a list per peda and frame, if
            true, otherwise each neighbor is in a single row.

    Returns:
        DataFrame containing the columns 'id', 'frame' and 'neighbors', where
        neighbors are a list of the neighbor's IDs if as_list is True.
        Otherwise the DataFrame contains the columns 'id', 'frame',
        'neighbor_id'.
    """
    if as_list:
        return _compute_neighbors_list(
            individual_voronoi_data=individual_voronoi_data
        )
    else:
        return _compute_neighbors_single(
            individual_voronoi_data=individual_voronoi_data
        )


def _compute_neighbors_list(
    individual_voronoi_data: pd.DataFrame,
) -> pd.DataFrame:
    neighbor_df = []

    for frame, frame_data in individual_voronoi_data.groupby(FRAME_COL):
        touching = shapely.dwithin(
            np.array(frame_data[POLYGON_COL])[:, np.newaxis],
            np.array(frame_data[POLYGON_COL])[np.newaxis, :],
            1e-9,  # Voronoi cells as close as 1 mm are touching
        )

        # the peds are not neighbors of themselves
        for i in range(len(touching)):
            touching[i, i] = False

        # create matrix with ped IDs
        ids = np.outer(
            np.ones_like(frame_data[ID_COL].to_numpy()),
            frame_data[ID_COL].to_numpy().reshape(1, -1),
        )

        neighbors = np.where(touching, ids, np.nan)

        neighbors_list = [
            np.array(neighbor)[~np.isnan(np.array(neighbor))]
            .astype(int)
            .tolist()
            for neighbor in neighbors
        ]

        frame_df = pd.DataFrame(
            zip(
                frame_data[ID_COL].to_numpy(),
                itertools.repeat(frame),
                neighbors_list,
            ),
            columns=[ID_COL, FRAME_COL, NEIGHBORS_COL],
        )
        neighbor_df.append(frame_df)

    return pd.concat(neighbor_df)


def _compute_neighbors_single(
    individual_voronoi_data: pd.DataFrame,
) -> pd.DataFrame:
    neighbor_df = []
    for frame, frame_data in individual_voronoi_data.groupby(FRAME_COL):
        touching = shapely.dwithin(
            np.array(frame_data[POLYGON_COL])[:, np.newaxis],
            np.array(frame_data[POLYGON_COL])[np.newaxis, :],
            1e-9,  # Voronoi cells as close as 1 mm are touching
        )

        # the peds are not neighbors of themselves
        for i in range(len(touching)):
            touching[i, i] = False

        # Create matrix of ped IDs
        ids = np.outer(
            np.ones_like(frame_data[ID_COL].to_numpy()),
            frame_data[ID_COL].to_numpy().reshape(1, -1),
        )

        # Filter neighbor relationships based on the touching matrix
        id_column = ids[touching]  # IDs for each row where touching is True
        neighbor_column = ids.T[touching]  # Transpose to get the neighbor IDs

        # Create DataFrame for this frame's neighbors
        frame_neighbors = pd.DataFrame(
            {
                ID_COL: id_column,
                FRAME_COL: frame,
                NEIGHBOR_ID_COL: neighbor_column,
            }
        )

        # Append to the result list
        neighbor_df.append(frame_neighbors)

    # Concatenate all frames' data into a single DataFrame
    return (
        pd.concat(neighbor_df, ignore_index=True)
        .sort_values(by=[FRAME_COL, ID_COL])
        .reset_index(drop=True)
    )


def compute_neighbor_distance(
    *,
    traj_data: TrajectoryData,
    neighborhood: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the distance between the neighbors.

    Computes the distance between the position of neighbors. As neighbors the
    result of :func:`~compute_neighbors` with parameter :code:`as_list=False`.

    Args:
        traj_data (TrajectoryData): trajectory data
        neighborhood (pd.DataFrame): DataFrame containing the columns 'id',
            'frame' and 'neighbor_id'. The result of :func:`~compute_neighbors`
            with parameter :code:`as_list=False` can be used here as input.

    Raises:
        ValueError: When passing a result of :func:`~compute_neighbors`
            with parameter :code:`as_list=True`.

    Returns:
        DataFrame containing the columns 'id', 'frame', 'neighbor_id' and
        'distance'.
    """
    if NEIGHBORS_COL in neighborhood.columns:
        raise ValueError(
            "For compute the distance between neighbors compute the "
            "neighborhood with `as_list=False`."
        )

    neighbors_with_position = neighborhood.merge(
        traj_data.data[[ID_COL, FRAME_COL, POINT_COL]],
        on=[ID_COL, FRAME_COL],
        how="left",
    )

    neighbors_with_position = neighbors_with_position.merge(
        traj_data.data[[ID_COL, FRAME_COL, POINT_COL]],
        left_on=[NEIGHBOR_ID_COL, FRAME_COL],
        right_on=[ID_COL, FRAME_COL],
        suffixes=("", "_neighbor"),
    )

    neighbors_with_position[DISTANCE_COL] = shapely.distance(
        neighbors_with_position[POINT_COL],
        neighbors_with_position["point_neighbor"],
    )

    return neighbors_with_position[
        [ID_COL, FRAME_COL, NEIGHBOR_ID_COL, DISTANCE_COL]
    ]


def compute_time_distance_line(
    *, traj_data: TrajectoryData, measurement_line: MeasurementLine
) -> pd.DataFrame:
    """Compute the time and distance to the measurement line.

    Compute the time (in frames) and distance to the first crossing of the
    measurement line for each pedestrian. For further information how the
    crossing frames are computed see :func:`~compute_crossing_frames`.
    All frames after a pedestrian has crossed the line will be omitted in the
    results.

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_line (MeasurementLine): line which is crossed

    Returns:
        DataFrame containing 'id', 'frame', 'time' (seconds until
        crossing),  and 'distance' (meters to measurement line)
    """
    df_distance_time = traj_data.data[[ID_COL, FRAME_COL, POINT_COL]].copy(
        deep=True
    )

    # Compute distance to measurement line
    df_distance_time[DISTANCE_COL] = shapely.distance(
        df_distance_time[POINT_COL], measurement_line.line
    )

    # Compute time to entrance
    crossing_frame = compute_crossing_frames(
        traj_data=traj_data, measurement_line=measurement_line
    ).rename(columns={FRAME_COL: CROSSING_FRAME_COL})
    df_distance_time = df_distance_time.merge(crossing_frame, on=ID_COL)
    df_distance_time[TIME_COL] = (
        df_distance_time[CROSSING_FRAME_COL] - df_distance_time[FRAME_COL]
    ) / traj_data.frame_rate

    # Delete all rows where the line has already been passed
    df_distance_time = df_distance_time[df_distance_time.time >= 0]

    return df_distance_time.loc[:, [ID_COL, FRAME_COL, DISTANCE_COL, TIME_COL]]


def compute_individual_voronoi_polygons(
    *,
    traj_data: TrajectoryData,
    walkable_area: WalkableArea,
    cut_off: Optional[Cutoff] = None,
    use_blind_points: bool = True,
) -> pd.DataFrame:
    """Compute the individual Voronoi polygon for each person and frame.

    The Voronoi cell will be computed based on the Voronoi tesselation of the
    pedestrians position. The resulting polygons will then be intersected with
    the walkable area.

    .. warning::

        In case of non-convex walkable areas it might happen that Voronoi cell
        will be cut at unexpected places.

    The computed Voronoi cells will stretch all the way to the boundaries of
    the walkable area. As seen below:

    .. image:: /images/voronoi_wo_cutoff.svg
        :width: 80 %
        :align: center

    In cases with only a few pedestrians not close to each other or large
    walkable areas this might not be desired behavior as the size of the
    Voronoi polygon is directly related to the individual density. In this
    case the size of the Voronoi polygon can be restricted by a
    :class:`Cutoff`, where you give a radius and the number of line segments
    used to approximate a quarter circle. The differences the number of
    line segments has on the circle can be seen in the plot below:

    .. image:: /images/voronoi_cutoff_differences.svg
        :width: 80 %
        :align: center

    Using this cut off information, the resulting Voronoi polygons would like
    this:

    .. image:: /images/voronoi_w_cutoff.svg
        :width: 80 %
        :align: center

    For allowing the computation of the Voronoi polygons when less than 4
    pedestrians are in the walkable area, 4 extra points will be added outside
    the walkable area with a significant distance. These will have no effect
    on the size of the computed Voronoi polygons. This behavior can be turned
    off by setting :code:`use_blind_points = False`. When turned off no Voronoi
    polygons will be computed for frames with less than 4 persons, also
    pedestrians walking in a line can lead to issues in the computation of the
    Voronoi tesselation.

    Args:
        traj_data (TrajectoryData): trajectory data
        walkable_area (WalkableArea): bounding area, where pedestrian are
                supposed to walk
        cut_off (Cutoff): cutoff information, which provide the largest
                possible extend of a single Voronoi polygon
        use_blind_points (bool): adds extra 4 points outside the walkable area
                to also compute voronoi cells when less than 4 peds are in the
                walkable area (default: on!)

    Returns:
        DataFrame containing the columns 'id', 'frame','polygon' (
        :class:`shapely.Polygon`), and 'individual_density' in :math:`1/m^2`.
    """
    dfs = []

    bounds = walkable_area.polygon.bounds
    x_diff = abs(bounds[2] - bounds[0])
    y_diff = abs(bounds[3] - bounds[1])
    clipping_diameter = 2 * max(x_diff, y_diff)

    blind_points = np.array(
        [
            [100 * (bounds[0] - x_diff), 100 * (bounds[1] - y_diff)],
            [100 * (bounds[2] + x_diff), 100 * (bounds[1] - y_diff)],
            [100 * (bounds[0] - x_diff), 100 * (bounds[3] + y_diff)],
            [100 * (bounds[2] + x_diff), 100 * (bounds[3] + y_diff)],
        ]
    )

    for frame, peds_in_frame in traj_data.data.groupby(traj_data.data.frame):
        points = peds_in_frame[[X_COL, Y_COL]].to_numpy()
        points = np.concatenate([points, blind_points])

        # only skip analysis if less than 4 peds are in the frame and blind
        # points are turned off
        if not use_blind_points and len(points) - len(blind_points) < 4:
            _log.warning(
                f"Not enough pedestrians (N="
                f"{len(points) - len(blind_points)}) available to "
                f"calculate Voronoi cells for frame = {frame}. "
                f"Consider enable use of blind points."
            )
            continue

        vor = Voronoi(points)
        voronoi_polygons = _clip_voronoi_polygons(vor, clipping_diameter)

        voronoi_polygons = voronoi_polygons[:-4]
        voronoi_in_frame = peds_in_frame.loc[:, (ID_COL, FRAME_COL, POINT_COL)]

        # Compute the intersecting area with the walkable area
        voronoi_in_frame[POLYGON_COL] = shapely.intersection(
            voronoi_polygons, walkable_area.polygon
        )

        if cut_off is not None:
            radius = cut_off.radius
            quad_segments = cut_off.quad_segments
            voronoi_in_frame.polygon = shapely.intersection(
                voronoi_in_frame.polygon,
                shapely.buffer(
                    peds_in_frame.point,
                    radius,
                    quad_segs=quad_segments,
                ),
            )

        # Only consider the parts of a multipolygon which contain the position
        # of the pedestrian
        voronoi_in_frame.loc[
            shapely.get_type_id(voronoi_in_frame.polygon) != 3,
            POLYGON_COL,
        ] = voronoi_in_frame.loc[
            shapely.get_type_id(voronoi_in_frame.polygon) != 3, :
        ].apply(
            lambda row: shapely.get_parts(row[POLYGON_COL])[
                shapely.within(row.point, shapely.get_parts(row.polygon))
            ][0],
            axis=1,
        )

        dfs.append(voronoi_in_frame)

    result = pd.concat(dfs)[[ID_COL, FRAME_COL, POLYGON_COL]]
    result[DENSITY_COL] = 1.0 / shapely.area(result.polygon)

    return result


def compute_intersecting_polygons(
    *,
    individual_voronoi_data: pd.DataFrame,
    measurement_area: MeasurementArea,
) -> pd.DataFrame:
    """Compute the intersection of the voronoi cells with the measurement area.

    .. image:: /images/voronoi_density.svg
        :width: 60 %
        :align: center

    Args:
        individual_voronoi_data (pandas.DataFrame): individual voronoi data,
            needs to contain a column 'polygon' (:class:`shapely.Polygon`),
            result from
            :func:`~method_utils.compute_individual_voronoi_polygons`
        measurement_area (MeasurementArea): measurement area for which the
            intersection will be computed.

    Returns:
        DataFrame containing the columns 'id', 'frame' and
        'intersection' which is the intersection of the individual Voronoi
        polygon and the given measurement area as :class:`shapely.Polygon`.
    """
    df_intersection = individual_voronoi_data[[ID_COL, FRAME_COL]].copy()
    df_intersection[INTERSECTION_COL] = shapely.intersection(
        individual_voronoi_data.polygon, measurement_area.polygon
    )
    return df_intersection


def compute_crossing_frames(
    *,
    traj_data: TrajectoryData,
    measurement_line: MeasurementLine,
) -> pd.DataFrame:
    """Compute the frames at the pedestrians pass the measurement line.

    As crossing we define a movement that moves across the measurement line.
    When the movement ends on the line, the line is not crossed. When it
    starts on the line, it counts as crossed. A visual representation is shown
    below, where the movement goes from left to right and each dot indicates
    the position at one frame. Red highlights where the person has crossed the
    measurement line.

    .. image:: /images/crossing_frames.svg
        :width: 80 %
        :align: center

    Note:
        Due to oscillations, it may happen that a pedestrian crosses the
        measurement line multiple times in a small-time interval.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        measurement_line (MeasurementLine): measurement line which is crossed

    Returns:
        DataFrame containing the columns 'id', 'frame', where 'frame' is
        the frame where the measurement line is crossed.
    """
    return _compute_crossing_frames(
        traj_data=traj_data,
        measurement_line=measurement_line,
        count_on_line=False,
    )


def _compute_crossing_frames(
    *,
    traj_data: TrajectoryData,
    measurement_line: MeasurementLine,
    count_on_line: bool,
) -> pd.DataFrame:
    """Compute the frames at which pedestrians pass the measurement line.

    If count_on_line is set to True, the crossing frame is the one where the
    pedestrian touches the line. Otherwise, it is the frame where the
    pedestrian crosses the line without stopping on it.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        measurement_line (MeasurementLine): measurement line which is crossed
        count_on_line (bool): Count movement ending on line (True) or only if
            movement crosses line, but does not end on line.

    Returns:
        DataFrame containing the columns 'id', 'frame', where 'frame' is
        the frame where the measurement line is crossed.
    """
    # stack is used to get the coordinates in the correct order, as pygeos
    # does not support creating linestring from points directly. The
    # resulting array looks as follows:
    # [[[x_0_start, y_0_start], [x_0_end, y_0_end]],
    #  [[x_1_start, y_1_start], [x_1_end, y_1_end]], ... ]
    df_movement = _compute_individual_movement(
        traj_data=traj_data, frame_step=1, bidirectional=False
    )
    df_movement["movement"] = shapely.linestrings(
        np.stack(
            [
                shapely.get_coordinates(df_movement.start_position),
                shapely.get_coordinates(df_movement.end_position),
            ],
            axis=1,
        )
    )

    movement_crosses_line = shapely.intersects(
        df_movement.movement, measurement_line.line
    )

    if count_on_line:
        crossing_frames = df_movement.loc[movement_crosses_line][
            [ID_COL, FRAME_COL]
        ]
    else:
        # Case when crossing means movement crosses the line, but the end point
        # is not on it

        # Minimum distance to consider crossing complete
        CROSSING_THRESHOLD: Final = 1e-5  # noqa: N806

        movement_ends_on_line = (
            shapely.distance(df_movement.end_position, measurement_line.line)
            < CROSSING_THRESHOLD
        )

        crossing_frames = df_movement.loc[
            (movement_crosses_line) & (~movement_ends_on_line)
        ][[ID_COL, FRAME_COL]]

    return crossing_frames


def _clip_voronoi_polygons(  # pylint: disable=too-many-locals,invalid-name
    voronoi: Voronoi, diameter: float
) -> List[shapely.Polygon]:
    """Generate Polygons from the Voronoi diagram.

    Generate shapely.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.
    from: https://stackoverflow.com/a/52727406/9601068
    """
    polygons = []
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(
        voronoi.ridge_points, voronoi.ridge_vertices, strict=False
    ):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p]  # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)  # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            polygons.append(shapely.polygons(voronoi.vertices[region]))
            continue
        # Infinite region.
        inf = region.index(-1)  # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)]  # Index of previous vertex.
        k = region[(inf + 1) % len(region)]  # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            (dir_j,) = ridge_direction[i, j]
            (dir_k,) = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1 :] + region[:inf]]
        extra_edge = np.array(
            [
                voronoi.vertices[j] + dir_j * length,
                voronoi.vertices[k] + dir_k * length,
            ]
        )
        polygons.append(
            shapely.polygons(np.concatenate((finite_part, extra_edge)))
        )
    return polygons


def _compute_individual_movement(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    bidirectional: bool = True,
    speed_border_method: SpeedCalculation = SpeedCalculation.BORDER_ADAPTIVE,
) -> pd.DataFrame:
    if speed_border_method == SpeedCalculation.BORDER_EXCLUDE:
        return _compute_movement_exclude_border(
            traj_data, frame_step, bidirectional
        )
    if speed_border_method == SpeedCalculation.BORDER_SINGLE_SIDED:
        return _compute_movement_single_sided_border(
            traj_data, frame_step, bidirectional
        )
    if speed_border_method == SpeedCalculation.BORDER_ADAPTIVE:
        return _compute_movememnt_adaptive_border(
            traj_data, frame_step, bidirectional
        )

    raise ValueError("speed border method not accepted")


def _compute_movement_exclude_border(
    traj_data: TrajectoryData,
    frame_step: int,
    bidirectional: bool,
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step.

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame these points will not be considered.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the future frame_step points will
            be used to determine the movement

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'start_position',
        'end_position', 'window_size'. Where 'start_position'/'end_position' are
        the points where the movement start/ends, and 'window_size' is the
        number of frames between the movement start and end.
    """
    df_movement = traj_data.data.copy(deep=True)

    df_movement[START_POSITION_COL] = df_movement.groupby(
        by=ID_COL
    ).point.shift(frame_step)
    df_movement["start_frame"] = df_movement.groupby(by=ID_COL).frame.shift(
        frame_step
    )

    if bidirectional:
        df_movement[END_POSITION_COL] = df_movement.groupby(
            df_movement.id
        ).point.shift(-frame_step)
        df_movement["end_frame"] = df_movement.groupby(
            df_movement.id
        ).frame.shift(-frame_step)
    else:
        df_movement[END_POSITION_COL] = df_movement.point
        df_movement["end_frame"] = df_movement.frame

    df_movement[WINDOW_SIZE_COL] = (
        df_movement.end_frame - df_movement.start_frame
    )
    return df_movement[
        [
            ID_COL,
            FRAME_COL,
            START_POSITION_COL,
            END_POSITION_COL,
            WINDOW_SIZE_COL,
        ]
    ].dropna()


def _compute_movement_single_sided_border(
    traj_data: TrajectoryData,
    frame_step: int,
    bidirectional: bool,
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step.

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame will be used as boundary. Hence, the intervals become [frame,
    frame + frame_step], or [frame - frame_step, frame] respectively.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the future frame_step points will
            be used to determine the movement

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'start_position',
        'end_position', 'window_size' .Where 'start_position'/'end_position' are
        the points where the movement start/ends, and 'window_size' is the
        number of frames between the movement start and end.
    """
    df_movement = traj_data.data.copy(deep=True)

    df_movement[START_POSITION_COL] = (
        df_movement.groupby(by=ID_COL)
        .point.shift(frame_step)
        .fillna(df_movement.point)
    )
    df_movement["start_frame"] = (
        df_movement.groupby(by=ID_COL)
        .frame.shift(frame_step)
        .fillna(df_movement.frame)
    )

    if bidirectional:
        df_movement[END_POSITION_COL] = (
            df_movement.groupby(df_movement.id)
            .point.shift(-frame_step)
            .fillna(df_movement.point)
        )
        df_movement["end_frame"] = (
            df_movement.groupby(df_movement.id)
            .frame.shift(-frame_step)
            .fillna(df_movement.frame)
        )
    else:
        df_movement[END_POSITION_COL] = df_movement.point
        df_movement["end_frame"] = df_movement.frame

    df_movement[WINDOW_SIZE_COL] = (
        df_movement.end_frame - df_movement.start_frame
    )
    return df_movement[
        [
            ID_COL,
            FRAME_COL,
            START_POSITION_COL,
            END_POSITION_COL,
            WINDOW_SIZE_COL,
        ]
    ]


def _compute_movememnt_adaptive_border(
    traj_data: TrajectoryData,
    frame_step: int,
    bidirectional: bool,
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step.

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame use the maximum available number of frames on this side and the same
    number of frames also on the other side.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the future frame_step points will
            be used to determine the movement

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'start_position',
        'end_position', 'window_size'. Where 'start_position'/'end_position' are
        the points where the movement start/ends, and 'window_size' is the
        number of frames between the movement start and end.
    """
    df_movement = traj_data.data.copy(deep=True)

    frame_infos = df_movement.groupby(by=ID_COL).agg(
        frame_min=(FRAME_COL, "min"), frame_max=(FRAME_COL, "max")
    )
    df_movement = df_movement.merge(frame_infos, on=ID_COL)

    df_movement["distance_min"] = np.abs(
        df_movement.frame - df_movement["frame_min"]
    )
    df_movement["distance_max"] = np.abs(
        df_movement.frame - df_movement["frame_max"]
    )
    df_movement[WINDOW_SIZE_COL] = np.minimum(
        frame_step,
        np.minimum(
            df_movement.distance_min.values, df_movement.distance_max.values
        ),
    )
    df_movement["start_frame"] = df_movement.frame - df_movement.window_size
    df_movement["end_frame"] = df_movement.frame + df_movement.window_size

    start = (
        df_movement[[ID_COL, FRAME_COL, "start_frame", WINDOW_SIZE_COL]]
        .merge(
            df_movement[[ID_COL, FRAME_COL, POINT_COL]],
            left_on=[ID_COL, "start_frame"],
            right_on=[ID_COL, FRAME_COL],
            suffixes=("", "_drop"),
        )
        .drop("frame_drop", axis=1)
        .rename({POINT_COL: START_POSITION_COL}, axis=1)
    )

    if bidirectional:
        end = (
            df_movement[[ID_COL, FRAME_COL, "end_frame"]]
            .merge(
                df_movement[[ID_COL, FRAME_COL, POINT_COL]],
                left_on=[ID_COL, "end_frame"],
                right_on=[ID_COL, FRAME_COL],
                suffixes=("", "_drop"),
            )
            .drop("frame_drop", axis=1)
            .rename({POINT_COL: END_POSITION_COL}, axis=1)
        )
        # as the window is used on both sides
        start[WINDOW_SIZE_COL] = 2 * start[WINDOW_SIZE_COL]

    else:
        df_movement[END_POSITION_COL] = df_movement[POINT_COL]
        end = df_movement[[ID_COL, FRAME_COL, END_POSITION_COL]].copy(deep=True)

    result = start.merge(end, on=[ID_COL, FRAME_COL])[
        [
            ID_COL,
            FRAME_COL,
            START_POSITION_COL,
            END_POSITION_COL,
            WINDOW_SIZE_COL,
        ]
    ]
    return result[result.window_size > 0]


def _compute_individual_movement_acceleration(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    acceleration_border_method: AccelerationCalculation = (
        AccelerationCalculation.BORDER_EXCLUDE
    ),
) -> pd.DataFrame:
    if acceleration_border_method == AccelerationCalculation.BORDER_EXCLUDE:
        return _compute_movement_acceleration_exclude_border(
            traj_data, frame_step
        )

    raise ValueError("acceleration border method not accepted")


def _compute_movement_acceleration_exclude_border(
    traj_data: TrajectoryData,
    frame_step: int,
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step.

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame these points will not be considered.

    Args:
        traj_data (pandas.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the future frame_step points
            will be used to determine the movement

    Returns:
        DataFrame containing the columns: 'id', 'frame', 'start_position',
        'mid_position', 'end_position', 'window_size'. Where
        'start_position'/'end_position' are the points where the movement
        start/ends, and 'window_size' is the number of frames between the
        movement start and end.
    """
    df_movement = traj_data.data.copy(deep=True)

    df_movement[START_POSITION_COL] = df_movement.groupby(
        by=ID_COL
    ).point.shift(2 * frame_step)
    df_movement["start_frame"] = df_movement.groupby(by=ID_COL).frame.shift(
        2 * frame_step
    )

    df_movement[MID_POSITION_COL] = df_movement.groupby(by=ID_COL).point.shift(
        frame_step
    )
    df_movement["mid_frame"] = df_movement.groupby(by=ID_COL).frame.shift(
        frame_step
    )

    df_movement[END_POSITION_COL] = df_movement.point
    df_movement["end_frame"] = df_movement.frame

    df_movement[WINDOW_SIZE_COL] = df_movement.end_frame - df_movement.mid_frame
    return df_movement[
        [
            ID_COL,
            FRAME_COL,
            START_POSITION_COL,
            MID_POSITION_COL,
            END_POSITION_COL,
            WINDOW_SIZE_COL,
        ]
    ].dropna()


def _get_continuous_parts_in_area(
    *, traj_data: TrajectoryData, measurement_area: MeasurementArea
) -> pd.DataFrame:
    """Compute the time-continuous parts of each pedestrian in the area.

    Compute the time-continuous parts in which the pedestrians are inside
    the given measurement area. As leaving the first frame outside the area is
    considered.

    Args:
        traj_data (TrajectoryData): trajectory data
        measurement_area (MeasurementArea): area which is considered

    Returns:
        DataFrame containing the columns: 'id', 'entering_frame',
        'leaving_frame'. Where 'entering_frame' is the first frame a pedestrian
        is inside the measurement area, and 'leaving_frame' is first frame a
        pedestrian after the pedestrian has left the measurement area.
    """
    inside = traj_data.data.loc[
        shapely.within(traj_data.data.point, measurement_area.polygon), :
    ].copy()
    inside.loc[:, "g"] = inside.groupby(
        by=ID_COL, group_keys=False
    ).frame.apply(lambda x: x.diff().ge(2).cumsum())
    inside_range = (
        inside.groupby([ID_COL, "g"])
        .agg(
            entering_frame=(FRAME_COL, "first"),
            leaving_frame=(FRAME_COL, "last"),
        )
        .reset_index()[[ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL]]
    )
    inside_range[LAST_FRAME_COL] += 1

    return inside_range


def _check_crossing_in_frame_range(
    *,
    inside_range: pd.DataFrame,
    crossing_frames: pd.DataFrame,
    check_column: str,
    column_name: str,
) -> pd.DataFrame:
    """Returns rows of inside_range which are also in crossing_frames.

    Args:
        inside_range (pandas.DataFrame): DataFrame containing the columns
            'ID' and check_column
        crossing_frames (pandas.DataFrame): DataFrame containing the columns
            'ID' and 'frame'
        check_column (str): name of the column in inside_range which represents
            a frame value. Needs to be 'frame_start' or 'frame_end'
        column_name (str): name of the result column

    Returns:
        DataFrame containing the columns 'id', 'entering_frame',
        'leaving_frame', and column_name
    """
    assert check_column in (
        FIRST_FRAME_COL,
        LAST_FRAME_COL,
    ), f"check_column needs to be '{FIRST_FRAME_COL}' or '{LAST_FRAME_COL}'"

    crossed = inside_range.merge(
        crossing_frames,
        left_on=[ID_COL, check_column],
        right_on=[ID_COL, FRAME_COL],
        how="left",
        indicator=column_name,
    )[[ID_COL, FIRST_FRAME_COL, LAST_FRAME_COL, column_name]]
    crossed[column_name] = crossed[column_name] == "both"
    crossed = crossed[crossed[column_name]]
    return crossed


def _compute_orthogonal_speed_in_relation_to_proportion(
    group: pd.DataFrame, measurement_line: MeasurementLine
) -> pd.DataFrame:
    """Calculates the speed orthogonal to the line times partial line length.

    group is a DataFrameGroupBy containing the columns
        'v_x', 'v_y' and 'polygon'.
    """
    normal_vector = measurement_line.normal_vector()
    return (
        group[V_X_COL] * normal_vector[0] + group[V_Y_COL] * normal_vector[1]
    ) * _compute_partial_line_length(group[POLYGON_COL], measurement_line)


def _compute_partial_line_length(
    polygon: shapely.Polygon, measurement_line: MeasurementLine
) -> float:
    """Calculates the fraction of the length that is intersected by the polygon.

    .
    """
    line = measurement_line.line
    return shapely.length(shapely.intersection(polygon, line)) / shapely.length(
        line
    )


def _apply_lambda_for_intersecting_frames(
    *,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
    species: pd.DataFrame,
    lambda_for_group: LambdaGroupFunction,
    column_id_sp1: str,
    column_id_sp2: str,
    individual_speed: pd.DataFrame = None,
) -> pd.DataFrame:
    """Apply a custom function to frames.

    Frames, where Voronoi polygons intersect with a measurement line,
    grouped by species.

    This function filters the `individual_voronoi_polygons` to include only
    those polygons that intersect with the given `measurement_line`.
    It then separates the data by species and applies a user-defined
    function (`lambda_for_group`) to each group on a per-frame basis.

    The results for both species are merged into a single DataFrame,
    with separate columns for each species' computed values.

    Args:
        individual_voronoi_polygons (pd.DataFrame):
            DataFrame containing Voronoi polygons for each individual, including
            a polygon geometry column.

        measurement_line (MeasurementLine):
            The measurement line used to filter intersecting Voronoi polygons.

        species (pd.DataFrame):
            DataFrame mapping individual IDs to species identifiers.
            Must contain the species column (`SPECIES_COL`).

        lambda_for_group (LambdaGroupFunction):
            A function applied to each group of species data intersecting
            the measurement line.
            It takes a species-specific group of data and the measurement line
            and returns a processed DataFrame.

        column_id_sp1 (str):
            The name of the output column for species 1 results in the
            merged DataFrame.

        column_id_sp2 (str):
            The name of the output column for species 2 results in the merged
            DataFrame.

        individual_speed (pd.DataFrame, optional):
            Optional DataFrame containing individual speed data, merged if
            provided, using `ID_COL` and `FRAME_COL`.

    Returns:
        pd.DataFrame:
            A DataFrame with computed results for both species, merged by frame.
            Contains `column_id_sp1` and `column_id_sp2` as result columns.
            The result is sorted by frame in descending order.

    Notes:
        - Species are identified by `1` and `-1` in the `SPECIES_COL`.
        - Frames without data for one species will contain `NaN` in the
          corresponding result column.
    """
    merged_table = individual_voronoi_polygons[
        shapely.intersects(
            individual_voronoi_polygons[POLYGON_COL], measurement_line.line
        )
    ]

    merged_table = merged_table.merge(species, on="id", how="left")
    if individual_speed is not None:
        merged_table = merged_table.merge(
            individual_speed,
            left_on=[ID_COL, FRAME_COL],
            right_on=[ID_COL, FRAME_COL],
        )

    species_1 = merged_table[merged_table[SPECIES_COL] == 1]
    species_2 = merged_table[merged_table[SPECIES_COL] == -1]

    if not species_1.empty:
        species_1 = (
            species_1.groupby(FRAME_COL, group_keys=False)
            .apply(lambda group: lambda_for_group(group, measurement_line))
            .reset_index()
        )
        species_1.columns = [FRAME_COL, column_id_sp1]
    else:
        species_1 = pd.DataFrame(columns=[FRAME_COL, column_id_sp1])

    if not species_2.empty:
        species_2 = (
            species_2.groupby(FRAME_COL, group_keys=False)
            .apply(lambda group: lambda_for_group(group, measurement_line))
            .reset_index()
        )
        species_2.columns = [FRAME_COL, column_id_sp2]
    else:
        species_2 = pd.DataFrame(columns=[FRAME_COL, column_id_sp2])

    result = species_1.merge(
        species_2, on=FRAME_COL, how="outer"
    ).infer_objects(copy=False)
    return result.sort_values(by=FRAME_COL, ascending=False)


def is_species_valid(
    *,
    species: pd.DataFrame,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
) -> bool:
    """Checks if there's species data of every pedestrian intersecting the line.

    Args:
        species (pd.DataFrame): dataframe containing information
            about the species of every pedestrian intersecting with the line,
            result from :func:`~speed_calculator.compute_species`

        individual_voronoi_polygons (pd.DataFrame): individual Voronoi data per
            frame, result
            from :func:`~method_utils.compute_individual_voronoi_polygons`

        measurement_line (MeasurementLine): measurement line

    Returns:
        True if all needed data is provided by the species dataframe else False.
    """
    intersecting_polygons = individual_voronoi_polygons[
        shapely.intersects(
            individual_voronoi_polygons[POLYGON_COL], measurement_line.line
        )
    ]
    return intersecting_polygons[ID_COL].isin(species[ID_COL]).all()


def is_individual_speed_valid(
    *,
    individual_speed: pd.DataFrame,
    individual_voronoi_polygons: pd.DataFrame,
    measurement_line: MeasurementLine,
) -> DataValidationStatus:
    """Checks for speed data in any entry a pedestrian is intersecting the line.

    Args:
        individual_speed (pd.DataFrame): individual speed data per frame,
            result from :func:`~speed_calculator.compute_individual_speed`
            using :code:`compute_velocity`

        individual_voronoi_polygons (pd.DataFrame): individual Voronoi data per
            frame, result
            from :func:`~method_utils.compute_individual_voronoi_polygons`

        measurement_line (MeasurementLine): measurement line

    Returns:
        DATA_CORRECT if all needed data is provided
            by the individual speed dataframe,
        COLUMN_MISSING if there is a column missing,
        ENTRY_MISSING if there is no matching entry
            for a frame where polygon and line intersect.
    """
    if not all(
        column in individual_speed.columns
        for column in [ID_COL, FRAME_COL, V_X_COL, V_Y_COL]
    ):
        return DataValidationStatus.COLUMN_MISSING
    intersecting_polygons = individual_voronoi_polygons[
        shapely.intersects(
            individual_voronoi_polygons[POLYGON_COL], measurement_line.line
        )
    ]
    if (
        not intersecting_polygons.merge(
            individual_speed, on=["id", "frame"], how="left"
        )
        .notna()
        .all()
        .all()
    ):
        return DataValidationStatus.ENTRY_MISSING

    return DataValidationStatus.DATA_CORRECT
