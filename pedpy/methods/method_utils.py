"""Helper functions for the analysis methods."""
# pylint: disable=C0302

import heapq
import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import Voronoi
from shapely.affinity import rotate, scale
from shapely.vectorized import contains

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
    POINT_COL,
    POLYGON_COL,
    SPEED_COL,
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


class SpeedCalculation(Enum):  # pylint: disable=too-few-public-methods
    """Method-identifier used to compute the movement at traj borders."""

    BORDER_EXCLUDE = auto()
    BORDER_ADAPTIVE = auto()
    BORDER_SINGLE_SIDED = auto()


class AccelerationCalculation(Enum):  # pylint: disable=too-few-public-methods
    """Method-identifier used to compute the movement at traj borders."""

    BORDER_EXCLUDE = auto()


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

    Returns:
        DataFrame containing the columns 'id', 'frame' and 'neighbors', where
        neighbors are a list of the neighbor's IDs
    """
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
                f"{len(points) -len(blind_points)}) available to "
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


SHAPE_COL = "shape"
STARTPOINT_COL = "start_points"
SPEEDMAP_COL = "speedmap"


@dataclass(
    kw_only=True,
)
class GridInfo:
    """Information about the Grid.

    The Grid is used to discretize the walkable area into squares of equal size
    when using the fast-marching method.

    Attributes:
        boundary (WalkableArea): walkable area that is discretize
        cell_size (float): Length of one side of a grid cell in Meter
    """

    def __init__(
        self, walkable_area: WalkableArea, cell_size: float = 0.05
    ) -> None:
        """Initialises the class object.

        Args:
            walkable_area (WalkableArea): walkable area that is discretize
            cell_size (float): Length of one side of a grid cell
        """
        self.boundary = walkable_area.polygon
        self.cell_size = cell_size


def compute_individual_areas(
    *, grid: GridInfo, shape_and_speedmap: pd.DataFrame, cutoff_distance: float
) -> pd.DataFrame:
    """Computes the individual areas using the fast marching method.

    Args:
        grid (Grid_info): information containing the walkable area
            and the cell size of the grid
        shape_and_speedmap (pd.Dataframe): a DataFrame that includes
            information about the shape of all pedestrian and the
            speedmap used by the fast-marching-method
        cutoff_distance (float): maximal distance of the
            individual area to the pedestrian

    Returns:
        DataFrame containing the columns 'id', 'frame','polygon' (
        :class:`shapely.Polygon`), and 'individual_density' in :math:`1/m^2`.

    """
    groups = shape_and_speedmap.groupby(FRAME_COL)

    results = [
        _process_frame(
            frame=frame,
            shape_and_speedmap_framegroup=group,
            grid=grid,
            cutoff_distance=cutoff_distance,
        )
        for frame, group in groups
    ]

    data = pd.concat(results).reset_index(drop=True)
    data[DENSITY_COL] = data[POLYGON_COL].apply(
        lambda polygon: 1 / polygon.area
    )
    return data[[ID_COL, FRAME_COL, POLYGON_COL, DENSITY_COL]]


def _process_frame(
    frame: int,
    shape_and_speedmap_framegroup: pd.DataFrame,
    grid: GridInfo,
    cutoff_distance: float,
) -> pd.DataFrame:
    """Process one frame with the Fast-Marching-Method.

    Processing a frame involves transferring the position of the pedestrians
    to the grid, assigning the grid cells to the pedestrians using the
    fast-marching method and extracting the polygons from the assigned grid.

    Args:
        frame: (int) currently processed frame
        shape_and_speedmap_framegroup: information on the shape and speedmap
            of pedestrain grouped by the frame currently processed
        grid: (Grid_info) information containing the walkable area
            and the cell size of the grid
        cutoff_distance: (float) maximal distance of the
            individual area to the pedestrian

    Returns:
        returns the entered group of shape and speedmap
        with the added column polygon.
    """
    shape_and_speedmap_framegroup[STARTPOINT_COL] = [
        _form_to_grid(grid=grid, form=form)
        for form in shape_and_speedmap_framegroup[SHAPE_COL]
    ]

    time_map, wave_map = _fast_marching_algorithm(
        shape_and_speedmap=shape_and_speedmap_framegroup[
            [ID_COL, STARTPOINT_COL, SHAPE_COL, SPEEDMAP_COL]
        ],
        cutoff_distance=cutoff_distance / grid.cell_size,
    )

    poly_dict = _get_polygons_by_shapely_boxes(wave_map=wave_map, grid=grid)

    new_poly = pd.DataFrame(
        list(poly_dict.items()), columns=[ID_COL, POLYGON_COL]
    )
    group = shape_and_speedmap_framegroup.merge(new_poly, how="left", on=ID_COL)
    return group


def _form_to_grid(
    grid: GridInfo,
    form: shapely.Point | shapely.Polygon | shapely.MultiPolygon,
) -> List[tuple[int, int]]:
    """Transfers the form to the grid.

    Args:
        grid: (Grid_info) information containing the walkable area
            and the cell size of the grid
        form: (shapely.Point | shapely.Polygon | shapely.MultiPolygon)
            shapely point or polygon representing the form of a pedestrain
    Returns:
    returns a list of grid cells occupied by the form.
    """
    if isinstance(form, shapely.Point):
        return [_cartesian_to_grid(grid=grid, point=form)]
    elif isinstance(form, shapely.Polygon):
        return _grid_cells_of_polygon(grid=grid, polygon=form)
    elif isinstance(form, shapely.MultiPolygon):
        # grid_cells_list is a list of lists of cells
        grid_cells_list = [
            _grid_cells_of_polygon(grid=grid, polygon=polygon)
            for polygon in list(form.geoms)
        ]
        # flatten list
        return [
            cell for list_element in grid_cells_list for cell in list_element
        ]

    else:
        # can not process form
        return []


def _cartesian_to_grid(grid: GridInfo, point: shapely.Point) -> tuple[int, int]:
    """Converts Cartesian coordinates of point to grid coordinates."""
    minx, miny, maxx, maxy = grid.boundary.bounds

    # Here a check could be added raising an exception
    # when a point is outside of the Boundary
    # if grid.boundary.contains(point) :
    #    raise Exception(f"Point {point.x, point.y} is not inside Boundaries")

    x = point.x
    y = point.y

    j = (x - minx - grid.cell_size / 2) / grid.cell_size
    i = (y - miny - grid.cell_size / 2) / grid.cell_size

    return int(round(i)), int(round(j))


def _grid_cells_of_polygon(
    grid: GridInfo, polygon: shapely.Polygon
) -> List[tuple[int, int]]:
    """Returns a list of all cells located on the border of the polygon."""
    cart_coordinates = list(polygon.exterior.coords)
    used_cells = []
    for i in range(len(cart_coordinates) - 1):
        pt1 = shapely.Point(cart_coordinates[i])
        pt2 = shapely.Point(cart_coordinates[i + 1])

        grid_x1, grid_y1 = _cartesian_to_grid(grid=grid, point=pt1)
        grid_x2, grid_y2 = _cartesian_to_grid(grid=grid, point=pt2)

        used_cells += _bresenham_line(grid_y1, grid_x1, grid_y2, grid_x2)

    unique_cells = list(set(used_cells))
    return unique_cells


def _bresenham_line(
    y1: int, x1: int, y2: int, x2: int
) -> list[tuple[int, int]]:
    """Determines all Grid cells used on a line with Bresenham's line algorithm.

    The line starts at the Point (x1, y1) and ends at (x2, y2).
    All Coordinates correspond to the index of a Gridcell.

    Returns:
        List with used Grid cells as tuple (x, y)
    """
    y = y1
    x = x1
    dx = x2 - x1
    dy = y2 - y1

    ystep = -1 + 2 * (dy >= 0)
    dy = abs(dy)

    xstep = -1 + 2 * (dx >= 0)
    dx = abs(dx)

    ddy = 2 * dy
    ddx = 2 * dx

    used_cells = [(x1, y1)]

    if ddx >= ddy:
        errorprev = error = dx
        for _ in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    used_cells.append((x, y - ystep))
                elif error + errorprev > ddx:
                    used_cells.append((x - xstep, y))
                else:
                    used_cells.append((x, y - ystep))
                    used_cells.append((x - xstep, y))
            used_cells.append((x, y))
            errorprev = error
    else:
        errorprev = error = dy
        for _ in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    used_cells.append((x - xstep, y))
                elif error + errorprev > ddy:
                    used_cells.append((x, y - ystep))
                else:
                    used_cells.append((x - xstep, y))
                    used_cells.append((x, y - ystep))
            used_cells.append((x, y))
            errorprev = error

    return used_cells


def _grid_contains(i: int, j: int, rows: int, cols: int) -> bool:
    """Returns True if the Grid contains an entry with index (i, j)."""
    return 0 <= i < rows and 0 <= j < cols


def _calc_arrivaltime(
    i: int,
    j: int,
    speed_map: np.array,
    time_map: np.array,
    wave_id: float,
    wave_map: np.array,
) -> float:
    """Computes the arrival time of cell with index (i, j).

    The new arrival time is computed by solving a local approximation
    of the Eikonal equation using the minimum arrival time of
    neighboring cells and the local speed of the wavefront from the speed map.

    Args:
        i (int): index of the cell whose arrival time is to be calculated
        j (int): index of the cell whose arrival time is to be calculated
        speed_map (np.array): array with the propagation velocities
        time_map (np.array): array with the calculated arrival times
        wave_id (float): ID of the pedestrian whose wavefront reaches this cell
        wave_map (np.array): Existing assignment of grid cells to pedestrian ids

    Returns:
        arrival-time (float)
    """
    surrounding_cells = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = time_map.shape
    neighbour_arrival_times = []
    for di, dj in surrounding_cells:
        ni, nj = i + di, j + dj

        if _grid_contains(ni, nj, rows, cols) and wave_id == wave_map[ni, nj]:
            neighbour_arrival_times.append(time_map[ni, nj])
        else:
            neighbour_arrival_times.append(np.inf)

    a = min(neighbour_arrival_times[0:2])
    b = min(neighbour_arrival_times[2:4])

    divided_by_f_ij = 1 / speed_map[i, j]

    if divided_by_f_ij > abs(a - b):
        new_time = (a + b + np.sqrt(2 * divided_by_f_ij**2 - (a - b) ** 2)) / 2

    else:
        new_time = divided_by_f_ij + min(a, b)

    return new_time


def _fast_marching_algorithm(
    shape_and_speedmap: pd.DataFrame, cutoff_distance: float = np.inf
) -> tuple[np.array, np.array]:
    """Uses the fast-marching method to assign grid cells to the pedestrians.

    for each pedestrian a wavefront is calculated. Each grid cell is assigned
    to the wavefront that reaches this cell first. The start points correspond
    to the grid cells that a pedestrian occupies. The speedmap shows how fast
    the wave front propagates for the associated pedestrians at different cells.

    Args:
        shape_and_speedmap (pd.Dataframe): contains speedmap and start points
            for each pedestrian in one frame

        cutoff_distance (float): maximal distance of the
            individual area to the pedestrian

    Returns:
    tuple containing the arrival time map and the assigned grid cells.
    """
    rows, cols = shape_and_speedmap[SPEEDMAP_COL].iloc[-1].shape
    time_map = np.full((rows, cols), np.inf)
    visited = np.zeros((rows, cols), dtype=bool)
    wave_map = np.full((rows, cols), np.nan)

    surrounding_cells = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    for wave_id, poly_coords, speed_map in zip(
        shape_and_speedmap[ID_COL],
        shape_and_speedmap[STARTPOINT_COL],
        shape_and_speedmap[SPEEDMAP_COL],
        strict=False,
    ):
        for point in poly_coords:
            i = point[0]
            j = point[1]
            if _grid_contains(i, j, rows, cols):
                visited[i, j] = True
                time_map[i, j] = 0
                wave_map[i, j] = wave_id

            for di, dj in surrounding_cells:
                ni, nj = i + di, j + dj

                if _grid_contains(ni, nj, rows, cols):
                    if visited[ni, nj] or np.isinf(speed_map[ni, nj]):
                        continue
                    else:
                        new_time = _calc_arrivaltime(
                            ni, nj, speed_map, time_map, wave_id, wave_map
                        )
                        if new_time < cutoff_distance:
                            heapq.heappush(heap, (new_time, (ni, nj), wave_id))
    while heap:
        current_time, (i, j), current_wave = heapq.heappop(heap)
        speed_map = shape_and_speedmap[
            shape_and_speedmap[ID_COL] == current_wave
        ][SPEEDMAP_COL].iloc[0]

        if visited[i, j]:
            continue

        visited[i, j] = True
        wave_map[i, j] = current_wave
        time_map[i, j] = current_time

        for di, dj in surrounding_cells:
            ni, nj = i + di, j + dj

            if _grid_contains(ni, nj, rows, cols):
                if visited[ni, nj] or np.isinf(speed_map[ni, nj]):
                    continue

                new_time = _calc_arrivaltime(
                    ni, nj, speed_map, time_map, current_wave, wave_map
                )
                if new_time < cutoff_distance:
                    heapq.heappush(heap, (new_time, (ni, nj), current_wave))

    return time_map, wave_map


def _get_polygons_by_shapely_boxes(
    wave_map: np.array, grid: GridInfo
) -> dict[int, shapely.Polygon]:
    """Determins the Polygons of the assigned wave_map.

    Determins the Polygons for every unique value exept np.isnan.
    To determin the Polygon a shapely boxes is created for each cell.
    All cells are joined via shapely unary union.

    Args:
        wave_map (np.array): Assignment of the grid cells.
            Each number corresponds to the ID of a pedestrian.

        grid (Grid_info): information containing the walkable area
            and the cell size of the grid

    Returns:
        A dictionary where each key corresponds to an associated pedestrian,
        and the value is the polygon representing that pedestrian's area.
    """
    m, n = wave_map.shape
    minx, miny, maxx, maxy = grid.boundary.bounds
    x_coords = np.linspace(minx, maxx, n + 1)
    y_coords = np.linspace(miny, maxy, m + 1)

    x1, y1 = np.meshgrid(x_coords[:-1], y_coords[1:])
    x2, y2 = np.meshgrid(x_coords[1:], y_coords[:-1])

    x1 = x1.flatten()
    y1 = y1.flatten()
    x2 = x2.flatten()
    y2 = y2.flatten()

    boxes = np.array(
        [shapely.box(x1[i], y1[i], x2[i], y2[i]) for i in range(len(x1))]
    ).reshape(m, n)

    unique_values = np.unique(wave_map)
    polygons = {}

    for value in unique_values:
        if np.isnan(value):  # background and not part of any polygon
            continue
        mask = wave_map == value
        shapes = boxes[mask]

        multi_polygon = shapely.unary_union(shapes)
        polygons[value] = multi_polygon

    return polygons


def compute_shape_and_speedmap(
    *,
    traj: TrajectoryData,
    grid: GridInfo,
    additions: pd.DataFrame,
    calc_shape_from_frame: Callable,
    calc_speedmap_from_frame: Callable,
) -> pd.DataFrame:
    """Computes pedestrian shapes and speed maps from trajectory data per frame.

    This function processes trajectory data for pedestrians by grouping the
    data into frames. For each frame group, two user-defined functions are
    applied: one to compute pedestrian shapes based on the provided frame data,
    and another to compute a speed map on a given grid for further analysis.

    Args:
        traj (TrajectoryData): trajectory data

        grid (Grid_info): information containing the walkable area
            and the cell size of the grid

        additions (pd.DataFrame):
            A DataFrame that contains additional information used when applying
            the callable functions calc_shape_from_frame and
            calc_speedmap_from_frame

        calc_shape_from_frame (Callable):
            A user-defined function that computes the shape of each pedestrian
            in the frame group. This function receives a group of trajectories
            (as a DataFrame), the grid information, and the additions DataFrame
            as inputs, and returns a DataFrame with an added shape column.

        calc_speedmap_from_frame (Callable):
            A user-defined function that computes the speed map for the
            pedestrians on the grid. This function receives the frame group,
            the grid, and the additions DataFrame, and returns a DataFrame
            with the added SPEEDMAP_COL.

    Returns:
        A DataFrame containing the columns 'id', 'frame', 'speedmap' and 'shape'

    .. note::
        The `traj` data is grouped by frames, and the shape and speed map
            calculations are done per group.
        The `additions` can but must not be a DataFrame.
        The function is flexible, allowing the user to define custom behavior
            for calculating pedestrian shapes and speed maps.

    """
    traj_data = traj.data
    result = traj_data.groupby(FRAME_COL)[traj_data.columns].apply(
        lambda group: calc_shape_from_frame(group, grid, additions)
    )
    traj_data = result.reset_index(drop=True)

    result = traj_data.groupby(FRAME_COL)[traj_data.columns].apply(
        lambda group: calc_speedmap_from_frame(group, grid, additions)
    )
    traj_data = result.reset_index(drop=True)
    return traj_data[[ID_COL, FRAME_COL, SHAPE_COL, SPEEDMAP_COL]]


def compute_speedmap(
    polygon: shapely.Polygon,
    cell_size: float,
    inside_speed: float = 1,
    outside_speed: float = np.inf,
) -> np.array:
    """Creates speedmap used by the Fast Marching Method from a polygonal area.

    This function creates a 2D grid where each cell is assigned a speed value
    depending on whether its center lies inside or outside the provided polygon.
    The resulting speedmap can be used to control the propagation of wavefronts
    within a defined area.

    Args:
        polygon (shapely.Polygon): polygon defining the boundary of the area.

        cell_size (float): The size of each cell in the grid.

        inside_speed (float, optional): The speed value for cells
            inside the polygon. Default is 1.

        outside_speed (float, optional): The speed value for cells
            outside the polygon. Default is np.inf.

    Returns:
        A 2D array representing the speedmap, where each cell contains
        either the inside_speed or outside_speed value.

    The grid is constructed such that the speed within the polygon is set to
    inside_speed, and outside the polygon, it is set to outside_speed. The
    center of each cell is checked to determine if it lies within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds

    rows = int(np.ceil((maxy - miny) / cell_size))
    cols = int(np.ceil((maxx - minx) / cell_size))

    x_coords = np.linspace(minx + cell_size / 2, maxx - cell_size / 2, cols)
    y_coords = np.linspace(miny + cell_size / 2, maxy - cell_size / 2, rows)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # contains from shapely.vectorized
    mask = contains(polygon, xv, yv)

    speed_map = np.full((rows, cols), outside_speed)

    speed_map[mask] = inside_speed

    return speed_map


def compute_individial_speedmap(
    grid: GridInfo,
    position: shapely.Point,
    v_x: float,
    v_y: float,
    inside_speed: float = 1.0,
    outside_speed: float = np.inf,
    blocked_multiplier: float = 0.25,
    not_seen_multiplier: float = 0.5,
) -> np.array:
    """Creates an speedmap for one person for the Fast-Marching-Method.

    This function generates a customized speed map for an individual pedestrian
    based on the grid's boundary, their current position, and the direction
    they are facing (specified by `v_x` and `v_y`). The speed map accounts for
    obstacles blocking the pedestrian's view, as well as areas outside their
    direct line of sight.

    The speed values for the map cells are assigned as follows:
    - Cells inside the boundary polygon are assigned the `inside_speed`.
    - Cells outside the boundary polygon are assigned the `outside_speed`.
    - If the line of sight between the pedestrian's position and the cell center
      is blocked, the cell's speed is multiplied by `blocked_multiplier`.
    - If the cell is behind the pedestrian's facing direction, the speed is
      multiplied by `not_seen_multiplier`.

    Args:
        grid (Grid_info): Information about the grid, including the boundary
            polygon and the cell size for constructing the speed map.

        position (shapely.Point): The pedestrian's current position.

        v_x (float): The x-component of the pedestrian's direction vector.

        v_y (float): The y-component of the pedestrian's direction vector.

        inside_speed (float, optional): The speed assigned to cells inside the
            boundary polygon. Default is 1.0.

        outside_speed (float, optional): The speed assigned to cells outside the
            boundary polygon. Default is `np.inf`.

        blocked_multiplier (float, optional): The multiplier applied to the
            speed of cells that are blocked by obstacles from the
            pedestrian's view. Default is 0.25.

        not_seen_multiplier (float, optional): The multiplier applied to the
            speed of cells that are behind the pedestrian's facing direction.
            Default is 0.5.

    Returns:
        numpy.ndarray: A 2D array representing the individual speed map for the
        pedestrian, where each cell contains an adjusted speed value based on
        its location, visibility, and the pedestrian's view.

    .. note::
        - A cell is considered blocked if an obstacle lies between the
          pedestrian's position and the cell center.
        - A cell is considered "not seen" if it lies behind the pedestrian, as
          determined by the facing vector (`v_x`, `v_y`).
        - The grid is divided into cells of size `grid.cell_size`, and the
          center of each cell is evaluated to determine its speed value.
    """
    minx, miny, maxx, maxy = grid.boundary.bounds
    cell_size = grid.cell_size
    rows = int(np.ceil((maxy - miny) / cell_size))
    cols = int(np.ceil((maxx - minx) / cell_size))
    x_coords = np.linspace(minx + cell_size / 2, maxx - cell_size / 2, cols)
    y_coords = np.linspace(miny + cell_size / 2, maxy - cell_size / 2, rows)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # contains from shapely.vectorized
    mask = contains(grid.boundary, xv, yv)

    speed_map = np.full((rows, cols), outside_speed)

    speed_map[mask] = inside_speed

    for i in range(rows):
        for j in range(cols):
            cell_center = shapely.Point(xv[i, j], yv[i, j])
            if np.isinf(speed_map[i, j]) or speed_map[i, j] == 0:
                # is unreachable anyway
                continue

            # Line between position and cell center
            line = shapely.LineString([position, cell_center])

            # if line of sight is blocked by the boundary polygon
            if not grid.boundary.contains(line):
                speed_map[i, j] *= blocked_multiplier

            # Vector to Cell Center
            cell_vector_x = cell_center.x - position.x
            cell_vector_y = cell_center.y - position.y

            # Normalization of the line of sight vector
            direction_length = np.hypot(v_x, v_y)
            if direction_length != 0:
                direction_x = v_x / direction_length
                direction_y = v_y / direction_length
            else:
                # no direction given
                continue

            dot_product = (
                cell_vector_x * direction_x + cell_vector_y * direction_y
            )

            if dot_product < 0:
                # gridcell i, j is not infront of the pedestrian
                speed_map[i, j] *= not_seen_multiplier

    return speed_map


def apply_point_as_shape(
    group: pd.DataFrame, grid: GridInfo, additions: None
) -> pd.DataFrame:
    """Assigns point column as shapes for each entry in the frame group.

    This function assigns the pedestrian positions (POINT_COL) directly as
    shapes (SHAPE_COL) without further calculations. It can be used as the
    shape computation function in :func:`compute_shape_and_speedmap`.

    Args:
        group (pd.DataFrame): A DataFrame representing a frame group of
            pedestrian trajectories.

        grid (Grid_info): Grid information, not used in this function but
            required for compatibility with :func:`compute_shape_and_speedmap`.

        additions (None): No additional data is required for this function.
            unused when given but required for compatibility with
            :func:`compute_shape_and_speedmap`.

    Returns:
        The modified frame group DataFrame with an added shape column
        where each pedestrian's position is assigned as their shape.
    """
    group[SHAPE_COL] = group[POINT_COL]
    return group


def apply_speedmap_from_additions(
    group: pd.DataFrame, grid: GridInfo, additions: dict[str, np.array]
) -> pd.DataFrame:
    """Assigns a precomputed speedmap to each entry in the frame group.

    This function retrieves a precomputed speed map from the `additions`
    dictionary (under the `SPEEDMAP_COL` key) and assigns it to each pedestrian
    entry in the frame group. The same speed map is applied to all entries
    within the group.

    Args:
        group (pd.DataFrame): A DataFrame representing a frame group of
            pedestrian trajectories.

        grid (Grid_info): Information about the grid, not used in this function
            but required for compatibility with
            :func:`compute_shape_and_speedmap`.

        additions (dict): A Dictionary containing additional data,
            including the precomputed speed map under the `SPEEDMAP_COL` key.

    Returns:
        pd.DataFrame: The modified frame group DataFrame with an added
        `SPEEDMAP_COL`, where all entries reference the same speed map from
        `additions`.

    Raises:
        KeyError: If the `SPEEDMAP_COL` key is not found in `additions`.

    """
    group[SPEEDMAP_COL] = [additions[SPEEDMAP_COL]] * len(group)
    return group


def apply_individual_speedmap(
    group: pd.DataFrame, grid: GridInfo, additions: pd.DataFrame
) -> pd.DataFrame:
    """Computes individual speedmaps of pedestrians with position and direction.

    This function merges the input group DataFrame with additional information
    from the `additions` DataFrame, then calculates an individual speed map
    for each pedestrian using their position and velocity components. The
    speed maps are computed using the :func:`compute_individual_speedmap`
    function, which considers local circumstances such as the pedestrian's
    position and facing direction. The resulting speedmaps are added to the
    DataFrame as a new column. This function can be used in conjunction
    with :func:`compute_shape_and_speedmap`.

    Args:
        group (pd.DataFrame): A DataFrame representing a frame group of
            pedestrian trajectories.

        grid (Grid_info): Information about the grid, including the boundary
            and cell size, necessary for computing individual speed maps.

        additions (pd.DataFrame): A DataFrame containing additional data
            associated with each pedestrian, which includes identifiers
            for merging with the `group` DataFrame. Needs to include the columns
            'id', 'frame', 'v_x', 'v_y'. The walking direction can be computed
            by :func:`~speed_calculator.compute_individual_speed` using the
            option :code:`compute_velocity`

    Returns:
        pd.DataFrame: The merged DataFrame containing all original columns
        from `group`, along with an added `SPEEDMAP_COL` that holds the
        individual speedmaps for each pedestrian.
    """
    merged = group.merge(additions, on=[ID_COL, FRAME_COL], how="left")
    merged[SPEEDMAP_COL] = merged.apply(
        lambda row: compute_individial_speedmap(
            grid=grid,
            position=row[POINT_COL],
            v_x=row[V_X_COL],
            v_y=row[V_Y_COL],
        ),
        axis=1,
    )
    return merged


def apply_computed_speedmap(
    group: pd.DataFrame, grid: GridInfo, additions: None
) -> pd.DataFrame:
    """Computes speedmap for the grid and assigns to the entire frame group.

    This function computes a speed map based on the provided grid's boundary
    and cell size using the `compute_speedmap` function. The same speed map is
    assigned to all pedestrians in the frame group. It can be used as the speed
    map computation function in :func:`compute_shape_and_speedmap`.

    Args:
        group (pd.DataFrame): A DataFrame representing a frame group of
            pedestrian trajectories.

        grid (Grid_info): Information containing the grid's walkable area
            (boundary) and the cell size used to compute the speed map.

        additions (pd.DataFrame): Additional data passed for compatibility
            with :func:`compute_shape_and_speedmap`, but not used in this
            function.

    Returns:
        The modified frame group DataFrame with an added
        `SPEEDMAP_COL` where a uniform speed map is assigned to each entry.

    """
    group[SPEEDMAP_COL] = [
        compute_speedmap(polygon=grid.boundary, cell_size=grid.cell_size)
    ] * len(group)
    return group


def create_elipse(
    point: shapely.Point, width: float, length: float, alpha: float
) -> shapely.Polygon:
    """Creates an ellipse at point with specified width, length and rotation.

    This function generates an elliptical shape using the Shapely library
    by calculating the necessary points and applying geometric transformations.
    The ellipse is defined by its center point, width, length, and angle of
    rotation.

    Args:
        point (shapely.geometry.Point): The center point of the ellipse.

        width (float): The width (semi-minor axis) of the ellipse.

        length (float): The length (semi-major axis) of the ellipse.

        alpha (float): The angle of rotation of the ellipse in radians.

    Returns:
        shapely.Polygon: A Shapely Polygon object representing the
        elliptical shape.

    Notes:
        This function is based on an implementation found at:
        https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely
    """
    # create a circle with radius 1
    c = point.buffer(1)

    # Resize this circle to create an ellipse that matches the length and width.
    c = scale(c, length / 2, width / 2)

    # Rotate the ellipse to the specified angle
    c = rotate(c, alpha, use_radians=True)
    return c


def compute_width_length_alpha(
    v: float,
    v_x: float,
    v_y: float,
    a_min: float = 0.2,
    b_min: float = 0.2,
    b_max: float = 0.4,
    v0: float = 1.2,
    a_v: float = 1,
) -> tuple[float, float, float]:
    r"""Computes the width, length, and angle of the ellipse of the GCFM.

    This function calculates the dimensions (width and length) and
    orientation (angle) of an ellipse representing a pedestrian's shape
    based on their speed and other parameters. The width and length are
    adjusted according to the pedestrian's speed relative to a reference
    speed (`v0`).

    .. math::
        length = a_{\\min} + a_v \\cdot v_i \\; \text{and}, \\
        width = b_{\\max} - (b_ {\\max} - b_{\\min}) \\cdot \frac{v_i}{v_0}.

    .. note::
        The default values for this functions are given in meters. When
        using a different unit like centimeters the values need to be adapted.

    Args:
        v (float): The current speed of the pedestrian.

        v_x (float): The x-component of the pedestrian's velocity vector.

        v_y (float): The y-component of the pedestrian's velocity vector.

        a_min (float): The minimum length of the ellipse.
            Default is 0.2.

        b_min (float): The minimum width of the ellipse.
            Default is 0.2.

        b_max (float): The maximum width of the ellipse.
            Default is 0.4

        v0 (float, optional): The desired speed.
            Default is 1.2.

        a_v (float, optional): The additional length increment based on speed.
            Default is 1.

    Returns:
        tuple: A tuple containing the calculated width, length, and angle
        of the ellipse (width, length, angle in radians).
    """
    width = b_max - (b_max - b_min) * (v / v0)
    length = a_min + a_v * v
    angle = math.atan2(v_y, v_x)
    return width, length, angle


def apply_ellipses_as_shape(group, grid, additions):
    """Computes and assigns elliptical shapes for pedestrians.

    This function is compatible with :func:`compute_shape_and_speedmap`,
    which can utilize the shapes computed here.
    This function merges trajectory data with additional parameters and
    calculates the width, length, and orientation of ellipses representing
    pedestrians. It uses the `_compute_width_length_alpha` function to
    derive the ellipse dimensions based on speed and optional parameters.
    Default values are used if not specified in the `additions` DataFrame.

    Args:
        group (pd.DataFrame): A DataFrame representing a frame group of
            pedestrian trajectories.

        grid (Grid_info): Information about the walkable area and grid cell
            size.

        additions (pd.DataFrame): A DataFrame containing additional parameters
            for the shape computation. Mandatory columns are 'id', 'frame',
            SPEED_COL, 'v_x', and 'v_y'. Optional columns include values for the
            calculation of length and width of the ellipsis with
            :func:`compute_length_width_alpha`.
            The optional columns are 'a_min', 'b_min', 'b_max', 'v0', and 'a_v'.

    Returns:
        pd.DataFrame: The modified DataFrame with an added column for
        elliptical shapes (SHAPE_COL) adjusted for overlap with the grid
        boundaries and other agents.

    Notes:
        - The function ensures that the computed shapes do not overlap
          with other agents' shapes.
    """
    default_values = {
        "a_min": 0.2,
        "b_min": 0.2,
        "b_max": 0.4,
        "v0": 1.2,
        "a_v": 1,
    }
    # merge additional information on group
    merged = group.merge(additions, on=[ID_COL, FRAME_COL], how="left")

    for key, value in default_values.items():
        if key not in merged:
            merged[key] = value

    merged["w_l_a"] = merged.apply(
        lambda row: compute_width_length_alpha(
            v=row[SPEED_COL],
            v_x=row[V_X_COL],
            v_y=row[V_Y_COL],
            a_min=row["a_min"],
            b_min=row["b_min"],
            b_max=row["b_max"],
            v0=row["v0"],
            a_v=row["a_v"],
        ),
        axis=1,
    )
    # add shape column
    merged[SHAPE_COL] = merged.apply(
        lambda row: create_elipse(row[POINT_COL], *row["w_l_a"]), axis=1
    )
    # remove intersection with boundary
    merged[SHAPE_COL] = shapely.intersection(
        merged[SHAPE_COL], grid.boundary.buffer(-grid.cell_size)
    )

    # remove overlap with other agents
    shapes_without_overlap = []
    for index, row in merged.iterrows():
        other_geoms = merged.drop(index)[SHAPE_COL]
        geom_union = (
            shapely.unary_union(other_geoms)
            .buffer(grid.cell_size)
            .simplify(tolerance=0.001)
        )
        shapes_without_overlap.append(
            row[SHAPE_COL].difference(geom_union).simplify(tolerance=0.001)
        )

    merged[SHAPE_COL] = shapes_without_overlap

    return merged
