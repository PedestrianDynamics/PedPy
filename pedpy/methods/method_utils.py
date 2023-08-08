"""Helper functions for the analysis methods."""
import itertools
import logging
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import Voronoi

from pedpy.data.geometry import MeasurementArea, MeasurementLine, WalkableArea
from pedpy.data.trajectory_data import TrajectoryData

log = logging.getLogger(__name__)


def is_trajectory_valid(
    *, traj: TrajectoryData, geometry: WalkableArea
) -> bool:
    """Checks if all trajectory data points lie within the given geometry.

    Args:
        traj (TrajectoryData): trajectory data
        geometry (WalkableArea): geometry

    Returns:
        All points lie within geometry
    """
    return get_invalid_trajectory(traj=traj, geometry=geometry).empty


def get_invalid_trajectory(
    *, traj: TrajectoryData, geometry: WalkableArea
) -> pd.DataFrame:
    """Returns all trajectory data points outside the given geometry.

    Args:
        traj (TrajectoryData): trajectory data
        geometry (WalkableArea): geometry

    Returns:
        DataFrame showing all data points outside the given geometry
    """
    return traj.data.loc[
        ~shapely.within(traj.data.points, geometry.walkable_area)
    ]


def compute_frame_range_in_area(
    *, traj_data: pd.DataFrame, measurement_line: MeasurementLine, width: float
) -> Tuple[pd.DataFrame, MeasurementArea]:
    """Compute the frame ranges for each pedestrian inside the measurement area.

    Note:
        Only pedestrians passing the complete measurement area will be
        considered. Meaning they need to cross measurement_line and the line
        with the given offset in one go. If leaving the area between two lines
        through the same line will be ignored.

        As passing we define the frame, the pedestrians enter the area and
        then move through the complete area without leaving it. Hence,
        doing a closed analysis of the movement area with several measuring
        ranges underestimates the actual movement time.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_line (MeasurementLine): measurement line
        width (float): distance to the second measurement line

    Returns:
        DataFrame containing the columns: 'ID', 'frame_start', 'frame_end' and
        the created measurement area
    """
    # Create the second with the given offset
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

    crossing_frames_first = compute_crossing_frames(
        traj_data=traj_data, measurement_line=measurement_line
    )
    crossing_frames_second = compute_crossing_frames(
        traj_data=traj_data, measurement_line=second_line
    )

    start_crossed_1 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_first,
        check_column="frame_start",
        column_name="start_crossed_1",
    )
    end_crossed_1 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_first,
        check_column="frame_end",
        column_name="end_crossed_1",
    )
    start_crossed_2 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_second,
        check_column="frame_start",
        column_name="start_crossed_2",
    )
    end_crossed_2 = _check_crossing_in_frame_range(
        inside_range=inside_range,
        crossing_frames=crossing_frames_second,
        check_column="frame_end",
        column_name="end_crossed_2",
    )

    frame_range_between_lines = (
        start_crossed_1.merge(
            start_crossed_2, how="outer", on=["ID", "frame_start", "frame_end"]
        )
        .merge(
            end_crossed_1, how="outer", on=["ID", "frame_start", "frame_end"]
        )
        .merge(
            end_crossed_2, how="outer", on=["ID", "frame_start", "frame_end"]
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
        frame_range_between_lines.loc[:, ("ID", "frame_start", "frame_end")],
        measurement_area,
    )


def compute_neighbors(individual_voronoi_data: pd.DataFrame) -> pd.DataFrame:
    """Compute the neighbors of each pedestrian based on the Voronoi cells.

    Computation of the neighborhood of each pedestrian per frame. Every other
    pedestrian is a neighbor if the Voronoi cells of both pedestrian intersect
    and some point.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data, needs
            to contain a column 'individual voronoi', which holds
            shapely.Polygon information

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and 'neighbors', where
        neighbors are a list of the neighbor's IDs
    """
    neighbor_df = []

    for frame, frame_data in individual_voronoi_data.groupby("frame"):
        touching = shapely.dwithin(
            np.array(frame_data["individual voronoi"])[:, np.newaxis],
            np.array(frame_data["individual voronoi"])[np.newaxis, :],
            1e-9,  # Voronoi cells as close as 1 mm are touching
        )

        # the peds are not neighbors of themselves
        for i in range(len(touching)):
            touching[i, i] = False

        # create matrix with ped IDs
        ids = np.outer(
            np.ones_like(frame_data["ID"].values),
            frame_data["ID"].values.reshape(1, -1),
        )

        neighbors = np.where(touching, ids, np.nan)

        neighbors_list = [
            np.array(l)[~np.isnan(np.array(l))].astype(int).tolist()
            for l in neighbors
        ]

        frame_df = pd.DataFrame(
            zip(
                frame_data["ID"].values, itertools.repeat(frame), neighbors_list
            ),
            columns=["ID", "frame", "neighbors"],
        )
        neighbor_df.append(frame_df)

    return pd.concat(neighbor_df)


def compute_time_distance_line(
    *, traj_data: pd.DataFrame, measurement_line: MeasurementLine
) -> pd.DataFrame:
    """Compute the time and distance to the measurement line.

    Compute the time (in frames) and distance to the first crossing of the
    measurement line. For further information how the crossing frames are
    computed see :obj:`~compute_crossing_frames`. All frames after a
    pedestrian has crossed the line will be omitted in the results.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_line (MeasurementLine): line which is crossed

    Returns: DataFrame containing 'ID', 'frame', 'time' (frames to crossing),
        and 'distance' (to measurement line)
    """
    df_distance_time = traj_data[["ID", "frame", "points"]].copy(deep=True)

    # Compute distance to measurement line
    df_distance_time["distance"] = shapely.distance(
        df_distance_time["points"], measurement_line.line
    )

    # Compute time to entrance
    crossing_frame = compute_crossing_frames(
        traj_data=traj_data, measurement_line=measurement_line
    ).rename(columns={"frame": "crossing_frame"})
    df_distance_time = df_distance_time.merge(crossing_frame, on="ID")
    df_distance_time["time"] = (
        df_distance_time["crossing_frame"] - df_distance_time["frame"]
    )

    # Delete all rows where the line has already been passed
    df_distance_time = df_distance_time[df_distance_time.time >= 0]

    return df_distance_time.loc[:, ["ID", "frame", "distance", "time"]]


def compute_individual_voronoi_polygons(
    *,
    traj_data: pd.DataFrame,
    geometry: WalkableArea,
    cut_off: Optional[Tuple[float, int]] = None,
    use_blind_points: bool = True,
) -> pd.DataFrame:
    """Compute the individual voronoi cells for each person and frame.

    Args:
        traj_data (pd.DataFrame): trajectory data
        geometry (WalkableArea): bounding area, where pedestrian are supposed to be
        cut_off (Tuple[float, int]): radius of max extended voronoi cell (in
                m), number of linear segments in the approximation of circular
                arcs, needs to be divisible by 4!
        use_blind_points (bool): adds extra 4 points outside the geometry to
                also compute voronoi cells when less than 4 peds are in the
                geometry (default: on!)

    Returns:
        DataFrame containing the columns: 'ID', 'frame','individual voronoi',
        and 'individual density' in 1/m^2.
    """
    dfs = []

    bounds = geometry.walkable_area.bounds
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

    for frame, peds_in_frame in traj_data.groupby(traj_data.frame):
        points = peds_in_frame[["X", "Y"]].values
        points = np.concatenate([points, blind_points])

        # only skip analysis if less than 4 peds are in the frame and blind
        # points are turned off
        if not use_blind_points and len(points) - len(blind_points) < 4:
            log.warning(
                f"Not enough pedestrians (N="
                f"{len(points) -len(blind_points)}) available to "
                f"calculate Voronoi cells for frame = {frame}. "
                f"Consider enable use of blind points."
            )
            continue

        vor = Voronoi(points)
        voronoi_polygons = _clip_voronoi_polygons(vor, clipping_diameter)

        voronoi_polygons = voronoi_polygons[:-4]
        voronoi_in_frame = peds_in_frame.loc[:, ("ID", "frame", "points")]

        # Compute the intersecting area with the walkable area
        voronoi_in_frame["individual voronoi"] = shapely.intersection(
            voronoi_polygons, geometry.walkable_area
        )

        if cut_off is not None:
            num_edges = cut_off[1]
            radius = cut_off[0]
            quad_edges = int(num_edges / 4)
            voronoi_in_frame["individual voronoi"] = shapely.intersection(
                voronoi_in_frame["individual voronoi"],
                shapely.buffer(
                    peds_in_frame["points"], radius, quad_segs=quad_edges
                ),
            )

        # Only consider the parts of a multipolygon which contain the position
        # of the pedestrian
        voronoi_in_frame.loc[
            shapely.get_type_id(voronoi_in_frame["individual voronoi"]) != 3,
            "individual voronoi",
        ] = voronoi_in_frame.loc[
            shapely.get_type_id(voronoi_in_frame["individual voronoi"]) != 3, :
        ].apply(
            lambda x: shapely.get_parts(x["individual voronoi"])[
                shapely.within(
                    x["points"], shapely.get_parts(x["individual voronoi"])
                )
            ][0],
            axis=1,
        )

        dfs.append(voronoi_in_frame)

    result = pd.concat(dfs)[["ID", "frame", "individual voronoi"]]
    result["individual density"] = 1.0 / shapely.area(
        result["individual voronoi"]
    )

    return result


def compute_intersecting_polygons(
    *, individual_voronoi_data: pd.DataFrame, measurement_area: MeasurementArea
) -> pd.DataFrame:
    """Compute the intersection of the voronoi cells with the measurement area.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data, needs
                to contain a column 'individual voronoi' which holds
                shapely.Polygon information
        measurement_area (MeasurementArea): measurement area for which the
            intersection will be computed.

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and
        'intersection voronoi'.
    """
    df_intersection = individual_voronoi_data[["ID", "frame"]].copy()
    df_intersection["intersection voronoi"] = shapely.intersection(
        individual_voronoi_data["individual voronoi"], measurement_area.polygon
    )
    return df_intersection


def _clip_voronoi_polygons(  # pylint: disable=too-many-locals,invalid-name
    voronoi: Voronoi, diameter: float
) -> List[shapely.Polygon]:
    """Generate Polygons from the Voronoi diagram.

    Generate shapely.geometry.Polygon objects corresponding to the
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
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
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
    *, traj_data: pd.DataFrame, frame_step: int, bidirectional: bool = True
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step.

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame will be used as boundary. Hence, the intervals become [frame,
    frame + frame_step], or [frame - frame_step, frame] respectively.

    Args:
        traj_data (pd.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the prev. frame_step points will
            be used to determine the movement

    Returns:
        DataFrame containing the columns: 'ID', 'frame', 'start', 'end',
        'start_frame, and 'end_frame'. Where 'start'/'end' are the points
        where the movement start/ends, and 'start_frame'/'end_frame' are the
        corresponding frames.
    """
    df_movement = traj_data.copy(deep=True)

    df_movement["start"] = (
        df_movement.groupby("ID")["points"]
        .shift(frame_step)
        .fillna(df_movement["points"])
    )
    df_movement["start_frame"] = (
        df_movement.groupby("ID")["frame"]
        .shift(frame_step)
        .fillna(df_movement["frame"])
    )

    if bidirectional:
        df_movement["end"] = (
            df_movement.groupby("ID")["points"]
            .shift(-frame_step)
            .fillna(df_movement["points"])
        )
        df_movement["end_frame"] = (
            df_movement.groupby("ID")["frame"]
            .shift(-frame_step)
            .fillna(df_movement["frame"])
        )
    else:
        df_movement["end"] = df_movement["points"]
        df_movement["end_frame"] = df_movement["frame"]

    return df_movement[
        ["ID", "frame", "start", "end", "start_frame", "end_frame"]
    ]


def compute_crossing_frames(
    *, traj_data: pd.DataFrame, measurement_line: MeasurementLine
) -> pd.DataFrame:
    """Compute the frames at the pedestrians pass the measurement line.

    As crossing we define a movement that moves across the measurement line.
    When the movement ends on the line, the line is not crossed. When it
    starts on the line, it counts as crossed.

    Note:
        Due to oscillations, it may happen that a pedestrian crosses the
        measurement line multiple times in a small-time interval.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_line (MeasurementLine):

    Returns:
        DataFrame containing the columns: 'ID', 'frame', where 'frame' are
        the frames where the measurement line is crossed.
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
                shapely.get_coordinates(df_movement["start"]),
                shapely.get_coordinates(df_movement["end"]),
            ],
            axis=1,
        )
    )

    # crossing means, the current movement crosses the line and the end point
    # of the movement is not on the line. The result is sorted by frame number
    crossing_frames = df_movement.loc[
        shapely.intersects(df_movement["movement"], measurement_line.line)
    ][["ID", "frame"]]

    return crossing_frames


def _get_continuous_parts_in_area(
    *, traj_data: pd.DataFrame, measurement_area: MeasurementArea
) -> pd.DataFrame:
    """Compute the time-continuous parts of each pedestrian in the area.

    Compute the time-continuous parts in which the pedestrians are inside
    the given measurement area. As leaving the first frame outside the area is
    considered.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_area (MeasurementArea): area which is considered

    Returns:
        DataFrame containing the columns: 'ID', 'frame_start', 'frame_end'
    """
    inside = traj_data.loc[
        shapely.within(traj_data.points, measurement_area.polygon), :
    ].copy()
    inside.loc[:, "g"] = inside.groupby("ID", group_keys=False)["frame"].apply(
        lambda x: x.diff().ge(2).cumsum()
    )
    inside_range = (
        inside.groupby(["ID", "g"])
        .agg(
            frame_start=("frame", "first"),
            frame_end=("frame", "last"),
        )
        .reset_index()[["ID", "frame_start", "frame_end"]]
    )
    inside_range["frame_end"] += 1

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
        inside_range (pd.DataFrame): DataFrame containing the columns 'ID'
            and check_column
        crossing_frames (pd.DataFrame): DataFrame containing the columns 'ID'
            and 'frame'
        check_column (str): name of the column in inside_range which represents
            a frame value. Needs to be 'frame_start' or 'frame_end'
        column_name (str): name of the result column

    Returns:
        DataFrame containing the columns 'ID', 'frame_start', 'frame_end',
        column_name
    """
    assert check_column in (
        "frame_start",
        "frame_end",
    ), "check_column needs to be 'frame_start' or 'frame_end'"

    crossed = pd.merge(
        inside_range,
        crossing_frames,
        left_on=["ID", check_column],
        right_on=["ID", "frame"],
        how="left",
        indicator=column_name,
    )[["ID", "frame_start", "frame_end", column_name]]
    crossed[column_name] = crossed[column_name] == "both"
    crossed = crossed[crossed[column_name]]
    return crossed
