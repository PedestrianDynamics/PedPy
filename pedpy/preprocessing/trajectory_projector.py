"""Provides functions for projecting trajectories."""

import copy
import math
import warnings
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely

from pedpy import GeometryError, InputError
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import get_invalid_trajectory, is_trajectory_valid


def correct_invalid_trajectories(
    trajectory_data: TrajectoryData,
    walkable_area: WalkableArea,
    back_distance_wall: float = -1,
    min_distance_wall: float = 0.01,
    max_distance_wall: float = 0.05,
    back_distance_obst: float = -1,
    min_distance_obst: float = 0.01,
    max_distance_obst: float = 0.05,
) -> tuple[TrajectoryData, list[int]]:
    r"""Corrects invalid trajectories.

    When dealing with head trajectories, it may happen that the participants lean
    over the obstacles. This means that their trajectory will leave the walkable area
    at some frames, this data can not be processed with PedPy.

    The function locates false points and points, are too close to a wall but
    still outside of it, and corrects them by pushing them slightly away. Depending
    on the geometry and the parameter values, it can be beneficial to buffer
    the geometry beforehand to create thicker walls.

    At the beginning it checks if the trajectory is valid, so that the whole process
    will not run unnecessary.

    It returns a corrected version of the trajectory input (which is also
    a pedpy.TrajectoryData).

    If a point lays inside the geometry or close to it, the  point will be moved away
    outside the geometry. The new distance is calculated by linear interpolation.
    Points that lay further inside an obstacle/wall have a smaller new distance compared
    to a point that lays at the end of the range. The range is between
    back_distance and max_distance. The unit for all values is meters.

    max_distance describes how far max. points can be moved out. Points that lay between
    the wall and this parameter are also slightly pushed away.
    Furthermore, a value >0 is necessary for a mostly accurate linear interpolation for
    all points, that need to be moved.

    .. image:: /images/parameters_preprocessing.png
        :width: 60 %
        :align: center

    The formula for the calculation for the moving points look like:

    .. math::
        x' = (x - a) \cdot (\frac{c - b}{c - a} )+ b

    x' is the new distance to the wall
    x is the old distance to the wall, \n
    which is either from the inside or not far enough away \n
    a is equal to back_distance \n
    b is equal to min_distance  \n
    c is equal to max_distance  \n

    Args:
        trajectory_data (pedpy.TrajectoryData): The trajectory data to be tested and
            corrected
        walkable_area (pedpy.WalkableArea): The belonging walkable area
        back_distance_wall (float): in meters, has to be <0. The distance behind the wall, till
            which the points inside  the walls should be corrected. Points, which are
            further inside the walls, are ignored. The parameter is needed for the
            interpolation for the correcting.
        min_distance_wall (float): in meters, has to be >0. The minimum distance, where the points
            should be moved outside the wall.
        max_distance_wall (float): in meters, has to be >0 and >= min_distance_wall. Points, which
            lay nearer to the wall than max_distance_wall will be also moved away. A value
            >0 ist needed for the linear interpolation, else the calculations do not work.
        back_distance_obst (float): in meters, has to be <0. Equivalent to max_distance_wall, but the
            value the concerns the obstacles.
        min_distance_obst (float): in meters, has to be >0. Equivalent to min_distance_wall, but
            the value concerns the obstacles.
        max_distance_obst (float): in meters, has to be >0 and >= min_distance_obst. Equivalent to
            max_distance_wall, but the value concerns the obstacles.

    Returns:
         pedpy.TrajectoryData, either the corrected version of the trajectory or the
         original trajectory, if the original trajectory was valid.
         A list off all personIDs that were changed.

    """
    # checks at first, if the trajectory is valid.
    traj_valid = is_trajectory_valid(traj_data=trajectory_data, walkable_area=walkable_area)

    _check_distance_parameters(
        back_distance_obst=back_distance_obst,
        back_distance_wall=back_distance_wall,
        max_distance_obst=max_distance_obst,
        max_distance_wall=max_distance_wall,
        min_distance_obst=min_distance_obst,
        min_distance_wall=min_distance_wall,
    )

    if not traj_valid:
        max_distance_all = max(max_distance_wall, max_distance_obst)
        # max_distance_wall and max_distance_obst do not necessarily have the same value.
        # In the next step, it is important to use the larger one to identify all points
        # that may lie within the area surrounding the geometry, where adjustments should
        # also be applied.
        try:
            walk_area_with_max_distance_all = shapely.buffer(walkable_area._polygon, -max_distance_all)
            invalid_traj_lines = get_invalid_trajectory(
                traj_data=trajectory_data, walkable_area=WalkableArea(walk_area_with_max_distance_all)
            )

            all_points_for_correcting = list(invalid_traj_lines.index)
            # Those points very likely have to be corrected, but not necessarily
        except GeometryError as p:
            raise InputError("max_distance parameter is not valid") from p

        normalized_walk_area_polygon = shapely.normalize(walkable_area._polygon)
        invalid_person_ids = invalid_traj_lines["id"][invalid_traj_lines["id"].duplicated()].unique()

        new_trajectories_df = _project_all_points_into_walkable_area(
            traj_data=trajectory_data,
            all_points_for_correcting=all_points_for_correcting,
            back_distance_wall=back_distance_wall,
            min_distance_wall=min_distance_wall,
            max_distance_wall=max_distance_wall,
            back_distance_obst=back_distance_obst,
            min_distance_obst=min_distance_obst,
            max_distance_obst=max_distance_obst,
            walkable_area=normalized_walk_area_polygon,
        )

        corrected_trajectory_data = TrajectoryData(new_trajectories_df, frame_rate=trajectory_data.frame_rate)
        traj_valid = is_trajectory_valid(traj_data=corrected_trajectory_data, walkable_area=walkable_area)
        if not traj_valid:
            warnings.warn(
                "Trajectory is still not valid. "
                "Please check whether max-/min-/back-distance "
                "parameters might be too large",
                stacklevel=2,
            )
        return corrected_trajectory_data, invalid_person_ids
    return trajectory_data, []


def _check_distance_parameters(
    back_distance_obst: float,
    back_distance_wall: float,
    max_distance_obst: float,
    max_distance_wall: float,
    min_distance_obst: float,
    min_distance_wall: float,
) -> None:
    """Check whether the distance parameters are valid.

    Raises:
        InputError: If any parameter is invalid.
    """
    if back_distance_wall >= 0:
        raise InputError(f"back_distance_wall has to be <0, is currently {back_distance_wall}")
    if back_distance_obst >= 0:
        raise InputError(f"back_distance_obst has to be <0, is currently {back_distance_obst}")
    if min_distance_wall <= 0 or min_distance_wall > max_distance_wall:
        raise InputError(
            "min_distance_wall and max_distance_wall have to be > 0 and "
            "max_distance_wall has to be >=  min_distance_wall"
        )
    if min_distance_obst <= 0 or min_distance_obst > max_distance_obst:
        raise InputError(
            "min_distance_obst and max_distance_obst have to be > 0 and "
            "max_distance_obst has to be >=  min_distance_obst"
        )


def _project_all_points_into_walkable_area(
    traj_data: TrajectoryData,
    all_points_for_correcting: list[int],
    back_distance_wall: float,
    min_distance_wall: float,
    max_distance_wall: float,
    back_distance_obst: float,
    min_distance_obst: float,
    max_distance_obst: float,
    walkable_area: shapely.Polygon,
) -> pd.DataFrame:
    """Extracts the x and y values of all invalid points and inserts the corrected points back into the traj data.

    This function goes through the trajectory and initializes calculations
    for correcting for every point, which does not lie within in the walkable
    area + max_distance.

    Args:
        traj_data (pedpy.TrajectoryData): The trajectory data to be tested and
            corrected
        all_points_for_correcting(list): A list, with every index of the trajData.data,
            where a point is, which may have to be corrected.
        back_distance_wall (float): in meters, has to be <0. The distance behind the wall, till
            which the points inside  the walls should be corrected. Points, which are
            further inside the walls, are ignored. The parameter is needed for the
            interpolation for the correcting.
        min_distance_wall (float): in meters, has to be >0. The minimum distance, where the points
            should be moved outside the wall.
        max_distance_wall (float): in meters, has to be >0 and >= min_distance_wall. Points, which
            lay nearer to the wall than max_distance_wall will be also moved away. A value
            >0 ist needed for the linear interpolation, else the calculations do not work.
        back_distance_obst (float): in meters, has to be <0. Equivalent to backDistance_wall, but the
            value the concerns the obstacles.
        min_distance_obst (float): in meters, has to be >0. Equivalent to min_distance_wall, but
            the value concerns the obstacles.
        max_distance_obst (float): in meters, has to be >0 and >= min_distance_obst. Equivalent to
            max_distance_wall, but the value concerns the obstacles.
        walkable_area (WalkableArea):  The belonging walkable area


    Returns:
         pedpy.TrajectoryData, either the corrected version of the trajectory or
         the original trajectory, if the original trajectory was valid
    """
    data_trajectories = copy.deepcopy(traj_data.data)
    prev_x = data_trajectories.loc[all_points_for_correcting, "x"].to_numpy()
    prev_y = data_trajectories.loc[all_points_for_correcting, "y"].to_numpy()
    new_x, new_y = _analyze_alignment_to_wall_types(
        x=prev_x,
        y=prev_y,
        back_distance_wall=back_distance_wall,
        min_distance_wall=min_distance_wall,
        max_distance_wall=max_distance_wall,
        back_distance_obst=back_distance_obst,
        min_distance_obst=min_distance_obst,
        max_distance_obst=max_distance_obst,
        walkable_area=walkable_area,
    )
    data_trajectories.loc[all_points_for_correcting, "x"] = new_x
    data_trajectories.loc[all_points_for_correcting, "y"] = new_y
    return data_trajectories


def _analyze_alignment_to_wall_types(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    back_distance_wall: float,
    min_distance_wall: float,
    max_distance_wall: float,
    back_distance_obst: float,
    min_distance_obst: float,
    max_distance_obst: float,
    walkable_area: shapely.Polygon,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Checks a single point, that is considered to be invalid.

    The function goes through the different sides of the geometry and tests for each
    one, if the point is too close to this wall. If this is the case, the new distance,
    the point should have to this wall, will be calculated, considering the given
    parameters and the distance between the point and the wall.

    Args:
        x (np.ndarray[float]): The x coordinates of the points which should be tested
        y (np.ndarray[float]): The y coordinates of the points which should be tested
        back_distance_wall (float): in meters, has to be <0. The distance behind the wall, till
            which the points inside  the walls should be corrected. Points, which are
            further inside the walls, are ignored. The parameter is needed for the
            interpolation for the correcting.
        min_distance_wall (float): in meters, has to be >0. The minimum distance, where the points
            should be moved outside the wall.
        max_distance_wall (float): in meters, has to be >0 and >= min_distance_wall. Points, which
            lay nearer to the wall than max_distance_wall will be also moved away. A value
            >0 ist needed for the linear interpolation, else the calculations do not work.
        back_distance_obst (float): in meters, has to be <0. Equivalent to backDistance_wall, but the
            value the concerns the obstacles.
        min_distance_obst (float): in meters, has to be >0. Equivalent to min_distance_wall, but
            the value concerns the obstacles.
        max_distance_obst (float): in meters, has to be >0 and >= min_distance_obst. Equivalent to
            max_distance_wall, but the value concerns the obstacles.
        walkable_area (WalkableArea):  The belonging walkable area

    Returns:
        tuple[np.ndarray[float], np.ndarray[float]]: The corrected x, y values

    """
    coords = [*walkable_area.interiors]
    points = shapely.points(x, y)
    for obstacle in coords:
        points = _adjust_around_corners(
            obstacle=shapely.Polygon(obstacle),
            back_distance_obst=back_distance_obst,
            max_distance_obst=max_distance_obst,
            min_distance_obst=min_distance_obst,
            points=points,
        )
        points = _push_out_of_obstacles(
            obstacle=shapely.Polygon(obstacle),
            back_distance_obst=back_distance_obst,
            max_distance_obst=max_distance_obst,
            min_distance_obst=min_distance_obst,
            points=points,
        )
    if (
        len(
            points[~shapely.within(points, shapely.buffer(shapely.Polygon(walkable_area.exterior), -max_distance_wall))]
        )
        > 0
    ):
        points = _push_out_of_wall(
            walkable_area_wall=shapely.Polygon(walkable_area.exterior),
            back_distance_wall=back_distance_wall,
            max_distance_wall=max_distance_wall,
            min_distance_wall=min_distance_wall,
            points=points,
        )

    x, y = _point_valid_test(
        points,
        walkable_area,
        back_distance_wall,
        min_distance_wall,
        max_distance_wall,
        back_distance_obst,
        min_distance_obst,
        max_distance_obst,
    )
    return x, y


def _push_out_of_wall(
    walkable_area_wall: shapely.Polygon,
    back_distance_wall: float,
    max_distance_wall: float,
    min_distance_wall: float,
    points: np.ndarray[shapely.Point],
) -> np.ndarray[shapely.Point]:
    """Checks for every edge of the outer wall if the point is inside of it or too close.

    Args:
        walkable_area_wall (shapely.Polygon): The surrounding wall of the walkable area as
            shapely.Polygon
        back_distance_wall (float): the distance behind the wall, ho far the points inside
            the walls should be corrected. Points, which are
            further inside the walls, are ignored.
        min_distance_wall (float): The minimum distance, where the points
            should be moved outside the wall.
        max_distance_wall (float): The maximum distance, how far points can be interpolated outside.
            Points, which lay nearer to the wall than end_distance_wall will be also moved away.
        points (np.ndarray[shapely.Point]): The points that probably need to be corrected.

    Returns:
        The moved x, y coordinates.
    """
    x_vertices = walkable_area_wall.exterior.coords.xy[0]
    y_vertices = walkable_area_wall.exterior.coords.xy[1]
    for i in range(len(x_vertices) - 1):
        edge = [x_vertices[i], y_vertices[i], x_vertices[i + 1], y_vertices[i + 1]]
        p1x, p1y, p2x, p2y = edge
        wall_edge = shapely.geometry.LineString([(p1x, p1y), (p2x, p2y)])
        distance = _distance_from_wall(points, walkable_area_wall, wall_edge, WallType.walls)
        points = _calculate_movement_wall(
            wall_edge, distance, back_distance_wall, max_distance_wall, min_distance_wall, points
        )
    return points


def _push_out_of_obstacles(
    obstacle: shapely.Polygon,
    back_distance_obst: float,
    max_distance_obst: float,
    min_distance_obst: float,
    points: np.ndarray[shapely.Point],
) -> np.ndarray[shapely.Point]:
    """Checks for every edge of the geometry if the points are inside or too close.

    Args:
        obstacle (shapely.Polygon): The polygon of the obstacle
        back_distance_obst (float): the distance behind the wall, ho far the points inside
            the walls should be corrected. points, which are further inside the walls, are ignored.
        max_distance_obst (float): The maximum distance, how far points can be interpolated outside.
            Points, which lay nearer to the wall than end_distance_wall will be also moved away.
        min_distance_obst (float): The minimum distance, where the points
            should be moved outside the wall.
                    points (np.ndarray[shapely.Point]): The points that probably need to be corrected.
        points (np.ndarray[shapely.Point]): The points that probably need to be tested and moved.

    Returns:
        The moved points.
    """
    # moving points away from inside the walls
    x_vertices = obstacle.exterior.coords.xy[0]
    y_vertices = obstacle.exterior.coords.xy[1]
    for i in range(len(x_vertices) - 1):
        edge = [x_vertices[i], y_vertices[i], x_vertices[i + 1], y_vertices[i + 1]]
        p1x, p1y, p2x, p2y = edge
        wall_line = shapely.geometry.LineString([(p1x, p1y), (p2x, p2y)])
        if _wall_facing_zero_point(line=wall_line, walltype=WallType.obstacle):
            distance = _distance_from_wall(points, obstacle, wall_line, WallType.obstacle)
            points = _calculate_movement_obstacle(
                wall_line, distance, back_distance_obst, max_distance_obst, min_distance_obst, points
            )
    return points


def _adjust_around_corners(
    obstacle: shapely.Polygon,
    back_distance_obst: float,
    max_distance_obst: float,
    min_distance_obst: float,
    points: np.ndarray[shapely.Point],
) -> np.ndarray[shapely.Point]:
    """Checks for every edge of the geometry if the point is inside or too close.

    Args:
        obstacle (shapely.Polygon): The polygon of the obstacle
        back_distance_obst (float): the distance behind the wall, ho far the points inside
            the walls should be corrected. Points, which are further inside the walls, are ignored.
        max_distance_obst (float): The maximum distance, how far points can be interpolated outside.
            Points, which lay nearer to the wall than end_distance_wall will be also moved away.
        min_distance_obst (float): The minimum distance, where the points
            should be moved outside the wall.
        points (np.ndarray[shapely.Point]): The points that probably need to be tested and moved.

    Returns:
        The moved points.
    """
    # gently pushing the points to avoid jumps around corners
    x_vertices = obstacle.exterior.coords.xy[0]
    y_vertices = obstacle.exterior.coords.xy[1]
    for i in range(len(x_vertices) - 1):
        edge = [x_vertices[i], y_vertices[i], x_vertices[i + 1], y_vertices[i + 1]]
        p1x, p1y, p2x, p2y = edge
        wall_line = shapely.geometry.LineString([(p1x, p1y), (p2x, p2y)])
        if _wall_facing_zero_point(line=wall_line, walltype=WallType.obstacle):
            distance = _distance_from_wall(points, obstacle, wall_line, WallType.obstacle)
            points = _calculate_movement_adjusting(
                wall_line, distance, back_distance_obst, max_distance_obst, min_distance_obst, points
            )
    return points


def _calculate_movement_wall(
    wall_line: shapely.LineString,
    distances: npt.NDArray[np.float64],
    back_distance_wall: float,
    max_distance_wall: float,
    min_distance_wall: float,
    points: np.ndarray[shapely.Point],
    failsafe: bool = False,
) -> np.ndarray[shapely.Point]:
    """Calculates the new distance, which the point should have after moving.

    Args:
        wall_line: the side of the obstacle from which the points need to will be viewed
            and corrected.
        distances: the distance between the edge and the point.
        back_distance_wall (float): the distance behind the wall, how far the points inside
            the walls should be corrected. Points, which are
            further inside the walls, are ignored.
        min_distance_wall (float): The minimum distance, where the points
            should be moved outside the wall.
        max_distance_wall (float): The maximum distance, how far points can be interpolated outside.
            Points, which lay nearer to the wall than max_distance_wall will be also moved away.
        points (np.ndarray[shapely.Point]): The points that probably need to be tested and moved.
        failsafe (bool): The parameter initiates a less accurate but more reliable way for
            calculating invalid points in case the original way fails.

    Returns:
        The moved points.
    """
    if not failsafe:
        mask_invalid_distances = (
            (distances > back_distance_wall) & (distances <= max_distance_wall) & _close_from_wall(points, wall_line)
        )
    else:
        mask_invalid_distances = (distances > back_distance_wall) & (distances <= max_distance_wall)
    invalid_distances = distances[mask_invalid_distances]
    if len(invalid_distances) > 0:
        points_to_move = points[mask_invalid_distances]
        new_distances = (invalid_distances - back_distance_wall) * (
            (max_distance_wall - min_distance_wall) / (max_distance_wall - back_distance_wall)
        ) + min_distance_wall
        moved = np.array(
            [
                _move_from_wall(point, wall_line, n, dist)
                for point, n, dist in zip(points_to_move, new_distances, invalid_distances, strict=True)
            ]
        )
        points[mask_invalid_distances] = moved
    return points


def _calculate_movement_obstacle(
    wall_line: shapely.LineString,
    distances: npt.NDArray[np.float64],
    back_distance_obst: float,
    max_distance_obst: float,
    min_distance_obst: float,
    points: np.ndarray[shapely.Point],
    failsafe: bool = False,
) -> np.ndarray[shapely.Point]:
    """Calculates the new distance, which the point should have after moving.

    Args:
        wall_line: the side of the obstacle from which the points need to will be viewed
            and corrected.
        distances: the distances between the edge and the point.
        back_distance_obst (float): the distance behind the wall, ho far the points inside
            the walls should be corrected. Points, which are
            further inside the walls, are ignored.
        max_distance_obst (float): The maximum distance, how far Points can be interpolated outside.
            Points, which lay nearer to the wall than max_distance_obst will be also moved away.
        min_distance_obst (float): The minimum distance, where the points
            should be moved outside the wall.
        points (np.ndarray[shapely.Point]): The points that probably need to be tested and moved.
        failsafe (bool): The parameter initiates a less accurate but more reliable way for
            calculating invalid points in case the original way fails.

    Returns:
        The moved points.
    """
    if not failsafe:
        mask_invalid_distances = (
            (distances > back_distance_obst) & (distances <= max_distance_obst) & _close_from_wall(points, wall_line)
        )
    else:
        mask_invalid_distances = (
            (distances > back_distance_obst)
            & (distances <= max_distance_obst)
            & _close_from_wall_triangle(points, wall_line)
        )
    invalid_distances = distances[mask_invalid_distances]
    if len(invalid_distances) > 0:
        points_to_move = points[mask_invalid_distances]
        new_distances = (invalid_distances - back_distance_obst) * (
            (max_distance_obst - min_distance_obst) / (max_distance_obst - back_distance_obst)
        ) + min_distance_obst
        if failsafe:
            new_distances = [min_distance_obst for _ in range(len(points_to_move))]
        moved = np.array(
            [
                _move_from_wall(point, wall_line, n, dist)
                for point, n, dist in zip(points_to_move, new_distances, invalid_distances, strict=True)
            ]
        )
        points[mask_invalid_distances] = moved
    return points


def _calculate_movement_adjusting(
    wall_line: shapely.LineString,
    distances: npt.NDArray[np.float64],
    back_distance_obst: float,
    max_distance_obst: float,
    min_distance_obst: float,
    points: np.ndarray[shapely.Point],
) -> np.ndarray[shapely.Point]:
    """Calculates the adjustment around corners for a smoother interpolation.

    Args:
        wall_line: the side of the obstacle from which the points need to will be viewed
            and corrected.
        distances: the distance between the edge and the point.
        back_distance_obst (float): the distance behind the wall, ho far the points inside
            the walls should be corrected. Points, which are
            further inside the walls, are ignored.
        max_distance_obst (float): The maximum distance, how far Points can be interpolated outside.
            Points, which lay nearer to the wall than max_distance_obst will be also moved away.
        min_distance_obst (float): The minimum distance, where the points
            should be moved outside the wall.
        points (np.ndarray[shapely.Point]): The points that probably need to be tested and moved.

    Returns:
        The moved points.
    """
    mask_invalid_distances = (
        (distances >= back_distance_obst)
        & (distances <= min_distance_obst)
        & _close_from_wall(points, wall_line, 4.0 * max_distance_obst)
    )
    invalid_distances = distances[mask_invalid_distances]
    if len(invalid_distances) > 0:
        points_to_move = points[mask_invalid_distances]
        new_distances = ((max_distance_obst - back_distance_obst) / (min_distance_obst - back_distance_obst)) * (
            invalid_distances - back_distance_obst
        ) + back_distance_obst

        moved = np.array(
            [
                _move_from_wall(point, wall_line, n, dist)
                for point, n, dist in zip(points_to_move, new_distances, invalid_distances, strict=True)
            ]
        )
        points[mask_invalid_distances] = moved
    return points


def _move_from_wall(
    point: shapely.Point, wall_line: shapely.LineString, new_distance: float, distance: float
) -> shapely.Point:
    """Moves the point away from the wall.

    Args:
        point: The point to be moved.
        wall_line: The belonging edge of the wall.
        new_distance: The new distance between the wall and the point.
        distance: The current distance between the wall and the point.

    Returns: The corrected point.

    """
    dist = wall_line.project(point)
    base = wall_line.interpolate(dist)  # Lotfusspunkt
    (p1x, p1y), (p2x, p2y) = wall_line.coords
    vertice1 = shapely.Point([p1x, p1y])
    vertice2 = shapely.Point([p2x, p2y])
    critical_dist = 0.05
    if shapely.distance(vertice1, base) < critical_dist:
        r = np.array([vertice2.x - base.x, vertice2.y - base.y])
        norm_r = np.linalg.norm(r)
        r = r / norm_r * 0.01
        base = shapely.Point([base.x + r[0], base.y + r[1]])
    if shapely.distance(vertice2, base) < critical_dist:
        r = np.array([vertice1.x - base.x, vertice1.y - base.y])
        norm_r = np.linalg.norm(r)
        r = r / norm_r * 0.01
        base = shapely.Point([base.x + r[0], base.y + r[1]])
    v = np.array([base.x - point.x, base.y - point.y])
    norm_v = np.linalg.norm(v)
    e = 10**-5
    if norm_v < e:
        # if the point does not lie exactly on the
        # wall, calculating with the norm of
        # [x - intersect_point_x, y - intersect_point_y] would cause
        # errors (devision by zero).
        v = np.array([[p1x - p2x], [p1y - p2y]])
        v = np.array([-v[1], v[0]])
        norm_v = np.linalg.norm(v)

    v = v / norm_v * new_distance
    sign = -np.sign(distance) if distance != 0 else 1
    x = base.x + v[0] * sign
    y = base.y + v[1] * sign
    return shapely.Point(x, y)


def _point_valid_test(
    points: np.ndarray[shapely.Point],
    walkable_area: shapely.Polygon,
    back_distance_wall: float,
    min_distance_wall: float,
    max_distance_wall: float,
    back_distance_obst: float,
    min_distance_obst: float,
    max_distance_obst: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Test, whether the interpolation has worked.

    Due to floating point precision errors it can occur that some points do not get moved
    out of the geometry even though they are invalid.
    The function checks, whether this is the case and moves the points, that are still
    invalid, outside the geometry initiating different, less accurate but more reliable
    calculations.
    """
    coords = [*walkable_area.interiors]
    for obstacle in coords:
        points_to_correct = points[shapely.within(points, shapely.Polygon(obstacle))]
        if len(points_to_correct) > 0:
            x_vertices = obstacle.coords.xy[0]
            y_vertices = obstacle.coords.xy[1]
            for i in range(len(x_vertices) - 1):
                edge = [x_vertices[i], y_vertices[i], x_vertices[i + 1], y_vertices[i + 1]]
                p1x, p1y, p2x, p2y = edge
                wall_line = shapely.geometry.LineString([(p1x, p1y), (p2x, p2y)])
                distance = _distance_from_wall(
                    points_to_correct, shapely.Polygon(obstacle), wall_line, WallType.obstacle
                )
                points_to_correct = _calculate_movement_obstacle(
                    wall_line,
                    distance,
                    back_distance_obst,
                    max_distance_obst,
                    min_distance_obst,
                    points_to_correct,
                    failsafe=True,
                )
            points[shapely.within(points, shapely.Polygon(obstacle))] = points_to_correct

    walls = walkable_area.exterior
    if (
        len(
            points[~shapely.within(points, shapely.buffer(shapely.Polygon(walkable_area.exterior), -max_distance_wall))]
        )
        > 0
    ):
        x_vertices = walls.coords.xy[0]
        y_vertices = walls.coords.xy[1]
        for i in range(len(x_vertices) - 1):
            edge = [x_vertices[i], y_vertices[i], x_vertices[i + 1], y_vertices[i + 1]]
            p1x, p1y, p2x, p2y = edge
            wall_edge = shapely.geometry.LineString([(p1x, p1y), (p2x, p2y)])
            distance = _distance_from_wall(points, shapely.Polygon(walls), wall_edge, WallType.walls)
            points = _calculate_movement_wall(
                wall_edge, distance, back_distance_wall, max_distance_wall, min_distance_wall, points, failsafe=True
            )
    return shapely.get_x(points), shapely.get_y(points)


class WallType(Enum):
    """Enum, which classifies a type of wall (obstacle or wall)."""

    obstacle = "obstacle"
    walls = "walls"


def _distance_from_wall(
    points: np.ndarray[shapely.Point], walk_area: shapely.Polygon, wall: shapely.LineString, walltype: WallType
) -> npt.NDArray[np.float64]:
    if len(points) == 1:
        distances = shapely.distance(points, wall)
        if shapely.within(points, walk_area):
            return -distances
        return distances
    distances = shapely.distance(points, wall)
    invalid = shapely.within(points, walk_area)
    if walltype == WallType.obstacle:
        distances[invalid] *= -1
    else:
        distances[~invalid] *= -1
    return distances


def _close_from_wall(points: np.ndarray[shapely.Point], line: shapely.LineString, bias: float = 0) -> np.ndarray:
    p = shapely.get_coordinates(points)
    x = p[:, 0]
    y = p[:, 1]
    (p1x, p1y), (p2x, p2y) = line.coords

    a = (p1x - p2x) ** 2 + (p1y - p2y) ** 2
    b = (p1x - x) ** 2 + (p1y - y) ** 2
    c = (x - p2x) ** 2 + (y - p2y) ** 2

    return a + bias > b + c


def _close_from_wall_triangle(points: np.ndarray[shapely.Point], line: shapely.LineString) -> bool:
    (p1x, p1y), (p2x, p2y) = line.coords
    p = shapely.get_coordinates(points)

    x = p[:, 0]
    y = p[:, 1]
    a = abs(p2x - p1x) + abs(p2y - p1y)

    return (a >= np.abs(x - p1x) + np.abs(y - p1y)) & (a >= np.abs(x - p2x) + np.abs(y - p2y))


# return true if the outside of the wall is in sight of the zero point
def _wall_facing_zero_point(line: shapely.LineString, walltype: WallType) -> int:
    if walltype == WallType.obstacle:
        direction = -1
    else:
        direction = 1
    (p1x, p1y), (p2x, p2y) = line.coords
    h = _zero_point_alignment(p1x, p1y, p2x, p2y, direction)
    if h >= 0:
        return 1
    else:
        return 0


def _zero_point_alignment(p1x: float, p1y: float, p2x: float, p2y: float, direction: int) -> float:
    vect_a = [p2x - p1x, p2y - p1y]
    vect_b = [0 - p1x, 0 - p1y]
    sin_p = np.linalg.det([vect_a, vect_b])
    sin_p = sin_p * direction
    a = (p2x - p1x) ** 2 + (p2y - p1y) ** 2
    b = (0 - p1x) ** 2 + (0 - p1y) ** 2
    c = (p2x - 0) ** 2 + (p2y - 0) ** 2
    if a == 0:
        return math.sqrt(math.sqrt(c))
    x_sq = ((a + b - c) ** 2) / (4 * a)
    return np.sign(sin_p) * math.sqrt(math.fabs(b - x_sq))
