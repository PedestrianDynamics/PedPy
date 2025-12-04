from distlib.database import new_dist_class

from pedpy import InputError
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.data.geometry import WalkableArea

import numpy as np
import numpy.linalg as npl
import math
import pandas as pd
import shapely

from pedpy.methods.method_utils import is_trajectory_valid, get_invalid_trajectory


def correct_invalid_trajectories(

        trajectory_data: TrajectoryData,
        walkable_area: WalkableArea,
        back_distance_wall: float = -1 ,
        start_distance_wall: float = 0.01,
        end_distance_wall:float = 0.05,
        back_distance_obst: float = -1 ,
        start_distance_obst: float = 0.01,
        end_distance_obst:float = 0.05
) -> TrajectoryData:

    """
    When dealing with head trajectories, it may happen that the participants lean
    over the obstacles. This means that their trajectory will leave the walkable area
    at some frames, these data than can not be processed with PedPy.

    The function locates false points and points, are too close to a wall but
    still outside of it, and corrects them or pushes them away.

    At the beginning it checks if the trajectory is valid, so that the whole process
    will not run unnecessary.

    It returns a corrected version of the trajectory input (which is also
    a pedpy.TrajectoryData).

    If a point lays inside the geometry or close to it, the  point will be moved away
    outside the geometry. The new distance is calculated by linear interpolation.
    Points that lie further inside an obstacle/wall have a smaller new distance compared
    to a point that lies at the end of the interval. The interval is between
    back_distance and end_distance.

    end_distance describes how far max. points can be moved out. Points that lie between
    the wall and this parameter are also slightly pushed away
    Furthermore, a value >0 is necessary for a mostly accurate linear interpolation for
    all points, that need to be moved.

    The formula for the calculation for the moving points look like:

    x' = (x-a)*((c-b)/(c-a))+b \n


    x' is the new distance to the wall \n
    x is the old distance to the wall,
    which is either from the inside or not far enough away \n
    a is equal to back_distance \n
    b is equal to start_distance  \n
    c is equal to end_distance  \n


    Args:
        trajectory_data (pedpy.TrajectoryData): The trajectory data to be tested and
            corrected
        walkable_area (pedpy.WalkableArea): The belonging walkable area
        back_distance_wall (float): Has to be <0. The distance behind the wall, till
            which the points inside  the walls should be corrected. Points, which are
            further inside the walls, are ignored. The parameter is needed for the
            interpolation for the correcting.
        start_distance_wall (float): Has to be >0. The minimum distance, where the points
            should be moved outside the wall
        end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points, which
            lay nearer to the wall than endDistance_wall will be also moved away. A value
            >0 ist needed for the linear interpolation, else the calculations do not work.
        back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall, but the
            value the concerns the obstacles
        start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall, but
            the value concerns the obstacles
        end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent to
            endDistance_wall, but the value concerns the obstacles

    Returns:
         pedpy.TrajectoryData, either the corrected version of the trajectory or the
         original trajectory, if the original trajectory was valid.
    """
    #checks at first, if the rajectory is valid.
    traj_valid = is_trajectory_valid(traj_data=trajectory_data,
                                     walkable_area=walkable_area)
    print("trajectory valid at beginning: ", traj_valid)

    _check_distance_parameters(back_distance_obst, back_distance_wall, end_distance_obst,
                               end_distance_wall, start_distance_obst, start_distance_wall)

    if not traj_valid:

        end_distance_all = max(end_distance_wall, end_distance_obst)
        # end_distance_wall and end_distance_obst do not necessarily have the same value.
        # In the next step, it is important to use the larger one to identify all points
        # that may lie within the area surrounding the geometry, where adjustments should
        # also be applied.
        walk_area_with_end_distance_all = shapely.buffer(walkable_area._polygon,
                                                        -end_distance_all)
        all_points_for_correcting = list(get_invalid_trajectory(
            traj_data=trajectory_data,
            walkable_area=WalkableArea(walk_area_with_end_distance_all)
        ).index)

        print("Number of points that are going to be corrected:  ",
              len(all_points_for_correcting))
        # to get an impression of the number of points, that need to be corrected.

        geo_data = _convert_walk_area_into_geo_data(walkable_area)
        # walkable area is converted into a list of lists like
        # [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]

        new_trajectories_df = _handle_all_point(
            traj_data= trajectory_data,
            geo_data= geo_data,
            all_points_for_correcting= all_points_for_correcting,
            back_distance_wall= back_distance_wall,
            start_distance_wall= start_distance_wall,
            end_distance_wall= end_distance_wall,
            back_distance_obst= back_distance_obst,
            start_distance_obst= start_distance_obst,
            end_distance_obst= end_distance_obst,
            walkable_area=walkable_area
        )

        trajectory_data = TrajectoryData(new_trajectories_df,
                                         frame_rate=trajectory_data.frame_rate)
        traj_valid = is_trajectory_valid(traj_data=trajectory_data,
                                         walkable_area=walkable_area)
        print("trajectory valid after : ", traj_valid)
    return trajectory_data


def _check_distance_parameters(

        back_distance_obst: float,
        back_distance_wall: float,
        end_distance_obst: float,
        end_distance_wall: float,
        start_distance_obst: float,
        start_distance_wall: float

):
    """
    The function checks, whether the distance parameters are valid and throws and InputError
    if not.

    """

    if back_distance_wall >= 0:
        raise InputError(
            f"back_distance_wall has to be <0, is currently {back_distance_wall}")
    if back_distance_obst >= 0:
        raise InputError(
            f"back_distance_obst has to be <0, is currently {back_distance_obst}")
    if start_distance_wall <= 0 or start_distance_wall > end_distance_wall:
        raise InputError(
            f"start_distance_wall and end_distance_wall have to be > 0 and "
            f"end_distance_wall has to be >=  start_distance_wall")
    if start_distance_obst <= 0 or start_distance_obst > end_distance_obst:
        raise InputError(
            f"start_distance_obst and end_distance_obst have to be > 0 and "
            f"end_distance_obst has to be >=  start_distance_obst")


def _handle_all_point(
        traj_data: TrajectoryData,
        geo_data: list,
        all_points_for_correcting: list,
        back_distance_wall: float ,
        start_distance_wall: float,
        end_distance_wall: float,
        back_distance_obst: float ,
        start_distance_obst:float,
        end_distance_obst: float,
        walkable_area: WalkableArea,
) ->pd.DataFrame:
    """
        This function goes through the trajectory and initializes calculations
        for correcting for every point, which does not lie within in the walkable
        area + end_distance.

        Args:
            traj_data (pedpy.TrajectoryData): The trajectory data to be tested and
                corrected
            geo_data (list):  The walkableArea modelled like list of
                [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
            all_points_for_correcting(list): A list, with every index of the trajData.data,
                where a point is, which may have to be corrected.
            back_distance_wall (float): Has to be <0. The distance behind the wall, till
                which the points inside  the walls should be corrected. Points, which are
                further inside the walls, are ignored. The parameter is needed for the
                interpolation for the correcting.
            start_distance_wall (float): Has to be >0. The minimum distance, where the
                points should be moved outside the wall
            end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points,
                which lay nearer to the wall than endDistance_wall will be also moved away.
                A value >0 ist needed for the linear interpolation, else the calculations
                do not work.
            back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall,
                but the value concerns the obstacles
            start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall,
                but the value concerns the obstacles
            end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent
                to endDistance_wall, but value concerns the obstacles


        Returns:
             pedpy.TrajectoryData, either the corrected version of the trajectory or
             the original trajectory, if the original trajectory was valid
    """

    #geo_data is modelled like list of [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
    data_trajectories = traj_data.data

    for i in range(len(data_trajectories)):
        # The loop goes through the whole len of the trajectory, but only, if the index is
        # part of the all_points_for_correcting, it tests further calculations.
        if i in all_points_for_correcting:
            line = data_trajectories.iloc[i]
            prev_x = line.iloc[2]
            prev_y = line.iloc[3]
            new_x, new_y = _handle_single_point(x= prev_x, y =prev_y, geo_data= geo_data,
                                               back_distance_wall= back_distance_wall,
                                               start_distance_wall= start_distance_wall,
                                               end_distance_wall= end_distance_wall,
                                               back_distance_obst= back_distance_obst,
                                               start_distance_obst= start_distance_obst,
                                               end_distance_obst= end_distance_obst,
                                               walkable_area= walkable_area)
            data_trajectories.loc[i, ["x", "y"]] = [float(new_x), float(new_y)]

    return data_trajectories


def _handle_single_point(
        x: float,
        y: float,
        geo_data:list,
        back_distance_wall: float,
        start_distance_wall: float,
        end_distance_wall: float,
        back_distance_obst: float,
        start_distance_obst: float,
        end_distance_obst: float,
        walkable_area: WalkableArea,
)-> tuple[float,float]:
    """
        The function goes through the different sides of the geometry and tests for each
        one, if the point is too close to this wall. If this is the case, the new distance,
        the point should have to this wall, will be calculated, considering the given
        parameters and the distance between the point and the wall.

        Args:
            x (float): The x coordinate of the point which should be tested
            y (float): The y coordinate of the point which should be tested
            geo_data (list):  The walkableArea modelled like list of
                [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
            back_distance_wall (float): Has to be <0. The distance behind the wall, till
                which the points inside  the walls should be corrected. Points, which are
                further inside the walls, are ignored. The parameter is needed for the
                interpolation for the correcting.
            start_distance_wall (float): Has to be >0. The minimum distance, where the
                points should be moved outside the wall
            end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points,
                which lay nearer to the wall than endDistance_wall will be also moved away.
                A value >0 ist needed for the linear interpolation, else the calculations
                do not work.
            back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall,
                but the value concerns the obstacles
            start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall,
                but the value concerns the obstacles
            end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent
                to endDistance_wall, but the value concerns the obstacles

        Returns:
            tuple[float,float]: The corrected x, y values

    """
    for geometry in geo_data:

        direction = geometry[1]

        if (geometry[0] == "walls"):  # first, testing all the sides, which are walls.
            x, y = _push_out_of_wall(direction, geometry, back_distance_wall,
                                     end_distance_wall, start_distance_wall, x, y)
        else:  # testing obstacles
            x, y = _adjust_around_corners(direction, geometry,
                                          back_distance_obst,
                                          end_distance_obst,
                                          start_distance_obst,
                                          x, y)

            x, y = _push_out_of_obstacles(direction, geometry, back_distance_obst,
                                          end_distance_obst, start_distance_obst,
                                          x, y)

    x, y = point_valid_test(start_distance_wall, end_distance_wall, back_distance_wall,
                            start_distance_obst, end_distance_obst, back_distance_obst,
                            geo_data, walkable_area, x, y)

    return x, y


def _push_out_of_wall(
        direction:float,
        geometry:list,
        back_distance_wall: float,
        end_distance_wall: float,
        start_distance_wall: float,
        x: float,
        y: float
) -> tuple[float, float]:

    """

    Args:
        direction(float): either 1 or -1, according to the direction of the vertices of a geometry
        geometry(list): a list of the type [type="walls",direction,list of [p1x, p1y, p2x, p2y]]
        back_distance_wall(float):
        end_distance_wall(float):
        start_distance_wall(float):
        x(float): The  x coordinate of the point to be tested and moved
        y(float): The  y coordinate of the point to be tested and moved

    Returns:
        The moved x, y coordinates
    """

    for edge in geometry[2]:
        if (_close_from_wall(x, y, edge[0], edge[1], edge[2], edge[3])):
            distance = _distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3],
                                           direction)
            x, y = _calculating_movement_wall(edge, direction, distance,
                                              back_distance_wall,
                                              end_distance_wall,
                                              start_distance_wall, x, y)
    return x, y

def _push_out_of_obstacles(
        direction:float,
        geometry:list,
        back_distance_obst: float,
        end_distance_obst: float,
        start_distance_obst: float,
        x: float,
        y: float,
) -> tuple[float, float]:
    """

    Args:
        direction(float): either 1 or -1, according to the direction of the vertices of a geometry
        geometry(list): a list of the type [type="walls",direction,list of [p1x, p1y, p2x, p2y]]
        back_distance_obst(float):
        end_distance_obst(float):
        start_distance_obst(float):
        x(float): The  x coordinate of the point to be tested and moved
        y(float): The  y coordinate of the point to be tested and moved

    Returns:
        The moved x, y coordinates
    """

    # moving points away from inside the walls
    for edge in geometry[2]:
        if (_close_from_wall_triangle(x, y, edge[0], edge[1], edge[2], edge[3])
                and _wall_facing_point(0, 0, edge[0], edge[1], edge[2], edge[3], direction)):
            distance = _distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], direction)
            x, y = _calculating_movement_obstacle(edge, direction, distance,
                                                  back_distance_obst,
                                                  end_distance_obst,
                                                  start_distance_obst, x, y)
    return x, y




def _adjust_around_corners(
        direction:float,
        geometry:list,
        back_distance_obst: float,
        end_distance_obst: float,
        start_distance_obst: float,
        x: float,
        y: float,
) -> tuple[float, float]:
    """

        Args:
            direction(float): either 1 or -1, according to the direction of the vertices of a geometry
            geometry(list): a list of the type [type="walls",direction,list of [p1x, p1y, p2x, p2y]]
            back_distance_obst(float):
            end_distance_obst(float):
            start_distance_obst(float):
            x(float): The  x coordinate of the point to be tested and adjusted
            y(float): The  y coordinate of the point to be tested and adjusted

        Returns:
            The moved x, y coordinates, if they had to be adjusted. Else the original x,y coordinates
        """

    # gently pushing the points to avoid jumps around corners
    for edge in geometry[2]:
        if (_close_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], 4.0 * end_distance_obst)
                and _wall_facing_point(0, 0, edge[0], edge[1], edge[2], edge[3], direction)):

            distance = _distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], direction)
            x, y = _calculating_movement_adjusting(edge, direction, distance,
                                                   back_distance_obst, end_distance_obst,
                                                   start_distance_obst, x, y)
    return x, y


def _calculating_movement_wall(
        edges: list,
        direction:int,
        distance: float,
        back_distance_wall: float,
        end_distance_wall: float,
        start_distance_wall: float,
        x: float,
        y: float
) -> tuple[float, float]:
    """
    Args:
        edges: The vertices of the edge which is too close
        direction:either 1 or -1, according to the direction of the vertices of a geometry
        distance: the distance between the edge and the point
        back_distance_wall:
        end_distance_wall:
        start_distance_wall:
        x(float): The  x coordinate of the point to be tested and moved
        y(float): The  y coordinate of the point to be tested and moved

    Returns:
         The moved x, y coordinates, if they had to be adjusted. Else the original x,y coordinates
    """
    if (back_distance_wall <= distance and distance <= end_distance_wall):
        new_distance = ((distance - back_distance_wall) * (
                (end_distance_wall - start_distance_wall)
                / (end_distance_wall - back_distance_wall))
                        + start_distance_wall)
        x, y = _move_from_wall(x, y, edges[0], edges[1], edges[2], edges[3],
                               new_distance, direction)
    return x, y


def _calculating_movement_obstacle(
        edges: list,
        direction:int,
        distance: float,
        back_distance_obst: float,
        end_distance_obst: float,
        start_distance_obst: float,
        x: float,
        y: float
) -> tuple[float, float]:
    """
       Args:
           edges: The vertices of the edge which is too close
           direction:either 1 or -1, according to the direction of the vertices of a geometry
           distance: the distance between the edge and the point
           back_distance_obst:
           end_distance_obst:
           start_distance_obst:
           x(float): The  x coordinate of the point to be tested and moved
           y(float): The  y coordinate of the point to be tested and moved

       Returns:
            The moved x, y coordinates, if they had to be adjusted. Else the original x,y coordinates
        """
    if (back_distance_obst <= distance and distance <= end_distance_obst):
        # x' = (x-a)*((c-b)/(c-a))+b
        new_distance = ((distance - back_distance_obst) * (
                (end_distance_obst - start_distance_obst)
                / (end_distance_obst - back_distance_obst))
                        + start_distance_obst)
        x, y = _move_from_wall(x, y, edges[0], edges[1], edges[2],
                               edges[3],
                               new_distance, direction)
    return x, y


def _calculating_movement_adjusting(
        edge:list,
        direction: float,
        distance: float ,
        back_distance_obst: float,
        end_distance_obst: float,
        start_distance_obst: float,
        x: float,
        y: float
) -> tuple[float, float]:
    """
       Args:
           edges: The vertices of the edge which is too close
           direction:either 1 or -1, according to the direction of the vertices of a geometry
           distance: the distance between the edge and the point
           back_distance_obst:
           end_distance_obst:
           start_distance_obst:
           x(float): The  x coordinate of the point to be tested and moved
           y(float): The  y coordinate of the point to be tested and moved

       Returns:
            The moved x, y coordinates, if they had to be adjusted. Else the original x,y coordinates
        """

    if (back_distance_obst <= distance and distance <= start_distance_obst):
        # x' =((c-a)/(b-a))*(x-a)+a, a slightly different calculation to have a
        # more gentle push concerning corners
        new_distance = (((end_distance_obst - back_distance_obst) /
                         (start_distance_obst - back_distance_obst)) *
                        (distance - back_distance_obst)
                        + back_distance_obst)

        x, y = _move_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], new_distance,
                               direction)
    return x, y


def _move_from_wall(
        x: float,
        y: float,
        p1x: float,
        p1y: float,
        p2x: float,
        p2y: float,
        new_distance: float,
        direction: float
)-> tuple[float,float]:
    """
    This function moves the points out of the obstacles/walls by
    using different methods of linear algebra.

    Args:
        x (float): x coordinate:
        y (float): y coordinate:
        p1x (float): x coordinate of the first vertice
        p1y (float): y coordinate of the first vertice
        p2x (float): x coordinate of the second vertice
        p2y (float): y coordinate of the second vertice, the two vertices
            are directly connected in the geometry
        new_distance: the calculated distance that should be between the
            wall and the point, that will be moved direction: the direction,
            in which the vertices are arranged (clockwise or anti-clockwise).
            With this value the function can decide, in which direction
            the point needs to be moved, to be outside the wall/obstacle.

    Returns: The moved x, y coordinates
    """
    # if the points are exactly on the corners of the edges, we move them slightly
    if (x == p1x and y == p1y):
        x = x - 0.001
    if (x == p2x and y == p2y):
        x = x - 0.001
    # the triangle is p1 p2 (x,y) where dist_a is the wall segment length
    # and dist_x  the length between p1 and the projection of (x,y) on the wall
    aSq = _distance_squared(p1x, p1y, p2x, p2y)
    bSq = _distance_squared(p1x, p1y, x, y)
    cSq = _distance_squared(x, y, p2x, p2y)
    xSq = ((aSq + bSq - cSq) * (aSq + bSq - cSq)) / (4 * aSq)
    h = _distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
    # the cosine is needed to tell if the projection is outside the segment
    vect_A = [p2x - p1x, p2y - p1y]
    vect_B = [x - p1x, y - p1y]
    cosP = np.dot(vect_A, vect_B) / (npl.norm(vect_A) * npl.norm(vect_B))
    dist_x = math.sqrt(xSq) * np.sign(cosP)
    dist_a = math.sqrt(aSq)
    #the point, that is projected orthogonal on the wall:
    intersect_point_x = p1x + (dist_x / dist_a) * (p2x - p1x)
    intersect_point_y = p1y + (dist_x / dist_a) * (p2y - p1y)

    if (h == 0):
        # if the Point is exactly on the edge, to move it away,
        # we rotate the point p1 around x by 90 degrees
        x = intersect_point_x - (p2y - intersect_point_y)
        y = intersect_point_y + (p2x - intersect_point_x)
        aSq = _distance_squared(p1x, p1y, p2x, p2y)
        bSq = _distance_squared(p1x, p1y, x, y)
        cSq = _distance_squared(x, y, p2x, p2y)
        xSq = ((aSq + bSq - cSq) * (aSq + bSq - cSq)) / (4 * aSq)
        h = _distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
        dist_x = math.sqrt(math.fabs(xSq))

    v = np.array([x - intersect_point_x, y - intersect_point_y])
    v_norm = np.linalg.norm(v)

    if v_norm == 0:  #if the point does not lie exactly on the
        # wall, calculating with the norm of
        # [x - intersect_point_x, y - intersect_point_y] would cause
        # errors (devision by zero).
        v = np.array([p1x - p2x], [p1y - p2y])
        v = np.array([-v[1], v[0]])
        v_norm = np.linalg.norm(v)

    v = v / v_norm * new_distance
    sign = np.sign(h) if h != 0 else 1
    x = intersect_point_x + v[0] * sign
    y = intersect_point_y + v[1] * sign


    return x, y

def point_valid_test(

        start_distance_wall: float,
        end_distance_wall: float,
        back_distance_wall: float,
        start_distance_obst: float,
        end_distance_obst: float,
        back_distance_obst: float,
        geo_data: list,
        walkable_area: WalkableArea,
        x: float,
        y: float

) -> tuple[ float, float]:

    point_is_valid = shapely.within(shapely.Point(x, y), walkable_area._polygon)
    if not point_is_valid:
        for geometry in geo_data:
            direction = geometry[1]
            for edge in geometry[2]:
                if ((_close_from_wall(x, y, edge[0], edge[1], edge[2], edge[3]))):
                    distance = _distance_from_wall(x, y, edge[0], edge[1], edge[2],
                                                   edge[3], direction)
                    if (geometry[0] == "obstacles"):
                        x, y = _calculating_movement_obstacle(edge, direction, distance,
                                                              back_distance_obst,
                                                              end_distance_obst,
                                                              start_distance_obst, x, y)
                    else: #geometry type = wall
                        x, y = _calculating_movement_wall(edge, direction, distance,
                                                          back_distance_wall,
                                                          end_distance_wall,
                                                          start_distance_wall, x, y)
    return x, y

def _convert_walk_area_into_geo_data(
        walk_area: WalkableArea
) -> list:
    """

    The following functions for correcting incorrect trajectory points work
    with a list instead od the WalkableArea. Every geometry
    inside the WalkableArea is converted into a list like

    [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]].

    Type is the information, whether the geometry is a wall or an obstacle.
    The direction is about, whether the following points of the geometry,
    which is analyzed, are initialized clockwise or counterclockwise. By this the functions
    can distinguish, if a point is inside or outside a geometry.
    [p1x, p1y, p2x, p2y] contain two vertices, which are directly connected to each other.

    Args:
        walk_area (WalkableArea): The belonging pedpy.WalkableArea for the trajectories:
    Returns:
        geoData (list), the WalkableArea is converted into a list of
        [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
    """
    obstacle_list = []
    obstacles = list(walk_area._polygon.interiors)
    for hole in obstacles:
        coords = np.array(hole.coords)
        obstacle_list.append(coords)

    wall_coords = walk_area.coords._coords

    geometry_list = []
    i = 0

    for obstacle in obstacle_list:
        vertices = []
        i = 0
        while i < len(obstacle) - 1:
            vertice = [float(obstacle[i][0])
                , float(obstacle[i][1])
                , float (obstacle[i + 1][0])
                , float(obstacle[i + 1][1])]
            vertices.append(vertice)
            i = i + 1
        geometry_list.append(["obstacle", -1 * _compute_direction(vertices), vertices])

    vertices = []
    i = 0
    while i < len(wall_coords) - 1:
        vertice = [float(wall_coords[i][0])
            , float (wall_coords[i][1])
            , float(wall_coords[i + 1][0])
            , float(wall_coords[i + 1][1])]
        vertices.append(vertice)
        i = i + 1
    geometry_list.append(["walls", _compute_direction(vertices), vertices])
    return geometry_list


def _compute_direction(
        vertices:list
)-> float:
    """
    Args:
        vertices (list): a list of all points of the walls/
        one single obstacle in the right order, like:
        [[p1x, p1y, p2x, p2y],[p2x, p2y, p3x, p3y,] ... ]
    Returns:
        1 (clockwise) or -1 (anticlockwise), according to the direction,
        in which the vertices are ordered. If the vertices do not make a
        loop, the total_angele is returned, and the sign can still be used.
    """


    total_angle = 0
    i = 0
    j = 1
    while i < len(vertices):
        vect_A = [vertices[i][2] - vertices[i][0], vertices[i][3] - vertices[i][1]]
        vect_B = [vertices[j][2] - vertices[j][0], vertices[j][3] - vertices[j][1]]
        cosP = np.dot(vect_A, vect_B) / (npl.norm(vect_A) * npl.norm(vect_B))
        sinP = (np.linalg.det([vect_A, vect_B]))
        angle = (np.sign(sinP) * np.arccos(cosP))
        total_angle += angle
        i = i + 1
        j = (i + 1) % len(vertices)
    total_angle = (total_angle / (2 * math.pi))
    if (total_angle == -1):
        return -1
    elif (total_angle == 1):
        return 1
    else:
        return total_angle


def _distance_squared(x1, y1, x2, y2):
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

# calculate distance between point and closest point on wall, the distance is signed according to direction
def _distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction):
    # detect
    vect_A = [p2x - p1x, p2y - p1y]
    vect_B = [x - p1x, y - p1y]
    sinP = (np.linalg.det([vect_A, vect_B]))
    sinP = sinP * direction
    a = _distance_squared(p1x, p1y, p2x, p2y)
    b = _distance_squared(p1x, p1y, x, y)
    c = _distance_squared(x, y, p2x, p2y)
    if (a == 0):
        return math.sqrt(math.sqrt(c))
    xSq = ((a + b - c) * (a + b - c)) / (4 * a)
    return np.sign(sinP) * math.sqrt(math.fabs(b - xSq))



# return 1 if close from wall (in a circle shape), else 0
def _close_from_wall(x, y, p1x, p1y, p2x, p2y, bias=0):
    a = _distance_squared(p1x, p1y, p2x, p2y)
    b = _distance_squared(p1x, p1y, x, y)
    c = _distance_squared(x, y, p2x, p2y)
    if (a + bias > b + c):
        return 1
    else:
        return 0



# return 1 if close from wall (in a triangle shape), else 0
def _close_from_wall_triangle(x, y, p1x, p1y, p2x, p2y):
    a = [p2x - p1x, p2y - p1y]
    b = [x - p1x, y - p1y]
    c = [x - p2x, y - p2y]
    a = np.abs(a)
    b = np.abs(b)
    c = np.abs(c)
    if (a[0] + a[1] >= b[0] + b[1] and a[0] + a[1] >= c[0] + c[1]):
        return 1
    else:
        return 0


# return true if the outside of the wall is in sight of the point
def _wall_facing_point(x, y, p1x, p1y, p2x, p2y, direction):
    h = _distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
    if (h >= 0):
        return 1
    else:
        return 0


