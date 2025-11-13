
from pedpy.data.trajectory import TrajectoryData
from pedpy.data.geometry import WalkableArea
from pedpy.methods.method_utils import get_trajectory_data, is_trajectory_data_valid

import numpy as np
import numpy.linalg as npl
import math
import pandas as pd
from pandas import DataFrame
import shapely


def correct_invalid_trajectories(

        trajectory_Data: TrajectoryData,
        walkable_area: WalkableArea,
        back_distance_wall: float = -1 ,
        start_distance_wall: float = 0.01,
        end_distance_wall:float = 0.03,
        back_distance_obst: float = -1 ,
        start_distance_obst: float = 0.01,
        end_distance_obst:float = 0.03
) -> TrajectoryData:

    """
    When dealing with head trajectories, it may happen that the participants lean
    over the obstacles. This means that their trajectory will leave the walkable area
    at some frames, these data than can not be processed with PedPy.

    The function locates false points and points, are too close to a wall but
    still outside of it, and corrects them or pushes them away.

    At the beginning it checks if the trajectory is valid, so that the whole process
    only runs, if the trajectory is not valid.

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
    x is the old distance to the wall, which is either from the inside or not far enough away \n
    a is equal to back_distance \n
    b is equal to start_distance  \n
    c is equal to end_distance  \n


    Args:
        trajectory_Data (pedpy.TrajectoryData): The trajectory data to be tested and corrected
        walkable_area (pedpy.WalkableArea): The belonging walkable area
        back_distance_wall (float): Has to be <0. The distance behind the wall, till which the
            points inside  the walls should be corrected. Points, which are further
            inside the walls, are ignored. The parameter is needed for the interpolation for the correcting.
        start_distance_wall (float): Has to be >0. The minimum distance, where the points should be moved
            outside the wall
        end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points, which lay nearer
            to the wall than endDistance_wall will be also moved away. A value >0 ist needed for the
            linear interpolation, else the calculations do not work.
        back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall, but value concerns
            the obstacles
        start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall, but value
            concerns the obstacles
        end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent to
            endDistance_wall, but value concerns the obstacles

    Returns:
         pedpy.TrajectoryData, either the corrected version of the trajectory or the
         original trajectory, if the original trajectory was valid.
    """
    traj_valid = is_trajectory_valid(traj_data=trajectory_Data,
                                     walkable_area=walkable_area)  # checks at first, if the rajectory is valid.
    print("trajectory valid at beginning: ", traj_valid)
    end_distance_all = max(end_distance_wall, end_distance_obst)
    #end_distance_wall and end_distance_obst do not necessarily have the same value. In the next step, it is important to use the larger one
    #to identify all points that may lie within the area surrounding the geometry, where corrections should also be applied.
    walkArea_with_end_distance_all = shapely.buffer(walkable_area._polygon, -end_distance_all)
    all_points_for_correcting = get_invalid_trajectory(traj_data=trajectory_Data,
                                                       walkable_area=WalkableArea(walkArea_with_end_distance_all)).index
    print("Number of points that are going to be corrected:  ",
          len(all_points_for_correcting))  # to get an impression of the number of points, that need to be corrected.
    if not traj_valid:
        geoData = convert_walk_Area_into_geoData(walkable_area)
        # walkable area is converted into a list of lists like [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
        new_trajectories_df = handle_all_point(trajectory_Data, geoData, all_points_for_correcting,
                                               back_distance_wall, start_distance_wall, end_distance_wall,
                                               back_distance_obst, start_distance_obst, end_distance_obst)
        trajectory_Data = TrajectoryData(new_trajectories_df, frame_rate=trajectory_Data.frame_rate)
        traj_valid = is_trajectory_valid(traj_data=trajectory_Data, walkable_area=walkable_area)
        print("trajectory valid after : ", traj_valid)
        if not traj_valid:
            invalid = get_invalid_trajectory(traj_data=trajectory_Data, walkable_area=walkable_area)
            print("Following ", len(invalid.index),
                  " points could not be interpolated, they will be moved out by forcing the new distance = startDistance: ")
            new_trajectories_df = handle_all_point(trajectory_Data, geoData, invalid.index,
                                                back_distance_wall= back_distance_wall,
                                                start_distance_wall= start_distance_wall,
                                                end_distance_wall=end_distance_wall,
                                                back_distance_obst=back_distance_obst,
                                                start_distance_obst=start_distance_obst,
                                                end_distance_obst=end_distance_obst,
                                                forcing_point_outside=True)
            trajectory_Data = TrajectoryData(new_trajectories_df, frame_rate=trajectory_Data.frame_rate)
            traj_valid = is_trajectory_valid(traj_data=trajectory_Data, walkable_area=walkable_area)
            print("trajectory valid after : ", traj_valid)
    return trajectory_Data



def handle_all_point(
        trajData: TrajectoryData,
        geoData: list,
        all_points_for_correcting: list,
        back_distance_wall: float ,
        start_distance_wall: float,
        end_distance_wall: float,
        back_distance_obst: float ,
        start_distance_obst:float,
        end_distance_obst: float,
        forcing_point_outside: bool = False,
) ->pd.DataFrame:
    """
        This function goes through the trajectory and initializes calculations
        for correcting for every point, which does not lie within in the walkable
        area + end_distance.

        Args:
            trajData (pedpy.TrajectoryData): The trajectory data to be tested and corrected
            geoData (list):  The walkableArea modelled like list of
                [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
            all_points_for_correcting(list): A list, with every index of the trajData.data, where a point is,
                which may have to be corrected.
            back_distance_wall (float): Has to be <0. The distance behind the wall, till which the
                points inside  the walls should be corrected. Points, which are further
                inside the walls, are ignored. The parameter is needed for the interpolation for the correcting.
            start_distance_wall (float): Has to be >0. The minimum distance, where the points should be moved
                outside the wall
            end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points, which lay nearer
                to the wall than endDistance_wall will be also moved away. A value >0 ist needed for the
                linear interpolation, else the calculations do not work.
            back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall, but value concerns
                the obstacles
            start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall, but value
                concerns the obstacles
            end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent to
                endDistance_wall, but value concerns the obstacles
            forcing_point_outside (bool): Bool, if the new corrected interpolated points should
                be forced to be outside the walls/obstacles. It may happen that the interpolation does not
                calculate the movement right. In this case the new distance is equal to startDistance.


        Returns:
             pedpy.TrajectoryData, either the corrected version of the trajectory or the original trajectory, if the original trajectory was valid
    """

    #reminder: geoData is modelled like list of [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
    data_trajectories = trajData.data

    for i in range(
            len(data_trajectories)):  # The loop goes through the whole len of the trajectory, but only, if the index is part of the all_points_for_correcting, it tests further calculations.
        if i in all_points_for_correcting:
            line = data_trajectories.iloc[i]
            new_x, new_y = handle_single_point(line.iloc[2], line.iloc[3], geoData,
                                               back_distance_wall, start_distance_wall, end_distance_wall,
                                               back_distance_obst, start_distance_obst, end_distance_obst,
                                               forcing_point_outside)
            if forcing_point_outside: print("Point, that could not be interpolated: \nx= ", prev_x,"\ny= ", prev_y,
                                            "\nNew calculated, corrected point: \nx =  ", new_x, "\ny =  ", new_y)
            data_trajectories.loc[i, ["x", "y"]] = [float(new_x), float(new_y)]

    return data_trajectories



def handle_single_point(
        x: float,
        y: float,
        geoData:list,
        back_distance_wall: float,
        start_distance_wall: float,
        end_distance_wall: float,
        back_distance_obst: float,
        start_distance_obst: float,
        end_distance_obst: float,
        forcing_point_outside: bool,
)-> tuple[float,float]:
    """
        The function goes through the different sides of the geometry and tests for each
        one, if the point is too close to this wall. If this is the case, the new distance,
        the point should have to this wall, will be calculated, considering the given
        parameters and the distance between the point and the wall.

        Args:
            x (float): The x coordinate of the point which should be tested
            y (float): The y coordinate of the point which should be tested
            geoData (list):  The walkableArea modelled like list of
                [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
            back_distance_wall (float): Has to be <0. The distance behind the wall, till which the
                points inside  the walls should be corrected. Points, which are further
                inside the walls, are ignored. The parameter is needed for the interpolation for the correcting.
            start_distance_wall (float): Has to be >0. The minimum distance, where the points should be moved
                outside the wall
            end_distance_wall (float): Has to be >0 and >= start_distance_wall. Points, which lay nearer
                to the wall than endDistance_wall will be also moved away. A value >0 ist needed for the
                linear interpolation, else the calculations do not work.
            back_distance_obst (float): Has to be <0. Equivalent to backDistance_wall, but value concerns
                the obstacles
            start_distance_obst (float): Has to be >0. Equivalent to startDistance_wall, but value
                concerns the obstacles
            end_distance_obst (float): Has to be >0 and >= start_distance_obst. Equivalent to
                endDistance_wall, but value concerns the obstacles
            forcing_point_outside (bool): Bool, if the new, interpolated points should
                be forced to be outside the walls/obstacles. Depending on the chosen values for
                the parameters, it can happen, that the interpolation calculated the new distance not far
                away enough. In this case the new distance is equal to startDistance

        Returns:
            tuple[float,float]: The corrected x, y values

    """
    for geometry in geoData:
        direction = geometry[1]

        if (geometry[0] == "walls"): #first, testing all the sides, which are walls.
            for edge in geometry[2]:
                if (close_from_wall(x, y, edge[0], edge[1], edge[2], edge[3])):
                    distance = distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], direction)
                    if (back_distance_wall <= distance and distance <= end_distance_wall):
                        # x' = (x-a)*((c-b)/(c-a))+b
                        newDistance = ((distance - back_distance_wall) *
                                       ((end_distance_wall - start_distance_wall) /
                                        (end_distance_wall - back_distance_wall))
                                       +start_distance_wall)

                        x, y = move_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], newDistance, direction)
        else:  # obstacle

            # gently pushing the points to avoid jumps around corners
            for edge in geometry[2]:
                if (close_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], 4 * end_distance_obst)
                        and wall_facing_point(0, 0, edge[0], edge[1], edge[2], edge[3], direction)):

                    distance = distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], direction)
                    if (back_distance_obst <= distance and distance <= start_distance_obst):
                        # x' =((c-a)/(b-a))*(x-a)+a, a slightly different calculation to have a more gentle push concerning corners
                        newDistance = (((end_distance_obst - back_distance_obst)  /
                                       (start_distance_obst - back_distance_obst)) *
                                       (distance - back_distance_obst)
                                       + back_distance_obst)

                        if forcing_point_outside and newDistance <= 0:
                            newDistance = start_distance_obst

                        x, y = move_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], newDistance, direction)


            # moving points away from inside the walls
            for edge in geometry[2]:
                if (close_from_wall_triangle(x, y, edge[0], edge[1], edge[2], edge[3])
                        and wall_facing_point(0, 0, edge[0], edge[1], edge[2], edge[3], direction)):
                    distance = distance_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], direction)
                    if (back_distance_obst <= distance and distance <= end_distance_obst):
                        # x' = (x-a)*((c-b)/(c-a))+b
                        newDistance = ((distance - back_distance_obst)
                                       * ((end_distance_obst - start_distance_obst) /
                                          (end_distance_obst - back_distance_obst))
                                       + start_distance_obst)

                        x, y = move_from_wall(x, y, edge[0], edge[1], edge[2], edge[3], newDistance, direction)
    return x, y


def move_from_wall(
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
    aSq = distance_squared(p1x, p1y, p2x, p2y)
    bSq = distance_squared(p1x, p1y, x, y)
    cSq = distance_squared(x, y, p2x, p2y)
    xSq = ((aSq + bSq - cSq) * (aSq + bSq - cSq)) / (4 * aSq)
    h = distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
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
        # if the Point is exactly on the edge, to move it away, we rotate the point p1 around x by 90 degrees
        x = intersect_point_x - (p2y - intersect_point_y)
        y = intersect_point_y + (p2x - intersect_point_x)
        aSq = distance_squared(p1x, p1y, p2x, p2y)
        bSq = distance_squared(p1x, p1y, x, y)
        cSq = distance_squared(x, y, p2x, p2y)
        xSq = ((aSq + bSq - cSq) * (aSq + bSq - cSq)) / (4 * aSq)
        h = distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
        dist_x = math.sqrt(math.fabs(xSq))
    #ratio = new_distance / h
    #x = intersect_point_x + ratio * (x - intersect_point_x)
    #y = intersect_point_y + ratio * (y - intersect_point_y)
    v = np.array([x - intersect_point_x, y - intersect_point_y])
    v_norm = np.linalg.norm(v)

    if v_norm == 0:  #if the point does not lie exactly on the wall, calculating with the norm of
                    # [x - intersect_point_x, y - intersect_point_y] would cause errors (devision by zero).
        v = np.array([p1x - p2x], [p1y - p2y])
        v = np.array([-v[1], v[0]])
        v_norm = np.linalg.norm(v)

    v = v / v_norm * new_distance
    sign = np.sign(h) if h != 0 else 1
    x = intersect_point_x + v[0] * sign
    y = intersect_point_y + v[1] * sign


    return x, y


def convert_walk_Area_into_geoData(
        walkArea: WalkableArea
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
        walkArea (WalkableArea): The belonging pedpy.WalkableArea for the trajectories:
    Returns:
        geoData (list), the WalkableArea is converted into a list of
        [type="obstacle"|"walls",direction,list of [p1x, p1y, p2x, p2y]]
    """
    obstacle_list = []
    obstacles = list(walkArea._polygon.interiors)
    for hole in obstacles:
        coords = np.array(hole.coords)
        obstacle_list.append(coords)

    wall_coords = walkArea.coords._coords

    geometry_list = []
    i = 0

    for obstacle in obstacle_list:
        vertices = []
        i = 0
        while i < len(obstacle) - 1:
            vertice = [float(obstacle[i][0])
                , float(obstacle[i][1])
                ,float (obstacle[i + 1][0])
                , float(obstacle[i + 1][1])]
            vertices.append(vertice)
            i = i + 1
        geometry_list.append(["obstacle", -1 * compute_direction(vertices), vertices])

    vertices = []
    i = 0
    while i < len(wall_coords) - 1:
        vertice = [float(wall_coords[i][0])
            ,float (wall_coords[i][1])
            , float(wall_coords[i + 1][0])
            , float(wall_coords[i + 1][1])]
        vertices.append(vertice)
        i = i + 1
    geometry_list.append(["walls", compute_direction(vertices), vertices])
    return geometry_list

def compute_direction(
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


def distance_squared(x1, y1, x2, y2):
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

# calculate distance between point and closest point on wall, the distance is signed according to direction
def distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction):
    # detect
    vect_A = [p2x - p1x, p2y - p1y]
    vect_B = [x - p1x, y - p1y]
    sinP = (np.linalg.det([vect_A, vect_B]))
    sinP = sinP * direction
    a = distance_squared(p1x, p1y, p2x, p2y)
    b = distance_squared(p1x, p1y, x, y)
    c = distance_squared(x, y, p2x, p2y)
    if (a == 0):
        return math.sqrt(math.sqrt(c))
    xSq = ((a + b - c) * (a + b - c)) / (4 * a)
    return np.sign(sinP) * math.sqrt(math.fabs(b - xSq))


# return 1 if close from wall (in a circle shape), else 0
def close_from_wall(x, y, p1x, p1y, p2x, p2y, bias=0):
    a = distance_squared(p1x, p1y, p2x, p2y)
    b = distance_squared(p1x, p1y, x, y)
    c = distance_squared(x, y, p2x, p2y)
    if (a + bias > b + c):
        return 1
    else:
        return 0


# return 1 if close from wall (in a triangle shape), else 0
def close_from_wall_triangle(x, y, p1x, p1y, p2x, p2y):
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
def wall_facing_point(x, y, p1x, p1y, p2x, p2y, direction):
    h = distance_from_wall(x, y, p1x, p1y, p2x, p2y, direction)
    if (h >= 0):
        return 1
    else:
        return 0


