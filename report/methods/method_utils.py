"""Helper functions for the analysis methods"""

from collections import defaultdict

import numpy as np
import pandas as pd
import pygeos
from scipy.spatial import Voronoi

from report.data.geometry import Geometry


def get_peds_in_area(traj_data: pd.DataFrame, measurement_area: pygeos.Geometry) -> pd.DataFrame:
    """Filters the trajectory date to pedestrians which are inside the given area.

    Args:
        traj_data (pd.DataFrame): trajectory data to filter
        measurement_area (pygeos.Geometry):

    Returns:
         Filtered data set, only containing data of pedestrians inside the measurement_area
    """
    return traj_data[pygeos.contains(measurement_area, traj_data["points"])]


def get_peds_in_frame_range(
    traj_data: pd.DataFrame, min_frame: int = None, max_frame: int = None
) -> pd.DataFrame:
    """Filters the trajectory data by the given min and max frames. If only one of them is given
    (not None) then the other is taken as filter.

    Note:
        min_frame >= max_frame is assumed!

    Args:
        traj_data (pd.DataFrame): trajectory data to filter
        min_frame (int): min frame number still in the filtered data set
        max_frame (int): max frame number still in the filtered data set

    Returns:
        Filtered data set, only containing data within the given frame range

    """
    if min_frame is None and max_frame is not None:
        return traj_data[traj_data["frame"].le(max_frame)]

    if max_frame is None and min_frame is not None:
        return traj_data[traj_data["frame"].ge(min_frame)]

    if min_frame is not None and max_frame is not None:
        return traj_data[traj_data["frame"].between(min_frame, max_frame, inclusive="both")]

    return traj_data


def get_num_peds_per_frame(traj_data: pd.DataFrame) -> pd.DataFrame:
    """Returns the number of pedestrians in each frame as DataFrame

    Args:
        traj_data (pd.DataFrame): trajectory data

    Returns:
        DataFrame containing the columns: 'frame' (as index) and 'num_peds'.

    """
    num_peds_per_frame = traj_data.groupby("frame").size()
    num_peds_per_frame = num_peds_per_frame.rename("num_peds")

    return num_peds_per_frame.to_frame()


def compute_individual_voronoi_polygons(
    traj_data: pd.DataFrame, geometry: Geometry
) -> pd.DataFrame:
    """Compute the individual voronoi cells for each person and frame

    Args:
        traj_data (pd.DataFrame): trajectory data
        geometry (Geometry): bounding area, where pedestrian are supposed to be

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and 'individual voronoi'.
    """
    dfs = []

    bounds = pygeos.bounds(geometry.walkable_area)
    clipping_diameter = 2 * max(abs(bounds[2] - bounds[0]), abs(bounds[3] - bounds[1]))

    for _, peds_in_frame in traj_data.groupby(traj_data.frame):
        points = peds_in_frame.sort_values(by="ID")[["X", "Y"]]
        if len(points) < 4:
            continue
        vor = Voronoi(points.to_numpy())
        vornoi_polygons = _clip_voronoi_polygons(vor, clipping_diameter)
        voronoi_in_frame = peds_in_frame.loc[:, ("ID", "frame")]
        voronoi_in_frame["individual voronoi"] = pygeos.intersection(
            vornoi_polygons, geometry.walkable_area
        )

        dfs.append(voronoi_in_frame)

    return pd.concat(dfs)


def compute_intersecting_polygons(
    individual_voronoi_data: pd.DataFrame, measurement_area: pygeos.Geometry
) -> pd.DataFrame:
    """Compute the intersection of each of the individual voronoi cells with the measurement area

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data, needs to contain a column
                'individual voronoi' which holds pygeos.Polygon information
        measurement_area (pygeos.Geometry):

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and 'intersection voronoi'.
    """
    df_intersection = individual_voronoi_data[["ID", "frame"]].copy()
    df_intersection["intersection voronoi"] = pygeos.intersection(
        individual_voronoi_data["individual voronoi"], measurement_area
    )
    return df_intersection


def _clip_voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
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
            polygons.append(pygeos.polygons(voronoi.vertices[region]))
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
        extra_edge = [voronoi.vertices[j] + dir_j * length, voronoi.vertices[k] + dir_k * length]

        polygons.append(pygeos.polygons(np.concatenate((finite_part, extra_edge))))
    return polygons
