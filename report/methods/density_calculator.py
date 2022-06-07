"""Module containing functions to compute densities"""
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import pygeos
from scipy.spatial import Voronoi

from report.data.geometry import Geometry
from report.methods.method_utils import get_peds_in_area


def compute_classic_density(
    traj_data: pd.DataFrame,
    measurement_area: pygeos.Geometry,
) -> pd.DataFrame:
    """Compute the classic density of the trajectory per frame inside the given measurement area

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (pygeos.Geometry): area for which the density is computed

    Returns:
        DataFrame containing the columns: 'frame' and 'classic density'
    """
    peds_in_area = get_peds_in_area(traj_data, measurement_area)
    peds_in_area_per_frame = _get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / pygeos.area(measurement_area)

    # Rename column and add missing zero values
    density.columns = ["classic density"]
    density = density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density


def compute_voronoi_density(
    traj_data: pd.DataFrame, measurement_area: pygeos.Geometry, geometry: Geometry
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the voronoi density of the trajectory per frame inside the given measurement area

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (pygeos.Geometry): area for which the density is computed
        geometry (Geometry): bounding area, where pedestrian are supposed to be

    Returns:
          DataFrame containing the columns: 'frame' and 'voronoi density',
          DataFrame containing the columns: 'ID', 'frame', 'individual voronoi'
    """
    df_individual = _compute_individual_voronoi_polygons(traj_data, geometry)
    df_intersecting = _compute_intersecting_polygons(df_individual, measurement_area)

    df_combined = pd.merge(df_individual, df_intersecting, on=["ID", "frame"], how="outer")
    df_combined["relation"] = pygeos.area(df_combined["intersection voronoi"]) / pygeos.area(
        df_combined["individual voronoi"]
    )

    df_voronoi_density = (
        df_combined.groupby("frame")["relation"].sum() / pygeos.area(measurement_area)
    ).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density.columns = ["voronoi density"]
    df_voronoi_density = df_voronoi_density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return df_voronoi_density, df_individual


def _get_num_peds_per_frame(traj_data: pd.DataFrame) -> pd.DataFrame:
    """Returns the number of pedestrians in each frame as DataFrame

    Args:
        traj_data (pd.DataFrame): trajectory data

    Returns:
        DataFrame containing the columns: 'frame' (as index) and 'num_peds'.

    """
    num_peds_per_frame = traj_data.groupby("frame").size()
    num_peds_per_frame = num_peds_per_frame.rename("num_peds")

    return num_peds_per_frame.to_frame()


def _compute_individual_voronoi_polygons(
    traj_data: pd.DataFrame, geometry: Geometry, use_blind_points: bool = True
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

    blind_points = np.array([])

    if use_blind_points:
        blind_points = np.array(
            [
                [100 * bounds[0], 100 * bounds[1]],
                [100 * bounds[2], 100 * bounds[1]],
                [100 * bounds[0], 100 * bounds[3]],
                [100 * bounds[2], 100 * bounds[3]],
            ]
        )

    for _, peds_in_frame in traj_data.groupby(traj_data.frame):
        points = peds_in_frame.sort_values(by="ID")[["X", "Y"]].to_numpy()
        if use_blind_points:
            points = np.concatenate([points, blind_points])
        if len(points) < 4:
            continue
        vor = Voronoi(points)
        voronoi_polygons = _clip_voronoi_polygons(vor, clipping_diameter)
        if use_blind_points:
            voronoi_polygons = voronoi_polygons[:-4]
        voronoi_in_frame = peds_in_frame.loc[:, ("ID", "frame")]
        voronoi_in_frame["individual voronoi"] = pygeos.intersection(
            voronoi_polygons, geometry.walkable_area
        )

        dfs.append(voronoi_in_frame)

    return pd.concat(dfs)


def _compute_intersecting_polygons(
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
