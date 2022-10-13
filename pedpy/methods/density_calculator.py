"""Module containing functions to compute densities"""
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import Voronoi
from shapely import Polygon

from pedpy.data.geometry import Geometry


def compute_classic_density(
    *,
    traj_data: pd.DataFrame,
    measurement_area: Polygon,
) -> pd.DataFrame:
    """Compute the classic density of the trajectory per frame inside the given
     measurement area.

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (shapely.Polygon): area for which the density is
            computed

    Returns:
        DataFrame containing the columns: 'frame' and 'classic density'
    """
    peds_in_area = traj_data[
        shapely.contains(measurement_area, traj_data["points"])
    ]
    peds_in_area_per_frame = _get_num_peds_per_frame(peds_in_area)

    density = peds_in_area_per_frame / measurement_area.area

    # Rename column and add missing zero values
    density.columns = ["classic density"]
    density = density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return density


def compute_voronoi_density(
    *,
    traj_data: pd.DataFrame,
    measurement_area: shapely.Polygon,
    geometry: Geometry,
    cut_off: Tuple[float, int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the voronoi density of the trajectory per frame inside the given
    measurement area.

    Args:
        traj_data (pd.DataFrame): trajectory data to analyze
        measurement_area (shapely.Polygon): area for which the density is
            computed
        geometry (Geometry): bounding area, where pedestrian are supposed to be
        cut_off (Tuple[float, int): radius of max extended voronoi cell (in m),
                number of linear segments in the approximation of circular
                arcs, needs to be divisible by 4!

    Returns:
          DataFrame containing the columns: 'frame' and 'voronoi density',
          DataFrame containing the columns: 'ID', 'frame', 'individual voronoi',
                'intersecting voronoi'
    """
    df_individual = compute_individual_voronoi_polygons(
        traj_data=traj_data, geometry=geometry, cut_off=cut_off
    )
    df_intersecting = _compute_intersecting_polygons(
        df_individual, measurement_area
    )

    df_combined = pd.merge(
        df_individual, df_intersecting, on=["ID", "frame"], how="outer"
    )
    df_combined["relation"] = shapely.area(
        df_combined["intersection voronoi"]
    ) / shapely.area(df_combined["individual voronoi"])

    df_voronoi_density = (
        df_combined.groupby("frame")["relation"].sum() / measurement_area.area
    ).to_frame()

    # Rename column and add missing zero values
    df_voronoi_density.columns = ["voronoi density"]
    df_voronoi_density = df_voronoi_density.reindex(
        list(range(traj_data.frame.min(), traj_data.frame.max() + 1)),
        fill_value=0.0,
    )

    return (
        df_voronoi_density,
        df_combined.loc[:, df_combined.columns != "relation"],
    )


def compute_passing_density(
    *, density_per_frame: pd.DataFrame, frames: pd.DataFrame
) -> pd.DataFrame:
    """Compute the individual density of the pedestrian who pass the area.

    Args:
        density_per_frame (pd.DataFrame): density per frame, DataFrame
                containing the columns: 'frame' (as index) and 'density'
        frames (pd.DataFrame): information for each pedestrian in the area,
                need to contain the following columns: 'ID','frame_start',
                'frame_end'

    Returns:
          DataFrame containing the columns: 'ID' and 'density' in 1/m
    """
    density = pd.DataFrame(frames["ID"], columns=["ID", "density"])

    densities = []
    for _, row in frames.iterrows():
        densities.append(
            density_per_frame[
                density_per_frame.index.to_series().between(
                    int(row.frame_start), int(row.frame_end), inclusive="left"
                )
            ].mean()
        )
    density["density"] = np.array(densities)
    return density


def _get_num_peds_per_frame(traj_data: pd.DataFrame) -> pd.DataFrame:
    """Returns the number of pedestrians in each frame as DataFrame

    Args:
        traj_data (pd.DataFrame): trajectory data

    Returns:
        DataFrame containing the columns: 'frame' (as index) and 'num_peds'.

    """
    num_peds_per_frame = traj_data.groupby("frame").agg(
        num_peds=("ID", "count")
    )

    return num_peds_per_frame


def compute_individual_voronoi_polygons(
    *,
    traj_data: pd.DataFrame,
    geometry: Geometry,
    cut_off: Tuple[float, int] = None,
) -> pd.DataFrame:
    """Compute the individual voronoi cells for each person and frame

    Args:
        traj_data (pd.DataFrame): trajectory data
        geometry (Geometry): bounding area, where pedestrian are supposed to be
        cut_off (Tuple[float, int]): radius of max extended voronoi cell (in m),
                number of linear segments in the approximation of circular arcs,
                needs to be divisible by 4!
    Returns:
        DataFrame containing the columns: 'ID', 'frame' and 'individual voronoi'.
    """
    dfs = []

    bounds = geometry.walkable_area.bounds
    clipping_diameter = 2 * max(
        abs(bounds[2] - bounds[0]), abs(bounds[3] - bounds[1])
    )

    for _, peds_in_frame in traj_data.groupby(traj_data.frame):
        points = peds_in_frame.sort_values(by="ID")[["X", "Y"]]
        if len(points) < 4:
            continue
        vor = Voronoi(points.to_numpy())
        vornoi_polygons = _clip_voronoi_polygons(vor, clipping_diameter)
        voronoi_in_frame = peds_in_frame.loc[:, ("ID", "frame", "points")]

        # Compute the intersecting area with the walkable area
        voronoi_in_frame["individual voronoi"] = shapely.intersection(
            vornoi_polygons, geometry.walkable_area
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

        if cut_off is not None:
            num_edges = cut_off[1]
            radius = cut_off[0]
            quad_edges = int(num_edges / 4)
            voronoi_in_frame["individual voronoi"] = shapely.intersection(
                voronoi_in_frame["individual voronoi"],
                shapely.buffer(
                    peds_in_frame["points"], radius, quadsegs=quad_edges
                ),
            )

        dfs.append(voronoi_in_frame)

    return pd.concat(dfs)[["ID", "frame", "individual voronoi"]]


def _compute_intersecting_polygons(
    individual_voronoi_data: pd.DataFrame, measurement_area: Polygon
) -> pd.DataFrame:
    """Compute the intersection of each of the individual voronoi cells with
    the measurement area.

    Args:
        individual_voronoi_data (pd.DataFrame): individual voronoi data, needs
                to contain a column 'individual voronoi' which holds
                shapely.Polygon information
        measurement_area (shapely.Polygon):

    Returns:
        DataFrame containing the columns: 'ID', 'frame' and
        'intersection voronoi'.
    """
    df_intersection = individual_voronoi_data[["ID", "frame"]].copy()
    df_intersection["intersection voronoi"] = shapely.intersection(
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
        extra_edge = [
            voronoi.vertices[j] + dir_j * length,
            voronoi.vertices[k] + dir_k * length,
        ]

        polygons.append(
            shapely.polygons(np.concatenate((finite_part, extra_edge)))
        )
    return polygons
