"""Computation of method CCM
"""
import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString

from report.data.configuration import ConfigurationMethodCCM
from report.data.geometry import Geometry
from report.data.trajectory_data import TrajectoryData
from report.methods.utils import compute_voronoi_polygons
from report.methods.velocity_calculator import VelocityCalculator


@dataclass(frozen=True)
class ResultMethodCCM:
    """Result of method CCM
    Attributes:
        df_mean (pd.DataFrame): data frame containing: frame, density and mean_velocity
        df_individual (pd.DataFrame): data frame containing when intersection not empty:
            frame, id, x, y, z, individual density, individual velocity, proportion of measurement line,
            voronoi polygon, intersection with measurement line
    """

    df_mean: pd.DataFrame
    df_individual: pd.DataFrame


def run_method_ccm(
    configurations: Dict[int, ConfigurationMethodCCM],
    trajectory: TrajectoryData,
    measurement_lines: Dict[int, LineString],
    geometry: Geometry,
    velocity_calculator: VelocityCalculator,
) -> Dict[int, ResultMethodCCM]:
    results = {}
    for measurement_line_id, configuration in configurations.items():
        results[measurement_line_id] = _run_method_ccm(
            configuration,
            trajectory,
            measurement_lines[measurement_line_id],
            geometry,
            velocity_calculator,
        )
    return results


def _run_method_ccm(
    configuration: ConfigurationMethodCCM,
    trajectory: TrajectoryData,
    measurement_line: LineString,
    geometry: Geometry,
    velocity_calculator: VelocityCalculator,
) -> ResultMethodCCM:
    # TODO get_main_movement_direction for each pedestrian and switch (second step)

    df_mean = pd.DataFrame(columns=["frame", "density", "velocity"])
    df_individual = pd.DataFrame(
        columns=[
            "frame",
            "id",
            "x",
            "y",
            "z",
            "individual density",
            "individual velocity",
            "proportion",
            "voronoi polygon",
            "intersection",
        ]
    )

    line_width = configuration.line_width
    # line_width = 0.5 # 0.25 # 0.05
    if line_width > 0:
        measurement_line = measurement_line.buffer(line_width, single_sided=True).union(
            measurement_line.buffer(-line_width, single_sided=True)
        )
        # measurement_line = geometry.get_as_polygon().intersection(measurement_line)
    min_frame = trajectory.get_min_frame()
    max_frame = trajectory.get_max_frame()

    # passing_frames = compute_passing_frames(trajectory, measurement_line)
    for frame in range(min_frame + 1, max_frame + 1):
        if frame % 100 == 0:
            logging.getLogger("JPSreport").info(f"Processing frame {frame} of {max_frame}")

        # passing_pedestrians = compute_passing_in_frame(trajectory, measurement_line, frame)
        # if not passing_pedestrians:
        #     continue

        # compute voronoi
        pedestrian_positions = trajectory.get_positions(frame)
        voronoi_polygons = compute_voronoi_polygons(
            list(pedestrian_positions.values()), geometry.get_as_polygon()
        )
        cut_off = configuration.cut_off_radius
        if cut_off > 0.0:
            voronoi_polygons = GeometryCollection(
                [
                    voronoi_polygons[i].intersection(
                        list(pedestrian_positions.values())[i].buffer(cut_off)
                    )
                    for i in range(len(voronoi_polygons.geoms))
                ]
            )

        # compute intersection with measurement line

        v = {
            list(pedestrian_positions.keys())[i]: voronoi_polygons.geoms[i]
            for i in range(len(voronoi_polygons.geoms))
            if not voronoi_polygons.geoms[i].intersection(measurement_line).is_empty
        }

        intersections = {
            ped_id: voronoi_polygon.intersection(measurement_line)
            for ped_id, voronoi_polygon in v.items()
        }

        scaling_factor = 1
        # compute densities
        if measurement_line.geom_type == "Polygon":
            # intersecting_proportion = {
            #     ped_id: intersection.area / measurement_line.area
            #     for ped_id, intersection in intersections.items()
            # }
            intersection_densities = {
                ped_id: intersection.area / v[ped_id].area
                for ped_id, intersection in intersections.items()
            }
            scaling_factor = 1.0 / measurement_line.area

        elif measurement_line.geom_type == "LineString":
            # intersecting_proportion = {
            #     ped_id: intersection.length / measurement_line.length
            #     for ped_id, intersection in intersections.items()
            # }
            intersection_densities = {
                ped_id: intersection.length / v[ped_id].area
                for ped_id, intersection in intersections.items()
            }
            scaling_factor = 1.0 / measurement_line.length

        rho = scaling_factor * np.sum(list(intersection_densities.values()))
        # compute velocities
        intersecting_velocities = {
            ped_id: velocity_calculator.compute_instantaneous_velocity(trajectory, ped_id, frame)
            for ped_id in intersections.keys()
        }

        velocity = np.mean(list(intersecting_velocities.values()))
        # compute mean
        df_mean.loc[df_mean.shape[0]] = [
            frame,
            rho,
            velocity,
        ]

        # print(f"frame: {frame} rho: {rho} v: {velocity}")
        # for ped_id in intersections.keys():
        #     x = pedestrian_positions[ped_id].x
        #     y = pedestrian_positions[ped_id].y
        #     z = 0
        #     density = intersection_densities[ped_id]
        #     velocity = intersecting_velocities[ped_id]
        #     proportion = intersecting_proportion[ped_id]
        #     polygon = intersections[ped_id].wkt
        #     intersection = intersections[ped_id].wkt
        #     df_individual.loc[df_individual.shape[0]] = [
        #         frame,
        #         ped_id,
        #         x,
        #         y,
        #         z,
        #         density,
        #         velocity,
        #         proportion,
        #         polygon,
        #         intersection,
        #     ]

        # df_flow.loc[df_flow.shape[0]] = [flow, np.mean(passed_velocity)]

        # if frame % 100 == 0:
        #     fig = plt.figure()
        #     fig.set_frameon(True)
        #     ax = fig.add_subplot(111)
        #
        #     for region in voronoi_polygons.geoms:
        #         patch = PolygonPatch(region, facecolor="blue", edgecolor="blue", alpha=0.5, zorder=2)
        #         ax.add_patch(patch)
        #
        #     for i, f in intersections.items():
        #         if f.geom_type == "Polygon":
        #             patch_inter = PolygonPatch(f, alpha=0.5, zorder=2)
        #             ax.add_patch(patch_inter)
        #
        #         if f.geom_type == "LineString":
        #             x, y = f.xy
        #             ax.plot(x, y, alpha=0.7, linewidth=3, zorder=2)
        #
        #         point = pedestrian_positions[i]
        #         plt.plot(point.x, point.y, "o", color="k")
        #     plt.show()
        #     print(f"frame: {frame} rho: {rho} v: {velocity}")
        # print(f"rho: {rho}")
        # print()
    # print(df_mean)
    # print(df_individual)

    return ResultMethodCCM(df_mean, df_individual)
