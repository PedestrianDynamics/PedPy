"""Computation of method CCM
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pygeos

from report.data.configuration import ConfigurationMethodCCM, ConfigurationVelocity
from report.data.geometry import Geometry
from report.data.trajectory_data import TrajectoryData
from report.methods.density_calculator import (
    _compute_individual_voronoi_polygons,
    _compute_intersecting_polygons,
)
from report.methods.velocity_calculator import compute_individual_velocity


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
    measurement_lines: Dict[int, pygeos.Geometry],
    geometry: Geometry,
    velocity_configuration: ConfigurationVelocity,
) -> Dict[int, ResultMethodCCM]:
    results = {}
    for measurement_line_id, configuration in configurations.items():
        result = _run_method_ccm(
            configuration,
            trajectory,
            measurement_lines[measurement_line_id],
            geometry,
            velocity_configuration,
        )
        results[measurement_line_id] = ResultMethodCCM(result[0], result[1])
    return results


def _run_method_ccm(
    configuration: ConfigurationMethodCCM,
    trajectory: TrajectoryData,
    measurement_line: pygeos.Geometry,
    geometry: Geometry,
    velocity_configuration: ConfigurationVelocity,
):
    # TODO get_main_movement_direction for each pedestrian and switch (second step)

    line_width = configuration.line_width
    if line_width > 0:
        measurement_line = pygeos.union(
            pygeos.buffer(measurement_line, line_width, single_sided=True),
            pygeos.buffer(measurement_line, -line_width, single_sided=True),
        )

    individual_speed = compute_individual_velocity(
        trajectory.data,
        trajectory.frame_rate,
        velocity_configuration.frame_step,
        velocity_configuration.movement_direction,
    )

    individual_voronoi = _compute_individual_voronoi_polygons(trajectory.data, geometry, True)
    intersection_voronoi = _compute_intersecting_polygons(individual_voronoi, measurement_line)

    combined = individual_voronoi.merge(intersection_voronoi, on=["ID", "frame"]).merge(
        individual_speed, on=["ID", "frame"]
    )

    if pygeos.area(measurement_line) > 0:
        combined["density"] = np.where(
            pygeos.area(combined["intersection voronoi"]) > 0,
            pygeos.area(combined["intersection voronoi"])
            / pygeos.area(combined["individual voronoi"]),
            0,
        )
    else:
        combined["density"] = np.where(
            pygeos.length(combined["intersection voronoi"]) > 0,
            pygeos.length(combined["intersection voronoi"])
            / pygeos.area(combined["individual voronoi"]),
            0,
        )

    scaling_factor = (
        pygeos.length(measurement_line)
        if pygeos.area(measurement_line) == 0
        else pygeos.area(measurement_line)
    )

    mean_rho_v = combined.groupby("frame")[["density", "speed"]].agg(
        {"density": "sum", "speed": "mean"}
    )
    mean_rho_v["density"] = mean_rho_v["density"] / scaling_factor

    return mean_rho_v, combined
