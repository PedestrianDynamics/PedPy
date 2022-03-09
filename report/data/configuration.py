from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List

from shapely.geometry import LineString, Polygon

from report.methods.velocity_calculator import VelocityCalculator


@dataclass(frozen=True)
class ConfigurationMethodA:
    """Configuration for method_A

    Attributes:
        frame_interval (int): specifies the size of frame interval for calculating flow rate.
    """

    frame_interval: int


@dataclass(frozen=True)
class Configuration:
    """Configuration to be used in the analysis

    Attributes:
        output_directory (pathlib.Path): directory in which to store the results
        trajectory_files (List[pathlib.Path]): trajectory files to be used in the analysis
        geometry_file (pathlib.Path): geometry file to be used in the analysis

        measurement_areas (Dict[int, Polygon]): measurement areas to be used in the analysis
        measurement_lines (Dict[int, LineString]): measurement line to be used in the analysis

        velocity_calculator (VelocityCalculator): configuration for the velocity computation
                (see :class:VelocityCalculator for more detail)

        config_method_a (Dict[int, ConfigurationMethodA]): configuration for the method a
                computation per measurement_line (see :class:ConfigurationMethodA for more detail)

    """

    output_directory: pathlib.Path
    trajectory_files: List[pathlib.Path]
    geometry_file: pathlib.Path

    measurement_areas: Dict[int, Polygon]
    measurement_lines: Dict[int, LineString]

    velocity_calculator: VelocityCalculator

    config_method_a: Dict[int, ConfigurationMethodA]
