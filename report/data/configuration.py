from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List

from shapely.geometry import LineString, Polygon


@dataclass(frozen=True)
class ConfigurationVelocity:
    """Configuration for velocity computation

    Attributes:
         frame_step (int): gives the size of time interval for calculating the velocity
         set_movement_direction (str): indicates in which direction the velocity will be projected
         ignore_backward_movement (bool):  indicates whether you want to ignore the movement opposite to
                                           the direction from `set_movement_direction`
    """

    frame_step: int
    set_movement_direction: str
    ignore_backward_movement: bool


@dataclass(frozen=True)
class Configuration:
    """Configuration to be used in the analysis

    Attributes:
        output_directory (pathlib.Path): directory in which to store the results
        trajectory_files (List[pathlib.Path]): trajectory files to be used in the analysis
        geometry_file (pathlib.Path): geometry file to be used in the analysis

        measurement_areas (Dict[int, Polygon]): measurement areas to be used in the analysis
        measurement_lines (Dict[int, LineString]): measurement line to be used in the analysis

        velocity_configuration (ConfigurationVelocity): configuration for the velocity compuation
                (see :class:ConfigurationVelocity for more detail)
    """

    output_directory: pathlib.Path
    trajectory_files: List[pathlib.Path]
    geometry_file: pathlib.Path

    measurement_areas: Dict[int, Polygon]
    measurement_lines: Dict[int, LineString]

    velocity_configuration: ConfigurationVelocity
