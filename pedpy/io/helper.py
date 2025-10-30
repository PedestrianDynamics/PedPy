"""Helper functions for data loading methods."""

import math
import pathlib
from enum import Enum
from typing import Tuple

import pandas as pd

from pedpy.column_identifier import ID_COL
from pedpy.errors import LoadTrajectoryError


class TrajectoryUnit(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of the unit of the trajectory coordinates."""

    METER = 1
    """meter (m)"""
    CENTIMETER = 100
    """centimeter (cm)"""


def _validate_is_file(file: pathlib.Path) -> None:
    """Validates if the given file is a valid file, if valid raises Exception.

    A file is considered invalid if:

    - it does not exist
    - is not a file, but a directory

    Args:
        file: File to check


    """
    if not file.exists():
        raise LoadTrajectoryError(f"{file} does not exist.")

    if not file.is_file():
        raise LoadTrajectoryError(f"{file} is not a file.")


def _calculate_frames_and_fps(
    traj_dataframe: pd.DataFrame,
) -> Tuple[pd.Series, int]:
    """Calculates fps and frames based on the time column of the dataframe."""
    mean_diff = traj_dataframe.groupby(ID_COL)["time"].diff().dropna().mean()
    if math.isnan(mean_diff):
        raise LoadTrajectoryError(
            "Can not determine the frame rate used to write the trajectory "
            "file. This may happen, if the file only contains data for a "
            "single frame."
        )

    fps = round(1 / mean_diff)
    frames = traj_dataframe["time"] * fps
    frames = frames.round().astype("int64")
    return frames, fps
