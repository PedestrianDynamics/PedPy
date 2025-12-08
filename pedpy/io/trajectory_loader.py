"""Load trajectories to the internal trajectory data format."""

import pathlib
from typing import Optional

from pedpy.data.trajectory_data import TrajectoryData
from pedpy.io.helper import TrajectoryUnit
from pedpy.io.ped_data_archive_loader import load_trajectory_from_txt


def load_trajectory(
    *,
    trajectory_file: pathlib.Path,
    default_frame_rate: Optional[float] = None,
    default_unit: Optional[TrajectoryUnit] = None,
) -> TrajectoryData:
    """Loads the trajectory file in the internal :class:`~trajectory_data.TrajectoryData` format.

    Loads the relevant data: trajectory data, frame rate, and type of
    trajectory from the given trajectory file. If the file does not contain
    some data, defaults can be submitted.

    Args:
        trajectory_file: file containing the trajectory
        default_frame_rate: frame rate of the file, None if frame rate
            from file is used
        default_unit: unit in which the coordinates are stored
                in the file, None if unit should be parsed from the file

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData`
        representation of the file data
    """
    return load_trajectory_from_txt(
        trajectory_file=trajectory_file,
        default_frame_rate=default_frame_rate,
        default_unit=default_unit,
    )
