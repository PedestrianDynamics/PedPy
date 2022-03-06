"""This module provides the functionalities to parse trajectory files to the internal
TrajectoryData format.
"""

import pathlib
from typing import List

import pandas as pd

from report.data.trajectory_data import TrajectoryData, TrajectoryType, TrajectoryUnit


def parse_trajectory_files(trajectory_files: List[pathlib.Path]) -> TrajectoryData:
    """Parses the given files for the relevant data.

    Raises:
        ValueError: if there is a mismatch of the frame rate or type in the given files
        ValueError: if there are the same (ID and frame) in multiple files

    Args:
        trajectory_files (List[pathlib.Path]): list of files containing the trajectory

    Returns:
        TrajectoryData containing the data from all trajectory files.
    """
    traj_dataframe = pd.DataFrame()
    traj_frame_rate = None
    traj_type = None

    for traj_file in trajectory_files:
        tmp_dataframe, tmp_frame_rate, tmp_traj_type = parse_trajectory_file(traj_file)

        if traj_frame_rate is None and traj_type is None:
            traj_frame_rate = tmp_frame_rate
            traj_type = tmp_traj_type

        if traj_frame_rate != tmp_frame_rate:
            raise ValueError(
                f"Frame rates of the trajectory files differ: {traj_frame_rate} != "
                f"{tmp_frame_rate}. Please check the trajectory files: {trajectory_files}."
            )

        if traj_type != tmp_traj_type:
            raise ValueError(
                f"Types of the trajectory files differ: {traj_type} != {tmp_traj_type}. "
                f"Please check the trajectory files: {trajectory_files}."
            )

        if traj_dataframe.empty:
            traj_dataframe = tmp_dataframe.copy(deep=True)
        else:
            traj_dataframe = traj_dataframe.append(tmp_dataframe)

        if traj_dataframe.duplicated(subset=["ID", "frame"]).any():
            raise ValueError(
                f"The trajectory data could not be stored in one data frame. This happens when "
                f"there is a ID + frame combination in multiple files. "
                f"Please check the trajectory files: {trajectory_files}."
            )

    return TrajectoryData(traj_dataframe, traj_frame_rate, traj_type, trajectory_files)


def parse_trajectory_file(trajectory_file: pathlib.Path) -> (pd.DataFrame, float, TrajectoryType):
    """Parse the trajectory file for the relevant data: trajectory data, frame rate, and type of
    trajectory.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory

    Returns:
        Tuple containing: trajectory data, frame rate, and type of trajectory.
    """
    traj_dataframe = parse_trajectory_data(trajectory_file)
    traj_frame_rate = parse_frame_rate(trajectory_file)
    traj_type = parse_trajectory_type(trajectory_file)

    return traj_dataframe, traj_frame_rate, traj_type


def parse_trajectory_data(trajectory_file: pathlib.Path) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory

    Returns:
        The trajectory data as data frame, the coordinates are converted to meter (m).
    """
    unit = parse_unit_of_coordinates(trajectory_file)

    try:
        data = pd.read_csv(
            trajectory_file,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["ID", "frame", "X", "Y", "Z"],
            usecols=[0, 1, 2, 3, 4],
            dtype={"ID": "int64", "frame": "int64", "X": "float64", "Y": "float64", "Z": "float64"},
        )

        if unit == TrajectoryUnit.CENTIMETER:
            data["X"] = data["X"].div(100)
            data["Y"] = data["Y"].div(100)
            data["Z"] = data["Z"].div(100)

        if data.empty:
            raise ValueError(
                f"The given trajectory file seem to be empty. It should contain at least 5 columns:"
                f"ID, frame, X, Y, Z. The values should be separated by any white space. Comment "
                f"line may start with a '#' and will be ignored. "
                f"Please check your trajectory file: {trajectory_file}."
            )
        return data
    except pd.errors.ParserError:
        raise ValueError(
            f"The given trajectory file could not be parsed. It should contain at least 5 columns: "
            f"ID, frame, X, Y, Z. The values should be separated by any white space. Comment "
            f"line may start with a '#' and will be ignored. "
            f"Please check your trajectory file: {trajectory_file}."
        )


def parse_frame_rate(trajectory_file: pathlib.Path) -> float:
    """Parse the trajectory file for the used framerate.

    Searches for the first line starting with '#' and containing the word 'framerate' and at
    least one floating point value. If float values are found, the first is returned.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory

    Returns:
        the frame rate used in the trajectory file
    """
    frame_rate = None
    with open(trajectory_file, "r") as file_content:
        for line in file_content:
            if not line.startswith("#"):
                break

            if "framerate" in line:
                for substring in line.split():
                    try:
                        frame_rate = float(substring)
                        break
                    except:
                        continue

    if frame_rate is None:
        raise ValueError(
            f"Frame rate is needed, but none could be found in the trajectory files. "
            f"Please check your trajectory file: {trajectory_file}."
        )

    if frame_rate <= 0:
        raise ValueError(
            f"Frame rate needs to be a positive value, but is {frame_rate}. "
            f"Please check your trajectory file: {trajectory_file}."
        )

    return frame_rate


def parse_trajectory_type(trajectory_file: pathlib.Path) -> TrajectoryType:
    """Parse the trajectory file for the type of trajectory, e.g., the origin of the trajectory file.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory

    Returns:
        The type of the trajectory
    """
    with open(trajectory_file, "r") as file_content:
        line = file_content.readline()
        if "PeTrack" in line:
            trajectory_type = TrajectoryType.PETRACK
        elif "jpscore" in line:
            trajectory_type = TrajectoryType.JUPEDSIM
        else:
            trajectory_type = TrajectoryType.FALLBACK
    return trajectory_type


def parse_unit_of_coordinates(trajectory_file: pathlib.Path) -> TrajectoryUnit:
    """Parse the trajectory file for the used units of the coordinates.

    Note:
        currently only works for PeTrack trajectory files

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory

    Returns:
        The unit used in the trajectory file for the coordinates. If no explicit unit is given, METER is returned.
    """
    unit = TrajectoryUnit.METER
    with open(trajectory_file, "r") as file_content:
        for line in file_content:
            if not line.startswith("#"):
                break

            if "x/cm" in line.lower():
                unit = TrajectoryUnit.CENTIMETER
                break
    return unit
