"""This module provides the functionalities to parse trajectory files to the internal
TrajectoryData format.
"""

import pathlib

import pandas as pd
import pygeos

from report.data.trajectory_data import TrajectoryData, TrajectoryType, TrajectoryUnit


def parse_trajectory(trajectory_file: pathlib.Path) -> TrajectoryData:
    """Parses the given file for the relevant data.

    Args:
        trajectory_file (pathlib.Path): files containing the trajectory data

    Returns:
        TrajectoryData containing the data from all trajectory files.
    """

    traj_dataframe, traj_frame_rate, traj_type = parse_trajectory_file(trajectory_file)

    return TrajectoryData(traj_dataframe, traj_frame_rate, traj_type, trajectory_file)


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

        data["points"] = pygeos.points(data["X"], data["Y"])

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
