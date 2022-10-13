"""This module provides the functionalities to parse trajectory files to the internal
TrajectoryData format.
"""

import pathlib

import pandas as pd

from pedpy.data.trajectory_data import TrajectoryData, TrajectoryUnit


def load_trajectory(
    *,
    trajectory_file: pathlib.Path,
    default_frame_rate: float = None,
    default_unit: TrajectoryUnit = None,
) -> TrajectoryData:
    """L the trajectory file for the relevant data: trajectory data, frame
    rate, and type of trajectory.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory
        default_frame_rate (float): frame rate of the file, None if frame rate from
                file is used
        default_unit (TrajectoryUnit): unit in which the coordinates are stored
                in the file, None if unit should be parsed from the file

    Returns:
        Tuple containing: trajectory data, frame rate, and type of trajectory.
    """

    if not trajectory_file.exists():
        raise IOError(f"{trajectory_file} does not exist.")

    if not trajectory_file.is_file():
        raise IOError(f"{trajectory_file} is not a file.")

    traj_frame_rate, traj_unit = _load_trajectory_meta_data(
        trajectory_file=trajectory_file,
        default_frame_rate=default_frame_rate,
        default_unit=default_unit,
    )
    traj_dataframe = _load_trajectory_data(
        trajectory_file=trajectory_file, unit=traj_unit
    )

    return TrajectoryData(traj_dataframe, traj_frame_rate, trajectory_file)


def _load_trajectory_data(
    *, trajectory_file: pathlib.Path, unit: TrajectoryUnit
) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory
        default_unit (TrajectoryUnit): unit in which the coordinates are stored
            in the file, None if unit should be parsed from the file

    Returns:
        The trajectory data as data frame, the coordinates are converted to
        meter (m).
    """
    try:
        data = pd.read_csv(
            trajectory_file,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["ID", "frame", "X", "Y", "Z"],
            usecols=[0, 1, 2, 3, 4],
            dtype={
                "ID": "int64",
                "frame": "int64",
                "X": "float64",
                "Y": "float64",
                "Z": "float64",
            },
        )

        if data.empty:
            raise ValueError(
                f"The given trajectory file seem to be empty. It should "
                f"contain at least 5 columns: ID, frame, X, Y, Z. The values "
                f"should be separated by any white space. Comment line may "
                f"start with a '#' and will be ignored. "
                f"Please check your trajectory file: {trajectory_file}."
            )

        if unit == TrajectoryUnit.CENTIMETER:
            data["X"] = data["X"].div(100)
            data["Y"] = data["Y"].div(100)
            data["Z"] = data["Z"].div(100)

        return data
    except pd.errors.ParserError:
        raise ValueError(
            f"The given trajectory file could not be parsed. It should "
            f"contain at least 5 columns: ID, frame, X, Y, Z. The values "
            f"should be separated by any white space. Comment line may start "
            f"with a '#' and will be ignored. "
            f"Please check your trajectory file: {trajectory_file}."
        )


def _load_trajectory_meta_data(
    *,
    trajectory_file: pathlib.Path,
    default_frame_rate: float,
    default_unit: TrajectoryUnit,
):
    parsed_frame_rate = None
    parsed_unit = None

    with open(trajectory_file, "r") as file_content:
        for line in file_content:
            if not line.startswith("#"):
                break

            if "framerate" in line:
                for substring in line.split():
                    try:
                        if parsed_frame_rate is None:
                            parsed_frame_rate = float(substring)
                    except:
                        continue

            if "x/cm" in line.lower() or "in cm" in line.lower():
                parsed_unit = TrajectoryUnit.CENTIMETER
            if "x/m" in line.lower() or "in m" in line.lower():
                parsed_unit = TrajectoryUnit.METER

    frame_rate = parsed_frame_rate
    if parsed_frame_rate is None and default_frame_rate is not None:
        if default_frame_rate <= 0:
            raise ValueError(
                f"Default frame needs to be positive but is "
                f"{default_frame_rate}"
            )

        frame_rate = default_frame_rate

    if parsed_frame_rate is None and default_frame_rate is None:
        raise ValueError(
            f"Frame rate is needed, but none could be found in the trajectory "
            f"file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            f"a default frame rate."
        )

    if parsed_frame_rate is not None and default_frame_rate is None:
        if parsed_frame_rate <= 0:
            raise ValueError(
                f"Frame rate needs to be a positive value, but is "
                f"{parsed_frame_rate}. "
                f"Please check your trajectory file: {trajectory_file}."
            )
    if parsed_frame_rate is not None and default_frame_rate is not None:
        if parsed_frame_rate != default_frame_rate:
            raise ValueError(
                "The given default frame rate seems to differ from the frame "
                "rate given in the trajectory file: "
                "{default_frame_rate} != {parsed_frame_rate}"
            )

    unit = parsed_unit
    if parsed_unit is None and default_unit is not None:
        unit = default_unit

    if parsed_unit is None and default_unit is None:
        raise ValueError(
            f"Unit is needed, but none could be found in the trajectory file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            f"a default unit."
        )

    if parsed_unit is not None and default_unit is not None:
        if parsed_unit != default_unit:
            raise ValueError(
                "The given default unit seems to differ from the unit given in "
                "the trajectory file: "
                "{default_unit} != {parsed_unit}"
            )

    return frame_rate, unit
