"""Load Pathfinder trajectories to the internal trajectory data format."""

import json
import logging
import pathlib

import pandas as pd

from pedpy.column_identifier import FRAME_COL, ID_COL, TIME_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import LoadTrajectoryError
from pedpy.io.helper import _calculate_frames_and_fps, _validate_is_file

_log = logging.getLogger(__name__)


def load_trajectory_from_pathfinder_json(
    *,
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads Pathfinder-JSON as :class:`~trajectory_data.TrajectoryData`.

    This function reads a JSON file containing trajectory data
    from Pathfinder simulations and converts it
    into a :class:`~trajectory_data.TrajectoryData` object.

    .. note::

        Pathfinder JSON data have a time-based structure that is going to be
        converted to a frame column for use with *PedPy*.

    Args:
        trajectory_file: The full path of the JSON containing the Pathfinder
            trajectory data. The expected format is a JSON file with agent IDs
            as top-level keys, and time-stamped position data as nested objects.

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
        of the file data

    Raises:
        LoadTrajectoryError: If the provided path does not exist or is not a
            file, or if the JSON structure is invalid.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_pathfinder_json(trajectory_file=trajectory_file)
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(traj_dataframe)

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_pathfinder_json(*, trajectory_file: pathlib.Path) -> pd.DataFrame:
    """Parse the trajectory JSON file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): The file containing the trajectory data.
            The expected format is a JSON file with agent IDs as top-level keys,
            and time-stamped position data as nested objects.

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        in meter (m).
    """
    common_error_message = (
        "The given trajectory file seems to be incorrect or empty. "
        "It should be a valid JSON file with agent IDs as top-level keys "
        "and time-stamped position data containing 'position' objects with "
        "'x' and 'y' coordinates. "
        f"Please check your trajectory file: {trajectory_file}."
    )

    try:
        with open(trajectory_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        raise LoadTrajectoryError(f"{common_error_message}\nOriginal error: {e}") from e

    if not isinstance(data, dict) or not data:
        raise LoadTrajectoryError(f"{common_error_message}\nEmpty or invalid JSON structure.")

    trajectory_records = []

    try:
        for agent_id_str, time_data in data.items():
            agent_id = int(agent_id_str)

            if not isinstance(time_data, dict):
                continue

            for time_str, agent_data in time_data.items():
                time_value = float(time_str)

                if not isinstance(agent_data, dict):
                    continue

                position = agent_data.get("position")
                if not isinstance(position, dict):
                    continue

                x_pos = position.get("x")
                y_pos = position.get("y")

                if x_pos is not None and y_pos is not None:
                    trajectory_records.append(
                        {
                            ID_COL: agent_id,
                            TIME_COL: time_value,
                            X_COL: float(x_pos),
                            Y_COL: float(y_pos),
                        }
                    )

    except (ValueError, KeyError, TypeError) as e:
        raise LoadTrajectoryError(f"{common_error_message}\nError parsing JSON structure: {e}") from e

    if not trajectory_records:
        raise LoadTrajectoryError(f"{common_error_message}\nNo valid trajectory data found.")

    traj_dataframe = pd.DataFrame(trajectory_records)

    # Sort by agent ID and time for consistency
    traj_dataframe = traj_dataframe.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    return traj_dataframe


def load_trajectory_from_pathfinder_csv(
    *,
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from Pathfinder-CSV file as :class:`~trajectory_data.TrajectoryData`.

    This function reads a CSV file containing trajectory data from Pathfinder
    simulations and converts it into a :class:`~trajectory_data.TrajectoryData`
    object.

    .. note::

        Pathfinder data have a time column, that is going to be converted to a
        frame column for use with *PedPy*.

    .. warning::

        Currently only Pathfinder files with a time column can be loaded.

    Args:
        trajectory_file: The full path of the CSV file containing the Pathfinder
            trajectory data. The expected format is a CSV file with comma
            as delimiter, and it should contain at least the following
            columns: id, t, x, y.

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
        of the file data

    Raises:
        LoadTrajectoryError: If the provided path does not exist or is not a
            file.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_pathfinder_csv(trajectory_file=trajectory_file)
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(traj_dataframe)

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_pathfinder_csv(*, trajectory_file: pathlib.Path) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): The file containing the trajectory data.
            The expected format is a CSV file with comma as delimiter, and it
            should contain at least the following columns: name, t, x, y.

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        in meter (m).
    """
    columns_to_keep = ["id", "t", "x", "y"]
    rename_mapping = {
        "id": ID_COL,
        "t": TIME_COL,
        "x": X_COL,
        "y": Y_COL,
    }
    column_types = {"id": int, "time": float, "x": float, "y": float}

    common_error_message = (
        "The given trajectory file seems to be incorrect or empty. "
        "It should contain at least the following columns: "
        "id, t, x, y, separated by comma. "
        f"Please check your trajectory file: {trajectory_file}."
    )
    # csv has a unit line. Usually the second line,
    # but not 100% sure if this is always the case.
    try:
        raw = pd.read_csv(
            trajectory_file,
            encoding="utf-8-sig",
        )
    except Exception as e:
        raise LoadTrajectoryError(f"{common_error_message}\nOriginal error: {e}") from e
    # filter out the unit line
    data = raw[raw["t"].apply(lambda v: str(v).replace(".", "", 1).isdigit())].copy()
    missing_columns = set(columns_to_keep) - set(data.columns)
    if missing_columns:
        raise LoadTrajectoryError(f"{common_error_message} Missing columns: {', '.join(missing_columns)}.")
    try:
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)
        data = data.astype(column_types)
    except Exception as e:
        raise LoadTrajectoryError(f"{common_error_message}\nOriginal error: {e}") from e

    if data.empty:
        raise LoadTrajectoryError(f"{common_error_message}.\n Empty dataframe.")

    return data
