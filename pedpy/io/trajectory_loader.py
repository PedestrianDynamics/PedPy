"""Load trajectories to the internal trajectory data format."""

import json
import logging
import math
import pathlib
import sqlite3
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import shapely
from shapely import Point, Polygon

from pedpy.column_identifier import FRAME_COL, ID_COL, TIME_COL, X_COL, Y_COL
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import LoadTrajectoryError, PedPyValueError

_log = logging.getLogger(__name__)


class TrajectoryUnit(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of the unit of the trajectory coordinates."""

    METER = 1
    """meter (m)"""
    CENTIMETER = 100
    """centimeter (cm)"""


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

    traj_dataframe = _load_trajectory_data_from_pathfinder_json(
        trajectory_file=trajectory_file
    )
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(
        traj_dataframe
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_pathfinder_json(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
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
        raise LoadTrajectoryError(
            f"{common_error_message}\nOriginal error: {e}"
        ) from e

    if not isinstance(data, dict) or not data:
        raise LoadTrajectoryError(
            f"{common_error_message}\nEmpty or invalid JSON structure."
        )

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
        raise LoadTrajectoryError(
            f"{common_error_message}\nError parsing JSON structure: {e}"
        ) from e

    if not trajectory_records:
        raise LoadTrajectoryError(
            f"{common_error_message}\nNo valid trajectory data found."
        )

    traj_dataframe = pd.DataFrame(trajectory_records)

    # Sort by agent ID and time for consistency
    traj_dataframe = traj_dataframe.sort_values([ID_COL, TIME_COL]).reset_index(
        drop=True
    )

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
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_pathfinder_csv(
        trajectory_file=trajectory_file
    )
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(
        traj_dataframe
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_pathfinder_csv(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
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
        raise LoadTrajectoryError(
            f"{common_error_message}\nOriginal error: {e}"
        ) from e
    # filter out the unit line
    data = raw[
        raw["t"].apply(lambda v: str(v).replace(".", "", 1).isdigit())
    ].copy()
    missing_columns = set(columns_to_keep) - set(data.columns)
    if missing_columns:
        raise LoadTrajectoryError(
            f"{common_error_message} "
            f"Missing columns: {', '.join(missing_columns)}."
        )
    try:
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)
        data = data.astype(column_types)
    except Exception as e:
        raise LoadTrajectoryError(
            f"{common_error_message}\nOriginal error: {e}"
        ) from e

    if data.empty:
        raise LoadTrajectoryError(f"{common_error_message}.\n Empty dataframe.")

    return data


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
        trajectory_file (pathlib.Path): file containing the trajectory
        default_frame_rate (float): frame rate of the file, None if frame rate
            from file is used
        default_unit (TrajectoryUnit): unit in which the coordinates are stored
                in the file, None if unit should be parsed from the file

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData`
            representation of the file data
    """  # noqa: E501
    return load_trajectory_from_txt(
        trajectory_file=trajectory_file,
        default_frame_rate=default_frame_rate,
        default_unit=default_unit,
    )


def load_trajectory_from_txt(
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
        trajectory_file (pathlib.Path): file containing the trajectory
        default_frame_rate (float): frame rate of the file, None if frame rate
            from file is used
        default_unit (TrajectoryUnit): unit in which the coordinates are stored
                in the file, None if unit should be parsed from the file

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData`
            representation of the file data
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    traj_frame_rate, traj_unit = _load_trajectory_meta_data_from_txt(
        trajectory_file=trajectory_file,
        default_frame_rate=default_frame_rate,
        default_unit=default_unit,
    )
    traj_dataframe = _load_trajectory_data_from_txt(
        trajectory_file=trajectory_file, unit=traj_unit
    )

    return TrajectoryData(data=traj_dataframe, frame_rate=traj_frame_rate)


def _load_trajectory_data_from_txt(
    *, trajectory_file: pathlib.Path, unit: TrajectoryUnit
) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory
        unit (TrajectoryUnit): unit in which the coordinates are stored
            in the file, None if unit should be parsed from the file

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        converted to meter (m).
    """
    try:
        data = pd.read_csv(
            trajectory_file,
            sep=r"\s+",
            comment="#",
            header=None,
            names=[ID_COL, FRAME_COL, X_COL, Y_COL],
            usecols=[0, 1, 2, 3],
            dtype={
                ID_COL: "int64",
                FRAME_COL: "int64",
                X_COL: "float64",
                Y_COL: "float64",
            },
        )

        if data.empty:
            raise PedPyValueError(
                "The given trajectory file seem to be empty. It should "
                "contain at least 5 columns: ID, frame, X, Y, Z. The values "
                "should be separated by any white space. Comment line may "
                "start with a '#' and will be ignored. "
                f"Please check your trajectory file: {trajectory_file}."
            )

        if unit == TrajectoryUnit.CENTIMETER:
            data.x = data.x.div(100)
            data.y = data.y.div(100)

        return data
    except pd.errors.ParserError as exc:
        raise PedPyValueError(
            "The given trajectory file could not be parsed. It should "
            "contain at least 5 columns: ID, frame, X, Y, Z. The values "
            "should be separated by any white space. Comment line may start "
            "with a '#' and will be ignored. "
            f"Please check your trajectory file: {trajectory_file}."
        ) from exc


def _load_trajectory_meta_data_from_txt(  # noqa: PLR0912
    *,
    trajectory_file: pathlib.Path,
    default_frame_rate: Optional[float],
    default_unit: Optional[TrajectoryUnit],
) -> Tuple[float, TrajectoryUnit]:
    """Extract the trajectory metadata from file, use defaults if none found.

    Check the given trajectory file for the used unit and frame-rate, if none
    were found, return the provided default values.

    In cases that there are differences between the found and given default
    values, an exception is raised. If no metadata found and no defaults given
    also an exception will be raised.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory
        default_frame_rate (float): frame rate of the file, None if frame rate
            from file is used
        default_unit (TrajectoryUnit): unit in which the coordinates are stored
                in the file, None if unit should be parsed from the file

    Returns:
        Tuple containing the frame-rate and used unit.
    """
    parsed_frame_rate: Any = None
    parsed_unit: Any = None

    with open(trajectory_file, "r", encoding="utf-8-sig") as file_content:
        for line in file_content:
            if not line.startswith("#"):
                break

            if "framerate" in line:
                for substring in line.split():
                    try:
                        if parsed_frame_rate is None:
                            parsed_frame_rate = float(substring)
                    except ValueError:  # noqa: PERF203
                        continue

            if "x/cm" in line.lower() or "in cm" in line.lower():
                parsed_unit = TrajectoryUnit.CENTIMETER

            if "x/m" in line.lower() or "in m" in line.lower():
                parsed_unit = TrajectoryUnit.METER

    frame_rate = parsed_frame_rate
    if parsed_frame_rate is None and default_frame_rate is not None:
        if default_frame_rate <= 0:
            raise PedPyValueError(
                f"Default frame needs to be positive but is "
                f"{default_frame_rate}"
            )

        frame_rate = default_frame_rate

    if parsed_frame_rate is None and default_frame_rate is None:
        raise PedPyValueError(
            "Frame rate is needed, but none could be found in the trajectory "
            "file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            "a default frame rate."
        )

    if parsed_frame_rate is not None and default_frame_rate is None:
        if parsed_frame_rate <= 0:
            raise PedPyValueError(
                "Frame rate needs to be a positive value, but is "
                f"{parsed_frame_rate}. "
                "Please check your trajectory file: {trajectory_file}."
            )
    if parsed_frame_rate is not None and default_frame_rate is not None:
        if parsed_frame_rate != default_frame_rate:
            raise PedPyValueError(
                "The given default frame rate seems to differ from the frame "
                "rate given in the trajectory file: "
                f"{default_frame_rate} != {parsed_frame_rate}"
            )

    unit = parsed_unit
    if parsed_unit is None and default_unit is not None:
        unit = default_unit

    if parsed_unit is None and default_unit is None:
        raise PedPyValueError(
            "Unit is needed, but none could be found in the trajectory file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            "a default unit."
        )

    if parsed_unit is not None and default_unit is not None:
        if parsed_unit != default_unit:
            raise PedPyValueError(
                "The given default unit seems to differ from the unit given "
                "in the trajectory file: "
                f"{default_unit} != {parsed_unit}"
            )

    return frame_rate, unit


def load_trajectory_from_jupedsim_sqlite(
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from the sqlite file as :class:`~trajectory_data.TrajectoryData`.

    Args:
        trajectory_file: trajectory file in JuPedSim sqlite format

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData`
            representation of the file data
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    with sqlite3.connect(trajectory_file) as con:
        try:
            data = pd.read_sql_query(
                "select frame, id, pos_x as x, pos_y as y from trajectory_data",
                con,
            )
        except Exception as exc:
            raise LoadTrajectoryError(
                "The given sqlite trajectory is not a valid JuPedSim format, "
                "it does not not contain a 'trajectory_data' table. Please "
                "check your file."
            ) from exc
        if data.empty:
            raise LoadTrajectoryError(
                "The given sqlite trajectory file seems to be empty. "
                "Please check your file."
            )

        try:
            fps_query_result = (
                con.cursor()
                .execute("select value from metadata where key = 'fps'")
                .fetchone()
            )
        except Exception as exc:
            raise LoadTrajectoryError(
                "The given sqlite trajectory is not a valid JuPedSim format, "
                "it does not not contain a 'metadata' table. Please check "
                "your file."
            ) from exc

        if fps_query_result is None:
            raise LoadTrajectoryError(
                "The given sqlite trajectory file seems not include a frame "
                "rate. Please check your file."
            )
        fps = float(fps_query_result[0])

    return TrajectoryData(data=data, frame_rate=fps)


def load_walkable_area_from_jupedsim_sqlite(
    trajectory_file: pathlib.Path,
) -> WalkableArea:
    """Loads the walkable area from the sqlite file as :class:`~geometry.WalkableArea`.

    .. note::

        When using a JuPedSim sqlite trajectory file with version 2, the
        walkable area is the union of all provided walkable areas in the file.

    Args:
        trajectory_file: trajectory file in JuPedSim sqlite format

    Returns:
        WalkableArea: :class:`~geometry.WalkableArea` used in the simulation
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    with sqlite3.connect(trajectory_file) as connection:
        db_version = _get_jupedsim_sqlite_version(connection)

        if db_version == 1:
            return _load_walkable_area_from_jupedsim_sqlite_v1(connection)

        if db_version == 2:
            return _load_walkable_area_from_jupedsim_sqlite_v2(connection)

        raise LoadTrajectoryError(
            f"The given sqlite trajectory has unsupported db version "
            f"{db_version}. Supported are versions: 1, 2."
        )


def _get_jupedsim_sqlite_version(connection: sqlite3.Connection) -> int:
    cur = connection.cursor()
    return int(
        cur.execute(
            "SELECT value FROM metadata WHERE key = ?", ("version",)
        ).fetchone()[0]
    )


def _load_walkable_area_from_jupedsim_sqlite_v1(
    con: sqlite3.Connection,
) -> WalkableArea:
    try:
        walkable_query_result = (
            con.cursor().execute("select wkt from geometry").fetchone()
        )
    except Exception as exc:
        raise LoadTrajectoryError(
            "The given sqlite trajectory is not a valid JuPedSim format, it "
            "does not not contain a 'geometry' table. Please check your file."
        ) from exc

    if walkable_query_result is None:
        raise LoadTrajectoryError(
            "The given sqlite trajectory file seems not include a geometry. "
            "Please check your file."
        )

    return WalkableArea(walkable_query_result[0])


def _load_walkable_area_from_jupedsim_sqlite_v2(
    con: sqlite3.Connection,
) -> WalkableArea:
    try:
        res = con.cursor().execute("SELECT wkt FROM geometry")
        geometries = [shapely.from_wkt(s) for s in res.fetchall()]
    except Exception as exc:
        raise LoadTrajectoryError(
            "The given sqlite trajectory is not a valid JuPedSim format, "
            "it does not not contain a 'geometry' table. Please check your "
            "file."
        ) from exc

    if not geometries:
        raise LoadTrajectoryError(
            "The given sqlite trajectory file seems not include a geometry. "
            "Please check your file."
        )

    return WalkableArea(shapely.union_all(geometries))


def load_trajectory_from_ped_data_archive_hdf5(
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from the hdf5 file as :class:`~trajectory_data.TrajectoryData`.

    Loads data from files in the
    `Pedestrian Dynamics Data Archive <https://ped.fz-juelich.de/da/doku.php>`_
    HDF5 format. The format is explained in more detail
    `here <https://ped.fz-juelich.de/da/doku.php?id=info>`_.

    In short: The file format includes the trajectory data in a data set
    `trajectory` which contains the trajectory data, e.g., x, y, z coordinates,
    frame number and a person identifier. The dataset is additionally annotated
    with an attribute `fps` which gives the frame rate in which the data was
    recorded.

    Args:
        trajectory_file: trajectory file in Pedestrian Dynamics Data Archive
            HDF5 format

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
            of the file data
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    with h5py.File(trajectory_file, "r") as hdf5_file:
        # with pd.HDFStore(str(trajectory_file), mode="r") as store:
        dataset_name = "trajectory"
        if dataset_name not in hdf5_file:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"it does not contain a 'trajectory' dataset."
            )

        trajectory_dataset = hdf5_file[dataset_name]

        # pylint: disable-next=no-member
        column_names = trajectory_dataset.dtype.names

        if not {ID_COL, FRAME_COL, X_COL, Y_COL}.issubset(set(column_names)):
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"the 'trajectory' dataset does not contain the following "
                f"columns: '{ID_COL}', '{FRAME_COL}', '{X_COL}', and "
                f"'{Y_COL}'."
            )

        if "fps" not in trajectory_dataset.attrs:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"the 'trajectory' dataset does not contain a 'fps' attribute."
            )

        df_trajectory = pd.DataFrame(
            trajectory_dataset[:], columns=column_names
        )
        fps = trajectory_dataset.attrs["fps"]

    return TrajectoryData(data=df_trajectory, frame_rate=fps)


def load_walkable_area_from_ped_data_archive_hdf5(
    trajectory_file: pathlib.Path,
) -> WalkableArea:
    """Loads the walkable area from the hdf5 file as :class:`~geometry.WalkableArea`.

    Loads walkable area from files in the
    `Pedestrian Dynamics Data Archive <https://ped.fz-juelich.de/da/doku.php>`_
    HDF5 format. The format is explained in more detail
    `here <https://ped.fz-juelich.de/da/doku.php?id=info>`_.

    In short: The file format includes an attribute `wkt_geometry` at root
    level, which contains the walkable area of the experiments.

    Args:
        trajectory_file: trajectory file in Pedestrian Dynamics Data Archive
            HDF5 format

    Returns:
        WalkableArea: :class:`~geometry.WalkableArea` used in the experiment
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    with h5py.File(trajectory_file, "r") as hdf5_file:
        if "wkt_geometry" not in hdf5_file.attrs:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"it does not contain a 'wkt_geometry' attribute."
            )

        walkable_area = WalkableArea(hdf5_file.attrs["wkt_geometry"])

    return walkable_area


def load_trajectory_from_viswalk(
    *,
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from Viswalk-csv file as :class:`~trajectory_data.TrajectoryData`.

    This function reads a CSV file containing trajectory data from Viswalk
    simulations and converts it into a :class:`~trajectory_data.TrajectoryData`
    object which can be used for further analysis and processing in the
    *PedPy* framework.

    .. note::

        Viswalk data have a time column, that is going to be converted to a
        frame column for use with *PedPy*.

    .. warning::

        Currently only Viswalk files with a time column can be loaded.

    Args:
        trajectory_file: The full path of the CSV file containing the Viswalk
            trajectory data. The expected format is a CSV file with :code:`;`
            as delimiter, and it should contain at least the following
            columns: NO, SIMSEC, COORDCENTX, COORDCENTY. Comment lines may
            start with a :code:`*` and will be ignored.

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
            of the file data

    Raises:
        LoadTrajectoryError: If the provided path does not exist or is not a
            file.
    """  # noqa: E501
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_viswalk(
        trajectory_file=trajectory_file
    )
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(
        traj_dataframe
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


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

    fps = int(round(1 / mean_diff))
    frames = traj_dataframe["time"] * fps
    frames = frames.round().astype("int64")
    return frames, fps


def _load_trajectory_data_from_viswalk(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): The file containing the trajectory data.
            The expected format is a CSV file with ';' as delimiter, and it
            should contain at least the following columns: NO, SIMSEC,
            COORDCENTX, COORDCENTY. Comment lines may start with a '*' and
            will be ignored.

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        in meter (m).
    """
    columns_to_keep = ["NO", "SIMSEC", "COORDCENTX", "COORDCENTY"]
    rename_mapping = {
        "NO": ID_COL,
        "SIMSEC": "time",
        "COORDCENTX": X_COL,
        "COORDCENTY": Y_COL,
    }
    common_error_message = (
        "The given trajectory file seems to be incorrect or empty. "
        "It should contain at least the following columns: "
        "NO, SIMSEC, COORDCENTX, COORDCENTY, separated by ';'. "
        "Comment lines may start with a '*' and will be ignored. "
        f"Please check your trajectory file: {trajectory_file}."
    )
    try:
        data = pd.read_csv(
            trajectory_file,
            delimiter=";",
            skiprows=1,  # skip first row containing '$VISION'
            comment="*",
            dtype={
                ID_COL: "int64",
                "time": "float64",
                X_COL: "float64",
                Y_COL: "float64",
            },
            encoding="utf-8-sig",
        )
        got_columns = data.columns
        cleaned_columns = got_columns.map(
            lambda x: x.replace("$PEDESTRIAN:", "")
        )
        set_columns_to_keep = set(columns_to_keep)
        set_cleaned_columns = set(cleaned_columns)
        missing_columns = set_columns_to_keep - set_cleaned_columns
        if missing_columns:
            raise LoadTrajectoryError(
                f"{common_error_message}"
                f"Missing columns: {', '.join(missing_columns)}."
            )

        data.columns = cleaned_columns
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)

        if data.empty:
            raise LoadTrajectoryError(common_error_message)

        return data
    except pd.errors.ParserError as exc:
        raise LoadTrajectoryError(common_error_message) from exc


def load_trajectory_from_vadere(
    *,
    trajectory_file: pathlib.Path,
    frame_rate: float,
    ignore_too_short_trajectories: bool = False,
) -> TrajectoryData:
    """Loads trajectory from Vadere-traj file.

    This function reads a traj file containing trajectory data from Vadere
    simulations and converts it into a :class:`~trajectory_data.TrajectoryData`
    object which can be used for further analysis and processing in *PedPy*.

    Args:
        trajectory_file: Full path of the trajectory file
            containing Vadere trajectory data. The expected
            format is a traj file with space character as
            delimiter, and it should contain
            the following columns: pedestrianId, simTime (in sec),
            startX (in m), startY (in m).
            Additional columns (e.g. endTime, endX, endY, targetId)
            will be ignored.
        frame_rate: Frame rate in frames per second.
        ignore_too_short_trajectories: If False (default), the operation will
            abort when a trajectory is detected to be too short.
            If True, such trajectories will be ignored and a warning logged.

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
        of the file data

    Raises:
        LoadTrajectoryError: If the provided path does not exist,
        is not a file or has fissing columns.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_vadere(
        trajectory_file=trajectory_file
    )

    traj_dataframe = _event_driven_traj_to_const_frame_rate(
        traj_dataframe=traj_dataframe,
        frame_rate=frame_rate,
        ignore_too_short_trajectories=ignore_too_short_trajectories,
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=frame_rate,
    )


def _load_trajectory_data_from_vadere(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): The full path of the trajectory file
        containing the Vadere trajectory data. The expected format is a
        traj file with space character as delimiter, and it should contain
        the following columns: pedestrianId, simTime (in sec),
        startX (in m), startY (in m).
        Additional columns (e.g. endTime, endX, endY, targetId)
        will be ignored.

    Returns:
        The trajectory data as :class:`DataFrame`,
        the coordinates are in meter (m).
    """
    vadere_comment = "#"  # Comment identifier in Vadere trajectory files
    vadere_key_id = "pedestrianId"
    vadere_key_time = "simTime"
    vadere_key_x = "startX"
    vadere_key_y = "startY"
    columns_to_keep = [
        vadere_key_id,
        vadere_key_time,
        vadere_key_x,
        vadere_key_y,
    ]
    name_mapping = {
        vadere_key_id: ID_COL,
        vadere_key_time: TIME_COL,
        vadere_key_x: X_COL,
        vadere_key_y: Y_COL,
    }

    common_error_message = (
        "The given trajectory file seems to be incorrect or empty. "
        "One possible reason is that it was exported using a version "
        "of Vadere prior to 2.0. "
        "It should contain the following columns, which should be "
        f"uniquely identifiably by: {', '.join(columns_to_keep)}. "
        f"Columns should be separated by a space character. "
        f"Comment lines may start with '{vadere_comment}' and will be ignored. "
        f"Please check your trajectory file: {trajectory_file}."
    )
    try:
        vadere_cols = list(
            pd.read_csv(
                trajectory_file, comment=vadere_comment, delimiter=" ", nrows=1
            ).columns
        )
        use_vadere_cols = []
        non_unique_cols = {}
        missing_cols = []
        rename_mapping = {}

        for col in columns_to_keep:
            matching = [vc for vc in vadere_cols if col in vc]
            if len(matching) == 1:
                use_vadere_cols += matching
                rename_mapping[matching[0]] = name_mapping[col]
            elif len(matching) > 1:
                non_unique_cols[col] = matching
            elif len(matching) == 0:
                missing_cols += [col]

        if non_unique_cols:
            raise LoadTrajectoryError(
                f"{common_error_message} "
                + ". ".join(
                    [
                        "The identifier '{0}' is non-unique. "
                        "It is contained in the columns: {1}".format(
                            k, ", ".join(v)
                        )
                        for k, v in non_unique_cols.items()
                    ]
                )
                + "."
            )

        if missing_cols:
            raise LoadTrajectoryError(
                f"{common_error_message} "
                f"Missing columns: {', '.join(missing_cols)}."
            )

        data = pd.read_csv(
            trajectory_file,
            delimiter=" ",
            usecols=use_vadere_cols,
            comment="#",
            dtype={
                vadere_key_id: "int64",
                vadere_key_time: "float64",
                vadere_key_x: "float64",
                vadere_key_y: "float64",
            },
            encoding="utf-8-sig",
        )

        data = data.rename(columns=rename_mapping)

        if data.empty:
            raise LoadTrajectoryError(common_error_message)

        return data
    except pd.errors.ParserError as exc:
        raise LoadTrajectoryError(common_error_message) from exc


def _event_driven_traj_to_const_frame_rate(
    traj_dataframe: pd.DataFrame,
    frame_rate: float,
    ignore_too_short_trajectories: bool,
) -> pd.DataFrame:
    """Interpolate trajectory data linearly for non-equidistant time steps.

    Args:
        traj_dataframe: trajectory data as :class:`DataFrame`
        frame_rate: Frame rate in frames per second.
        ignore_too_short_trajectories: If False (default), the operation will
            abort when a trajectory is detected to be too short.
            If True, such trajectories will be ignored and a warning logged.

    Returns:
        The trajectory data as :class:`DataFrame` with positions x and y being
        linearly interpolated for frames between two recorded time steps.
    """
    _validate_is_deviation_vadere_pedpy_traj_transform_below_threshold(
        traj_dataframe, frame_rate
    )

    trajectory_too_short_messages = []
    traj_dataframe = traj_dataframe.set_index(TIME_COL)
    traj_by_ped = traj_dataframe.groupby(ID_COL)
    traj_dataframe_interpolated = pd.DataFrame()
    for ped_id, traj in traj_by_ped:
        t = traj.index
        t_start = traj.index.to_numpy().min()
        t_stop = traj.index.to_numpy().max()

        # Round t_start up (t_stop down) to nearest multiple of
        # frame period (= 1/frame_rate) to avoid extrapolation of trajectories
        # to times before first (after last) pedestrian step.
        precision = 14
        t_start_ = (
            math.ceil(np.round(t_start * frame_rate, precision)) / frame_rate
        )
        t_stop_ = (
            math.floor(np.round(t_stop * frame_rate, precision)) / frame_rate
        )

        if t_start == t_stop:
            msg = (
                f"Trajectory of pedestrian {ped_id} is too short in time "
                f"to be captured by the chosen frame rate of {frame_rate}. "
                f"Therefore, this trajectory will be ignored."
            )
            if ignore_too_short_trajectories:
                _log.warning(msg)
                continue
            else:
                trajectory_too_short_messages.append(msg)
        else:
            equidist_time_steps = np.linspace(
                start=t_start_,
                stop=t_stop_,
                num=int(np.round((t_stop_ - t_start_) * frame_rate, 0)) + 1,
                endpoint=True,
            )

            r = pd.Index(equidist_time_steps, name=t.name)
            interpolated_traj = (
                traj.reindex(t.union(r)).interpolate(method="index").loc[r]
            )

            interpolated_traj[ID_COL] = interpolated_traj[ID_COL].astype(int)

            traj_dataframe_interpolated = pd.concat(
                [traj_dataframe_interpolated, interpolated_traj]
            )

    if trajectory_too_short_messages and not ignore_too_short_trajectories:
        raise LoadTrajectoryError(
            "One or more pedestrian trajectories are too short to be captured "
            f"at frame rate {frame_rate}:\n- "
            + "\n- ".join(trajectory_too_short_messages)
        )

    if traj_dataframe_interpolated.empty:
        raise LoadTrajectoryError("No valid trajectories were captured.")

    traj_dataframe_interpolated = traj_dataframe_interpolated.reset_index()

    traj_dataframe_interpolated[FRAME_COL] = (
        (traj_dataframe_interpolated[TIME_COL] * frame_rate)
        .round(decimals=0)
        .astype(int)
    )
    traj_dataframe_interpolated = traj_dataframe_interpolated.drop(
        labels=TIME_COL, axis="columns"
    )

    traj_dataframe_interpolated = traj_dataframe_interpolated.sort_values(
        by=[FRAME_COL, ID_COL], ignore_index=True
    )
    return traj_dataframe_interpolated


def _validate_is_deviation_vadere_pedpy_traj_transform_below_threshold(
    traj_dataframe: pd.DataFrame,
    frame_rate: float,
    deviation_threshold: float = 0.1,
) -> None:
    """Validates the maximum deviation.

    Validates whether the maximum deviation between event-based vadere
    trajectories and their interpolated version with fixed frames is
    below given threshold.

    Max difference occurs when first (last) step of a trajectory happens
    just after (before) the last (next) frame.
    Example for an agent that moves with a certain speed, s:
        First frame at t_f1, second frame at t_f2 = t_f1 + 1 / frame_rate
        First step at t_s1 = t_f1 + t_offset
        Distance walked between t_s1 and t_f2 will not be captured:
        x_s1f2 =  s * (1 / frame_rate - t_offset)
        with t_offset --> 0s: x_s1f2 = s / frame_rate

    Args:
        traj_dataframe: trajectory data as :class:`DataFrame`
        frame_rate: Frame rate in frames per second.
        deviation_threshold: acceptable max. difference in meter (m),
                             otherwise log warning
    """
    traj_groups = traj_dataframe.groupby(ID_COL)

    max_speed = 0  # max pedestrian speed that actually reads from the traj file
    for _, traj in traj_groups:
        diff = traj.diff().dropna()
        dx_dt = (np.sqrt(diff[[X_COL, Y_COL]].pow(2).sum(axis=1))).divide(
            diff[TIME_COL]
        )
        max_speed = max([max_speed, round(max(dx_dt), 2)])

    max_deviation = round(max_speed / frame_rate, 2)
    if max_deviation > deviation_threshold:
        _log.warning(
            f"The interpolated trajectory potentially deviates up to "
            f"{max_deviation!s} m from the original trajectory, at least "
            f"for the fastest pedestrian with max. speed of {max_speed!s} m/s. "
            f"If smaller deviations are required, choose a higher frame rate. "
            f"The current frame rate is {frame_rate!s} fps."
        )


def load_walkable_area_from_vadere_scenario(
    vadere_scenario_file: pathlib.Path,
    margin: float = 0,
    decimals: int = 6,
) -> WalkableArea:
    """Loads walkable area from Vadere scenario file.

    Loads walkable area from Vadere scenario file
    as :class:`~geometry.WalkableArea`.

    Args:
        vadere_scenario_file: Vadere scenario file (json format)
        margin: Increases the walkable area by the value of margin to avoid
                that the topography bound touches obstacles because shapely
                Polygons used in PedPy do not allow this.
                By default (margin = .0), the bound of the walkable area in
                PedPy coincides with the inner bound of
                the bounding box (obstacle) in Vadere.
                PedPy cannot process the case where obstacles touch the
                bounding box defined in Vadere. To avoid errors, either
                increase the value of margin (e.g. to 1e-3) or make
                sure that the obstacles in Vadere do not touch the
                bounding box.
        decimals: Integer defining the decimals of the coordinates
                  of the walkable area

    Returns:
        WalkableArea: :class:`~geometry.WalkableArea` used in the simulation
    """
    _validate_is_file(vadere_scenario_file)

    if margin != 0 and margin < 10**-decimals:
        raise LoadTrajectoryError(
            f"Margin ({margin!s}) should be greater than 10 ** (-{decimals!s})."
        )

    with open(vadere_scenario_file, "r") as f:
        data = json.load(f)
        topography = data["scenario"]["topography"]
        scenario_attributes = topography["attributes"]

        # bound
        complete_area = scenario_attributes["bounds"]
        bounding_box_with = scenario_attributes["boundingBoxWidth"]
        complete_area["x"] = complete_area["x"] + bounding_box_with - margin
        complete_area["y"] = complete_area["y"] + bounding_box_with - margin
        complete_area["width"] = complete_area["width"] - 2 * (
            bounding_box_with - margin
        )
        complete_area["height"] = complete_area["height"] - 2 * (
            bounding_box_with - margin
        )
        complete_area["type"] = "RECTANGLE"
        complete_area_points = _vadere_shape_to_point_list(
            complete_area, decimals=decimals
        )
        area_poly = shapely.Polygon(complete_area_points)

        # obstacles
        walkable_area_poly = Polygon(area_poly.exterior.coords)
        obstacles = topography["obstacles"]
        error_obst_ids = []
        for obstacle in obstacles:
            obst_points = _vadere_shape_to_point_list(
                obstacle["shape"], decimals=decimals
            )
            obstacle_polygon = shapely.Polygon(obst_points)
            if area_poly.contains_properly(obstacle_polygon):
                walkable_area_poly = walkable_area_poly.difference(
                    obstacle_polygon
                )
            else:
                error_obst_ids += [str(obstacle["id"])]

        if error_obst_ids:
            error_obst_ids = list({", ".join(error_obst_ids)})
            raise LoadTrajectoryError(
                f"Cannot convert obstacles with IDs {error_obst_ids} because "
                f"they touch the bound of the walkable area (inner bound of "
                f"the bounding box in Vadere). Increase the walkable area by "
                f"adjusting 'margin' or adapt the scenario file to make "
                f"sure that obstacles have no common points with the bounding "
                f"box."
            )

    return WalkableArea(walkable_area_poly)


def _vadere_shape_to_point_list(
    shape: dict[str, Any], decimals: int
) -> list[Point]:
    """Transforms dict describing a rectangle or polygon into a list of points.

    Args:
        shape: Dict containing the shape as RECTANGLE or POLYGON
               * 'shape' RECTANGLE requires key value pairs
                for 'x', 'y', 'width', 'height'
               * 'shape' POLYGON requires key value pair
               for 'points': [{'x': ..., 'y': ...},
               {'x': ..., 'y': ...}, ...]

        decimals: Integer defining the decimals of the returned coordinates

    Returns:
        list[Point]

    """
    _supported_types = ["RECTANGLE", "POLYGON"]

    shape_type = shape["type"]
    if shape_type not in _supported_types:
        raise LoadTrajectoryError(
            f"The given Vadere scenario contains an unsupported obstacle "
            f"shape '{shape_type}'. "
        )

    if shape_type == "RECTANGLE":
        # lower left corner (x1, y1)
        x1 = shape["x"]
        y1 = shape["y"]

        # upper right corner (x2, y2)
        x2 = x1 + shape["width"]
        y2 = y1 + shape["height"]

        points = [
            shapely.Point(x1, y1),
            shapely.Point(x2, y1),
            shapely.Point(x2, y2),
            shapely.Point(x1, y2),
        ]

    elif shape_type == "POLYGON":
        points = [shapely.Point(p["x"], p["y"]) for p in shape["points"]]

    # handle floating point errors
    points = [
        shapely.Point(np.round(p.x, decimals), np.round(p.y, decimals))
        for p in points
    ]
    return points


def load_trajectory_from_crowdit(
    *,
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from Crowdit file as :class:`~trajectory_data.TrajectoryData`.

    This function reads a CSV file containing trajectory data from Crowdit
    simulations and converts it into a :class:`~trajectory_data.TrajectoryData`
    object.

    Args:
        trajectory_file: The full path of the CSV file containing the Crowdit
            trajectory data. The expected format is a CSV file with comma
            as delimiter, and it should contain at least the following
            columns: pedID, time, posX, posY.

    Returns:
        TrajectoryData: :class:`~trajectory_data.TrajectoryData` representation
            of the file data

    Raises:
        LoadTrajectoryError: If the provided path does not exist or is not a
            file.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_crowdit(
        trajectory_file=trajectory_file
    )
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(
        traj_dataframe
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_crowdit(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
    """Parse the Crowdit trajectory file for trajectory data.

    Args:
        trajectory_file: The file containing the trajectory data.
            The expected format is a CSV file with comma as delimiter, and it
            should contain at least the following columns:
            pedID, time, posX, posY.

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        in meter (m).
    """
    columns_to_keep = ["pedID", "time", "posX", "posY"]
    rename_mapping = {
        "pedID": ID_COL,
        "time": TIME_COL,
        "posX": X_COL,
        "posY": Y_COL,
    }
    column_types = {"id": int, "time": float, "x": float, "y": float}

    common_error_message = (
        "The given trajectory file seems to be incorrect or empty. "
        "It should contain at least the following columns: "
        "pedID, time, posX, posY, separated by comma. "
        f"Please check your trajectory file: {trajectory_file}."
    )

    try:
        data = pd.read_csv(
            trajectory_file,
            encoding="utf-8-sig",
        ).dropna()
    except Exception as e:
        raise LoadTrajectoryError(
            f"{common_error_message}\nOriginal error: {e}"
        ) from e

    missing_columns = set(columns_to_keep) - set(data.columns)
    if missing_columns:
        raise LoadTrajectoryError(
            f"{common_error_message}"
            f"Missing columns: {', '.join(missing_columns)}."
        )

    try:
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)
        data = data.astype(column_types)
    except Exception as e:
        raise LoadTrajectoryError(
            f"{common_error_message}\nOriginal error: {e}"
        ) from e

    if data.empty:
        raise LoadTrajectoryError(common_error_message)

    return data


def load_walkable_area_from_crowdit(
    *, geometry_file: pathlib.Path, buffer: float = 1e-3
) -> WalkableArea:
    """Load walkable area from a Crowdit XML geometry file.

    Args:
        geometry_file: Path to the XML geometry file.
        buffer: Optional padding around the bounding box to avoid
            overlap with obstacles.

    Returns:
        WalkableArea: representation of the walkable area.
    """
    _validate_is_file(geometry_file)

    try:
        tree = ET.parse(geometry_file)
        root = tree.getroot()
    except Exception as e:
        raise LoadTrajectoryError(
            f"Could not parse Crowdit geometry file: {geometry_file}\n"
            f"Original error: {e}"
        ) from e

    all_points = []
    walls = []

    # Get walls from all layers, ignore WunderZone
    for layer in root.findall("layer"):
        for geom in layer.findall("wall"):
            points = []
            for pt in geom.findall("point"):
                x = pt.get("x")
                y = pt.get("y")
                if x is None or y is None:
                    raise LoadTrajectoryError(
                        f"Invalid point found in {geometry_file}."
                        "missing x or y attribute"
                    )
                points.append((float(x), float(y)))
            if points:
                if points[0] != points[-1]:
                    points.append(points[0])
                all_points.extend(points)
                walls.append(points)

    if not all_points:
        raise LoadTrajectoryError(
            f"No wall polygons found in Crowdit geometry file: {geometry_file}"
        )

    # Exception for single wall → directly as WalkableArea
    if len(walls) == 1:
        return WalkableArea(polygon=walls[0])

    # Normal case: Bounding Box + all walls as obstacles
    xs, ys = zip(*all_points, strict=True)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    outer = [
        (minx - buffer, miny - buffer),
        (maxx + buffer, miny - buffer),
        (maxx + buffer, maxy + buffer),
        (minx - buffer, maxy + buffer),
        (minx - buffer, miny - buffer),
    ]

    return WalkableArea(polygon=outer, obstacles=walls)
