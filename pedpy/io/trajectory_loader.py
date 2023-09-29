"""Load trajectories to the internal trajectory data format."""

import pathlib
import sqlite3
from typing import Any, Optional, Tuple

import pandas as pd
from aenum import Enum

from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData


class LoadTrajectoryError(Exception):
    """Class reflecting errors when loading trajectories with PedPy."""

    def __init__(self, message):
        """Create LoadTrajectoryError with the given message.

        Args:
            message: Error message
        """
        self.message = message


class TrajectoryUnit(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of the unit of the trajectory coordinates."""

    _init_ = "value __doc__"
    METER = 1, "meter (m)"
    CENTIMETER = 100, "centimeter (cm)"


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
        :class:`~trajectory_data.TrajectoryData` representation of the file data
    """
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
        :class:`~trajectory_data.TrajectoryData` representation of the file data
    """
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


def _calculate_frames_and_fps(
    traj_dataframe: pd.DataFrame,
) -> Tuple[pd.Series, int]:
    """Calculates fps and frames based on the time column of the dataframe."""
    mean_diff = traj_dataframe.groupby(ID_COL)["time"].diff().dropna().mean()
    fps = int(round(1 / mean_diff))
    frames = (traj_dataframe["time"] * fps).astype(int)
    return frames, fps


def _load_viswalk_trajectory_data(
    *, trajectory_file: pathlib.Path
) -> pd.DataFrame:
    """Parse the trajectory file for trajectory data.

    Args:
        trajectory_file (pathlib.Path): file containing the trajectory
        Unit is always in meter.

    Returns:
        The trajectory data as :class:`DataFrame`, the coordinates are
        converted to meter (m).
    """
    columns_to_keep = ["NO", "SIMSEC", "COORDCENTX", "COORDCENTY"]
    rename_mapping = {
        "NO": ID_COL,
        "SIMSEC": "time",
        "COORDCENTX": X_COL,
        "COORDCENTY": Y_COL,
    }
    try:
        data = pd.read_csv(
            trajectory_file,
            delimiter=";",
            skiprows=1,
            dtype={
                ID_COL: "int64",
                "time": "float64",
                X_COL: "float64",
                Y_COL: "float64",
                "COORDCENT": "float64",
            },
        )
        got_columns = data.columns
        cleaned_columns = got_columns.map(
            lambda x: x.replace("$PEDESTRIAN:", "")
        )
        data.columns = cleaned_columns
        data = data[columns_to_keep]
        data.rename(columns=rename_mapping, inplace=True)
        set_columns_to_keep = set(columns_to_keep)
        set_cleaned_columns = set(cleaned_columns)
        missing_columns = set_columns_to_keep - set_cleaned_columns
        if missing_columns:
            error_message = (
                f"The following columns in {columns_to_keep} are not present"
                f"in {cleaned_columns}: {', '.join(missing_columns)}."
                f"Please check your trajectory file: {trajectory_file}."
            )
            raise ValueError(error_message)

        if data.empty:
            raise ValueError(
                "The given trajectory file seem to be empty. It should "
                "contain at least 4 columns: NO, SIMSEC, COORDCENTX, COORDCENTY."
                "The values should be separated by ';'. Comment line may "
                "start with a '*' and will be ignored."
                f"Please check your trajectory file: {trajectory_file}."
            )

        return data
    except pd.errors.ParserError as exc:
        raise ValueError(
            "The given trajectory file seem to be empty. It should "
            "contain at least 4 columns: NO, SIMSEC, COORDCENTX, COORDCENTY."
            "The values should be separated by ';'. Comment line may "
            "start with a '*' and will be ignored."
            f"Please check your trajectory file: {trajectory_file}."
        ) from exc


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
            raise ValueError(
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
        raise ValueError(
            "The given trajectory file could not be parsed. It should "
            "contain at least 5 columns: ID, frame, X, Y, Z. The values "
            "should be separated by any white space. Comment line may start "
            "with a '#' and will be ignored. "
            f"Please check your trajectory file: {trajectory_file}."
        ) from exc


def _load_trajectory_meta_data_from_txt(  # pylint: disable=too-many-branches
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

    with open(trajectory_file, "r", encoding="utf-8") as file_content:
        for line in file_content:
            if not line.startswith("#"):
                break

            if "framerate" in line:
                for substring in line.split():
                    try:
                        if parsed_frame_rate is None:
                            parsed_frame_rate = float(substring)
                    except ValueError:
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
            "Frame rate is needed, but none could be found in the trajectory "
            "file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            "a default frame rate."
        )

    if parsed_frame_rate is not None and default_frame_rate is None:
        if parsed_frame_rate <= 0:
            raise ValueError(
                "Frame rate needs to be a positive value, but is "
                f"{parsed_frame_rate}. "
                "Please check your trajectory file: {trajectory_file}."
            )
    if parsed_frame_rate is not None and default_frame_rate is not None:
        if parsed_frame_rate != default_frame_rate:
            raise ValueError(
                "The given default frame rate seems to differ from the frame "
                "rate given in the trajectory file: "
                f"{default_frame_rate} != {parsed_frame_rate}"
            )

    unit = parsed_unit
    if parsed_unit is None and default_unit is not None:
        unit = default_unit

    if parsed_unit is None and default_unit is None:
        raise ValueError(
            "Unit is needed, but none could be found in the trajectory file. "
            f"Please check your trajectory file: {trajectory_file} or provide "
            "a default unit."
        )

    if parsed_unit is not None and default_unit is not None:
        if parsed_unit != default_unit:
            raise ValueError(
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
        :class:`~trajectory_data.TrajectoryData` representation of the file data
    """
    _validate_is_file(trajectory_file)

    with sqlite3.connect(trajectory_file) as con:
        data = pd.read_sql_query(
            "select frame, id, pos_x as x, pos_y as y from trajectory_data",
            con,
        )
        fps = float(
            con.cursor()
            .execute("select value from metadata where key = 'fps'")
            .fetchone()[0]
        )
    return TrajectoryData(data=data, frame_rate=fps)


def load_walkable_area_from_jupedsim_sqlite(
    trajectory_file: pathlib.Path,
) -> WalkableArea:
    """Loads the walkable area from the sqlite file as :class:`~geometry.WalkableArea`.

    Args:
        trajectory_file: trajectory file in JuPedSim sqlite format

    Returns:
        :class:`~geometry.WalkableArea` used in the simulation
    """
    _validate_is_file(trajectory_file)

    with sqlite3.connect(trajectory_file) as con:
        walkable_area = (
            con.cursor().execute("select wkt from geometry").fetchone()[0]
        )
    return WalkableArea(walkable_area)


def load_trajectory_from_ped_data_archive_hdf5(
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads data from the hdf5 file as :class:`~trajectory_data.TrajectoryData`.

    Loads data from files in the
    `Pedestrian Dynamics Data Archive <https://ped.fz-juelich.de/da/doku.php>`_ HDF5
    format. The format is explained in more detail
    `here <https://ped.fz-juelich.de/da/doku.php?id=info>`_.

    In short: The file format includes the trajectory data in a data set `trajectory` which
    contains the trajectory data, e.g., x, y, z coordinates, frame number and a person identifier.
    The dataset is additionally annotated with an attribute `fps` which gives the frame rate in
    which the data was recorded.

    Args:
        trajectory_file: trajectory file in Pedestrian Dynamics Data Archive HDF5 format

    Returns:
        :class:`~trajectory_data.TrajectoryData` representation of the file data
    """
    _validate_is_file(trajectory_file)

    with pd.HDFStore(str(trajectory_file), mode="r") as store:
        if store.get_node("trajectory") is None:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"it does not contain a 'trajectory' dataset."
            )
        df_trajectory = store["trajectory"]

        if not (
            [ID_COL, FRAME_COL, X_COL, Y_COL] == df_trajectory.columns[:4]
        ).all():
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"the 'trajectory' dataset does not contain the following columns: "
                f"'{ID_COL}', '{FRAME_COL}', '{X_COL}', and '{Y_COL}'."
            )

        if "fps" not in store.get_storer("trajectory").attrs:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"the 'trajectory' dataset does not contain a 'fps' attribute."
            )
        fps = store.get_storer("trajectory").attrs.fps

    return TrajectoryData(data=df_trajectory, frame_rate=fps)


def load_walkable_area_from_ped_data_archive_hdf5(
    trajectory_file: pathlib.Path,
) -> WalkableArea:
    """Loads the walkable area from the hdf5 file as :class:`~geometry.WalkableArea`.

    Loads walkable area from files in the
    `Pedestrian Dynamics Data Archive <https://ped.fz-juelich.de/da/doku.php>`_ HDF5
    format. The format is explained in more detail
    `here <https://ped.fz-juelich.de/da/doku.php?id=info>`_.

    In short: The file format includes an attribute `wkt_geometry` at root level, which contains
    the walkable area of the experiments.

    Args:
        trajectory_file: trajectory file in Pedestrian Dynamics Data Archive HDF5 format

    Returns:
        :class:`~geometry.WalkableArea` used in the experiment
    """
    _validate_is_file(trajectory_file)

    with pd.HDFStore(str(trajectory_file), mode="r") as store:
        # pylint: disable=protected-access
        if "wkt_geometry" not in store._handle.get_node("/")._v_attrs:
            raise LoadTrajectoryError(
                f"{trajectory_file} seems to be not a supported hdf5 file, "
                f"it does not contain a 'wkt_geometry' attribute."
            )

        walkable_area = WalkableArea(
            store._handle.get_node("/")._v_attrs["wkt_geometry"]
        )
        # pylint: enable=protected-access

    return walkable_area


def load_viswalk_trajectories(
    *,
    trajectory_file: pathlib.Path,
) -> TrajectoryData:
    """Loads trajectory data from a Viswalk-CSV file and returns a PedPy.TrajectoryData object.

    This function reads a CSV file containing trajectory data from Viswalk simulations and
    converts it into a PedPy TrajectoryData object which can be used for further analysis and
    processing in the PedPy framework.

    Note: viswalk data have a time column, that is going to be converted to
          a frame column for use with PedPy.

    Args:
    filename (str): The full path of the CSV file containing the Viswalk
                    trajectory data.
                    The file should be formatted with the following columns:
                    'id', 'time', 'x', 'y', 'COORDCENT'.

    Returns:
    pedpy.TrajectoryData: A PedPy TrajectoryData object containing the loaded
                          trajectory data


    See Also:
    https://pedpy.readthedocs.io/ for more information on the PedPy library
    and TrajectoryData object.
    """
    if not trajectory_file.exists():
        raise IOError(f"{trajectory_file} does not exist.")

    if not trajectory_file.is_file():
        raise IOError(f"{trajectory_file} is not a file.")

    traj_dataframe = _load_viswalk_trajectory_data(
        trajectory_file=trajectory_file
    )
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(
        traj_dataframe
    )

    return TrajectoryData(data=traj_dataframe, frame_rate=traj_frame_rate)
