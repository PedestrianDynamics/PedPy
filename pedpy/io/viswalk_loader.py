"""Load Viswalk trajectories to the internal trajectory data format."""

import pathlib

import pandas as pd

from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import LoadTrajectoryError
from pedpy.io.helper import _calculate_frames_and_fps, _validate_is_file


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
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_viswalk(trajectory_file=trajectory_file)
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(traj_dataframe)

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_viswalk(*, trajectory_file: pathlib.Path) -> pd.DataFrame:
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
        cleaned_columns = got_columns.map(lambda x: x.replace("$PEDESTRIAN:", ""))
        set_columns_to_keep = set(columns_to_keep)
        set_cleaned_columns = set(cleaned_columns)
        missing_columns = set_columns_to_keep - set_cleaned_columns
        if missing_columns:
            raise LoadTrajectoryError(f"{common_error_message}Missing columns: {', '.join(missing_columns)}.")

        data.columns = cleaned_columns
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)

        if data.empty:
            raise LoadTrajectoryError(common_error_message)

        return data
    except pd.errors.ParserError as exc:
        raise LoadTrajectoryError(common_error_message) from exc
