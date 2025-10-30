"""Load crowd:it trajectories to the internal trajectory data format."""

import pathlib
import xml.etree.ElementTree as ET

import pandas as pd

from pedpy.column_identifier import FRAME_COL, ID_COL, TIME_COL, X_COL, Y_COL
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import LoadTrajectoryError
from pedpy.io.helper import _calculate_frames_and_fps, _validate_is_file


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
        LoadTrajectoryError: If the provided path does not exist or is not a file.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_crowdit(trajectory_file=trajectory_file)
    traj_dataframe["frame"], traj_frame_rate = _calculate_frames_and_fps(traj_dataframe)

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=traj_frame_rate,
    )


def _load_trajectory_data_from_crowdit(*, trajectory_file: pathlib.Path) -> pd.DataFrame:
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
        raise LoadTrajectoryError(f"{common_error_message}\nOriginal error: {e}") from e

    missing_columns = set(columns_to_keep) - set(data.columns)
    if missing_columns:
        raise LoadTrajectoryError(f"{common_error_message}Missing columns: {', '.join(missing_columns)}.")

    try:
        data = data[columns_to_keep]
        data = data.rename(columns=rename_mapping)
        data = data.astype(column_types)
    except Exception as e:
        raise LoadTrajectoryError(f"{common_error_message}\nOriginal error: {e}") from e

    if data.empty:
        raise LoadTrajectoryError(common_error_message)

    return data


def load_walkable_area_from_crowdit(*, geometry_file: pathlib.Path, buffer: float = 1e-3) -> WalkableArea:
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
        raise LoadTrajectoryError(f"Could not parse Crowdit geometry file: {geometry_file}\nOriginal error: {e}") from e

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
                    raise LoadTrajectoryError(f"Invalid point found in {geometry_file}.missing x or y attribute")
                points.append((float(x), float(y)))
            if points:
                if points[0] != points[-1]:
                    points.append(points[0])
                all_points.extend(points)
                walls.append(points)

    if not all_points:
        raise LoadTrajectoryError(f"No wall polygons found in Crowdit geometry file: {geometry_file}")

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
