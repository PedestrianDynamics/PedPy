"""Helper functions for the analysis methods"""
from typing import Tuple

import numpy as np
import pandas as pd
import shapely
from shapely import LineString, Polygon

from pedpy import Geometry, TrajectoryData


def is_trajectory_valid(*, traj: TrajectoryData, geometry: Geometry) -> bool:
    """Checks if all trajectory data points lie within the given geometry

    Args:
        traj (TrajectoryData): trajectory data
        geometry (Geometry): geometry

    Returns:
        All points lie within geometry
    """
    return get_invalid_trajectory(traj=traj, geometry=geometry).empty


def get_invalid_trajectory(
    *, traj: TrajectoryData, geometry: Geometry
) -> pd.DataFrame:
    """Returns all trajectory data points outside the given geometry

    Args:
        traj (TrajectoryData): trajectory data
        geometry (Geometry): geometry

    Returns:
        DataFrame showing all data points outside the given geometry
    """
    return traj.data.loc[
        ~shapely.within(traj.data.points, geometry.walkable_area)
    ]


def compute_frame_range_in_area(
    *, traj_data: pd.DataFrame, measurement_line: LineString, width: float
) -> Tuple[pd.DataFrame, Polygon]:
    """Compute the frame ranges for each pedestrian inside the measurement area.

    Note:
        Only pedestrians passing the complete measurement area will be
        considered. Meaning they need to cross measurement_line and the line
        with the given offset in one go. If leaving the area between two lines
         through the same line will be ignored.

        As passing we define the frame the pedestrians enter the area and
        then moves through the complete area without leaving it. Hence,
        doing a closed analysis of the movement area with several measuring
        ranges underestimates the actual movement time.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_line (shapely.LineString): measurement line
        width (float): distance to the second measurement line

    Returns:
        DataFrame containing the columns: 'ID', 'frame_start', 'frame_end' and
        the created measurement area
    """
    assert len(shapely.get_coordinates(measurement_line)) == 2, (
        f"The measurement line needs to be a straight line but has more  "
        f"coordinates (expected 2, got "
        f"{len(shapely.get_coordinates(measurement_line))})"
    )

    # Create the second with the given offset
    second_line = shapely.offset_curve(measurement_line, distance=width)

    # Reverse the order of the coordinates for the second line string to
    # create a rectangular area between the lines
    measurement_area = Polygon(
        [
            *measurement_line.coords,
            *second_line.coords[::-1],
        ]
    )

    inside_range = _get_continuous_parts_in_area(traj_data, measurement_area)

    crossing_frames_first = _compute_crossing_frames(
        traj_data, measurement_line
    )
    crossing_frames_second = _compute_crossing_frames(traj_data, second_line)

    start_crossed_1 = _check_crossing_in_frame_range(
        inside_range, crossing_frames_first, "frame_start", "start_crossed_1"
    )
    end_crossed_1 = _check_crossing_in_frame_range(
        inside_range, crossing_frames_first, "frame_end", "end_crossed_1"
    )
    start_crossed_2 = _check_crossing_in_frame_range(
        inside_range, crossing_frames_second, "frame_start", "start_crossed_2"
    )
    end_crossed_2 = _check_crossing_in_frame_range(
        inside_range, crossing_frames_second, "frame_end", "end_crossed_2"
    )

    frame_range_between_lines = (
        start_crossed_1.merge(
            start_crossed_2, how="outer", on=["ID", "frame_start", "frame_end"]
        )
        .merge(
            end_crossed_1, how="outer", on=["ID", "frame_start", "frame_end"]
        )
        .merge(
            end_crossed_2, how="outer", on=["ID", "frame_start", "frame_end"]
        )
    )

    frame_range_between_lines = frame_range_between_lines[
        (
            frame_range_between_lines.start_crossed_1
            & frame_range_between_lines.end_crossed_2
        )
        | (
            frame_range_between_lines.start_crossed_2
            & frame_range_between_lines.end_crossed_1
        )
    ]

    return (
        frame_range_between_lines.loc[:, ("ID", "frame_start", "frame_end")],
        measurement_area,
    )


def _compute_individual_movement(
    traj_data: pd.DataFrame, frame_step: int, bidirectional: bool = True
) -> pd.DataFrame:
    """Compute the individual movement in the time interval frame_step

    The movement is computed for the interval [frame - frame_step: frame +
    frame_step], if one of the boundaries is not contained in the trajectory
    frame will be used as boundary. Hence, the intervals become [frame,
    frame + frame_step], or [frame - frame_step, frame] respectively.

    Args:
        traj_data (pd.DataFrame): trajectory data
        frame_step (int): how many frames back and forwards are used to compute
            the movement.
        bidirectional (bool): if True also the prev. frame_step points will
            be used to determine the movement

    Returns:
        DataFrame containing the columns: 'ID', 'frame', 'start', 'end',
        'start_frame, and 'end_frame'. Where 'start'/'end' are the points
        where the movement start/ends, and 'start_frame'/'end_frame' are the
        corresponding frames.
    """
    df_movement = traj_data.copy(deep=True)

    df_movement["start"] = (
        df_movement.groupby("ID")["points"]
        .shift(frame_step)
        .fillna(df_movement["points"])
    )
    df_movement["start_frame"] = (
        df_movement.groupby("ID")["frame"]
        .shift(frame_step)
        .fillna(df_movement["frame"])
    )

    if bidirectional:
        df_movement["end"] = (
            df_movement.groupby("ID")["points"]
            .shift(-frame_step)
            .fillna(df_movement["points"])
        )
        df_movement["end_frame"] = (
            df_movement.groupby("ID")["frame"]
            .shift(-frame_step)
            .fillna(df_movement["frame"])
        )
    else:
        df_movement["end"] = df_movement["points"]
        df_movement["end_frame"] = df_movement["frame"]

    return df_movement[
        ["ID", "frame", "start", "end", "start_frame", "end_frame"]
    ]


def _compute_crossing_frames(
    traj_data: pd.DataFrame, measurement_line: LineString
) -> pd.DataFrame:
    """Compute the frames at which a pedestrian crosses a specific
    measurement line.

    As crossing we define a movement that moves across the measurement line.
    When the movement ends on the line, the line is not crossed. When it
    starts on the line, it counts as crossed.

    Note:
        Due to oscillations it may happen that a pedestrian crosses the
        measurement line multiple time in a small time interval.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_line (shapely.LineString):

    Returns:
        DataFrame containing the columns: 'ID', 'frame', where 'frame' are
        the frames where the measurement line is crossed.
    """
    # stack is used to get the coordinates in the correct order, as pygeos
    # does not support creating linestring from points directly. The
    # resulting array looks as follows:
    # [[[x_0_start, y_0_start], [x_0_end, y_0_end]],
    #  [[x_1_start, y_1_start], [x_1_end, y_1_end]], ... ]
    df_movement = _compute_individual_movement(traj_data, 1, False)
    df_movement["movement"] = shapely.linestrings(
        np.stack(
            [
                shapely.get_coordinates(df_movement["start"]),
                shapely.get_coordinates(df_movement["end"]),
            ],
            axis=1,
        )
    )

    # crossing means, the current movement crosses the line and the end point
    # of the movement is not on the line. The result is sorted by frame number
    crossing_frames = df_movement.loc[
        shapely.intersects(df_movement["movement"], measurement_line)
    ][["ID", "frame"]]

    return crossing_frames


def _get_continuous_parts_in_area(
    traj_data: pd.DataFrame, measurement_area: Polygon
) -> pd.DataFrame:
    """Returns the time-continuous parts in which the pedestrians are inside
    the given measurement area. As leaving the first frame outside the area is
    considered.

    Args:
        traj_data (pd.DataFrame): trajectory data
        measurement_area (shapely.Polygon): area which is considered

    Returns:
        DataFrame containing the columns: 'ID', 'frame_start', 'frame_end'
    """
    inside = traj_data.loc[
        shapely.within(traj_data.points, measurement_area), :
    ].copy()
    inside.loc[:, "g"] = inside.groupby("ID")["frame"].apply(
        lambda x: x.diff().ge(2).cumsum()
    )
    inside_range = (
        inside.groupby(["ID", "g"])
        .agg(
            frame_start=("frame", "first"),
            frame_end=("frame", "last"),
        )
        .reset_index()[["ID", "frame_start", "frame_end"]]
    )
    inside_range["frame_end"] += 1

    return inside_range


def _check_crossing_in_frame_range(
    inside_range: pd.DataFrame,
    crossing_frames: pd.DataFrame,
    check_column: str,
    column_name: str,
) -> pd.DataFrame:
    """Returns rows of inside_range which are also in crossing_frames.

    Args:
        inside_range (pd.DataFrame): DataFrame containing the columns 'ID'
            and check_column
        crossing_frames (pd.DataFrame): DataFrame containing the columns 'ID'
            and 'frame'
        check_column (str): name of the column in inside_range which represents
            a frame value. Needs to be 'frame_start' or 'frame_end'
        column_name (str): name of the result column

    Returns:
        DataFrame containing the columns 'ID', 'frame_start', 'frame_end',
        column_name
    """
    assert check_column in (
        "frame_start",
        "frame_end",
    ), "check_column needs to be 'frame_start' or 'frame_end'"

    crossed = pd.merge(
        inside_range,
        crossing_frames,
        left_on=["ID", check_column],
        right_on=["ID", "frame"],
        how="left",
        indicator=column_name,
    )[["ID", "frame_start", "frame_end", column_name]]
    crossed[column_name] = crossed[column_name] == "both"
    crossed = crossed[crossed[column_name]]
    return crossed
