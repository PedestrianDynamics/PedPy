"""Load Vadere trajectories to the internal trajectory data format."""

import json
import logging
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely import Point, Polygon

from pedpy.column_identifier import FRAME_COL, ID_COL, TIME_COL, X_COL, Y_COL
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import LoadTrajectoryError
from pedpy.io.helper import _validate_is_file

_log = logging.getLogger(__name__)


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
            is not a file or has missing columns.
    """
    _validate_is_file(trajectory_file)

    traj_dataframe = _load_trajectory_data_from_vadere(trajectory_file=trajectory_file)

    traj_dataframe = _event_driven_traj_to_const_frame_rate(
        traj_dataframe=traj_dataframe,
        frame_rate=frame_rate,
        ignore_too_short_trajectories=ignore_too_short_trajectories,
    )

    return TrajectoryData(
        data=traj_dataframe[[ID_COL, FRAME_COL, X_COL, Y_COL]],
        frame_rate=frame_rate,
    )


def _load_trajectory_data_from_vadere(*, trajectory_file: pathlib.Path) -> pd.DataFrame:
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
        vadere_cols = list(pd.read_csv(trajectory_file, comment=vadere_comment, delimiter=" ", nrows=1).columns)
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
                        "The identifier '{0}' is non-unique. It is contained in the columns: {1}".format(
                            k, ", ".join(v)
                        )
                        for k, v in non_unique_cols.items()
                    ]
                )
                + "."
            )

        if missing_cols:
            raise LoadTrajectoryError(f"{common_error_message} Missing columns: {', '.join(missing_cols)}.")

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
    _validate_is_deviation_vadere_pedpy_traj_transform_below_threshold(traj_dataframe, frame_rate)

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
        t_start_ = math.ceil(np.round(t_start * frame_rate, precision)) / frame_rate
        t_stop_ = math.floor(np.round(t_stop * frame_rate, precision)) / frame_rate

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
            interpolated_traj = traj.reindex(t.union(r)).interpolate(method="index").loc[r]

            interpolated_traj[ID_COL] = interpolated_traj[ID_COL].astype(int)

            traj_dataframe_interpolated = pd.concat([traj_dataframe_interpolated, interpolated_traj])

    if trajectory_too_short_messages and not ignore_too_short_trajectories:
        raise LoadTrajectoryError(
            "One or more pedestrian trajectories are too short to be captured "
            f"at frame rate {frame_rate}:\n- " + "\n- ".join(trajectory_too_short_messages)
        )

    if traj_dataframe_interpolated.empty:
        raise LoadTrajectoryError("No valid trajectories were captured.")

    traj_dataframe_interpolated = traj_dataframe_interpolated.reset_index()

    traj_dataframe_interpolated[FRAME_COL] = (
        (traj_dataframe_interpolated[TIME_COL] * frame_rate).round(decimals=0).astype(int)
    )
    traj_dataframe_interpolated = traj_dataframe_interpolated.drop(labels=TIME_COL, axis="columns")

    traj_dataframe_interpolated = traj_dataframe_interpolated.sort_values(by=[FRAME_COL, ID_COL], ignore_index=True)
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
        dx_dt = (np.sqrt(diff[[X_COL, Y_COL]].pow(2).sum(axis=1))).divide(diff[TIME_COL])
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
        raise LoadTrajectoryError(f"Margin ({margin!s}) should be greater than 10 ** (-{decimals!s}).")

    with open(vadere_scenario_file, "r") as f:
        data = json.load(f)
        topography = data["scenario"]["topography"]
        scenario_attributes = topography["attributes"]

        # bound
        complete_area = scenario_attributes["bounds"]
        bounding_box_with = scenario_attributes["boundingBoxWidth"]
        complete_area["x"] = complete_area["x"] + bounding_box_with - margin
        complete_area["y"] = complete_area["y"] + bounding_box_with - margin
        complete_area["width"] = complete_area["width"] - 2 * (bounding_box_with - margin)
        complete_area["height"] = complete_area["height"] - 2 * (bounding_box_with - margin)
        complete_area["type"] = "RECTANGLE"
        complete_area_points = _vadere_shape_to_point_list(complete_area, decimals=decimals)
        area_poly = shapely.Polygon(complete_area_points)

        # obstacles
        walkable_area_poly = Polygon(area_poly.exterior.coords)
        obstacles = topography["obstacles"]
        error_obst_ids = []
        for obstacle in obstacles:
            obst_points = _vadere_shape_to_point_list(obstacle["shape"], decimals=decimals)
            obstacle_polygon = shapely.Polygon(obst_points)
            if area_poly.contains_properly(obstacle_polygon):
                walkable_area_poly = walkable_area_poly.difference(obstacle_polygon)
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


def _vadere_shape_to_point_list(shape: dict[str, Any], decimals: int) -> list[Point]:
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
        raise LoadTrajectoryError(f"The given Vadere scenario contains an unsupported obstacle shape '{shape_type}'. ")

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
    points = [shapely.Point(np.round(p.x, decimals), np.round(p.y, decimals)) for p in points]
    return points
