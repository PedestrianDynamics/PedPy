import json
import math
import pathlib
import re
import sqlite3
import textwrap
from datetime import datetime
from typing import Any, List, Optional

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import shapely
from numpy import dtype
from shapely import LinearRing, Polygon

from pedpy.column_identifier import *
from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import (
    LoadTrajectoryError,
    TrajectoryUnit,
    _load_trajectory_data_from_txt,
    _load_trajectory_meta_data_from_txt,
    _validate_is_file,
    load_trajectory_from_crowdit,
    load_trajectory_from_jupedsim_sqlite,
    load_trajectory_from_pathfinder_csv,
    load_trajectory_from_pathfinder_json,
    load_trajectory_from_ped_data_archive_hdf5,
    load_trajectory_from_txt,
    load_trajectory_from_vadere,
    load_trajectory_from_viswalk,
    load_walkable_area_from_crowdit,
    load_walkable_area_from_jupedsim_sqlite,
    load_walkable_area_from_ped_data_archive_hdf5,
    load_walkable_area_from_vadere_scenario,
)


def prepare_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data set for comparison with the result of the parsing.

    Trims the data frame to the first 4 columns and sets the column dtype to
    float. This needs to be done, as only the relevant data are read from the
    file, and the rest will be ignored.

    Args:
        data_frame (pd.DataFrame): data frame to prepare

    Result:
        prepared data frame
    """
    result = data_frame.iloc[:, :4].copy(deep=True)
    result = result.astype("float64")

    return result


def get_data_frame_to_write(
    data_frame: pd.DataFrame, unit: TrajectoryUnit
) -> pd.DataFrame:
    """Get the data frame which should be written to file.

    Args:
        data_frame (pd.DataFrame): data frame to prepare
        unit (TrajectoryUnit): unit used to write the data frame

    Result:
        copy
    """
    data_frame_to_write = data_frame.copy(deep=True)
    if unit == TrajectoryUnit.CENTIMETER:
        data_frame_to_write[2] = pd.to_numeric(data_frame_to_write[2]).mul(100)
        data_frame_to_write[3] = pd.to_numeric(data_frame_to_write[3]).mul(100)
    return data_frame_to_write


def write_txt_trajectory_file(
    *,
    data: pd.DataFrame,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    unit: Optional[TrajectoryUnit] = None,
) -> None:
    with file.open("w") as f:
        if frame_rate is not None:
            f.write(f"#framerate: {frame_rate}\n")

        if unit is not None:
            if unit == TrajectoryUnit.CENTIMETER:
                f.write("# id frame x/cm y/cm z/cm\n")
            else:
                f.write("# id frame x/m y/m z/m\n")

        f.write(data.to_csv(sep=" ", header=False, index=False))


def prepare_jupedsim_sqlite_trajectory_file_v1(
    *, file, create_trajectory=True, create_meta_data=True, create_geometry=True
):
    con = sqlite3.connect(file)

    cur = con.cursor()
    cur.execute("BEGIN")
    if create_trajectory:
        cur.execute("DROP TABLE IF EXISTS trajectory_data")
        cur.execute(
            "CREATE TABLE trajectory_data ("
            "   frame INTEGER NOT NULL,"
            "   id INTEGER NOT NULL,"
            "   pos_x REAL NOT NULL,"
            "   pos_y REAL NOT NULL,"
            "   ori_x REAL NOT NULL,"
            "   ori_y REAL NOT NULL)"
        )
        cur.execute("CREATE INDEX frame_id_idx ON trajectory_data(frame, id)")

    if create_meta_data:
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute(
            "CREATE TABLE metadata(key TEXT NOT NULL UNIQUE PRIMARY KEY, value TEXT NOT NULL)"
        )
        cur.execute(
            "INSERT INTO metadata VALUES(?, ?)",
            (("version", "1")),
        )

    if create_geometry:
        cur.execute("DROP TABLE IF EXISTS geometry")
        cur.execute("CREATE TABLE geometry(wkt TEXT NOT NULL)")
    cur.execute("COMMIT")


def prepare_jupedsim_sqlite_trajectory_file_v2(
    *, file, create_trajectory=True, create_meta_data=True, create_geometry=True
):
    con = sqlite3.connect(file)

    cur = con.cursor()
    cur.execute("BEGIN")
    if create_trajectory:
        cur.execute("DROP TABLE IF EXISTS trajectory_data")
        cur.execute(
            "CREATE TABLE trajectory_data ("
            "   frame INTEGER NOT NULL,"
            "   id INTEGER NOT NULL,"
            "   pos_x REAL NOT NULL,"
            "   pos_y REAL NOT NULL,"
            "   ori_x REAL NOT NULL,"
            "   ori_y REAL NOT NULL)"
        )
        cur.execute("CREATE INDEX frame_id_idx ON trajectory_data(frame, id)")

    if create_meta_data:
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute(
            "CREATE TABLE metadata(key TEXT NOT NULL UNIQUE PRIMARY KEY, value TEXT NOT NULL)"
        )
        cur.execute(
            "INSERT INTO metadata VALUES(?, ?)",
            (("version", "2")),
        )

    if create_geometry:
        cur.execute("DROP TABLE IF EXISTS geometry")
        cur.execute(
            "CREATE TABLE geometry("
            "   hash INTEGER NOT NULL, "
            "   wkt TEXT NOT NULL)"
        )
    cur.execute("DROP TABLE IF EXISTS frame_data")
    cur.execute(
        "CREATE TABLE frame_data("
        "   frame INTEGER NOT NULL,"
        "   geometry_hash INTEGER NOT NULL)"
    )

    cur.execute("COMMIT")


def write_jupedsim_sqlite_trajectory_file_v1(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    geometry: Optional[shapely.Polygon] = None,
    create_trajectory=True,
    create_meta_data=True,
    create_geometry=True,
):
    prepare_jupedsim_sqlite_trajectory_file_v1(
        file=file,
        create_trajectory=create_trajectory,
        create_geometry=create_geometry,
        create_meta_data=create_meta_data,
    )

    con = sqlite3.connect(file)

    cur = con.cursor()

    if frame_rate:
        cur.execute(
            "INSERT INTO metadata VALUES(?, ?)",
            (("fps", frame_rate)),
        )

    if geometry:
        geometry_wkt = shapely.to_wkt(
            geometry,
            rounding_precision=-1,
        )
        cur.execute("INSERT INTO geometry VALUES(?)", (geometry_wkt,))

    if cur.rowcount > 0:
        cur.execute("COMMIT")

    if data is not None:
        data.columns = ["id", "frame", "pos_x", "pos_y"]
        data["ori_x"] = 0
        data["ori_y"] = 0
        data.to_sql("trajectory_data", con, index=False, if_exists="append")


def write_jupedsim_sqlite_trajectory_file_v2(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    geometries: Optional[List[shapely.Polygon]] = None,
    create_trajectory=True,
    create_meta_data=True,
    create_geometry=True,
):
    prepare_jupedsim_sqlite_trajectory_file_v2(
        file=file,
        create_trajectory=create_trajectory,
        create_geometry=create_geometry,
        create_meta_data=create_meta_data,
    )

    con = sqlite3.connect(file)

    cur = con.cursor()

    if frame_rate:
        cur.execute(
            "INSERT INTO metadata VALUES(?, ?)",
            (("fps", frame_rate)),
        )

    if geometries:
        geometry_wkts = [
            shapely.to_wkt(
                geometry,
                rounding_precision=-1,
            )
            for geometry in geometries
        ]

        geometry_to_add = [(hash(wkt), wkt) for wkt in geometry_wkts]
        cur.executemany(
            "INSERT INTO geometry VALUES(?, ?)",
            geometry_to_add,
        )

    if cur.rowcount > 0:
        cur.execute("COMMIT")

    if data is not None:
        data.columns = ["id", "frame", "pos_x", "pos_y"]
        data["ori_x"] = 0
        data["ori_y"] = 0
        data.to_sql("trajectory_data", con, index=False, if_exists="append")


def write_jupedsim_sqlite_trajectory_file(
    *,
    version: int,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    geometries: Optional[List[shapely.Polygon]] = None,
    geometry: Optional[shapely.Polygon] = None,
    create_trajectory=True,
    create_meta_data=True,
    create_geometry=True,
):
    if version == 1:
        write_jupedsim_sqlite_trajectory_file_v1(
            data=data,
            file=file,
            frame_rate=frame_rate,
            geometry=geometry,
            create_trajectory=create_trajectory,
            create_meta_data=create_meta_data,
            create_geometry=create_geometry,
        )
    elif version == 2:
        write_jupedsim_sqlite_trajectory_file_v2(
            data=data,
            file=file,
            frame_rate=frame_rate,
            geometries=geometries,
            create_trajectory=create_trajectory,
            create_meta_data=create_meta_data,
            create_geometry=create_geometry,
        )
    else:
        raise RuntimeError(
            f"Internal Error: Trying to write unsupported JuPedSim Version {version}."
        )


def write_data_archive_hdf5_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    geometry: Optional[shapely.Polygon] = None,
):
    with h5py.File(file, "w") as h5_file:
        dt = h5py.special_dtype(vlen=str)
        h5_file.attrs["file_version"] = "1.0.0"
        if geometry:
            h5_file.attrs["wkt_geometry"] = shapely.to_wkt(
                geometry.normalize(), rounding_precision=-1
            )
        if data is not None:
            records = data.to_records(
                index=False,
                column_dtypes={
                    col: dt for col in data.columns[data.dtypes == "object"]
                },
            )
            ds_traj = h5_file.create_dataset(
                "trajectory",
                data=records,
            )
            if frame_rate:
                ds_traj.attrs["fps"] = frame_rate
            ds_traj.attrs["id"] = "unique identifier for pedestrian"
            ds_traj.attrs["frame"] = "frame number"
            ds_traj.attrs["x"] = "pedestrian x-coordinate (meter [m])"
            ds_traj.attrs["y"] = "pedestrian y-coordinate (meter [m])"


def write_viswalk_csv_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: float = 0,
    start_time: float = 0,
):
    data = data.rename(
        columns={
            ID_COL: "$PEDESTRIAN:NO",
            FRAME_COL: "SIMSEC",
            X_COL: "COORDCENTX",
            Y_COL: "COORDCENTY",
        }
    )
    data["SIMSEC"] = start_time + data["SIMSEC"] / frame_rate
    data.columns = [column.upper() for column in data.columns]

    write_header_viswalk(file=file, data=data)
    data.to_csv(file, sep=";", index=False, mode="a", encoding="utf-8-sig")


def write_vadere_csv_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: float,
):
    data = data.rename(
        columns={
            ID_COL: "pedestrianId",
            FRAME_COL: "simTime",
            X_COL: "startX-PID1",  # "-PID1" stands for processor id 1 and is used in Vadere
            # outputs as extension of the generic column name startX
            Y_COL: "startY-PID1",  # "-PID1" see comment above
        }
    )
    data["simTime"] = data["simTime"] / frame_rate

    vadere_traj_header = "#IDXCOL=2,DATACOL=2,SEP=' '\n"
    with open(file, "w", encoding="utf-8-sig") as writer:
        writer.write(vadere_traj_header)

    data.to_csv(file, sep=" ", index=False, mode="a", encoding="utf-8-sig")


def is_rectangular(poly: shapely.Polygon):
    return math.isclose(
        a=poly.area,
        b=poly.minimum_rotated_rectangle.area,
        abs_tol=0,
    )


def write_vadere_scenario_file(
    file,
    complete_area,
    obstacles,
    bounding_box_width,
):
    obstacles_ = list()

    for obstacle in obstacles:
        obst_coords = list(obstacle.coords)

        if is_rectangular(shapely.Polygon(obstacle)):
            minx, miny, maxx, maxy = shapely.Polygon(obstacle).bounds
            obstacles_ += [
                {
                    "shape": {
                        "type": "RECTANGLE",
                        "x": obst_coords[0][0],
                        "y": obst_coords[0][1],
                        "width": abs(maxx - minx),
                        "height": abs(maxy - miny),
                    }
                }
            ]
        else:  # is polygon
            obstacles_ += [
                {
                    "shape": {
                        "type": "POLYGON",
                        "points": [{"x": p[0], "y": p[1]} for p in obst_coords],
                    }
                }
            ]

    if is_rectangular(shapely.Polygon(complete_area)):
        scenario = {
            "name": "vadere_test",
            "release": str(),
            "scenario": {
                "topography": {
                    "attributes": {
                        "bounds": {
                            "x": complete_area.bounds[0],
                            "y": complete_area.bounds[1],
                            "width": abs(
                                complete_area.bounds[2]
                                - complete_area.bounds[0]
                            ),
                            "height": abs(
                                complete_area.bounds[3]
                                - complete_area.bounds[1]
                            ),
                        },
                        "boundingBoxWidth": bounding_box_width,
                    },
                    "obstacles": obstacles_,
                }
            },
        }
    else:
        raise RuntimeError(
            "Internal Error: Trying to write non-rectangular shape as Vadere "
            "scenario bound."
        )

    # Convert and write JSON object to file
    with open(file, "w") as f:
        json.dump(scenario, f, indent=2)


def write_header_viswalk(file, data):
    column_description = {
        "$PEDESTRIAN:NO": "No, Number (Unique pedestrian number)",
        "SIMSEC": "SimSec, Simulation second (Simulation time [s]) [s]",
        "COORDCENTX": "CoordCentX, Coordinate center (x) (X-coordinate of pedestrian’s center)",
        "COORDCENTY": "CoordCentY, Coordinate center (y) (Y-coordinate of pedestrian’s center)",
    }

    with open(file, "w", encoding="utf-8-sig") as writer:
        writer.write(
            textwrap.dedent(
                f"""\
                $VISION
                * File: {file.parent.absolute()}/{file.stem}.inpx
                * Comment: 
                * Date: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                * Application: PedPy Testing Module
                *
                * Table: Pedestrians In Network
                *
                """
            )
        )

        for column in data.columns:
            if column in column_description:
                writer.write(
                    f"* {column.replace('$PEDESTRIAN:', '')}: {column_description[column]}\n"
                )
            else:
                writer.write(
                    f"* {column.replace('$PEDESTRIAN:', '')}: Dummy description\n"
                )
        writer.write("*\n")

        writer.write("* ")
        for column in data.columns[:-1]:
            if column in column_description:
                description = column_description[column]
                writer.write(f"{description[: description.find(',')]};")
            else:
                writer.write(f"{column.capitalize()}")

        column = data.columns[-1]
        if column in column_description:
            description = column_description[column]
            writer.write(f"{description[: description.find(',')]};")
        else:
            writer.write(f"{column.capitalize()}")
        writer.write("\n")

        writer.write("* ")
        for column in data.columns[:-1]:
            if column in column_description:
                res = re.search(
                    r"(?<=, ).*(?= \([A-Z])", column_description[column]
                )
                writer.write(f"{res.group(0)};")
            else:
                writer.write(f"{column.capitalize()}")

        column = data.columns[-1]
        if column in column_description:
            res = re.search(
                r"(?<=, ).*(?= \([A-Z])", column_description[column]
            )
            writer.write(f"{res.group(0)};")
        else:
            writer.write(f"{column.capitalize()}")
        writer.write("\n")
        writer.write("*\n")


def test_validate_file_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        _validate_is_file(file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_validate_file_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        _validate_is_file(file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "data, expected_frame_rate, expected_unit",
    [
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
            TrajectoryUnit.METER,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            15.0,
            TrajectoryUnit.METER,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([[0, 0, 5, 1, 123], [1, 0, -5, -1, 123]]),
            50.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array(
                [
                    [0, 0, 5, 1, "should be ignored"],
                    [1, 0, -5, -1, "this too"],
                ]
            ),
            50.0,
            TrajectoryUnit.CENTIMETER,
        ),
    ],
)
def test_load_trajectory_from_txt_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
    expected_unit: TrajectoryUnit,
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, expected_unit)
    write_txt_trajectory_file(
        file=trajectory_txt,
        frame_rate=expected_frame_rate,
        unit=expected_unit,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_txt(
        trajectory_file=trajectory_txt,
        default_unit=None,
        default_frame_rate=None,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_txt_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/ped_data_archive_text.txt"
    )
    load_trajectory_from_txt(trajectory_file=traj_txt)


@pytest.mark.parametrize(
    "data, separator, expected_unit",
    [
        (
            np.array(
                [(0, 0, 5, 1), (1, 0, -5, -1)],
            ),
            " ",
            TrajectoryUnit.METER,
        ),
        (
            np.array(
                [(0, 0, 5, 1), (1, 0, -5, -1)],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array(
                [(0, 0, 5, 1, 99999), (1, 0, -5, -1, -99999)],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array(
                [
                    (0, 0, 5, 1, "test"),
                    (1, 0, -5, -1, "will be ignored"),
                ],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
    ],
)
def test_parse_trajectory_data_from_txt_success(
    tmp_path: pathlib.Path,
    data: npt.NDArray[np.float64],
    separator: str,
    expected_unit: TrajectoryUnit,
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, expected_unit)

    write_txt_trajectory_file(
        file=trajectory_txt,
        unit=expected_unit,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)

    data_from_file = _load_trajectory_data_from_txt(
        trajectory_file=trajectory_txt, unit=expected_unit
    )

    assert list(data_from_file.dtypes.values) == [
        dtype("int64"),
        dtype("int64"),
        dtype("float64"),
        dtype("float64"),
    ]
    assert (
        data_from_file[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()


@pytest.mark.parametrize(
    "data, expected_message",
    [
        (np.array([]), "The given trajectory file seem to be empty."),
        (
            np.array(
                [
                    (0, 0, 5),
                    (
                        1,
                        0,
                        -5,
                    ),
                ]
            ),
            "The given trajectory file could not be parsed.",
        ),
    ],
)
def test_parse_trajectory_data_from_txt_failure(
    tmp_path: pathlib.Path, data: npt.NDArray[np.float64], expected_message: str
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")
    written_data = pd.DataFrame(data=data)

    write_txt_trajectory_file(
        file=trajectory_txt,
        data=written_data,
    )

    with pytest.raises(ValueError) as error_info:
        _load_trajectory_data_from_txt(
            trajectory_file=trajectory_txt,
            unit=TrajectoryUnit.METER,  # type: ignore
        )

    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "file_content, expected_frame_rate, expected_unit",
    [
        (
            "#framerate: 8.00\n#ID frame x/m y/m z/m",
            8.0,
            TrajectoryUnit.METER,
        ),
        (
            "#framerate: 8.\n#ID frame x[in m] y[in m] z[in m]",
            8.0,
            TrajectoryUnit.METER,
        ),
        (
            "#framerate: 25\n#ID frame x/cm y/cm z/cm",
            25.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "#framerate: 25e0\n#ID frame x[in cm] y[in cm] z[in cm]",
            25.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "#framerate: 25 10 8\n#ID frame x/m y/m z/m",
            25.0,
            TrajectoryUnit.METER,
        ),
        (
            "#framerate: \n#framerate: 25\n#ID frame x[in m] y[in m] z[in m]",
            25.0,
            TrajectoryUnit.METER,
        ),
        (
            "# framerate: 8.0 fps\n#ID frame x/cm y/cm z/cm",
            8.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "# framerate: 25 fps\n#ID frame x[in cm] y[in cm] z[in cm]",
            25.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "# framerate: 25e0 fps\n#ID frame x/m y/m z/m",
            25.0,
            TrajectoryUnit.METER,
        ),
        (
            "# framerate: 25 10 fps\n#ID frame x[in m] y[in m] z[in m]",
            25.0,
            TrajectoryUnit.METER,
        ),
        (
            "# framerate: 25 fps 10\n#ID frame x/cm y/cm z/cm",
            25.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "# framerate: 25 fps 10\n#ID frame x[in cm] y[in cm] z[in cm]",
            25.0,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            "# framerate: \n# framerate: 25 fp\n#ID frame x/m y/m z/ms",
            25.0,
            TrajectoryUnit.METER,
        ),
    ],
)
def test_load_trajectory_meta_data_from_txt_success(
    tmp_path: pathlib.Path,
    file_content: str,
    expected_frame_rate: float,
    expected_unit: TrajectoryUnit,
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    frame_rate_from_file, unit_from_file = _load_trajectory_meta_data_from_txt(
        trajectory_file=trajectory_txt,
        default_frame_rate=None,
        default_unit=None,
    )

    assert frame_rate_from_file == expected_frame_rate
    assert unit_from_file == expected_unit


@pytest.mark.parametrize(
    "file_content, default_frame_rate, default_unit, expected_exception, "
    "expected_message",
    [
        (
            "",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "framerate: -8.00",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "#framerate:",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "#framerate: asdasd",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "framerate: 25 fps",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "#framerate: fps",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "#framerate: asdasd fps",
            None,
            None,
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory "
            "file.",
        ),
        (
            "#framerate: 0",
            None,
            None,
            ValueError,
            "Frame rate needs to be a positive value,",
        ),
        (
            "#framerate: -25.",
            None,
            None,
            ValueError,
            "Frame rate needs to be a positive value,",
        ),
        (
            "#framerate: 0.00 fps",
            None,
            None,
            ValueError,
            "Frame rate needs to be a positive value,",
        ),
        (
            "#framerate: -10.00 fps",
            None,
            None,
            ValueError,
            "Frame rate needs to be a positive value,",
        ),
        (
            "#framerate: 25.00 fps",
            30.0,
            None,
            ValueError,
            "The given default frame rate seems to differ from the frame rate "
            "given in",
        ),
        (
            "#framerate: asdasd fps",
            0,
            None,
            ValueError,
            "Default frame needs to be positive but is",
        ),
        (
            "#framerate: asdasd fps",
            -12,
            None,
            ValueError,
            "Default frame needs to be positive but is",
        ),
        (
            "#framerate: asdasd fps",
            0.0,
            None,
            ValueError,
            "Default frame needs to be positive but is",
        ),
        (
            "#framerate: asdasd fps",
            -12.0,
            None,
            ValueError,
            "Default frame needs to be positive but is",
        ),
        (
            "#framerate: 8.00\n#ID frame x y z",
            None,
            None,
            ValueError,
            "Unit is needed, but none could be found in the trajectory file.",
        ),
        (
            "#framerate: 8.00\n#ID frame x/m y/m z/m",
            None,
            TrajectoryUnit.CENTIMETER,
            ValueError,
            "The given default unit seems to differ from the unit given in the "
            "trajectory file:",
        ),
        (
            "#framerate: 8.00\n#ID frame x/cm y/cm z/cm",
            None,
            TrajectoryUnit.METER,
            ValueError,
            "The given default unit seems to differ from the unit given in the "
            "trajectory file:",
        ),
        (
            "#framerate: 8.00\n#ID frame x[in m] y[in m] z[in m]",
            None,
            TrajectoryUnit.CENTIMETER,
            ValueError,
            "The given default unit seems to differ from the unit given in the "
            "trajectory file:",
        ),
        (
            "#framerate: 8.00\n#ID frame x[in cm] y[in cm] z[in cm]",
            None,
            TrajectoryUnit.METER,
            ValueError,
            "The given default unit seems to differ from the unit given in the "
            "trajectory file:",
        ),
    ],
)
def test_load_trajectory_meta_data_from_txt_failure(
    tmp_path: pathlib.Path,
    file_content: str,
    default_frame_rate: float,
    default_unit: TrajectoryUnit,
    expected_exception: Any,
    expected_message: str,
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    with pytest.raises(expected_exception) as error_info:
        _load_trajectory_meta_data_from_txt(
            trajectory_file=trajectory_txt,
            default_unit=default_unit,
            default_frame_rate=default_frame_rate,
        )

    assert expected_message in str(error_info.value)


def test_load_trajectory_from_txt_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_txt_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            15.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
    ],
)
def test_load_trajectory_from_jupedsim_sqlite_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
) -> None:
    for version in [1, 2]:
        trajectory_sqlite = pathlib.Path(
            tmp_path / f"trajectory-v{version}.sqlite"
        )

        expected_data = pd.DataFrame(data=data)

        written_data = get_data_frame_to_write(
            expected_data, TrajectoryUnit.METER
        )

        write_jupedsim_sqlite_trajectory_file(
            file=trajectory_sqlite,
            frame_rate=expected_frame_rate,
            data=written_data,
            version=version,
        )
        expected_data = prepare_data_frame(expected_data)
        traj_data_from_file = load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )

        assert (
            traj_data_from_file.data[
                [ID_COL, FRAME_COL, X_COL, Y_COL]
            ].to_numpy()
            == expected_data.to_numpy()
        ).all()
        assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_jupedsim_sqlite_v1_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim_v1.sqlite"
    )
    load_trajectory_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_trajectory_from_jupedsim_sqlite_v2_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim_v2.sqlite"
    )
    load_trajectory_from_jupedsim_sqlite(trajectory_file=traj_txt)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_trajectory_data(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10.0,
        create_trajectory=False,
        version=version,
    )
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_empty_trajectory_data(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    empty_data = pd.DataFrame(columns=[ID_COL, FRAME_COL, X_COL, Y_COL])

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10.0,
        data=empty_data,
        version=version,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory file seems to be empty." in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_meta_data(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")
    data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        create_meta_data=False,
        data=data,
        version=version,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_frame_rate_in_meta_data(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite, data=data, version=version
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert (
        "The given sqlite trajectory file seems not include a frame rate."
        in str(error_info.value)
    )


def test_load_trajectory_from_jupedsim_sqlite_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_jupedsim_sqlite_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "version, data, frame_rate, walkable_area",
    [
        (
            1,
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
            shapely.Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]),
        ),
        (
            1,
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            10,
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
        ),
        (
            2,
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
            shapely.Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]),
        ),
        (
            2,
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            10,
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
        ),
    ],
)
def test_load_walkable_area_from_jupedsim_sqlite_success(
    tmp_path: pathlib.Path, version, data, frame_rate, walkable_area
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=frame_rate,
        data=written_data,
        geometry=walkable_area,
        geometries=[walkable_area],
        version=version,
    )

    expected_walkable_area = WalkableArea(walkable_area)
    walkable_area = load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert expected_walkable_area.polygon.equals(walkable_area.polygon)


@pytest.mark.parametrize(
    "data, frame_rate, geometries",
    [
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
            [shapely.box(-5, -5, 5, 5), shapely.box(-2.5, -2.5, 10, 10)],
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            10,
            [
                shapely.Polygon(
                    [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                    [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
                ),
                shapely.box(10, -5, 20, 5),
                shapely.box(15, 0, 20, 20),
            ],
        ),
    ],
)
def test_load_walkable_area_from_jupedsim_sqlite_v2_multiple_geometries_success(
    tmp_path: pathlib.Path, data, frame_rate, geometries
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=frame_rate,
        data=written_data,
        geometries=geometries,
        version=2,
    )

    expected_walkable_area = WalkableArea(shapely.union_all(geometries))
    walkable_area = load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert expected_walkable_area.polygon.equals(walkable_area.polygon)


def test_load_walkable_area_from_jupedsim_sqlite_v1_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim_v1.sqlite"
    )
    load_walkable_area_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_walkable_area_from_jupedsim_sqlite_v2_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim_v2.sqlite"
    )
    load_walkable_area_from_jupedsim_sqlite(trajectory_file=traj_txt)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_walkable_area_from_jupedsim_sqlite_no_geometry_table(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
    )

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10,
        data=written_data,
        create_geometry=False,
        version=version,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_walkable_area_from_jupedsim_sqlite_no_geometry(
    tmp_path: pathlib.Path, version
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
    )

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10,
        data=written_data,
        version=version,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert (
        "The given sqlite trajectory file seems not include a geometry"
        in str(error_info.value)
    )


def test_load_walkable_area_from_jupedsim_sqlite_non_supported_version(
    tmp_path: pathlib.Path,
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")
    prepare_jupedsim_sqlite_trajectory_file_v1(file=trajectory_sqlite)
    con = sqlite3.connect(trajectory_sqlite)

    cur = con.cursor()
    cur.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (999, "version"),
    )
    cur.execute("COMMIT")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert "The given sqlite trajectory has unsupported db version" in str(
        error_info.value
    )


def test_load_walkable_area_from_jupedsim_sqlite_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_walkable_area_from_jupedsim_sqlite_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            15.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            50.0,
        ),
    ],
)
def test_load_trajectory_from_data_archive_hdf5_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
) -> None:
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.h5")

    expected_data = pd.DataFrame(
        data=data,
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
        frame_rate=expected_frame_rate,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_ped_data_archive_hdf5(
        trajectory_file=trajectory_hdf5,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_data_archive_hdf5_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/ped_data_archive_hdf5.h5"
    )
    load_trajectory_from_ped_data_archive_hdf5(trajectory_file=traj_txt)


def test_load_trajectory_from_ped_data_archive_hdf5_no_trajectory_dataset(
    tmp_path: pathlib.Path,
):
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.h5")

    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_ped_data_archive_hdf5(
            trajectory_file=trajectory_hdf5
        )

    assert "it does not contain a 'trajectory' dataset." in str(
        error_info.value
    )


def test_load_trajectory_from_ped_data_archive_hdf5_trajectory_wrong_columns(
    tmp_path: pathlib.Path,
):
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.h5")
    traj_df = pd.DataFrame(
        np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=["foo", FRAME_COL, X_COL, Y_COL],
    )

    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
        data=traj_df,
        frame_rate=10,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_ped_data_archive_hdf5(
            trajectory_file=trajectory_hdf5
        )

    assert (
        "'trajectory' dataset does not contain the following columns: 'id', 'frame', 'x', and 'y'"
        in str(error_info.value)
    )


def test_load_trajectory_from_ped_data_archive_hdf5_trajectory_dataset_no_fps(
    tmp_path: pathlib.Path,
):
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.h5")
    traj_df = pd.DataFrame(
        np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
        data=traj_df,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_ped_data_archive_hdf5(
            trajectory_file=trajectory_hdf5
        )

    assert "the 'trajectory' dataset does not contain a 'fps' attribute" in str(
        error_info.value
    )


def test_load_trajectory_from_ped_data_archive_hdf5_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_ped_data_archive_hdf5(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_ped_data_archive_hdf5_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_ped_data_archive_hdf5(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "data, frame_rate, walkable_area",
    [
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            7.0,
            shapely.Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]),
        ),
        (
            np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
            10,
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
        ),
    ],
)
def test_load_walkable_area_from_ped_data_archive_hdf5_success(
    tmp_path: pathlib.Path, data, frame_rate, walkable_area
):
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
        frame_rate=frame_rate,
        data=written_data,
        geometry=walkable_area,
    )

    expected_walkable_area = WalkableArea(walkable_area)
    walkable_area = load_walkable_area_from_ped_data_archive_hdf5(
        trajectory_hdf5
    )
    assert expected_walkable_area.polygon.equals(walkable_area.polygon)


def test_load_walkable_area_from_data_archive_hdf5_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/ped_data_archive_hdf5.h5"
    )
    load_walkable_area_from_ped_data_archive_hdf5(trajectory_file=traj_txt)


def test_load_walkable_area_from_ped_data_archive_hdf5_no_wkt_in_file(
    tmp_path: pathlib.Path,
):
    trajectory_hdf5 = pathlib.Path(tmp_path / "trajectory.h5")
    traj_df = pd.DataFrame(
        np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_data_archive_hdf5_file(
        file=trajectory_hdf5,
        frame_rate=10,
        data=traj_df,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_ped_data_archive_hdf5(
            trajectory_file=trajectory_hdf5
        )

    assert "it does not contain a 'wkt_geometry' attribute" in str(
        error_info.value
    )


def test_load_walkable_area_from_ped_data_archive_hdf5_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_ped_data_archive_hdf5(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_walkable_area_from_ped_data_archive_hdf5_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_ped_data_archive_hdf5(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            15.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
    ],
)
def test_load_trajectory_from_viswalk_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    expected_data = pd.DataFrame(
        data=data,
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        frame_rate=expected_frame_rate,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_viswalk(
        trajectory_file=trajectory_viswalk,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_viswalk_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/viswalk.pp"
    )
    load_trajectory_from_viswalk(trajectory_file=traj_txt)


def test_load_trajectory_from_viswalk_no_data(
    tmp_path: pathlib.Path,
):
    data_empty = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    written_data = get_data_frame_to_write(data_empty, TrajectoryUnit.METER)
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_viswalk_frame_rate_zero(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_in_single_frame = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(
        data_in_single_frame, TrajectoryUnit.METER
    )
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert (
        "Can not determine the frame rate used to write the trajectory file."
        in str(error_info.value)
    )


def test_load_trajectory_from_viswalk_columns_missing(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "FOO!"],
    )

    written_data = get_data_frame_to_write(
        data_with_missing_column, TrajectoryUnit.METER
    )
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_viswalk_data_not_parseable(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(
        data_with_missing_column, TrajectoryUnit.METER
    )
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )
    with open(trajectory_viswalk, "a") as writer:
        writer.write("0; 2; This; is; a; line; to; break; the; parsing\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_viswalk_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_viswalk_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


def test_load_trajectory_from_vadere_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/vadere_postvis.traj"
    )
    load_trajectory_from_vadere(trajectory_file=traj_txt, frame_rate=24.0)


def test_load_trajectory_from_vadere_no_data(
    tmp_path: pathlib.Path,
):
    data_empty = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    written_data = get_data_frame_to_write(data_empty, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_vadere,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            15.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
    ],
)
def test_load_trajectory_from_vadere_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    expected_data = pd.DataFrame(
        data=data,
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_vadere_csv_file(
        file=trajectory_vadere,
        frame_rate=expected_frame_rate,
        data=written_data,
    )
    expected_data = prepare_data_frame(expected_data)

    traj_data_from_file = load_trajectory_from_vadere(
        trajectory_file=trajectory_vadere,
        frame_rate=expected_frame_rate,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_vadere_columns_missing(
    tmp_path: pathlib.Path,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "FOO!"],
    )

    written_data = get_data_frame_to_write(
        data_with_missing_column, TrajectoryUnit.METER
    )
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(
            trajectory_file=trajectory_vadere, frame_rate=24.0
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_vadere_columns_non_unique(
    tmp_path: pathlib.Path,
):
    trajectory_vadere = pathlib.Path(tmp_path / "postvis.traj")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 5, 1], [0, 1, -5, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "startX-PID2", Y_COL],
    )

    written_data = get_data_frame_to_write(
        data_with_missing_column, TrajectoryUnit.METER
    )
    write_vadere_csv_file(
        file=trajectory_vadere,
        data=written_data,
        frame_rate=24.0,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_vadere(
            trajectory_file=trajectory_vadere,
            frame_rate=24.0,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "area_poly, margin, bounding_box",
    [
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            0,
            0,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            1e-6,
            0,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            0,
            0.5,
        ),
        (
            shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            1e-6,
            0.5,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            0,
            0,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            1e-6,
            0,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 1), (2, 0)], [(-2, -2), (-3, -3), (-4, -2)]],
            ),
            0,
            0.5,
        ),
        (
            shapely.Polygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)],
                [[(0, 0), (1, 0), (1, 1), (0, 1)]],  # rectangular hole
            ),
            1e-6,
            0.5,
        ),
    ],
)
def test_load_walkable_area_from_vadere_scenario_success(
    tmp_path: pathlib.Path,
    area_poly: shapely.Polygon,
    margin: bool,
    bounding_box: float,
):
    file_path = pathlib.Path(tmp_path / "vadere_test.scenario")
    decimals = 6

    complete_area = area_poly.exterior
    if len(area_poly.interiors) > 0:
        obstacles = area_poly.interiors
    else:
        obstacles = []

    write_vadere_scenario_file(
        file=file_path,
        complete_area=complete_area,
        obstacles=obstacles,
        bounding_box_width=bounding_box,
    )
    walkable_area_from_file = load_walkable_area_from_vadere_scenario(
        file_path,
        margin=margin,
        decimals=decimals,
    )

    # convert test input to expected WalkableArea
    # 1) shrink expected walkable area to area without bounding box
    expected_shell = shapely.Polygon(area_poly.exterior)
    expected_shell = expected_shell.buffer(
        distance=-bounding_box + margin,
        single_sided=True,
        join_style="mitre",
    )
    # convert polygon shell to points to use it as input to WalkableArea
    expected_shell_points = list(expected_shell.exterior.coords)
    # handle floating point errors
    expected_shell_points = np.round(expected_shell_points, decimals)
    # 2) treat holes separately to avoid buffering of holes
    expected_holes = [list(p.coords) for p in area_poly.interiors]
    expected_walkable_area = WalkableArea(
        expected_shell_points, obstacles=expected_holes
    )

    expected_walkable_area_polygon = force_cw(expected_walkable_area.polygon)
    walkable_area_from_file_polygon = force_cw(walkable_area_from_file.polygon)

    # The polygons coordinates may differ by up to 'margin' on both axes.
    # The maximum possible difference between coordinates is therefore the diagonal of the square formed
    maximum_coordinate_difference_due_to_margin = math.sqrt(
        margin * margin + margin * margin
    )

    assert expected_walkable_area_polygon.equals_exact(
        walkable_area_from_file_polygon,
        maximum_coordinate_difference_due_to_margin,
    )


def force_cw(polygon):
    if is_ccw(polygon):
        return Polygon(polygon.exterior.coords[::-1])
    else:
        return polygon


def is_ccw(polygon):
    return LinearRing(polygon.exterior.coords).is_ccw

# ------------------ Pathfinder tests ------------------ #


def write_pathfinder_csv_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: float = 1,
    start_time: float = 0,
    unit: str = "m",
):
    """Write test data in Pathfinder CSV format."""
    if data is not None:
        data = data.rename(
            columns={
                ID_COL: "id",
                FRAME_COL: "t",
                X_COL: "x",
                Y_COL: "y",
            }
        )
        data["t"] = start_time + data["t"] / frame_rate
        with open(file, "w", encoding="utf-8-sig") as f:
            # Header-Zeile
            f.write("id,t,x,y\n")
            # Unit-Zeile (Pathfinder schreibt so etwas rein)
            f.write(f"unit,,{unit},{unit}\n")
            # Daten
            data.to_csv(f, index=False, header=False, encoding="utf-8-sig")


def write_pathfinder_json_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: float = 1,
    start_time: float = 0.0,
    unit: str = "m",
):
    """Write test data in Pathfinder JSON format."""
    result = {}

    if data is not None:
        data = data.rename(
            columns={
                "id": "id",
                "t": "t",
                "x": "x",
                "y": "y",
            }
        )

        for _, row in data.iterrows():
            agent_id = str(int(row["id"])).zfill(5)
            frame_time = start_time + row["t"] / frame_rate
            frame_key = f"{frame_time:.1f}"

            if agent_id not in result:
                result[agent_id] = {}

            result[agent_id][frame_key] = {
                "name": agent_id,
                "isActive": True,
                "position": {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": 0.0,
                },
                "velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "magnitude": 0.0,
                },
                "distance": 0.0,
                "location": f"Floor 0.0 {unit}->Room00",
                "terrainType": "level",
                "trigger": "None",
                "target": "None",
                "tagsApplied": [],
            }

    with open(file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


@pytest.mark.parametrize(
    "data, expected_frame_rate",
    [
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            7.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            50.0,
        ),
        (
            np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
            15.0,
        ),
        (
            np.array([[1, 0, 5, 1], [1, 1, 6, 2], [1, 2, 7, 3]]),
            10.0,
        ),
    ],
)
def test_load_trajectory_from_pathfinder_success_csv(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    expected_data = pd.DataFrame(
        data=data,
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_pathfinder_csv_file(
        file=trajectory_pathfinder,
        frame_rate=expected_frame_rate,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_pathfinder_csv(
        trajectory_file=trajectory_pathfinder,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_pathfinder_no_data_csv(
    tmp_path: pathlib.Path,
):
    data_empty = pd.DataFrame(
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    written_data = get_data_frame_to_write(data_empty, TrajectoryUnit.METER)
    write_pathfinder_csv_file(
        file=trajectory_pathfinder,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_pathfinder_frame_rate_zero_csv(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    data_in_single_frame = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(
        data_in_single_frame, TrajectoryUnit.METER
    )
    write_pathfinder_csv_file(
        file=trajectory_pathfinder,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder,
        )

    assert (
        "Can not determine the frame rate used to write the trajectory file."
        in str(error_info.value)
    )


def test_load_trajectory_from_pathfinder_columns_missing_csv(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    # Create CSV with missing 'y' column
    with open(trajectory_pathfinder, "w", encoding="utf-8-sig") as f:
        f.write("id,t,x\n")
        f.write("0,0.0,5.0\n")
        f.write("0,0.1,-5.0\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder,
        )
    assert "Missing columns: y." in str(error_info.value)


def test_load_trajectory_from_pathfinder_data_not_parseable_csv(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    # Create malformed CSV
    with open(trajectory_pathfinder, "w", encoding="utf-8-sig") as f:
        f.write("id,t,x,y\n")
        f.write("0,0.0,5.0,1.0\n")
        f.write("This,is,a,malformed,line,with,too,many,columns\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(
        error_info.value
    )


def test_load_trajectory_from_pathfinder_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    # TODO: also for json
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_pathfinder_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(trajectory_file=tmp_path)

    # TODO: also for json

    assert "is not a file" in str(error_info.value)


def test_load_trajectory_from_pathfinder_reference_file_csv():
    traj_csv = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/pathfinder.csv"
    )
    load_trajectory_from_pathfinder_csv(trajectory_file=traj_csv)


def test_load_trajectory_from_pathfinder_wrong_types_csv(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    with open(trajectory_pathfinder, "w", encoding="utf-8-sig") as f:
        f.write("id,t,x,y\n")
        # id="not_an_int" fails astype(int)
        f.write("not_an_int,0.0,1.0,2.0\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder
        )

    assert "Original error" in str(error_info.value)


@pytest.fixture
def json_file_path():
    return pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/pathfinder.json"
    )


def test_load_trajectory_from_pathfinder_reference_file_json(json_file_path):
    load_trajectory_from_pathfinder_json(trajectory_file=json_file_path)


def test_load_pathfinder_json_not_empty(json_file_path):
    traj = load_trajectory_from_pathfinder_json(trajectory_file=json_file_path)
    assert len(traj.data) > 0, "JSON loader should return non-empty data"


def test_pathfinder_json_contains_expected_fields(json_file_path):
    traj = load_trajectory_from_pathfinder_json(trajectory_file=json_file_path)
    data = traj.data
    expected_cols = {"id", "frame", "x", "y"}
    missing = expected_cols - set(data.columns)
    assert not missing, f"Missing fields: {missing}"


def test_pathfinder_json_at_least_one_agent(json_file_path):
    traj = load_trajectory_from_pathfinder_json(trajectory_file=json_file_path)
    data = traj.data
    agents = data["id"].astype(str).unique()

    assert "0" in agents


def test_load_trajectory_from_pathfinder_frame_rate_zero_json(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = tmp_path / "trajectory.json"

    data_in_single_frame = pd.DataFrame(
        np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=["id", "t", "x", "y"],
    )

    write_pathfinder_json_file(
        data=data_in_single_frame,
        file=trajectory_pathfinder,
        frame_rate=1,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_json(
            trajectory_file=trajectory_pathfinder,
        )

    assert (
        "Can not determine the frame rate used to write the trajectory file."
        in str(error_info.value)
    )

# ================== crowd:it ===================
def test_load_trajectory_from_crowdit_valid(tmp_path: pathlib.Path):
    trajectory_crowdit = tmp_path / "trajectory.csv"

    with open(trajectory_crowdit, "w", encoding="utf-8-sig") as f:
        f.write("pedID,time,posX,posY\n")
        f.write("0,0.0,1.0,2.0\n")
        f.write("0,0.5,1.5,2.5\n")  # gleicher Agent, späterer Zeitpunkt
        f.write("1,0.0,3.0,4.0\n")

    traj = load_trajectory_from_crowdit(trajectory_file=trajectory_crowdit)

    assert not traj.data.empty
    for col in (ID_COL, FRAME_COL, X_COL, Y_COL):
        assert col in traj.data.columns
    assert traj.frame_rate > 0


def test_load_trajectory_from_crowdit_wrong_types(tmp_path: pathlib.Path):
    trajectory_crowdit = tmp_path / "trajectory.csv"
    with open(trajectory_crowdit, "w", encoding="utf-8-sig") as f:
        f.write("pedID,time,posX,posY\n")
        f.write("not_an_int,0.0,1.0,2.0\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_crowdit(trajectory_file=trajectory_crowdit)

    assert "Original error" in str(error_info.value)


def test_load_trajectory_from_crowdit_missing_columns(tmp_path: pathlib.Path):
    trajectory_crowdit = tmp_path / "trajectory.csv"
    with open(trajectory_crowdit, "w", encoding="utf-8-sig") as f:
        f.write("pedID,time,posX\n")  # posY fehlt
        f.write("0,0.0,1.0\n")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_crowdit(trajectory_file=trajectory_crowdit)

    assert "Missing columns: posY" in str(error_info.value)


def test_load_walkable_area_from_crowdit_non_existing_file():
    with pytest.raises(LoadTrajectoryError):
        load_walkable_area_from_crowdit(
            geometry_file=pathlib.Path("non_existing_geometry.xml")
        )


def test_load_walkable_area_from_crowdit_no_walls(tmp_path: pathlib.Path):
    geometry_file = tmp_path / "geometry.xml"
    xml_content = "<geometry><layer></layer></geometry>"
    geometry_file.write_text(xml_content, encoding="utf-8")

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_crowdit(geometry_file=geometry_file)

    assert "No wall polygons found" in str(error_info.value)


def test_load_walkable_area_from_crowdit_invalid_xml(tmp_path: pathlib.Path):
    geometry_file = tmp_path / "geometry.xml"
    geometry_file.write_text("Not valid XML", encoding="utf-8")

    with pytest.raises(LoadTrajectoryError):
        load_walkable_area_from_crowdit(geometry_file=geometry_file)


def test_load_walkable_area_from_crowdit_valid(
    tmp_path: pathlib.Path, buffer=0
):
    geometry_file = tmp_path / "geometry.xml"
    xml_content = """
    <geometry>
        <layer>
            <wall>
                <point x="0" y="0"/>
                <point x="10" y="0"/>
                <point x="10" y="10"/>
                <point x="0" y="10"/>
            </wall>
        </layer>
    </geometry>
    """
    geometry_file.write_text(xml_content, encoding="utf-8")

    area = load_walkable_area_from_crowdit(geometry_file=geometry_file)

    assert isinstance(area, WalkableArea)
    assert area.polygon.area == 100.0


def test_load_trajectory_from_crowdit_reference_data_file():
    traj_file = pathlib.Path(__file__).parent / "test-data/crowdit.csv.gz"

    traj = load_trajectory_from_crowdit(trajectory_file=traj_file)
    assert not traj.data.empty
    for col in (ID_COL, FRAME_COL, X_COL, Y_COL):
        assert col in traj.data.columns
    assert traj.frame_rate > 0


def test_load_trajectory_from_crowdit_reference_geometry_file():
    geom_file = pathlib.Path(__file__).parent / "test-data/crowdit.floor"
    area = load_walkable_area_from_crowdit(geometry_file=geom_file)
    assert isinstance(area, WalkableArea)
    assert area.polygon.is_valid
    assert area.polygon.area > 0
