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

from pedpy.column_identifier import *
from pedpy.data.geometry import WalkableArea
from pedpy.io.trajectory_loader import (
    LoadTrajectoryError,
    TrajectoryUnit,
    _load_trajectory_data_from_txt,
    _load_trajectory_meta_data_from_txt,
    _validate_is_file,
    load_trajectory,
    load_trajectory_from_fcd_data,
    load_trajectory_from_jupedsim_sqlite,
    load_trajectory_from_ped_data_archive_hdf5,
    load_trajectory_from_txt,
    load_walkable_area_from_jupedsim_sqlite,
    load_walkable_area_from_ped_data_archive_hdf5,
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


def prepare_jupedsim_sqlite_trajectory_file(
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


def write_jupedsim_sqlite_trajectory_file(
    *,
    data: Optional[pd.DataFrame] = None,
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
    geometry: Optional[shapely.Polygon] = None,
    create_trajectory=True,
    create_meta_data=True,
    create_geometry=True,
):
    prepare_jupedsim_sqlite_trajectory_file(
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


def write_fcd_file(
    *,
    data: Optional[pd.DataFrame],
    file: pathlib.Path,
    frame_rate: Optional[float] = None,
):
    data.columns = ["id", "frame", "x", "y"]
    print()
    print(data)
    with file.open("w") as f:
        f.write("<fcd-export>")

        for frame, frame_data in data.groupby(by=FRAME_COL):
            if frame_rate is not None:
                time = frame / frame_rate
            else:
                time = frame * frame
                print(time)
            if "x" in frame_data.columns:
                f.write(f'  <timestep time="{time}">')
                for _, row in frame_data.iterrows():
                    f.write(
                        f'      <person id="{row.id}" '
                        f'x="{row.x}" '
                        f'y="{row.y}" '
                        f'angle="{0}" '
                        f'type="{0}" '
                        f'speed="{0}" '
                        f'edge="{0}" '
                        f'slope="{0}"/>'
                    )
                f.write("  </timestep>")
            else:
                f.write(f'  <timestep time="{time}"/>')

        f.write("</fcd-export>")


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
            trajectory_file=trajectory_txt, unit=TrajectoryUnit.METER  # type: ignore
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
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=expected_frame_rate,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_jupedsim_sqlite(
        trajectory_file=trajectory_sqlite,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_jupedsim_sqlite_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim.sqlite"
    )
    load_trajectory_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_trajectory_from_jupedsim_sqlite_no_trajectory_data(
    tmp_path: pathlib.Path,
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10.0,
        create_trajectory=False,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


def test_load_trajectory_from_jupedsim_sqlite_empty_trajectory_data(
    tmp_path: pathlib.Path,
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    empty_data = pd.DataFrame(columns=[ID_COL, FRAME_COL, X_COL, Y_COL])

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=10.0,
        data=empty_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory file seems to be empty." in str(
        error_info.value
    )


def test_load_trajectory_from_jupedsim_sqlite_no_meta_data(
    tmp_path: pathlib.Path,
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")
    data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite, create_meta_data=False, data=data
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


def test_load_trajectory_from_jupedsim_sqlite_no_frame_rate_in_meta_data(
    tmp_path: pathlib.Path,
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        data=data,
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
def test_load_walkable_area_from_jupedsim_sqlite_success(
    tmp_path: pathlib.Path, data, frame_rate, walkable_area
):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_jupedsim_sqlite_trajectory_file(
        file=trajectory_sqlite,
        frame_rate=frame_rate,
        data=written_data,
        geometry=walkable_area,
    )

    expected_walkable_area = WalkableArea(walkable_area)
    walkable_area = load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert expected_walkable_area.polygon.equals(walkable_area.polygon)


def test_load_walkable_area_from_jupedsim_sqlite_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path(
        "test-data/jupedsim.sqlite"
    )
    load_walkable_area_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_walkable_area_from_jupedsim_sqlite_no_geometry_table(
    tmp_path: pathlib.Path,
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
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(
        error_info.value
    )


def test_load_walkable_area_from_jupedsim_sqlite_no_geometry(
    tmp_path: pathlib.Path,
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
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_sqlite)
    assert (
        "The given sqlite trajectory file seems not include a geometry"
        in str(error_info.value)
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
    print(traj_txt.absolute())
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
            0.1,
        ),
    ],
)
def test_load_trajectory_from_fcd_success(
    tmp_path: pathlib.Path,
    data: List[npt.NDArray[np.float64]],
    expected_frame_rate: float,
) -> None:
    trajectory_file_fcd = pathlib.Path(tmp_path / "trajectory.fcd")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_fcd_file(
        file=trajectory_file_fcd,
        frame_rate=expected_frame_rate,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = load_trajectory_from_fcd_data(
        trajectory_file=trajectory_file_fcd,
    )

    assert (
        traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


@pytest.mark.parametrize(
    "data",
    [
        (np.array([[0, 0, 5, 1], [0, 0, -5, -1]])),
        (np.array([[0, 0, 5, 1], [0, 0, -5, -1]])),
    ],
)
def test_load_trajectory_from_fcd_not_enough_frames(tmp_path, data):
    trajectory_file_fcd = pathlib.Path(tmp_path / "trajectory.fcd")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_fcd_file(
        file=trajectory_file_fcd,
        frame_rate=10,
        data=written_data,
    )
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_fcd_data(trajectory_file=trajectory_file_fcd)
    assert "Need at least two time steps to compute the frame rate" in str(
        error_info.value
    )


@pytest.mark.parametrize(
    "data",
    [
        (np.array([[0, 0, 5, 1], [0, 1, -5, -1], [0, 2, -5, -1]])),
        (np.array([[0, 0, 5, 1], [0, 1, -5, -1], [0, 2, -5, -1]])),
    ],
)
def test_load_trajectory_from_fcd_changing_frame_rate(tmp_path, data):
    trajectory_file_fcd = pathlib.Path(tmp_path / "trajectory.fcd")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)
    write_fcd_file(
        file=trajectory_file_fcd,
        data=written_data,
    )
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_fcd_data(trajectory_file=trajectory_file_fcd)
    assert "The time step seems to vary in the file" in str(error_info.value)


def test_load_trajectory_from_fcd_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_fcd_data(
            trajectory_file=pathlib.Path("non_existing_file")
        )
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_fcd_archive_hdf5_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_fcd_data(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)
