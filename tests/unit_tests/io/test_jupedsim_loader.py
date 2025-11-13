import pathlib
import sqlite3
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import shapely
from tests.unit_tests.io.utils import get_data_frame_to_write, prepare_data_frame

from pedpy import FRAME_COL, ID_COL, X_COL, Y_COL, LoadTrajectoryError
from pedpy.data.geometry import WalkableArea
from pedpy.io.jupedsim_loader import load_trajectory_from_jupedsim_sqlite, load_walkable_area_from_jupedsim_sqlite
from pedpy.io.trajectory_loader import TrajectoryUnit


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
        cur.execute("CREATE TABLE metadata(key TEXT NOT NULL UNIQUE PRIMARY KEY, value TEXT NOT NULL)")
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
        cur.execute("CREATE TABLE metadata(key TEXT NOT NULL UNIQUE PRIMARY KEY, value TEXT NOT NULL)")
        cur.execute(
            "INSERT INTO metadata VALUES(?, ?)",
            (("version", "2")),
        )

    if create_geometry:
        cur.execute("DROP TABLE IF EXISTS geometry")
        cur.execute("CREATE TABLE geometry(   hash INTEGER NOT NULL,    wkt TEXT NOT NULL)")
    cur.execute("DROP TABLE IF EXISTS frame_data")
    cur.execute("CREATE TABLE frame_data(   frame INTEGER NOT NULL,   geometry_hash INTEGER NOT NULL)")

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
        raise RuntimeError(f"Internal Error: Trying to write unsupported JuPedSim Version {version}.")


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
        trajectory_sqlite = pathlib.Path(tmp_path / f"trajectory-v{version}.sqlite")

        expected_data = pd.DataFrame(data=data)

        written_data = get_data_frame_to_write(expected_data, TrajectoryUnit.METER)

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
            traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy() == expected_data.to_numpy()
        ).all()
        assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_jupedsim_sqlite_v1_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/jupedsim_v1.sqlite")
    load_trajectory_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_trajectory_from_jupedsim_sqlite_v2_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/jupedsim_v2.sqlite")
    load_trajectory_from_jupedsim_sqlite(trajectory_file=traj_txt)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_trajectory_data(tmp_path: pathlib.Path, version):
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
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(error_info.value)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_empty_trajectory_data(tmp_path: pathlib.Path, version):
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
    assert "The given sqlite trajectory file seems to be empty." in str(error_info.value)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_meta_data(tmp_path: pathlib.Path, version):
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
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(error_info.value)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_trajectory_from_jupedsim_sqlite_no_frame_rate_in_meta_data(tmp_path: pathlib.Path, version):
    trajectory_sqlite = pathlib.Path(tmp_path / "trajectory.sqlite")

    data = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    write_jupedsim_sqlite_trajectory_file(file=trajectory_sqlite, data=data, version=version)

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_sqlite,
        )
    assert "The given sqlite trajectory file seems not include a frame rate." in str(error_info.value)


def test_load_trajectory_from_jupedsim_sqlite_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_jupedsim_sqlite(trajectory_file=pathlib.Path("non_existing_file"))
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
    walkable_area = load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_sqlite)
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
    walkable_area = load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_sqlite)
    assert expected_walkable_area.polygon.equals(walkable_area.polygon)


def test_load_walkable_area_from_jupedsim_sqlite_v1_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/jupedsim_v1.sqlite")
    load_walkable_area_from_jupedsim_sqlite(trajectory_file=traj_txt)


def test_load_walkable_area_from_jupedsim_sqlite_v2_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/jupedsim_v2.sqlite")
    load_walkable_area_from_jupedsim_sqlite(trajectory_file=traj_txt)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_walkable_area_from_jupedsim_sqlite_no_geometry_table(tmp_path: pathlib.Path, version):
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
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_sqlite)
    assert "The given sqlite trajectory is not a valid JuPedSim format" in str(error_info.value)


@pytest.mark.parametrize(
    "version",
    [1, 2],
)
def test_load_walkable_area_from_jupedsim_sqlite_no_geometry(tmp_path: pathlib.Path, version):
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
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_sqlite)
    assert "The given sqlite trajectory file seems not include a geometry" in str(error_info.value)


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
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_sqlite)
    assert "The given sqlite trajectory has unsupported db version" in str(error_info.value)


def test_load_walkable_area_from_jupedsim_sqlite_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_walkable_area_from_jupedsim_sqlite_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_jupedsim_sqlite(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)
