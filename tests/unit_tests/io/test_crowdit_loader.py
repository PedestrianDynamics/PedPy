import pathlib

import pytest

from pedpy import LoadTrajectoryError
from pedpy.column_identifier import *
from pedpy.data.geometry import WalkableArea
from pedpy.io.crowdit_loader import load_trajectory_from_crowdit, load_walkable_area_from_crowdit


def test_load_trajectory_from_crowdit_valid(tmp_path: pathlib.Path):
    trajectory_crowdit = tmp_path / "trajectory.csv"

    with open(trajectory_crowdit, "w", encoding="utf-8-sig") as f:
        f.write("pedID,time,posX,posY\n")
        f.write("0,0.0,1.0,2.0\n")
        f.write("0,0.5,1.5,2.5\n")  # same agent, later in time
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


def test_load_trajectory_from_crowdit_reference_data_file():
    traj_file = pathlib.Path(__file__).parent / "test-data/crowdit.csv.gz"

    traj = load_trajectory_from_crowdit(trajectory_file=traj_file)
    assert not traj.data.empty
    for col in (ID_COL, FRAME_COL, X_COL, Y_COL):
        assert col in traj.data.columns
    assert traj.frame_rate > 0


def test_load_trajectory_from_crowdit_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_crowdit(trajectory_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_crowdit_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_crowdit(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


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


def test_load_walkable_area_from_crowdit_valid(tmp_path: pathlib.Path, buffer=0):
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


def test_load_walkable_area_from_crowdit_reference_geometry_file():
    geom_file = pathlib.Path(__file__).parent / "test-data/crowdit.floor"
    area = load_walkable_area_from_crowdit(geometry_file=geom_file)
    assert isinstance(area, WalkableArea)
    assert area.polygon.is_valid
    assert area.polygon.area > 0


def test_load_walkable_area_from_crowdit_non_existing_file():
    with pytest.raises(LoadTrajectoryError):
        load_walkable_area_from_crowdit(geometry_file=pathlib.Path("non_existing_geometry.xml"))


def test_load_walkable_area_from_crowdit_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_walkable_area_from_crowdit(geometry_file=tmp_path)

    assert "is not a file" in str(error_info.value)
