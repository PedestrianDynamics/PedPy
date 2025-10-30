import json
import pathlib
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tests.unit_tests.io.utils import get_data_frame_to_write, prepare_data_frame

from pedpy import FRAME_COL, ID_COL, X_COL, Y_COL, LoadTrajectoryError
from pedpy.io.pathfinder_loader import load_trajectory_from_pathfinder_csv, load_trajectory_from_pathfinder_json
from pedpy.io.trajectory_loader import TrajectoryUnit


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
            # header-row
            f.write("id,t,x,y\n")
            # unit-row (Pathfinder specifies the unit in file)
            f.write(f"unit,,{unit},{unit}\n")
            # data
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

    assert (traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy() == expected_data.to_numpy()).all()
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
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_pathfinder_frame_rate_zero_csv(
    tmp_path: pathlib.Path,
):
    trajectory_pathfinder = pathlib.Path(tmp_path / "trajectory.csv")

    data_in_single_frame = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(data_in_single_frame, TrajectoryUnit.METER)
    write_pathfinder_csv_file(
        file=trajectory_pathfinder,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(
            trajectory_file=trajectory_pathfinder,
        )

    assert "Can not determine the frame rate used to write the trajectory file." in str(error_info.value)


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
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_pathfinder_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(trajectory_file=pathlib.Path("non_existing_file"))

    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_pathfinder_reference_file_csv():
    traj_csv = pathlib.Path(__file__).parent / pathlib.Path("test-data/pathfinder.csv")
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
        load_trajectory_from_pathfinder_csv(trajectory_file=trajectory_pathfinder)
        assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_pathfinder_json_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_json(trajectory_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_pathfinder_csv_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)


@pytest.fixture
def json_file_path():
    return pathlib.Path(__file__).parent / pathlib.Path("test-data/pathfinder.json")


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

    assert "Can not determine the frame rate used to write the trajectory file." in str(error_info.value)


def test_load_trajectory_from_pathfinder_csv_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_csv(trajectory_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_pathfinder_json_sqlite_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_pathfinder_json(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)
