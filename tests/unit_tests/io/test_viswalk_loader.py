import pathlib
import re
import textwrap
from datetime import datetime
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tests.unit_tests.io.utils import (
    get_data_frame_to_write,
    prepare_data_frame,
)

from pedpy import FRAME_COL, ID_COL, X_COL, Y_COL, LoadTrajectoryError
from pedpy.io.trajectory_loader import TrajectoryUnit
from pedpy.io.viswalk_loader import load_trajectory_from_viswalk


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
                writer.write(f"* {column.replace('$PEDESTRIAN:', '')}: {column_description[column]}\n")
            else:
                writer.write(f"* {column.replace('$PEDESTRIAN:', '')}: Dummy description\n")
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
                res = re.search(r"(?<=, ).*(?= \([A-Z])", column_description[column])
                writer.write(f"{res.group(0)};")
            else:
                writer.write(f"{column.capitalize()}")

        column = data.columns[-1]
        if column in column_description:
            res = re.search(r"(?<=, ).*(?= \([A-Z])", column_description[column])
            writer.write(f"{res.group(0)};")
        else:
            writer.write(f"{column.capitalize()}")
        writer.write("\n")
        writer.write("*\n")


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

    assert (traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy() == expected_data.to_numpy()).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_from_viswalk_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/viswalk.pp")
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
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_viswalk_frame_rate_zero(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_in_single_frame = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [1, 0, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(data_in_single_frame, TrajectoryUnit.METER)
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert "Can not determine the frame rate used to write the trajectory file." in str(error_info.value)


def test_load_trajectory_from_viswalk_columns_missing(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, "FOO!"],
    )

    written_data = get_data_frame_to_write(data_with_missing_column, TrajectoryUnit.METER)
    write_viswalk_csv_file(
        file=trajectory_viswalk,
        data=written_data,
    )

    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(
            trajectory_file=trajectory_viswalk,
        )
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_viswalk_data_not_parseable(
    tmp_path: pathlib.Path,
):
    trajectory_viswalk = pathlib.Path(tmp_path / "trajectory.pp")

    data_with_missing_column = pd.DataFrame(
        data=np.array([[0, 0, 5, 1], [0, 1, -5, -1]]),
        columns=[ID_COL, FRAME_COL, X_COL, Y_COL],
    )

    written_data = get_data_frame_to_write(data_with_missing_column, TrajectoryUnit.METER)
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
    assert "The given trajectory file seems to be incorrect or empty." in str(error_info.value)


def test_load_trajectory_from_viswalk_non_existing_file():
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(trajectory_file=pathlib.Path("non_existing_file"))
    assert "does not exist" in str(error_info.value)


def test_load_trajectory_from_viswalk_non_file(tmp_path):
    with pytest.raises(LoadTrajectoryError) as error_info:
        load_trajectory_from_viswalk(trajectory_file=tmp_path)

    assert "is not a file" in str(error_info.value)
