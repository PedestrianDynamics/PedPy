import pathlib
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from tests.unit_tests.io.test_ped_data_archive_loader import write_txt_trajectory_file
from tests.unit_tests.io.utils import get_data_frame_to_write, prepare_data_frame

from pedpy import FRAME_COL, ID_COL, X_COL, Y_COL, TrajectoryUnit, load_trajectory


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
def test_load_trajectory_success(
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
    traj_data_from_file = load_trajectory(
        trajectory_file=trajectory_txt,
        default_unit=None,
        default_frame_rate=None,
    )

    assert (traj_data_from_file.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].to_numpy() == expected_data.to_numpy()).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate


def test_load_trajectory_reference_file():
    traj_txt = pathlib.Path(__file__).parent / pathlib.Path("test-data/ped_data_archive_text.txt")
    load_trajectory(trajectory_file=traj_txt)


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
def test_load_trajectory_data_failure(
    tmp_path: pathlib.Path, data: npt.NDArray[np.float64], expected_message: str
) -> None:
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")
    written_data = pd.DataFrame(data=data)

    write_txt_trajectory_file(file=trajectory_txt, data=written_data, frame_rate=25.0)

    with pytest.raises(ValueError) as error_info:
        load_trajectory(trajectory_file=trajectory_txt, default_unit=TrajectoryUnit.METER, default_frame_rate=25.0)

    assert expected_message in str(error_info.value)
