import numpy as np
import pandas as pd
import pytest
from numpy import dtype

from analyzer.data.trajectory_data import *
from analyzer.io.trajectory_parser import *


def prepare_data_frame(data_frame: pd.DataFrame):
    """Prepare the data set for comparison with the result of the parsing.

    Trims the data frame to the first 5 columns and sets the column dtype to float. This needs to
    be done, as only the relevant data are read from the file, and the rest will be ignored.

    Args:
        data_frame (pd.DataFrame): data frame to prepare

    Result:
        prepared data frame
    """
    result = data_frame[[0, 1, 2, 3, 4]].copy(deep=True)
    result = result.astype("float64")

    return result


def get_data_frame_to_write(data_frame: pd.DataFrame, unit: TrajectoryUnit):
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
        data_frame_to_write[4] = pd.to_numeric(data_frame_to_write[4]).mul(100)
    return data_frame_to_write


def write_trajectory_file(
    *,
    data: pd.DataFrame,
    file: pathlib.Path,
    frame_rate: float = None,
    traj_type: TrajectoryType = None,
    unit: TrajectoryUnit = None,
):
    with file.open("w") as f:
        if traj_type is not None:
            if traj_type == TrajectoryType.JUPEDSIM:
                f.write("#description: jpscore (0.8.4)\n")
            elif traj_type == TrajectoryType.PETRACK:
                f.write("# PeTrack project: project.pet\n")
            else:
                f.write("# neither\n")

        if frame_rate is not None:
            f.write(f"#framerate: {frame_rate}\n")

        if unit is not None:
            if unit == TrajectoryUnit.CENTIMETER:
                f.write("# id frame x/cm y/cm z/cm\n")
            else:
                f.write("# id frame x/m y/m z/m\n")

        f.write(data.to_csv(sep=" ", header=False, index=False))


@pytest.mark.parametrize(
    "data, expected_frame_rate, expected_type, expected_unit",
    [
        (
            np.array([[0, 0, 5, 1, 10], [1, 0, -5, -1, -10]]),
            7.0,
            TrajectoryType.JUPEDSIM,
            TrajectoryUnit.METER,
        ),
        (
            np.array([[0, 0, 5, 1, 10], [1, 0, -5, -1, -10]]),
            50.0,
            TrajectoryType.JUPEDSIM,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([[0, 0, 5, 1, 10], [1, 0, -5, -1, -10]]),
            15.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.METER,
        ),
        (
            np.array([[0, 0, 5, 1, 10], [1, 0, -5, -1, -10]]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([[0, 0, 5, 1, 10, 123], [1, 0, -5, -1, -10, 123]]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([[0, 0, 5, 1, 10, "should be ignore"], [1, 0, -5, -1, -10, "this too"]]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
    ],
)
def test_parse_trajectory_files_success(
    tmp_path,
    data: List[np.array],
    expected_frame_rate: float,
    expected_type: TrajectoryType,
    expected_unit: TrajectoryUnit,
):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, expected_unit)
    write_trajectory_file(
        file=trajectory_txt,
        frame_rate=expected_frame_rate,
        traj_type=expected_type,
        unit=expected_unit,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)
    traj_data_from_file = parse_trajectory(trajectory_txt)

    assert (
        traj_data_from_file.data[["ID", "frame", "X", "Y", "Z"]].to_numpy()
        == expected_data.to_numpy()
    ).all()
    assert traj_data_from_file.frame_rate == expected_frame_rate
    assert traj_data_from_file.trajectory_type == expected_type


@pytest.mark.parametrize(
    "data, expected_frame_rate, expected_type, expected_unit",
    [
        (
            np.array([(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)]),
            50.0,
            TrajectoryType.JUPEDSIM,
            TrajectoryUnit.METER,
        ),
        (
            np.array([(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)]),
            50.0,
            TrajectoryType.JUPEDSIM,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.METER,
        ),
        (
            np.array([(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array([(0, 0, 5, 1, 10, "should be ignored"), (1, 0, -5, -1, -10, "this too")]),
            50.0,
            TrajectoryType.PETRACK,
            TrajectoryUnit.CENTIMETER,
        ),
    ],
)
def test_parse_trajectory_file(
    tmp_path,
    data: np.array,
    expected_frame_rate: float,
    expected_type: TrajectoryType,
    expected_unit: TrajectoryUnit,
):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    expected_data = pd.DataFrame(data=data)
    written_data = get_data_frame_to_write(expected_data, expected_unit)

    write_trajectory_file(
        file=trajectory_txt,
        frame_rate=expected_frame_rate,
        traj_type=expected_type,
        unit=expected_unit,
        data=written_data,
    )

    data_from_file, frame_rate_from_file, type_from_file = parse_trajectory_file(trajectory_txt)
    expected_data = prepare_data_frame(expected_data)

    assert (
        data_from_file[["ID", "frame", "X", "Y", "Z"]].to_numpy() == expected_data.to_numpy()
    ).all()
    assert frame_rate_from_file == expected_frame_rate
    assert type_from_file == expected_type


@pytest.mark.parametrize(
    "data, separator, expected_unit",
    [
        (
            np.array(
                [(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)],
            ),
            " ",
            TrajectoryUnit.METER,
        ),
        (
            np.array(
                [(0, 0, 5, 1, 10), (1, 0, -5, -1, -10)],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array(
                [(0, 0, 5, 1, 10, 99999), (1, 0, -5, -1, -10, -99999)],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
        (
            np.array(
                [(0, 0, 5, 1, 10, "test"), (1, 0, -5, -1, -10, "will be ignored")],
            ),
            " ",
            TrajectoryUnit.CENTIMETER,
        ),
    ],
)
def test_parse_trajectory_data_success(
    tmp_path, data: np.array, separator: str, expected_unit: TrajectoryUnit
):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    expected_data = pd.DataFrame(data=data)

    written_data = get_data_frame_to_write(expected_data, expected_unit)

    write_trajectory_file(
        file=trajectory_txt,
        unit=expected_unit,
        data=written_data,
    )

    expected_data = prepare_data_frame(expected_data)

    data_from_file = parse_trajectory_data(trajectory_txt)
    print(list(data_from_file.dtypes.values))
    assert list(data_from_file.dtypes.values) == [
        dtype("int64"),
        dtype("int64"),
        dtype("float64"),
        dtype("float64"),
        dtype("float64"),
    ]
    assert (
        data_from_file[["ID", "frame", "X", "Y", "Z"]].to_numpy() == expected_data.to_numpy()
    ).all()


@pytest.mark.parametrize(
    "data, expected_message",
    [
        (np.array([]), "The given trajectory file seem to be empty."),
        (
            np.array(
                [
                    (0, 0, 5, 1),
                    (
                        1,
                        0,
                        -5,
                        -1,
                    ),
                ]
            ),
            "The given trajectory file could not be parsed.",
        ),
    ],
)
def test_parse_trajectory_data_failure(tmp_path, data: np.array, expected_message: str):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")
    written_data = pd.DataFrame(data=data)

    write_trajectory_file(
        file=trajectory_txt,
        data=written_data,
    )

    with pytest.raises(ValueError) as error_info:
        parse_trajectory_data(trajectory_txt)

    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "file_content, expected_frame_rate",
    [
        ("#framerate: 8.00", 8.0),
        ("#framerate: 8.", 8.0),
        ("#framerate: 25", 25.0),
        ("#framerate: 25e0", 25.0),
        ("#framerate: 25 10 8", 25.0),
        ("#framerate: \n#framerate: 25", 25.0),
        ("# framerate: 8.0 fps", 8.0),
        ("# framerate: 25 fps", 25.0),
        ("# framerate: 25e0 fps", 25.0),
        ("# framerate: 25 10 fps", 25.0),
        ("# framerate: 25 fps 10", 25.0),
        ("# framerate: 25 fps 10", 25.0),
        ("# framerate: \n# framerate: 25 fps", 25.0),
    ],
)
def test_parse_frame_rate_success(tmp_path, file_content: str, expected_frame_rate: float):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    frame_rate_from_file = parse_frame_rate(trajectory_txt)
    assert frame_rate_from_file == expected_frame_rate


@pytest.mark.parametrize(
    "file_content, expected_exception, expected_message, default_frame_rate",
    [
        (
            "",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "framerate: -8.00",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "#framerate:",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "#framerate: asdasd",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "framerate: 25 fps",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "#framerate: fps",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "#framerate: asdasd fps",
            ValueError,
            "Frame rate is needed, but none could be found in the trajectory files.",
            None,
        ),
        (
            "#framerate: 0",
            ValueError,
            "Frame rate needs to be a positive value,",
            None,
        ),
        (
            "#framerate: -25.",
            ValueError,
            "Frame rate needs to be a positive value,",
            None,
        ),
        (
            "#framerate: 0.00 fps",
            ValueError,
            "Frame rate needs to be a positive value,",
            None,
        ),
        (
            "#framerate: -10.00 fps",
            ValueError,
            "Frame rate needs to be a positive value,",
            None,
        ),
        (
            "#framerate: 25.00 fps",
            ValueError,
            "The given default frame rate seems to differ from the frame rate given in",
            30.0,
        ),
        (
            "#framerate: asdasd fps",
            ValueError,
            "Default frame needs to be positive but is",
            0,
        ),
        (
            "#framerate: asdasd fps",
            ValueError,
            "Default frame needs to be positive but is",
            -12,
        ),
        (
            "#framerate: asdasd fps",
            ValueError,
            "Default frame needs to be positive but is",
            0.0,
        ),
        (
            "#framerate: asdasd fps",
            ValueError,
            "Default frame needs to be positive but is",
            -12.0,
        ),
    ],
)
def test_parse_frame_rate_failure(
    tmp_path,
    file_content: str,
    expected_exception,
    expected_message: str,
    default_frame_rate: float,
):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    with pytest.raises(expected_exception) as error_info:
        if default_frame_rate is None:
            parse_frame_rate(trajectory_txt)
        else:
            parse_frame_rate(trajectory_txt, default_frame_rate)

    assert expected_message in str(error_info.value)


@pytest.mark.parametrize(
    "file_content, expected_type",
    [
        ("", TrajectoryType.FALLBACK),
        ("# line does not contain petrack nor jupedsim, result Fallback", TrajectoryType.FALLBACK),
        ("# PeTrack project: project.pet", TrajectoryType.PETRACK),
        ("#description: jpscore (0.8.4)", TrajectoryType.JUPEDSIM),
    ],
)
def test_parse_trajectory_type(tmp_path, file_content: str, expected_type: TrajectoryType):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    type_from_file = parse_trajectory_type(trajectory_txt)
    assert type_from_file == expected_type


@pytest.mark.parametrize(
    "file_content, expected_unit",
    [
        ("", TrajectoryUnit.METER),
        ("here is no unit given, so default meter should be returned", TrajectoryUnit.METER),
        ("#ID	FR	X	Y	Z	A	B	ANGLE	COLOR	", TrajectoryUnit.METER),
        ("# id frame x/cm y/cm z/cm", TrajectoryUnit.CENTIMETER),
        ("# id frame x/cm y/cm z/cm".upper(), TrajectoryUnit.CENTIMETER),
        ("# id frame x/m y/m z/m", TrajectoryUnit.METER),
        ("# id frame x/m y/m z/m".upper(), TrajectoryUnit.METER),
    ],
)
def test_parse_unit_of_coordinates(tmp_path, file_content: str, expected_unit: TrajectoryUnit):
    trajectory_txt = pathlib.Path(tmp_path / "trajectory.txt")

    with trajectory_txt.open("w") as f:
        f.write(file_content)

    unit_from_file = parse_unit_of_coordinates(trajectory_txt)
    assert unit_from_file == expected_unit
