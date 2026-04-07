import pytest

import pathlib
import logging

from pedpy import InputError
from pedpy.io.helper import TrajectoryUnit
from pedpy.io.trajectory_loader import load_trajectory
from pedpy.preprocessing.trajectory_outlier_detection import detect_outliers


@pytest.fixture()
def setup_trajectory_data():
    trajectory_data = load_trajectory(
        trajectory_file=pathlib.Path(
            pathlib.Path(__file__).parent.parent / "io/test-data/uni_corr_500_08_modified.txt"),
        default_unit=TrajectoryUnit.METER,
    )
    return trajectory_data

def test_detect_outliers_correction(setup_trajectory_data):
    corrected_trajectory = detect_outliers(setup_trajectory_data,  tolerance=6, deleting=True)[0]
    valid_trajectory, changed_index_orig, changed_index_new = (
        detect_outliers(corrected_trajectory,  tolerance=6))
    assert changed_index_orig ==[]
    assert changed_index_new ==[]

def test_detect_outliers_changes_index_list_deleting(setup_trajectory_data):
    corrected_trajectory, changed_index_orig, changed_index_new  = (
        detect_outliers(setup_trajectory_data,  tolerance=6, deleting=True))
    assert changed_index_orig == [55, 89, 123, 209, 210, 211, 463]
    assert changed_index_new == [55, 89, 123, 209, 210, 211, 462]

def test_detect_outliers_logging_default_tolerance6(setup_trajectory_data, caplog):
    with caplog.at_level(logging.INFO):
        detect_outliers(
            traj_data=setup_trajectory_data,
            tolerance=6,
            deleting=True,
            quantile=0.97,
            max_length=7
        )
    expected_messages = [
        "Outliers found: personID 55 at frames [2018] ",
        "Outliers found: personID 89 at frames [438] ",
        "Outliers found: personID 123 at frames [3081, 3082, 3083, 3084, 3085, 3086, 3087] ",
        "Outliers found: personID 209 at frames [479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490] ",
        "Outliers found: personID 210 at frames [1387, 1388, 1389] ",
        "Outliers found: personID 211 at frames [1186, 1187, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078] ",
        "Trajectory with personID 422 has to many invalid points and cannot be corrected",
        "Trajectory with personID 422 was deleted",
        "Outliers found: personID 463 at frames [350, 713, 714, 715, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766] "
    ]
    assert all(
        any(msg in r.message and r.levelname == "INFO" for r in caplog.records)
        for msg in expected_messages
    )

def test_detect_outliers_logging_split_only(setup_trajectory_data, caplog):
    with caplog.at_level(logging.INFO):
        detect_outliers(
            traj_data=setup_trajectory_data,
            tolerance=6,
            drops_only=True
        )
    expected_messages = [
    "Frames in trajectory with original personID 56 were cut out after frame 2017 ",
    "Frames in trajectory with original personID 210 were cut out after frame 478 ",
    "Frames in trajectory with original personID 211 were cut out before frame 1389",
    "Frames in trajectory with original personID 212 were cut out before frame 1188 and after frame 2054 ",
    "Trajectory with personID 422 has to many invalid points and cannot be corrected",
    "Trajectory with personID 422 will be returned unchanged",
    "Frames in trajectory with original personID 464 were cut out before frame 351 and after frame 749 "]
    assert all(
        any(msg in r.message and r.levelname == "INFO" for r in caplog.records)
        for msg in expected_messages
    )

def test_detect_outliers_invalid_tolerance(setup_trajectory_data):
    with pytest.raises(
        InputError,
            match=r"The tolerance has to be greater than 1 for meaningful results output",
    ):
        detect_outliers(setup_trajectory_data, tolerance=0.1, deleting=True)

def test_detect_outliers_invalid_quantile(setup_trajectory_data):
    with pytest.raises(
        InputError,
        match=
            r"The quantile of all distances between following points, that is used "
            "as a guideline for the expected distance between two points to locate "
            "outlier should be less that one 1 but at least 0.5",

    ):
        detect_outliers(setup_trajectory_data, quantile=0.1)

def test_detect_outliers_invalid_percentage(setup_trajectory_data):
    with pytest.raises(
        InputError,
        match=r"A value for a percentage has to be between 0 and 100",
    ):
        detect_outliers(setup_trajectory_data, percentage_invalid=-1)

def test_detect_outliers_invalid_max_length(setup_trajectory_data):
    with pytest.raises(
        InputError,
        match=f"The maximum length of an outlier has to be positive and greater than 1 "
            f"to perform the expected outlier detection"
    ):
        detect_outliers(setup_trajectory_data, max_length=0)

def test_detect_outliers_invalid_critical(setup_trajectory_data):
    with pytest.raises(
        InputError,
        match=f"The minimal length, that a trajectory has to have, has to be greater than 1. "
    ):
        detect_outliers(setup_trajectory_data, critical=1)