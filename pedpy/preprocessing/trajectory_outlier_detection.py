import copy
import logging
from enum import Enum
from itertools import chain

import numpy as np
import pandas as pd

from pedpy import InputError
from pedpy.data.trajectory_data import TrajectoryData

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
_log = logging.getLogger(__name__)


def detect_outliers(
    traj_data: TrajectoryData,
    tolerance: float = 3.0,
    quantile: float = 0.99,
    percentage_invalid: float = 0.2,
    deleting: bool = False,
    max_length: int = 8,
    critical: int = None,
    drops_only: bool = False,
) -> tuple[TrajectoryData, list[int], list[int]]:
    """

    Args:
        traj_data:
        tolerance: The tolerance equals to the factor the quantile of the distance is
            multiplied with. A low value means a low tolerance for potential outliers,
            which can be useful in trajectories where the speed of the pedestrians
            stays in a similar range. If the pedestrians speed variates, for
            example in bottleneck experiments, the tolerance should be chosen larger.
        quantile: The value, that is used as the guideline for the expected distance
            between 2 points is calculated by the quantile of all distances in the
            whole trajectory. A high quantile is recommended.
        percentage_invalid: If more that percentage_invalid % of the trajectory was
            detected as an outlier, the trajectory cannot be corrected properly and
            is considered as completely invalid.
        deleting: A bool parameter whether completely invalid trajectories should
            be deleted or not.
        max_length: Sometimes it happens, that a few outliers are directly after another
            without a jump back to the correct trajectory. The max_length value defines,
            how many points long those multiple outliers can be, before the program will
            check, whether it is a split in the trajectory or, if the outlier occur at
            the beginning of the trajectory, the points before it will be treated as
            outliers. The default value is 8.
        critical: Only relevant in cases, where anomalies occur in the beginning of a
            trajectory and the previous outlier finds need to be checked again, also
            critical length that a trajectory has to have. If the lowest position of
            anomalies in the begging is lower than the critical value, the anomalies are
            considered as outlier. Otherwise, it is a split and the points and every point
            after is going to be deleted. The default value is 10% of the length of the
            trajectory.
        drops_only: A bool parameter whether the program should only search and correct
            major jumps within the trajectory, that do not have a jump back. This includes
            outlier group that contain the very first or the very last frame of the person
            and drops in the middle of the trajectory, where the tracker caused problems.

    Returns:
        The corrected and modified copy of the original trajectory as pedpy.TrajectoryData.
    """
    _check_parameters(
        tolerance=tolerance,
        quantile=quantile,
        percentage_invalid=percentage_invalid,
        max_length=max_length,
        critical=critical,
    )
    all_single_trajectories, all_distances = _concat_traj_data(traj_data)
    median = float(np.median(all_distances))
    quan = float(np.quantile(all_distances, quantile))
    single_trajectories_to_keep = []
    index_original = []

    for i in range(len(all_single_trajectories)):
        trajectory_single = all_single_trajectories[i]
        if critical is None:
            critical = int(0.1 * len(trajectory_single))
        outliers, deal = _detect_single_outliers(
            trajectory_single=trajectory_single,
            quantile=quan,
            tolerance=tolerance,
            critical=critical,
            max_length=max_length,
            drops_only=drops_only,
        )
        if len(outliers) > 0:
            if (
                len(list(chain.from_iterable(outliers))) > percentage_invalid * len(trajectory_single)
                or deal == DealTrajectory.delete
            ):
                _log.info(f"Trajectory with personID {i + 1} has to many invalid points and cannot be corrected")
                deal = DealTrajectory.delete
            else:
                if not drops_only:
                    frames = []
                    for following_outliers in outliers:
                        for index in following_outliers:
                            frames.append(trajectory_single.iloc[index, 1].tolist())
                    _log.info(f"Outliers found: personID {i + 1} at frames {frames} ")
            mean_direction = _direction_of_trajectory(_traj_to_xy(trajectory_single))
            match deal:
                case DealTrajectory.delete:
                    if deleting:
                        _log.info(f"Trajectory with personID {i + 1} was deleted")
                        continue
                    else:
                        _log.info(f"Trajectory with personID {i + 1} will be returned unchanged")
                case DealTrajectory.split:
                    if not drops_only:
                        split_index = outliers[-1][0]
                        trajectory_single = trajectory_single[:split_index]
                        outliers = outliers[:-1]
                        trajectory_single = _correct_trajectory(
                            single_trajectory=trajectory_single,
                            outliers=outliers,
                            median=median,
                            mean_direction=mean_direction,
                        )
                    else:
                        trajectory_single = _split_trajectory(
                            trajectory_to_split=trajectory_single, outliers=outliers, person_id=i + 1, critical=critical
                        )
                case DealTrajectory.correct:
                    trajectory_single = _correct_trajectory(
                        single_trajectory=trajectory_single,
                        outliers=outliers,
                        median=median,
                        mean_direction=mean_direction,
                    )
            index_original.append(i + 1)
        single_trajectories_to_keep.append(trajectory_single)
    corrected_trajectory, index_corrected = _to_traj_data(traj_data, single_trajectories_to_keep, index_original)
    return corrected_trajectory, index_original, index_corrected


def _check_parameters(
    tolerance: float,
    quantile: float,
    percentage_invalid: float,
    max_length: int,
    critical: int,
) -> None:
    if tolerance < 1:
        raise InputError(f"The tolerance has to be greater than 1 for meaningful results output")
    if quantile < 0.5 or quantile >= 1:
        raise InputError(
            f"The quantile of all distances between following points, that is used "
            f"as a guideline for the expected distance between two points to locate "
            f"outlier should be less that one 1 but at least 0.5"
        )
    if percentage_invalid > 100 or percentage_invalid < 0:
        raise InputError(f"A value for a percentage has to be between 0 and 100")
    if max_length <= 1:
        raise InputError(
            f"The maximum length of an outlier has to be positive and greater than 1 "
            f"to perform the expected outlier detection"
        )
    if critical != None and critical <= 1:
        raise InputError(f"The minimal length, that a trajectory has to have, has to be greater than 1. ")


def _concat_traj_data(traj_data: TrajectoryData) -> tuple[list[pd.DataFrame], list[int]]:
    trajectory_dataframe = copy.deepcopy(traj_data.data)
    trajectories_per_person = []
    all_distances = []
    for i in range(1, traj_data.number_pedestrians + 1):
        trajectory_single_person = trajectory_dataframe[trajectory_dataframe["id"] == i].copy()
        distances = _calc_distances(trajectory_single_person)
        all_distances.extend(distances)
        distances_to_insert = np.insert(distances, 0, 0)
        trajectory_single_person.loc[:, "distance prev. point"] = distances_to_insert
        trajectories_per_person.append(trajectory_single_person)
    return trajectories_per_person, all_distances


def _split_trajectory(trajectory_to_split, outliers: list[list[int]], person_id: int, critical):
    if len(outliers) == 2:
        trajectory_to_split = trajectory_to_split[outliers[0][-1] + 1 : outliers[1][0]]
        _log.info(
            f"Frames in trajectory with original personID {person_id + 1} were cut out before frame "
            f"{trajectory_to_split.iloc[0, 1]} and after frame "
            f"{trajectory_to_split.iloc[-1, 1]} "
        )
    elif len(outliers) == 1:
        if outliers[0][0] > critical:
            trajectory_to_split = trajectory_to_split[: outliers[0][0]]
            _log.info(
                f"Frames in trajectory with original personID {person_id + 1} were cut out after frame "
                f"{trajectory_to_split.iloc[-1, 1]} "
            )
        else:
            trajectory_to_split = trajectory_to_split[outliers[0][-1] :]
            _log.info(
                f"Frames in trajectory with original personID {person_id + 1} were cut out before frame "
                f"{trajectory_to_split.iloc[0, 1]}"
            )
    else:
        _log.warning(f"Trajectory with personID {person_id + 1} contains uncorrected jumps")
    return trajectory_to_split


def _calc_distances(traj_data: pd.DataFrame) -> np.ndarray:
    x = traj_data["x"].to_numpy()
    y = traj_data["y"].to_numpy()

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    distances = np.sqrt(dx**2 + dy**2)
    return distances


class DealTrajectory(Enum):
    delete = "delete"
    split = "split"
    correct = "correct"


def _detect_single_outliers(
    trajectory_single: pd.DataFrame, quantile: float, tolerance: float, critical: int, max_length: int, drops_only: bool
) -> tuple[list[list[int]], DealTrajectory]:
    distances = trajectory_single["distance prev. point"].to_numpy()
    outliers = []
    for i in range(len(distances)):
        # if the points current points were already tested by the multiple_outlier function
        # it could cause mistakes to check them again because of the different algorithms used
        # for single and for multiple outliers
        if len(outliers) > 0 and outliers[-1][-1] + 1 >= i:
            continue
        if (
            distances[i] > tolerance * quantile
        ):  # Testing, whether the distance between to points is abnormal -> sign of outlier
            following_outliers = []
            if i == 1:
                # if the outlier is at the second distance, it is not clear if the first or second point is incorrect,
                # and it needs to be tested further.
                if _outliers_beginning(trajectory_single, quantile, tolerance):
                    following_outliers.append(1)
                    if i + 1 < len(distances):
                        _detect_multiple_outliers(quantile, trajectory_single, following_outliers)
                else:
                    following_outliers.append(0)
            else:
                following_outliers.append(i)
                if i + 1 < len(distances):
                    _detect_multiple_outliers(quantile, trajectory_single, following_outliers)

            if following_outliers[0] < critical and len(following_outliers) > max_length:
                # The list of following_outliers exceeds the maximum length for outliers and is in the
                # beginning of the trajectory, therefore the points before the supposed outliers are
                # considered as the actual anomalies
                following_outliers = [i for i in range(following_outliers[0])]
                if following_outliers[0] == 0 and len(outliers) > 0 and outliers[0] == [0]:
                    # This case only happen, when the second point (and further) is an outlier, but the first one
                    # is correct and there is a split within the trajectory. In this case, the logic of the even/ uneven number of
                    # outliers is broken and those concrete cases need to be handled again
                    del following_outliers[0]
                    del outliers[0]
                outliers = [group for group in outliers if not any(x in following_outliers for x in group)]
            # following_outliers[0]<critical and len(following_outliers)< max_length
            # -> are considered outliers, because the index list it not long enough to count as a trajectory
            elif following_outliers[0] > critical and len(following_outliers) > max_length:
                # Recognized and treated as a split
                deal = DealTrajectory.split
                outliers.append(following_outliers)
                if following_outliers[0] < critical:
                    deal = DealTrajectory.delete
                if drops_only:
                    if len(outliers) > 0:
                        outliers = [group for group in outliers if 0 in group]
                        outliers.append(following_outliers)

                return outliers, deal
            outliers.append(following_outliers)
    deal = DealTrajectory.correct
    if drops_only and len(outliers) > 0:
        outliers = [group for group in outliers if 0 in group or len(distances) - 1 in group]
        deal = DealTrajectory.split
    return outliers, deal


def _detect_multiple_outliers(quantile: float, traj_data: pd.DataFrame, following_outliers: list[int]) -> None:
    """
    Searches for outliers directly after a single outlier.
    As the distance between multiple outliers directly after another do not necessarily has to be
    larger that the expected distance between two points, the distance last correct point is taken
    as a reference. The tolerance is very low to avoid points to be accidental "right" again.
    """
    correct_point = np.array(
        [traj_data.iloc[following_outliers[-1] - 1]["x"], traj_data.iloc[following_outliers[-1] - 1]["y"]]
    )  # The last correct point is the point before the outlier
    following_point = np.array(
        [traj_data.iloc[following_outliers[-1] + 1]["x"], traj_data.iloc[following_outliers[-1] + 1]["y"]]
    )  # The outlier is already tested and documented, so we'll proceed with the point afterward
    dist_to_corr_point = np.linalg.norm(correct_point - following_point)
    n = 2  # As we test with the correct point, which is not the direct point before the tested point, we'll multiply the critical distance correspondingly often
    index = following_outliers[-1] + 1
    tolerance = 1.5
    while dist_to_corr_point > quantile * n * tolerance:
        following_outliers.append(index)
        index += 1
        if index == len(traj_data):
            break
        n += 1
        following_point = np.array([traj_data.iloc[index]["x"], traj_data.iloc[index]["y"]])
        dist_to_corr_point = np.linalg.norm(correct_point - following_point)


def _outliers_beginning(trajectory_single: pd.DataFrame, quantile: float, tolerance: float) -> bool:
    """
    Returns True if the first point is correct and False if the first point is an outlier.
    """
    distances = trajectory_single["distance prev. point"]
    index_outliers = np.where(distances > tolerance * quantile)[0]
    if len(index_outliers) % 2:  # Uneven Number, first or last point is an outlier, not both
        return len(trajectory_single) - 1 in index_outliers  # if the number of jumps is uneven, it still can be that
        # the second point is the outlier, when the last point is an outlier too
    else:  # Even Number of outliers: Either the first and last point is an outlier, or none
        return (
            len(trajectory_single) - 1 not in index_outliers
        )  # if the last step of the person is an outlier, there is
        # still an even number of outliers even when the first point is incorrect (an outlier)


def _traj_to_xy(single_traj_data: pd.DataFrame) -> np.ndarray:
    return np.column_stack([single_traj_data["x"].to_numpy(), single_traj_data["y"].to_numpy()])


def _direction_of_trajectory(points: np.ndarray) -> np.ndarray:
    vectors = points[:-1] - points[1:]
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms != 0
    unit_vectors = vectors[valid] / norms[valid][:, None]
    if unit_vectors.shape[0] == 0:
        return np.array([0.0, 0.0])
    mean_direction = np.mean(unit_vectors, axis=0)
    return mean_direction / np.linalg.norm(mean_direction)


def _correct_trajectory(
    single_trajectory: pd.DataFrame, outliers: list[list[int]], median: float, mean_direction: np.ndarray
) -> pd.DataFrame:
    index_outliers = []
    for following_outliers in outliers:
        index_outliers.append(single_trajectory.index[following_outliers])
        single_trajectory = _correct_outliers(single_trajectory, following_outliers, median, mean_direction)

    return single_trajectory  #


def _correct_outliers(
    traj_single: pd.DataFrame, outlier_index: list[int], median: float, mean_direction: np.ndarray
) -> pd.DataFrame:
    if outlier_index[-1] + 1 == len(traj_single):
        _false_points_at_end(median, traj_single, outlier_index, -mean_direction)
    elif outlier_index[0] == 0:
        _false_points_at_beginning(median, traj_single, outlier_index, mean_direction)
    else:
        last_correct_point = np.array(
            [traj_single.iloc[outlier_index[0] - 1]["x"], traj_single.iloc[outlier_index[0] - 1]["y"]]
        )
        first_correct_point = np.array(
            [traj_single.iloc[outlier_index[-1] + 1]["x"], traj_single.iloc[outlier_index[-1] + 1]["y"]]
        )
        interpolated_points = np.linspace(first_correct_point, last_correct_point, len(outlier_index))
        index_labels = traj_single.index[outlier_index]
        traj_single.loc[index_labels, "x"] = interpolated_points[:, 0]
        traj_single.loc[index_labels, "y"] = interpolated_points[:, 1]
    return traj_single


def _false_points_at_end(
    median: float, traj_single: pd.DataFrame, outlier_index: list[int], mean_direction: np.ndarray
) -> pd.DataFrame:
    last_correct_point = np.array(
        [traj_single.iloc[outlier_index[0] - 1]["x"], traj_single.iloc[outlier_index[0] - 1]["y"]]
    )
    last_traj_point = last_correct_point + mean_direction * len(outlier_index) * median
    insert_points = np.linspace(last_correct_point, last_traj_point, len(outlier_index) - 1)
    interpolated_points = np.vstack([insert_points, last_traj_point])
    index_labels = traj_single.index[outlier_index]
    traj_single.loc[index_labels, "x"] = interpolated_points[:, 0]
    traj_single.loc[index_labels, "y"] = interpolated_points[:, 1]
    return traj_single


def _false_points_at_beginning(
    median: float, traj_single: pd.DataFrame, outlier_index: list[int], mean_direction: np.ndarray
) -> pd.DataFrame:
    last_correct_point = np.array(
        [traj_single.iloc[outlier_index[-1] + 1]["x"], traj_single.iloc[outlier_index[-1] + 1]["y"]]
    )
    first_traj_point = last_correct_point + mean_direction * len(outlier_index) * median
    insert_points = np.linspace(first_traj_point, last_correct_point, len(outlier_index) - 1)
    interpolated_points = np.vstack([first_traj_point, insert_points])
    index_labels = traj_single.index[outlier_index]
    traj_single.loc[index_labels, "x"] = interpolated_points[:, 0]
    traj_single.loc[index_labels, "y"] = interpolated_points[:, 1]
    return traj_single


def _to_traj_data(
    original_traj_data: TrajectoryData, single_trajectories_to_keep: list[pd.DataFrame], index_original: list[int]
) -> tuple[TrajectoryData, list[int]]:
    index_corrected = []
    for i in range(len(single_trajectories_to_keep)):
        single_trajectories_to_keep[i] = single_trajectories_to_keep[i].drop(columns=["distance prev. point"])
        if single_trajectories_to_keep[i].iloc[0, 0] in index_original:
            index_corrected.append(i + 1)
        if (
            single_trajectories_to_keep[i].iloc[0, 0] != i + 1
        ):  # i+1 corresponds to the current right personID in the file
            single_trajectories_to_keep[i].iloc[:, 0] = i + 1
    trajectory_data = pd.concat(single_trajectories_to_keep, ignore_index=True)
    modified_traj_data = TrajectoryData(data=trajectory_data, frame_rate=original_traj_data.frame_rate)
    return modified_traj_data, index_corrected
