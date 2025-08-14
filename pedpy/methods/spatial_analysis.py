"""Module containing functions to compute spatial analysis methods.

For example: the pair distribution function.
"""

import warnings
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.distance import cdist

from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData


def compute_pair_distribution_function(
    *,
    traj_data: TrajectoryData,
    radius_bin_size: float,
    randomisation_stacking: int = 1,
) -> Tuple[npt.NDArray[np.float16], npt.NDArray[np.float16]]:
    """Computes the pair distribution function g(r).

    This function calculates the spatial distribution of positions :math:`g(r)`
    :math:`g(r)` here referred to the Euclidean distance between interacting
    pedestrians, i.e., pedestrians that are in the same walkable area at the
    same moment. The pdf is given by the probability that two pedestrians are
    separated by :math:`r` normalized by the probability :math:`PNI(r)` that
    two non-interacting pedestrians are separated by :math:`r`, specifically

    .. math::
        g(r) = P(r)/PNI(r),

    Args:
        traj_data: TrajectoryData, an object containing the trajectories.
        radius_bin_size: float, the size of the bins for the radial
            distribution function in the same units as the positions.
        randomisation_stacking: int, Number of time the dataset will be
            stacked before being randomly shuffled to exact distances of
            non-interacting pedestrians. Larger stacking number will lead to
            closer approximation of true pairwise distribution of non-
            interacting pedestrians but with also increase computation cost.


    Returns:
        A tuple of two numpy arrays. The first array contains the bin edges
        (excluding the first bin edge), and the second array contains the values
        of the pair-distribution function :math:`g(r)` for each bin.
    """
    data_df = traj_data.data

    # Create Dataframe with all pairwise distances
    pairwise_dist_array = _calculate_pair_distances(data_df)

    # Concatenate the working dataframe (data_df) to match the number of
    # randomization cycles
    concatenated_random_df = pd.concat(
        [data_df] * randomisation_stacking, ignore_index=True
    )
    # Scramble time-information to mitigate finite-size effects and calculate
    # pairwise distances of scrambled dataset
    concatenated_random_df.frame = concatenated_random_df.frame.sample(
        frac=1
    ).reset_index(drop=True)
    pairwise_dist_ni_array = _calculate_pair_distances(concatenated_random_df)

    ## Create the bin for data
    radius_bins = np.arange(0, pairwise_dist_array.max(), radius_bin_size)

    # Calculate pair distribution: g(r)
    ## Actual distribution
    pd_bins = pd.cut(pairwise_dist_array, radius_bins)
    pd_bins_normalised = (pd_bins.value_counts().sort_index().to_numpy()) / len(
        pairwise_dist_array
    )  # Normalising by the number of pairwise distances in the dataframe
    ## Scrambled distribution
    pd_ni_bins = pd.cut(pairwise_dist_ni_array, radius_bins)
    pd_ni_bins_normalised = (
        pd_ni_bins.value_counts().sort_index().to_numpy()
    ) / len(
        pairwise_dist_ni_array
    )  # Normalising by the number of pairwise distances in the dataframe

    # Suppress warnings
    warnings.filterwarnings("ignore")

    try:
        with np.errstate(divide="raise"):
            pair_distribution = pd_bins_normalised / pd_ni_bins_normalised
        warnings.filterwarnings("default")  # reset warning-values

    except FloatingPointError:
        warnings.filterwarnings("default")  # reset warning-values
        pair_distribution = pd_bins_normalised / pd_ni_bins_normalised
        warning_message = (
            "Random probability distribution contains null values,"
            + "try using larger dx or more randomization cycles."
        )
        warnings.warn(warning_message, stacklevel=2)

    return radius_bins[1:], pair_distribution


def _calculate_pair_distances(
    data_df: pd.DataFrame,
) -> npt.NDArray[np.float16]:
    """Calculates the pairwise distances of pedestrians.

    This function calculates the pairwise Euclidean distances between all
    pedestrian positions and returns an array containning Euclidean distances
    between every possible pair of pedestrian positions at every time frame.

    Args:
        data_df: pandas.DataFrame, a DataFrame containing pedestrian positions,
            where columns are
        identified by FRAME_COL, ID_COL, X_COL, and Y_COL constants.

    Returns:
        npt.NDArray[np.float16]: A 1D numpy array of pairwise distances.
    """
    distances_list = []

    for _, frame_df in data_df.groupby(FRAME_COL):
        number_pedestrians = len(frame_df[ID_COL].unique())
        if number_pedestrians > 1:
            x_values = frame_df[X_COL].to_numpy()
            y_values = frame_df[Y_COL].to_numpy()
            coordinates = np.stack((x_values, y_values), axis=-1)
            # Calculate pairwise distances for the current frame using cdist
            frame_distances = cdist(
                coordinates, coordinates, metric="euclidean"
            )

            # Extract the upper triangle without the diagonal
            distances_upper_triangle = frame_distances[
                np.triu_indices_from(frame_distances, k=1)
            ]

            distances_list.extend(distances_upper_triangle)

    return np.array(distances_list)
