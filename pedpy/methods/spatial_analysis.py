"""Module containing functions to compute spatial analysis methods.

For example: the pair distribution function, and the intrusion and
avoidance numbers of Cordes et al., PNAS Nexus 2024
(https://doi.org/10.1093/pnasnexus/pgae120).
"""

import warnings
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from scipy.spatial.distance import cdist

from pedpy.column_identifier import (
    AVOIDANCE_COL,
    FRAME_COL,
    ID_COL,
    INTRUSION_COL,
    X_COL,
    Y_COL,
)
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.method_utils import (
    SpeedCalculation,
    _check_trajectory_data,
    compute_individual_distances,
)
from pedpy.methods.speed_calculator import compute_individual_speed


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
    _check_trajectory_data(traj_data)
    data_df = traj_data.data

    # Create Dataframe with all pairwise distances
    pairwise_dist_array = _calculate_pair_distances(data_df)

    # Concatenate the working dataframe (data_df) to match the number of
    # randomization cycles
    concatenated_random_df = pd.concat([data_df] * randomisation_stacking, ignore_index=True)
    # Scramble time-information to mitigate finite-size effects and calculate
    # pairwise distances of scrambled dataset
    concatenated_random_df.frame = concatenated_random_df.frame.sample(frac=1).reset_index(drop=True)
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
    pd_ni_bins_normalised = (pd_ni_bins.value_counts().sort_index().to_numpy()) / len(
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
            frame_distances = cdist(coordinates, coordinates, metric="euclidean")

            # Extract the upper triangle without the diagonal
            distances_upper_triangle = frame_distances[np.triu_indices_from(frame_distances, k=1)]

            distances_list.extend(distances_upper_triangle)

    return np.array(distances_list)


class IntrusionMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier of method to compute the intrusion."""

    MAX = "max"
    """Use the max value in neighborhood as intrusion."""
    SUM = "sum"
    """Use the sum of all values in neighborhood as intrusion."""


def compute_intrusion(
    *,
    traj_data: TrajectoryData,
    r_soc: float = 0.8,
    l_min: float = 0.2,
    method: IntrusionMethod = IntrusionMethod.SUM,
    cap: Optional[float] = 400.0,
) -> pd.DataFrame:
    r"""Compute the intrusion number for each pedestrian per frame.

    The intrusion variable :math:`\mathcal{I}n_i` quantifies how much
    other agents encroach on pedestrian *i*'s personal space
    (Cordes et al. 2024, Eq. 1):

    .. math::

        \mathcal{I}n_i = \sum_{j \in \mathcal{N}_i}
        \left(\frac{r_\text{soc} - \ell_\text{min}}
        {r_{ij} - \ell_\text{min}}\right)^{k_I},

    with :math:`k_I = 2`. The neighbor set :math:`\mathcal{N}_i` contains
    all agents *j* with :math:`r_{ij} \leq 3\,r_\text{soc}`.

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        r_soc (float): social radius in m (default 0.8)
        l_min (float): pedestrian diameter in m (default 0.2)
        method (IntrusionMethod): aggregation over the neighbors.
            ``SUM`` is the definition of the paper (Eq. 1): the
            encroachments of all neighbors add up, as mechanical forces
            do. ``MAX`` keeps only the strongest single encroachment, so
            it does not grow with the number of neighbors; it is offered
            for comparison and is not the paper's definition.
        cap (float, optional): upper bound applied to every per-pair term
            before aggregation (default 400, the paper's value, which
            limits artifacts from apparent overlaps of the discs in
            noisy trajectories). Pairs at or below ``l_min`` distance,
            where Eq. 1 diverges, take the cap. ``None`` disables the
            cap; such pairs are then infinite.

    Returns:
        DataFrame with columns 'id', 'frame', and 'intrusion'.
        Agents with no neighbors within the cutoff are absent from
        the result.
    """
    intrusion = compute_individual_distances(traj_data=traj_data)
    intrusion = intrusion.loc[intrusion.distance <= 3 * r_soc].copy()
    with np.errstate(divide="ignore"):
        intrusion.loc[:, INTRUSION_COL] = np.where(
            intrusion.distance > l_min,
            ((r_soc - l_min) / (intrusion.distance - l_min)) ** 2,
            np.inf,
        )
    if cap is not None:
        intrusion.loc[:, INTRUSION_COL] = np.minimum(intrusion[INTRUSION_COL], cap)
    intrusion = (
        intrusion.groupby(by=[ID_COL, FRAME_COL])
        .agg(
            intrusion=(INTRUSION_COL, method.value),
        )
        .reset_index()
    )

    return intrusion


def compute_avoidance(
    *,
    traj_data: TrajectoryData,
    frame_step: int,
    l_min: float = 0.2,
    tau_0: float,
    cap: Optional[float] = 60.0,
) -> pd.DataFrame:
    r"""Compute the avoidance number for each pedestrian per frame.

    The avoidance variable :math:`\mathcal{A}v_i` quantifies the
    imminence of collisions that pedestrian *i* faces, based on
    the time-to-collision (TTC) with neighbors
    (Cordes et al. 2024, Eq. 2):

    .. math::

        \mathcal{A}v_i = \sum_{j \in \mathcal{N}'_i}
        \left(\frac{\tau_0}{\tau_{ij}}\right)^{k_A},

    with :math:`k_A = 1`. Unlike Eq. 1, the neighbor set
    :math:`\mathcal{N}'_i` contains only the neighbor with the shortest
    time-to-collision :math:`\tau_{ij}`, the most imminent risk. The sum
    therefore has a single term, and because :math:`\tau_0 / \tau_{ij}`
    decreases with :math:`\tau_{ij}`, :math:`\mathcal{A}v_i` equals the
    largest :math:`\tau_0 / \tau_{ij}` over all neighbors *j* in the same
    frame. That maximum is what this function computes. The TTC is the
    time until the discs of diameter ``l_min`` around *i* and *j* first
    touch if both keep their current velocity; pairs that are not on a
    collision course have :math:`\tau_{ij} = \infty` and contribute 0.

    Pairs whose discs already touch or overlap (:math:`r_{ij} \leq \ell`)
    are in collision now, :math:`\tau_{ij} = 0`, and take the cap.

    Args:
        traj_data (TrajectoryData): trajectory data to analyze
        frame_step (int): number of frames used for velocity computation
        l_min (float): pedestrian diameter :math:`\ell` in m at which
            two pedestrians touch (default 0.2, as the paper uses for
            recorded trajectories; its simulations use
            :math:`\ell_\text{soc} = 0.4` m instead)
        tau_0 (float): reference timescale in s (paper uses 3.0)
        cap (float, optional): upper bound applied to every per-pair term
            before taking the maximum (default 60, the paper's value,
            which limits artifacts from apparent overlaps of the discs
            in noisy trajectories). ``None`` disables the cap; touching
            or overlapping pairs are then infinite.

    Returns:
        DataFrame with columns 'id', 'frame', and 'avoidance'
    """
    velocity = compute_individual_speed(
        traj_data=traj_data,
        frame_step=frame_step,
        compute_velocity=True,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    data = traj_data.data.merge(velocity, on=[ID_COL, FRAME_COL])
    data["velocity"] = shapely.points(data.v_x, data.v_y)

    matrix = data.merge(data, how="inner", on=FRAME_COL, suffixes=("", "_neighbor"))
    matrix = matrix[matrix[ID_COL] != matrix[f"{ID_COL}_neighbor"]]

    pos = shapely.get_coordinates(matrix.point)
    pos_neighbor = shapely.get_coordinates(matrix.point_neighbor)
    distance = np.linalg.norm(pos - pos_neighbor, axis=1)

    delta_v = shapely.get_coordinates(matrix.velocity) - shapely.get_coordinates(matrix.velocity_neighbor)
    delta_v_norm = np.linalg.norm(delta_v, axis=1)

    # solve the quadratic only for separated pairs with a relative velocity
    computable = (distance > l_min) & (delta_v_norm > 0)

    ttc = np.full(matrix.shape[0], np.inf)
    ttc[distance <= l_min] = 0.0  # already in contact

    if np.any(computable):
        d_c = distance[computable]
        dv_c = delta_v[computable]
        dv_norm_c = delta_v_norm[computable]

        e_v = (pos[computable] - pos_neighbor[computable]) / d_c[:, np.newaxis]
        v_rel_hat = dv_c / dv_norm_c[:, np.newaxis]
        cos_alpha = np.sum(e_v * v_rel_hat, axis=1)

        capital_a = (cos_alpha**2 - 1) * d_c**2 + l_min**2
        sqrt_a_safe = np.sqrt(np.maximum(capital_a, 0.0))

        valid = (capital_a >= 0) & (-cos_alpha * d_c - sqrt_a_safe >= 0)

        idx = np.where(computable)[0][valid]
        ttc[idx] = (-cos_alpha[valid] * d_c[valid] - sqrt_a_safe[valid]) / dv_norm_c[valid]

    matrix = matrix.copy()
    with np.errstate(divide="ignore"):
        matrix[AVOIDANCE_COL] = tau_0 / ttc
    if cap is not None:
        matrix[AVOIDANCE_COL] = np.minimum(matrix[AVOIDANCE_COL], cap)

    avoidance = matrix.groupby(by=[ID_COL, FRAME_COL], as_index=False).agg(avoidance=(AVOIDANCE_COL, "max"))
    return avoidance
