"""Module containing functions to compute spatial analysis methods.

Includes the pair distribution function :math:`g(r)` with time or spatial
randomization and the rate-of-approach pair distribution function :math:`g(v)`
/ :math:`g(r, v)`.
"""

import warnings
from enum import Enum, auto
from typing import Final, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from scipy.spatial.distance import cdist

from pedpy.column_identifier import FRAME_COL, ID_COL, V_X_COL, V_Y_COL, X_COL, Y_COL
from pedpy.data.geometry import WalkableArea
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import PedPyTypeError, PedPyValueError, SpeedError
from pedpy.methods.method_utils import _check_trajectory_data

_MIN_PAIR_DISTANCE: Final = 1e-12
_REJECTION_OVERSAMPLE: Final = 2
_MIN_SAMPLE_BATCH: Final = 256
_MAX_SAMPLING_ITERATIONS: Final = 10_000


class PairNormalizationMethod(Enum):  # pylint: disable=too-few-public-methods
    """Normalization ensemble for pair-distribution calculations.

    Attributes:
        TIME_RANDOMIZATION: Estimate non-interacting pair statistics by
            shuffling frame timestamps across the trajectory dataset (default).
        SPATIAL_RANDOMIZATION: Estimate non-interacting pair statistics by
            sampling positions uniformly at random inside a ``WalkableArea``
            (suited for dense static or semi-static crowds).
    """

    TIME_RANDOMIZATION = auto()
    SPATIAL_RANDOMIZATION = auto()


def _sample_uniform_points_in_walkable_area(
    walkable_area: WalkableArea,
    n_points: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Sample ``n_points`` 2D coordinates uniformly inside ``walkable_area``."""
    if walkable_area.area <= 0.0:
        raise PedPyValueError("'walkable_area' must have positive area for spatial randomization.")
    if n_points == 0:
        return np.empty((0, 2), dtype=np.float64)

    poly = walkable_area.polygon
    shapely.prepare(poly)
    min_x, min_y, max_x, max_y = walkable_area.bounds

    accepted: list[npt.NDArray[np.float64]] = []
    collected = 0
    for _ in range(_MAX_SAMPLING_ITERATIONS):
        needed = n_points - collected
        batch_size = max(needed * _REJECTION_OVERSAMPLE, _MIN_SAMPLE_BATCH)
        xs = rng.uniform(min_x, max_x, size=batch_size)
        ys = rng.uniform(min_y, max_y, size=batch_size)
        inside = shapely.contains_xy(poly, xs, ys)
        if np.any(inside):
            pts = np.column_stack((xs[inside], ys[inside]))
            accepted.append(pts)
            collected += len(pts)
            if collected >= n_points:
                return np.vstack(accepted)[:n_points]

    raise PedPyValueError("Failed to sample uniform points inside 'walkable_area'.")


def _build_randomized_dataframe(
    data_df: pd.DataFrame,
    randomisation_stacking: int,
    normalization_method: PairNormalizationMethod,
    walkable_area: Optional[WalkableArea],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create the non-interacting reference DataFrame for pair normalization."""
    if randomisation_stacking < 1:
        raise PedPyValueError(f"'randomisation_stacking' must be >= 1, got {randomisation_stacking}.")
    if not isinstance(normalization_method, PairNormalizationMethod):
        raise PedPyValueError(f"Unsupported normalization method: {normalization_method!r}.")

    random_df = pd.concat([data_df] * randomisation_stacking, ignore_index=True)

    if normalization_method == PairNormalizationMethod.TIME_RANDOMIZATION:
        permuted_idx = rng.permutation(len(random_df))
        random_df[FRAME_COL] = random_df[FRAME_COL].to_numpy()[permuted_idx]
        return random_df

    if walkable_area is None:
        raise PedPyValueError(
            "'walkable_area' must be provided when using PairNormalizationMethod.SPATIAL_RANDOMIZATION."
        )
    if not isinstance(walkable_area, WalkableArea):
        raise PedPyTypeError(
            f"Expected 'walkable_area' to be a WalkableArea, got {type(walkable_area).__name__!r} instead."
        )

    sampled_xy = _sample_uniform_points_in_walkable_area(walkable_area, len(random_df), rng)
    random_df[X_COL] = sampled_xy[:, 0]
    random_df[Y_COL] = sampled_xy[:, 1]
    if V_X_COL in random_df.columns and V_Y_COL in random_df.columns and len(random_df) > 0:
        vel_perm = rng.permutation(len(random_df))
        random_df[V_X_COL] = random_df[V_X_COL].to_numpy()[vel_perm]
        random_df[V_Y_COL] = random_df[V_Y_COL].to_numpy()[vel_perm]

    return random_df


def _safe_divide_distributions(
    actual_prob: npt.NDArray[np.float64],
    ni_prob: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Divide ``actual_prob / ni_prob`` while warning on zero reference bins."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = actual_prob / ni_prob
    if np.any(ni_prob == 0):
        warnings.warn(
            "Random probability distribution contains null values, "
            "try using larger bin sizes or more randomization cycles.",
            stacklevel=3,
        )
    return ratio


def compute_pair_distribution_function(
    *,
    traj_data: TrajectoryData,
    radius_bin_size: float,
    randomisation_stacking: int = 1,
    normalization_method: PairNormalizationMethod = PairNormalizationMethod.TIME_RANDOMIZATION,
    walkable_area: Optional[WalkableArea] = None,
    seed: Optional[int] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Computes the pair distribution function :math:`g(r)`.

    Calculates the spatial pair distribution :math:`g(r) = P(r) / P_{NI}(r)` of
    Euclidean distances :math:`r` between pedestrians present in the same frame,
    normalized by the non-interacting pair distance distribution :math:`P_{NI}(r)`
    obtained either via frame-timestamp shuffling
    (:attr:`PairNormalizationMethod.TIME_RANDOMIZATION`) or uniform spatial
    sampling inside ``walkable_area``
    (:attr:`PairNormalizationMethod.SPATIAL_RANDOMIZATION`).

    Args:
        traj_data (TrajectoryData): TrajectoryData object containing positions.
        radius_bin_size (float): Radial bin width in position units.
        randomisation_stacking (int): Number of dataset replications used to
            estimate the non-interacting pair distribution.
        normalization_method (PairNormalizationMethod): Method used to construct
            the non-interacting reference distribution (default:
            ``PairNormalizationMethod.TIME_RANDOMIZATION``).
        walkable_area (WalkableArea, optional): Walkable area required when
            ``normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION``.
        seed (int, optional): Random seed for reproducible randomization.

    Returns:
        Tuple ``(bin_edges, g_r)`` where ``bin_edges`` are the upper radial bin
        edges and ``g_r`` are the corresponding :math:`g(r)` values.
    """
    _check_trajectory_data(traj_data)
    if radius_bin_size <= 0:
        raise PedPyValueError(f"'radius_bin_size' must be > 0, got {radius_bin_size}.")

    rng = np.random.default_rng(seed)
    data_df = traj_data.data[[ID_COL, FRAME_COL, X_COL, Y_COL]]

    pairwise_dist_array = _calculate_pair_distances(data_df)
    if len(pairwise_dist_array) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    random_df = _build_randomized_dataframe(
        data_df=data_df,
        randomisation_stacking=randomisation_stacking,
        normalization_method=normalization_method,
        walkable_area=walkable_area,
        rng=rng,
    )
    pairwise_dist_ni_array = _calculate_pair_distances(random_df)

    max_dist = float(np.max(pairwise_dist_array))
    stop_edge = max_dist if max_dist > radius_bin_size else radius_bin_size + max_dist
    radius_bins = np.arange(0.0, stop_edge, radius_bin_size, dtype=np.float64)

    pd_bins = pd.cut(pairwise_dist_array, radius_bins)
    pd_bins_normalised = pd_bins.value_counts().sort_index().to_numpy(dtype=np.float64) / len(pairwise_dist_array)

    pd_ni_bins = pd.cut(pairwise_dist_ni_array, radius_bins)
    pd_ni_bins_normalised = pd_ni_bins.value_counts().sort_index().to_numpy(dtype=np.float64) / len(
        pairwise_dist_ni_array
    )

    pair_distribution = _safe_divide_distributions(pd_bins_normalised, pd_ni_bins_normalised)
    return radius_bins[1:], pair_distribution


def _calculate_pair_distances(
    data_df: pd.DataFrame,
) -> npt.NDArray[np.float64]:
    """Calculates pairwise Euclidean distances across all frames in ``data_df``."""
    distances_list = []

    for _, frame_df in data_df.groupby(FRAME_COL):
        if len(frame_df) > 1:
            coordinates = frame_df[[X_COL, Y_COL]].to_numpy(dtype=np.float64)
            frame_distances = cdist(coordinates, coordinates, metric="euclidean")
            distances_upper_triangle = frame_distances[np.triu_indices_from(frame_distances, k=1)]
            distances_list.append(distances_upper_triangle)

    if not distances_list:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(distances_list)


def _calculate_pair_distances_and_approach_rates(
    data_df: pd.DataFrame,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Calculate pairwise distances :math:`r_{ij}` and approach rates :math:`-\dot{r}_{ij}`."""
    dist_chunks: list[npt.NDArray[np.float64]] = []
    rate_chunks: list[npt.NDArray[np.float64]] = []

    for _, frame_df in data_df.groupby(FRAME_COL):
        n_peds = len(frame_df)
        if n_peds > 1:
            coords = frame_df[[X_COL, Y_COL]].to_numpy(dtype=np.float64)
            vels = frame_df[[V_X_COL, V_Y_COL]].to_numpy(dtype=np.float64)
            idx_i, idx_j = np.triu_indices(n_peds, k=1)

            dx = coords[idx_i, 0] - coords[idx_j, 0]
            dy = coords[idx_i, 1] - coords[idx_j, 1]
            dvx = vels[idx_i, 0] - vels[idx_j, 0]
            dvy = vels[idx_i, 1] - vels[idx_j, 1]

            r_pair = np.hypot(dx, dy)
            dot_rv = dx * dvx + dy * dvy
            v_pair = np.divide(
                -dot_rv,
                r_pair,
                out=np.zeros_like(r_pair),
                where=r_pair > _MIN_PAIR_DISTANCE,
            )
            dist_chunks.append(r_pair)
            rate_chunks.append(v_pair)

    if not dist_chunks:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    return np.concatenate(dist_chunks), np.concatenate(rate_chunks)


def _resolve_approach_rate_edges(
    v_values: npt.NDArray[np.float64],
    approach_rate_bin_size: Optional[float],
    approach_rate_bins: Optional[Sequence[float]],
) -> npt.NDArray[np.float64]:
    """Validate and construct 1D bin edges for rate-of-approach binning."""
    if (approach_rate_bin_size is None) == (approach_rate_bins is None):
        raise PedPyValueError("Provide exactly one of 'approach_rate_bin_size' or 'approach_rate_bins'.")

    if approach_rate_bins is not None:
        edges = np.asarray(approach_rate_bins, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise PedPyValueError("'approach_rate_bins' must be a strictly increasing 1D sequence of >= 2 edges.")
        return edges

    assert approach_rate_bin_size is not None
    if approach_rate_bin_size <= 0:
        raise PedPyValueError(f"'approach_rate_bin_size' must be > 0, got {approach_rate_bin_size}.")

    v_min = float(np.min(v_values))
    v_max = float(np.max(v_values))
    if v_max - v_min < approach_rate_bin_size:
        return np.array([v_min, v_min + approach_rate_bin_size], dtype=np.float64)
    return np.arange(v_min, v_max + approach_rate_bin_size, approach_rate_bin_size, dtype=np.float64)


def compute_approach_rate_pair_distribution(
    *,
    traj_data: TrajectoryData,
    individual_speed: pd.DataFrame,
    approach_rate_bin_size: Optional[float] = None,
    approach_rate_bins: Optional[Sequence[float]] = None,
    radius_bin_size: Optional[float] = None,
    randomisation_stacking: int = 1,
    normalization_method: PairNormalizationMethod = PairNormalizationMethod.TIME_RANDOMIZATION,
    walkable_area: Optional[WalkableArea] = None,
    seed: Optional[int] = None,
) -> Union[
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    r"""Compute the pair distribution function for the rate of approach.

    For each interacting pair :math:`(i, j)` in the same frame, the rate of
    approach is defined as (Karamouzas et al., 2014,
    ``doi:10.1103/PhysRevLett.113.238701``; Cordes et al., 2024,
    ``doi:10.1093/pnasnexus/pgae120``):

    .. math::

        v_{ij} = -\frac{dr_{ij}}{dt}
        = -\frac{(\vec{x}_i - \vec{x}_j) \cdot (\vec{v}_i - \vec{v}_j)}
        {\|\vec{x}_i - \vec{x}_j\|},

    which is positive when pedestrians :math:`i` and :math:`j` move closer
    together and negative when moving apart.

    - When ``radius_bin_size`` is ``None`` (default), evaluates the 1D pair
      distribution :math:`g(v) = P(v) / P_{NI}(v)` and returns
      ``(v_bin_edges, g_v)``.
    - When ``radius_bin_size`` is provided, evaluates the 2D distance /
      rate-of-approach pair distribution :math:`g(r, v) = P(r, v) / P_{NI}(r, v)`
      and returns ``(r_bin_edges, v_bin_edges, g_rv)``.

    Args:
        traj_data (TrajectoryData): TrajectoryData object containing positions.
        individual_speed (pd.DataFrame): Individual velocity DataFrame computed
            with ``compute_velocity=True`` (must contain ``v_x`` and ``v_y``).
        approach_rate_bin_size (float, optional): Bin width for the rate of
            approach in m/s. Mutually exclusive with ``approach_rate_bins``.
        approach_rate_bins (Sequence[float], optional): Explicit strictly
            increasing bin edges for the rate of approach (e.g.
            ``[0.0, 0.5, 1.0, 1.5]``). Mutually exclusive with
            ``approach_rate_bin_size``.
        radius_bin_size (float, optional): If provided, evaluates the 2D
            pair distribution :math:`g(r, v)` using radial bins of this width.
        randomisation_stacking (int): Number of dataset replications used to
            estimate the non-interacting pair distribution.
        normalization_method (PairNormalizationMethod): Ensemble used for
            non-interacting normalization (default:
            ``PairNormalizationMethod.TIME_RANDOMIZATION``).
        walkable_area (WalkableArea, optional): Walkable area required when
            ``normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION``.
        seed (int, optional): Random seed for reproducible randomization.

    Returns:
        ``(v_bin_edges, g_v)`` when ``radius_bin_size`` is ``None``, otherwise
        ``(r_bin_edges, v_bin_edges, g_rv)``.
    """
    _check_trajectory_data(traj_data)
    if not {ID_COL, FRAME_COL, V_X_COL, V_Y_COL}.issubset(individual_speed.columns):
        raise SpeedError(
            "Required velocity columns ('v_x', 'v_y') are missing from 'individual_speed'. "
            "Compute individual speed with `compute_velocity=True`."
        )
    if radius_bin_size is not None and radius_bin_size <= 0:
        raise PedPyValueError(f"'radius_bin_size' must be > 0, got {radius_bin_size}.")

    merged_df = traj_data.data[[ID_COL, FRAME_COL, X_COL, Y_COL]].merge(
        individual_speed[[ID_COL, FRAME_COL, V_X_COL, V_Y_COL]],
        on=[ID_COL, FRAME_COL],
    )

    r_actual, v_actual = _calculate_pair_distances_and_approach_rates(merged_df)
    v_edges = _resolve_approach_rate_edges(v_actual, approach_rate_bin_size, approach_rate_bins)

    rng = np.random.default_rng(seed)
    random_df = _build_randomized_dataframe(
        data_df=merged_df,
        randomisation_stacking=randomisation_stacking,
        normalization_method=normalization_method,
        walkable_area=walkable_area,
        rng=rng,
    )
    r_ni, v_ni = _calculate_pair_distances_and_approach_rates(random_df)

    if radius_bin_size is None:
        hist_actual, _ = np.histogram(v_actual, bins=v_edges)
        hist_ni, _ = np.histogram(v_ni, bins=v_edges)
        prob_actual = hist_actual.astype(np.float64) / max(len(v_actual), 1)
        prob_ni = hist_ni.astype(np.float64) / max(len(v_ni), 1)
        return v_edges[1:], _safe_divide_distributions(prob_actual, prob_ni)

    max_dist = float(np.max(r_actual)) if len(r_actual) > 0 else radius_bin_size
    stop_edge = max_dist if max_dist > radius_bin_size else radius_bin_size + max_dist
    r_edges = np.arange(0.0, stop_edge, radius_bin_size, dtype=np.float64)

    hist2d_actual, _, _ = np.histogram2d(r_actual, v_actual, bins=[r_edges, v_edges])
    hist2d_ni, _, _ = np.histogram2d(r_ni, v_ni, bins=[r_edges, v_edges])
    prob2d_actual = hist2d_actual.astype(np.float64) / max(len(r_actual), 1)
    prob2d_ni = hist2d_ni.astype(np.float64) / max(len(r_ni), 1)
    return r_edges[1:], v_edges[1:], _safe_divide_distributions(prob2d_actual, prob2d_ni)
