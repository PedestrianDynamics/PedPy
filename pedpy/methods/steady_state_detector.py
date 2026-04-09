"""Steady state detection using the modified CUSUM algorithm.

Implements the method from:
    Liao et al., "Measuring the steady state of pedestrian flow in
    bottleneck experiments", Physica A 461 (2016) 248-261.

The algorithm detects intervals in density/speed time series where
the process is stationary ("steady state"), using a modified
Cumulative Sum Control Chart (CUSUM) with a detection threshold
calibrated via an autoregressive model.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import norm, pearsonr


@dataclass(frozen=True)
class SteadyStateResult:
    """Result of steady state detection on a single time series.

    Attributes:
        frame_start: start frames of detected steady intervals
        frame_end: end frames of detected steady intervals
        cusum: the CUSUM statistics array (one value per observation)
        theta: the calibrated detection threshold
        mean: mean of the reference process
        std: standard deviation of the reference process
        acf: lag-1 autocorrelation of the reference process
    """

    frame_start: np.ndarray
    frame_end: np.ndarray
    cusum: np.ndarray
    theta: float
    mean: float
    std: float
    acf: float


def detect_steady_state(
    *,
    frames: np.ndarray,
    values: np.ndarray,
    ref_start: int,
    ref_end: int,
    alpha: float = 0.99,
    gamma: float = 0.99,
    s_max: int = 100,
    grid_size: int = 500,
    grid_half_width: float = 3.2,
) -> SteadyStateResult:
    """Detect steady state intervals in a time series.

    Uses the modified CUSUM algorithm from Liao et al. (2016) to find
    intervals where the time series is stationary relative to a
    user-selected reference process.

    Args:
        frames: array of frame numbers (monotonically increasing)
        values: array of observed values (density or speed), same length
            as frames
        ref_start: start frame of the reference process (inclusive)
        ref_end: end frame of the reference process (inclusive)
        alpha: critical probability for the step function F
            (default 0.99)
        gamma: confidence level for the detection threshold theta
            (default 0.99)
        s_max: upper boundary of the CUSUM statistics (default 100)
        grid_size: number of grid points K for the numerical theta
            computation (default 500)
        grid_half_width: half-width of the discretisation grid
            (default 3.2)

    Returns:
        SteadyStateResult with detected steady intervals and diagnostics

    Raises:
        ValueError: if reference range is invalid or data is too short.
    """
    frames = np.asarray(frames, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(frames) != len(values):
        raise ValueError("frames and values must have the same length.")

    if ref_start > ref_end:
        raise ValueError(f"ref_start ({ref_start}) must be <= ref_end ({ref_end}).")

    # --- reference statistics ---
    ref_mask = (frames >= ref_start) & (frames <= ref_end)
    ref_series = values[ref_mask]
    if len(ref_series) < 3:
        raise ValueError("Reference range must contain at least 3 observations.")

    ref_mean = float(np.mean(ref_series))
    ref_std = float(np.std(ref_series))
    ref_acf = float(pearsonr(ref_series[:-1], ref_series[1:])[0])

    if ref_std == 0:
        raise ValueError("Reference series has zero standard deviation.")

    # --- calibrate detection threshold theta ---
    q = norm.ppf(alpha)
    theta = _compute_theta(
        acf=ref_acf,
        q=q,
        gamma=gamma,
        s_max=s_max,
        grid_size=grid_size,
        grid_half_width=grid_half_width,
    )

    # --- compute CUSUM statistics ---
    cusum = _compute_cusum(values, ref_mean, ref_std, q, s_max)

    # --- extract steady intervals ---
    starts, ends = _find_steady_intervals(
        frames=frames,
        cusum=cusum,
        theta=theta,
        s_max=s_max,
    )

    return SteadyStateResult(
        frame_start=starts,
        frame_end=ends,
        cusum=cusum,
        theta=theta,
        mean=ref_mean,
        std=ref_std,
        acf=ref_acf,
    )


def combine_steady_states(
    results: List[SteadyStateResult],
) -> List[Tuple[float, float]]:
    """Find the overlapping steady intervals across multiple series.

    For pedestrian flow analysis, the final steady state is the
    intersection of the steady states detected independently for
    density and speed.

    Args:
        results: list of SteadyStateResult objects (e.g. one for
            density, one for speed)

    Returns:
        list of (start_frame, end_frame) tuples for the combined
        steady intervals
    """
    if not results:
        return []

    # collect all intervals from all results
    all_intervals = []
    for r in results:
        for s, e in zip(r.frame_start, r.frame_end, strict=True):
            all_intervals.append((s, e))

    if not all_intervals:
        return []

    if len(results) == 1:
        return all_intervals

    # find pairwise overlaps across all results
    # start with intervals from the first result, intersect with each
    # subsequent result
    current = list(zip(results[0].frame_start, results[0].frame_end, strict=True))
    for r in results[1:]:
        next_intervals = list(zip(r.frame_start, r.frame_end, strict=True))
        current = _intersect_interval_lists(current, next_intervals)

    return current


# --------------- internal helpers ---------------


def _compute_cusum(
    values: np.ndarray,
    ref_mean: float,
    ref_std: float,
    q: float,
    s_max: int,
) -> np.ndarray:
    """Compute the modified CUSUM statistics (Eq. 7 in Liao 2016).

    s_i = min(max(0, s_{i-1} + F(x_tilde_i)), s_max)
    with s_0 = s_max and F(x) = 1 if |x| > q else -1.
    """
    x_tilde = (values - ref_mean) / ref_std
    f_vals = np.where(np.abs(x_tilde) > q, 1.0, -1.0)
    s = np.empty(len(values), dtype=float)
    s[0] = s_max
    for i in range(1, len(s)):
        s[i] = min(max(0.0, s[i - 1] + f_vals[i]), s_max)
    return s


def _find_steady_intervals(
    *,
    frames: np.ndarray,
    cusum: np.ndarray,
    theta: float,
    s_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract steady state frame ranges from CUSUM statistics.

    A point is in steady state when cusum < theta.  The raw intervals
    are then adjusted for the reaction times:
        t_reaching = (s_max - theta) / fps  (entering steady)
        t_leaving  = theta / fps             (leaving steady)
    Since we work in frame units (fps=1 frame), the corrections are
    applied directly in frame counts.
    """
    steady_mask = cusum < theta
    if not np.any(steady_mask):
        return np.array([]), np.array([])

    steady_frames = frames[steady_mask]

    # split into contiguous runs
    starts = []
    ends = []
    run_start = steady_frames[0]
    for i in range(1, len(steady_frames)):
        # detect gap (non-consecutive frames)
        if steady_frames[i] - steady_frames[i - 1] != 1:
            run_end = steady_frames[i - 1]
            starts.append(run_start)
            ends.append(run_end)
            run_start = steady_frames[i]
    # last run
    starts.append(run_start)
    ends.append(steady_frames[-1])

    # apply reaction time corrections (Eq. 8 in Liao 2016)
    # t_reaching = (s_max - theta): entering steady state
    # t_leaving = theta: leaving steady state
    reaching = s_max - theta
    leaving = theta

    adjusted_starts = []
    adjusted_ends = []
    for s, e in zip(starts, ends, strict=True):
        adj_s = s - reaching
        adj_e = e - leaving
        if adj_s < adj_e:
            adjusted_starts.append(adj_s)
            adjusted_ends.append(adj_e)

    return np.array(adjusted_starts), np.array(adjusted_ends)


def _compute_theta(
    *,
    acf: float,
    q: float,
    gamma: float,
    s_max: int,
    grid_size: int,
    grid_half_width: float,
) -> float:
    """Calibrate the detection threshold theta via the autoregressive model.

    Uses the block-tridiagonal Thomas Algorithm to solve for the
    stationary distribution of the Markov chain (y_i, s_i) as
    described in Appendix A of Liao et al. (2016).

    The autoregressive process is:
        y_i = c * y_{i-1} + sqrt(1 - c^2) * epsilon_i

    theta is the upper gamma-percentile of the marginal distribution
    of s from the stationary distribution.
    """
    c = acf
    n_blocks = s_max  # number of s-blocks (0..s_max)

    d = 2 * grid_half_width / grid_size
    xi = np.arange(-grid_half_width, grid_half_width + 0.0001, d)
    kp1 = grid_size + 1  # grid points per block

    # find grid indices ia, ib corresponding to -q and +q
    ia, ib, dnorm = _init_grid_params(xi, q)

    # build block matrices b1 (outside [-q,q]) and b2 (inside [-q,q])
    b1, b2 = _build_block_matrices(xi, ia, ib, kp1, c, d)
    eye = np.eye(kp1)

    # solve block-tridiagonal system via Thomas algorithm
    td = _solve_thomas(b1, b2, eye, kp1, n_blocks, dnorm)

    # assemble full solution vector and normalise
    tms = np.vstack(td)
    tms = tms / (d * np.sum(tms))

    # marginal distribution of s: integrate over x for each s
    tm = np.zeros(n_blocks + 1)
    for j in range(n_blocks + 1):
        begin = j * kp1
        end = begin + kp1
        tm[j] = d * np.sum(tms[begin:end])

    # find theta as upper gamma-percentile
    cumulative = tm[0]
    theta = 1
    while theta + 1 < len(tm) and cumulative + tm[theta] < gamma:
        cumulative += tm[theta]
        theta += 1

    return float(theta)


def _init_grid_params(xi: np.ndarray, q: float) -> Tuple[int, int, float]:
    """Find grid indices for -q and +q, and compute normalisation constant."""
    ia = ib = 0
    dnorm = 0.0
    for i in range(len(xi)):
        if ia == 0 and i + 1 < len(xi) and xi[i + 1] > -q:
            ia = i
        if ib == 0 and xi[i] > q:
            ib = i
        dnorm += (1.0 / math.sqrt(2 * math.pi)) * math.exp(-xi[i] * xi[i] / 2.0)
    return ia, ib, dnorm


def _build_block_matrices(
    xi: np.ndarray,
    ia: int,
    ib: int,
    kp1: int,
    c: float,
    d: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build block transition matrices b1 and b2."""

    def func_b(i: int, k: int) -> float:
        if abs(c) >= 1.0:
            return d / math.sqrt(2 * math.pi) * math.exp(-((xi[i] - xi[k]) ** 2) / 2.0)
        return (
            d * math.exp(-((xi[i] - xi[k] * c) ** 2) / (2.0 * (1.0 - c * c))) / math.sqrt(2 * math.pi * (1.0 - c * c))
        )

    b1 = np.zeros((kp1, kp1))
    b2 = np.zeros((kp1, kp1))
    for i in range(kp1):
        for j in range(kp1):
            if i < ia or i >= ib:
                b1[i, j] = func_b(i, j)
            if ia <= i < ib:
                b2[i, j] = func_b(i, j)
    return b1, b2


def _solve_thomas(
    b1: np.ndarray,
    b2: np.ndarray,
    eye: np.ndarray,
    kp1: int,
    n_blocks: int,
    dnorm: float,
) -> List[np.ndarray]:
    """Solve the block-tridiagonal system using the Thomas algorithm."""
    td: List[np.ndarray] = [np.zeros((kp1, 1)) for _ in range(n_blocks + 1)]
    td[n_blocks][-1, 0] = dnorm

    tc_arr: List[np.ndarray] = [np.zeros((kp1, kp1)) for _ in range(n_blocks + 1)]

    # block 0: Tb = b2 - I, Tc = b2
    tb0 = b2 - eye
    tc_arr[0] = np.linalg.solve(tb0, b2.copy())
    td[0] = np.linalg.solve(tb0, td[0])

    # blocks 1..n_blocks
    for i in range(1, n_blocks + 1):
        if i < n_blocks:
            tb_i = -eye
            tc_i = b2.copy()
        else:
            tb_i = b1 - eye
            tc_i = np.zeros((kp1, kp1))

        aa = tb_i - b1 @ tc_arr[i - 1]
        if i < n_blocks:
            tc_arr[i] = np.linalg.solve(aa, tc_i)
        td[i] = np.linalg.solve(aa, td[i] - b1 @ td[i - 1])

    # backward sweep
    for i in range(n_blocks - 1, -1, -1):
        td[i] = td[i] - tc_arr[i] @ td[i + 1]

    return td


def _intersect_interval_lists(
    a: List[Tuple[float, float]],
    b: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Compute pairwise intersections of two lists of intervals."""
    result = []
    for a_s, a_e in a:
        for b_s, b_e in b:
            start = max(a_s, b_s)
            end = min(a_e, b_e)
            if start < end:
                result.append((start, end))
    return result
