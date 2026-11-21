"""Functions for smoothing positional jitter in pedestrian trajectories."""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import PedPyTypeError, PedPyValueError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

_EVEN_MODULO: Final = 2
_MIN_SAVGOL_ADAPTIVE_WINDOW: Final = 3


class SmoothingBorderMethod(Enum):
    """Strategy for handling incomplete windows near trajectory boundaries."""

    EXCLUDE = auto()
    """Drop boundary frames where a full centered window is unavailable.

    Each smoothed pedestrian trajectory becomes shorter by ``window_size - 1``
    frames, ensuring every retained sample uses an exact full-width window.
    """

    ADAPTIVE = auto()
    """Shrink the symmetric window width toward trajectory start and end.

    Preserves all frames without extrapolating synthetic values outside the
    recorded trajectory: frame ``k`` uses half-width
    ``min(k, n_frames - 1 - k, window_size // 2)``.
    """

    NEAREST = auto()
    """Pad boundaries by repeating the first and last coordinate values.

    Corresponds to ``mode="nearest"`` in :func:`scipy.ndimage.uniform_filter1d`
    and :func:`scipy.signal.savgol_filter`.
    """

    INTERP = auto()
    """Fit polynomials up to boundaries without padding (Savitzky-Golay) or reflect (moving average).

    Corresponds to ``mode="interp"`` in :func:`scipy.signal.savgol_filter` and
    ``mode="reflect"`` in :func:`scipy.ndimage.uniform_filter1d`.
    """


def _validate_common_inputs(
    traj_data: TrajectoryData,
    window_size: int,
    border_method: SmoothingBorderMethod,
    ped_ids: Sequence[int] | None,
) -> set[int] | None:
    if not isinstance(traj_data, TrajectoryData):
        raise PedPyTypeError(f"traj_data must be an instance of TrajectoryData, got {type(traj_data).__name__}.")
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise PedPyTypeError(f"window_size must be an integer, got {type(window_size).__name__}.")
    if window_size <= 0:
        raise PedPyValueError(f"window_size must be a positive integer (> 0), got {window_size}.")
    if not isinstance(border_method, SmoothingBorderMethod):
        raise PedPyTypeError(
            f"border_method must be an instance of SmoothingBorderMethod, got {type(border_method).__name__}."
        )
    if ped_ids is None:
        return None
    if isinstance(ped_ids, (str, bytes)):
        raise PedPyTypeError("ped_ids must be a sequence of integer IDs, not str.")
    target_ids: set[int] = set()
    for pid in ped_ids:
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise PedPyTypeError(f"Each entry in ped_ids must be an int, got {type(pid).__name__}.")
        target_ids.add(pid)
    return target_ids


def _moving_average_1d(
    values: npt.NDArray[np.float64],
    window_size: int,
    border_method: SmoothingBorderMethod,
) -> tuple[npt.NDArray[np.float64], slice]:
    n_samples = len(values)
    if window_size == 1:
        return values.copy(), slice(0, n_samples)

    origin = -1 if window_size % _EVEN_MODULO == 0 else 0
    if border_method == SmoothingBorderMethod.EXCLUDE:
        filtered = np.asarray(
            uniform_filter1d(values, size=window_size, mode="nearest", origin=origin),
            dtype=np.float64,
        )
        left_trim = (window_size - 1) // 2
        right_trim = window_size // 2
        stop = n_samples - right_trim
        return filtered[left_trim:stop], slice(left_trim, stop)

    if border_method == SmoothingBorderMethod.ADAPTIVE:
        half_max = window_size // 2
        cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        idx = np.arange(n_samples)
        half = np.minimum(np.minimum(idx, n_samples - 1 - idx), half_max)
        starts = idx - half
        ends = idx + half + 1
        adaptive = (cumsum[ends] - cumsum[starts]) / (ends - starts)
        return adaptive, slice(0, n_samples)

    scipy_mode = "nearest" if border_method == SmoothingBorderMethod.NEAREST else "reflect"
    filtered = np.asarray(
        uniform_filter1d(values, size=window_size, mode=scipy_mode, origin=origin),
        dtype=np.float64,
    )
    return filtered, slice(0, n_samples)


def _savgol_1d(
    values: npt.NDArray[np.float64],
    window_size: int,
    polyorder: int,
    border_method: SmoothingBorderMethod,
) -> tuple[npt.NDArray[np.float64], slice]:
    n_samples = len(values)
    if window_size == 1:
        return values.copy(), slice(0, n_samples)

    if border_method == SmoothingBorderMethod.EXCLUDE:
        filtered = np.asarray(
            savgol_filter(values, window_length=window_size, polyorder=polyorder, mode="interp"),
            dtype=np.float64,
        )
        half = window_size // 2
        stop = n_samples - half
        return filtered[half:stop], slice(half, stop)

    if border_method == SmoothingBorderMethod.ADAPTIVE:
        interior = np.asarray(
            savgol_filter(values, window_length=window_size, polyorder=polyorder, mode="interp"),
            dtype=np.float64,
        )
        half_max = window_size // 2
        out = interior.copy()
        out[0] = values[0]
        out[-1] = values[-1]
        for k in range(1, half_max):
            win_k = 2 * k + 1
            if win_k >= _MIN_SAVGOL_ADAPTIVE_WINDOW:
                ord_k = min(polyorder, win_k - 1)
                left_seg = savgol_filter(values[:win_k], window_length=win_k, polyorder=ord_k, mode="interp")
                right_seg = savgol_filter(
                    values[n_samples - win_k :],
                    window_length=win_k,
                    polyorder=ord_k,
                    mode="interp",
                )
                out[k] = float(left_seg[k])
                out[n_samples - 1 - k] = float(right_seg[k])
        return out, slice(0, n_samples)

    scipy_mode = "nearest" if border_method == SmoothingBorderMethod.NEAREST else "interp"
    filtered = np.asarray(
        savgol_filter(values, window_length=window_size, polyorder=polyorder, mode=scipy_mode),
        dtype=np.float64,
    )
    return filtered, slice(0, n_samples)


def moving_average_filter(
    *,
    traj_data: TrajectoryData,
    window_size: int,
    ped_ids: Sequence[int] | None = None,
    border_method: SmoothingBorderMethod = SmoothingBorderMethod.ADAPTIVE,
) -> TrajectoryData:
    """Smooth positional jitter in trajectories using a moving-average filter.

    Applies a 1-D uniform moving average independently to the ``x`` and ``y``
    coordinates of each selected pedestrian (ordered by ``frame``), preserving
    pedestrian IDs, frame numbers, non-coordinate columns, and frame rate.

    Choosing ``window_size``:
        Small odd windows (e.g., ``3`` to ``7`` frames at 10-25 Hz, roughly
        0.15-0.35 s) attenuate high-frequency camera/tracking noise without
        excessively rounding sharp turns or decelerations. Larger windows give
        stronger noise suppression at the cost of corner cutting.

    Border behavior (``border_method``):
        * :attr:`SmoothingBorderMethod.ADAPTIVE` (default): Shrinks the
          symmetric window near start/end frames so trajectory length is
          preserved without extrapolating outside observed frames.
        * :attr:`SmoothingBorderMethod.EXCLUDE`: Drops the ``window_size - 1``
          boundary frames where a full window is unavailable.
        * :attr:`SmoothingBorderMethod.NEAREST`: Replicates endpoint positions
          via ``scipy.ndimage.uniform_filter1d(..., mode="nearest")``.
        * :attr:`SmoothingBorderMethod.INTERP`: Reflects boundary samples via
          ``scipy.ndimage.uniform_filter1d(..., mode="reflect")``.

    Preconditions:
        Frames for each pedestrian should be sampled at uniform ``frame_rate``
        intervals. If a pedestrian's trajectory has fewer frames than
        ``window_size``, a :class:`UserWarning` is emitted and that
        pedestrian's trajectory is kept unchanged.

    Args:
        traj_data: Input :class:`~pedpy.data.trajectory_data.TrajectoryData`.
        window_size: Positive filter window length in frames (``>= 1``).
        ped_ids: Optional subset of pedestrian IDs to smooth. Other pedestrians
            in ``traj_data`` are retained unmodified.
        border_method: Boundary handling mode (default:
            :attr:`SmoothingBorderMethod.ADAPTIVE`).

    Returns:
        Smoothed :class:`~pedpy.data.trajectory_data.TrajectoryData`.
    """
    target_ids = _validate_common_inputs(traj_data, window_size, border_method, ped_ids)
    if traj_data.data.empty:
        return TrajectoryData(data=traj_data.data.copy(), frame_rate=traj_data.frame_rate)

    chunks: list[pd.DataFrame] = []
    for pid, group in traj_data.data.groupby(ID_COL, sort=False):
        ordered = group.sort_values(FRAME_COL).copy()
        if target_ids is not None and int(pid) not in target_ids:
            chunks.append(ordered)
            continue
        if len(ordered) < window_size:
            warnings.warn(
                f"Pedestrian {pid} has {len(ordered)} frames, which is shorter than "
                f"window_size={window_size}; keeping original trajectory.",
                category=UserWarning,
                stacklevel=2,
            )
            chunks.append(ordered)
            continue

        x_vals = ordered[X_COL].to_numpy(dtype=np.float64)
        y_vals = ordered[Y_COL].to_numpy(dtype=np.float64)
        x_smooth, keep_slice = _moving_average_1d(x_vals, window_size, border_method)
        y_smooth, _ = _moving_average_1d(y_vals, window_size, border_method)

        trimmed = ordered.iloc[keep_slice].copy()
        trimmed[X_COL] = x_smooth
        trimmed[Y_COL] = y_smooth
        chunks.append(trimmed)

    combined = pd.concat(chunks, ignore_index=True)
    return TrajectoryData(data=combined, frame_rate=traj_data.frame_rate)


def savitzky_golay_filter(
    *,
    traj_data: TrajectoryData,
    window_size: int,
    polyorder: int,
    ped_ids: Sequence[int] | None = None,
    border_method: SmoothingBorderMethod = SmoothingBorderMethod.INTERP,
) -> TrajectoryData:
    """Smooth positional jitter in trajectories using a Savitzky-Golay filter.

    Fits local polynomials of degree ``polyorder`` over odd-length windows of
    ``window_size`` frames via :func:`scipy.signal.savgol_filter` independently
    on ``x`` and ``y`` for each pedestrian (ordered by ``frame``). Compared to
    a moving average, Savitzky-Golay filtering preserves path curvature,
    turning radii, and velocity/acceleration peaks while suppressing tracking
    jitter.

    Choosing ``window_size`` and ``polyorder``:
        * ``window_size`` must be an odd positive integer (typically ``5`` to
          ``15`` frames depending on frame rate and noise amplitude).
        * ``polyorder`` must satisfy ``0 <= polyorder < window_size``. Quadratic
          or cubic orders (``2`` or ``3``) preserve constant-acceleration bends
          well; higher orders preserve sharper maneuvers but reduce smoothing.

    Border behavior (``border_method``):
        * :attr:`SmoothingBorderMethod.INTERP` (default): Evaluates polynomial
          fits fitted to the first/last ``window_size`` samples at boundary
          frames (`scipy` ``mode="interp"``), preserving full length.
        * :attr:`SmoothingBorderMethod.ADAPTIVE`: Shrinks the odd window width
          and polynomial order toward boundaries so no boundary polynomial
          extrapolation is required while keeping all frames.
        * :attr:`SmoothingBorderMethod.NEAREST`: Replicates endpoint positions
          (`scipy` ``mode="nearest"``).
        * :attr:`SmoothingBorderMethod.EXCLUDE`: Drops the ``window_size - 1``
          boundary frames where a centered full window cannot be formed.

    Preconditions:
        Frames for each pedestrian should be uniformly spaced. If a
        pedestrian's trajectory contains fewer frames than ``window_size``, a
        :class:`UserWarning` is emitted and that pedestrian's trajectory is
        returned unchanged.

    Args:
        traj_data: Input :class:`~pedpy.data.trajectory_data.TrajectoryData`.
        window_size: Odd positive window length in frames.
        polyorder: Non-negative polynomial order (``< window_size``).
        ped_ids: Optional subset of pedestrian IDs to smooth.
        border_method: Boundary handling mode (default:
            :attr:`SmoothingBorderMethod.INTERP`).

    Returns:
        Smoothed :class:`~pedpy.data.trajectory_data.TrajectoryData`.
    """
    target_ids = _validate_common_inputs(traj_data, window_size, border_method, ped_ids)
    if window_size % _EVEN_MODULO == 0:
        raise PedPyValueError(f"window_size for savitzky_golay_filter must be odd, got {window_size}.")
    if not isinstance(polyorder, int) or isinstance(polyorder, bool):
        raise PedPyTypeError(f"polyorder must be an integer, got {type(polyorder).__name__}.")
    if polyorder < 0 or polyorder >= window_size:
        raise PedPyValueError(f"polyorder must satisfy 0 <= polyorder < window_size ({window_size}), got {polyorder}.")

    if traj_data.data.empty:
        return TrajectoryData(data=traj_data.data.copy(), frame_rate=traj_data.frame_rate)

    chunks: list[pd.DataFrame] = []
    for pid, group in traj_data.data.groupby(ID_COL, sort=False):
        ordered = group.sort_values(FRAME_COL).copy()
        if target_ids is not None and int(pid) not in target_ids:
            chunks.append(ordered)
            continue
        if len(ordered) < window_size:
            warnings.warn(
                f"Pedestrian {pid} has {len(ordered)} frames, which is shorter than "
                f"window_size={window_size}; keeping original trajectory.",
                category=UserWarning,
                stacklevel=2,
            )
            chunks.append(ordered)
            continue

        x_vals = ordered[X_COL].to_numpy(dtype=np.float64)
        y_vals = ordered[Y_COL].to_numpy(dtype=np.float64)
        x_smooth, keep_slice = _savgol_1d(x_vals, window_size, polyorder, border_method)
        y_smooth, _ = _savgol_1d(y_vals, window_size, polyorder, border_method)

        trimmed = ordered.iloc[keep_slice].copy()
        trimmed[X_COL] = x_smooth
        trimmed[Y_COL] = y_smooth
        chunks.append(trimmed)

    combined = pd.concat(chunks, ignore_index=True)
    return TrajectoryData(data=combined, frame_rate=traj_data.frame_rate)
