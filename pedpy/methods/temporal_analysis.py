"""Module containing functions to compute temporal analysis methods.

For example: Short-Time Fourier Transform (STFT) and Welch's method for
spectral analysis.

Unlike the other calculators in :mod:`pedpy.methods`, the functions in this
module operate on a plain :class:`pandas.Series` rather than on
:class:`~pedpy.data.trajectory_data.TrajectoryData` directly. Frequency
analysis is commonly applied to derived per-pedestrian signals (e.g. a sway
feature extracted from the raw trajectory) that have no fixed representation
in :class:`~pedpy.data.trajectory_data.TrajectoryData`, so the functions stay
signal-agnostic and take whichever series the caller wants to analyze.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import stft, welch

from pedpy.column_identifier import FREQUENCY_COL, MAGNITUDE_COL, PHASE_COL, POWER_COL, TIME_COL
from pedpy.errors import InputError


def _validate_signal_series_input(
    *,
    signal_series: pd.Series,
    frame_rate: float,
    segments_length: Optional[int],
    overlap_length: Optional[int],
) -> None:
    if signal_series.empty:
        raise InputError("signal_series must not be empty.")

    if frame_rate <= 0:
        raise InputError(f"frame_rate must be positive, but is {frame_rate}.")

    if segments_length is not None:
        if segments_length <= 0:
            raise InputError(f"segments_length must be positive, but is {segments_length}.")

        if segments_length > len(signal_series):
            raise InputError(
                "segments_length must not be larger than the length of "
                f"signal_series ({len(signal_series)}), but is {segments_length}."
            )

        if overlap_length is not None and overlap_length >= segments_length:
            raise InputError(
                f"overlap_length ({overlap_length}) must be smaller than segments_length ({segments_length})."
            )


def compute_stft(
    *,
    signal_series: pd.Series,
    frame_rate: float,
    segments_length: Optional[int] = None,
    overlap_length: Optional[int] = None,
    zeros_padded: Optional[int] = None,
    window: str = "hann",
) -> pd.DataFrame:
    r"""Computes the Short-Time Fourier Transform (STFT) of a signal.

    This function calculates the time-frequency representation of a signal
    using the Short-Time Fourier Transform (STFT). The STFT provides
    information about the frequency content of the signal over time by
    computing the Fourier Transform within a sliding window.

    The output consists of the magnitude and phase of the STFT, allowing
    for both amplitude and phase analysis.

    .. math::
        STFT\{x[n]\}(m, k) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j 2 \pi k n / N}

    where :math:`x[n]` is the discrete-time signal, :math:`w[n]` is the
    window function, :math:`m` is the time index, :math:`k` is the
    frequency index, and :math:`N` is the number of FFT points.

    Args:
        signal_series (pd.Series): A pandas Series containing data values measured at a constant time interval.
        frame_rate (float): The frame rate of the signal data. The frame rate
            has to remain constant throughout the whole dataset.
        segments_length (int, optional): Length of each segment for the STFT window.
            Defaults to 5 times `frame_rate`.
        overlap_length (int, optional): Number of overlapping points between
            segments. Defaults to None (half of `segments_length` is used).
        zeros_padded (int, optional): Number of FFT points. Defaults to None
            (5 times `segments_length`).
        window (str, optional): The window function to apply before
            computing the STFT. Defaults to `'hann'`. Other options are
            `'hamming'`, `'bartlett'`, `'blackman'`, `'boxcar'`, `'triang'`, etc.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - :data:`~pedpy.column_identifier.FREQUENCY_COL`: The frequency bins of the STFT.
            - :data:`~pedpy.column_identifier.TIME_COL`: The time bins corresponding to the STFT computation.
            - :data:`~pedpy.column_identifier.MAGNITUDE_COL`: The absolute magnitude of the STFT at each
              time-frequency point.
            - :data:`~pedpy.column_identifier.PHASE_COL`: The phase of the STFT at each time-frequency point.
    """
    _validate_signal_series_input(
        signal_series=signal_series,
        frame_rate=frame_rate,
        segments_length=segments_length,
        overlap_length=overlap_length,
    )

    if segments_length is None:
        segments_length = int(frame_rate * 5)

    if overlap_length is None:
        overlap_length = segments_length // 2

    if zeros_padded is None:
        zeros_padded = 5 * segments_length

    f, t, zxx = stft(
        signal_series.values,
        fs=frame_rate,
        nperseg=segments_length,
        noverlap=overlap_length,
        nfft=zeros_padded,
        window=window,
    )

    return pd.DataFrame(
        {
            FREQUENCY_COL: np.repeat(f, len(t)),
            TIME_COL: np.tile(t, len(f)),
            MAGNITUDE_COL: np.abs(zxx).flatten(),
            PHASE_COL: np.angle(zxx).flatten(),
        }
    )


def compute_welch_spectral_distribution(
    *,
    signal_series: pd.Series,
    frame_rate: float,
    segments_length: Optional[int] = None,
    overlap_length: Optional[int] = None,
    zeros_padded: Optional[int] = None,
    window: str = "hann",
) -> pd.DataFrame:
    """Computes the power spectral density of a signal using Welch's method.

    This function estimates the power spectral density (PSD) of a signal by
    splitting it into overlapping segments, computing a modified periodogram
    for each segment, and averaging the periodograms. This is Welch's
    method, as implemented by :func:`scipy.signal.welch`.

    Args:
        signal_series (pd.Series): A pandas Series containing data values measured at a constant time interval.
        frame_rate (float): The frame rate of the signal data. The frame rate
            has to remain constant throughout the whole dataset.
        segments_length (int, optional): Length of each segment used to
            estimate the PSD. Defaults to one third of `signal_series` length.
        overlap_length (int, optional): Number of overlapping points between
            segments. Defaults to None (half of `segments_length` is used).
        zeros_padded (int, optional): Number of FFT points. Defaults to None
            (5 times `segments_length`).
        window (str, optional): The window function to apply before
            computing the PSD. Defaults to `'hann'`. Other options are
            `'hamming'`, `'bartlett'`, `'blackman'`, `'boxcar'`, `'triang'`, etc.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - :data:`~pedpy.column_identifier.FREQUENCY_COL`: The frequency bins of the spectral distribution.
            - :data:`~pedpy.column_identifier.POWER_COL`: The power spectral density at each frequency bin.
    """
    _validate_signal_series_input(
        signal_series=signal_series,
        frame_rate=frame_rate,
        segments_length=segments_length,
        overlap_length=overlap_length,
    )

    if segments_length is None:
        segments_length = len(signal_series) // 3

    if overlap_length is None:
        overlap_length = segments_length // 2

    if zeros_padded is None:
        zeros_padded = 5 * segments_length

    f, pxx = welch(
        signal_series.values,
        fs=frame_rate,
        nperseg=segments_length,
        noverlap=overlap_length,
        nfft=zeros_padded,
        window=window,
    )

    return pd.DataFrame(
        {
            FREQUENCY_COL: f,
            POWER_COL: pxx,
        }
    )
