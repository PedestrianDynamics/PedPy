"""Module containing functions to compute temporal analysis methods.

For example: Short-Time Fourier Transform (STFT) and Welch's method for spectral analysis.
"""

# import warnings
# from typing import Tuple

# import numpy as np
# import numpy.typing as npt
# import pandas as pd
# from scipy.spatial.distance import cdist

# from pedpy.column_identifier import FRAME_COL, ID_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData


import pandas as pd
import numpy as np
from scipy.signal import stft


def compute_STFT(
    signal_series: str,
    frame_rate: float,
    segments_length: int = None,
    overlap_length: int = None,
    zeros_padded: int = None,
    window: str = "hann",
) -> pd.DataFrame:
    """Computes the Short-Time Fourier Transform (STFT) of a signal.

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
        signal_series (pd.Series): A pandas Series containing data values mesured with a constant time interval.
        # frame_index_series (pd.Series): the pandas Series containing the time frame index associated
        # with each signal value.
        frame_rate (float): The frame rate of the signal data. The frame rate
            have to remain constant thought the whole dataset.
        segments_length (int, optional): Length of each segment for the STFT window.
            Defaults to one teenth of signal_series length.
        overlap_length (int, optional): Number of overlapping points between
            segments. Defaults to None (half of `segments_length` is used).
        zeros_padded (int, optional): Number of FFT points. Defaults to None
            (same as `segments_length`).
        window (str or tuple, optional): The window function to apply before
            computing the STFT. Defaults is `'hann'` (corresponds to a `'hann'`
            window). Other options are `'hamming'`, `'bartlett'`, `'blackman'`,
            `'boxcar'`, `'triang'`, etc.
    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - `"Frequency"`: The frequency bins of the STFT.
            - `"Time"`: The time bins corresponding to the STFT computation.
            - `"Magnitude"`: The absolute magnitude of the STFT at each
              time-frequency point.
            - `"Phase"`: The phase of the STFT at each time-frequency point.
    """

    # Set default segments_length if None
    if segments_length is None:
        segments_length = len(signal_series) // 10

    # Set default overlap_length if None
    if overlap_length is None:
        overlap_length = segments_length // 2

    # Extract signal data and compute STFT
    f, t, Zxx = stft(
        signal_series.values,
        fs=frame_rate,
        nperseg=segments_length,
        noverlap=overlap_length,
        nfft=zeros_padded,
        window=window,
    )

    # Convert STFT result into a Pandas DataFrame
    stft_df = pd.DataFrame(
        {
            "Frequency": np.repeat(
                f, len(t)
            ),  # Repeat each frequency value for each time step
            "Time": np.tile(t, len(f)),  # Tile time values across frequencies
            "Magnitude": np.abs(Zxx).flatten(),  # Flatten the STFT magnitude values
            "Phase": np.angle(Zxx).flatten(),  # Flatten the phase values
        }
    )

    return stft_df


def compute_welch_spectral_distribution(
    *,
    traj_data: TrajectoryData,
    column_series: tuple,  # or array like object
    windowing_method: str = "square",
    overlapping: float = 0.0,
    averaging: int = 1,
    zerro_padding: int = 0,
) -> pd.DataFrame:
    """Computes the spectral distribution of given temporal series.

    some decription

    .. math::
        some equation,

    Args:
        traj_data: TrajectoryData, an object containing the trajectories.
        column_series: tuple, an array-like object containing the temporal
            series for which the spectral distribution is to be computed using fft mothod from ...library...
        windowing_method: str, the method to be used for windowing the
            temporal series. Can be "square", "hanning", or "hamming". #and more
        lower_bound: int, the lower bound of the frequency range for the
            spectral distribution.
        upper_bound: int, the upper bound of the frequency range for the
            spectral distribution.
        zerro_padding: int, the number of zeros to be added to the end of the
            temporal series for zero-padding.
        temporal_averaging_window: float, the length of the window to be used
            for temporal averaging.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the frequency bins and the
            corresponding values of the spectral distribution.
    """


# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch


# Also interesting: STFFT to analysi temporal evolution of the spectum
# based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT
