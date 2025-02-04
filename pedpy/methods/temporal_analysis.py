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


def compute_spectral_distribution(
    *,
    traj_data: TrajectoryData,
    column_series: tuple,  # or array like object
    windowing_method: str = "square",
    lower_bound: int = 0,
    upper_bound: int = 100,
    zerro_padding: int = 0,
    temporal_averaging_window: float,
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
