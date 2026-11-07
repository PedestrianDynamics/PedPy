import numpy as np
import pandas as pd
import pytest

from pedpy.column_identifier import *
from pedpy.errors import InputError
from pedpy.methods.temporal_analysis import (
    compute_stft,
    compute_welch_spectral_distribution,
)

FRAME_RATE = 30.0
SIGNAL_FREQUENCY = 2.0
DURATION = 20.0


@pytest.fixture
def sine_signal() -> pd.Series:
    t = np.arange(0, DURATION, 1 / FRAME_RATE)
    return pd.Series(np.sin(2 * np.pi * SIGNAL_FREQUENCY * t))


def test_compute_stft_columns_and_shape(sine_signal):
    result = compute_stft(
        signal_series=sine_signal,
        frame_rate=FRAME_RATE,
        segments_length=60,
        overlap_length=30,
        zeros_padded=120,
    )

    assert list(result.columns) == [FREQUENCY_COL, TIME_COL, MAGNITUDE_COL, PHASE_COL]

    num_frequencies = result[FREQUENCY_COL].nunique()
    num_times = result[TIME_COL].nunique()
    assert num_frequencies == 120 // 2 + 1
    assert len(result) == num_frequencies * num_times


def test_compute_stft_recovers_known_frequency(sine_signal):
    result = compute_stft(signal_series=sine_signal, frame_rate=FRAME_RATE)

    mean_magnitude_per_frequency = result.groupby(FREQUENCY_COL)[MAGNITUDE_COL].mean()
    dominant_frequency = mean_magnitude_per_frequency.idxmax()

    assert dominant_frequency == pytest.approx(SIGNAL_FREQUENCY, abs=0.5)


def test_compute_stft_default_segments_length(sine_signal):
    default = compute_stft(signal_series=sine_signal, frame_rate=FRAME_RATE)
    explicit = compute_stft(
        signal_series=sine_signal,
        frame_rate=FRAME_RATE,
        segments_length=int(FRAME_RATE * 5),
    )

    pd.testing.assert_frame_equal(default, explicit)


def test_compute_welch_columns_and_shape(sine_signal):
    result = compute_welch_spectral_distribution(
        signal_series=sine_signal,
        frame_rate=FRAME_RATE,
        segments_length=60,
        overlap_length=30,
        zeros_padded=120,
    )

    assert list(result.columns) == [FREQUENCY_COL, POWER_COL]
    assert len(result) == 120 // 2 + 1


def test_compute_welch_recovers_known_frequency(sine_signal):
    result = compute_welch_spectral_distribution(signal_series=sine_signal, frame_rate=FRAME_RATE)

    dominant_frequency = result.loc[result[POWER_COL].idxmax(), FREQUENCY_COL]

    assert dominant_frequency == pytest.approx(SIGNAL_FREQUENCY, abs=0.5)


def test_compute_welch_default_segments_length(sine_signal):
    default = compute_welch_spectral_distribution(signal_series=sine_signal, frame_rate=FRAME_RATE)
    explicit = compute_welch_spectral_distribution(
        signal_series=sine_signal,
        frame_rate=FRAME_RATE,
        segments_length=len(sine_signal) // 3,
    )

    pd.testing.assert_frame_equal(default, explicit)


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
def test_empty_signal_series_raises(compute_fn):
    with pytest.raises(InputError, match="signal_series must not be empty"):
        compute_fn(signal_series=pd.Series(dtype=float), frame_rate=FRAME_RATE)


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
@pytest.mark.parametrize("frame_rate", [0, -1.0])
def test_non_positive_frame_rate_raises(compute_fn, sine_signal, frame_rate):
    with pytest.raises(InputError, match="frame_rate must be positive"):
        compute_fn(signal_series=sine_signal, frame_rate=frame_rate)


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
def test_non_positive_segments_length_raises(compute_fn, sine_signal):
    with pytest.raises(InputError, match="segments_length must be positive"):
        compute_fn(signal_series=sine_signal, frame_rate=FRAME_RATE, segments_length=0)


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
def test_segments_length_larger_than_signal_raises(compute_fn, sine_signal):
    with pytest.raises(InputError, match="segments_length must not be larger"):
        compute_fn(
            signal_series=sine_signal,
            frame_rate=FRAME_RATE,
            segments_length=len(sine_signal) + 1,
        )


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
def test_overlap_length_not_smaller_than_segments_length_raises(compute_fn, sine_signal):
    with pytest.raises(InputError, match="must be smaller than"):
        compute_fn(
            signal_series=sine_signal,
            frame_rate=FRAME_RATE,
            segments_length=60,
            overlap_length=60,
        )


@pytest.mark.parametrize("compute_fn", [compute_stft, compute_welch_spectral_distribution])
def test_positional_arguments_are_rejected(compute_fn, sine_signal):
    with pytest.raises(TypeError):
        compute_fn(sine_signal, FRAME_RATE)
