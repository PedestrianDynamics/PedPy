"""Unit tests for moving-average and Savitzky-Golay trajectory smoothing filters."""

import numpy as np
import pandas as pd
import pytest

from pedpy import (
    FRAME_COL,
    ID_COL,
    X_COL,
    Y_COL,
    PedPyTypeError,
    PedPyValueError,
    SmoothingBorderMethod,
    TrajectoryData,
    moving_average_filter,
    savitzky_golay_filter,
)


def _noisy_two_ped_traj() -> TrajectoryData:
    frames = list(range(11))
    # Ped 1: quadratic curve + alternating jitter
    jitter = np.array([(-1) ** k * 0.1 for k in frames])
    x1 = np.array(frames, dtype=float) + jitter
    y1 = 0.05 * np.array(frames, dtype=float) ** 2 - jitter
    # Ped 2: far away constant position (must never bleed into Ped 1)
    x2 = np.full(len(frames), 100.0)
    y2 = np.full(len(frames), -50.0)

    df = pd.DataFrame(
        {
            ID_COL: [1] * len(frames) + [2] * len(frames),
            FRAME_COL: frames + frames,
            X_COL: np.concatenate([x1, x2]),
            Y_COL: np.concatenate([y1, y2]),
            "marker_id": [101] * len(frames) + [202] * len(frames),
        }
    )
    return TrajectoryData(data=df, frame_rate=25.0)


def test_moving_average_per_pedestrian_isolation_and_metadata():
    traj = _noisy_two_ped_traj()
    smoothed = moving_average_filter(
        traj_data=traj,
        window_size=3,
        border_method=SmoothingBorderMethod.ADAPTIVE,
    )
    assert smoothed.frame_rate == traj.frame_rate
    assert "marker_id" in smoothed.data.columns
    assert len(smoothed.data) == len(traj.data)

    ped2 = smoothed.data[smoothed.data[ID_COL] == 2]
    np.testing.assert_allclose(ped2[X_COL], 100.0)
    np.testing.assert_allclose(ped2[Y_COL], -50.0)

    # Interior jitter on ped 1 is strongly attenuated
    ped1_orig = traj.data[traj.data[ID_COL] == 1]
    ped1_smooth = smoothed.data[smoothed.data[ID_COL] == 1]
    orig_err = np.abs(ped1_orig[X_COL].to_numpy()[2:-2] - np.arange(2, 9))
    smooth_err = np.abs(ped1_smooth[X_COL].to_numpy()[2:-2] - np.arange(2, 9))
    assert smooth_err.mean() < orig_err.mean()


def test_savitzky_golay_preserves_exact_quadratic_and_ped_ids_subset():
    frames = np.arange(9)
    x_quad = 2.0 + 0.5 * frames + 0.1 * frames**2
    y_quad = -1.0 + 0.3 * frames - 0.05 * frames**2
    df = pd.DataFrame(
        {
            ID_COL: [1] * 9 + [2] * 9,
            FRAME_COL: list(frames) + list(frames),
            X_COL: np.concatenate([x_quad, x_quad + 5.0]),
            Y_COL: np.concatenate([y_quad, y_quad + 5.0]),
        }
    )
    # Add jitter only to ped 2 to verify ped_ids=[1] leaves ped 2 untouched
    df.loc[df[ID_COL] == 2, X_COL] += 0.25
    traj = TrajectoryData(data=df, frame_rate=10.0)

    res = savitzky_golay_filter(
        traj_data=traj,
        window_size=5,
        polyorder=2,
        ped_ids=[1],
        border_method=SmoothingBorderMethod.INTERP,
    )
    p1 = res.data[res.data[ID_COL] == 1]
    p2 = res.data[res.data[ID_COL] == 2]
    np.testing.assert_allclose(p1[X_COL], x_quad, atol=1e-12)
    np.testing.assert_allclose(p1[Y_COL], y_quad, atol=1e-12)
    np.testing.assert_allclose(p2[X_COL], traj.data[traj.data[ID_COL] == 2][X_COL])


@pytest.mark.parametrize(
    "border_method, expected_len",
    [
        (SmoothingBorderMethod.EXCLUDE, 7),
        (SmoothingBorderMethod.ADAPTIVE, 11),
        (SmoothingBorderMethod.NEAREST, 11),
        (SmoothingBorderMethod.INTERP, 11),
    ],
)
def test_border_methods_for_both_filters(border_method, expected_len):
    traj = _noisy_two_ped_traj()
    ma = moving_average_filter(traj_data=traj, window_size=5, border_method=border_method)
    sg = savitzky_golay_filter(
        traj_data=traj,
        window_size=5,
        polyorder=2,
        border_method=border_method,
    )
    assert len(ma.data[ma.data[ID_COL] == 1]) == expected_len
    assert len(sg.data[sg.data[ID_COL] == 1]) == expected_len


def test_short_trajectory_warns_and_keeps_original():
    df = pd.DataFrame(
        {
            ID_COL: [1, 1, 1],
            FRAME_COL: [0, 1, 2],
            X_COL: [0.0, 1.0, 2.0],
            Y_COL: [0.5, 0.5, 0.5],
        }
    )
    traj = TrajectoryData(data=df, frame_rate=10.0)

    with pytest.warns(UserWarning, match="shorter than window_size"):
        ma = moving_average_filter(traj_data=traj, window_size=5)
    np.testing.assert_allclose(ma.data[X_COL], traj.data[X_COL])

    with pytest.warns(UserWarning, match="shorter than window_size"):
        sg = savitzky_golay_filter(traj_data=traj, window_size=5, polyorder=2)
    np.testing.assert_allclose(sg.data[X_COL], traj.data[X_COL])


def test_invalid_parameters_raise_pedpy_exceptions():
    traj = _noisy_two_ped_traj()

    with pytest.raises(PedPyValueError, match="positive"):
        moving_average_filter(traj_data=traj, window_size=0)
    with pytest.raises(PedPyValueError, match="positive"):
        moving_average_filter(traj_data=traj, window_size=-3)
    with pytest.raises(PedPyTypeError):
        moving_average_filter(traj_data=traj, window_size=True)  # type: ignore[arg-type]

    with pytest.raises(PedPyValueError, match="odd"):
        savitzky_golay_filter(traj_data=traj, window_size=4, polyorder=2)
    with pytest.raises(PedPyValueError, match="polyorder"):
        savitzky_golay_filter(traj_data=traj, window_size=5, polyorder=5)
    with pytest.raises(PedPyValueError, match="polyorder"):
        savitzky_golay_filter(traj_data=traj, window_size=5, polyorder=-1)
    with pytest.raises(PedPyTypeError):
        savitzky_golay_filter(traj_data="bad", window_size=5, polyorder=2)  # type: ignore[arg-type]
