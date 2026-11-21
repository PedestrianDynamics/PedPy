import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from tests.utils.utils import get_trajectory

from pedpy import (
    ACC_COL,
    A_X_COL,
    A_Y_COL,
    FRAME_COL,
    ID_COL,
    SPEED_COL,
    V_X_COL,
    V_Y_COL,
    X_COL,
    Y_COL,
    AccelerationCalculation,
    BorderFillMethod,
    MeasurementLine,
    PedPyValueError,
    SpeedCalculation,
    TrajectoryData,
    WalkableArea,
    compute_individual_acceleration,
    compute_individual_speed,
    compute_individual_voronoi_polygons,
    compute_time_distance_line,
    plot_acceleration,
    plot_speed,
    plot_speed_distribution,
    plot_time_distance,
    plot_voronoi_cells,
)


def _make_linear_trajectory(number_frames: int = 12, frame_rate: float = 10.0) -> TrajectoryData:
    return TrajectoryData(
        data=get_trajectory(
            shape=[2, 1],
            number_frames=number_frames,
            start_position=np.array([0.0, 0.0]),
            movement_direction=np.array([0.2, 0.0]),
            ped_distance=1.0,
        ),
        frame_rate=frame_rate,
    )


@pytest.mark.parametrize(
    "speed_calculation",
    [SpeedCalculation.BORDER_EXCLUDE, SpeedCalculation.BORDER_ADAPTIVE],
)
def test_individual_speed_border_fill_default_unchanged(speed_calculation: SpeedCalculation):
    traj_data = _make_linear_trajectory(number_frames=12)
    speed_default = compute_individual_speed(
        traj_data=traj_data,
        frame_step=3,
        compute_velocity=True,
        speed_calculation=speed_calculation,
    )
    speed_none = compute_individual_speed(
        traj_data=traj_data,
        frame_step=3,
        compute_velocity=True,
        speed_calculation=speed_calculation,
        border_fill=BorderFillMethod.NONE,
    )

    pd.testing.assert_frame_equal(speed_default, speed_none)
    assert len(speed_default) < len(traj_data.data)


@pytest.mark.parametrize(
    "speed_calculation",
    [SpeedCalculation.BORDER_EXCLUDE, SpeedCalculation.BORDER_ADAPTIVE],
)
def test_individual_speed_border_fill_nan(speed_calculation: SpeedCalculation):
    traj_data = _make_linear_trajectory(number_frames=12)
    speed_truncated = compute_individual_speed(
        traj_data=traj_data,
        frame_step=3,
        compute_velocity=True,
        speed_calculation=speed_calculation,
    )
    speed_nan = compute_individual_speed(
        traj_data=traj_data,
        frame_step=3,
        compute_velocity=True,
        speed_calculation=speed_calculation,
        border_fill=BorderFillMethod.NAN,
    )

    assert len(speed_nan) == len(traj_data.data)
    pd.testing.assert_frame_equal(
        speed_nan[[ID_COL, FRAME_COL]].reset_index(drop=True),
        traj_data.data[[ID_COL, FRAME_COL]].reset_index(drop=True),
    )

    valid_rows = speed_nan.dropna().reset_index(drop=True)
    pd.testing.assert_frame_equal(
        valid_rows.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True),
        speed_truncated.sort_values([ID_COL, FRAME_COL]).reset_index(drop=True),
    )

    # First and last frame of each pedestrian must be NaN
    for _, ped_df in speed_nan.groupby(ID_COL):
        ped_sorted = ped_df.sort_values(FRAME_COL)
        assert np.isnan(ped_sorted.iloc[0][SPEED_COL])
        assert np.isnan(ped_sorted.iloc[-1][SPEED_COL])
        assert np.isnan(ped_sorted.iloc[0][V_X_COL])
        assert np.isnan(ped_sorted.iloc[-1][V_Y_COL])


@pytest.mark.parametrize(
    "speed_calculation",
    [SpeedCalculation.BORDER_EXCLUDE, SpeedCalculation.BORDER_ADAPTIVE],
)
def test_individual_speed_border_fill_last_value(speed_calculation: SpeedCalculation):
    # Accelerating single pedestrian so start and end valid speeds differ
    frames = np.arange(10)
    traj_df = pd.DataFrame(
        {
            ID_COL: [1] * 10,
            FRAME_COL: frames,
            X_COL: 0.1 * frames**2,
            Y_COL: np.zeros_like(frames, dtype=float),
        }
    )
    traj_data = TrajectoryData(data=traj_df, frame_rate=10.0)

    speed_none = compute_individual_speed(
        traj_data=traj_data,
        frame_step=2,
        compute_velocity=True,
        speed_calculation=speed_calculation,
    ).sort_values(FRAME_COL)
    speed_last = compute_individual_speed(
        traj_data=traj_data,
        frame_step=2,
        compute_velocity=True,
        speed_calculation=speed_calculation,
        border_fill=BorderFillMethod.LAST_VALUE,
    ).sort_values(FRAME_COL)

    assert len(speed_last) == len(traj_data.data)
    assert not speed_last[[SPEED_COL, V_X_COL, V_Y_COL]].isna().any().any()

    first_valid_frame = int(speed_none[FRAME_COL].min())
    last_valid_frame = int(speed_none[FRAME_COL].max())
    first_valid_speed = float(speed_none.iloc[0][SPEED_COL])
    last_valid_speed = float(speed_none.iloc[-1][SPEED_COL])

    leading = speed_last[speed_last[FRAME_COL] < first_valid_frame][SPEED_COL]
    trailing = speed_last[speed_last[FRAME_COL] > last_valid_frame][SPEED_COL]
    assert np.allclose(leading, first_valid_speed)
    assert np.allclose(trailing, last_valid_speed)


@pytest.mark.parametrize(
    "border_fill",
    [BorderFillMethod.NONE, BorderFillMethod.NAN, BorderFillMethod.LAST_VALUE],
)
def test_individual_speed_single_sided_ignores_border_fill(border_fill: BorderFillMethod):
    # Include a 1-frame pedestrian (whose single-sided speed is NaN) to verify
    # border_fill has strictly zero effect when BORDER_SINGLE_SIDED is selected.
    traj_df = pd.DataFrame(
        {
            ID_COL: [1, 1, 1, 1, 2],
            FRAME_COL: [0, 1, 2, 3, 0],
            X_COL: [0.0, 0.5, 1.0, 1.5, 2.0],
            Y_COL: [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    traj_data = TrajectoryData(data=traj_df, frame_rate=10.0)

    baseline = compute_individual_speed(
        traj_data=traj_data,
        frame_step=2,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    with_fill = compute_individual_speed(
        traj_data=traj_data,
        frame_step=2,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
        border_fill=border_fill,
    )

    pd.testing.assert_frame_equal(baseline, with_fill)


def test_individual_acceleration_border_fill_strategies():
    frames = np.arange(12)
    traj_df = pd.DataFrame(
        {
            ID_COL: [1] * 12,
            FRAME_COL: frames,
            X_COL: 0.02 * frames**3,
            Y_COL: np.zeros_like(frames, dtype=float),
        }
    )
    traj_data = TrajectoryData(data=traj_df, frame_rate=10.0)

    acc_default = compute_individual_acceleration(
        traj_data=traj_data,
        frame_step=2,
        compute_acceleration_components=True,
        acceleration_calculation=AccelerationCalculation.BORDER_EXCLUDE,
    )
    acc_nan = compute_individual_acceleration(
        traj_data=traj_data,
        frame_step=2,
        compute_acceleration_components=True,
        acceleration_calculation=AccelerationCalculation.BORDER_EXCLUDE,
        border_fill=BorderFillMethod.NAN,
    )
    acc_last = compute_individual_acceleration(
        traj_data=traj_data,
        frame_step=2,
        compute_acceleration_components=True,
        acceleration_calculation=AccelerationCalculation.BORDER_EXCLUDE,
        border_fill=BorderFillMethod.LAST_VALUE,
    )

    assert len(acc_default) == 8
    assert len(acc_nan) == len(traj_data.data)
    assert len(acc_last) == len(traj_data.data)

    # First 2 * frame_step = 4 frames are dropped in acceleration calculation
    assert acc_nan.iloc[:4][ACC_COL].isna().all()
    assert acc_nan.iloc[:4][A_X_COL].isna().all()
    assert acc_nan.iloc[:4][A_Y_COL].isna().all()
    pd.testing.assert_frame_equal(
        acc_nan.iloc[4:].reset_index(drop=True),
        acc_default.reset_index(drop=True),
    )

    assert not acc_last[[ACC_COL, A_X_COL, A_Y_COL]].isna().any().any()
    assert np.allclose(acc_last.iloc[:4][ACC_COL], acc_default.iloc[0][ACC_COL])


def test_border_fill_last_value_does_not_leak_across_pedestrians():
    # Ped 2 has only 2 frames (too short for frame_step=2) between Ped 1 and Ped 3
    traj_df = pd.DataFrame(
        {
            ID_COL: [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            FRAME_COL: [0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 4],
            X_COL: [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            Y_COL: [0.0] * 12,
        }
    )
    # Shuffle row order to also verify 1:1 row order preservation
    shuffled_df = traj_df.iloc[[5, 0, 7, 1, 6, 8, 2, 9, 3, 10, 4, 11]].reset_index(drop=True)
    traj_data = TrajectoryData(data=shuffled_df, frame_rate=1.0)

    speed_last = compute_individual_speed(
        traj_data=traj_data,
        frame_step=2,
        speed_calculation=SpeedCalculation.BORDER_EXCLUDE,
        border_fill=BorderFillMethod.LAST_VALUE,
    )

    pd.testing.assert_frame_equal(
        speed_last[[ID_COL, FRAME_COL]],
        traj_data.data[[ID_COL, FRAME_COL]],
    )
    assert np.allclose(speed_last[speed_last[ID_COL] == 1][SPEED_COL], 1.0)
    assert np.allclose(speed_last[speed_last[ID_COL] == 3][SPEED_COL], 2.0)
    assert speed_last[speed_last[ID_COL] == 2][SPEED_COL].isna().all()


def test_invalid_border_fill_raises_value_error():
    traj_data = _make_linear_trajectory(number_frames=6)
    with pytest.raises(PedPyValueError, match="border fill method"):
        compute_individual_speed(
            traj_data=traj_data,
            frame_step=1,
            border_fill="nan",  # type: ignore[arg-type]
        )
    with pytest.raises(PedPyValueError, match="border fill method"):
        compute_individual_acceleration(
            traj_data=traj_data,
            frame_step=1,
            border_fill="last_value",  # type: ignore[arg-type]
        )


def test_plotting_handles_nan_filled_speed_and_acceleration():
    traj_data = _make_linear_trajectory(number_frames=10)
    walkable_area = WalkableArea([(-2, -2), (10, -2), (10, 5), (-2, 5)])
    measurement_line = MeasurementLine([(5, -2), (5, 5)])

    speed_nan = compute_individual_speed(
        traj_data=traj_data,
        frame_step=3,
        speed_calculation=SpeedCalculation.BORDER_EXCLUDE,
        border_fill=BorderFillMethod.NAN,
    )
    acc_nan = compute_individual_acceleration(
        traj_data=traj_data,
        frame_step=2,
        border_fill=BorderFillMethod.NAN,
    )

    fig, axes = plt.subplots(2, 3)
    try:
        plot_speed(speed=speed_nan[speed_nan[ID_COL] == 0], axes=axes[0, 0])
        plot_acceleration(acceleration=acc_nan[acc_nan[ID_COL] == 0], axes=axes[0, 1])
        plot_speed_distribution(speed=speed_nan, axes=axes[0, 2])

        # Also check all-NaN speed distribution does not raise
        all_nan_speed = speed_nan.copy()
        all_nan_speed[SPEED_COL] = np.nan
        plot_speed_distribution(speed=all_nan_speed, axes=axes[1, 0])

        td = compute_time_distance_line(traj_data=traj_data, measurement_line=measurement_line)
        plot_time_distance(time_distance=td, speed=speed_nan, axes=axes[1, 1])

        voronoi = compute_individual_voronoi_polygons(traj_data=traj_data, walkable_area=walkable_area).merge(
            speed_nan, on=[ID_COL, FRAME_COL]
        )
        plot_voronoi_cells(voronoi_data=voronoi, frame=0, color_by_column=SPEED_COL, axes=axes[1, 2])
    finally:
        plt.close(fig)
