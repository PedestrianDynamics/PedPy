import numpy as np
import pandas as pd
import pytest

from pedpy import (
    FRAME_COL,
    ID_COL,
    SPEED_COL,
    V_X_COL,
    V_Y_COL,
    X_COL,
    Y_COL,
    PairNormalizationMethod,
    PedPyTypeError,
    PedPyValueError,
    SpeedCalculation,
    SpeedError,
    TrajectoryData,
    WalkableArea,
    compute_approach_rate_pair_distribution,
    compute_individual_speed,
    compute_pair_distribution_function,
)


def test_pair_distribution_spatial_randomization_static_crowd():
    # Static crowd: 4 pedestrians standing on a 1m x 1m square over 20 frames
    # inside a 10m x 10m room with an obstacle hole.
    coords = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)]
    rows = [
        {ID_COL: pid, FRAME_COL: frame, X_COL: x, Y_COL: y}
        for frame in range(20)
        for pid, (x, y) in enumerate(coords, start=1)
    ]
    traj = TrajectoryData(data=pd.DataFrame(rows), frame_rate=10.0)
    walkable_area = WalkableArea(
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        obstacles=[[(5.0, 5.0), (7.0, 5.0), (7.0, 7.0), (5.0, 7.0)]],
    )

    r_bins, g_r = compute_pair_distribution_function(
        traj_data=traj,
        radius_bin_size=0.6,
        randomisation_stacking=25,
        normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION,
        walkable_area=walkable_area,
        seed=42,
    )

    # All actual pair distances are 1.0 or sqrt(2) ~= 1.414 -> 0 pairs in (0, 0.6], concentrated in (0.6, 1.2]
    assert len(r_bins) == len(g_r)
    assert g_r[0] == 0.0
    assert g_r[1] > 5.0

    # Seed reproducibility
    _, g_r_repeat = compute_pair_distribution_function(
        traj_data=traj,
        radius_bin_size=0.6,
        randomisation_stacking=25,
        normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION,
        walkable_area=walkable_area,
        seed=42,
    )
    assert np.allclose(g_r, g_r_repeat, equal_nan=True)


def test_pair_distribution_spatial_randomization_requires_walkable_area():
    traj_df = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            X_COL: [0.0, 2.0],
            Y_COL: [0.0, 0.0],
        }
    )
    traj = TrajectoryData(data=traj_df, frame_rate=1.0)

    with pytest.raises(PedPyValueError, match="walkable_area"):
        compute_pair_distribution_function(
            traj_data=traj,
            radius_bin_size=0.5,
            normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION,
        )

    with pytest.raises(PedPyTypeError, match="WalkableArea"):
        compute_pair_distribution_function(
            traj_data=traj,
            radius_bin_size=0.5,
            normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION,
            walkable_area="invalid",  # type: ignore[arg-type]
        )


def test_approach_rate_pair_distribution_1d_and_2d():
    # Two pedestrians approach head-on for frames 0..4 (x1 increases, x2 decreases -> approach rate +2.0 m/s)
    # and separate after passing for frames 6..9 (approach rate -2.0 m/s).
    rows = []
    for frame in range(10):
        t = float(frame)
        rows.append({ID_COL: 1, FRAME_COL: frame, X_COL: -4.5 + 1.0 * t, Y_COL: 0.0})
        rows.append({ID_COL: 2, FRAME_COL: frame, X_COL: 4.5 - 1.0 * t, Y_COL: 0.2})
    traj = TrajectoryData(data=pd.DataFrame(rows), frame_rate=1.0)
    walkable_area = WalkableArea([(-10.0, -5.0), (10.0, -5.0), (10.0, 5.0), (-10.0, 5.0)])

    speed_df = compute_individual_speed(
        traj_data=traj,
        frame_step=1,
        compute_velocity=True,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    v_edges, g_v = compute_approach_rate_pair_distribution(
        traj_data=traj,
        individual_speed=speed_df,
        approach_rate_bins=[-3.0, -1.0, 1.0, 3.0],
        randomisation_stacking=40,
        normalization_method=PairNormalizationMethod.SPATIAL_RANDOMIZATION,
        walkable_area=walkable_area,
        seed=7,
    )
    assert np.allclose(v_edges, [-1.0, 1.0, 3.0])
    assert len(g_v) == 3
    # Middle bin (-1, 1] has near-zero actual observations (except closest pass), outer bins > 0
    assert g_v[0] > 0.0
    assert g_v[2] > 0.0

    # 2D (r, v) evaluation with time randomization and automatic approach_rate_bin_size
    res_2d = compute_approach_rate_pair_distribution(
        traj_data=traj,
        individual_speed=speed_df,
        approach_rate_bin_size=1.5,
        radius_bin_size=2.5,
        randomisation_stacking=30,
        seed=11,
    )
    assert len(res_2d) == 3
    r_edges, v_edges_auto, g_rv = res_2d
    assert g_rv.shape == (len(r_edges), len(v_edges_auto))
    assert np.any(np.isfinite(g_rv) & (g_rv > 0))


def test_approach_rate_pair_distribution_validation_errors():
    traj_df = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            X_COL: [0.0, 1.0],
            Y_COL: [0.0, 0.0],
        }
    )
    traj = TrajectoryData(data=traj_df, frame_rate=1.0)
    speed_no_vel = pd.DataFrame({ID_COL: [1, 2], FRAME_COL: [0, 0], SPEED_COL: [1.0, 1.0]})
    speed_vel = pd.DataFrame(
        {
            ID_COL: [1, 2],
            FRAME_COL: [0, 0],
            SPEED_COL: [1.0, 1.0],
            V_X_COL: [1.0, -1.0],
            V_Y_COL: [0.0, 0.0],
        }
    )

    with pytest.raises(SpeedError, match="compute_velocity=True"):
        compute_approach_rate_pair_distribution(
            traj_data=traj,
            individual_speed=speed_no_vel,
            approach_rate_bin_size=0.5,
        )

    with pytest.raises(PedPyValueError, match="exactly one"):
        compute_approach_rate_pair_distribution(
            traj_data=traj,
            individual_speed=speed_vel,
        )

    with pytest.raises(PedPyValueError, match="strictly increasing"):
        compute_approach_rate_pair_distribution(
            traj_data=traj,
            individual_speed=speed_vel,
            approach_rate_bins=[1.0, 0.5],
        )
