"""Tests for steady state detection (modified CUSUM algorithm)."""

import numpy as np
import pytest

from pedpy.methods.steady_state_detector import (
    SteadyStateResult,
    _compute_cusum,
    _compute_theta,
    _find_steady_intervals,
    _intersect_interval_lists,
    combine_steady_states,
    detect_steady_state,
)


class TestComputeCusum:
    """Tests for the CUSUM statistics computation."""

    def test_constant_series_at_reference_mean_decreases(self):
        """CUSUM should decrease when observations match reference."""
        values = np.full(200, 5.0)
        q = 2.326  # norm.ppf(0.99)
        cusum = _compute_cusum(values, ref_mean=5.0, ref_std=1.0, q=q, s_max=100)
        assert cusum[0] == 100  # starts at s_max
        # should decrease since all x_tilde = 0, F = -1
        assert cusum[-1] == 0

    def test_outlier_series_stays_high(self):
        """CUSUM should stay high when observations are far from reference."""
        values = np.full(50, 100.0)
        q = 2.326
        cusum = _compute_cusum(values, ref_mean=5.0, ref_std=1.0, q=q, s_max=100)
        # all values are outliers, F = +1 always, stays at s_max
        assert cusum[-1] == 100

    def test_cusum_bounded(self):
        """CUSUM should always be in [0, s_max]."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 3, size=500)
        q = 2.326
        cusum = _compute_cusum(values, ref_mean=0.0, ref_std=1.0, q=q, s_max=100)
        assert np.all(cusum >= 0)
        assert np.all(cusum <= 100)


class TestComputeTheta:
    """Tests for theta calibration via autoregressive model."""

    def test_theta_matches_original_implementation(self):
        """Theta values should match the original SteadyState.py code."""
        # These values were verified against the original implementation
        expected = {0.95: 84, 0.97: 78, 0.99: 53}
        for acf, expected_theta in expected.items():
            theta = _compute_theta(
                acf=acf,
                q=2.326,
                gamma=0.99,
                s_max=100,
                grid_size=500,
                grid_half_width=3.2,
            )
            assert theta == expected_theta, f"acf={acf}: got theta={theta}, expected {expected_theta}"

    def test_theta_in_valid_range(self):
        """Theta should be between 1 and s_max."""
        theta = _compute_theta(
            acf=0.9,
            q=2.326,
            gamma=0.99,
            s_max=100,
            grid_size=500,
            grid_half_width=3.2,
        )
        assert 1 <= theta <= 100


class TestFindSteadyIntervals:
    """Tests for extracting steady intervals from CUSUM."""

    def test_no_steady_state(self):
        """When CUSUM never drops below theta, no intervals found."""
        frames = np.arange(100, dtype=float)
        cusum = np.full(100, 90.0)
        starts, ends = _find_steady_intervals(
            frames=frames,
            cusum=cusum,
            theta=50.0,
            s_max=100,
        )
        assert len(starts) == 0
        assert len(ends) == 0

    def test_single_steady_interval(self):
        """Detect a single steady interval."""
        frames = np.arange(200, dtype=float)
        cusum = np.full(200, 80.0)
        # create a dip below theta in the middle
        cusum[50:150] = 10.0
        starts, ends = _find_steady_intervals(
            frames=frames,
            cusum=cusum,
            theta=50.0,
            s_max=100,
        )
        assert len(starts) == 1
        # the interval should be adjusted by reaction times


class TestDetectSteadyState:
    """Integration tests for the full detection pipeline."""

    def test_stationary_signal(self):
        """A stationary signal should have a large steady interval."""
        rng = np.random.default_rng(123)
        n = 1000
        frames = np.arange(n, dtype=float)
        values = rng.normal(5.0, 0.5, size=n)

        result = detect_steady_state(
            frames=frames,
            values=values,
            ref_start=200,
            ref_end=400,
        )
        # should detect at least one steady interval
        assert len(result.frame_start) >= 1
        assert result.theta > 0
        assert result.mean != 0

    def test_step_function_detects_transition(self):
        """A step function should yield a steady interval only in flat parts."""
        n = 600
        frames = np.arange(n, dtype=float)
        values = np.concatenate(
            [
                np.full(200, 3.0) + np.random.default_rng(1).normal(0, 0.1, 200),
                np.full(200, 6.0) + np.random.default_rng(2).normal(0, 0.1, 200),
                np.full(200, 3.0) + np.random.default_rng(3).normal(0, 0.1, 200),
            ]
        )
        # reference from first plateau
        result = detect_steady_state(
            frames=frames,
            values=values,
            ref_start=50,
            ref_end=150,
        )
        # steady intervals should NOT include the middle plateau (around 200-400)
        for s, e in zip(result.frame_start, result.frame_end):
            # no steady interval should be fully contained in the step region
            assert not (s >= 210 and e <= 390), f"Unexpected steady interval {s}-{e} in step region"

    def test_invalid_reference_range(self):
        """Should raise ValueError for invalid reference range."""
        frames = np.arange(100, dtype=float)
        values = np.ones(100)
        with pytest.raises(ValueError):
            detect_steady_state(
                frames=frames,
                values=values,
                ref_start=50,
                ref_end=10,
            )

    def test_mismatched_lengths(self):
        """Should raise ValueError for mismatched input lengths."""
        with pytest.raises(ValueError):
            detect_steady_state(
                frames=np.arange(10),
                values=np.ones(5),
                ref_start=0,
                ref_end=5,
            )

    def test_zero_std_reference(self):
        """Should raise ValueError when reference has zero std."""
        frames = np.arange(100, dtype=float)
        values = np.ones(100)  # all identical
        with pytest.raises(ValueError, match="zero standard deviation"):
            detect_steady_state(
                frames=frames,
                values=values,
                ref_start=10,
                ref_end=50,
            )


class TestCombineSteadyStates:
    """Tests for combining steady states across multiple series."""

    def test_overlapping_intervals(self):
        """Overlapping intervals from two series should be intersected."""
        r1 = SteadyStateResult(
            frame_start=np.array([100.0]),
            frame_end=np.array([500.0]),
            cusum=np.array([]),
            theta=50.0,
            mean=3.0,
            std=0.5,
            acf=0.9,
        )
        r2 = SteadyStateResult(
            frame_start=np.array([200.0]),
            frame_end=np.array([600.0]),
            cusum=np.array([]),
            theta=70.0,
            mean=0.7,
            std=0.1,
            acf=0.95,
        )
        combined = combine_steady_states([r1, r2])
        assert len(combined) == 1
        assert combined[0][0] == 200.0
        assert combined[0][1] == 500.0

    def test_no_overlap(self):
        """Non-overlapping intervals should yield empty result."""
        r1 = SteadyStateResult(
            frame_start=np.array([100.0]),
            frame_end=np.array([200.0]),
            cusum=np.array([]),
            theta=50.0,
            mean=3.0,
            std=0.5,
            acf=0.9,
        )
        r2 = SteadyStateResult(
            frame_start=np.array([300.0]),
            frame_end=np.array([400.0]),
            cusum=np.array([]),
            theta=70.0,
            mean=0.7,
            std=0.1,
            acf=0.95,
        )
        combined = combine_steady_states([r1, r2])
        assert len(combined) == 0

    def test_single_result(self):
        """Single result should return its intervals directly."""
        r1 = SteadyStateResult(
            frame_start=np.array([100.0, 300.0]),
            frame_end=np.array([200.0, 400.0]),
            cusum=np.array([]),
            theta=50.0,
            mean=3.0,
            std=0.5,
            acf=0.9,
        )
        combined = combine_steady_states([r1])
        assert len(combined) == 2

    def test_empty_results(self):
        """Empty result list should return empty."""
        assert combine_steady_states([]) == []


class TestIntersectIntervalLists:
    """Tests for interval list intersection."""

    def test_basic_overlap(self):
        """Two overlapping intervals should produce their intersection."""
        result = _intersect_interval_lists(
            [(1, 10)],
            [(5, 15)],
        )
        assert result == [(5, 10)]

    def test_multiple_overlaps(self):
        """Multiple intervals should produce all pairwise intersections."""
        result = _intersect_interval_lists(
            [(0, 10), (20, 30)],
            [(5, 25)],
        )
        assert len(result) == 2
        assert (5, 10) in result
        assert (20, 25) in result


class TestReferenceData:
    """Validation against the original SteadyState.py reference results."""

    @pytest.fixture
    def reference_data(self):
        """Load the reference dataset."""
        import os

        data_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "materials",
            "SteadyState",
            "rho_v_Voronoi_traj_AO_b240.txt_id_1.dat",
        )
        if not os.path.exists(data_path):
            pytest.skip("Reference data not available")
        data = np.loadtxt(data_path, usecols=[0, 1, 2])
        data = data[data[:, 1] != 0]
        return data

    def test_density_steady_state(self, reference_data):
        """Density steady state should match reference within tolerance."""
        data = reference_data
        result = detect_steady_state(
            frames=data[:, 0],
            values=data[:, 1],
            ref_start=240,
            ref_end=640,
        )
        # Reference from the legacy SteadyState.py implementation.
        np.testing.assert_array_equal(result.frame_start, np.array([161.0]))
        np.testing.assert_array_equal(result.frame_end, np.array([866.0]))

    def test_speed_steady_state(self, reference_data):
        """Speed steady state should match reference within tolerance."""
        data = reference_data
        result = detect_steady_state(
            frames=data[:, 0],
            values=data[:, 2],
            ref_start=240,
            ref_end=640,
        )
        # Reference from the legacy SteadyState.py implementation.
        np.testing.assert_array_equal(result.frame_start, np.array([231.0]))
        np.testing.assert_array_equal(result.frame_end, np.array([965.0]))

    def test_combined_steady_state(self, reference_data):
        """Combined steady state should match reference within tolerance."""
        data = reference_data
        result_rho = detect_steady_state(
            frames=data[:, 0],
            values=data[:, 1],
            ref_start=240,
            ref_end=640,
        )
        result_v = detect_steady_state(
            frames=data[:, 0],
            values=data[:, 2],
            ref_start=240,
            ref_end=640,
        )
        combined = combine_steady_states([result_rho, result_v])
        assert combined == [(231.0, 866.0)]
