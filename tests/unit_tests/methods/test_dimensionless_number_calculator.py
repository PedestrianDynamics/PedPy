import pandas as pd
import pytest

from pedpy.column_identifier import AVOIDANCE_COL, FRAME_COL, ID_COL, INTRUSION_COL
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.methods.dimensionless_number_calculator import (
    IntrusionMethod,
    compute_avoidance,
    compute_intrusion,
)


def _make_traj(positions, frame_rate=10):
    """Build a single-frame TrajectoryData from a list of (x, y) positions."""
    rows = []
    for pid, (x, y) in enumerate(positions):
        rows.append({"id": pid, "frame": 0, "x": x, "y": y})
    return TrajectoryData(data=pd.DataFrame(rows), frame_rate=frame_rate)


# ---------------------------------------------------------------------------
# Intrusion
# ---------------------------------------------------------------------------


class TestComputeIntrusion:
    """Tests for compute_intrusion (Cordes et al. 2024, Eq. 1)."""

    def test_two_agents_known_distance(self):
        """Two agents at known distance, check In_i by hand."""
        r_soc = 0.8
        l_min = 0.2
        d = 0.5  # distance between the two agents
        # Expected per-pair intrusion: ((0.8-0.2)/(0.5-0.2))^2 = (0.6/0.3)^2 = 4.0
        traj = _make_traj([(0, 0), (d, 0)])
        result = compute_intrusion(traj_data=traj, r_soc=r_soc, l_min=l_min)

        for pid in [0, 1]:
            val = result.loc[result[ID_COL] == pid, INTRUSION_COL].iloc[0]
            assert val == pytest.approx(4.0)

    def test_neighbor_cutoff_excludes_distant_agents(self):
        """Agents beyond 3*r_soc should not contribute to intrusion."""
        r_soc = 0.8
        l_min = 0.2
        far = 3 * r_soc + 0.1  # just beyond cutoff
        traj = _make_traj([(0, 0), (far, 0)])
        result = compute_intrusion(traj_data=traj, r_soc=r_soc, l_min=l_min)

        # Both agents should be absent (no neighbors within cutoff)
        assert result.empty

    def test_three_agents_sum(self):
        """Three agents: agent 0 has two neighbors, check sum."""
        r_soc = 0.8
        l_min = 0.2
        d1 = 0.5
        d2 = 0.6
        traj = _make_traj([(0, 0), (d1, 0), (0, d2)])

        result = compute_intrusion(
            traj_data=traj, r_soc=r_soc, l_min=l_min, method=IntrusionMethod.SUM
        )

        in_01 = ((r_soc - l_min) / (d1 - l_min)) ** 2
        in_02 = ((r_soc - l_min) / (d2 - l_min)) ** 2
        expected_0 = in_01 + in_02

        val = result.loc[result[ID_COL] == 0, INTRUSION_COL].iloc[0]
        assert val == pytest.approx(expected_0)

    def test_method_max(self):
        """IntrusionMethod.MAX returns the max over neighbors."""
        r_soc = 0.8
        l_min = 0.2
        d1 = 0.5
        d2 = 0.6
        traj = _make_traj([(0, 0), (d1, 0), (0, d2)])

        result = compute_intrusion(
            traj_data=traj, r_soc=r_soc, l_min=l_min, method=IntrusionMethod.MAX
        )

        in_01 = ((r_soc - l_min) / (d1 - l_min)) ** 2
        in_02 = ((r_soc - l_min) / (d2 - l_min)) ** 2
        expected_0 = max(in_01, in_02)

        val = result.loc[result[ID_COL] == 0, INTRUSION_COL].iloc[0]
        assert val == pytest.approx(expected_0)


# ---------------------------------------------------------------------------
# Avoidance
# ---------------------------------------------------------------------------


def _make_moving_traj(agent_data, frame_rate=10):
    """Build multi-frame TrajectoryData for agents with constant velocity.

    agent_data: list of (x0, y0, vx, vy) per agent.
    Creates two frames separated by dt = 1/frame_rate so velocities resolve.
    """
    rows = []
    dt = 1.0 / frame_rate
    for pid, (x0, y0, vx, vy) in enumerate(agent_data):
        for frame in range(3):
            rows.append(
                {
                    "id": pid,
                    "frame": frame,
                    "x": x0 + vx * frame * dt,
                    "y": y0 + vy * frame * dt,
                }
            )
    return TrajectoryData(data=pd.DataFrame(rows), frame_rate=frame_rate)


class TestComputeAvoidance:
    """Tests for compute_avoidance (Cordes et al. 2024, Eq. 2)."""

    def test_head_on_collision(self):
        """Two agents approaching head-on: TTC is analytically known."""
        # Agent 0 at x=-2, moving right at v=1
        # Agent 1 at x=+2, moving left at v=-1
        # At frame 1 (dt=0.1): positions are -1.9 and +1.9, distance=3.8
        # Relative speed = 2, TTC = (d - R) / |delta_v| = (3.8 - 0.2) / 2 = 1.8
        radius = 0.2
        tau_0 = 3.0
        d_at_frame1 = 3.8
        expected_ttc = (d_at_frame1 - radius) / 2.0
        expected_av = tau_0 / expected_ttc

        traj = _make_moving_traj([(-2, 0, 1, 0), (2, 0, -1, 0)], frame_rate=10)
        result = compute_avoidance(
            traj_data=traj, frame_step=1, radius=radius, tau_0=tau_0
        )

        # Check frame 1 (middle frame where velocity is computed)
        row = result[(result[ID_COL] == 0) & (result[FRAME_COL] == 1)]
        assert len(row) == 1
        assert row[AVOIDANCE_COL].iloc[0] == pytest.approx(expected_av, rel=0.05)

    def test_diverging_agents_zero_avoidance(self):
        """Two agents moving apart should have Av = 0 (TTC = inf)."""
        tau_0 = 3.0
        traj = _make_moving_traj([(-2, 0, -1, 0), (2, 0, 1, 0)], frame_rate=10)
        result = compute_avoidance(
            traj_data=traj, frame_step=1, radius=0.2, tau_0=tau_0
        )

        row = result[(result[ID_COL] == 0) & (result[FRAME_COL] == 1)]
        assert len(row) == 1
        assert row[AVOIDANCE_COL].iloc[0] == pytest.approx(0.0, abs=1e-10)
