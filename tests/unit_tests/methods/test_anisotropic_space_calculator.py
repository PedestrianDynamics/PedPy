"""Unit tests for pedpy.methods.anisotropic_space_calculator."""

import math

import numpy as np
import pandas as pd
import pytest

import pedpy
from pedpy import (
    PedPyTypeError,
    PedPyValueError,
    TrajectoryData,
    compute_anisotropic_personal_space,
    compute_shared_space_cpa,
    detect_social_dyads,
)


def _make_linear_traj(
    agents: list[tuple[int, float, float, float, float]],
    n_frames: int = 10,
    fps: float = 10.0,
) -> TrajectoryData:
    """Build synthetic linear trajectories from (id, x0, y0, vx, vy) tuples."""
    rows = []
    for frame in range(n_frames):
        t = frame / fps
        for pid, x0, y0, vx, vy in agents:
            rows.append(
                {
                    "id": pid,
                    "frame": frame,
                    "x": x0 + vx * t,
                    "y": y0 + vy * t,
                }
            )
    return TrajectoryData(data=pd.DataFrame(rows), frame_rate=fps)


class TestAnisotropicPersonalSpace:
    """Tests for compute_anisotropic_personal_space."""

    def test_top_level_exports(self):
        assert hasattr(pedpy, "compute_anisotropic_personal_space")
        assert hasattr(pedpy, "detect_social_dyads")
        assert hasattr(pedpy, "compute_shared_space_cpa")

    def test_semi_axes_area_and_distraction_modulation(self):
        traj = _make_linear_traj([(1, 0.0, 0.0, 1.5, 0.0)], n_frames=6, fps=10.0)
        res_att = compute_anisotropic_personal_space(traj_data=traj)

        expected_a = 0.42 + 0.36 * 1.5
        expected_b = 0.28
        assert np.allclose(res_att["a_m"], expected_a, atol=1e-9)
        assert np.allclose(res_att["b_m"], expected_b, atol=1e-9)
        assert np.allclose(res_att["a_b_ratio"], expected_a / expected_b, atol=1e-9)
        assert np.allclose(res_att["ellipse_area_m2"], math.pi * expected_a * expected_b, atol=1e-9)
        assert np.all(np.isinf(res_att["min_d_eff"]))

        # Distracted via gaze_state == 0 (ON_PHONE)
        att_df = pd.DataFrame({"id": [1], "gaze_state": [0]})
        res_dist = compute_anisotropic_personal_space(traj_data=traj, attention_df=att_df)
        exp_a_dist = expected_a + 0.22
        exp_b_dist = expected_b + 0.16
        assert np.allclose(res_dist["a_m"], exp_a_dist, atol=1e-9)
        assert np.allclose(res_dist["b_m"], exp_b_dist, atol=1e-9)
        assert np.all(res_dist["ellipse_area_m2"] > res_att["ellipse_area_m2"])

    def test_directional_elliptical_distance_ahead_vs_lateral(self):
        # Agent 1 walks along +x at 1.4 m/s; Agent 2 is 1.0 m ahead along +x
        traj_ahead = _make_linear_traj(
            [(1, 0.0, 0.0, 1.4, 0.0), (2, 1.0, 0.0, 1.4, 0.0)],
            n_frames=5,
            fps=10.0,
        )
        # Agent 1 walks along +x at 1.4 m/s; Agent 2 is 1.0 m laterally along +y
        traj_lat = _make_linear_traj(
            [(1, 0.0, 0.0, 1.4, 0.0), (2, 0.0, 1.0, 1.4, 0.0)],
            n_frames=5,
            fps=10.0,
        )

        df_ahead = compute_anisotropic_personal_space(traj_data=traj_ahead)
        df_lat = compute_anisotropic_personal_space(traj_data=traj_lat)

        d_eff_ahead = df_ahead.loc[df_ahead["id"] == 1, "min_d_eff"].iloc[2]
        d_eff_lat = df_lat.loc[df_lat["id"] == 1, "min_d_eff"].iloc[2]

        a_expected = 0.42 + 0.36 * 1.4
        b_expected = 0.28
        assert math.isclose(d_eff_ahead, 1.0 / a_expected, rel_tol=1e-9)
        assert math.isclose(d_eff_lat, 1.0 / b_expected, rel_tol=1e-9)
        assert d_eff_ahead < d_eff_lat

    def test_validation_errors(self):
        with pytest.raises(PedPyTypeError, match="Expected 'traj_data' to be a TrajectoryData"):
            compute_anisotropic_personal_space(traj_data=pd.DataFrame())  # type: ignore[arg-type]

        traj = _make_linear_traj([(1, 0.0, 0.0, 1.0, 0.0)], n_frames=3)
        with pytest.raises(PedPyValueError):
            compute_anisotropic_personal_space(traj_data=traj, a0=-0.1)


class TestDetectSocialDyads:
    """Tests for detect_social_dyads."""

    def test_abreast_vs_single_file_and_counterflow_rejection(self):
        # Dyad (1, 2): abreast side-by-side along +x (dy = 0.80 m -> 90 deg)
        # Dyad (10, 11): single-file leader-follower along +y (dy = 0.95 m -> 0 deg)
        # Passerby 99: counterflow passing near agent 1 briefly
        traj = _make_linear_traj(
            [
                (1, 0.0, 0.0, 1.3, 0.0),
                (2, 0.0, 0.80, 1.3, 0.0),
                (10, 5.0, 0.0, 0.0, 1.2),
                (11, 5.0, 0.95, 0.0, 1.2),
                (99, 1.2, 0.3, -1.3, 0.0),
            ],
            n_frames=10,
            fps=10.0,
        )

        dyads = detect_social_dyads(
            traj_data=traj,
            max_distance=1.25,
            max_speed_diff=0.35,
            min_shared_frames=5,
        )
        assert len(dyads) == 2

        row_12 = dyads[(dyads["ped_id_1"] == 1) & (dyads["ped_id_2"] == 2)].iloc[0]
        assert row_12["formation"] == "abreast"
        assert math.isclose(row_12["formation_angle_deg"], 90.0, abs_tol=1e-6)
        assert row_12["shared_frames"] == 10
        assert math.isclose(row_12["mean_distance_m"], 0.80, abs_tol=1e-6)

        row_1011 = dyads[(dyads["ped_id_1"] == 10) & (dyads["ped_id_2"] == 11)].iloc[0]
        assert row_1011["formation"] == "single_file"
        assert math.isclose(row_1011["formation_angle_deg"], 0.0, abs_tol=1e-6)

        # Counter-flowing agent 99 must never appear in detected social dyads
        assert 99 not in set(dyads["ped_id_1"]).union(set(dyads["ped_id_2"]))


class TestSharedSpaceCPA:
    """Tests for compute_shared_space_cpa."""

    def test_converging_overtake_and_diverging_wheeler(self):
        # Pedestrian at (0, 0) walking along +x at 1.0 m/s
        ped_traj = _make_linear_traj([(1, 0.0, 0.0, 1.0, 0.0)], n_frames=5, fps=10.0)

        # E-scooter 100 overtaking along +x at 5.0 m/s with lateral offset y = 0.50 m (< 0.75 m)
        # Cyclist 200 already ahead and diverging along +x at 4.0 m/s with y = 1.20 m
        whl_rows = []
        for f in range(5):
            t = f / 10.0
            whl_rows.append(
                {
                    "id": 100,
                    "frame": f,
                    "x": -8.0 + 5.0 * t,
                    "y": 0.50,
                    "wheeler_type": "e_scooter",
                }
            )
            whl_rows.append(
                {
                    "id": 200,
                    "frame": f,
                    "x": 4.0 + 4.0 * t,
                    "y": 1.20,
                    "wheeler_type": "cyclist",
                }
            )
        wheeler_df = pd.DataFrame(whl_rows)

        cpa = compute_shared_space_cpa(ped_traj=ped_traj, wheeler_df=wheeler_df, d_min=0.75)
        # At frame 0 (t=0): dr = (-8.0, 0.50), dv = (+4.0, 0.0)
        # tau_cpa = -(-8.0 * 4.0) / 16.0 = 2.0 s, d_cpa = |0.50| = 0.50 m
        f0_scooter = cpa[(cpa["frame"] == 0) & (cpa["wheeler_id"] == 100)].iloc[0]
        assert math.isclose(f0_scooter["tau_cpa_s"], 2.0, rel_tol=1e-6)
        assert math.isclose(f0_scooter["d_cpa_m"], 0.50, rel_tol=1e-6)
        assert bool(f0_scooter["is_encroachment"]) is True
        assert f0_scooter["wheeler_type"] == "e_scooter"

        # Diverging cyclist at frame 0: dr = (+4.0, 1.20), dv = (+3.0, 0.0) -> tau_cpa = 0
        f0_cyclist = cpa[(cpa["frame"] == 0) & (cpa["wheeler_id"] == 200)].iloc[0]
        assert math.isclose(f0_cyclist["tau_cpa_s"], 0.0, abs_tol=1e-9)
        assert math.isclose(f0_cyclist["d_cpa_m"], math.hypot(4.0, 1.20), rel_tol=1e-6)
        assert bool(f0_cyclist["is_encroachment"]) is False

    def test_cpa_validation_errors(self):
        ped_traj = _make_linear_traj([(1, 0.0, 0.0, 1.0, 0.0)], n_frames=3)
        with pytest.raises(PedPyTypeError, match="Expected 'ped_traj' to be a TrajectoryData"):
            compute_shared_space_cpa(ped_traj="bad", wheeler_df=pd.DataFrame())  # type: ignore[arg-type]

        with pytest.raises(PedPyValueError, match="Missing required columns"):
            compute_shared_space_cpa(ped_traj=ped_traj, wheeler_df=pd.DataFrame({"id": [1]}))


class TestPoseDistractionCOCO17:
    """Tests for MS-COCO 17 smartphone distraction posture classifier."""

    @staticmethod
    def _make_coco17_skeleton(phone_viewing: bool = False) -> np.ndarray:
        # Image pixel coordinates (+y downward): hips at y=300, shoulders at y=200
        kp = np.zeros((17, 2), dtype=float)
        kp[11] = [90.0, 300.0]  # left_hip
        kp[12] = [110.0, 300.0]  # right_hip
        kp[5] = [85.0, 200.0]  # left_shoulder
        kp[6] = [115.0, 200.0]  # right_shoulder
        kp[3] = [92.0, 162.0]  # left_ear
        kp[4] = [108.0, 162.0]  # right_ear
        kp[13] = [90.0, 380.0]
        kp[14] = [110.0, 380.0]
        kp[15] = [90.0, 460.0]
        kp[16] = [110.0, 460.0]

        if not phone_viewing:
            # Upright head looking forward; relaxed straight arms along sides
            kp[0] = [100.0, 160.0]  # nose level with ears
            kp[1] = [96.0, 156.0]
            kp[2] = [104.0, 156.0]
            kp[7] = [82.0, 248.0]  # left_elbow
            kp[9] = [80.0, 295.0]  # left_wrist near hip
            kp[8] = [118.0, 248.0]  # right_elbow
            kp[10] = [120.0, 295.0]  # right_wrist near hip
        else:
            # Head pitched down toward chest; right elbow flexed holding phone at chest
            kp[0] = [100.0, 186.0]  # nose sagged well below ears toward chest
            kp[1] = [96.0, 180.0]
            kp[2] = [104.0, 180.0]
            kp[7] = [82.0, 248.0]
            kp[9] = [80.0, 295.0]
            kp[8] = [122.0, 248.0]  # right_elbow
            kp[10] = [102.0, 222.0]  # right_wrist raised in front of chest (~75 deg bend)
        return kp

    def test_single_and_batch_coco17_classification_and_attention_enrichment(self):
        from pedpy import classify_pose_distraction_coco17

        kp_att = self._make_coco17_skeleton(phone_viewing=False)
        kp_ph = self._make_coco17_skeleton(phone_viewing=True)

        res_att = classify_pose_distraction_coco17(kp_att)
        res_ph = classify_pose_distraction_coco17(kp_ph)

        assert isinstance(res_att, dict) and isinstance(res_ph, dict)
        assert res_att["distracted"] is False
        assert res_att["posture_class"] == "attentive"
        assert res_att["gaze_state"] == 1

        assert res_ph["distracted"] is True
        assert res_ph["posture_class"] == "phone_viewing"
        assert res_ph["head_pitch_deg"] > 25.0
        assert res_ph["elbow_angle_deg"] < 110.0
        assert res_ph["gaze_state"] == 0

        # Batch classification directly enriches TrajectoryData attention_df
        batch_df = classify_pose_distraction_coco17(np.stack([kp_att, kp_ph], axis=0))
        assert isinstance(batch_df, pd.DataFrame)
        assert list(batch_df["gaze_state"]) == [1, 0]

        traj = _make_linear_traj([(1, 0.0, 0.0, 1.3, 0.0), (2, 3.0, 0.0, 1.3, 0.0)], n_frames=4, fps=10.0)
        batch_df["id"] = [1, 2]
        ps_df = compute_anisotropic_personal_space(traj_data=traj, attention_df=batch_df)
        a_ped1 = ps_df.loc[ps_df["id"] == 1, "a_m"].iloc[0]
        a_ped2 = ps_df.loc[ps_df["id"] == 2, "a_m"].iloc[0]
        assert a_ped2 > a_ped1

        # Low-confidence keypoints safely abstain
        low_scores = np.full(17, 0.10)
        res_low = classify_pose_distraction_coco17(kp_ph, scores_17=low_scores)
        assert res_low["posture_class"] == "low_confidence"
        assert res_low["distracted"] is False

        with pytest.raises(PedPyValueError):
            classify_pose_distraction_coco17(np.zeros((10, 2)))
