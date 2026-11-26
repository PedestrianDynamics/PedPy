"""Module containing anisotropic personal space, social dyad, and shared-space CPA methods."""

import math
from typing import Final, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from pedpy.column_identifier import FRAME_COL, ID_COL, SPEED_COL, V_X_COL, V_Y_COL, X_COL, Y_COL
from pedpy.data.trajectory_data import TrajectoryData
from pedpy.errors import PedPyTypeError, PedPyValueError
from pedpy.methods.method_utils import _check_trajectory_data

_SPEED_EPS: Final = 1e-6
_DIST_EPS: Final = 1e-9
_REL_VEL_SQ_EPS: Final = 1e-12
_MOVING_SPEED_MIN: Final = 0.05
_CODIR_COS_MIN: Final = 0.5
_ABREAST_ANGLE_DEG: Final = 45.0
_STRAIGHT_ELBOW_DEG: Final = 180.0
_NECK_REF_PROJ: Final = 0.36
_NECK_PITCH_SCALE: Final = 185.0
_MAX_NECK_PITCH_DEG: Final = 65.0
_EAR_PITCH_WEIGHT: Final = 0.65
_NECK_PITCH_WEIGHT: Final = 0.35
_CHEST_MAX_ELBOW_DEG: Final = 120.0
_CHEST_MIN_WRIST_H: Final = 0.35
_CHEST_MAX_WRIST_H: Final = 1.05
_CHEST_MAX_WRIST_LAT: Final = 0.70
_EAR_MAX_ELBOW_DEG: Final = 95.0
_EAR_MIN_WRIST_H: Final = 1.02
_EAR_MAX_WRIST_LAT: Final = 0.65
_PITCH_SCORE_OFFSET_DEG: Final = 18.0
_PITCH_SCORE_SPAN_DEG: Final = 22.0
_FLEX_SCORE_REF_DEG: Final = 135.0
_FLEX_SCORE_SPAN_DEG: Final = 55.0
_PHONE_VIEW_MIN_PITCH_DEG: Final = 24.0
_DOWNWARD_GAZE_MIN_PITCH_DEG: Final = 35.0
_ATTENTIVE_SCORE_CAP: Final = 0.45


def _compute_segment_velocities(
    f_seg: npt.NDArray[np.float64],
    x_seg: npt.NDArray[np.float64],
    y_seg: npt.NDArray[np.float64],
    frame_rate: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute central/boundary finite-difference velocities for one agent."""
    n = len(f_seg)
    vx_seg = np.zeros(n, dtype=float)
    vy_seg = np.zeros(n, dtype=float)
    if n <= 1:
        return vx_seg, vy_seg

    dt0 = (f_seg[1] - f_seg[0]) / frame_rate
    if dt0 > 0.0:
        vx_seg[0] = (x_seg[1] - x_seg[0]) / dt0
        vy_seg[0] = (y_seg[1] - y_seg[0]) / dt0

    dt_end = (f_seg[-1] - f_seg[-2]) / frame_rate
    if dt_end > 0.0:
        vx_seg[-1] = (x_seg[-1] - x_seg[-2]) / dt_end
        vy_seg[-1] = (y_seg[-1] - y_seg[-2]) / dt_end

    if n > 2:
        dt_mid = (f_seg[2:] - f_seg[:-2]) / frame_rate
        valid = dt_mid > 0.0
        vx_mid = np.zeros(n - 2, dtype=float)
        vy_mid = np.zeros(n - 2, dtype=float)
        vx_mid[valid] = (x_seg[2:][valid] - x_seg[:-2][valid]) / dt_mid[valid]
        vy_mid[valid] = (y_seg[2:][valid] - y_seg[:-2][valid]) / dt_mid[valid]
        vx_seg[1:-1] = vx_mid
        vy_seg[1:-1] = vy_mid

    return vx_seg, vy_seg


def _estimate_agent_kinematics(
    df: pd.DataFrame,
    frame_rate: float,
) -> pd.DataFrame:
    """Compute per-agent velocity, speed, and heading using finite differences."""
    if frame_rate <= 0.0:
        raise PedPyValueError(f"Expected positive frame_rate, got {frame_rate}.")

    if df.empty:
        out = df.copy()
        for col in (V_X_COL, V_Y_COL, SPEED_COL, "heading_rad"):
            out[col] = pd.Series(dtype=float)
        return out

    work = df[[ID_COL, FRAME_COL, X_COL, Y_COL]].sort_values([ID_COL, FRAME_COL]).copy()
    if V_X_COL in df.columns and V_Y_COL in df.columns:
        vel_sorted = df.sort_values([ID_COL, FRAME_COL])
        vx_all = vel_sorted[V_X_COL].to_numpy(dtype=float)
        vy_all = vel_sorted[V_Y_COL].to_numpy(dtype=float)
    else:
        vx_all = np.zeros(len(work), dtype=float)
        vy_all = np.zeros(len(work), dtype=float)
        ids = work[ID_COL].to_numpy()
        frames = work[FRAME_COL].to_numpy(dtype=float)
        xs = work[X_COL].to_numpy(dtype=float)
        ys = work[Y_COL].to_numpy(dtype=float)

        change_idx = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], change_idx))
        ends = np.concatenate((change_idx, [len(work)]))
        for s, e in zip(starts, ends, strict=True):
            vx_seg, vy_seg = _compute_segment_velocities(frames[s:e], xs[s:e], ys[s:e], frame_rate)
            vx_all[s:e] = vx_seg
            vy_all[s:e] = vy_seg

    speed_all = np.hypot(vx_all, vy_all)
    work[V_X_COL] = vx_all
    work[V_Y_COL] = vy_all
    work[SPEED_COL] = speed_all
    work["heading_rad"] = np.where(speed_all > _SPEED_EPS, np.arctan2(vy_all, vx_all), 0.0)
    return work.reset_index(drop=True)


def _extract_distraction_level(
    kin_df: pd.DataFrame,
    attention_df: Optional[pd.DataFrame],
) -> npt.NDArray[np.float64]:
    """Resolve per-row distraction weight in [0, 1] from optional attention_df."""
    if attention_df is None:
        return np.zeros(len(kin_df), dtype=float)
    if not isinstance(attention_df, pd.DataFrame):
        raise PedPyTypeError(
            f"Expected 'attention_df' to be a pandas DataFrame or None, got {type(attention_df).__name__!r}."
        )
    if ID_COL not in attention_df.columns:
        raise PedPyValueError(f"Expected column {ID_COL!r} in 'attention_df'.")

    join_keys = [ID_COL, FRAME_COL] if FRAME_COL in attention_df.columns else [ID_COL]
    att = attention_df.copy()

    if "distracted" in att.columns:
        att["_distraction_val"] = att["distracted"].astype(float)
    elif "gaze_state" in att.columns:
        att["_distraction_val"] = (att["gaze_state"].astype(float) == 0.0).astype(float)
    elif "attention" in att.columns:
        att["_distraction_val"] = 1.0 - att["attention"].astype(float)
    else:
        raise PedPyValueError("Expected 'attention_df' to contain one of {'distracted', 'gaze_state', 'attention'}.")

    merged = kin_df[join_keys].merge(
        att[[*join_keys, "_distraction_val"]].drop_duplicates(subset=join_keys),
        on=join_keys,
        how="left",
    )
    vals = merged["_distraction_val"].fillna(0.0).to_numpy(dtype=float)
    return np.clip(vals, 0.0, 1.0)


def compute_anisotropic_personal_space(
    *,
    traj_data: TrajectoryData,
    a0: float = 0.42,
    tau_stride: float = 0.36,
    delta_a_distracted: float = 0.22,
    b0: float = 0.28,
    delta_b_distracted: float = 0.16,
    attention_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    r"""Compute velocity- and attention-modulated elliptical personal space metrics.

    Each pedestrian :math:`i` at frame :math:`t` is modelled with a heading-aligned
    anisotropic personal space ellipse with longitudinal semi-axis :math:`a_i(t)`
    and lateral semi-axis :math:`b_i(t)`:

    .. math::

        a_i(t) = a_0 + \tau_{\mathrm{stride}} v_i(t) + \Delta a_{\mathrm{distracted}} u_i(t),
        \qquad
        b_i(t) = b_0 + \Delta b_{\mathrm{distracted}} u_i(t),

    where :math:`v_i(t)` is walking speed and :math:`u_i(t) \in [0, 1]` is the
    cognitive/visual distraction indicator. The effective normalized elliptical
    distance to neighbor :math:`j` is

    .. math::

        d_{\mathrm{eff}}(i, j) = \sqrt{
            \left(\frac{\Delta x_{\parallel, ij}}{a_i}\right)^2
            + \left(\frac{\Delta y_{\perp, ij}}{b_i}\right)^2
        },

    where :math:`d_{\mathrm{eff}} < 1` indicates intrusion into pedestrian :math:`i`'s
    personal space ellipse.

    Args:
        traj_data: Pedestrian trajectory data (:class:`~pedpy.data.trajectory_data.TrajectoryData`).
        a0: Baseline longitudinal semi-axis at standstill in meters.
        tau_stride: Longitudinal stride time headway scaling in seconds.
        delta_a_distracted: Additional longitudinal buffer under distraction in meters.
        b0: Baseline lateral sway semi-axis in meters.
        delta_b_distracted: Additional lateral sway buffer under distraction in meters.
        attention_df: Optional DataFrame keyed by ``id`` (and optionally ``frame``)
            with column ``distracted`` (in ``[0, 1]``), ``gaze_state`` (``0`` = phone,
            ``1`` = road), or ``attention`` (in ``[0, 1]``).

    Returns:
        DataFrame with columns ``id``, ``frame``, ``speed``, ``heading_rad``,
        ``a_m``, ``b_m``, ``a_b_ratio``, ``ellipse_area_m2``, and ``min_d_eff``.
    """
    _check_trajectory_data(traj_data, param_name="traj_data")
    if a0 <= 0.0 or b0 <= 0.0:
        raise PedPyValueError("Baseline semi-axes 'a0' and 'b0' must be strictly positive.")
    if tau_stride < 0.0 or delta_a_distracted < 0.0 or delta_b_distracted < 0.0:
        raise PedPyValueError("'tau_stride', 'delta_a_distracted', and 'delta_b_distracted' must be non-negative.")

    out_cols = [
        ID_COL,
        FRAME_COL,
        SPEED_COL,
        "heading_rad",
        "a_m",
        "b_m",
        "a_b_ratio",
        "ellipse_area_m2",
        "min_d_eff",
    ]
    kin = _estimate_agent_kinematics(traj_data.data, traj_data.frame_rate)
    if kin.empty:
        return pd.DataFrame(columns=out_cols)

    distraction = _extract_distraction_level(kin, attention_df)
    speeds = kin[SPEED_COL].to_numpy(dtype=float)

    a_m = a0 + tau_stride * speeds + delta_a_distracted * distraction
    b_m = b0 + delta_b_distracted * distraction
    kin["a_m"] = a_m
    kin["b_m"] = b_m
    kin["a_b_ratio"] = a_m / b_m
    kin["ellipse_area_m2"] = math.pi * a_m * b_m

    min_d_eff = np.full(len(kin), np.inf, dtype=float)
    for _, group in kin.groupby(FRAME_COL, sort=False):
        idx = group.index.to_numpy()
        if len(idx) < 2:
            continue
        gx = group[X_COL].to_numpy(dtype=float)
        gy = group[Y_COL].to_numpy(dtype=float)
        ga = group["a_m"].to_numpy(dtype=float)
        gb = group["b_m"].to_numpy(dtype=float)
        gh = group["heading_rad"].to_numpy(dtype=float)

        dx = gx[np.newaxis, :] - gx[:, np.newaxis]
        dy = gy[np.newaxis, :] - gy[:, np.newaxis]
        cos_h = np.cos(gh)[:, np.newaxis]
        sin_h = np.sin(gh)[:, np.newaxis]

        d_par = dx * cos_h + dy * sin_h
        d_perp = -dx * sin_h + dy * cos_h
        d_eff_mat = np.hypot(d_par / ga[:, np.newaxis], d_perp / gb[:, np.newaxis])
        np.fill_diagonal(d_eff_mat, np.inf)
        min_d_eff[idx] = np.min(d_eff_mat, axis=1)

    kin["min_d_eff"] = min_d_eff
    return kin[out_cols].sort_values([FRAME_COL, ID_COL]).reset_index(drop=True)


def _summarize_cohesive_dyads(
    cohesive_pairs: pd.DataFrame,
    min_shared_frames: int,
    out_cols: list[str],
) -> pd.DataFrame:
    """Aggregate frame-level cohesive pairs into persistent dyad records."""
    records = []
    for (id1, id2), grp in cohesive_pairs.groupby([f"{ID_COL}_1", f"{ID_COL}_2"], sort=True):
        n_frames = len(grp)
        if n_frames < min_shared_frames:
            continue
        mean_angle = float(grp["phi_deg"].mean())
        records.append(
            {
                "ped_id_1": id1,
                "ped_id_2": id2,
                "shared_frames": n_frames,
                "mean_distance_m": float(grp["dist"].mean()),
                "mean_speed_diff_mps": float(grp["speed_diff"].mean()),
                "formation_angle_deg": mean_angle,
                "formation": "abreast" if mean_angle >= _ABREAST_ANGLE_DEG else "single_file",
            }
        )
    if not records:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(records, columns=out_cols)


def detect_social_dyads(
    *,
    traj_data: TrajectoryData,
    max_distance: float = 1.25,
    max_speed_diff: float = 0.35,
    min_shared_frames: int = 5,
) -> pd.DataFrame:
    r"""Detect persistent walking dyads and classify abreast vs. single-file formation.

    Co-present pedestrian pairs :math:`(i, j)` with :math:`i < j` are evaluated across
    shared frames for spatial proximity (:math:`d_{ij} \le d_{\max}`), speed matching
    (:math:`|v_i - v_j| \le \Delta v_{\max}`), and co-directional heading alignment.
    Pairs meeting these criteria for at least ``min_shared_frames`` frames are
    classified by their mean acute formation angle :math:`\bar{\phi}_{ij} \in [0^\circ, 90^\circ]`
    between the inter-member separation vector and group velocity vector:

    * ``"abreast"`` when :math:`\bar{\phi}_{ij} \ge 45^\circ` (side-by-side walking),
    * ``"single_file"`` when :math:`\bar{\phi}_{ij} < 45^\circ` (leader-follower walking).

    Args:
        traj_data: Pedestrian trajectory data (:class:`~pedpy.data.trajectory_data.TrajectoryData`).
        max_distance: Maximum Euclidean separation in meters for dyad cohesion.
        max_speed_diff: Maximum absolute speed difference in m/s between dyad members.
        min_shared_frames: Minimum number of cohesive co-walking frames required.

    Returns:
        DataFrame with columns ``ped_id_1``, ``ped_id_2``, ``shared_frames``,
        ``mean_distance_m``, ``mean_speed_diff_mps``, ``formation_angle_deg``,
        and ``formation``.
    """
    _check_trajectory_data(traj_data, param_name="traj_data")
    if max_distance <= 0.0:
        raise PedPyValueError("'max_distance' must be strictly positive.")
    if max_speed_diff < 0.0:
        raise PedPyValueError("'max_speed_diff' must be non-negative.")
    if min_shared_frames < 1:
        raise PedPyValueError("'min_shared_frames' must be at least 1.")

    out_cols = [
        "ped_id_1",
        "ped_id_2",
        "shared_frames",
        "mean_distance_m",
        "mean_speed_diff_mps",
        "formation_angle_deg",
        "formation",
    ]
    kin = _estimate_agent_kinematics(traj_data.data, traj_data.frame_rate)
    if kin.empty or kin[ID_COL].nunique() < 2:
        return pd.DataFrame(columns=out_cols)

    pairs = kin.merge(kin, on=FRAME_COL, suffixes=("_1", "_2"))
    pairs = pairs[pairs[f"{ID_COL}_1"] < pairs[f"{ID_COL}_2"]].copy()
    if pairs.empty:
        return pd.DataFrame(columns=out_cols)

    dx = pairs[f"{X_COL}_2"].to_numpy(dtype=float) - pairs[f"{X_COL}_1"].to_numpy(dtype=float)
    dy = pairs[f"{Y_COL}_2"].to_numpy(dtype=float) - pairs[f"{Y_COL}_1"].to_numpy(dtype=float)
    dist = np.hypot(dx, dy)

    v1 = pairs[f"{SPEED_COL}_1"].to_numpy(dtype=float)
    v2 = pairs[f"{SPEED_COL}_2"].to_numpy(dtype=float)
    speed_diff = np.abs(v1 - v2)

    vx1 = pairs[f"{V_X_COL}_1"].to_numpy(dtype=float)
    vy1 = pairs[f"{V_Y_COL}_1"].to_numpy(dtype=float)
    vx2 = pairs[f"{V_X_COL}_2"].to_numpy(dtype=float)
    vy2 = pairs[f"{V_Y_COL}_2"].to_numpy(dtype=float)

    both_moving = (v1 > _MOVING_SPEED_MIN) & (v2 > _MOVING_SPEED_MIN)
    cos_heading_diff = np.ones(len(pairs), dtype=float)
    cos_heading_diff[both_moving] = (vx1[both_moving] * vx2[both_moving] + vy1[both_moving] * vy2[both_moving]) / (
        v1[both_moving] * v2[both_moving]
    )

    cohesive = (dist <= max_distance) & (speed_diff <= max_speed_diff) & (cos_heading_diff >= _CODIR_COS_MIN)
    cohesive_pairs = pairs.loc[cohesive, [f"{ID_COL}_1", f"{ID_COL}_2"]].copy()
    if cohesive_pairs.empty:
        return pd.DataFrame(columns=out_cols)

    mean_vx = 0.5 * (vx1[cohesive] + vx2[cohesive])
    mean_vy = 0.5 * (vy1[cohesive] + vy2[cohesive])
    mean_v_norm = np.hypot(mean_vx, mean_vy)
    dist_c = dist[cohesive]

    valid_geom = (dist_c > _DIST_EPS) & (mean_v_norm > _SPEED_EPS)
    abs_cos_phi = np.zeros(len(cohesive_pairs), dtype=float)
    abs_cos_phi[valid_geom] = np.clip(
        np.abs(dx[cohesive][valid_geom] * mean_vx[valid_geom] + dy[cohesive][valid_geom] * mean_vy[valid_geom])
        / (dist_c[valid_geom] * mean_v_norm[valid_geom]),
        0.0,
        1.0,
    )

    cohesive_pairs["dist"] = dist_c
    cohesive_pairs["speed_diff"] = speed_diff[cohesive]
    cohesive_pairs["phi_deg"] = np.degrees(np.arccos(abs_cos_phi))
    return _summarize_cohesive_dyads(cohesive_pairs, min_shared_frames, out_cols)


def compute_shared_space_cpa(
    *,
    ped_traj: TrajectoryData,
    wheeler_df: pd.DataFrame,
    d_min: float = 0.75,
) -> pd.DataFrame:
    r"""Compute Closest Point of Approach (CPA) metrics between pedestrians and micro-mobility wheelers.

    For each co-occurring pedestrian :math:`p` and cyclist/e-scooter :math:`w` at frame :math:`f`,
    with relative position :math:`\Delta \mathbf{r} = \mathbf{p}_w - \mathbf{p}_p` and
    relative velocity :math:`\Delta \mathbf{v} = \mathbf{v}_w - \mathbf{v}_p`, the
    future Time-to-Closest-Approach and minimum passing clearance are:

    .. math::

        \tau_{\mathrm{CPA}} = \max\!\left(
            0, -\frac{\Delta \mathbf{r} \cdot \Delta \mathbf{v}}{\|\Delta \mathbf{v}\|_2^2}
        \right),
        \qquad
        d_{\mathrm{CPA}} = \|\Delta \mathbf{r} + \tau_{\mathrm{CPA}} \Delta \mathbf{v}\|_2.

    Args:
        ped_traj: Pedestrian trajectory data (:class:`~pedpy.data.trajectory_data.TrajectoryData`).
        wheeler_df: DataFrame of cyclist/e-scooter states containing ``id``, ``frame``,
            ``x``, ``y`` (and optionally ``v_x``, ``v_y``, and ``wheeler_type`` / ``type``).
        d_min: Minimum safe passing clearance threshold in meters (default ``0.75`` m).

    Returns:
        DataFrame with columns ``frame``, ``ped_id``, ``wheeler_id``,
        ``current_distance_m``, ``closing_speed_mps``, ``tau_cpa_s``,
        ``d_cpa_m``, and ``is_encroachment`` (plus ``wheeler_type`` if present).
    """
    _check_trajectory_data(ped_traj, param_name="ped_traj")
    if not isinstance(wheeler_df, pd.DataFrame):
        raise PedPyTypeError(
            f"Expected 'wheeler_df' to be a pandas DataFrame, got {type(wheeler_df).__name__!r} instead."
        )
    if d_min <= 0.0:
        raise PedPyValueError("'d_min' must be strictly positive.")

    missing = {ID_COL, FRAME_COL, X_COL, Y_COL} - set(wheeler_df.columns)
    if missing:
        raise PedPyValueError(f"Missing required columns in 'wheeler_df': {sorted(missing)}.")

    type_col = (
        "wheeler_type" if "wheeler_type" in wheeler_df.columns else ("type" if "type" in wheeler_df.columns else None)
    )
    base_cols = [
        FRAME_COL,
        "ped_id",
        "wheeler_id",
        "current_distance_m",
        "closing_speed_mps",
        "tau_cpa_s",
        "d_cpa_m",
        "is_encroachment",
    ]
    if type_col is not None:
        base_cols.insert(3, "wheeler_type")

    if ped_traj.data.empty or wheeler_df.empty:
        return pd.DataFrame(columns=base_cols)

    ped_kin = _estimate_agent_kinematics(ped_traj.data, ped_traj.frame_rate)
    whl_kin = _estimate_agent_kinematics(wheeler_df, ped_traj.frame_rate)
    if type_col is not None:
        type_lookup = wheeler_df[[ID_COL, FRAME_COL, type_col]].drop_duplicates(subset=[ID_COL, FRAME_COL])
        whl_kin = whl_kin.merge(type_lookup, on=[ID_COL, FRAME_COL], how="left")

    merged = ped_kin.merge(whl_kin, on=FRAME_COL, suffixes=("_ped", "_whl"))
    if merged.empty:
        return pd.DataFrame(columns=base_cols)

    dr_x = merged[f"{X_COL}_whl"].to_numpy(dtype=float) - merged[f"{X_COL}_ped"].to_numpy(dtype=float)
    dr_y = merged[f"{Y_COL}_whl"].to_numpy(dtype=float) - merged[f"{Y_COL}_ped"].to_numpy(dtype=float)
    dv_x = merged[f"{V_X_COL}_whl"].to_numpy(dtype=float) - merged[f"{V_X_COL}_ped"].to_numpy(dtype=float)
    dv_y = merged[f"{V_Y_COL}_whl"].to_numpy(dtype=float) - merged[f"{V_Y_COL}_ped"].to_numpy(dtype=float)

    curr_dist = np.hypot(dr_x, dr_y)
    dv_sq = dv_x**2 + dv_y**2
    dr_dot_dv = dr_x * dv_x + dr_y * dv_y

    tau_cpa = np.zeros(len(merged), dtype=float)
    moving_rel = dv_sq > _REL_VEL_SQ_EPS
    tau_cpa[moving_rel] = np.maximum(0.0, -dr_dot_dv[moving_rel] / dv_sq[moving_rel])
    d_cpa = np.hypot(dr_x + tau_cpa * dv_x, dr_y + tau_cpa * dv_y)

    closing_speed = np.where(
        curr_dist > _DIST_EPS,
        np.maximum(0.0, -dr_dot_dv / np.maximum(curr_dist, _DIST_EPS)),
        0.0,
    )

    out = pd.DataFrame(
        {
            FRAME_COL: merged[FRAME_COL],
            "ped_id": merged[f"{ID_COL}_ped"],
            "wheeler_id": merged[f"{ID_COL}_whl"],
            "current_distance_m": curr_dist,
            "closing_speed_mps": closing_speed,
            "tau_cpa_s": tau_cpa,
            "d_cpa_m": d_cpa,
            "is_encroachment": d_cpa < d_min,
        }
    )
    if type_col is not None:
        out.insert(3, "wheeler_type", merged[type_col])

    return out.sort_values([FRAME_COL, "ped_id", "wheeler_id"]).reset_index(drop=True)


def _estimate_head_pitch_deg(
    kp: npt.NDArray[np.float64],
    sc: npt.NDArray[np.float64],
    mid_shoulder: npt.NDArray[np.float64],
    trunk_up: npt.NDArray[np.float64],
    torso_len: float,
    min_confidence: float,
) -> float:
    """Estimate torso-normalized downward head pitch in degrees from nose/ears."""
    if sc[0] < min_confidence:
        return 0.0
    neck_proj = float(np.dot(kp[0] - mid_shoulder, trunk_up)) / torso_len
    pitch_neck = float(np.clip((_NECK_REF_PROJ - neck_proj) * _NECK_PITCH_SCALE, 0.0, _MAX_NECK_PITCH_DEG))
    ears_ok = [idx for idx in (3, 4) if sc[idx] >= min_confidence]
    if not ears_ok:
        return pitch_neck

    ear_to_nose = kp[0] - np.mean(kp[ears_ok], axis=0)
    en_dist = float(np.linalg.norm(ear_to_nose))
    if en_dist <= _SPEED_EPS:
        return pitch_neck
    sag = -float(np.dot(ear_to_nose, trunk_up)) / en_dist
    pitch_ear = math.degrees(math.asin(float(np.clip(sag, -1.0, 1.0))))
    return max(0.0, _EAR_PITCH_WEIGHT * max(0.0, pitch_ear) + _NECK_PITCH_WEIGHT * pitch_neck)


def _evaluate_arm_zones(
    kp: npt.NDArray[np.float64],
    sc: npt.NDArray[np.float64],
    mid_hip: npt.NDArray[np.float64],
    trunk_up: npt.NDArray[np.float64],
    torso_len: float,
    min_confidence: float,
) -> Tuple[float, bool, bool]:
    """Compute minimum elbow angle and chest/ear wrist-zone flags."""
    min_elbow_deg = _STRAIGHT_ELBOW_DEG
    hand_at_chest = False
    hand_at_ear = False

    for sh_i, el_i, wr_i in ((5, 7, 9), (6, 8, 10)):
        if sc[sh_i] < min_confidence or sc[el_i] < min_confidence or sc[wr_i] < min_confidence:
            continue
        u_up = kp[sh_i] - kp[el_i]
        u_fore = kp[wr_i] - kp[el_i]
        nu = float(np.linalg.norm(u_up))
        nf = float(np.linalg.norm(u_fore))
        if nu < _SPEED_EPS or nf < _SPEED_EPS:
            continue
        cos_ang = float(np.clip(np.dot(u_up, u_fore) / (nu * nf), -1.0, 1.0))
        ang_deg = math.degrees(math.acos(cos_ang))
        min_elbow_deg = min(min_elbow_deg, ang_deg)

        wr_rel_hip = kp[wr_i] - mid_hip
        wr_h = float(np.dot(wr_rel_hip, trunk_up)) / torso_len
        wr_lat = float(np.linalg.norm(wr_rel_hip - float(np.dot(wr_rel_hip, trunk_up)) * trunk_up)) / torso_len

        if (
            ang_deg <= _CHEST_MAX_ELBOW_DEG
            and (_CHEST_MIN_WRIST_H <= wr_h <= _CHEST_MAX_WRIST_H)
            and (wr_lat <= _CHEST_MAX_WRIST_LAT)
        ):
            hand_at_chest = True
        if ang_deg <= _EAR_MAX_ELBOW_DEG and (wr_h > _EAR_MIN_WRIST_H) and (wr_lat <= _EAR_MAX_WRIST_LAT):
            hand_at_ear = True

    return min_elbow_deg, hand_at_chest, hand_at_ear


def _classify_single_coco17_pose(
    kp: npt.NDArray[np.float64],
    sc: npt.NDArray[np.float64],
    min_confidence: float = 0.30,
) -> dict[str, Union[float, bool, int, str]]:
    """Classify a single (17, 2) MS-COCO skeleton into smartphone distraction posture."""
    low_conf_res: dict[str, Union[float, bool, int, str]] = {
        "distracted": False,
        "distraction_score": 0.0,
        "posture_class": "low_confidence",
        "head_pitch_deg": 0.0,
        "elbow_angle_deg": _STRAIGHT_ELBOW_DEG,
        "gaze_state": 1,
    }
    if np.any(sc[[5, 6, 11, 12]] < min_confidence):
        return low_conf_res

    mid_shoulder = 0.5 * (kp[5] + kp[6])
    mid_hip = 0.5 * (kp[11] + kp[12])
    trunk_vec = mid_shoulder - mid_hip
    torso_len = float(np.linalg.norm(trunk_vec))
    if torso_len < _SPEED_EPS:
        return low_conf_res

    trunk_up = trunk_vec / torso_len
    head_pitch_deg = _estimate_head_pitch_deg(kp, sc, mid_shoulder, trunk_up, torso_len, min_confidence)
    min_elbow_deg, hand_at_chest, hand_at_ear = _evaluate_arm_zones(
        kp, sc, mid_hip, trunk_up, torso_len, min_confidence
    )

    pitch_score = float(np.clip((head_pitch_deg - _PITCH_SCORE_OFFSET_DEG) / _PITCH_SCORE_SPAN_DEG, 0.0, 1.0))
    flex_score = float(np.clip((_FLEX_SCORE_REF_DEG - min_elbow_deg) / _FLEX_SCORE_SPAN_DEG, 0.0, 1.0))

    if hand_at_chest and head_pitch_deg >= _PHONE_VIEW_MIN_PITCH_DEG:
        score = float(np.clip(0.55 * pitch_score + 0.45 * flex_score + 0.25, 0.0, 1.0))
        return {
            "distracted": True,
            "distraction_score": score,
            "posture_class": "phone_viewing",
            "head_pitch_deg": head_pitch_deg,
            "elbow_angle_deg": min_elbow_deg,
            "gaze_state": 0,
        }
    if hand_at_ear:
        score = float(np.clip(0.70 * flex_score + 0.20, 0.5, 1.0))
        return {
            "distracted": True,
            "distraction_score": score,
            "posture_class": "phone_call",
            "head_pitch_deg": head_pitch_deg,
            "elbow_angle_deg": min_elbow_deg,
            "gaze_state": 1,
        }
    if head_pitch_deg >= _DOWNWARD_GAZE_MIN_PITCH_DEG:
        return {
            "distracted": True,
            "distraction_score": pitch_score,
            "posture_class": "downward_gaze",
            "head_pitch_deg": head_pitch_deg,
            "elbow_angle_deg": min_elbow_deg,
            "gaze_state": 0,
        }

    att_score = 0.5 * pitch_score + 0.2 * flex_score
    return {
        "distracted": False,
        "distraction_score": float(min(_ATTENTIVE_SCORE_CAP, att_score)),
        "posture_class": "attentive",
        "head_pitch_deg": head_pitch_deg,
        "elbow_angle_deg": min_elbow_deg,
        "gaze_state": 1,
    }


def classify_pose_distraction_coco17(
    keypoints_17x2: npt.ArrayLike,
    scores_17: Optional[npt.ArrayLike] = None,
    min_confidence: float = 0.30,
) -> Union[dict[str, Union[float, bool, int, str]], pd.DataFrame]:
    """Classify pedestrian smartphone distraction from MS-COCO 17-keypoint 2D skeletons.

    Evaluates torso-normalized downward head pitch (nose-to-ear sag and neck
    projection) combined with upper-limb elbow flexion and wrist-to-chest/head
    zones. Invariant to image ``+y``-down vs. Cartesian ``+y``-up coordinates.

    Args:
        keypoints_17x2: Array of shape ``(17, 2)`` for a single pedestrian or
            ``(N, 17, 2)`` for a batch of detections.
        scores_17: Optional keypoint confidence scores of shape ``(17,)`` or ``(N, 17)``.
            Defaults to ``1.0`` for all keypoints when omitted.
        min_confidence: Minimum landmark confidence threshold (default ``0.30``).

    Returns:
        A dictionary of distraction metrics for a single ``(17, 2)`` input, or a
        :class:`pandas.DataFrame` with one row per skeleton for ``(N, 17, 2)`` batch input
        (directly usable to populate ``attention_df`` for
        :func:`compute_anisotropic_personal_space`).
    """
    kp_arr = np.asarray(keypoints_17x2, dtype=float)
    if kp_arr.ndim == 2:
        if kp_arr.shape != (17, 2):
            raise PedPyValueError(f"Expected single skeleton shape (17, 2), got {kp_arr.shape}.")
        if scores_17 is None:
            sc_arr = np.ones(17, dtype=float)
        else:
            sc_arr = np.asarray(scores_17, dtype=float).reshape(-1)
            if sc_arr.shape != (17,):
                raise PedPyValueError(f"Expected scores_17 shape (17,), got {sc_arr.shape}.")
        return _classify_single_coco17_pose(kp_arr, sc_arr, min_confidence=min_confidence)

    if kp_arr.ndim == 3 and kp_arr.shape[1:] == (17, 2):
        n_batch = kp_arr.shape[0]
        if scores_17 is None:
            sc_batch = np.ones((n_batch, 17), dtype=float)
        else:
            sc_batch = np.asarray(scores_17, dtype=float)
            if sc_batch.shape != (n_batch, 17):
                raise PedPyValueError(f"Expected batch scores_17 shape {(n_batch, 17)}, got {sc_batch.shape}.")
        rows = [
            _classify_single_coco17_pose(kp_arr[i], sc_batch[i], min_confidence=min_confidence) for i in range(n_batch)
        ]
        return pd.DataFrame(rows)

    raise PedPyValueError(f"Expected 'keypoints_17x2' of shape (17, 2) or (N, 17, 2), got {kp_arr.shape}.")
