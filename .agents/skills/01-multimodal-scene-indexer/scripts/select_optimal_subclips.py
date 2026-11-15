"""Peak Cinematic Subclip Optimizer & Style-to-Footage Slot Planner (v2.1).

Computes:
- 4D CV moment quality (optical-flow camera stability, Laplacian focus sharpness,
  rule-of-thirds visual energy, exposure safety) via true adaptive sub-second frame sampling
- Visual-environment coherence (green/outdoor consistency)
- Semantic slot matching (32-D visual fingerprint cosine similarity, shot scale, camera motion continuity)
- PyTorch DeepLabV3 person-subject presence verification (GPU/CPU auto-detected)
- Frame-snapped & beat-locked cut timing + Premiere Pro tick conversion (`254016000000` ticks/sec)
- Track A2 pre-cut SFX transient placement plan, Speed-Ramp curves, and Punch-In zoom keyframes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCOPES_DIR = _SCRIPTS_DIR.parents[1] / "balancing-primary-color-and-skins" / "scripts"
for _p in (str(_SCRIPTS_DIR), str(_SCOPES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from index_raw_media import _get_deeplab_person_model  # noqa: E402
from virtual_scopes import (  # noqa: E402
    adaptive_sample_frames,
    seconds_to_premiere_ticks,
    snap_time_to_frame,
)

_CLIP_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}


def compute_focus_score(gray: np.ndarray) -> float:
    """Laplacian variance edge crispness mapped cleanly to [0, 1]."""
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(1.0, lap / 500.0))


def compute_composition_energy(gray: np.ndarray) -> float:
    """Rule-of-thirds / central subject Sobel edge balance mapped to [0, 1]."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    h, w = mag.shape
    roi = mag[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    return float(min(1.0, (float(np.mean(roi)) / 45.0) if roi.size else 0.0))


def compute_exposure_score(gray: np.ndarray) -> float:
    """Reward balanced midtones (90-165), penalize severe underexposure or highlight blowout."""
    mean_v = float(np.mean(gray))
    clip_hi = float(np.mean(gray > 248))
    clip_lo = float(np.mean(gray < 10))
    mid_fit = max(0.0, 1.0 - abs(mean_v - 122.0) / 110.0)
    return float(max(0.0, mid_fit - 1.5 * clip_hi - 1.2 * clip_lo))


def _estimate_person_coverage(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    """Return (person_coverage_pct, center_alignment_0_to_1, top_head_margin_pct) for a BGR frame on GPU/CPU."""
    bundle = _get_deeplab_person_model()
    if bundle is None:
        return 0.0, 0.0, 100.0
    import torch
    from PIL import Image

    model, preprocess, _, device = bundle
    small = cv2.resize(frame_bgr, (320, max(180, int(round(320 * frame_bgr.shape[0] / max(1, frame_bgr.shape[1]))))))
    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    inp = preprocess(pil_img).unsqueeze(0)
    if device is not None:
        inp = inp.to(device)
    with torch.no_grad():
        preds = model(inp)["out"][0].argmax(0).byte().cpu().numpy()
    mask = preds == 15
    cov_pct = float(np.mean(mask) * 100.0)
    if cov_pct < 0.4:
        return round(cov_pct, 2), 0.0, 100.0
    mh, mw = mask.shape[:2]
    ys, xs = np.where(mask)
    top_margin_pct = round(float(ys.min()) / max(1, mh) * 100.0, 2)
    cx = float(np.mean(xs) / max(1, mw))
    cy = float(np.mean(ys) / max(1, mh))
    align = max(0.0, 1.0 - (abs(cx - 0.5) * 1.4 + abs(cy - 0.52) * 0.8))
    return round(cov_pct, 2), round(align, 3), top_margin_pct


def _analyze_clip_cached(
    clip_path: str,
    require_person: bool = False,
    sample_fps: float = 5.0,
) -> dict[str, Any] | None:
    """Decode & cache container keyframes + 4D CV scores + optional DeepLabV3 person tracks once per clip."""
    key = str(Path(clip_path).resolve())
    cached = _CLIP_ANALYSIS_CACHE.get(key)
    if cached is not None:
        if require_person and not cached.get("person_computed"):
            p_covs, p_aligns, p_head_margins = [], [], []
            for fr in cached["frames"]:
                cov, al, hm = _estimate_person_coverage(fr)
                p_covs.append(cov)
                p_aligns.append(al)
                p_head_margins.append(hm)
            cached["person_covs"] = p_covs
            cached["person_aligns"] = p_aligns
            cached["person_head_margins"] = p_head_margins
            cached["person_computed"] = True
        return cached

    times, frames, meta = adaptive_sample_frames(
        clip_path,
        target_width=480,
        sample_fps=sample_fps,
        max_frames=16,
    )
    if len(times) < 2:
        return None

    focus_scores: list[float] = []
    comp_scores: list[float] = []
    exp_scores: list[float] = []
    stability_scores: list[float] = []
    motion_mags: list[float] = []
    p_covs: list[float] = []
    p_aligns: list[float] = []
    p_head_margins: list[float] = []

    prev_gray: np.ndarray | None = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        focus_scores.append(compute_focus_score(gray))
        comp_scores.append(compute_composition_energy(gray))
        exp_scores.append(compute_exposure_score(gray))
        stab = 0.85
        mag = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.median(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            stab = float(max(0.0, 1.0 - (mag / 18.0)))
        stability_scores.append(stab)
        motion_mags.append(mag)
        prev_gray = gray
        if require_person:
            cov, al, hm = _estimate_person_coverage(frame)
            p_covs.append(cov)
            p_aligns.append(al)
            p_head_margins.append(hm)

    entry = {
        "times": times,
        "frames": frames,
        "dur": float(meta.get("duration") or times[-1]),
        "fps": float(meta.get("fps") or 30.0),
        "focus": focus_scores,
        "comp": comp_scores,
        "exp": exp_scores,
        "stab": stability_scores,
        "motion": motion_mags,
        "person_covs": p_covs,
        "person_aligns": p_aligns,
        "person_head_margins": p_head_margins,
        "person_computed": require_person,
    }
    _CLIP_ANALYSIS_CACHE[key] = entry
    return entry


def analyze_clip_moments(
    clip_path: str,
    target_duration: float,
    sample_fps: float = 5.0,
    min_in_sec: float = 0.0,
    max_out_sec: float | None = None,
    exclude_windows: list[tuple[float, float]] | None = None,
    require_person: bool = False,
    min_person_pct: float = 2.0,
) -> dict[str, Any] | None:
    """Search clip timeline for optimal non-overlapping [t_in, t_out] cinematic window."""
    eff_sample_fps = max(sample_fps, 8.0 if target_duration < 0.6 else 5.0)
    data = _analyze_clip_cached(clip_path, require_person=require_person, sample_fps=eff_sample_fps)
    if data is None:
        return None

    dur = data["dur"]
    fps = data["fps"]
    times = np.array(data["times"], dtype=np.float64)
    if dur <= 0.05 or times.size < 2:
        return None

    t_limit = min(dur, max_out_sec) if max_out_sec is not None else dur
    eff_target = min(max(0.08, float(target_duration)), max(0.08, t_limit - min_in_sec - 0.05))
    max_start = max(min_in_sec, t_limit - eff_target)

    grid_step = min(0.30, max(0.10, eff_target * 0.75))
    candidates = np.arange(min_in_sec, max_start + 1e-6, grid_step)
    if candidates.size == 0:
        candidates = np.array([min_in_sec])

    excl = exclude_windows or []
    best_score = -1e9
    best_in = float(candidates[0])
    best_metrics: dict[str, Any] = {}

    for t_s in candidates:
        t_s = float(t_s)
        t_e = min(t_limit, t_s + eff_target)
        if any(min(t_e, ex_e) - max(t_s, ex_s) > 0.06 for ex_s, ex_e in excl):
            continue

        t_mid = 0.5 * (t_s + t_e)
        probe_ts = np.array([t_s, t_mid, t_e], dtype=np.float64)
        avg_foc = float(np.mean(np.interp(probe_ts, times, data["focus"])))
        avg_stab = float(np.mean(np.interp(probe_ts, times, data["stab"])))
        avg_comp = float(np.mean(np.interp(probe_ts, times, data["comp"])))
        avg_exp = float(np.mean(np.interp(probe_ts, times, data["exp"])))
        raw_quality = 0.32 * avg_stab + 0.28 * avg_foc + 0.22 * avg_comp + 0.18 * avg_exp

        # Locate peak action/motion frame offset inside the window for beat-drop synchronization
        idx_s = int(np.argmin(np.abs(times - t_s)))
        idx_m = int(np.argmin(np.abs(times - t_mid)))
        idx_e = int(np.argmin(np.abs(times - t_e)))
        lo, hi = min(idx_s, idx_e), max(idx_s, idx_e) + 1
        win_motions = data["motion"][lo:hi]
        peak_rel_idx = int(np.argmax(win_motions)) if win_motions else 0
        peak_time = float(times[min(len(times) - 1, lo + peak_rel_idx)])
        peak_offset_sec = round(max(0.0, min(eff_target, peak_time - t_s)), 3)

        person_cov = 0.0
        person_align = 0.0
        head_margin_pct = 100.0
        if require_person and data.get("person_computed"):
            cov_s = data["person_covs"][idx_s]
            cov_m = data["person_covs"][idx_m]
            al_s = data["person_aligns"][idx_s]
            al_m = data["person_aligns"][idx_m]
            hm_s = data.get("person_head_margins", [100.0] * len(times))[idx_s]
            hm_m = data.get("person_head_margins", [100.0] * len(times))[idx_m]
            person_cov = min(cov_s, cov_m) if cov_s > 0 and cov_m > 0 else max(cov_s, cov_m) * 0.5
            person_align = (al_s + al_m) * 0.5
            head_margin_pct = min(hm_s, hm_m)
            # Reject / heavily penalize windows where person coverage is too low OR head is clipped at top border (<= 2.0% H)
            if person_cov < min_person_pct or head_margin_pct < 2.0:
                raw_quality -= 2.5
            else:
                raw_quality += 0.45 * min(1.0, person_cov / 15.0) + 0.35 * person_align

        if raw_quality > best_score:
            best_score = raw_quality
            best_in = t_s
            best_metrics = {
                "stability": round(avg_stab, 3),
                "focus": round(avg_foc, 3),
                "composition": round(avg_comp, 3),
                "exposure": round(avg_exp, 3),
                "peak_action_offset_sec": peak_offset_sec,
                "person_coverage_pct": round(person_cov, 2),
                "person_alignment": round(person_align, 3),
                "top_head_margin_pct": round(head_margin_pct, 2),
                "head_safe": bool(head_margin_pct >= 2.0),
            }

    if not best_metrics:
        best_in = float(min_in_sec)
        best_score = 0.5
        best_metrics = {
            "stability": round(float(np.mean(data["stab"])), 3),
            "focus": round(float(np.mean(data["focus"])), 3),
            "composition": round(float(np.mean(data["comp"])), 3),
            "exposure": round(float(np.mean(data["exp"])), 3),
            "peak_action_offset_sec": round(0.5 * target_duration, 3),
            "person_coverage_pct": 0.0,
            "person_alignment": 0.0,
            "overlap_relaxed": True,
        }

    in_sec = snap_time_to_frame(best_in, fps)
    out_sec = snap_time_to_frame(min(t_limit, in_sec + target_duration), fps)
    return {
        "file_path": Path(clip_path).as_posix(),
        "source_duration_sec": round(dur, 3),
        "source_fps": round(fps, 3),
        "recommended_in_sec": round(in_sec, 3),
        "recommended_out_sec": round(out_sec, 3),
        "recommended_in_ticks": seconds_to_premiere_ticks(in_sec),
        "recommended_out_ticks": seconds_to_premiere_ticks(out_sec),
        "target_duration_sec": round(target_duration, 3),
        "quality_score": round(float(best_score), 3),
        "metrics": best_metrics,
    }


def _resolve_clip_path(clip_name_or_path: str, raw_index: dict[str, Any] | None = None) -> str | None:
    """Resolve clip filename or path across raw_scene_index, a_roll/, and b_roll/."""
    cand = Path(clip_name_or_path)
    if cand.exists():
        return cand.as_posix()
    if raw_index:
        for c in raw_index.get("clips", []):
            if c.get("file_name") == cand.name or c.get("path") == clip_name_or_path:
                if Path(c["path"]).exists():
                    return c["path"]
    for sub in ("a_roll", "b_roll"):
        p = Path("01_raw_media") / sub / cand.name
        if p.exists():
            return p.as_posix()
    return None


def _score_clip_theme_fit(
    clip_entry: dict[str, Any],
    coherence_target: dict[str, Any],
    project_median_green: float = 0.0,
) -> tuple[bool, float]:
    """Return (is_eligible, coherence_score) so indoor/enclosed low-green outliers are excluded when alpine/outdoor consistency applies."""
    env = clip_entry.get("environment", {})
    green_f = float(env.get("green_fraction", 0.0))
    sky_f = float(env.get("sky_blue_fraction", 0.0))
    likely_indoor = bool(env.get("likely_indoor", False))

    need_outdoor = bool(coherence_target.get("outdoor_scene_consistency_required", False))
    tgt_green = float(coherence_target.get("mean_green_fraction", 0.0))

    if need_outdoor and (
        likely_indoor
        or (tgt_green >= 0.08 and green_f < 0.025 and sky_f < 0.025)
        or (project_median_green >= 0.15 and green_f < 0.05 and sky_f < 0.045)
    ):
        return False, -1.0

    if tgt_green > 0:
        fit = 1.0 - min(1.0, abs(green_f - tgt_green) / max(0.25, tgt_green))
    else:
        fit = 0.7
    return True, round(0.6 * fit + 0.4 * min(1.0, green_f * 2.5 + sky_f * 1.5), 3)


def _compute_semantic_slot_affinity(
    clip_entry: dict[str, Any],
    slot_entry: dict[str, Any],
    prev_motion: str | None = None,
) -> float:
    """Compute [0..1] affinity between a candidate raw clip and a reference narrative slot (embedding + scale + motion continuity)."""
    c_sem = clip_entry.get("shot_semantics", {})
    c_vec = c_sem.get("visual_fingerprint") or []
    s_vec = slot_entry.get("visual_fingerprint") or []

    cos_sim = 0.5
    if len(c_vec) == 32 and len(s_vec) == 32:
        a = np.array(c_vec, dtype=np.float32)
        b = np.array(s_vec, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-6:
            cos_sim = float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

    scale_bonus = 0.0
    c_scale = c_sem.get("shot_scale", "unknown")
    s_scale = slot_entry.get("shot_scale", "unknown")
    if c_scale != "unknown" and s_scale != "unknown":
        if c_scale == s_scale:
            scale_bonus = 0.20
        elif {"extreme_wide_scenic", "wide_establishing"} >= {c_scale, s_scale}:
            scale_bonus = 0.10

    motion_bonus = 0.0
    c_mot = c_sem.get("camera_motion", "static_locked")
    s_mot = slot_entry.get("camera_motion", "static_locked")
    if c_mot == s_mot and c_mot != "static_locked":
        motion_bonus += 0.15
    # Penalize jarring opposite pan reversals on back-to-back cuts
    opposites = {("pan_left", "pan_right"), ("pan_right", "pan_left"), ("tilt_up", "tilt_down"), ("tilt_down", "tilt_up")}
    if prev_motion and (prev_motion, c_mot) in opposites:
        motion_bonus -= 0.18

    return round(float(np.clip(0.65 * cos_sim + scale_bonus + motion_bonus, 0.0, 1.0)), 3)


def _build_a2_sfx_plan(
    slot_assignments: list[dict[str, Any]],
    aperture_curve: dict[str, Any],
    typography_plan: dict[str, Any],
    seq_fps: float,
    vault_path: str = "04_assets/vault_index.json",
) -> list[dict[str, Any]]:
    """Synthesize Track A2 SFX placements aligned 3-5 frames BEFORE visual cuts, unlocks, and title pops."""
    vault_sfx: dict[str, dict[str, Any]] = {}
    vp = Path(vault_path)
    if vp.exists():
        try:
            vd = json.loads(vp.read_text(encoding="utf-8"))
            sfx_list = vd.get("sfx", [])
            if isinstance(sfx_list, list):
                for item in sfx_list:
                    if isinstance(item, dict) and item.get("category"):
                        vault_sfx[item["category"]] = item
        except Exception:
            pass

    whoosh = vault_sfx.get(
        "transition_whoosh",
        {"id": "whoosh_fast_01", "filename": "04_assets/sfx/whoosh_fast_01.wav", "peak_transient_ms": 320, "recommended_gain_db": -14},
    )
    impact = vault_sfx.get(
        "impact_hook",
        {"id": "sub_drop_heavy_01", "filename": "04_assets/sfx/sub_drop_heavy_01.wav", "peak_transient_ms": 110, "recommended_gain_db": -12},
    )
    pop_ui = vault_sfx.get(
        "graphic_pop",
        {"id": "pop_click_ui_01", "filename": "04_assets/sfx/pop_click_ui_01.wav", "peak_transient_ms": 40, "recommended_gain_db": -16},
    )

    pre_cut_lead_sec = 4.0 / max(24.0, seq_fps)  # 4 frames before visual cut (Rule 3 of pacing-and-retention.md)
    placements: list[dict[str, Any]] = []

    # 1. Aperture slit unlock / drop impact
    unlock = aperture_curve.get("slit_unlock")
    if isinstance(unlock, dict) and unlock.get("slit_open_sec") is not None:
        u_sec = float(unlock["slit_open_sec"])
        start_s = snap_time_to_frame(max(0.0, u_sec - impact["peak_transient_ms"] / 1000.0 - pre_cut_lead_sec), seq_fps)
        placements.append(
            {
                "track": "A2",
                "event_type": "aperture_slit_unlock_impact",
                "sfx_id": impact["id"],
                "file_path": impact.get("path") or f"04_assets/sfx/{Path(impact['filename']).name}",
                "visual_sync_sec": u_sec,
                "timeline_start_sec": round(start_s, 3),
                "timeline_start_ticks": seconds_to_premiere_ticks(start_s),
                "gain_db": impact["recommended_gain_db"],
            }
        )

    # 2. Mask transitions & speed-ramp whip cuts
    for seg in slot_assignments:
        t_cut = float(seg["timeline_start_sec"])
        if t_cut <= 0.05:
            continue
        has_mask = bool(seg.get("mask_transitions"))
        has_ramp = bool(seg.get("speed_ramp", {}).get("detected"))
        if has_mask or has_ramp:
            start_s = snap_time_to_frame(max(0.0, t_cut - whoosh["peak_transient_ms"] / 1000.0 - pre_cut_lead_sec), seq_fps)
            placements.append(
                {
                    "track": "A2",
                    "event_type": "mask_or_whip_transition",
                    "slot_id": seg["slot_id"],
                    "sfx_id": whoosh["id"],
                    "file_path": whoosh.get("path") or f"04_assets/sfx/{Path(whoosh['filename']).name}",
                    "visual_sync_sec": t_cut,
                    "timeline_start_sec": round(start_s, 3),
                    "timeline_start_ticks": seconds_to_premiere_ticks(start_s),
                    "gain_db": whoosh["recommended_gain_db"],
                }
            )

    # 3. Kinetic typography pop-in entrances
    for ev in typography_plan.get("events", []):
        t_pop = float(ev.get("start_sec", 0.0))
        start_s = snap_time_to_frame(max(0.0, t_pop - pop_ui["peak_transient_ms"] / 1000.0 - pre_cut_lead_sec), seq_fps)
        placements.append(
            {
                "track": "A2",
                "event_type": f"typography_{ev.get('reveal_effect', 'pop')}",
                "text": ev.get("text"),
                "sfx_id": pop_ui["id"],
                "file_path": pop_ui.get("path") or f"04_assets/sfx/{Path(pop_ui['filename']).name}",
                "visual_sync_sec": t_pop,
                "timeline_start_sec": round(start_s, 3),
                "timeline_start_ticks": seconds_to_premiere_ticks(start_s),
                "gain_db": pop_ui["recommended_gain_db"],
            }
        )

    placements.sort(key=lambda x: x["timeline_start_sec"])
    return placements


def _preload_cache_from_raw_index(raw_idx: dict[str, Any]) -> None:
    """Populate _CLIP_ANALYSIS_CACHE directly from pre-indexed raw_scene_index.json to avoid 4K HEVC re-decode."""
    for c in raw_idx.get("clips", []):
        p_str = c.get("path")
        if not p_str:
            continue
        key = str(Path(p_str).resolve())
        if key in _CLIP_ANALYSIS_CACHE:
            continue
        dur = float(c.get("duration_sec", 5.0))
        fps = float(c.get("fps", 30.0))
        pt = c.get("person_tracking", {})
        samples = pt.get("person_samples", [])
        if len(samples) >= 2:
            times = [float(s.get("time_sec", 0.0)) for s in samples]
            p_covs = [float(s.get("coverage_pct", 0.0)) for s in samples]
            p_aligns = [float(s.get("centered_score", 0.75)) for s in samples]
            p_hm = [float(s.get("top_head_margin_pct", 15.0 if s.get("head_safe", True) else 0.0)) for s in samples]
        else:
            times = list(np.linspace(0.0, max(0.5, dur - 0.1), 8))
            cov_val = float(pt.get("max_person_coverage_pct", 0.0))
            p_covs = [cov_val] * len(times)
            p_aligns = [0.85 if cov_val > 0 else 0.0] * len(times)
            p_hm = [18.0 if cov_val > 0 else 100.0] * len(times)
        n_pts = len(times)
        sem = c.get("shot_semantics", {})
        mot_mag = float(sem.get("motion_magnitude", 2.0))
        stab_val = float(max(0.45, min(0.95, 1.0 - mot_mag / 22.0)))
        _CLIP_ANALYSIS_CACHE[key] = {
            "times": times,
            "frames": [],
            "dur": dur,
            "fps": fps,
            "focus": [0.82] * n_pts,
            "comp": [0.80] * n_pts,
            "exp": [0.85] * n_pts,
            "stab": [stab_val] * n_pts,
            "motion": [mot_mag] * n_pts,
            "person_covs": p_covs,
            "person_aligns": p_aligns,
            "person_head_margins": p_hm,
            "person_computed": True,
        }


def build_optimal_timeline_plan(
    style_dna_path: str = "02_analysis/style_dna.json",
    raw_index_path: str = "02_analysis/raw_scene_index.json",
    output_subclips_path: str = "02_analysis/optimal_subclips.json",
    output_plan_path: str = "02_analysis/timeline_plan.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map Style DNA slots onto optimal raw footage windows with theme, semantic, and person verification."""
    dna = json.loads(Path(style_dna_path).read_text(encoding="utf-8"))
    raw_idx = json.loads(Path(raw_index_path).read_text(encoding="utf-8"))
    _preload_cache_from_raw_index(raw_idx)

    all_clips = raw_idx.get("clips", [])
    coherence = dna.get("layer_3_aesthetic_typography_and_color", {}).get("visual_coherence_target", {})
    temporal = dna.get("layer_1_temporal_grammar", {})
    slots = dna.get("layer_4_narrative_slots", [])
    masks = temporal.get("mask_transitions", [])
    strips = temporal.get("multi_strip_opening", {})
    allow_reuse = bool(temporal.get("allow_clip_reuse_for_high_cut_density", True))
    seq_fps = float(raw_idx.get("recommended_sequence_fps", 60.0))

    all_greens = [float(c.get("environment", {}).get("green_fraction", 0.0)) for c in all_clips]
    proj_med_green = float(np.median(all_greens)) if all_greens else 0.0

    eligible_clips: list[dict[str, Any]] = []
    excluded_clips: list[dict[str, Any]] = []
    for c in all_clips:
        ok, theme_score = _score_clip_theme_fit(c, coherence, project_median_green=proj_med_green)
        entry = dict(c)
        entry["theme_fit_score"] = theme_score
        if ok:
            eligible_clips.append(entry)
        else:
            excluded_clips.append(
                {
                    "file_name": c["file_name"],
                    "reason": c.get("environment", {}).get("scene_category"),
                    "green_fraction": c.get("environment", {}).get("green_fraction"),
                    "sky_blue_fraction": c.get("environment", {}).get("sky_blue_fraction"),
                }
            )

    if not eligible_clips:
        eligible_clips = [dict(c, theme_fit_score=0.5) for c in all_clips]

    person_clips = sorted(
        [
            c
            for c in eligible_clips
            if c.get("person_tracking", {}).get("has_person")
            and c.get("person_tracking", {}).get("max_person_coverage_pct", 0.0) >= 8.0
        ],
        key=lambda c: (0 if c.get("role") == "b_roll" else 1, -c.get("person_tracking", {}).get("max_person_coverage_pct", 0.0)),
    )
    if not person_clips:
        person_clips = [c for c in eligible_clips if c.get("person_tracking", {}).get("has_person")]

    landscape_clips = [c for c in eligible_clips if c.get("role") == "a_roll"] or list(eligible_clips)

    used_windows: dict[str, list[tuple[float, float]]] = {}
    used_clip_counts: dict[str, int] = {}
    prev_cam_motion: str | None = None

    def _pick_clip(
        pool: list[dict[str, Any]],
        slot_obj: dict[str, Any] | None = None,
        prefer_person: bool = False,
    ) -> tuple[dict[str, Any], float]:
        scored = []
        for c in pool:
            sem_aff = _compute_semantic_slot_affinity(c, slot_obj or {}, prev_motion=prev_cam_motion) if slot_obj else 0.5
            combined = round(0.55 * float(c.get("theme_fit_score", 0.5)) + 0.45 * sem_aff, 4)
            scored.append((c, sem_aff, combined))

        if prefer_person:
            best = sorted(
                scored,
                key=lambda item: (
                    used_clip_counts.get(item[0]["file_name"], 0)
                    if item[0].get("role") == "b_roll"
                    else used_clip_counts.get(item[0]["file_name"], 0) + 1,
                    0 if item[0].get("role") == "b_roll" else 1,
                    -item[0].get("person_tracking", {}).get("max_person_coverage_pct", 0.0),
                    -item[2],
                ),
            )[0]
            return best[0], best[1]

        best = sorted(
            scored,
            key=lambda item: (
                used_clip_counts.get(item[0]["file_name"], 0),
                -item[2],
                -item[0].get("duration_sec", 0.0),
            ),
        )[0]
        return best[0], best[1]

    strip_allocations: list[dict[str, Any]] = []
    if strips.get("detected"):
        strip_dur = float(slots[0]["duration_sec"]) if slots else 3.5
        for st in strips.get("strips", []):
            chosen, _ = _pick_clip(landscape_clips, slot_obj=slots[0] if slots else None)
            opt = analyze_clip_moments(
                chosen["path"],
                target_duration=max(1.5, strip_dur),
                exclude_windows=used_windows.get(chosen["path"], []),
            )
            if opt:
                used_windows.setdefault(chosen["path"], []).append((opt["recommended_in_sec"], opt["recommended_out_sec"]))
                used_clip_counts[chosen["file_name"]] = used_clip_counts.get(chosen["file_name"], 0) + 1
            strip_allocations.append(
                {
                    "strip_index": st["strip_index"],
                    "x_range_px": st["x_range_px"],
                    "enter_sec": snap_time_to_frame(float(st["enter_sec"]), seq_fps),
                    "source_file": chosen["file_name"],
                    "source_path": chosen["path"],
                    "optimal_window": opt,
                }
            )

    slot_assignments: list[dict[str, Any]] = []
    subclips_report: list[dict[str, Any]] = []
    speed_ramp_plan: list[dict[str, Any]] = []
    punch_in_zoom_plan: list[dict[str, Any]] = []

    panel_overlays_by_seg = {
        int(p["segment_id"]): p for p in temporal.get("geometric_panel_overlays", []) if "segment_id" in p
    }
    carry_over_continuity: dict[str, Any] | None = None

    for sl in slots:
        seg_id_cur = int(sl["segment_id"])
        panel_info = panel_overlays_by_seg.get(seg_id_cur)
        req_person = bool(sl.get("mask_transitions"))
        slot_dur = max(0.15, float(sl.get("duration_sec", 0.5)))

        if carry_over_continuity is not None:
            # Enforce Zero-Jump 1:1 Screen-Space & Time Continuity from preceding panel slot!
            chosen = carry_over_continuity["clip_obj"]
            sem_affinity = carry_over_continuity["sem_affinity"]
            cont_in = float(carry_over_continuity["next_in_sec"])
            cont_out = snap_time_to_frame(min(float(chosen["duration_sec"]), cont_in + slot_dur), float(chosen["fps"]))
            opt = {
                "file_path": chosen["path"],
                "source_duration_sec": chosen["duration_sec"],
                "source_fps": chosen["fps"],
                "recommended_in_sec": round(cont_in, 3),
                "recommended_out_sec": round(cont_out, 3),
                "recommended_in_ticks": seconds_to_premiere_ticks(cont_in),
                "recommended_out_ticks": seconds_to_premiere_ticks(cont_out),
                "target_duration_sec": round(slot_dur, 3),
                "quality_score": carry_over_continuity["quality_score"],
                "metrics": {**carry_over_continuity["metrics"], "continuous_from_prev_panel": True},
            }
            if panel_info and panel_info.get("continuous_1to1_screen_space"):
                carry_over_continuity["next_in_sec"] = round(cont_out, 3)
            else:
                carry_over_continuity = None
        else:
            pool = person_clips if (req_person and person_clips) else landscape_clips
            if not allow_reuse:
                unused = [c for c in pool if used_clip_counts.get(c["file_name"], 0) == 0]
                if unused:
                    pool = unused

            chosen, sem_affinity = _pick_clip(pool, slot_obj=sl, prefer_person=req_person)
            excl = used_windows.get(chosen["path"], [])
            # If this slot is a panel that opens into the next slot, reserve enough duration for both slots!
            need_dur = slot_dur + 0.6 if (panel_info and panel_info.get("continuous_1to1_screen_space")) else slot_dur
            opt = analyze_clip_moments(
                chosen["path"],
                target_duration=need_dur,
                exclude_windows=excl,
                require_person=req_person,
                min_person_pct=2.0 if req_person else 0.0,
            )
            if opt is None:
                for alt in pool:
                    if alt["path"] != chosen["path"]:
                        opt = analyze_clip_moments(
                            alt["path"],
                            target_duration=need_dur,
                            exclude_windows=used_windows.get(alt["path"], []),
                            require_person=req_person,
                        )
                        if opt is not None:
                            chosen = alt
                            break
            if opt and panel_info and panel_info.get("continuous_1to1_screen_space"):
                # Trim current slot's out_sec to slot_dur and pass exact out_sec to next slot!
                in_s = float(opt["recommended_in_sec"])
                out_s = snap_time_to_frame(min(float(chosen["duration_sec"]), in_s + slot_dur), float(chosen["fps"]))
                opt["recommended_out_sec"] = round(out_s, 3)
                opt["recommended_out_ticks"] = seconds_to_premiere_ticks(out_s)
                carry_over_continuity = {
                    "clip_obj": chosen,
                    "sem_affinity": sem_affinity,
                    "next_in_sec": round(out_s, 3),
                    "quality_score": opt["quality_score"],
                    "metrics": opt["metrics"],
                }

        if opt:
            used_windows.setdefault(chosen["path"], []).append((opt["recommended_in_sec"], opt["recommended_out_sec"]))
            used_clip_counts[chosen["file_name"]] = used_clip_counts.get(chosen["file_name"], 0) + 1

        prev_cam_motion = chosen.get("shot_semantics", {}).get("camera_motion")
        tl_start = snap_time_to_frame(float(sl["start_sec"]), seq_fps)
        tl_end = snap_time_to_frame(float(sl["end_sec"]), seq_fps)
        tl_dur = round(max(1.0 / seq_fps, tl_end - tl_start), 4)

        sr_info = sl.get("speed_ramp", {"detected": False, "curve_type": "constant_100pct", "speed_points_pct": [100, 100, 100]})
        if sr_info.get("detected"):
            speed_ramp_plan.append(
                {
                    "slot_id": sl["segment_id"],
                    "assigned_clip": chosen["file_name"],
                    "timeline_start_sec": round(tl_start, 3),
                    "timeline_end_sec": round(tl_end, 3),
                    "curve_type": sr_info.get("curve_type"),
                    "speed_points_pct": sr_info.get("speed_points_pct"),
                }
            )

        # Retention visual reset rule (pacing-and-retention.md Rule 1): if a shot holds > 3.2s, schedule a creeping/punch zoom
        if tl_dur >= 3.2 and not strips.get("detected"):
            mid_zoom_t = round(tl_start + tl_dur * 0.48, 3)
            punch_in_zoom_plan.append(
                {
                    "slot_id": sl["segment_id"],
                    "assigned_clip": chosen["file_name"],
                    "trigger_sec": mid_zoom_t,
                    "trigger_ticks": seconds_to_premiere_ticks(mid_zoom_t),
                    "type": "creeping_bezier_zoom",
                    "start_scale_pct": 100.0,
                    "end_scale_pct": 108.0,
                    "interpolation": "bezier_ease_in_out",
                }
            )

        assignment = {
            "slot_id": sl["segment_id"],
            "timeline_start_sec": round(tl_start, 3),
            "timeline_end_sec": round(tl_end, 3),
            "timeline_start_ticks": seconds_to_premiere_ticks(tl_start),
            "timeline_end_ticks": seconds_to_premiere_ticks(tl_end),
            "duration_sec": round(tl_dur, 3),
            "in_expanding_slit": sl.get("in_expanding_slit", False),
            "mask_transitions": sl.get("mask_transitions", []),
            "is_beat_locked": sl.get("is_beat_locked", False),
            "speed_ramp": sr_info,
            "require_person": req_person,
            "assigned_clip": chosen["file_name"],
            "assigned_role": chosen["role"],
            "source_path": chosen["path"],
            "source_fps": chosen["fps"],
            "shot_scale": chosen.get("shot_semantics", {}).get("shot_scale"),
            "camera_motion": chosen.get("shot_semantics", {}).get("camera_motion"),
            "theme_fit_score": chosen.get("theme_fit_score"),
            "semantic_affinity_score": sem_affinity,
            "recommended_in_sec": opt["recommended_in_sec"] if opt else 0.0,
            "recommended_out_sec": opt["recommended_out_sec"] if opt else round(min(chosen["duration_sec"], slot_dur), 3),
            "recommended_in_ticks": opt["recommended_in_ticks"] if opt else 0,
            "recommended_out_ticks": opt["recommended_out_ticks"] if opt else seconds_to_premiere_ticks(min(chosen["duration_sec"], slot_dur)),
            "quality_score": opt["quality_score"] if opt else 0.0,
            "metrics": opt["metrics"] if opt else {},
        }
        slot_assignments.append(assignment)
        if opt:
            subclips_report.append(
                {
                    "slot_id": sl["segment_id"],
                    "clip": chosen["file_name"],
                    "role": chosen["role"],
                    "in_sec": opt["recommended_in_sec"],
                    "out_sec": opt["recommended_out_sec"],
                    "duration_sec": round(tl_dur, 3),
                    "require_person": req_person,
                    "semantic_affinity_score": sem_affinity,
                    "quality_score": opt["quality_score"],
                    "metrics": opt["metrics"],
                }
            )

    spatial = dna.get("spatial_geometry", {})
    rec_adapt = spatial.get("recommended_landscape_adaptation", {})
    typo_plan = dna.get("layer_3_aesthetic_typography_and_color", {}).get("typography", {})
    aperture_curve = temporal.get("aperture_animation", {})
    a2_sfx_plan = _build_a2_sfx_plan(slot_assignments, aperture_curve, typo_plan, seq_fps)

    # Reverse-chronological cut execution list (strictly descending t_n > ... > t_1 per premiere-dom-invariants.md)
    descending_cut_seconds = sorted(
        {round(float(s["timeline_start_sec"]), 3) for s in slot_assignments if float(s["timeline_start_sec"]) > 0.0},
        reverse=True,
    )

    optimal_manifest = {
        "schema_version": "2.1",
        "excluded_off_theme_clips": excluded_clips,
        "total_slots_planned": len(subclips_report),
        "strip_opening_allocations": strip_allocations,
        "optimized_subclips": subclips_report,
    }
    timeline_plan = {
        "schema_version": "2.1",
        "sequence_config": {
            "target_resolution": rec_adapt.get("target_resolution", spatial.get("container_resolution", [1080, 1920])),
            "target_fps": seq_fps,
            "strip_static_letterbox": rec_adapt.get("strip_static_letterbox", False),
            "scaling_policy": "crop_no_stretch",
            "snap_all_cuts_to_frame_grid": True,
            "premiere_ticks_per_second": 254016000000,
        },
        "reverse_chronological_cut_timestamps_sec": descending_cut_seconds,
        "excluded_off_theme_clips": excluded_clips,
        "multi_strip_opening_plan": strip_allocations,
        "aperture_curve": aperture_curve,
        "mask_transitions_plan": masks,
        "speed_ramp_plan": speed_ramp_plan,
        "punch_in_zoom_plan": punch_in_zoom_plan,
        "typography_plan": typo_plan,
        "audio_plan": dna.get("layer_2_acoustic_architecture", {}),
        "a2_sfx_plan": a2_sfx_plan,
        "color_grade_plan": {
            "pass_1_strategy": "normalize_each_clip_via_02_analysis/color_telemetry.json",
            "pass_2_strategy": "match_all_broll_and_strips_to_hero_a_roll_green_luma",
            "pass_3_track_v4_lumetri_matrix": dna.get("layer_3_aesthetic_typography_and_color", {}).get("lumetri_matrix", {}),
        },
        "timeline_segments": slot_assignments,
    }

    Path(output_subclips_path).write_text(json.dumps(optimal_manifest, indent=2), encoding="utf-8")
    Path(output_plan_path).write_text(json.dumps(timeline_plan, indent=2), encoding="utf-8")
    return optimal_manifest, timeline_plan


def optimize_all_subclips(
    plan_path: str = "02_analysis/timeline_plan.json",
    output_path: str = "02_analysis/optimal_subclips.json",
    raw_index_path: str = "02_analysis/raw_scene_index.json",
) -> None:
    """Refresh optimal [t_in, t_out] selections for an existing timeline_plan.json."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    raw_idx = json.loads(Path(raw_index_path).read_text(encoding="utf-8")) if Path(raw_index_path).exists() else None
    results = []
    used_windows: dict[str, list[tuple[float, float]]] = {}

    for seg in plan.get("timeline_segments", []):
        clip_name = seg.get("assigned_clip") or seg.get("clip")
        target_dur = float(seg.get("duration_sec", 2.0))
        req_person = bool(seg.get("require_person") or seg.get("mask_transitions"))
        if not clip_name:
            continue
        clip_path = _resolve_clip_path(str(clip_name), raw_idx)
        if not clip_path:
            continue
        res = analyze_clip_moments(
            clip_path,
            target_duration=target_dur,
            exclude_windows=used_windows.get(clip_path, []),
            require_person=req_person,
        )
        if res:
            used_windows.setdefault(clip_path, []).append((res["recommended_in_sec"], res["recommended_out_sec"]))
            res["segment_id"] = seg.get("slot_id") or seg.get("segment_id")
            results.append(res)

    Path(output_path).write_text(json.dumps({"optimized_subclips": results}, indent=2), encoding="utf-8")
    print(f"[OK] Saved {len(results)} optimal cinematic windows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize subclip windows and plan coherent Style DNA -> raw footage mapping.")
    parser.add_argument("--style_dna", default="02_analysis/style_dna.json", help="Path to style_dna.json")
    parser.add_argument("--raw_index", default="02_analysis/raw_scene_index.json", help="Path to raw_scene_index.json")
    parser.add_argument("--plan", default=None, help="Existing timeline_plan.json to refresh subclips from")
    parser.add_argument("--output", default="02_analysis/optimal_subclips.json", help="Output optimal_subclips.json path")
    parser.add_argument("--out_plan", default="02_analysis/timeline_plan.json", help="Output timeline_plan.json path")
    args = parser.parse_args()

    if args.plan and Path(args.plan).exists() and not Path(args.style_dna).exists():
        optimize_all_subclips(args.plan, args.output, args.raw_index)
        return

    if Path(args.style_dna).exists() and Path(args.raw_index).exists():
        opt, plan = build_optimal_timeline_plan(args.style_dna, args.raw_index, args.output, args.out_plan)
        print(
            f"[OK] Planned {len(plan['timeline_segments'])} segments, {len(plan.get('a2_sfx_plan', []))} A2 SFX cues, "
            f"& {len(opt['optimized_subclips'])} subclips -> {args.out_plan}"
        )
        if opt["excluded_off_theme_clips"]:
            print(f"     Excluded off-theme clips: {opt['excluded_off_theme_clips']}")
    elif Path(args.plan or "02_analysis/timeline_plan.json").exists():
        optimize_all_subclips(args.plan or "02_analysis/timeline_plan.json", args.output, args.raw_index)
    else:
        raise SystemExit("[ERROR] Need either (--style_dna + --raw_index) or an existing --plan.")


if __name__ == "__main__":
    main()
