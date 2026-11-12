"""Peak Cinematic Subclip Optimizer & Style-to-Footage Slot Planner.

Computes 4D CV moment quality (optical-flow camera stability, Laplacian focus sharpness,
rule-of-thirds visual energy, exposure safety) PLUS visual-environment coherence (green/outdoor
consistency) PLUS PyTorch DeepLabV3 person-subject presence verification, with adaptive
micro-cut sampling and non-overlapping multi-window reuse across rapid beat montages.
"""

from __future__ import annotations

import argparse
import json
import os
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
from virtual_scopes import fast_sample_keyframes  # noqa: E402

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


def _estimate_person_coverage(frame_bgr: np.ndarray) -> tuple[float, float]:
    """Return (person_coverage_pct, center_alignment_0_to_1) for a BGR frame."""
    bundle = _get_deeplab_person_model()
    if bundle is None:
        return 0.0, 0.0
    import torch
    from PIL import Image

    model, preprocess = bundle
    small = cv2.resize(frame_bgr, (320, max(180, int(round(320 * frame_bgr.shape[0] / max(1, frame_bgr.shape[1]))))))
    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    inp = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        preds = model(inp)["out"][0].argmax(0).byte().cpu().numpy()
    mask = preds == 15
    cov_pct = float(np.mean(mask) * 100.0)
    if cov_pct < 0.4:
        return round(cov_pct, 2), 0.0
    mh, mw = mask.shape[:2]
    ys, xs = np.where(mask)
    cx = float(np.mean(xs) / max(1, mw))
    cy = float(np.mean(ys) / max(1, mh))
    align = max(0.0, 1.0 - (abs(cx - 0.5) * 1.4 + abs(cy - 0.52) * 0.8))
    return round(cov_pct, 2), round(align, 3)


def _analyze_clip_cached(clip_path: str, require_person: bool = False) -> dict[str, Any] | None:
    """Decode & cache container keyframes + 4D CV scores + optional DeepLabV3 person tracks once per clip."""
    key = str(Path(clip_path).resolve())
    cached = _CLIP_ANALYSIS_CACHE.get(key)
    if cached is not None:
        if require_person and not cached.get("person_computed"):
            p_covs, p_aligns = [], []
            for fr in cached["frames"]:
                cov, al = _estimate_person_coverage(fr)
                p_covs.append(cov)
                p_aligns.append(al)
            cached["person_covs"] = p_covs
            cached["person_aligns"] = p_aligns
            cached["person_computed"] = True
        return cached

    times, frames, meta = fast_sample_keyframes(clip_path, target_width=480, max_frames=16)
    if len(times) < 2:
        return None

    focus_scores: list[float] = []
    comp_scores: list[float] = []
    exp_scores: list[float] = []
    stability_scores: list[float] = []
    p_covs: list[float] = []
    p_aligns: list[float] = []

    prev_gray: np.ndarray | None = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        focus_scores.append(compute_focus_score(gray))
        comp_scores.append(compute_composition_energy(gray))
        exp_scores.append(compute_exposure_score(gray))
        stab = 0.85
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.median(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            stab = float(max(0.0, 1.0 - (mag / 18.0)))
        stability_scores.append(stab)
        prev_gray = gray
        if require_person:
            cov, al = _estimate_person_coverage(frame)
            p_covs.append(cov)
            p_aligns.append(al)

    entry = {
        "times": times,
        "frames": frames,
        "dur": float(meta.get("duration") or times[-1]),
        "fps": float(meta.get("fps") or 30.0),
        "focus": focus_scores,
        "comp": comp_scores,
        "exp": exp_scores,
        "stab": stability_scores,
        "person_covs": p_covs,
        "person_aligns": p_aligns,
        "person_computed": require_person,
    }
    _CLIP_ANALYSIS_CACHE[key] = entry
    return entry


def analyze_clip_moments(
    clip_path: str,
    target_duration: float,
    sample_fps: float = 2.0,
    min_in_sec: float = 0.0,
    max_out_sec: float | None = None,
    exclude_windows: list[tuple[float, float]] | None = None,
    require_person: bool = False,
    min_person_pct: float = 2.0,
) -> dict[str, Any] | None:
    """Search clip timeline for optimal non-overlapping [t_in, t_out] cinematic window."""
    data = _analyze_clip_cached(clip_path, require_person=require_person)
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

    # Dense candidate start grid enables clean disjoint reuse of micro-cut windows across long clips
    grid_step = min(0.35, max(0.15, eff_target * 0.9))
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
        # Bracket indices spanning [t_s, t_e]
        idx_s = int(np.argmin(np.abs(times - t_s)))
        idx_m = int(np.argmin(np.abs(times - t_mid)))
        idx_e = int(np.argmin(np.abs(times - t_e)))
        lo, hi = min(idx_s, idx_e), max(idx_s, idx_e) + 1

        avg_foc = float(np.mean(data["focus"][lo:hi]))
        avg_stab = float(np.mean(data["stab"][lo:hi]))
        avg_comp = float(np.mean(data["comp"][lo:hi]))
        avg_exp = float(np.mean(data["exp"][lo:hi]))
        raw_quality = 0.32 * avg_stab + 0.28 * avg_foc + 0.22 * avg_comp + 0.18 * avg_exp

        person_cov = 0.0
        person_align = 0.0
        if require_person and data.get("person_computed"):
            cov_s = data["person_covs"][idx_s]
            cov_m = data["person_covs"][idx_m]
            al_s = data["person_aligns"][idx_s]
            al_m = data["person_aligns"][idx_m]
            person_cov = min(cov_s, cov_m) if cov_s > 0 and cov_m > 0 else max(cov_s, cov_m) * 0.5
            person_align = (al_s + al_m) * 0.5
            if person_cov < min_person_pct:
                raw_quality -= 2.0
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
                "person_coverage_pct": round(person_cov, 2),
                "person_alignment": round(person_align, 3),
            }

    if not best_metrics:
        best_in = float(min_in_sec)
        best_score = 0.5
        best_metrics = {
            "stability": round(float(np.mean(data["stab"])), 3),
            "focus": round(float(np.mean(data["focus"])), 3),
            "composition": round(float(np.mean(data["comp"])), 3),
            "exposure": round(float(np.mean(data["exp"])), 3),
            "person_coverage_pct": 0.0,
            "person_alignment": 0.0,
            "overlap_relaxed": True,
        }

    in_sec = round(best_in, 3)
    out_sec = round(min(t_limit, in_sec + target_duration), 3)
    return {
        "file_path": Path(clip_path).as_posix(),
        "source_duration_sec": round(dur, 3),
        "source_fps": round(fps, 3),
        "recommended_in_sec": in_sec,
        "recommended_out_sec": out_sec,
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

    # Reject indoor/enclosed shots OR dark low-green/low-sky bus-window outliers when green valleys dominate
    if need_outdoor and (
        likely_indoor
        or (tgt_green >= 0.08 and green_f < 0.025 and sky_f < 0.025)
        or (project_median_green >= 0.15 and green_f < 0.05 and sky_f < 0.045)
    ):
        return False, -1.0

    # Higher score for matching greenery/alpine visual density
    if tgt_green > 0:
        fit = 1.0 - min(1.0, abs(green_f - tgt_green) / max(0.25, tgt_green))
    else:
        fit = 0.7
    return True, round(0.6 * fit + 0.4 * min(1.0, green_f * 2.5 + sky_f * 1.5), 3)


def build_optimal_timeline_plan(
    style_dna_path: str = "02_analysis/style_dna.json",
    raw_index_path: str = "02_analysis/raw_scene_index.json",
    output_subclips_path: str = "02_analysis/optimal_subclips.json",
    output_plan_path: str = "02_analysis/timeline_plan.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map Style DNA slots onto optimal raw footage windows with theme & person verification."""
    dna = json.loads(Path(style_dna_path).read_text(encoding="utf-8"))
    raw_idx = json.loads(Path(raw_index_path).read_text(encoding="utf-8"))

    all_clips = raw_idx.get("clips", [])
    coherence = dna.get("layer_3_aesthetic_typography_and_color", {}).get("visual_coherence_target", {})
    temporal = dna.get("layer_1_temporal_grammar", {})
    slots = dna.get("layer_4_narrative_slots", [])
    masks = temporal.get("mask_transitions", [])
    strips = temporal.get("multi_strip_opening", {})
    allow_reuse = bool(temporal.get("allow_clip_reuse_for_high_cut_density", True))

    all_greens = [float(c.get("environment", {}).get("green_fraction", 0.0)) for c in all_clips]
    proj_med_green = float(np.median(all_greens)) if all_greens else 0.0

    # Filter & rank consistent raw clips
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

    # Reserve dedicated B-roll/high-coverage subjects for mask slots; allow ALL consistent outdoor clips in landscape strips/cuts
    person_clips = sorted(
        [c for c in eligible_clips if c.get("person_tracking", {}).get("has_person") and c.get("person_tracking", {}).get("max_person_coverage_pct", 0.0) >= 8.0],
        key=lambda c: (0 if c.get("role") == "b_roll" else 1, -c.get("person_tracking", {}).get("max_person_coverage_pct", 0.0)),
    )
    if not person_clips:
        person_clips = [c for c in eligible_clips if c.get("person_tracking", {}).get("has_person")]

    # Prioritize a_roll lush outdoor clips first for general cuts/strips, keeping dedicated b_roll walkers for hero mask payoffs
    landscape_clips = [c for c in eligible_clips if c.get("role") == "a_roll"] or list(eligible_clips)

    used_windows: dict[str, list[tuple[float, float]]] = {}
    used_clip_counts: dict[str, int] = {}

    def _pick_clip(pool: list[dict[str, Any]], prefer_person: bool = False) -> dict[str, Any]:
        if prefer_person:
            return sorted(
                pool,
                key=lambda c: (
                    used_clip_counts.get(c["file_name"], 0) if c.get("role") == "b_roll" else used_clip_counts.get(c["file_name"], 0) + 1,
                    0 if c.get("role") == "b_roll" else 1,
                    -c.get("person_tracking", {}).get("max_person_coverage_pct", 0.0),
                ),
            )[0]
        return sorted(
            pool,
            key=lambda c: (used_clip_counts.get(c["file_name"], 0), -c.get("theme_fit_score", 0.0), -c.get("duration_sec", 0.0)),
        )[0]

    # Allocate strip opening visuals (if multi-strip intro exists)
    strip_allocations: list[dict[str, Any]] = []
    if strips.get("detected"):
        strip_dur = float(slots[0]["duration_sec"]) if slots else 3.5
        for st in strips.get("strips", []):
            chosen = _pick_clip(landscape_clips)
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
                    "enter_sec": st["enter_sec"],
                    "source_file": chosen["file_name"],
                    "source_path": chosen["path"],
                    "optimal_window": opt,
                }
            )

    # Allocate each narrative slot cleanly (distributing unique clips first, then disjoint reuse windows)
    slot_assignments: list[dict[str, Any]] = []
    subclips_report: list[dict[str, Any]] = []

    for sl in slots:
        req_person = bool(sl.get("mask_transitions"))
        slot_dur = max(0.15, float(sl.get("duration_sec", 0.5)))

        pool = person_clips if (req_person and person_clips) else landscape_clips
        if not allow_reuse:
            unused = [c for c in pool if used_clip_counts.get(c["file_name"], 0) == 0]
            if unused:
                pool = unused

        chosen = _pick_clip(pool, prefer_person=req_person)
        # Keep micro-cut reuse windows non-overlapping on source clips
        excl = used_windows.get(chosen["path"], [])
        opt = analyze_clip_moments(
            chosen["path"],
            target_duration=slot_dur,
            exclude_windows=excl,
            require_person=req_person,
            min_person_pct=2.0 if req_person else 0.0,
        )
        if opt is None:
            # Fallback if clip too short: try another clip in pool
            for alt in pool:
                if alt["path"] != chosen["path"]:
                    opt = analyze_clip_moments(
                        alt["path"],
                        target_duration=slot_dur,
                        exclude_windows=used_windows.get(alt["path"], []),
                        require_person=req_person,
                    )
                    if opt is not None:
                        chosen = alt
                        break

        if opt:
            used_windows.setdefault(chosen["path"], []).append((opt["recommended_in_sec"], opt["recommended_out_sec"]))
            used_clip_counts[chosen["file_name"]] = used_clip_counts.get(chosen["file_name"], 0) + 1

        assignment = {
            "slot_id": sl["segment_id"],
            "timeline_start_sec": sl["start_sec"],
            "timeline_end_sec": sl["end_sec"],
            "duration_sec": sl["duration_sec"],
            "in_expanding_slit": sl.get("in_expanding_slit", False),
            "mask_transitions": sl.get("mask_transitions", []),
            "require_person": req_person,
            "assigned_clip": chosen["file_name"],
            "assigned_role": chosen["role"],
            "source_path": chosen["path"],
            "source_fps": chosen["fps"],
            "theme_fit_score": chosen.get("theme_fit_score"),
            "recommended_in_sec": opt["recommended_in_sec"] if opt else 0.0,
            "recommended_out_sec": opt["recommended_out_sec"] if opt else round(min(chosen["duration_sec"], slot_dur), 3),
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
                    "duration_sec": sl["duration_sec"],
                    "require_person": req_person,
                    "quality_score": opt["quality_score"],
                    "metrics": opt["metrics"],
                }
            )

    # Frame-exact export config (never bake static letterbox when reference active region is 16:9!)
    spatial = dna.get("spatial_geometry", {})
    rec_adapt = spatial.get("recommended_landscape_adaptation", {})
    seq_fps = raw_idx.get("recommended_sequence_fps", 60.0)

    optimal_manifest = {
        "schema_version": "2.0",
        "excluded_off_theme_clips": excluded_clips,
        "total_slots_planned": len(subclips_report),
        "strip_opening_allocations": strip_allocations,
        "optimized_subclips": subclips_report,
    }
    timeline_plan = {
        "schema_version": "2.0",
        "sequence_config": {
            "target_resolution": rec_adapt.get("target_resolution", spatial.get("container_resolution", [1080, 1920])),
            "target_fps": seq_fps,
            "strip_static_letterbox": rec_adapt.get("strip_static_letterbox", False),
            "scaling_policy": "crop_no_stretch",
            "snap_all_cuts_to_frame_grid": True,
        },
        "excluded_off_theme_clips": excluded_clips,
        "multi_strip_opening_plan": strip_allocations,
        "aperture_curve": temporal.get("aperture_animation", {}),
        "mask_transitions_plan": masks,
        "typography_plan": dna.get("layer_3_aesthetic_typography_and_color", {}).get("typography", {}),
        "audio_plan": dna.get("layer_2_acoustic_architecture", {}),
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
        print(f"[OK] Planned {len(plan['timeline_segments'])} segments & {len(opt['optimized_subclips'])} subclips -> {args.out_plan}")
        if opt["excluded_off_theme_clips"]:
            print(f"     Excluded off-theme clips: {opt['excluded_off_theme_clips']}")
    elif Path(args.plan or "02_analysis/timeline_plan.json").exists():
        optimize_all_subclips(args.plan or "02_analysis/timeline_plan.json", args.output, args.raw_index)
    else:
        raise SystemExit("[ERROR] Need either (--style_dna + --raw_index) or an existing --plan.")


if __name__ == "__main__":
    main()
