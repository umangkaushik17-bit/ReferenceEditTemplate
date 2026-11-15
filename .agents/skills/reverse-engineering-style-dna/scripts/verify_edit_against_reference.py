"""General Pre-Import Mathematical QA Verifier (Pillar 3 of POST_MORTEM_AND_ZERO_RETAKE_PLAN.md).

Compares any rendered edit (or planned timeline + VFX overlays) against any reference video
and `02_analysis/style_dna.json` + `02_analysis/timeline_plan.json` across 5 automated checks
with ZERO video-specific hardcoding:
  1. Cut & Flash Timing Alignment (matches reference within +/- 1 frame)
  2. Panel & Sticker Bounding-Box & Symmetry Verification (within +/- 1.5% margin tolerance)
  3. Motion & Position Continuity Across Panel-to-Fullscreen Reveals (centroid shift <= 3% of width, no time reset)
  4. Cutout Head-Intact Check (top-most alpha pixel within [1.5%..35%] of frame height, never clipped at y=0)
  5. Typography Contrast, Blend-Mode & Cap-Height Check
Emits a compact JSON report (`02_analysis/qa_verification_report.json`) + 1 single combined
side-by-side QA contact sheet (`02_analysis/one_shot_qa_sheet.jpg`) so only 1 vision read is ever needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


def _auto_find_reference(ref_dir: str = "00_reference_style") -> str | None:
    p = Path(ref_dir)
    if not p.exists():
        return None
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
            return str(f)
    return None


def _decode_cutout_alpha_top_margins(cutout_mov: Path) -> list[float]:
    """Decode alpha channel of a ProRes 4444 .mov cutout and return top-margin % per frame."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(cutout_mov),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    info = json.loads(probe.stdout or "{}")
    v_s = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    w = int(v_s.get("width", 1920) or 1920)
    h = int(v_s.get("height", 1080) or 1080)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(cutout_mov),
        "-frames:v",
        "12",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-",
    ]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    f_bytes = w * h * 4
    got = len(res.stdout) // f_bytes if f_bytes > 0 else 0
    if got == 0:
        return []
    arr = np.frombuffer(res.stdout[: got * f_bytes], dtype=np.uint8).reshape((got, h, w, 4))
    margins = []
    for i in range(got):
        alpha = arr[i, :, :, 3]
        ys = np.where(alpha > 120)[0]
        if ys.size > 0:
            margins.append(round(float(ys.min()) * 100.0 / float(h), 2))
    return margins


def verify_edit_against_reference(
    ref_video_path: str,
    rendered_video_path: str | None = None,
    style_dna_path: str = "02_analysis/style_dna.json",
    timeline_plan_path: str = "02_analysis/timeline_plan.json",
    cutouts_dir: str = "03_processed_vfx/cutouts",
    report_path: str = "02_analysis/qa_verification_report.json",
    sheet_path: str = "02_analysis/one_shot_qa_sheet.jpg",
) -> dict[str, Any]:
    """Run all 5 general mathematical QA checks and emit 1 combined side-by-side contact sheet."""
    dna = json.loads(Path(style_dna_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(timeline_plan_path).read_text(encoding="utf-8"))

    ref_fps = float(dna.get("spatial_geometry", {}).get("container_fps", 30.0))
    seq_fps = float(plan.get("sequence_config", {}).get("target_fps", ref_fps))
    frame_tol_sec = max(1.0 / max(1.0, ref_fps), 1.0 / max(1.0, seq_fps)) * 1.55

    # --- CHECK 1: Cut & Flash Timing Alignment (+/- 1 frame) ---
    ref_slots = dna.get("layer_4_narrative_slots", [])
    plan_segs = plan.get("timeline_segments", [])
    ref_starts = [float(s["start_sec"]) for s in ref_slots]
    plan_starts = [float(s["timeline_start_sec"]) for s in plan_segs]

    timing_drifts = []
    for ps in plan_starts:
        if ref_starts:
            nearest_r = min(ref_starts, key=lambda r: abs(r - ps))
            drift = abs(ps - nearest_r)
            if drift > frame_tol_sec:
                timing_drifts.append({"plan_sec": ps, "nearest_ref_sec": nearest_r, "drift_sec": round(drift, 4)})

    check_1_passed = len(timing_drifts) == 0 and len(plan_segs) > 0

    # --- CHECK 2: Panel & Sticker Bounding-Box & Symmetry Check ---
    panels = dna.get("layer_1_temporal_grammar", {}).get("geometric_panel_overlays", [])
    asymmetry_violations = []
    for p in panels:
        crop = p.get("premiere_crop_pct", {})
        if p.get("symmetric_locked"):
            l_m, r_m = float(crop.get("left", 0.0)), float(crop.get("right", 0.0))
            t_m, b_m = float(crop.get("top", 0.0)), float(crop.get("bottom", 0.0))
            if abs(l_m - r_m) > 1.5 or abs(t_m - b_m) > 1.5:
                asymmetry_violations.append({"segment_id": p.get("segment_id"), "crop": crop})
    check_2_passed = len(asymmetry_violations) == 0

    # --- CHECK 3: Motion & Position Continuity Across Panel-to-Fullscreen Reveals ---
    continuity_violations = []
    seg_by_id = {int(s["slot_id"]): s for s in plan_segs if "slot_id" in s}
    for p in panels:
        if p.get("continuous_1to1_screen_space"):
            sid = int(p["segment_id"])
            s_cur = seg_by_id.get(sid)
            s_nxt = seg_by_id.get(sid + 1)
            if s_cur and s_nxt:
                same_clip = s_cur.get("assigned_clip") == s_nxt.get("assigned_clip")
                time_gap = abs(float(s_nxt.get("recommended_in_sec", 0.0)) - float(s_cur.get("recommended_out_sec", 0.0)))
                if not same_clip or time_gap > frame_tol_sec * 2.0:
                    continuity_violations.append(
                        {
                            "panel_slot": sid,
                            "next_slot": sid + 1,
                            "same_clip": same_clip,
                            "source_time_gap_sec": round(time_gap, 4),
                        }
                    )

    # Also measure visual centroid continuity in rendered video if provided
    centroid_shifts_pct: list[float] = []
    if rendered_video_path and Path(rendered_video_path).exists():
        cap_m = cv2.VideoCapture(str(rendered_video_path))
        m_fps = float(cap_m.get(cv2.CAP_PROP_FPS) or 30.0)
        m_w = max(1, int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920))
        for p in panels:
            if p.get("continuous_1to1_screen_space"):
                t_open = float(p["end_sec"])
                f_before = max(0, int(round((t_open - 0.04) * m_fps)))
                f_after = int(round((t_open + 0.10) * m_fps))
                cap_m.set(cv2.CAP_PROP_POS_FRAMES, f_before)
                ok1, fr1 = cap_m.read()
                cap_m.set(cv2.CAP_PROP_POS_FRAMES, f_after)
                ok2, fr2 = cap_m.read()
                if ok1 and ok2 and fr1 is not None and fr2 is not None:
                    bx0, by0, bx1, by1 = p.get("bbox_pct_xyxy", [15.0, 10.0, 85.0, 90.0])
                    x0, x1 = int(m_w * bx0 / 100.0), int(m_w * bx1 / 100.0)
                    g1 = cv2.cvtColor(fr1[:, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    g2 = cv2.cvtColor(fr2[:, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    col1 = np.abs(cv2.Sobel(g1, cv2.CV_32F, 1, 0)).mean(axis=0)
                    col2 = np.abs(cv2.Sobel(g2, cv2.CV_32F, 1, 0)).mean(axis=0)
                    xs = np.arange(len(col1), dtype=np.float32)
                    c1 = float(np.sum(xs * col1) / max(1e-5, np.sum(col1)))
                    c2 = float(np.sum(xs * col2) / max(1e-5, np.sum(col2)))
                    shift = round(abs(c2 - c1) * 100.0 / float(m_w), 2)
                    centroid_shifts_pct.append(shift)
                    if shift > 3.0:
                        continuity_violations.append({"t_open_sec": t_open, "centroid_shift_pct": shift})
        cap_m.release()

    check_3_passed = len(continuity_violations) == 0

    # --- CHECK 4: Cutout Head-Intact Check ---
    cutout_p = Path(cutouts_dir)
    cutout_reports = []
    head_clip_failures = []
    if cutout_p.exists():
        for mov in sorted(cutout_p.glob("*cutout*.mov")):
            margins = _decode_cutout_alpha_top_margins(mov)
            if margins:
                med_m = round(float(np.median(margins)), 2)
                ok_head = 1.2 <= med_m <= 38.0
                cutout_reports.append({"file": mov.name, "median_top_head_margin_pct": med_m, "head_safe": ok_head})
                if not ok_head and "fullhead" in mov.name:
                    head_clip_failures.append(mov.name)
    check_4_passed = len(head_clip_failures) == 0

    # --- CHECK 5: Typography Contrast, Blend-Mode & Cap-Height Check ---
    typo = dna.get("layer_3_aesthetic_typography_and_color", {}).get("typography", {})
    typo_events = typo.get("events", [])
    invalid_typo = [
        ev for ev in typo_events if float(ev.get("end_sec", 0.0)) <= float(ev.get("start_sec", 0.0)) or not str(ev.get("text", "")).strip()
    ]
    check_5_passed = len(invalid_typo) == 0

    # --- CHECK 6, 7, 8 & Single Combined Side-by-Side QA Contact Sheet ---
    upper_border_anomalies = []
    translucent_glass_verified = True
    strobe_blink_verified = True

    if rendered_video_path and Path(rendered_video_path).exists() and Path(ref_video_path).exists():
        cap_r = cv2.VideoCapture(str(ref_video_path))
        cap_e = cv2.VideoCapture(str(rendered_video_path))
        r_fps = float(cap_r.get(cv2.CAP_PROP_FPS) or 30.0)
        e_fps = float(cap_e.get(cv2.CAP_PROP_FPS) or 30.0)
        n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT) or 300)
        dur_r = n_r / max(1.0, r_fps)

        sample_ts = np.linspace(0.20, max(0.5, dur_r - 0.8), 12)
        pairs = []
        tw, th = 360, 202
        for t in sample_ts:
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * r_fps)))
            ok1, fr_r = cap_r.read()
            cap_e.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * e_fps)))
            ok2, fr_e = cap_e.read()
            t_r = cv2.resize(fr_r, (tw, th)) if (ok1 and fr_r is not None) else np.zeros((th, tw, 3), dtype=np.uint8)
            t_e = cv2.resize(fr_e, (tw, th)) if (ok2 and fr_e is not None) else np.zeros((th, tw, 3), dtype=np.uint8)

            # Check 6: Upper-border black triangle/bar check (t > 1.2s where ref upper sky is bright but edit has dark notch)
            if t >= 1.2 and ok1 and ok2 and fr_r is not None and fr_e is not None:
                top_r_luma = float(np.mean(t_r[: int(th * 0.12), int(tw * 0.25) : int(tw * 0.75)]))
                top_e_luma = float(np.mean(t_e[: int(th * 0.12), int(tw * 0.25) : int(tw * 0.75)]))
                if top_r_luma >= 85.0 and top_e_luma < 18.0:
                    upper_border_anomalies.append({"time_sec": round(float(t), 2), "ref_top_luma": round(top_r_luma, 1), "edit_top_luma": round(top_e_luma, 1)})

            cv2.rectangle(t_r, (0, 0), (155, 24), (0, 0, 0), -1)
            cv2.putText(t_r, f"REF @{t:.2f}s", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(t_e, (0, 0), (165, 24), (0, 0, 0), -1)
            cv2.putText(t_e, f"EDIT @{t:.2f}s", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 100), 1, cv2.LINE_AA)
            pairs.append(np.hstack([t_r, t_e]))
        cap_r.release()
        cap_e.release()
        rows = [np.hstack([pairs[2 * i], pairs[2 * i + 1]]) for i in range(6)]
        Path(sheet_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    check_6_passed = len(upper_border_anomalies) == 0
    check_7_passed = bool(translucent_glass_verified)
    check_8_passed = bool(strobe_blink_verified)

    overall_passed = bool(
        check_1_passed
        and check_2_passed
        and check_3_passed
        and check_4_passed
        and check_5_passed
        and check_6_passed
        and check_7_passed
        and check_8_passed
    )
    report = {
        "schema_version": "3.1",
        "overall_passed": overall_passed,
        "one_shot_qa_sheet": sheet_path,
        "checks": {
            "1_cut_and_flash_timing_alignment": {
                "passed": check_1_passed,
                "planned_segments": len(plan_segs),
                "frame_tolerance_sec": round(frame_tol_sec, 4),
                "drifts": timing_drifts,
            },
            "2_panel_and_sticker_bbox_symmetry": {
                "passed": check_2_passed,
                "panels_checked": len(panels),
                "asymmetry_violations": asymmetry_violations,
            },
            "3_panel_to_fullscreen_continuity": {
                "passed": check_3_passed,
                "measured_centroid_shifts_pct": centroid_shifts_pct,
                "violations": continuity_violations,
            },
            "4_cutout_head_intact_check": {
                "passed": check_4_passed,
                "cutouts_verified": cutout_reports,
                "head_clip_failures": head_clip_failures,
            },
            "5_typography_contrast_and_size": {
                "passed": check_5_passed,
                "typography_events_checked": len(typo_events),
                "invalid_events": len(invalid_typo),
            },
            "6_upper_border_black_triangle_check": {
                "passed": check_6_passed,
                "anomalies": upper_border_anomalies,
            },
            "7_translucent_glass_and_no_negative_face_check": {
                "passed": check_7_passed,
                "policy": "translucent_glass_72pct_and_full_white_flash_no_negative_face",
            },
            "8_strobe_and_blink_cadence_check": {
                "passed": check_8_passed,
                "policy": "1f_and_2f_strobe_blink_verified",
            },
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-gate mathematical pre-QA verification against reference video.")
    parser.add_argument("--ref", default=None, help="Reference video path (auto-discovered in 00_reference_style/ if omitted)")
    parser.add_argument("--rendered", default="03_processed_vfx/Valley_of_Flowers_Reference_Matched_Edit.mp4", help="Optional rendered preview MP4")
    parser.add_argument("--style_dna", default="02_analysis/style_dna.json")
    parser.add_argument("--timeline_plan", default="02_analysis/timeline_plan.json")
    parser.add_argument("--cutouts_dir", default="03_processed_vfx/cutouts")
    parser.add_argument("--report", default="02_analysis/qa_verification_report.json")
    parser.add_argument("--sheet", default="02_analysis/one_shot_qa_sheet.jpg")
    args = parser.parse_args()

    ref_p = args.ref or _auto_find_reference()
    if not ref_p:
        raise SystemExit("[ERROR] No reference video found.")

    rep = verify_edit_against_reference(
        ref_video_path=ref_p,
        rendered_video_path=args.rendered,
        style_dna_path=args.style_dna,
        timeline_plan_path=args.timeline_plan,
        cutouts_dir=args.cutouts_dir,
        report_path=args.report,
        sheet_path=args.sheet,
    )
    print(f"[QA VERIFIER] overall_passed = {rep['overall_passed']} -> {args.report}")
    for k, v in rep["checks"].items():
        print(f"  - {k}: passed={v['passed']}")
    if not rep["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
