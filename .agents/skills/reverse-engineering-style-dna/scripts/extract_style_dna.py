"""General 4-Layer + Visual-Grammar Style DNA Extractor (v2.1).

Deconstructs any reference reel (vertical/horizontal, letterboxed/full-bleed, rapid
beat-montage/lyrical cuts, multi-strip intros, stepped/expanding apertures, white
silhouette mask transitions, kinetic typography, audio onset/RMS/downbeat/sub-beat
grids, optical-flow speed-ramp curves, semantic shot descriptors, and active-region
Lumetri profiles) without any video-specific hardcoding.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import cv2
import librosa
import numpy as np

# Reuse virtual_scopes from balancing-primary-color-and-skins skill (zero duplication)
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCOPES_DIR = _SCRIPTS_DIR.parents[1] / "balancing-primary-color-and-skins" / "scripts"
if str(_SCOPES_DIR) not in sys.path:
    sys.path.insert(0, str(_SCOPES_DIR))

from virtual_scopes import (  # noqa: E402
    analyze_frame_scopes,
    analyze_shot_semantics,
    probe_video_stream_info,
)

try:
    import pytesseract
    from pytesseract import Output as TessOutput
except ImportError:
    pytesseract = None
    TessOutput = None


def _configure_tesseract() -> bool:
    """Locate Tesseract binary on PATH or default Windows installation path."""
    if pytesseract is None:
        return False
    exe = shutil.which("tesseract")
    if not exe:
        win_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if win_default.exists():
            exe = str(win_default)
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
        return True
    return False


def _probe_video_stream(video_path: str) -> dict[str, Any]:
    """Query authoritative stream geometry and rational/VFR-aware FPS via ffprobe."""
    meta = probe_video_stream_info(video_path)
    return {
        "width": meta["width"],
        "height": meta["height"],
        "fps": meta["fps"],
        "is_vfr": meta["is_vfr"],
        "duration_sec": meta["duration"],
    }


def _slugify_stem(stem: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return clean[:48] or "reference"


def extract_and_clone_reference_music(
    ref_video_path: str,
    out_dir: str = "04_assets/music",
    vault_path: str = "04_assets/vault_index.json",
) -> dict[str, Any]:
    """Extract reference reel audio bed to 48kHz WAV and analyze beats/downbeats/sub-beats/RMS/onsets."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir_p / "reference_audio.wav"
    slug_wav = out_dir_p / f"reference_audio_{_slugify_stem(Path(ref_video_path).stem)}.wav"

    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        ref_video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(out_wav),
    ]
    subprocess.run(ffmpeg_cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    if slug_wav != out_wav:
        shutil.copy2(out_wav, slug_wav)

    y, sr = librosa.load(str(out_wav), sr=22050)
    duration = float(len(y) / sr) if sr else 0.0
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.flat[0])
    beat_times = [round(float(t), 3) for t in librosa.frames_to_time(beat_frames, sr=sr)]

    # Fine 50ms RMS envelope + onset envelope for tune-unlock & silence detection
    hop = max(256, int(round(sr * 0.05)))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False)
    strong_onsets = [
        round(float(times[idx]), 3)
        for idx in onset_frames
        if idx < len(onset_env) and onset_env[idx] >= float(np.percentile(onset_env, 75))
    ]

    # Estimate 4/4 bar downbeats (phase 0..3 with highest average onset strength) and half-beat grid
    downbeat_times: list[float] = []
    sub_beat_times: list[float] = []
    if len(beat_times) >= 4:
        phase_scores = []
        for phase in range(4):
            p_beats = beat_times[phase::4]
            p_idxs = [int(np.argmin(np.abs(times - bt))) for bt in p_beats]
            phase_scores.append(float(np.mean([onset_env[i] for i in p_idxs if i < len(onset_env)])))
        best_phase = int(np.argmax(phase_scores))
        downbeat_times = [round(bt, 3) for bt in beat_times[best_phase::4]]
    elif beat_times:
        downbeat_times = [beat_times[0]]

    for i in range(len(beat_times)):
        sub_beat_times.append(beat_times[i])
        if i + 1 < len(beat_times):
            mid_b = round(0.5 * (beat_times[i] + beat_times[i + 1]), 3)
            sub_beat_times.append(mid_b)

    # Detect silence dips and strong audio resurgence points
    silence_windows: list[dict[str, float]] = []
    in_silence = False
    sil_start = 0.0
    for t_val, r_val in zip(times, rms):
        if r_val < 0.004 and not in_silence:
            in_silence = True
            sil_start = float(t_val)
        elif r_val >= 0.004 and in_silence:
            in_silence = False
            if t_val - sil_start >= 0.2:
                silence_windows.append(
                    {"start_sec": round(sil_start, 3), "end_sec": round(float(t_val), 3)}
                )
    if in_silence and duration - sil_start >= 0.2:
        silence_windows.append({"start_sec": round(sil_start, 3), "end_sec": round(duration, 3)})

    # Major energy resurgence after quiet dip (drop / tune unlock cue)
    drop_cues: list[dict[str, float]] = []
    for sw in silence_windows:
        post_idx = np.where((times >= sw["end_sec"]) & (times <= sw["end_sec"] + 2.5))[0]
        if post_idx.size:
            peak_i = int(post_idx[np.argmax(onset_env[post_idx])])
            drop_cues.append(
                {
                    "silence_end_sec": sw["end_sec"],
                    "resurgence_onset_sec": round(float(times[peak_i]), 3),
                    "onset_strength": round(float(onset_env[peak_i]), 2),
                }
            )

    audio_info = {
        "cloned_audio_path": out_wav.as_posix(),
        "archived_audio_path": slug_wav.as_posix(),
        "duration_sec": round(duration, 3),
        "bpm": round(float(tempo), 1),
        "beat_count": len(beat_times),
        "beat_timestamps": beat_times,
        "downbeat_timestamps": downbeat_times,
        "sub_beat_timestamps": sub_beat_times[:300],
        "strong_onset_timestamps": strong_onsets[:150],
        "silence_windows": silence_windows,
        "energy_resurgence_cues": drop_cues,
        "peak_rms": round(float(np.max(rms)) if rms.size else 0.0, 4),
        "mean_rms": round(float(np.mean(rms)) if rms.size else 0.0, 4),
    }

    # Register cloned track cleanly in vault_index.json without corrupting sfx list schema
    vault_p = Path(vault_path)
    vault_data: dict[str, Any] = {"version": "1.0", "sfx": [], "music": []}
    if vault_p.exists():
        try:
            loaded = json.loads(vault_p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                vault_data = loaded
                if not isinstance(vault_data.get("sfx"), list):
                    vault_data["sfx"] = []
        except Exception:
            pass
    music_list = [
        m
        for m in vault_data.get("music", [])
        if isinstance(m, dict) and m.get("name") not in (out_wav.name, slug_wav.name)
    ]
    music_list.insert(
        0,
        {
            "name": out_wav.name,
            "archived_path": slug_wav.as_posix(),
            "path": out_wav.as_posix(),
            "bpm": audio_info["bpm"],
            "duration_sec": audio_info["duration_sec"],
            "source_reference": Path(ref_video_path).name,
        },
    )
    vault_data["music"] = music_list
    vault_p.parent.mkdir(parents=True, exist_ok=True)
    vault_p.write_text(json.dumps(vault_data, indent=2), encoding="utf-8")
    return audio_info


def _largest_true_run(mask_1d: np.ndarray, max_gap: int = 4) -> tuple[int, int] | None:
    """Return (start, end) of longest contiguous True segment allowing small gaps."""
    idx = np.where(mask_1d)[0]
    if idx.size == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur - prev <= max_gap + 1:
            prev = cur
        else:
            runs.append((start, prev + 1))
            start = prev = cur
    runs.append((start, prev + 1))
    return max(runs, key=lambda r: r[1] - r[0])


def _scan_video_stream(video_path: str) -> dict[str, Any]:
    """Single streaming pass collecting spatial activity profiles, optical flow, and compact thumbnails."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open reference video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    thumb_w = 240
    thumb_h = max(80, int(round(height * thumb_w / max(1, width))))

    row_fracs: list[np.ndarray] = []
    col_fracs: list[np.ndarray] = []
    thumbs_bgr: list[np.ndarray] = []
    row_p35_list: list[np.ndarray] = []
    col_p35_list: list[np.ndarray] = []
    white_blob_candidates: list[dict[str, Any]] = []
    frame_means: list[float] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        max_ch = np.max(frame, axis=2)
        nonblack = max_ch > 14
        row_fracs.append(nonblack.mean(axis=1).astype(np.float32))
        col_fracs.append(nonblack.mean(axis=0).astype(np.float32))
        row_p35_list.append(np.percentile(max_ch, 35, axis=1).astype(np.uint8))
        col_p35_list.append(np.percentile(max_ch, 35, axis=0).astype(np.uint8))
        frame_means.append(float(np.mean(max_ch)))

        small = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs_bgr.append(small)

        hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pure_white = ((hsv_s[:, :, 2] >= 238) & (hsv_s[:, :, 1] <= 24)).astype(np.uint8)
        white_pct = float(np.mean(pure_white))
        if white_pct >= 0.025:
            white_blob_candidates.append({"frame": frame_idx, "white_pct_full": white_pct})

        frame_idx += 1

    cap.release()
    n_frames = len(thumbs_bgr)
    if n_frames == 0:
        raise RuntimeError(f"Video yielded 0 frames: {video_path}")

    row_fracs_arr = np.stack(row_fracs)
    col_fracs_arr = np.stack(col_fracs)
    row_p35_arr = np.stack(row_p35_list)
    col_p35_arr = np.stack(col_p35_list)

    frame_cov = (row_fracs_arr > 0.20).mean(axis=1)
    bright_idx = np.where(frame_cov >= max(0.15, float(np.percentile(frame_cov, 65))))[0]
    if bright_idx.size == 0:
        bright_idx = np.arange(n_frames)

    median_row_frac = np.median(row_fracs_arr[bright_idx], axis=0)
    median_col_frac = np.median(col_fracs_arr[bright_idx], axis=0)

    r_run = _largest_true_run(median_row_frac > 0.15, max_gap=max(6, int(0.015 * height)))
    c_run = _largest_true_run(median_col_frac > 0.15, max_gap=max(6, int(0.015 * width)))
    y0_act, y1_act = r_run if r_run else (0, height)
    x0_act, x1_act = c_run if c_run else (0, width)

    if (y1_act - y0_act) >= int(0.95 * height):
        y0_act, y1_act = 0, height
    if (x1_act - x0_act) >= int(0.95 * width):
        x0_act, x1_act = 0, width

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "duration_sec": round(n_frames / fps, 3),
        "thumb_w": thumb_w,
        "thumb_h": thumb_h,
        "active_bounds": [x0_act, y0_act, x1_act - x0_act, y1_act - y0_act],
        "row_fracs": row_fracs_arr,
        "col_fracs": col_fracs_arr,
        "row_p35": row_p35_arr,
        "col_p35": col_p35_arr,
        "frame_means": np.array(frame_means, dtype=np.float32),
        "thumbs_bgr": np.stack(thumbs_bgr),
        "white_candidates": white_blob_candidates,
    }


def _compute_visible_row_spans(scan: dict[str, Any], video_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame (y0, y1) visible video aperture directly from cached thumbnails + row_p35 (zero 2nd decode)."""
    ax, ay, aw, ah = scan["active_bounds"]
    W, H = scan["width"], scan["height"]
    tw, th = scan["thumb_w"], scan["thumb_h"]
    thumbs = scan["thumbs_bgr"]
    n_frames = scan["n_frames"]

    vis_y0 = np.full(n_frames, ay, dtype=np.int32)
    vis_y1 = np.full(n_frames, ay, dtype=np.int32)

    tx0 = max(0, int(round(ax * tw / max(1, W))))
    tx1 = min(tw, int(round((ax + aw) * tw / max(1, W))))
    ty0 = max(0, int(round(ay * th / max(1, H))))
    ty1 = min(th, int(round((ay + ah) * th / max(1, H))))
    tah = max(1, ty1 - ty0)
    taw = max(1, tx1 - tx0)

    edge_w = max(4, int(round(taw * 0.14)))
    lx0, lx1 = tx0, min(tx1, tx0 + edge_w)
    rx0, rx1 = max(tx0, tx1 - edge_w), tx1
    scale_y = float(ah) / float(tah)

    for t in range(n_frames):
        fr = thumbs[t]
        left_band = np.max(fr[ty0:ty1, lx0:lx1], axis=2) > 16
        right_band = np.max(fr[ty0:ty1, rx0:rx1], axis=2) > 16
        edge_active = np.maximum(left_band.mean(axis=1), right_band.mean(axis=1)) > 0.28

        run = _largest_true_run(edge_active, max_gap=max(2, int(0.02 * tah)))
        if run is not None and (run[1] - run[0]) >= max(2, int(round(6.0 / scale_y))):
            vis_y0[t] = ay + int(round(run[0] * scale_y))
            vis_y1[t] = min(ay + ah, ay + int(round(run[1] * scale_y)))
        else:
            full_active = scan["row_p35"][t, ay : ay + ah] > 14
            run_full = _largest_true_run(full_active, max_gap=max(6, int(0.02 * ah)))
            if run_full is not None and (run_full[1] - run_full[0]) >= 6:
                vis_y0[t] = ay + run_full[0]
                vis_y1[t] = ay + run_full[1]

    return vis_y0, vis_y1


def _detect_aperture_and_strips(
    scan: dict[str, Any],
    vis_y0: np.ndarray,
    vis_y1: np.ndarray,
    video_path: str,
    audio_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect staggered multi-strip split screens, divider lines, stepped close plateaus, and slit expansion."""
    fps = scan["fps"]
    n_frames = scan["n_frames"]
    ax, ay, aw, ah = scan["active_bounds"]
    vis_h = (vis_y1 - vis_y0).astype(np.float32)
    vis_ratio = vis_h / max(1.0, float(ah))

    intro_end = min(n_frames, int(round(fps * 4.0)))
    h_scale = scan["height"] / max(1.0, float(ah))
    active_col_fracs = np.clip(scan["col_fracs"][:intro_end, ax : ax + aw] * h_scale, 0.0, 1.0)
    open_frames = [
        t for t in range(intro_end) if vis_ratio[t] >= 0.68 and active_col_fracs[t].mean() >= 0.80
    ]

    strips_info: dict[str, Any] = {
        "detected": False,
        "strip_count": 1,
        "divider_line_px": 0,
        "divider_color": None,
        "strips": [],
    }

    if open_frames:
        min_strip_w = max(36, int(round(aw * 0.08)))
        max_intro_f = open_frames[-1] + 1

        col_enter = np.full(aw, max_intro_f, dtype=np.int32)
        for x in range(aw):
            act_t = np.where(active_col_fracs[:max_intro_f, x] >= 0.45)[0]
            if act_t.size > 0:
                col_enter[x] = int(act_t[0])

        col_enter_med = np.array(
            [int(np.median(col_enter[max(0, x - 2) : min(aw, x + 3)])) for x in range(aw)], dtype=np.int32
        )
        stagger_jumps = np.abs(np.diff(col_enter_med, prepend=col_enter_med[0]))
        stagger_xs: list[int] = []
        for x in range(min_strip_w, aw - min_strip_w):
            if stagger_jumps[x] >= 2 and all(abs(x - sx) >= min_strip_w for sx in stagger_xs):
                stagger_xs.append(x)

        sample_ts = sorted(set([open_frames[0], open_frames[len(open_frames) // 2], open_frames[-1]]))
        cap = cv2.VideoCapture(video_path)
        seam_accum = np.zeros(aw, dtype=np.float32)
        white_col_scores = np.zeros(aw, dtype=np.float32)
        for st in sample_ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            roi = fr[ay : ay + ah, ax : ax + aw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])).mean(axis=0)
            seam_accum += diff_x
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_col = ((hsv[:, :, 2] > 220) & (hsv[:, :, 1] < 35)).mean(axis=0)
            white_col_scores += white_col
        cap.release()
        seam_accum /= max(1, len(sample_ts))
        white_col_scores /= max(1, len(sample_ts))

        interior_grad = seam_accum.copy()
        interior_grad[:min_strip_w] = 0.0
        interior_grad[-min_strip_w:] = 0.0
        med_g = float(np.median(seam_accum[min_strip_w:-min_strip_w])) if aw > 2 * min_strip_w else 5.0

        grad_xs: list[int] = []
        for x in np.argsort(interior_grad)[::-1]:
            if interior_grad[x] < max(8.5, med_g * 1.9):
                break
            if all(abs(int(x) - cx) >= min_strip_w for cx in grad_xs):
                grad_xs.append(int(x))

        candidate_xs = list(stagger_xs)
        for gx in grad_xs:
            if all(abs(gx - cx) >= min_strip_w for cx in candidate_xs):
                candidate_xs.append(gx)
        candidate_xs.sort()

        if 1 <= len(candidate_xs) <= 9:
            bounds = [0] + candidate_xs + [aw]
            strip_entries = []
            enter_frames = []
            for idx in range(len(bounds) - 1):
                sx0, sx1 = ax + bounds[idx], ax + bounds[idx + 1]
                ent_f = int(np.median(col_enter_med[bounds[idx] + 4 : max(bounds[idx] + 5, bounds[idx + 1] - 4)]))
                enter_frames.append(ent_f)
                strip_entries.append(
                    {
                        "strip_index": idx + 1,
                        "x_range_px": [sx0, sx1],
                        "width_pct": round((sx1 - sx0) * 100.0 / aw, 1),
                        "enter_frame": ent_f,
                        "enter_sec": round(ent_f / fps, 3),
                    }
                )

            staggered = len(set(enter_frames)) >= 2 and (max(enter_frames) - min(enter_frames)) >= 3
            if staggered or len(grad_xs) >= 2:
                line_widths = []
                for bx in candidate_xs:
                    w_band = white_col_scores[max(0, bx - 4) : min(aw, bx + 5)]
                    line_widths.append(int(np.sum(w_band >= 0.72)))
                divider_px = int(round(float(np.median(line_widths)))) if line_widths else 0
                strips_info = {
                    "detected": True,
                    "strip_count": len(strip_entries),
                    "staggered_entrance": staggered,
                    "divider_line_px": divider_px,
                    "divider_color": "#FFFFFF" if divider_px > 0 else None,
                    "border_style": f"white_{divider_px}px" if divider_px > 0 else "borderless",
                    "strips": strip_entries,
                }

    # --- 2. Aperture Dynamics (Stepped Beat-Close, Blackout, Expanding Slit + Tune Sync) ---
    plateaus: list[dict[str, Any]] = []
    t = 0
    while t < n_frames:
        if 0.08 <= vis_ratio[t] <= 0.90:
            run_end = t + 1
            anchor_r = vis_ratio[t]
            while run_end < n_frames and abs(vis_ratio[run_end] - anchor_r) <= 0.065 and 0.08 <= vis_ratio[run_end] <= 0.90:
                run_end += 1
            if run_end - t >= 4:
                med_r = float(np.median(vis_ratio[t:run_end]))
                plateaus.append(
                    {
                        "start_frame": int(t),
                        "end_frame": int(run_end - 1),
                        "start_sec": round(t / fps, 3),
                        "end_sec": round((run_end - 1) / fps, 3),
                        "open_pct": round(med_r * 100.0, 1),
                        "close_pct": round((1.0 - med_r) * 100.0, 1),
                        "visible_height_px": int(round(float(np.median(vis_h[t:run_end])))),
                    }
                )
            t = run_end
        else:
            t += 1

    white_cand_set = {c["frame"] for c in scan["white_candidates"] if c["white_pct_full"] >= 0.028}
    blackout_runs: list[dict[str, Any]] = []
    t = 0
    while t < n_frames:
        if vis_h[t] < 4 and t not in white_cand_set:
            e = t + 1
            while e < n_frames and vis_h[e] < 4 and e not in white_cand_set:
                e += 1
            if e - t >= 6:
                blackout_runs.append(
                    {
                        "start_frame": int(t),
                        "end_frame": int(e - 1),
                        "start_sec": round(t / fps, 3),
                        "end_sec": round((e - 1) / fps, 3),
                        "duration_sec": round((e - t) / fps, 3),
                    }
                )
            t = e
        else:
            t += 1

    aperture_unlock: dict[str, Any] | None = None
    mid_blackouts = [b for b in blackout_runs if b["start_sec"] > 0.5 and b["end_sec"] < (scan["duration_sec"] - 0.8)]
    if mid_blackouts:
        bo = mid_blackouts[0]
        unlock_f = bo["end_frame"] + 1
        full_open_f = unlock_f
        while full_open_f < n_frames and vis_ratio[full_open_f] < 0.90:
            full_open_f += 1
        unlock_sec = round(unlock_f / fps, 3)

        nearest_cue = None
        min_dt = 999.0
        for cue in audio_info.get("energy_resurgence_cues", []):
            dt = abs(cue["resurgence_onset_sec"] - unlock_sec)
            if dt < min_dt:
                min_dt = dt
                nearest_cue = cue

        aperture_unlock = {
            "blackout_start_sec": bo["start_sec"],
            "slit_open_frame": int(unlock_f),
            "slit_open_sec": unlock_sec,
            "full_open_frame": int(min(full_open_f, n_frames - 1)),
            "full_open_sec": round(min(full_open_f, n_frames - 1) / fps, 3),
            "scaling_policy": "crop_no_stretch",
            "audio_unlock_sync": nearest_cue if min_dt <= 1.2 else None,
        }

    crop_keyframes: list[dict[str, Any]] = []
    closing_steps = [p for p in plateaus if not mid_blackouts or p["end_frame"] <= mid_blackouts[0]["start_frame"] + 2]
    has_aperture = bool(closing_steps or mid_blackouts)
    if has_aperture:
        sample_step = max(2, int(round(fps * 0.2)))
        prev_r = -1.0
        for t in range(0, n_frames, sample_step):
            r = round(float(vis_ratio[t]), 3)
            if abs(r - prev_r) >= 0.03 or t == 0:
                crop_keyframes.append(
                    {
                        "time_sec": round(t / fps, 3),
                        "open_pct": round(r * 100.0, 1),
                        "close_pct": round((1.0 - r) * 100.0, 1),
                    }
                )
                prev_r = r

    aperture_info = {
        "detected": has_aperture,
        "scaling_policy": "crop_no_stretch",
        "stepped_close_levels": closing_steps,
        "blackout_holds": mid_blackouts,
        "slit_unlock": aperture_unlock,
        "crop_keyframes": crop_keyframes,
    }
    return strips_info, aperture_info


def _detect_cuts_and_masks(
    scan: dict[str, Any],
    vis_y0: np.ndarray,
    vis_y1: np.ndarray,
    aperture_info: dict[str, Any],
    audio_info: dict[str, Any],
    video_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Detect all cuts (down to ~0.13s beat cuts), white masks, speed-ramp curves, semantic tags, and outro fade."""
    fps = scan["fps"]
    n_frames = scan["n_frames"]
    ax, ay, aw, ah = scan["active_bounds"]
    W, H = scan["width"], scan["height"]
    tw, th = scan["thumb_w"], scan["thumb_h"]
    thumbs = scan["thumbs_bgr"]

    tx0 = max(0, int(round(ax * tw / W)))
    tx1 = min(tw, int(round((ax + aw) * tw / W)))
    margin_w = max(4, int(round((tx1 - tx0) * 0.16)))

    hsv_thumbs = [cv2.cvtColor(fr, cv2.COLOR_BGR2HSV) for fr in thumbs]
    gray_thumbs = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in thumbs]
    slit_mad = np.zeros(n_frames, dtype=np.float32)
    slit_hist = np.zeros(n_frames, dtype=np.float32)
    flow_mags = np.zeros(n_frames, dtype=np.float32)

    prev_hist = None
    for t in range(n_frames):
        ty0 = max(0, int(round(vis_y0[t] * th / H)))
        ty1 = min(th, int(round(vis_y1[t] * th / H)))
        if ty1 - ty0 < 4:
            prev_hist = None
            continue
        if (vis_y1[t] - vis_y0[t]) < 0.72 * ah and (tx1 - tx0) > 2 * margin_w + 4:
            roi_t = np.concatenate([thumbs[t][ty0:ty1, tx0 : tx0 + margin_w], thumbs[t][ty0:ty1, tx1 - margin_w : tx1]], axis=1)
            hsv_t = np.concatenate([hsv_thumbs[t][ty0:ty1, tx0 : tx0 + margin_w], hsv_thumbs[t][ty0:ty1, tx1 - margin_w : tx1]], axis=1)
        else:
            roi_t = thumbs[t][ty0:ty1, tx0:tx1]
            hsv_t = hsv_thumbs[t][ty0:ty1, tx0:tx1]

        hist = cv2.calcHist([hsv_t], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)

        if t > 0 and (vis_y1[t - 1] - vis_y0[t - 1]) >= 6:
            py0 = max(ty0, int(round(vis_y0[t - 1] * th / H)))
            py1 = min(ty1, int(round(vis_y1[t - 1] * th / H)))
            if py1 - py0 >= 4:
                if (vis_y1[t] - vis_y0[t]) < 0.72 * ah and (tx1 - tx0) > 2 * margin_w + 4:
                    cur_ov = np.concatenate([thumbs[t][py0:py1, tx0 : tx0 + margin_w], thumbs[t][py0:py1, tx1 - margin_w : tx1]], axis=1)
                    pre_ov = np.concatenate([thumbs[t - 1][py0:py1, tx0 : tx0 + margin_w], thumbs[t - 1][py0:py1, tx1 - margin_w : tx1]], axis=1)
                else:
                    cur_ov = thumbs[t][py0:py1, tx0:tx1]
                    pre_ov = thumbs[t - 1][py0:py1, tx0:tx1]
                slit_mad[t] = float(np.mean(np.abs(cur_ov.astype(np.float32) - pre_ov.astype(np.float32))))
            if prev_hist is not None:
                slit_hist[t] = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_hist = hist

    # --- Verify silhouette mask bursts at full native resolution ---
    mask_events: list[dict[str, Any]] = []
    cand_frames = [c["frame"] for c in scan["white_candidates"]]
    cand_runs: list[list[int]] = []
    for cf in cand_frames:
        if not cand_runs or cf - cand_runs[-1][-1] > 2:
            cand_runs.append([cf])
        else:
            cand_runs[-1].append(cf)

    cap = cv2.VideoCapture(video_path)
    mask_frame_set: set[int] = set()
    for run in cand_runs:
        if len(run) > int(round(fps * 1.8)):
            continue
        valid_frames = []
        bg_corners: list[float] = []
        white_pcts: list[float] = []
        for f_idx in range(max(0, run[0] - 1), min(n_frames, run[-1] + 2)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            roi = fr[ay : ay + ah, ax : ax + aw]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            pw = ((hsv[:, :, 2] >= 238) & (hsv[:, :, 1] <= 24)).astype(np.uint8)
            pw_area = float(np.mean(pw))
            if pw_area < 0.03:
                continue
            num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(pw, connectivity=8)
            if num_lbl <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            best_i = int(np.argmax(areas)) + 1
            best_area_pct = float(stats[best_i, cv2.CC_STAT_AREA]) / max(1, roi.shape[0] * roi.shape[1])
            best_h_pct = float(stats[best_i, cv2.CC_STAT_HEIGHT]) / max(1, roi.shape[0])
            if best_area_pct < 0.032 or best_h_pct < 0.28:
                continue

            # Reject multi-letter giant impact words (e.g. 'BUT' which has 3 equal-height letter blobs) from silhouette masks
            tall_blobs = [
                k for k in range(1, num_lbl)
                if (stats[k, cv2.CC_STAT_HEIGHT] / max(1, roi.shape[0])) >= 0.24
                and (stats[k, cv2.CC_STAT_AREA] / max(1, roi.shape[0] * roi.shape[1])) >= 0.015
            ]
            if len(tall_blobs) >= 2:
                top_ys = [stats[k, cv2.CC_STAT_TOP] for k in tall_blobs]
                bot_ys = [stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] for k in tall_blobs]
                if (max(top_ys) - min(top_ys)) <= 0.04 * roi.shape[0] and (max(bot_ys) - min(bot_ys)) <= 0.04 * roi.shape[0]:
                    continue

            bx, by, bw, bh = [int(stats[best_i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
            crop_pw = pw[by : by + bh, bx : bx + bw]
            eroded = cv2.erode(crop_pw, np.ones((5, 5), np.uint8))
            if int(np.sum(eroded)) < 150:
                continue
            blob_std = float(np.std(cv2.cvtColor(roi[by : by + bh, bx : bx + bw], cv2.COLOR_BGR2GRAY)[eroded > 0]))
            if blob_std > 9.0:
                continue

            c_sz = max(16, int(round(min(aw, ah) * 0.08)))
            corners = [
                roi[:c_sz, :c_sz],
                roi[:c_sz, -c_sz:],
                roi[-c_sz:, :c_sz],
                roi[-c_sz:, -c_sz:],
            ]
            bg_mean = float(np.mean([np.mean(c) for c in corners]))
            valid_frames.append(f_idx)
            bg_corners.append(bg_mean)
            white_pcts.append(round(best_area_pct * 100.0, 2))

        if len(valid_frames) >= 2:
            s_f, e_f = valid_frames[0], valid_frames[-1]
            mean_bg = float(np.mean(bg_corners))
            mode = "mask_on_black" if mean_bg < 12.0 else "mask_over_live_bg"
            mask_events.append(
                {
                    "type": mode,
                    "start_frame": s_f,
                    "end_frame": e_f,
                    "start_sec": round(s_f / fps, 3),
                    "end_sec": round((e_f + 1) / fps, 3),
                    "duration_frames": e_f - s_f + 1,
                    "duration_sec": round((e_f - s_f + 1) / fps, 3),
                    "mean_corner_bg_luma": round(mean_bg, 1),
                    "mask_area_pct": round(float(np.mean(white_pcts)), 2),
                    "require_person_subject": True,
                }
            )
            for mf in range(max(0, s_f - 1), min(n_frames, e_f + 3)):
                mask_frame_set.add(mf)
    cap.release()

    # --- Peak-pick editorial cuts (down to 0.13s rapid beat-montage spacing) ---
    min_gap_frames = max(4, int(round(fps * 0.133)))
    blackout_set = set()
    for bo in aperture_info.get("blackout_holds", []):
        for bf in range(bo["start_frame"], bo["end_frame"] + 1):
            blackout_set.add(bf)

    raw_candidates: list[tuple[int, float]] = []
    for t in range(2, n_frames - 2):
        if t in blackout_set or t in mask_frame_set:
            continue
        if (vis_y1[t] - vis_y0[t]) < 12:
            continue
        h_d = float(slit_hist[t])
        m_d = float(slit_mad[t])
        in_slit = (vis_y1[t] - vis_y0[t]) < 0.85 * ah
        if (in_slit and h_d >= 0.25 and m_d >= 12.0) or (not in_slit and ((h_d >= 0.29 and m_d >= 12.0) or m_d >= 28.0)):
            score = h_d * 2.0 + m_d / 35.0
            raw_candidates.append((t, score))

    selected_cuts: list[int] = []
    for t, score in sorted(raw_candidates, key=lambda x: x[1], reverse=True):
        if all(abs(t - sc) >= min_gap_frames for sc in selected_cuts):
            selected_cuts.append(t)

    for me in mask_events:
        split_pt = int(me["start_frame"])
        if all(abs(split_pt - sc) >= min_gap_frames for sc in selected_cuts) and 2 <= split_pt < n_frames - 2:
            selected_cuts.append(split_pt)
    selected_cuts.sort()

    # Detect outro fade-to-black tail
    tail_start = max(0, int(round(n_frames * 0.82)))
    end_mean = float(np.mean(scan["frame_means"][max(0, n_frames - 3) : n_frames]))
    outro_info: dict[str, Any] = {"type": "hard_cut", "fade_duration_sec": 0.0}
    if end_mean < 12.0:
        fade_f = n_frames - 1
        while fade_f > tail_start and scan["frame_means"][fade_f - 1] <= scan["frame_means"][fade_f] + 1.5 and scan["frame_means"][fade_f] < 80.0:
            fade_f -= 1
        if n_frames - fade_f >= 4:
            outro_info = {
                "type": "fade_to_black",
                "start_frame": int(fade_f),
                "start_sec": round(fade_f / fps, 3),
                "end_sec": round(n_frames / fps, 3),
                "fade_duration_sec": round((n_frames - fade_f) / fps, 3),
            }

    # Beat grid reference for beat-lock diagnostics
    all_musical_cues = sorted(
        set(audio_info.get("beat_timestamps", []) + audio_info.get("strong_onset_timestamps", []))
    )

    cut_boundaries = [0] + selected_cuts + [n_frames]
    segments: list[dict[str, Any]] = []
    seg_id = 1
    for i in range(len(cut_boundaries) - 1):
        sf, ef = cut_boundaries[i], cut_boundaries[i + 1]
        while sf < ef and sf in blackout_set:
            sf += 1
        while ef > sf and (ef - 1) in blackout_set:
            ef -= 1
        if ef - sf < 2:
            continue
        dur = round((ef - sf) / fps, 3)
        start_sec = round(sf / fps, 3)
        end_sec = round(ef / fps, 3)
        in_slit_seg = bool(np.median(vis_y1[sf:ef] - vis_y0[sf:ef]) < 0.85 * ah)
        seg_masks = [m for m in mask_events if m["start_frame"] < ef and m["end_frame"] >= sf]

        # Beat alignment check
        beat_offset_ms = None
        is_beat_locked = False
        if all_musical_cues:
            nearest_b = min(all_musical_cues, key=lambda b: abs(b - start_sec))
            beat_offset_ms = int(round((start_sec - nearest_b) * 1000.0))
            is_beat_locked = abs(beat_offset_ms) <= 90

        # Speed-ramp curve detection within segment (excluding cut boundary spike)
        inner_flow = slit_mad[min(ef - 1, sf + 1) : max(sf + 2, ef - 1)]
        speed_ramp_info: dict[str, Any] = {"detected": False, "curve_type": "constant_100pct", "speed_points_pct": [100, 100, 100]}
        if inner_flow.size >= 6:
            third = max(1, inner_flow.size // 3)
            f_start = float(np.mean(inner_flow[:third]))
            f_mid = float(np.mean(inner_flow[third : 2 * third]))
            f_end = float(np.mean(inner_flow[2 * third :]))
            if max(f_start, f_end) >= max(1.6, f_mid * 1.85):
                s_in = int(np.clip(round(100 * (f_start / max(0.4, f_mid))), 100, 320))
                s_out = int(np.clip(round(100 * (f_end / max(0.4, f_mid))), 100, 320))
                speed_ramp_info = {
                    "detected": True,
                    "curve_type": "whip_to_slowmo_valley",
                    "speed_points_pct": [s_in, 50, s_out],
                }
            elif f_mid >= max(1.8, max(f_start, f_end) * 1.75):
                speed_ramp_info = {
                    "detected": True,
                    "curve_type": "mid_pulse_accelerate",
                    "speed_points_pct": [80, 220, 80],
                }

        # Semantic shot scale, camera motion & 32-D visual fingerprint from segment thumbnails
        seg_thumb_idxs = np.linspace(sf, ef - 1, min(5, ef - sf), dtype=int)
        seg_thumbs = [thumbs[int(k)] for k in seg_thumb_idxs]
        semantics = analyze_shot_semantics(seg_thumbs, auto_crop_active=True)

        segments.append(
            {
                "segment_id": seg_id,
                "start_frame": int(sf),
                "end_frame": int(ef),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur,
                "in_expanding_slit": in_slit_seg,
                "mask_transitions": [m["type"] for m in seg_masks],
                "is_beat_locked": is_beat_locked,
                "nearest_beat_offset_ms": beat_offset_ms,
                "speed_ramp": speed_ramp_info,
                "shot_scale": semantics["shot_scale"],
                "camera_motion": semantics["camera_motion"],
                "motion_magnitude": semantics["motion_magnitude"],
                "visual_fingerprint": semantics["visual_fingerprint"],
            }
        )
        seg_id += 1

    return segments, mask_events, outro_info


def _detect_geometric_masks_and_panels(
    scan: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect rectangular inset boxes, symmetrical 2-panel splits, 3-pill bars, circle iris reveals, and sticker cards.
    Enforces Symmetry & Continuity Invariant (`continuous_1to1_screen_space: True`) when a panel precedes a full-screen reveal.
    """
    thumbs = scan["thumbs_bgr"]
    th, tw = thumbs.shape[1:3]
    panel_events: list[dict[str, Any]] = []

    for i, seg in enumerate(segments):
        sf, ef = seg["start_frame"], seg["end_frame"]
        if ef - sf < 3 or sf < 2:
            continue
        mid_f = (sf + ef) // 2
        ref_f = max(0, sf - 2)
        diff = np.mean(np.abs(thumbs[mid_f].astype(np.float32) - thumbs[ref_f].astype(np.float32)), axis=2)
        active = (diff > 18.0).astype(np.uint8)
        cov = float(np.mean(active))
        if not (0.12 <= cov <= 0.85):
            continue

        col_act = active.mean(axis=0) > 0.35
        row_act = active.mean(axis=1) > 0.35
        xs = np.where(col_act)[0]
        ys = np.where(row_act)[0]
        if xs.size < 4 or ys.size < 4:
            continue

        x0_pct = round(float(xs[0]) * 100.0 / tw, 2)
        x1_pct = round(float(xs[-1] + 1) * 100.0 / tw, 2)
        y0_pct = round(float(ys[0]) * 100.0 / th, 2)
        y1_pct = round(float(ys[-1] + 1) * 100.0 / th, 2)

        # True geometric panel/inset/sticker must be strictly inset on X or Y with preserved background margins outside the box!
        inset_x = x0_pct >= 6.0 and x1_pct <= 94.0
        inset_y = y0_pct >= 6.0 and y1_pct <= 94.0
        if not (inset_x or inset_y):
            continue

        bx0, bx1 = int(xs[0]), int(xs[-1] + 1)
        by0, by1 = int(ys[0]), int(ys[-1] + 1)
        outer_mask = np.ones_like(active, dtype=bool)
        outer_mask[by0:by1, bx0:bx1] = False
        outer_change = float(np.mean(active[outer_mask])) if np.any(outer_mask) else 1.0
        if outer_change > 0.22:
            continue

        # Classify corner shape: circle_iris vs stadium_pill vs sharp_rect
        sub_mask = active[by0:by1, bx0:bx1]
        box_fill = float(np.mean(sub_mask)) if sub_mask.size else 1.0
        ar_box = (bx1 - bx0) / max(1.0, float(by1 - by0))
        if 0.70 <= box_fill <= 0.84 and 0.75 <= ar_box <= 1.35:
            shape_type = "circle_iris"
        elif 0.84 < box_fill <= 0.93 and ar_box > 1.8:
            shape_type = "stadium_pill"
        else:
            shape_type = "sharp_rect"

        # Symmetry Lock: if left & right outer margins are within 3.5%, lock them symmetrically
        left_m, right_m = x0_pct, 100.0 - x1_pct
        top_m, bot_m = y0_pct, 100.0 - y1_pct
        is_sym_x = abs(left_m - right_m) <= 3.5
        is_sym_y = abs(top_m - bot_m) <= 3.5
        if is_sym_x:
            avg_mx = round(0.5 * (left_m + right_m), 2)
            x0_pct, x1_pct = avg_mx, round(100.0 - avg_mx, 2)
        if is_sym_y:
            avg_my = round(0.5 * (top_m + bot_m), 2)
            y0_pct, y1_pct = avg_my, round(100.0 - avg_my, 2)

        # Check if next segment is a continuous reveal of the same shot inside [by0:by1, bx0:bx1]
        cont_next = False
        if i + 1 < len(segments):
            nxt_mid = (segments[i + 1]["start_frame"] + segments[i + 1]["end_frame"]) // 2
            roi_cur = thumbs[mid_f][by0:by1, bx0:bx1].astype(np.float32)
            roi_nxt = thumbs[nxt_mid][by0:by1, bx0:bx1].astype(np.float32)
            mad_inner = float(np.mean(np.abs(roi_cur - roi_nxt))) if roi_cur.size else 999.0
            v_cur = np.array(seg.get("visual_fingerprint", []), dtype=np.float32)
            v_nxt = np.array(segments[i + 1].get("visual_fingerprint", []), dtype=np.float32)
            sim = float(np.dot(v_cur, v_nxt) / max(1e-6, np.linalg.norm(v_cur) * np.linalg.norm(v_nxt))) if (v_cur.size == 32 and v_nxt.size == 32) else 0.0
            cont_next = bool(sim >= 0.86 and mad_inner <= 24.0)

        panel_events.append(
            {
                "segment_id": seg["segment_id"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "shape_type": shape_type,
                "bbox_pct_xyxy": [x0_pct, y0_pct, x1_pct, y1_pct],
                "premiere_crop_pct": {
                    "left": x0_pct,
                    "right": round(100.0 - x1_pct, 2),
                    "top": y0_pct,
                    "bottom": round(100.0 - y1_pct, 2),
                },
                "symmetric_locked": bool(is_sym_x and is_sym_y),
                "continuous_1to1_screen_space": cont_next,
                "continuous_with_next_slot": cont_next,
            }
        )
    return panel_events


_detect_geometric_panels_and_stickers = _detect_geometric_masks_and_panels


def _detect_strobe_and_blink_schedules(
    scan: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect 1-frame and 2-frame alternating strobe/blink cycles across full frame and center text ROI.
    Catches opening mountain cutout strobes (f002-f011), 00:07:21 inset white flash cutouts (f229-f233),
    and 2-frame translucent text blink strobes ('SO' f470-f515) without sparse-sampling blindness.
    """
    thumbs = scan["thumbs_bgr"]
    fps = float(scan["fps"])
    n_frames, th, tw = thumbs.shape[:3]
    if n_frames < 6:
        return []

    gray = np.mean(thumbs.astype(np.float32), axis=3)
    cx0, cx1 = int(tw * 0.32), int(tw * 0.68)
    cy0, cy1 = int(th * 0.25), int(th * 0.75)
    center_gray = gray[:, cy0:cy1, cx0:cx1]

    full_diff = np.mean(gray[1:] - gray[:-1], axis=(1, 2))
    center_mad = np.mean(np.abs(center_gray[1:] - center_gray[:-1]), axis=(1, 2))
    outer_mad = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))

    strobe_events: list[dict[str, Any]] = []

    # 1. Detect 1-frame or 2-frame full/inset flash strobes (high alternating luminance delta)
    f = 1
    while f < n_frames - 3:
        if abs(full_diff[f - 1]) >= 14.0:
            # Check for alternating sign within 1..2 frames
            run_end = f
            flips = 0
            last_sign = 1 if full_diff[f - 1] > 0 else -1
            for k in range(f, min(n_frames - 1, f + 24)):
                if abs(full_diff[k - 1]) >= 10.0:
                    cur_sign = 1 if full_diff[k - 1] > 0 else -1
                    if cur_sign != last_sign:
                        flips += 1
                        last_sign = cur_sign
                        run_end = k + 1
                elif k - run_end > 2:
                    break
            if flips >= 2:
                strobe_events.append(
                    {
                        "type": "Clean_White_Or_Cutout_Strobe",
                        "region": "full_or_inset_subject",
                        "start_frame": int(f),
                        "end_frame": int(run_end),
                        "start_sec": round(f / fps, 3),
                        "end_sec": round(run_end / fps, 3),
                        "alternating_flips": int(flips),
                        "flash_policy": "full_white_flash_no_negative_face",
                    }
                )
                f = run_end + 1
                continue
        f += 1

    # 2. Detect 2-frame center text blink strobes (e.g. 'SO' f470..f515 where center ROI blinks every 2 frames while outer bg is smooth)
    for seg in segments:
        sf, ef = int(seg["start_frame"]), int(seg["end_frame"])
        if ef - sf < 16:
            continue
        seg_c_mad = center_mad[sf : ef - 1]
        seg_o_mad = outer_mad[sf : ef - 1]
        spikes = [sf + i + 1 for i in range(len(seg_c_mad)) if seg_c_mad[i] >= max(3.2, seg_o_mad[i] * 1.45)]
        if len(spikes) >= 6:
            gaps = [spikes[i + 1] - spikes[i] for i in range(len(spikes) - 1)]
            two_frame_gaps = sum(1 for g in gaps if g in (2, 3))
            if two_frame_gaps >= 4:
                strobe_events.append(
                    {
                        "type": "Center_Text_Blink_Strobe",
                        "region": "center_text_roi",
                        "segment_id": int(seg["segment_id"]),
                        "start_frame": int(spikes[0]),
                        "end_frame": int(spikes[-1] + 2),
                        "start_sec": round(spikes[0] / fps, 3),
                        "end_sec": round((spikes[-1] + 2) / fps, 3),
                        "period_frames_on_off": [2, 2],
                        "blend_policy": "translucent_glass_or_difference_center",
                    }
                )

    return strobe_events


def _detect_intra_shot_collage_and_splits(
    scan: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan every frame inside segments >= 1.2s for rectangular PiP border collage tiles popping in/out."""
    thumbs = scan["thumbs_bgr"]
    fps = float(scan["fps"])
    n_frames, th, tw = thumbs.shape[:3]
    collage_tiles: list[dict[str, Any]] = []

    for seg in segments:
        sf, ef = int(seg["start_frame"]), int(seg["end_frame"])
        if (ef - sf) < int(round(fps * 1.2)):
            continue
        for f in range(sf + 2, ef - 2):
            diff = np.mean(np.abs(thumbs[f].astype(np.float32) - thumbs[f - 1].astype(np.float32)), axis=2)
            active = (diff > 16.0).astype(np.uint8)
            area_frac = float(np.mean(active))
            if not (0.035 <= area_frac <= 0.42):
                continue
            n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
            if n_lbl <= 1:
                continue
            best_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bx, by, bw, bh, b_area = [int(stats[best_i, k]) for k in range(5)]
            rect_area = max(1, bw * bh)
            if (b_area / float(rect_area)) >= 0.55 and (rect_area / float(tw * th)) >= 0.035:
                collage_tiles.append(
                    {
                        "segment_id": int(seg["segment_id"]),
                        "event_frame": int(f),
                        "event_sec": round(f / fps, 3),
                        "bbox_xywh_pct": [
                            round(bx * 100.0 / tw, 1),
                            round(by * 100.0 / th, 1),
                            round(bw * 100.0 / tw, 1),
                            round(bh * 100.0 / th, 1),
                        ],
                    }
                )
    return collage_tiles


def _generate_postmortem_contact_sheet(
    ref_video_path: str,
    segments: list[dict[str, Any]],
    out_sheet: str = "02_analysis/ref_postmortem_sheet.jpg",
) -> str:
    """Emit a single compact contact sheet of all reference segment midpoints with MM:SS:FF (fNNN) burned in
    so an isolated subagent can complete a 1-turn Reference Post-Mortem without bloating the main context.
    """
    cap = cv2.VideoCapture(str(ref_video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    tw, th = 280, 158
    thumbs = []
    for seg in segments[:48]:
        mf = (int(seg["start_frame"]) + int(seg["end_frame"])) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mf)
        ok, fr = cap.read()
        im = cv2.resize(fr, (tw, th)) if (ok and fr is not None) else np.zeros((th, tw, 3), dtype=np.uint8)
        sec = int(mf // round(fps))
        rem = int(mf % round(fps))
        cv2.rectangle(im, (0, 0), (195, 22), (0, 0, 0), -1)
        cv2.putText(
            im,
            f"S{seg['segment_id']:02d} {sec:02d}:{rem:02d} (f{mf})",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(im)
    cap.release()
    if not thumbs:
        return out_sheet
    cols = 6
    while len(thumbs) % cols != 0:
        thumbs.append(np.zeros((th, tw, 3), dtype=np.uint8))
    rows = [np.hstack(thumbs[i * cols : (i + 1) * cols]) for i in range(len(thumbs) // cols)]
    out_p = Path(out_sheet)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out_p.as_posix()


def _ocr_bright_overlay(
    frame_bgr: np.ndarray,
    ref_bg_bgr: np.ndarray | None = None,
) -> tuple[str, list[dict[str, Any]], tuple[int, int, int, int] | None, str, float]:
    """Extract solid OR Difference/Exclusion-blended overlay text, returning (text, words, bbox, blend_mode, stroke_ratio)."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw_white = ((hsv[:, :, 2] >= 205) & (hsv[:, :, 1] <= 50)).astype(np.uint8)

    # Detect Difference/Exclusion blend-mode glyphs inside the center title band [y: 22%..78%, x: 12%..88%]
    # by checking pixels whose RGB values approximate (255 - local_spatial_background) with crisp glyph edges
    diff_glyph_mask = np.zeros_like(raw_white, dtype=np.uint8)
    cy0, cy1 = int(h * 0.22), int(h * 0.78)
    cx0, cx1 = int(w * 0.12), int(w * 0.88)
    center_roi = frame_bgr[cy0:cy1, cx0:cx1]
    local_bg = cv2.GaussianBlur(center_roi, (31, 31), 0).astype(np.float32)
    roi_f = center_roi.astype(np.float32)
    distinct_from_bg = np.mean(np.abs(roi_f - local_bg), axis=2) > 36.0
    matches_invert = np.mean(np.abs(roi_f - (255.0 - local_bg)), axis=2) < 38.0
    diff_glyph_mask[cy0:cy1, cx0:cx1] = (distinct_from_bg & matches_invert).astype(np.uint8)

    combined_cand = cv2.bitwise_or(raw_white, diff_glyph_mask)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(combined_cand, connectivity=8)
    glyph_mask = np.zeros_like(raw_white, dtype=np.uint8)
    max_glyph_h = max(18, int(round(h * 0.36)))
    max_glyph_w = max(20, int(round(w * 0.28)))
    stroke_ratios: list[float] = []

    for lbl in range(1, num_lbl):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if 6 <= ch <= max_glyph_h and 3 <= cw <= max_glyph_w and 14 <= area <= int(max_glyph_h * max_glyph_w * 0.80):
            glyph_mask[labels == lbl] = 255
            stroke_ratios.append(float(area) / max(1.0, float(cw * ch)))

    if int(np.count_nonzero(glyph_mask)) < 60:
        return "", [], None, "Normal", 0.0

    # Determine whether detected glyphs came primarily from Difference inversion or Normal solid white
    diff_overlap = int(np.count_nonzero((glyph_mask > 0) & (diff_glyph_mask > 0)))
    total_glyph_px = max(1, int(np.count_nonzero(glyph_mask > 0)))
    blend_mode = "Difference" if (diff_overlap / float(total_glyph_px)) >= 0.35 else "Normal"
    med_stroke_ratio = round(float(np.median(stroke_ratios)) if stroke_ratios else 0.35, 3)

    ys_nz, xs_nz = np.where(glyph_mask > 0)
    if ys_nz.size == 0:
        return "", [], None, "Normal", 0.0
    gx0 = max(0, int(xs_nz.min()) - 14)
    gy0 = max(0, int(ys_nz.min()) - 14)
    gx1 = min(w, int(xs_nz.max()) + 15)
    gy1 = min(h, int(ys_nz.max()) + 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.dilate(glyph_mask[gy0:gy1, gx0:gx1], kernel, iterations=1)
    inv = cv2.bitwise_not(clean)
    cw_crop, ch_crop = gx1 - gx0, gy1 - gy0
    scale_f = 2 if max(cw_crop, ch_crop) <= 640 else 1
    up = cv2.resize(inv, (cw_crop * scale_f, ch_crop * scale_f), interpolation=cv2.INTER_CUBIC) if scale_f > 1 else inv
    data = pytesseract.image_to_data(up, output_type=TessOutput.DICT, config="--psm 6")

    lines: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for i, raw_w in enumerate(data["text"]):
        txt = raw_w.strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1.0
        if conf < 48.0 or not txt or not any(ch.isalnum() for ch in txt):
            continue
        bx = gx0 + int(data["left"][i] // scale_f)
        by = gy0 + int(data["top"][i] // scale_f)
        bw = int(data["width"][i] // scale_f)
        bh = int(data["height"][i] // scale_f)
        if bh > max_glyph_h * 1.2:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines.setdefault(key, []).append({"text": txt, "conf": conf, "box": [bx, by, bw, bh]})

    valid_words: list[dict[str, Any]] = []
    for lw in lines.values():
        alpha_chars = sum(sum(c.isalpha() for c in wd["text"]) for wd in lw)
        mean_conf = sum(wd["conf"] for wd in lw) / len(lw)
        multi_char = sum(1 for wd in lw if len(wd["text"]) >= 2 and any(c.isalpha() for c in wd["text"]))
        if alpha_chars >= 2 and multi_char >= 1 and mean_conf >= 52.0:
            valid_words.extend(lw)

    if not valid_words:
        return "", [], None, blend_mode, med_stroke_ratio
    full_txt = " ".join(wd["text"] for wd in valid_words)
    xs = [wd["box"][0] for wd in valid_words]
    ys = [wd["box"][1] for wd in valid_words]
    x2s = [wd["box"][0] + wd["box"][2] for wd in valid_words]
    y2s = [wd["box"][1] + wd["box"][3] for wd in valid_words]
    bbox = (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))
    return full_txt, valid_words, bbox, blend_mode, med_stroke_ratio


def _detect_typography(scan: dict[str, Any], video_path: str) -> dict[str, Any]:
    """Detect overlay captions/titles, spatial box, casing, serif heuristics, and reveal animation."""
    if not _configure_tesseract():
        return {"detected": False, "reason": "tesseract_unavailable", "events": []}

    fps = scan["fps"]
    n_frames = scan["n_frames"]
    W, H = scan["width"], scan["height"]
    sample_step = max(4, int(round(fps * 0.40)))
    mask_cand_set = {c["frame"] for c in scan["white_candidates"] if c["white_pct_full"] >= 0.03}

    cap = cv2.VideoCapture(video_path)
    raw_samples: list[dict[str, Any]] = []
    last_ocr_res: dict[str, Any] | None = None
    last_crop_bin: np.ndarray | None = None
    prev_full_fr: np.ndarray | None = None

    for t in range(0, n_frames, sample_step):
        if t in mask_cand_set:
            last_ocr_res = None
            last_crop_bin = None
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue

        if last_ocr_res is not None and last_crop_bin is not None:
            bx, by, bw, bh = last_ocr_res["bbox"]
            hsv_b = cv2.cvtColor(fr[by : by + bh, bx : bx + bw], cv2.COLOR_BGR2HSV)
            cur_bin = ((hsv_b[:, :, 2] >= 205) & (hsv_b[:, :, 1] <= 50)).astype(np.uint8)
            inter = np.logical_and(cur_bin > 0, last_crop_bin > 0).sum()
            union = max(1, np.logical_or(cur_bin > 0, last_crop_bin > 0).sum())
            if inter / union >= 0.88:
                raw_samples.append({**last_ocr_res, "frame": t})
                prev_full_fr = fr
                continue

        txt, words, bbox, blend_mode, stroke_ratio = _ocr_bright_overlay(fr, ref_bg_bgr=prev_full_fr)
        prev_full_fr = fr
        if len(txt) >= 2 and sum(w["conf"] for w in words) / max(1, len(words)) >= 54.0 and bbox is not None:
            entry = {
                "frame": t,
                "text": txt,
                "words": words,
                "bbox": bbox,
                "blend_mode": blend_mode,
                "stroke_to_height_ratio": stroke_ratio,
            }
            raw_samples.append(entry)
            last_ocr_res = entry
            bx, by, bw, bh = bbox
            hsv_b = cv2.cvtColor(fr[by : by + bh, bx : bx + bw], cv2.COLOR_BGR2HSV)
            last_crop_bin = ((hsv_b[:, :, 2] >= 205) & (hsv_b[:, :, 1] <= 50)).astype(np.uint8)
        else:
            last_ocr_res = None
            last_crop_bin = None

    events: list[dict[str, Any]] = []
    cur_group: list[dict[str, Any]] = []
    for smp in raw_samples:
        if not cur_group:
            cur_group = [smp]
            continue
        gap_sec = (smp["frame"] - cur_group[-1]["frame"]) / fps
        sim_prev = SequenceMatcher(None, cur_group[-1]["text"].lower(), smp["text"].lower()).ratio()
        first_txt = cur_group[0]["text"]
        is_scramble_prefix = (
            len(cur_group) <= 2
            and gap_sec <= 0.65
            and abs(smp["bbox"][1] - cur_group[-1]["bbox"][1]) <= 0.08 * H
            and any(ch.isdigit() for ch in first_txt)
        )
        if gap_sec <= 0.85 and (sim_prev >= 0.52 or is_scramble_prefix):
            cur_group.append(smp)
        else:
            events.append({"samples": cur_group})
            cur_group = [smp]
    if cur_group:
        events.append({"samples": cur_group})

    final_events: list[dict[str, Any]] = []
    for ev in events:
        smps = ev["samples"]
        if len(smps) < 2 and (n_frames - smps[0]["frame"]) > sample_step * 2:
            continue
        settled = smps[len(smps) // 2 :]
        best_s = max(settled, key=lambda s: sum(w["conf"] for w in s["words"]) / max(1, len(s["words"])))
        bx, by, bw, bh = best_s["bbox"]

        first_probe = smps[0]["frame"]
        probe_start = max(0, first_probe - sample_step)
        probe_end = min(n_frames, first_probe + int(round(fps * 0.75)))
        roi_white_series: list[tuple[int, float]] = []

        px0, py0 = max(0, bx - 16), max(0, by - 12)
        px1, py1 = min(W, bx + bw + 16), min(H, by + bh + 12)
        for pf in range(probe_start, probe_end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pf)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            crop = fr[py0:py1, px0:px1]
            hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            w_density = float(np.mean((hsv_c[:, :, 2] >= 205) & (hsv_c[:, :, 1] <= 50)))
            roi_white_series.append((pf, w_density))

        onset_f = first_probe
        peak_density = max((d for _, d in roi_white_series), default=0.05)
        for pf, wd in roi_white_series:
            if wd >= max(0.012, peak_density * 0.25):
                onset_f = pf
                break

        early_texts: list[str] = []
        for offset_f in (1, 4):
            tf = min(n_frames - 1, onset_f + offset_f)
            cap.set(cv2.CAP_PROP_POS_FRAMES, tf)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            crop = fr[py0:py1, px0:px1]
            hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            inv_c = cv2.bitwise_not(((hsv_c[:, :, 2] >= 205) & (hsv_c[:, :, 1] <= 50)).astype(np.uint8) * 255)
            up_c = cv2.resize(inv_c, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            t_c = pytesseract.image_to_string(up_c, config="--psm 6").strip()
            if t_c:
                early_texts.append(t_c)

        ramp_frames = sum(1 for _, wd in roi_white_series if 0.20 * peak_density <= wd <= 0.78 * peak_density)
        scramble_diffs = [
            SequenceMatcher(None, early_texts[i].lower(), best_s["text"].lower()).ratio()
            for i in range(min(3, len(early_texts)))
        ]
        if scramble_diffs and min(scramble_diffs) < 0.52 and len(early_texts) >= 2:
            reveal_effect = "scramble_decode"
        elif ramp_frames >= 3:
            reveal_effect = "fade_in"
        else:
            reveal_effect = "step_cut_no_fade"

        end_f = min(n_frames - 1, smps[-1]["frame"] + sample_step // 2)
        y_center_pct = round(((by + bh / 2.0) / H) * 100.0, 1)
        pos_zone = "upper_third" if y_center_pct < 38.0 else ("lower_third" if y_center_pct > 62.0 else "center")
        text_clean = best_s["text"]
        is_all_caps = text_clean == text_clean.upper() and any(c.isalpha() for c in text_clean)

        box_h_pct = round(bh * 100.0 / H, 2)
        blend_m = best_s.get("blend_mode", "Normal")
        s_ratio = float(best_s.get("stroke_to_height_ratio", 0.35))
        font_weight_class = (
            "heavy_impact_solid"
            if (box_h_pct >= 20.0 and blend_m == "Normal" and s_ratio >= 0.42)
            else ("regular_sans_difference" if blend_m == "Difference" else "regular_sans_solid")
        )

        final_events.append(
            {
                "slot_index": len(final_events) + 1,
                "text": text_clean,
                "start_sec": round(onset_f / fps, 3),
                "end_sec": round(end_f / fps, 3),
                "start_frame": int(onset_f),
                "end_frame": int(end_f),
                "reveal_effect": reveal_effect,
                "blend_mode": blend_m,
                "font_weight_class": font_weight_class,
                "stroke_to_height_ratio": round(s_ratio, 3),
                "position_zone": pos_zone,
                "y_center_pct": y_center_pct,
                "box_height_pct": box_h_pct,
                "is_all_caps": is_all_caps,
            }
        )

    cap.release()
    effects_found = [e["reveal_effect"] for e in final_events]
    dominant_effect = max(set(effects_found), key=effects_found.count) if effects_found else "none"
    dominant_pos = max(set(e["position_zone"] for e in final_events), key=lambda z: sum(1 for e in final_events if e["position_zone"] == z)) if final_events else "center"

    return {
        "detected": bool(final_events),
        "event_count": len(final_events),
        "dominant_reveal_effect": dominant_effect,
        "dominant_position_zone": dominant_pos,
        "events": final_events,
    }


def _compute_reference_lumetri_matrix(scan: dict[str, Any], vis_y0: np.ndarray, vis_y1: np.ndarray, video_path: str) -> dict[str, Any]:
    """Compute real active-region reference color profile & lumetri_matrix using virtual_scopes."""
    ax, ay, aw, ah = scan["active_bounds"]
    n_frames = scan["n_frames"]
    vis_ratio = (vis_y1 - vis_y0) / max(1.0, float(ah))

    white_f_set = {c["frame"] for c in scan["white_candidates"] if c["white_pct_full"] > 0.08}
    eligible = [t for t in range(n_frames) if vis_ratio[t] >= 0.85 and scan["frame_means"][t] >= 18.0 and t not in white_f_set]
    if not eligible:
        eligible = [t for t in range(n_frames) if scan["frame_means"][t] >= 12.0]
    if not eligible:
        eligible = [n_frames // 2]

    pick_idx = np.linspace(0, len(eligible) - 1, min(10, len(eligible)), dtype=int)
    probe_frames = [eligible[i] for i in pick_idx]

    cap = cv2.VideoCapture(video_path)
    scope_list: list[dict[str, Any]] = []
    vignette_ratios: list[float] = []
    shadow_casts: list[tuple[float, float, float]] = []
    high_casts: list[tuple[float, float, float]] = []

    for pf in probe_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pf))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        roi = fr[ay : ay + ah, ax : ax + aw]
        sc = analyze_frame_scopes(roi, auto_crop_active=False)
        scope_list.append(sc)

        img_f = roi.astype(np.float32) / 255.0
        luma = 0.2126 * img_f[:, :, 2] + 0.7152 * img_f[:, :, 1] + 0.0722 * img_f[:, :, 0]
        ch, cw = roi.shape[:2]
        center_y = float(np.mean(luma[ch // 4 : (3 * ch) // 4, cw // 4 : (3 * cw) // 4]))
        edge_y = float(np.mean([np.mean(luma[: ch // 8, : cw // 8]), np.mean(luma[-ch // 8 :, -cw // 8 :])]))
        if center_y > 0.08:
            vignette_ratios.append(edge_y / center_y)

        sh_mask = (luma >= 0.06) & (luma <= 0.30)
        if np.count_nonzero(sh_mask) > 100:
            sb, sg, sr = [float(np.mean(img_f[:, :, k][sh_mask]) * 100.0) for k in range(3)]
            savg = (sb + sg + sr) / 3.0
            shadow_casts.append((sr - savg, sg - savg, sb - savg))

        hi_mask = (luma >= 0.68) & (luma <= 0.95)
        if np.count_nonzero(hi_mask) > 100:
            hb, hg, hr = [float(np.mean(img_f[:, :, k][hi_mask]) * 100.0) for k in range(3)]
            havg = (hb + hg + hr) / 3.0
            high_casts.append((hr - havg, hg - havg, hb - havg))

    cap.release()

    med_p01 = round(float(np.median([s["luma_ire"]["p01_black"] for s in scope_list])), 1)
    med_p25 = round(float(np.median([s["luma_ire"]["p25_shadow"] for s in scope_list])), 1)
    med_p50 = round(float(np.median([s["luma_ire"]["p50_mid"] for s in scope_list])), 1)
    med_p75 = round(float(np.median([s["luma_ire"]["p75_high"] for s in scope_list])), 1)
    med_p99 = round(float(np.median([s["luma_ire"]["p99_white"] for s in scope_list])), 1)
    med_rb = round(float(np.median([s["rgb_parade_cast"]["r_bias"] for s in scope_list])), 2)
    med_gb = round(float(np.median([s["rgb_parade_cast"]["g_bias"] for s in scope_list])), 2)
    med_bb = round(float(np.median([s["rgb_parade_cast"]["b_bias"] for s in scope_list])), 2)
    med_sat = round(float(np.median([s["environment_metrics"]["mean_saturation_pct"] for s in scope_list])), 1)
    med_green = round(float(np.median([s["environment_metrics"]["green_fraction"] for s in scope_list])), 4)
    med_sky = round(float(np.median([s["environment_metrics"]["sky_blue_fraction"] for s in scope_list])), 4)
    med_warm = round(float(np.median([s["environment_metrics"]["warm_fraction"] for s in scope_list])), 4)
    vig_ratio = round(float(np.median(vignette_ratios)) if vignette_ratios else 1.0, 2)

    sh_mean = [round(float(np.mean([c[k] for c in shadow_casts])), 2) for k in range(3)] if shadow_casts else [0.0, 0.0, 0.0]
    hi_mean = [round(float(np.mean([c[k] for c in high_casts])), 2) for k in range(3)] if high_casts else [0.0, 0.0, 0.0]

    faded_film = int(np.clip((med_p01 - 5.5) * 2.2, 0, 18))
    ref_dyn = med_p99 - med_p01
    contrast_shift = int(np.clip((ref_dyn - 82.0) * 0.55, -18, 22))
    highlights_shift = int(np.clip((med_p99 - 90.0) * 1.4, -30, 10))
    shadows_shift = int(np.clip((med_p25 - 22.0) * 1.1, -25, 20))
    blacks_shift = int(np.clip((med_p01 - 7.0) * 2.0 - faded_film, -25, 18))
    temp_shift = int(np.clip((med_rb - med_bb) * 1.8, -18, 18))
    tint_shift = int(np.clip(-med_gb * 1.8, -15, 15))
    sat_target = int(np.clip(100 + (med_sat - 32.0) * 0.85, 80, 118))
    vibrance_target = int(np.clip((med_sat - 28.0) * 0.9, -15, 22))
    vignette_amt = round(float(np.clip((0.92 - vig_ratio) * -4.0, -2.5, 0.0)), 2)

    if med_p01 >= 9.0 and med_sat < 30.0:
        tone_style = "moody_matte_filmic"
    elif med_bb > med_rb + 1.5 or med_sky > 0.14:
        tone_style = "crisp_cool_alpine"
    elif med_rb > med_bb + 1.5 or med_warm > 0.14:
        tone_style = "warm_nostalgic_golden"
    else:
        tone_style = "balanced_natural_editorial"

    return {
        "tone_style": tone_style,
        "measured_active_scopes": {
            "luma_ire": {
                "p01_black": med_p01,
                "p25_shadow": med_p25,
                "p50_mid": med_p50,
                "p75_high": med_p75,
                "p99_white": med_p99,
                "dynamic_range": round(ref_dyn, 1),
            },
            "rgb_parade_cast": {"r_bias": med_rb, "g_bias": med_gb, "b_bias": med_bb},
            "shadow_cast_rgb": sh_mean,
            "highlight_cast_rgb": hi_mean,
            "environment_profile": {
                "mean_saturation_pct": med_sat,
                "green_fraction": med_green,
                "sky_blue_fraction": med_sky,
                "warm_fraction": med_warm,
                "vignette_edge_to_center_ratio": vig_ratio,
            },
        },
        "lumetri_matrix": {
            "exposure": round(float(np.clip((med_p50 - 48.0) / 28.0, -0.45, 0.45)), 2),
            "contrast": contrast_shift,
            "highlights": highlights_shift,
            "shadows": shadows_shift,
            "whites": 0 if med_p99 >= 88.0 else 5,
            "blacks": blacks_shift,
            "temperature": temp_shift,
            "tint": tint_shift,
            "saturation": sat_target,
            "vibrance": vibrance_target,
            "faded_film": faded_film,
            "sharpen": 12,
            "vignette_amount": vignette_amt,
            "look_intensity": 65,
        },
    }


def extract_style_dna(
    ref_video_path: str,
    output_json_path: str = "02_analysis/style_dna.json",
    assets_dir: str = "04_assets",
) -> dict[str, Any]:
    """Run complete 4-layer reverse-engineering pipeline on a reference video."""
    ref_p = Path(ref_video_path)
    if not ref_p.exists():
        raise FileNotFoundError(f"Reference video not found: {ref_video_path}")

    stream_meta = _probe_video_stream(str(ref_p))
    audio_info = extract_and_clone_reference_music(
        str(ref_p),
        out_dir=str(Path(assets_dir) / "music"),
        vault_path=str(Path(assets_dir) / "vault_index.json"),
    )

    scan = _scan_video_stream(str(ref_p))
    vis_y0, vis_y1 = _compute_visible_row_spans(scan)
    strips_info, aperture_info = _detect_aperture_and_strips(scan, vis_y0, vis_y1, str(ref_p), audio_info)
    segments, mask_events, outro_info = _detect_cuts_and_masks(scan, vis_y0, vis_y1, aperture_info, audio_info, str(ref_p))
    panel_overlays = _detect_geometric_panels_and_stickers(scan, segments)
    strobe_schedules = _detect_strobe_and_blink_schedules(scan, segments)
    collage_events = _detect_intra_shot_collage_and_splits(scan, segments)
    postmortem_sheet = _generate_postmortem_contact_sheet(str(ref_p), segments)
    typo_info = _detect_typography(scan, str(ref_p))
    color_info = _compute_reference_lumetri_matrix(scan, vis_y0, vis_y1, str(ref_p))

    ax, ay, aw, ah = scan["active_bounds"]
    W, H = scan["width"], scan["height"]
    letterboxed = ah < int(0.92 * H) or aw < int(0.92 * W)
    active_ar = round(aw / max(1, ah), 4)

    durations = [s["duration_sec"] for s in segments]
    asl = round(sum(durations) / max(1, len(durations)), 3)
    slit_durs = [s["duration_sec"] for s in segments if s["in_expanding_slit"]]
    rapid_beat_mode = bool(durations and min(durations) < 0.45 and sum(1 for d in durations if d <= 0.55) >= 4)
    beat_locked_ratio = round(sum(1 for s in segments if s.get("is_beat_locked")) / max(1, len(segments)), 3)

    style_dna = {
        "schema_version": "2.1",
        "reference_file": ref_p.as_posix(),
        "spatial_geometry": {
            "container_resolution": [W, H],
            "container_fps": stream_meta["fps"],
            "is_vfr": stream_meta["is_vfr"],
            "active_picture_bounds_xywh": [ax, ay, aw, ah],
            "active_aspect_ratio": active_ar,
            "letterbox_or_pillarbox_detected": letterboxed,
            "recommended_landscape_adaptation": (
                {
                    "strip_static_letterbox": True,
                    "target_resolution": [3840, 2160],
                    "scaling_policy": "crop_no_stretch",
                    "fps_policy": "match_project_raw_fps_e.g_60fps_and_snap_cuts_to_frame",
                }
                if letterboxed and active_ar >= 1.45
                else {
                    "strip_static_letterbox": letterboxed,
                    "target_aspect_ratio": active_ar,
                    "scaling_policy": "crop_no_stretch",
                    "fps_policy": "match_project_raw_fps_when_higher_and_snap_cuts_to_frame",
                }
            ),
        },
        "layer_1_temporal_grammar": {
            "duration_sec": scan["duration_sec"],
            "total_frames": scan["n_frames"],
            "cut_count": max(0, len(segments) - 1),
            "segment_count": len(segments),
            "average_shot_length_sec": asl,
            "min_shot_sec": min(durations) if durations else 0.0,
            "max_shot_sec": max(durations) if durations else 0.0,
            "slit_median_shot_sec": round(float(np.median(slit_durs)), 3) if slit_durs else None,
            "cadence_profile": "rapid_beat_montage" if rapid_beat_mode else "lyrical_cuts",
            "beat_locked_cut_ratio": beat_locked_ratio,
            "allow_clip_reuse_for_high_cut_density": len(segments) >= 15,
            "multi_strip_opening": strips_info,
            "aperture_animation": aperture_info,
            "geometric_panel_overlays": panel_overlays,
            "strobe_and_blink_schedules": strobe_schedules,
            "intra_shot_collage_events": collage_events,
            "postmortem_contact_sheet": postmortem_sheet,
            "mask_transitions": mask_events,
            "outro_transition": outro_info,
        },
        "layer_2_acoustic_architecture": {
            "bpm": audio_info["bpm"],
            "beat_count": audio_info["beat_count"],
            "cloned_music_track": audio_info["cloned_audio_path"],
            "archived_music_track": audio_info["archived_audio_path"],
            "silence_windows": audio_info["silence_windows"],
            "energy_resurgence_cues": audio_info["energy_resurgence_cues"],
            "strong_onset_timestamps": audio_info["strong_onset_timestamps"],
            "beat_timestamps": audio_info["beat_timestamps"],
            "downbeat_timestamps": audio_info["downbeat_timestamps"],
            "sub_beat_timestamps": audio_info["sub_beat_timestamps"],
            "track_targets_dbfs": {
                "A1_dialogue": -6.0,
                "A2_sfx": -14.0,
                "A3_music_ambient": -18.0,
                "A3_music_ducked": -22.0,
            },
        },
        "layer_3_aesthetic_typography_and_color": {
            "tone_style": color_info["tone_style"],
            "lumetri_matrix": color_info["lumetri_matrix"],
            "measured_active_scopes": color_info["measured_active_scopes"],
            "visual_coherence_target": {
                "mean_green_fraction": color_info["measured_active_scopes"]["environment_profile"]["green_fraction"],
                "mean_sky_blue_fraction": color_info["measured_active_scopes"]["environment_profile"]["sky_blue_fraction"],
                "mean_warm_fraction": color_info["measured_active_scopes"]["environment_profile"]["warm_fraction"],
                "outdoor_scene_consistency_required": color_info["measured_active_scopes"]["environment_profile"]["green_fraction"] >= 0.05
                or color_info["measured_active_scopes"]["environment_profile"]["sky_blue_fraction"] >= 0.05,
            },
            "typography": typo_info,
        },
        "layer_4_narrative_slots": segments,
    }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(style_dna, indent=2), encoding="utf-8")
    return style_dna


def _auto_find_reference(ref_dir: str = "00_reference_style") -> str | None:
    p = Path(ref_dir)
    if not p.exists():
        return None
    for f in sorted(p.iterdir()):
        if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
            return str(f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-engineer a reference reel into 4-layer style_dna.json.")
    parser.add_argument("pos_ref", nargs="?", default=None, help="Optional positional reference video path")
    parser.add_argument("pos_out", nargs="?", default=None, help="Optional positional output JSON path")
    parser.add_argument("--ref", "--reference", dest="ref", default=None, help="Path to reference video")
    parser.add_argument("--out", "--output", dest="out", default="02_analysis/style_dna.json", help="Output JSON path")
    parser.add_argument("--assets_dir", default="04_assets", help="Assets vault root directory")
    args = parser.parse_args()

    ref_path = args.ref or args.pos_ref or _auto_find_reference()
    if not ref_path:
        raise SystemExit("[ERROR] No reference video specified and none found in 00_reference_style/")
    out_path = args.pos_out or args.out

    dna = extract_style_dna(ref_path, out_path, assets_dir=args.assets_dir)
    t = dna["layer_1_temporal_grammar"]
    c = dna["layer_3_aesthetic_typography_and_color"]
    print(f"[OK] Saved style_dna.json -> {out_path}")
    print(f"     Geometry : {dna['spatial_geometry']['container_resolution']} active={dna['spatial_geometry']['active_picture_bounds_xywh']} letterbox={dna['spatial_geometry']['letterbox_or_pillarbox_detected']}")
    print(f"     Temporal : {t['segment_count']} segments ({t['cut_count']} cuts), ASL={t['average_shot_length_sec']}s, mode={t['cadence_profile']}, beat_locked={t['beat_locked_cut_ratio']}")
    print(f"     Strips   : detected={t['multi_strip_opening']['detected']} count={t['multi_strip_opening']['strip_count']} border={t['multi_strip_opening'].get('border_style')}")
    print(f"     Aperture : detected={t['aperture_animation']['detected']} close_steps={len(t['aperture_animation']['stepped_close_levels'])} slit_unlock={t['aperture_animation']['slit_unlock']}")
    print(f"     Masks    : {[(m['type'], m['start_sec'], m['end_sec']) for m in t['mask_transitions']]}")
    print(f"     Typo     : detected={c['typography']['detected']} events={c['typography']['event_count']} effect={c['typography']['dominant_reveal_effect']}")
    print(f"     Color    : style={c['tone_style']} lumetri={c['lumetri_matrix']}")


if __name__ == "__main__":
    main()
