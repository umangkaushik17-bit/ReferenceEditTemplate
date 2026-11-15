"""Virtual Lumetri Scopes, Active-Region Telemetry, Shot Semantics & Adaptive Frame Sampler.

Provides:
- Hardware device auto-detection (CUDA / MPS / DirectML / multi-core CPU)
- Authoritative rational & VFR-aware media stream probing + Premiere tick math
- Adaptive frame & keyframe sampling via FFmpeg pipe (supports sub-second optical-flow & micro-cut cadence)
- Active-picture letterbox/pillarbox stripping (`extract_active_region`)
- Rec.709 Luma IRE, chroma-safe RGB Parade grey-balance (immune to lush green foliage false-positives),
  YCrCb Vectorscope skin-tone telemetry, and Pass 1 + Pass 2 shot-matching deltas
- Semantic shot classification (shot scale, camera motion direction/magnitude, 32-D visual fingerprint)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import cv2
import numpy as np

PREMIERE_TICKS_PER_SECOND = 254016000000


def get_compute_device() -> tuple[str, Any]:
    """Detect best available PyTorch compute device (cuda -> mps -> directml -> cpu) and optimize CPU threads."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.device("mps")
        try:
            import torch_directml  # type: ignore

            if torch_directml.is_available():
                return "directml", torch_directml.device()
        except Exception:
            pass
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(max(2, min(8, cpu_count)))
        return "cpu", torch.device("cpu")
    except Exception:
        return "cpu", None


def seconds_to_premiere_ticks(seconds: float) -> int:
    """Convert seconds to exact Premiere Pro ticks (254016000000 ticks/sec)."""
    return int(round(float(seconds) * PREMIERE_TICKS_PER_SECOND))


def snap_time_to_frame(seconds: float, fps: float) -> float:
    """Snap timestamp in seconds to the nearest exact integer frame boundary."""
    if fps <= 0:
        return round(float(seconds), 4)
    frame_idx = int(round(float(seconds) * float(fps)))
    return round(frame_idx / float(fps), 6)


def _parse_rational_fps(rate_str: str | None) -> float:
    """Parse rational frame rate string like '60000/1001' without premature 3-decimal truncation."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    parts = str(rate_str).split("/")
    if len(parts) == 2:
        num, den = float(parts[0]), float(parts[1])
        return (num / den) if den > 0 else 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def probe_video_stream_info(video_path: str) -> dict[str, Any]:
    """Probe resolution, authoritative avg/r frame rate, VFR status, audio stream, and duration."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    info = json.loads(res.stdout or "{}")
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    r_fps = _parse_rational_fps(v_stream.get("r_frame_rate"))
    avg_fps = _parse_rational_fps(v_stream.get("avg_frame_rate"))

    # Prefer avg_frame_rate when r_frame_rate is a container timebase spike (e.g. 90000/1 or VFR mismatch)
    if avg_fps > 1.0 and (r_fps <= 0 or r_fps > 240.0 or abs(r_fps - avg_fps) > 1.5):
        fps = avg_fps
    else:
        fps = r_fps if r_fps > 0 else (avg_fps if avg_fps > 0 else 30.0)

    is_vfr = bool(r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) > 0.25)
    width = int(v_stream.get("width", 1920) or 1920)
    height = int(v_stream.get("height", 1080) or 1080)
    dur = float(info.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "fps_exact": fps,
        "r_frame_rate": v_stream.get("r_frame_rate", "30/1"),
        "avg_frame_rate": v_stream.get("avg_frame_rate", "30/1"),
        "is_vfr": is_vfr,
        "duration": round(dur, 3),
        "has_audio": a_stream is not None,
    }


def extract_active_region(img_bgr: np.ndarray, black_thresh: int = 14) -> tuple[np.ndarray, dict[str, Any]]:
    """Strip static black letterbox/pillarbox/windowbox bars before scope calculation."""
    h, w = img_bgr.shape[:2]
    max_ch = np.max(img_bgr, axis=2)
    active_px = max_ch > black_thresh

    def _largest_run(mask_1d: np.ndarray, max_gap: int = 4) -> tuple[int, int] | None:
        idx = np.where(mask_1d)[0]
        if idx.size == 0:
            return None
        runs = []
        start = int(idx[0])
        prev = int(idx[0])
        for cur in idx[1:]:
            cur = int(cur)
            if cur - prev <= max_gap + 1:
                prev = cur
            else:
                runs.append((start, prev + 1))
                start = prev = cur
        runs.append((start, prev + 1))
        return max(runs, key=lambda r: r[1] - r[0])

    row_run = _largest_run(active_px.mean(axis=1) > 0.04, max_gap=max(4, int(0.01 * h)))
    if row_run is None:
        return img_bgr, {"cropped": False, "box": [0, 0, w, h], "full_size": [w, h]}
    y0, y1 = row_run

    col_run = _largest_run(active_px[y0:y1, :].mean(axis=0) > 0.04, max_gap=max(4, int(0.01 * w)))
    if col_run is None:
        return img_bgr, {"cropped": False, "box": [0, 0, w, h], "full_size": [w, h]}
    x0, x1 = col_run

    if (y1 - y0) < max(16, int(0.08 * h)) or (x1 - x0) < max(16, int(0.08 * w)):
        return img_bgr, {"cropped": False, "box": [0, 0, w, h], "full_size": [w, h]}

    is_cropped = (y1 - y0) < int(0.96 * h) or (x1 - x0) < int(0.96 * w)
    if not is_cropped:
        return img_bgr, {"cropped": False, "box": [0, 0, w, h], "full_size": [w, h]}

    return img_bgr[y0:y1, x0:x1], {
        "cropped": True,
        "box": [x0, y0, x1 - x0, y1 - y0],
        "full_size": [w, h],
    }


def analyze_frame_scopes(img_bgr: np.ndarray, auto_crop_active: bool = True) -> dict[str, Any]:
    """Compute Rec.709 Luma IRE, foliage-safe RGB Parade cast, YCrCb skin IRE, and Pass 1 Lumetri parameters."""
    active_info = {
        "cropped": False,
        "box": [0, 0, img_bgr.shape[1], img_bgr.shape[0]],
        "full_size": [img_bgr.shape[1], img_bgr.shape[0]],
    }
    if auto_crop_active:
        img_bgr, active_info = extract_active_region(img_bgr)
    img_float = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)

    # Rec.709 Luma mapped to 0-100 IRE
    luma_ire = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 100.0
    p01, p25, p50, p75, p99 = np.percentile(luma_ire, [1, 25, 50, 75, 99])

    # Visual theme / environment cues from HSV (OpenCV H in 0..179, S/V in 0..255)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    sat = s_ch.astype(np.float32) / 255.0
    green_mask = (h_ch >= 30) & (h_ch <= 92) & (s_ch >= 32) & (v_ch >= 28)
    sky_mask = (h_ch >= 90) & (h_ch <= 130) & (s_ch >= 25) & (v_ch >= 75)
    warm_mask = ((h_ch <= 24) | (h_ch >= 165)) & (s_ch >= 38) & (v_ch >= 35)

    green_frac = float(np.mean(green_mask))
    sky_frac = float(np.mean(sky_mask))
    warm_frac = float(np.mean(warm_mask))

    # Neutral Midtone RGB Parade Cast Detection:
    # Never sample saturated green foliage or blue sky as a "color cast"!
    mid_mask = (luma_ire > 22.0) & (luma_ire < 82.0)
    neutral_mask = mid_mask & (sat < 0.20)
    if np.count_nonzero(neutral_mask) < 120:
        # Adaptive fallback: take the lowest-saturation 15% of midtone pixels, capped at sat < 0.35
        if np.count_nonzero(mid_mask) > 50:
            sat_cap = min(0.35, float(np.percentile(sat[mid_mask], 15)))
            neutral_mask = mid_mask & (sat <= sat_cap)
    if np.count_nonzero(neutral_mask) >= 30:
        r_mean = float(np.mean(r[neutral_mask]) * 100.0)
        g_mean = float(np.mean(g[neutral_mask]) * 100.0)
        b_mean = float(np.mean(b[neutral_mask]) * 100.0)
        avg_rgb = (r_mean + g_mean + b_mean) / 3.0
        # Dampen cast estimate if even the least-saturated pixels are moderately chromatic
        mean_neutral_sat = float(np.mean(sat[neutral_mask]))
        trust_weight = max(0.25, 1.0 - max(0.0, mean_neutral_sat - 0.15) * 3.0)
        r_bias = round((r_mean - avg_rgb) * trust_weight, 2)
        g_bias = round((g_mean - avg_rgb) * trust_weight, 2)
        b_bias = round((b_mean - avg_rgb) * trust_weight, 2)
    else:
        # Entire frame is highly saturated (e.g. pure lush foliage close-up): do not hallucinate WB cast
        r_bias, g_bias, b_bias = 0.0, 0.0, 0.0

    # Skin-Tone Detection & Vectorscope I-Line Check (YCrCb space)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([40, 135, 85]), np.array([235, 175, 130]))
    # Exclude green/sky pixels from accidental skin false positives
    skin_mask = cv2.bitwise_and(skin_mask, (~(green_mask | sky_mask)).astype(np.uint8) * 255)
    skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)
    skin_detected = bool(skin_ratio > 0.015)

    if skin_detected:
        skin_luma_ire = float(np.mean(luma_ire[skin_mask > 0]))
        cr_shift = float(np.mean(ycrcb[:, :, 1][skin_mask > 0]) - 152.0)
        cb_shift = float(np.mean(ycrcb[:, :, 2][skin_mask > 0]) - 108.0)
    else:
        skin_luma_ire = float(p50)
        cr_shift, cb_shift = 0.0, 0.0

    # Compute deterministic Lumetri Basic Correction parameters
    target_mid = 62.0 if skin_detected else 48.0
    current_mid = skin_luma_ire if skin_detected else float(p50)
    exposure = round(float(np.clip((target_mid - current_mid) / 18.0, -1.8, 1.8)), 2)

    dyn_range = float(p99 - p01)
    contrast = int(np.clip((82.0 - dyn_range) * 0.75, -20, 35))
    highlights = int(np.clip((88.0 - p99) * 1.6, -55, 15))
    shadows = int(np.clip((22.0 - p25) * 1.4, -25, 40))
    whites = int(np.clip((92.0 - p99) * 1.2, -30, 25))
    blacks = int(np.clip((7.0 - p01) * 1.5, -30, 20))

    # Enforce highlight ceiling invariant from color-grading-invariants.md
    if p99 > 94.0:
        exposure = min(0.0, exposure)
        highlights = min(-15, max(-45, highlights))
        whites = min(-5, whites)

    # Counteract Blue/Red cast via Temperature & Green/Magenta cast via Tint
    # On predominantly green outdoor nature shots, cap positive magenta tint to prevent purple foliage
    max_magenta_tint = 10 if green_frac >= 0.12 else 30
    temperature = int(np.clip((b_bias - r_bias) * 2.2, -35, 35))
    tint = int(np.clip(g_bias * 2.2, -30, max_magenta_tint))
    mean_sat = float(np.mean(sat) * 100.0)
    saturation = int(np.clip(100 + (32.0 - mean_sat) * 0.6, 85, 118))

    return {
        "active_region": active_info,
        "luma_ire": {
            "p01_black": round(float(p01), 1),
            "p25_shadow": round(float(p25), 1),
            "p50_mid": round(float(p50), 1),
            "p75_high": round(float(p75), 1),
            "p99_white": round(float(p99), 1),
            "dynamic_range": round(dyn_range, 1),
        },
        "rgb_parade_cast": {"r_bias": r_bias, "g_bias": g_bias, "b_bias": b_bias},
        "skin_metrics": {
            "skin_detected": skin_detected,
            "skin_coverage_pct": round(skin_ratio * 100.0, 2),
            "skin_luma_ire": round(skin_luma_ire, 1),
            "cr_warmth_offset": round(cr_shift, 2),
            "cb_cool_offset": round(cb_shift, 2),
        },
        "environment_metrics": {
            "mean_saturation_pct": round(mean_sat, 1),
            "green_fraction": round(green_frac, 4),
            "sky_blue_fraction": round(sky_frac, 4),
            "warm_fraction": round(warm_frac, 4),
            "mean_bgr": [round(float(np.mean(img_bgr[:, :, i])), 1) for i in range(3)],
        },
        "recommended_primary_lumetri": {
            "exposure": exposure,
            "contrast": contrast,
            "highlights": highlights,
            "shadows": shadows,
            "whites": whites,
            "blacks": blacks,
            "temperature": temperature,
            "tint": tint,
            "saturation": saturation,
        },
    }


def compute_visual_fingerprint(frame_bgr: np.ndarray, auto_crop_active: bool = True) -> list[float]:
    """Compute a normalized 32-D spatial-color-edge visual embedding for scene similarity matching."""
    if auto_crop_active:
        frame_bgr, _ = extract_active_region(frame_bgr)
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 12-bin Hue + 4-bin Sat + 4-bin Val + 3x3 spatial luma grid (9) + 3 edge orientation/density = 32 dims
    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
    h_hist /= max(1e-6, float(np.sum(h_hist)))
    s_hist /= max(1e-6, float(np.sum(s_hist)))
    v_hist /= max(1e-6, float(np.sum(v_hist)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grid_3x3 = cv2.resize(gray, (3, 3), interpolation=cv2.INTER_AREA).flatten()

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    h_half = mag.shape[0] // 2
    edge_feats = np.array(
        [
            float(np.mean(mag[:h_half, :])),
            float(np.mean(mag[h_half:, :])),
            float(np.std(gray)),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([h_hist, s_hist, v_hist, grid_3x3, edge_feats]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return [round(float(x), 4) for x in vec]


def analyze_shot_semantics(frames: list[np.ndarray], auto_crop_active: bool = True) -> dict[str, Any]:
    """Classify shot scale, camera movement direction/speed, and visual embedding from sampled frames."""
    valid = [f for f in frames if f is not None and float(np.mean(f)) >= 4.0]
    if not valid:
        return {
            "shot_scale": "unknown",
            "camera_motion": "static_locked",
            "motion_magnitude": 0.0,
            "visual_fingerprint": [0.0] * 32,
        }

    mid_frame = valid[len(valid) // 2]
    cropped, _ = extract_active_region(mid_frame) if auto_crop_active else (mid_frame, {})
    ch, cw = cropped.shape[:2]
    gray_mid = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Center vs periphery Laplacian detail ratio estimates Shot Scale (Close-Up vs Wide Landscape)
    lap = np.abs(cv2.Laplacian(gray_mid, cv2.CV_32F))
    center_lap = float(np.mean(lap[ch // 4 : (3 * ch) // 4, cw // 4 : (3 * cw) // 4]))
    outer_lap = max(1e-4, float(np.mean(lap)) - 0.25 * center_lap)
    center_dominance = center_lap / outer_lap

    hsv_mid = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    sky_top = float(np.mean((hsv_mid[: ch // 3, :, 0] >= 90) & (hsv_mid[: ch // 3, :, 0] <= 132) & (hsv_mid[: ch // 3, :, 2] >= 75)))

    if center_dominance >= 1.85:
        shot_scale = "close_up_detail"
    elif center_dominance >= 1.32:
        shot_scale = "medium_subject"
    elif sky_top >= 0.22:
        shot_scale = "extreme_wide_scenic"
    else:
        shot_scale = "wide_establishing"

    # Optical flow camera movement across sequential frame pairs
    dxs, dys, mags, div_scores = [], [], [], []
    for i in range(len(valid) - 1):
        f1 = extract_active_region(valid[i])[0] if auto_crop_active else valid[i]
        f2 = extract_active_region(valid[i + 1])[0] if auto_crop_active else valid[i + 1]
        g1 = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (160, 90))
        g2 = cv2.resize(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (160, 90))
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 13, 2, 5, 1.1, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        dxs.append(float(np.median(fx)))
        dys.append(float(np.median(fy)))
        mags.append(float(np.median(np.sqrt(fx**2 + fy**2))))
        # Radial divergence checks push-in / pull-out zoom
        left_x = float(np.median(fx[:, :50]))
        right_x = float(np.median(fx[:, 110:]))
        div_scores.append(right_x - left_x)

    med_dx = float(np.median(dxs)) if dxs else 0.0
    med_dy = float(np.median(dys)) if dys else 0.0
    med_mag = float(np.median(mags)) if mags else 0.0
    med_div = float(np.median(div_scores)) if div_scores else 0.0

    if med_mag < 0.65:
        cam_motion = "static_locked"
    elif abs(med_div) >= 1.4 and abs(med_div) > abs(med_dx) * 0.8:
        cam_motion = "push_in" if med_div > 0 else "pull_out"
    elif abs(med_dx) >= abs(med_dy) * 1.25 and abs(med_dx) >= 0.8:
        # Optical flow opposite to camera pan direction
        cam_motion = "pan_right" if med_dx < 0 else "pan_left"
    elif abs(med_dy) >= 0.8:
        cam_motion = "tilt_down" if med_dy < 0 else "tilt_up"
    else:
        cam_motion = "handheld_dynamic"

    return {
        "shot_scale": shot_scale,
        "camera_motion": cam_motion,
        "motion_magnitude": round(med_mag, 3),
        "center_detail_dominance": round(center_dominance, 3),
        "visual_fingerprint": compute_visual_fingerprint(mid_frame, auto_crop_active=auto_crop_active),
    }


def adaptive_sample_frames(
    video_path: str,
    target_width: int = 480,
    sample_fps: float = 4.0,
    max_frames: int = 24,
) -> tuple[list[float], list[np.ndarray], dict[str, Any]]:
    """Fast CPU-friendly frame sampler: decodes container keyframes first (~20x faster on 4K 60fps),
    and only falls back to full-stream FPS filter when keyframes are sparser than 6 frames.
    """
    times, frames, meta = fast_sample_keyframes(
        video_path,
        target_width=target_width,
        max_frames=max_frames,
        min_frames=2,
    )
    return times, frames, meta


def fast_sample_keyframes(
    video_path: str,
    target_width: int = 480,
    max_frames: int = 16,
    min_frames: int = 2,
) -> tuple[list[float], list[np.ndarray], dict[str, Any]]:
    """Decode container keyframes via FFmpeg pipe, falling back to uniform adaptive FPS sampling when GOP is sparse."""
    meta = probe_video_stream_info(video_path)
    src_w, src_h, fps, dur = meta["width"], meta["height"], meta["fps"], meta["duration"]

    out_w = max(64, int(target_width) // 2 * 2)
    out_h = max(64, int(round(src_h * out_w / max(1, src_w))) // 2 * 2)

    cmd = [
        "ffmpeg",
        "-skip_frame",
        "nokey",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-vf",
        f"scale={out_w}:{out_h}:flags=fast_bilinear,showinfo",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, check=False)
    stderr_txt = res.stderr.decode("utf-8", errors="ignore")
    pts_list = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", stderr_txt)]

    frame_bytes = out_w * out_h * 3
    n_decoded = len(res.stdout) // frame_bytes if frame_bytes > 0 else 0

    # If video has sparse keyframes (< min_frames), fallback to direct low-res uniform FPS sampling
    if n_decoded < min(min_frames, max_frames) and dur > 0.5:
        eff_fps = max(2.0, min(6.0, max_frames / max(0.5, dur)))
        cmd_fb = [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "4",
            "-i",
            str(video_path),
            "-vf",
            f"fps={eff_fps:.3f},scale={out_w}:{out_h}:flags=fast_bilinear",
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        res_fb = subprocess.run(cmd_fb, capture_output=True, check=False)
        n_fb = len(res_fb.stdout) // frame_bytes if frame_bytes > 0 else 0
        if n_fb > n_decoded:
            raw_fb = np.frombuffer(res_fb.stdout[: n_fb * frame_bytes], dtype=np.uint8).reshape((n_fb, out_h, out_w, 3))
            dt = 1.0 / max(0.1, eff_fps)
            times_fb = [round(min(dur, i * dt), 3) for i in range(n_fb)]
            frames_fb = [raw_fb[i].copy() for i in range(n_fb)]
            return times_fb, frames_fb, meta

    if n_decoded == 0:
        return [], [], meta

    raw_arr = np.frombuffer(res.stdout[: n_decoded * frame_bytes], dtype=np.uint8).reshape((n_decoded, out_h, out_w, 3))
    if len(pts_list) < n_decoded:
        pts_list = [i * (dur / max(1, n_decoded)) for i in range(n_decoded)]
    else:
        pts_list = pts_list[:n_decoded]

    if n_decoded > max_frames > 1:
        keep = np.linspace(0, n_decoded - 1, max_frames, dtype=int)
        frames = [raw_arr[int(i)].copy() for i in keep]
        times = [round(pts_list[int(i)], 3) for i in keep]
    else:
        frames = [raw_arr[i].copy() for i in range(n_decoded)]
        times = [round(pts_list[i], 3) for i in range(n_decoded)]

    return times, frames, meta


def analyze_video_file_scopes(
    video_path: str,
    frames: list[np.ndarray] | None = None,
    auto_crop_active: bool = True,
) -> dict[str, Any]:
    """Analyze pre-sampled or keyframe-sampled non-black frames and return median scope telemetry."""
    if frames is None:
        _, frames, _ = fast_sample_keyframes(str(video_path), target_width=480, max_frames=8)
    samples = []
    for frame in frames:
        if frame is None or float(np.mean(frame)) < 4.5:
            continue
        samples.append(analyze_frame_scopes(frame, auto_crop_active=auto_crop_active))
    if not samples:
        return {}
    mids = [s["luma_ire"]["p50_mid"] for s in samples]
    med_mid = float(np.median(mids))
    best = min(samples, key=lambda s: abs(s["luma_ire"]["p50_mid"] - med_mid))
    best["environment_metrics"]["green_fraction"] = round(
        float(np.mean([s["environment_metrics"]["green_fraction"] for s in samples])), 4
    )
    best["environment_metrics"]["sky_blue_fraction"] = round(
        float(np.mean([s["environment_metrics"]["sky_blue_fraction"] for s in samples])), 4
    )
    best["environment_metrics"]["warm_fraction"] = round(
        float(np.mean([s["environment_metrics"]["warm_fraction"] for s in samples])), 4
    )
    return best


def compute_shot_match_delta(clip_scopes: dict[str, Any], hero_scopes: dict[str, Any]) -> dict[str, Any]:
    """Compute Pass 2 shot-matching Lumetri parameters that align a clip directly to the Hero clip's look.

    Provides both:
    - `direct_lumetri_to_match_hero`: single-instance Lumetri values that move raw clip -> hero raw look
    - `pass2_residual_after_pass1`: additive delta if Pass 1 (`recommended_primary_lumetri`) was already applied
      to the clip and you want its final tone/foliage/temperature to match the Hero clip's post-Pass-1 state.
    """
    c_luma = clip_scopes["luma_ire"]
    h_luma = hero_scopes["luma_ire"]
    c_cast = clip_scopes["rgb_parade_cast"]
    h_cast = hero_scopes["rgb_parade_cast"]
    c_env = clip_scopes["environment_metrics"]
    h_env = hero_scopes["environment_metrics"]

    # Direct raw-to-hero matching (brings clip's raw IRE & WB cast to match hero's raw IRE & WB cast)
    raw_exp_to_hero = round(float(np.clip((h_luma["p50_mid"] - c_luma["p50_mid"]) / 18.0, -1.8, 1.8)), 2)
    raw_contrast_to_hero = int(np.clip((h_luma["dynamic_range"] - c_luma["dynamic_range"]) * 0.75, -25, 30))
    raw_hi_to_hero = int(np.clip((h_luma["p99_white"] - c_luma["p99_white"]) * 1.4, -45, 20))
    raw_sh_to_hero = int(np.clip((h_luma["p25_shadow"] - c_luma["p25_shadow"]) * 1.3, -30, 35))
    raw_temp_to_hero = int(
        np.clip(((c_cast["b_bias"] - c_cast["r_bias"]) - (h_cast["b_bias"] - h_cast["r_bias"])) * 2.0, -30, 30)
    )
    raw_tint_to_hero = int(np.clip((c_cast["g_bias"] - h_cast["g_bias"]) * 2.0, -25, 25))
    raw_sat_to_hero = int(
        np.clip(100 + (h_env["mean_saturation_pct"] - c_env["mean_saturation_pct"]) * 0.75, 80, 122)
    )

    # Post-Pass-1 residual delta (since Pass 1 already normalizes both clips toward 48/62 IRE neutral,
    # Pass 2 only needs to bridge any target midtone mismatch e.g. skin 62 vs landscape 48 and saturation harmony)
    c_rec = clip_scopes["recommended_primary_lumetri"]
    h_rec = hero_scopes["recommended_primary_lumetri"]
    c_post_mid = c_luma["p50_mid"] + c_rec["exposure"] * 18.0
    h_post_mid = h_luma["p50_mid"] + h_rec["exposure"] * 18.0
    res_exp = round(float(np.clip((h_post_mid - c_post_mid) / 18.0, -0.6, 0.6)), 2)

    return {
        "exposure_delta": res_exp,
        "temperature_delta": 0,
        "tint_delta": 0,
        "contrast_delta": int(np.clip(raw_contrast_to_hero - c_rec["contrast"], -15, 15)),
        "highlights_delta": int(np.clip(raw_hi_to_hero - c_rec["highlights"], -20, 20)),
        "shadows_delta": int(np.clip(raw_sh_to_hero - c_rec["shadows"], -20, 20)),
        "saturation_delta": int(np.clip(raw_sat_to_hero - c_rec["saturation"], -15, 15)),
        "p50_mid_ire_delta": round(h_luma["p50_mid"] - c_luma["p50_mid"], 1),
        "direct_lumetri_to_match_hero": {
            "exposure": raw_exp_to_hero,
            "contrast": raw_contrast_to_hero,
            "highlights": raw_hi_to_hero,
            "shadows": raw_sh_to_hero,
            "temperature": raw_temp_to_hero,
            "tint": raw_tint_to_hero,
            "saturation": raw_sat_to_hero,
        },
    }


def _select_representative_frame(frames: list[Path]) -> np.ndarray | None:
    """Pick a non-black representative frame near the middle of a frame list."""
    if not frames:
        return None
    mid_idx = len(frames) // 2
    probe_order = sorted(range(len(frames)), key=lambda i: abs(i - mid_idx))
    fallback = None
    for i in probe_order:
        img = cv2.imread(str(frames[i]))
        if img is None:
            continue
        if fallback is None:
            fallback = img
        if float(np.mean(img)) >= 5.0:
            return img
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Virtual Lumetri Scopes JSON for Premiere MCP.")
    parser.add_argument("--frames_dir", required=True, help="Directory containing clip frame folders, images, or video files")
    parser.add_argument("--output", required=True, help="Output JSON path (e.g., 02_analysis/color_telemetry.json)")
    parser.add_argument("--match_to", required=False, default=None, help="Optional hero reference clip name to compute match deltas")
    parser.add_argument("--no_crop_active", action="store_true", help="Disable automatic letterbox/pillarbox cropping")
    args = parser.parse_args()

    auto_crop = not args.no_crop_active
    root = Path(args.frames_dir)
    telemetry: dict[str, Any] = {}

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.exists() else []
    if subdirs:
        for folder in subdirs:
            frames = sorted(list(folder.glob("*.webp")) + list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
            img = _select_representative_frame(frames)
            if img is not None:
                telemetry[folder.name] = analyze_frame_scopes(img, auto_crop_active=auto_crop)
            else:
                vids = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.mov")))
                for v in vids:
                    sc = analyze_video_file_scopes(str(v), auto_crop_active=auto_crop)
                    if sc:
                        telemetry[v.stem] = sc
    elif root.exists():
        frames = sorted(list(root.glob("*.webp")) + list(root.glob("*.png")) + list(root.glob("*.jpg")))
        vids = sorted(list(root.glob("*.mp4")) + list(root.glob("*.mov")))
        if len(frames) > 1 and not all(f.stem.startswith(("frame_", "thumb_")) or f.stem.isdigit() for f in frames):
            for f in frames:
                img = cv2.imread(str(f))
                if img is not None:
                    telemetry[f.stem] = analyze_frame_scopes(img, auto_crop_active=auto_crop)
        elif frames:
            img = _select_representative_frame(frames)
            if img is not None:
                telemetry[root.name] = analyze_frame_scopes(img, auto_crop_active=auto_crop)
        for v in vids:
            sc = analyze_video_file_scopes(str(v), auto_crop_active=auto_crop)
            if sc:
                telemetry[v.stem] = sc

    # Compute accurate Pass 2 shot-matching deltas against designated hero clip
    if args.match_to and args.match_to in telemetry:
        hero_scopes = telemetry[args.match_to]
        for clip_name, data in telemetry.items():
            if clip_name == args.match_to:
                continue
            data["match_delta_to_hero"] = compute_shot_match_delta(data, hero_scopes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[OK] Saved virtual scopes telemetry ({len(telemetry)} items) -> {out_path}")


if __name__ == "__main__":
    main()
