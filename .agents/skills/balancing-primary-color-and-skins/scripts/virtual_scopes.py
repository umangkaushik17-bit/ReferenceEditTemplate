import argparse
import json
from pathlib import Path
import re
import subprocess
import cv2
import numpy as np

def extract_active_region(img_bgr: np.ndarray, black_thresh: int = 14) -> tuple[np.ndarray, dict]:
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


def analyze_frame_scopes(img_bgr: np.ndarray, auto_crop_active: bool = True) -> dict:
    active_info = {"cropped": False, "box": [0, 0, img_bgr.shape[1], img_bgr.shape[0]], "full_size": [img_bgr.shape[1], img_bgr.shape[0]]}
    if auto_crop_active:
        img_bgr, active_info = extract_active_region(img_bgr)
    img_float = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img_float)

    # Rec.709 Luma mapped to 0-100 IRE
    luma_ire = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 100.0
    p01, p25, p50, p75, p99 = np.percentile(luma_ire, [1, 25, 50, 75, 99])

    # Neutral Midtone RGB Parade Cast Detection (low-saturation pixels between 25 and 80 IRE)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] / 255.0
    neutral_mask = (luma_ire > 25.0) & (luma_ire < 80.0) & (sat < 0.22)
    if np.count_nonzero(neutral_mask) < 200:
        neutral_mask = (luma_ire > 20.0) & (luma_ire < 85.0)

    r_mean = float(np.mean(r[neutral_mask]) * 100.0)
    g_mean = float(np.mean(g[neutral_mask]) * 100.0)
    b_mean = float(np.mean(b[neutral_mask]) * 100.0)
    avg_rgb = (r_mean + g_mean + b_mean) / 3.0

    r_bias = round(r_mean - avg_rgb, 2)
    g_bias = round(g_mean - avg_rgb, 2)
    b_bias = round(b_mean - avg_rgb, 2)

    # Skin-Tone Detection & Vectorscope I-Line Check (YCrCb space)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([40, 135, 85]), np.array([235, 175, 130]))
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
    temperature = int(np.clip((b_bias - r_bias) * 2.2, -35, 35))
    tint = int(np.clip(g_bias * 2.5, -30, 30))
    mean_sat = float(np.mean(sat) * 100.0)
    saturation = int(np.clip(100 + (32.0 - mean_sat) * 0.6, 85, 118))

    # Visual theme / environment cues from HSV (OpenCV H in 0..179, S/V in 0..255)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    green_mask = (h_ch >= 30) & (h_ch <= 92) & (s_ch >= 32) & (v_ch >= 28)
    sky_mask = (h_ch >= 90) & (h_ch <= 130) & (s_ch >= 25) & (v_ch >= 75)
    warm_mask = ((h_ch <= 24) | (h_ch >= 165)) & (s_ch >= 38) & (v_ch >= 35)

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
            "green_fraction": round(float(np.mean(green_mask)), 4),
            "sky_blue_fraction": round(float(np.mean(sky_mask)), 4),
            "warm_fraction": round(float(np.mean(warm_mask)), 4),
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


def fast_sample_keyframes(
    video_path: str,
    target_width: int = 480,
    max_frames: int = 16,
) -> tuple[list[float], list[np.ndarray], dict]:
    """Decode only container keyframes via FFmpeg pipe (~10-40x faster than random 4K seeks)."""
    p = str(video_path)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            p,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    info = json.loads(probe.stdout or "{}")
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    src_w = int(v_stream.get("width", 1920))
    src_h = int(v_stream.get("height", 1080))
    rate_parts = str(v_stream.get("r_frame_rate", "30/1")).split("/")
    fps = float(rate_parts[0]) / float(rate_parts[1]) if len(rate_parts) == 2 and float(rate_parts[1]) else 30.0
    dur = float(info.get("format", {}).get("duration", 0.0))

    out_w = max(64, int(target_width) // 2 * 2)
    out_h = max(64, int(round(src_h * out_w / max(1, src_w))) // 2 * 2)

    cmd = [
        "ffmpeg",
        "-skip_frame",
        "nokey",
        "-i",
        p,
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
    if n_decoded == 0:
        return [], [], {"width": src_w, "height": src_h, "fps": round(fps, 3), "duration": round(dur, 3)}

    raw_arr = np.frombuffer(res.stdout[: n_decoded * frame_bytes], dtype=np.uint8).reshape((n_decoded, out_h, out_w, 3))
    if len(pts_list) < n_decoded:
        pts_list = [i * (dur / max(1, n_decoded)) for i in range(n_decoded)]
    else:
        pts_list = pts_list[:n_decoded]

    # Uniformly thin down to max_frames while retaining first and last keyframes
    if n_decoded > max_frames > 1:
        keep = np.linspace(0, n_decoded - 1, max_frames, dtype=int)
        frames = [raw_arr[int(i)].copy() for i in keep]
        times = [round(pts_list[int(i)], 3) for i in keep]
    else:
        frames = [raw_arr[i].copy() for i in range(n_decoded)]
        times = [round(pts_list[i], 3) for i in range(n_decoded)]

    return times, frames, {"width": src_w, "height": src_h, "fps": round(fps, 3), "duration": round(dur, 3)}


def analyze_video_file_scopes(
    video_path: str,
    frames: list[np.ndarray] | None = None,
    auto_crop_active: bool = True,
) -> dict:
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
    # Pick sample closest to median p50 midtone IRE so exposure metrics match a typical frame
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


def main():
    parser = argparse.ArgumentParser(description="Generate Virtual Lumetri Scopes JSON for Premiere MCP.")
    parser.add_argument("--frames_dir", required=True, help="Directory containing clip frame folders or images")
    parser.add_argument("--output", required=True, help="Output JSON path (e.g., 02_analysis/color_telemetry.json)")
    parser.add_argument("--match_to", required=False, default=None, help="Optional hero reference clip name to compute match deltas")
    parser.add_argument("--no_crop_active", action="store_true", help="Disable automatic letterbox/pillarbox cropping")
    args = parser.parse_args()

    auto_crop = not args.no_crop_active
    root = Path(args.frames_dir)
    telemetry = {}

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if subdirs:
        for folder in subdirs:
            frames = sorted(list(folder.glob("*.webp")) + list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
            img = _select_representative_frame(frames)
            if img is not None:
                telemetry[folder.name] = analyze_frame_scopes(img, auto_crop_active=auto_crop)
    else:
        frames = sorted(list(root.glob("*.webp")) + list(root.glob("*.png")) + list(root.glob("*.jpg")))
        if len(frames) > 1 and not all(f.stem.startswith(("frame_", "thumb_")) or f.stem.isdigit() for f in frames):
            for f in frames:
                img = cv2.imread(str(f))
                if img is not None:
                    telemetry[f.stem] = analyze_frame_scopes(img, auto_crop_active=auto_crop)
        else:
            img = _select_representative_frame(frames)
            if img is not None:
                telemetry[root.name] = analyze_frame_scopes(img, auto_crop_active=auto_crop)

    # If match_to is requested and in telemetry, compute residual offsets
    if args.match_to and args.match_to in telemetry:
        hero = telemetry[args.match_to]["recommended_primary_lumetri"]
        hero_luma = telemetry[args.match_to]["luma_ire"]
        for clip_name, data in telemetry.items():
            if clip_name == args.match_to:
                continue
            clip_rec = data["recommended_primary_lumetri"]
            clip_luma = data["luma_ire"]
            data["match_delta_to_hero"] = {
                "exposure_delta": round(clip_rec["exposure"] - hero["exposure"], 2),
                "temperature_delta": clip_rec["temperature"] - hero["temperature"],
                "tint_delta": clip_rec["tint"] - hero["tint"],
                "contrast_delta": clip_rec["contrast"] - hero["contrast"],
                "highlights_delta": clip_rec["highlights"] - hero["highlights"],
                "shadows_delta": clip_rec["shadows"] - hero["shadows"],
                "saturation_delta": clip_rec["saturation"] - hero["saturation"],
                "p50_mid_ire_delta": round(hero_luma["p50_mid"] - clip_luma["p50_mid"], 1),
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

if __name__ == "__main__":
    main()

