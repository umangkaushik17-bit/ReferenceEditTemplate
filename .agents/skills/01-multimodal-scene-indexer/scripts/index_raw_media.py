"""Multimodal Raw Media & Subject Indexer.

Indexes a_roll, b_roll, and audio media into 02_analysis/raw_scene_index.json and
02_analysis/raw_transcript_words.json with authoritative stream FPS, active-picture
virtual scopes, environment/color coherence tags, and PyTorch DeepLabV3 person tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

# Reuse virtual_scopes from balancing-primary-color-and-skins skill
_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCOPES_DIR = _SCRIPTS_DIR.parents[1] / "balancing-primary-color-and-skins" / "scripts"
if str(_SCOPES_DIR) not in sys.path:
    sys.path.insert(0, str(_SCOPES_DIR))

from virtual_scopes import analyze_video_file_scopes, fast_sample_keyframes  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".flac")

_DEEPLAB_BUNDLE: tuple[Any, Any] | None = None


def _get_deeplab_person_model() -> tuple[Any, Any] | None:
    """Lazy-load torchvision DeepLabV3 MobileNetV3-Large for COCO class-15 person detection."""
    global _DEEPLAB_BUNDLE
    if _DEEPLAB_BUNDLE is not None:
        return _DEEPLAB_BUNDLE
    try:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
        model.eval()
        _DEEPLAB_BUNDLE = (model, weights.transforms())
        return _DEEPLAB_BUNDLE
    except Exception as exc:
        print(f"[WARN] DeepLabV3 person detector unavailable ({exc}); falling back to skin heuristic.")
        return None


def _probe_media_info(path: str) -> dict[str, Any]:
    """Probe resolution, authoritative frame rate, audio presence, and duration."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    info = json.loads(res.stdout or "{}")
    v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    rate_parts = str(v_stream.get("r_frame_rate", "30/1")).split("/")
    fps = float(rate_parts[0]) / float(rate_parts[1]) if len(rate_parts) == 2 and float(rate_parts[1]) else 30.0
    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))
    dur = float(info.get("format", {}).get("duration", 0.0))

    return {
        "duration": round(dur, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "aspect_ratio": round(width / max(1, height), 3) if height else 0.0,
        "has_audio": a_stream is not None,
    }


def _detect_silences(path: str, total_dur: float) -> list[dict[str, float]]:
    """Run FFmpeg silencedetect (>0.35s below -35dB) and extract active speech spans."""
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-af",
        "silencedetect=noise=-35dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", res.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", res.stderr)]

    speech_segments: list[dict[str, Any]] = []
    cur_t = 0.0
    for i, s_t in enumerate(starts):
        if s_t > cur_t + 0.1:
            speech_segments.append({"start": round(cur_t, 3), "end": round(s_t, 3), "type": "speech"})
        e_t = ends[i] if i < len(ends) else total_dur
        if e_t > s_t:
            speech_segments.append(
                {
                    "start": round(s_t, 3),
                    "end": round(e_t, 3),
                    "type": "silence_cut_candidate",
                    "duration": round(e_t - s_t, 3),
                }
            )
        cur_t = e_t
    if total_dur > cur_t + 0.1:
        speech_segments.append({"start": round(cur_t, 3), "end": round(total_dur, 3), "type": "speech"})
    return speech_segments


def _scan_person_presence(
    video_path: str,
    duration: float,
    sample_times: list[float] | None = None,
    sample_frames: list[np.ndarray] | None = None,
    n_samples: int = 8,
) -> dict[str, Any]:
    """Sample clip frames using DeepLabV3 COCO class-15 to locate foreground human subjects."""
    bundle = _get_deeplab_person_model()
    if bundle is None:
        return {"has_person": False, "max_person_coverage_pct": 0.0, "person_samples": [], "detector": "unavailable"}

    import torch
    from PIL import Image

    model, preprocess = bundle
    if sample_frames is None or sample_times is None:
        sample_times, sample_frames, _ = fast_sample_keyframes(video_path, target_width=480, max_frames=n_samples)
    if not sample_frames:
        return {"has_person": False, "max_person_coverage_pct": 0.0, "person_samples": [], "detector": "no_frames"}

    person_samples: list[dict[str, Any]] = []
    for t_sec, frame in zip(sample_times, sample_frames):
        if frame is None:
            continue
        small = cv2.resize(frame, (384, max(216, int(round(384 * frame.shape[0] / max(1, frame.shape[1]))))))
        pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        inp = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)["out"][0]
        preds = out.argmax(0).byte().cpu().numpy()
        person_mask = preds == 15  # COCO person class
        cov_pct = float(np.mean(person_mask) * 100.0)
        if cov_pct >= 0.8:
            mh, mw = person_mask.shape[:2]
            ys, xs = np.where(person_mask)
            cx_pct = float(np.mean(xs) / max(1, mw) * 100.0)
            cy_pct = float(np.mean(ys) / max(1, mh) * 100.0)
            center_dist = abs(cx_pct - 50.0) / 50.0
            centered_score = round(max(0.0, 1.0 - center_dist), 3)
            person_samples.append(
                {
                    "time_sec": round(float(t_sec), 3),
                    "coverage_pct": round(cov_pct, 2),
                    "center_x_pct": round(cx_pct, 1),
                    "center_y_pct": round(cy_pct, 1),
                    "centered_score": centered_score,
                }
            )

    max_cov = max((p["coverage_pct"] for p in person_samples), default=0.0)
    # Require clear visible subject (>= 1.5% of frame or multiple detections)
    has_person = bool(max_cov >= 1.5 and person_samples)
    best_sample = max(person_samples, key=lambda p: p["coverage_pct"] * (0.6 + 0.4 * p["centered_score"])) if person_samples else None
    valid_times = [p["time_sec"] for p in person_samples if p["coverage_pct"] >= 1.5]
    best_window = [round(min(valid_times), 3), round(max(valid_times), 3)] if valid_times else None

    return {
        "has_person": has_person,
        "max_person_coverage_pct": round(max_cov, 2),
        "best_timestamp_sec": best_sample["time_sec"] if best_sample else None,
        "best_person_window_sec": best_window,
        "person_samples": person_samples,
        "detector": "deeplabv3_mobilenet_v3_large",
    }


def _classify_environment(scopes: dict[str, Any]) -> dict[str, Any]:
    """Derive scene environment category & visual-coherence flags from active scopes."""
    env = scopes.get("environment_metrics", {})
    luma = scopes.get("luma_ire", {})
    green_f = float(env.get("green_fraction", 0.0))
    sky_f = float(env.get("sky_blue_fraction", 0.0))
    warm_f = float(env.get("warm_fraction", 0.0))
    mid_ire = float(luma.get("p50_mid", 50.0))

    if green_f >= 0.08:
        category = "outdoor_green_landscape"
    elif sky_f >= 0.06 or (green_f >= 0.05 and mid_ire >= 40.0):
        category = "outdoor_scenic"
    elif green_f < 0.05 and sky_f < 0.045:
        category = "indoor_or_enclosed"
    else:
        category = "neutral_mixed"

    return {
        "scene_category": category,
        "green_fraction": round(green_f, 4),
        "sky_blue_fraction": round(sky_f, 4),
        "warm_fraction": round(warm_f, 4),
        "mean_saturation_pct": env.get("mean_saturation_pct", 0.0),
        "is_outdoor_green": green_f >= 0.06,
        "likely_indoor": category == "indoor_or_enclosed",
    }


def index_raw_media(
    raw_dir: str = "01_raw_media",
    analysis_dir: str = "02_analysis",
    scan_person: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Index all a_roll, b_roll, and audio files into structured perception manifests."""
    raw_p = Path(raw_dir)
    analysis_p = Path(analysis_dir)
    analysis_p.mkdir(parents=True, exist_ok=True)

    scene_manifest: list[dict[str, Any]] = []
    transcript_manifest: list[dict[str, Any]] = []
    clip_idx = 0

    for role in ("a_roll", "b_roll"):
        role_dir = raw_p / role
        if not role_dir.exists():
            continue
        files = sorted(f for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
        for f in files:
            clip_idx += 1
            media_info = _probe_media_info(str(f))
            speech_segments = _detect_silences(str(f), media_info["duration"]) if media_info["has_audio"] else []
            silence_cuts = [s for s in speech_segments if s["type"] == "silence_cut_candidate"]

            sample_times, sample_frames, _ = fast_sample_keyframes(str(f), target_width=480, max_frames=8)
            scopes = analyze_video_file_scopes(str(f), frames=sample_frames)
            env_tags = _classify_environment(scopes) if scopes else {}
            person_info = (
                _scan_person_presence(str(f), media_info["duration"], sample_times=sample_times, sample_frames=sample_frames)
                if scan_person
                else {"has_person": False}
            )

            scene_manifest.append(
                {
                    "clip_id": f"{role}_{clip_idx:02d}",
                    "role": role,
                    "file_name": f.name,
                    "path": f.as_posix(),
                    "duration_sec": media_info["duration"],
                    "resolution": f"{media_info['width']}x{media_info['height']}",
                    "width": media_info["width"],
                    "height": media_info["height"],
                    "fps": media_info["fps"],
                    "aspect_ratio": media_info["aspect_ratio"],
                    "has_audio": media_info["has_audio"],
                    "silence_count": len(silence_cuts),
                    "environment": env_tags,
                    "person_tracking": person_info,
                    "scopes_baseline": scopes,
                    "speech_timeline": speech_segments,
                }
            )
            transcript_manifest.append(
                {
                    "file": f.name,
                    "role": role,
                    "path": f.as_posix(),
                    "segments": speech_segments,
                }
            )

    audio_files: list[dict[str, Any]] = []
    audio_dir = raw_p / "audio"
    if audio_dir.exists():
        for af in sorted(audio_dir.iterdir()):
            if af.is_file() and af.suffix.lower() in AUDIO_EXTS:
                ai = _probe_media_info(str(af))
                audio_files.append({"file_name": af.name, "path": af.as_posix(), "duration_sec": ai["duration"]})

    # Compute project-wide dominant FPS (so sequence creation / timeline export uses true camera FPS e.g. 60fps)
    fps_values = [c["fps"] for c in scene_manifest if c["fps"] > 0]
    project_max_fps = max(fps_values) if fps_values else 30.0

    manifest_out = {
        "schema_version": "2.0",
        "total_clips": len(scene_manifest),
        "a_roll_count": sum(1 for c in scene_manifest if c["role"] == "a_roll"),
        "b_roll_count": sum(1 for c in scene_manifest if c["role"] == "b_roll"),
        "recommended_sequence_fps": project_max_fps,
        "clips": scene_manifest,
        "audio_files": audio_files,
    }

    (analysis_p / "raw_scene_index.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    (analysis_p / "raw_transcript_words.json").write_text(json.dumps(transcript_manifest, indent=2), encoding="utf-8")
    return manifest_out, transcript_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Index raw a_roll, b_roll, and audio media with CV scopes & subject detection.")
    parser.add_argument("--raw_dir", default="01_raw_media", help="Raw media folder containing a_roll/b_roll/audio")
    parser.add_argument("--analysis_dir", default="02_analysis", help="Output analysis directory")
    parser.add_argument("--no_person_scan", action="store_true", help="Skip DeepLabV3 person scan")
    args = parser.parse_args()

    manifest, _ = index_raw_media(args.raw_dir, args.analysis_dir, scan_person=not args.no_person_scan)
    print(f"[OK] Indexed {manifest['total_clips']} video clips (a_roll={manifest['a_roll_count']}, b_roll={manifest['b_roll_count']}), seq_fps={manifest['recommended_sequence_fps']}")
    for c in manifest["clips"]:
        env = c.get("environment", {})
        pt = c.get("person_tracking", {})
        print(
            f"  - [{c['role']}] {c['file_name']} ({c['duration_sec']}s @ {c['fps']}fps {c['resolution']}) "
            f"| env={env.get('scene_category')} (green={env.get('green_fraction')}) "
            f"| person={pt.get('has_person')} ({pt.get('max_person_coverage_pct', 0)}%)"
        )


if __name__ == "__main__":
    main()
