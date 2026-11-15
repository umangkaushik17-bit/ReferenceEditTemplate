"""Multimodal Raw Media, Word-Level ASR Transcript & Subject Indexer (v2.1).

Indexes a_roll, b_roll, and audio media into 02_analysis/raw_scene_index.json and
02_analysis/raw_transcript_words.json with:
- Authoritative rational/VFR-aware stream FPS & resolution
- Active-picture virtual scopes & environment/color coherence tags
- Semantic shot classification (shot scale, camera motion direction/speed, 32-D visual fingerprint)
- Hardware-accelerated (CUDA / MPS / DirectML / multi-core CPU) PyTorch DeepLabV3 person tracking
- Word-level speech transcription (faster-whisper int8 CPU / float16 GPU) with filler-word detection,
  retake pruning, vocal emphasis scoring for punch-in zooms, and FFmpeg silence detection (>0.35s).
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
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

from virtual_scopes import (  # noqa: E402
    analyze_shot_semantics,
    analyze_video_file_scopes,
    fast_sample_keyframes,
    get_compute_device,
    probe_video_stream_info,
)

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".flac")
FILLER_WORDS = {"um", "uh", "erm", "er", "ah", "hmm", "uhh", "umm"}

_DEEPLAB_BUNDLE: tuple[Any, Any, str, Any] | None = None
_WHISPER_BUNDLE: tuple[Any, str] | None = None


def _get_deeplab_person_model() -> tuple[Any, Any, str, Any] | None:
    """Lazy-load torchvision DeepLabV3 MobileNetV3-Large on best available device (CUDA/MPS/DirectML/CPU)."""
    global _DEEPLAB_BUNDLE
    if _DEEPLAB_BUNDLE is not None:
        return _DEEPLAB_BUNDLE
    try:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        dev_name, device = get_compute_device()
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
        if device is not None:
            model = model.to(device)
        model.eval()
        _DEEPLAB_BUNDLE = (model, weights.transforms(), dev_name, device)
        return _DEEPLAB_BUNDLE
    except Exception as exc:
        print(f"[WARN] DeepLabV3 person detector unavailable ({exc}); falling back to skin heuristic.")
        return None


def _get_whisper_model(model_size: str = "base") -> tuple[Any, str] | None:
    """Lazy-load faster-whisper ASR model using int8 on CPU or float16 on CUDA GPU."""
    global _WHISPER_BUNDLE
    if _WHISPER_BUNDLE is not None:
        return _WHISPER_BUNDLE
    try:
        from faster_whisper import WhisperModel

        dev_name, _ = get_compute_device()
        whisper_dev = "cuda" if dev_name == "cuda" else "cpu"
        compute_type = "float16" if whisper_dev == "cuda" else "int8"
        cpu_threads = max(2, min(8, os.cpu_count() or 4))
        try:
            model = WhisperModel(
                model_size,
                device=whisper_dev,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                local_files_only=True,
            )
        except Exception:
            model = WhisperModel(
                model_size,
                device=whisper_dev,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        _WHISPER_BUNDLE = (model, f"{whisper_dev}:{compute_type}")
        return _WHISPER_BUNDLE
    except Exception as exc:
        print(f"[INFO] faster-whisper ASR not active ({exc}); using FFmpeg silence + acoustic VAD.")
        return None


def _probe_media_info(path: str) -> dict[str, Any]:
    """Probe resolution, authoritative frame rate, VFR status, audio presence, and duration."""
    info = probe_video_stream_info(path)
    width = info["width"]
    height = info["height"]
    return {
        "duration": info["duration"],
        "width": width,
        "height": height,
        "fps": info["fps"],
        "is_vfr": info["is_vfr"],
        "aspect_ratio": round(width / max(1, height), 3) if height else 0.0,
        "has_audio": info["has_audio"],
    }


def _detect_silences(path: str, total_dur: float) -> list[dict[str, Any]]:
    """Run FFmpeg silencedetect (>0.35s below -35dB) with -vn (audio-only decode, 50x faster on 4K video)."""
    cmd = [
        "ffmpeg",
        "-vn",
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


def _transcribe_words_and_takes(
    path: str,
    speech_segments: list[dict[str, Any]],
    enable_asr: bool = True,
    whisper_model_size: str = "base",
) -> dict[str, Any]:
    """Extract word-level timestamps, filler words, retake pruning flags, and emphasis words."""
    active_speech_dur = sum(
        float(s["end"]) - float(s["start"]) for s in speech_segments if s.get("type") == "speech"
    )
    if not enable_asr or active_speech_dur < 0.25:
        return {
            "asr_engine": "silence_vad_only",
            "language": None,
            "full_text": "",
            "word_count": 0,
            "words": [],
            "phrases": [],
            "emphasis_words": [],
            "retake_cuts": [],
        }

    bundle = _get_whisper_model(model_size=whisper_model_size)
    if bundle is None:
        return {
            "asr_engine": "silence_vad_only",
            "language": None,
            "full_text": "",
            "word_count": 0,
            "words": [],
            "phrases": [],
            "emphasis_words": [],
            "retake_cuts": [],
        }

    model, backend_tag = bundle
    try:
        segments_gen, info = model.transcribe(
            path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
        )
        words_out: list[dict[str, Any]] = []
        phrases_out: list[dict[str, Any]] = []
        for seg in segments_gen:
            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue
            seg_words = []
            for w in seg.words or []:
                w_clean = w.word.strip()
                if not w_clean:
                    continue
                norm = re.sub(r"[^a-zA-Z0-9']+", "", w_clean).lower()
                is_filler = norm in FILLER_WORDS
                dur_w = max(0.04, float(w.end) - float(w.start))
                # Emphasis heuristic: longer articulated content words with high confidence
                emphasis = round(
                    min(1.0, (0.55 * float(w.probability) + 0.45 * min(1.0, dur_w / 0.42)))
                    if (len(norm) >= 4 and not is_filler)
                    else 0.2,
                    3,
                )
                w_entry = {
                    "word": w_clean,
                    "normalized": norm,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "duration": round(dur_w, 3),
                    "probability": round(float(w.probability), 3),
                    "is_filler": is_filler,
                    "emphasis_score": emphasis,
                }
                words_out.append(w_entry)
                seg_words.append(w_entry)

            phrases_out.append(
                {
                    "phrase_id": len(phrases_out) + 1,
                    "text": seg_text,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "avg_logprob": round(float(seg.avg_logprob), 3),
                    "no_speech_prob": round(float(seg.no_speech_prob), 3),
                    "is_retake_candidate": False,
                    "keep_take": True,
                    "j_cut_lead_frames": 5,
                    "l_cut_trail_frames": 5,
                    "words": seg_words,
                }
            )

        # Detect repeated takes: if consecutive phrases have high text similarity (>= 0.68),
        # mark the earlier attempt as a retake candidate and keep the later/confident take.
        retake_cuts: list[dict[str, Any]] = []
        for i in range(len(phrases_out) - 1):
            t_cur = re.sub(r"[^a-z0-9 ]+", "", phrases_out[i]["text"].lower()).strip()
            t_nxt = re.sub(r"[^a-z0-9 ]+", "", phrases_out[i + 1]["text"].lower()).strip()
            if len(t_cur) >= 6 and len(t_nxt) >= 6:
                sim = SequenceMatcher(None, t_cur, t_nxt).ratio()
                prefix_match = t_nxt.startswith(t_cur[: min(12, len(t_cur))])
                if sim >= 0.68 or prefix_match:
                    phrases_out[i]["is_retake_candidate"] = True
                    phrases_out[i]["keep_take"] = False
                    retake_cuts.append(
                        {
                            "discard_phrase_id": phrases_out[i]["phrase_id"],
                            "discard_start": phrases_out[i]["start"],
                            "discard_end": phrases_out[i]["end"],
                            "kept_phrase_id": phrases_out[i + 1]["phrase_id"],
                            "similarity": round(sim, 3),
                        }
                    )

        emphasis_words = [
            w for w in words_out if w["emphasis_score"] >= 0.72 and not w["is_filler"]
        ]
        return {
            "asr_engine": f"faster_whisper_{whisper_model_size} ({backend_tag})",
            "language": getattr(info, "language", None),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            "full_text": " ".join(p["text"] for p in phrases_out if p["keep_take"]),
            "word_count": len(words_out),
            "words": words_out,
            "phrases": phrases_out,
            "emphasis_words": emphasis_words[:50],
            "retake_cuts": retake_cuts,
        }
    except Exception as exc:
        return {
            "asr_engine": f"fallback_silence_only ({exc})",
            "language": None,
            "full_text": "",
            "word_count": 0,
            "words": [],
            "phrases": [],
            "emphasis_words": [],
            "retake_cuts": [],
        }


def _scan_person_presence(
    video_path: str,
    duration: float,
    sample_times: list[float] | None = None,
    sample_frames: list[np.ndarray] | None = None,
    n_samples: int = 8,
) -> dict[str, Any]:
    """Sample clip frames using DeepLabV3 COCO class-15 on GPU/CPU to locate foreground human subjects."""
    bundle = _get_deeplab_person_model()
    if bundle is None:
        return {"has_person": False, "max_person_coverage_pct": 0.0, "person_samples": [], "detector": "unavailable"}

    import torch
    from PIL import Image

    model, preprocess, dev_name, device = bundle
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
        if device is not None:
            inp = inp.to(device)
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
            top_head_margin_pct = round(float(ys.min()) / max(1, mh) * 100.0, 2)
            head_safe = bool(top_head_margin_pct >= 2.0)
            center_dist = abs(cx_pct - 50.0) / 50.0
            centered_score = round(max(0.0, 1.0 - center_dist), 3)
            person_samples.append(
                {
                    "time_sec": round(float(t_sec), 3),
                    "coverage_pct": round(cov_pct, 2),
                    "center_x_pct": round(cx_pct, 1),
                    "center_y_pct": round(cy_pct, 1),
                    "top_head_margin_pct": top_head_margin_pct,
                    "head_safe": head_safe,
                    "centered_score": centered_score,
                }
            )

    max_cov = max((p["coverage_pct"] for p in person_samples), default=0.0)
    has_person = bool(max_cov >= 1.5 and person_samples)
    head_safe_samples = [p for p in person_samples if p.get("head_safe", True) and p["coverage_pct"] >= 1.5]
    cand_pool = head_safe_samples if head_safe_samples else person_samples
    best_sample = max(cand_pool, key=lambda p: p["coverage_pct"] * (0.6 + 0.4 * p["centered_score"])) if cand_pool else None
    valid_times = [p["time_sec"] for p in cand_pool if p["coverage_pct"] >= 1.5]
    best_window = [round(min(valid_times), 3), round(max(valid_times), 3)] if valid_times else None

    return {
        "has_person": has_person,
        "max_person_coverage_pct": round(max_cov, 2),
        "best_timestamp_sec": best_sample["time_sec"] if best_sample else None,
        "best_person_window_sec": best_window,
        "person_samples": person_samples,
        "detector": f"deeplabv3_mobilenet_v3_large ({dev_name})",
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
    transcribe_speech: bool = True,
    whisper_model: str = "base",
    force_reindex: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Index all a_roll, b_roll, and audio files into structured perception & word-level ASR manifests."""
    raw_p = Path(raw_dir)
    analysis_p = Path(analysis_dir)
    analysis_p.mkdir(parents=True, exist_ok=True)

    cached_scenes: dict[str, dict[str, Any]] = {}
    cached_transcripts: dict[str, dict[str, Any]] = {}
    scene_index_file = analysis_p / "raw_scene_index.json"
    transcript_file = analysis_p / "raw_transcript_words.json"
    if not force_reindex and scene_index_file.exists():
        try:
            old_scene_data = json.loads(scene_index_file.read_text(encoding="utf-8"))
            for c in old_scene_data.get("clips", []):
                if c.get("file_name") and c.get("shot_semantics") and c.get("person_tracking"):
                    cached_scenes[f"{c.get('role')}/{c.get('file_name')}"] = c
        except Exception:
            pass
    if not force_reindex and transcript_file.exists():
        try:
            old_tr_data = json.loads(transcript_file.read_text(encoding="utf-8"))
            for t in old_tr_data:
                if t.get("file"):
                    cached_transcripts[f"{t.get('role')}/{t.get('file')}"] = t
        except Exception:
            pass

    dev_name, _ = get_compute_device()
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
            cache_key = f"{role}/{f.name}"
            if cache_key in cached_scenes and cache_key in cached_transcripts:
                cached_clip = dict(cached_scenes[cache_key])
                cached_clip["clip_id"] = f"{role}_{clip_idx:02d}"
                cached_clip["path"] = f.as_posix()
                scene_manifest.append(cached_clip)
                cached_tr = dict(cached_transcripts[cache_key])
                cached_tr["path"] = f.as_posix()
                transcript_manifest.append(cached_tr)
                continue

            media_info = _probe_media_info(str(f))
            speech_segments = _detect_silences(str(f), media_info["duration"]) if media_info["has_audio"] else []
            silence_cuts = [s for s in speech_segments if s["type"] == "silence_cut_candidate"]

            # Run word-level ASR primarily on a_roll (or b_roll when speech is enabled)
            asr_data = (
                _transcribe_words_and_takes(
                    str(f),
                    speech_segments,
                    enable_asr=transcribe_speech and (role == "a_roll" or media_info["has_audio"]),
                    whisper_model_size=whisper_model,
                )
                if media_info["has_audio"]
                else {
                    "asr_engine": "no_audio_stream",
                    "full_text": "",
                    "word_count": 0,
                    "words": [],
                    "phrases": [],
                    "emphasis_words": [],
                    "retake_cuts": [],
                }
            )

            sample_times, sample_frames, _ = fast_sample_keyframes(str(f), target_width=480, max_frames=8)
            scopes = analyze_video_file_scopes(str(f), frames=sample_frames)
            env_tags = _classify_environment(scopes) if scopes else {}
            shot_semantics = analyze_shot_semantics(sample_frames, auto_crop_active=True)
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
                    "is_vfr": media_info["is_vfr"],
                    "aspect_ratio": media_info["aspect_ratio"],
                    "has_audio": media_info["has_audio"],
                    "silence_count": len(silence_cuts),
                    "word_count": asr_data.get("word_count", 0),
                    "environment": env_tags,
                    "shot_semantics": shot_semantics,
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
                    "asr_engine": asr_data.get("asr_engine"),
                    "language": asr_data.get("language"),
                    "full_text": asr_data.get("full_text", ""),
                    "word_count": asr_data.get("word_count", 0),
                    "words": asr_data.get("words", []),
                    "phrases": asr_data.get("phrases", []),
                    "emphasis_words": asr_data.get("emphasis_words", []),
                    "retake_cuts": asr_data.get("retake_cuts", []),
                    "segments": speech_segments,
                }
            )

    audio_files: list[dict[str, Any]] = []
    audio_dir = raw_p / "audio"
    if audio_dir.exists():
        for af in sorted(audio_dir.iterdir()):
            if af.is_file() and af.suffix.lower() in AUDIO_EXTS:
                ai = _probe_media_info(str(af))
                sp_segs = _detect_silences(str(af), ai["duration"])
                asr_af = _transcribe_words_and_takes(
                    str(af),
                    sp_segs,
                    enable_asr=transcribe_speech,
                    whisper_model_size=whisper_model,
                )
                audio_files.append(
                    {
                        "file_name": af.name,
                        "path": af.as_posix(),
                        "duration_sec": ai["duration"],
                        "word_count": asr_af.get("word_count", 0),
                    }
                )
                transcript_manifest.append(
                    {
                        "file": af.name,
                        "role": "audio",
                        "path": af.as_posix(),
                        "asr_engine": asr_af.get("asr_engine"),
                        "language": asr_af.get("language"),
                        "full_text": asr_af.get("full_text", ""),
                        "word_count": asr_af.get("word_count", 0),
                        "words": asr_af.get("words", []),
                        "phrases": asr_af.get("phrases", []),
                        "emphasis_words": asr_af.get("emphasis_words", []),
                        "retake_cuts": asr_af.get("retake_cuts", []),
                        "segments": sp_segs,
                    }
                )

    fps_values = [c["fps"] for c in scene_manifest if c["fps"] > 0]
    project_max_fps = max(fps_values) if fps_values else 30.0

    manifest_out = {
        "schema_version": "2.1",
        "compute_device": dev_name,
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
    parser = argparse.ArgumentParser(description="Index raw a_roll, b_roll, and audio media with CV scopes, ASR words & subject detection.")
    parser.add_argument("--raw_dir", default="01_raw_media", help="Raw media folder containing a_roll/b_roll/audio")
    parser.add_argument("--analysis_dir", default="02_analysis", help="Output analysis directory")
    parser.add_argument("--no_person_scan", action="store_true", help="Skip DeepLabV3 person scan")
    parser.add_argument("--no_transcribe", action="store_true", help="Skip faster-whisper word-level ASR transcription")
    parser.add_argument("--whisper_model", default="base", help="faster-whisper model size (tiny, base, small)")
    args = parser.parse_args()

    manifest, _ = index_raw_media(
        args.raw_dir,
        args.analysis_dir,
        scan_person=not args.no_person_scan,
        transcribe_speech=not args.no_transcribe,
        whisper_model=args.whisper_model,
    )
    print(
        f"[OK] Indexed {manifest['total_clips']} video clips "
        f"(a_roll={manifest['a_roll_count']}, b_roll={manifest['b_roll_count']}), "
        f"seq_fps={manifest['recommended_sequence_fps']}, device={manifest['compute_device']}"
    )
    for c in manifest["clips"]:
        env = c.get("environment", {})
        pt = c.get("person_tracking", {})
        sem = c.get("shot_semantics", {})
        print(
            f"  - [{c['role']}] {c['file_name']} ({c['duration_sec']}s @ {c['fps']}fps {c['resolution']}) "
            f"| env={env.get('scene_category')} (green={env.get('green_fraction')}) "
            f"| shot={sem.get('shot_scale')}/{sem.get('camera_motion')} "
            f"| person={pt.get('has_person')} ({pt.get('max_person_coverage_pct', 0)}%) "
            f"| words={c.get('word_count', 0)}"
        )


if __name__ == "__main__":
    main()
