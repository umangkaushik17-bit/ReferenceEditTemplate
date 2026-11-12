"""Foreground Subject Isolation & Silhouette Compositing (PyTorch DeepLabV3 + FFmpeg ProRes 4444).

Supports three compositing modes without any silent dummy fallback:
  - alpha_cutout     : ProRes 4444 (.mov, yuva444p10le) with real alpha matte for V1/V2/V3 depth titles.
  - silhouette_black : Solid white (#FFFFFF) foreground subject over pure black (#000000) background.
  - silhouette_live  : Solid white (#FFFFFF) foreground subject composited over original live background.

Enforces subject presence verification (>= min_coverage_pct) so empty grass/scenery shots fail loudly
instead of silently generating blank or broken mattes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def _load_deeplab_model(backbone: str = "mobilenet"):
    """Load PyTorch DeepLabV3 segmentation model targeting COCO class 15 (person)."""
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights,
        DeepLabV3_ResNet50_Weights,
        deeplabv3_mobilenet_v3_large,
        deeplabv3_resnet50,
    )

    if backbone == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    else:
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()
    return model, weights.transforms(), torch


def _predict_person_alpha(
    frame_bgr: np.ndarray,
    model,
    preprocess,
    torch_mod,
    infer_width: int = 512,
    feather_px: int = 3,
) -> np.ndarray:
    """Return soft float32 alpha matte in [0.0, 1.0] at native frame resolution."""
    from PIL import Image

    orig_h, orig_w = frame_bgr.shape[:2]
    scale_h = max(64, int(round(orig_h * infer_width / max(1, orig_w))))
    small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (infer_width, scale_h)), cv2.COLOR_BGR2RGB)
    inp = preprocess(Image.fromarray(small_rgb)).unsqueeze(0)

    with torch_mod.no_grad():
        logits = model(inp)["out"][0]  # (21, H, W)
        probs = torch_mod.softmax(logits, dim=0)
        # COCO class 15 = person; combine soft class probability + argmax confidence
        person_prob = probs[15].cpu().numpy()
        cls_map = logits.argmax(0).cpu().numpy()
        mask = np.where((cls_map == 15) | (person_prob >= 0.48), person_prob, 0.0).astype(np.float32)

    alpha_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # Crisp morphological cleanup + feathering
    binary = (alpha_full >= 0.42).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    k = max(1, feather_px * 2 + 1)
    soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (k, k), 0)
    return np.clip(soft, 0.0, 1.0)


def isolate_subject(
    input_video: str,
    output_mov: str,
    mode: str = "alpha_cutout",
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    backbone: str = "mobilenet",
    min_coverage_pct: float = 0.5,
) -> dict:
    """Segment foreground human subject and encode ProRes 4444 RGBA or silhouette transition clip."""
    in_p = Path(input_video)
    if not in_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    out_p = Path(output_mov)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_f = max(0, int(round(start_sec * fps)))
    end_f = min(total_frames, start_f + int(round(duration_sec * fps))) if duration_sec else total_frames
    if end_f <= start_f:
        cap.release()
        raise ValueError(f"Empty frame range [{start_f}, {end_f}] for {input_video}")

    model, preprocess, torch_mod = _load_deeplab_model(backbone=backbone)

    tmp_dir = out_p.parent / f"_tmp_{out_p.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    coverages: list[float] = []
    written = 0

    for f_idx in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        alpha = _predict_person_alpha(frame, model, preprocess, torch_mod)
        cov_pct = float(np.mean(alpha > 0.5) * 100.0)
        coverages.append(cov_pct)

        if mode == "silhouette_black":
            white_fg = np.full_like(frame, 255, dtype=np.float32)
            out_rgb = (white_fg * alpha[..., None]).astype(np.uint8)
            cv2.imwrite(str(tmp_dir / f"frame_{written:05d}.png"), out_rgb)
        elif mode == "silhouette_live":
            white_fg = np.full_like(frame, 255, dtype=np.float32)
            comp = white_fg * alpha[..., None] + frame.astype(np.float32) * (1.0 - alpha[..., None])
            cv2.imwrite(str(tmp_dir / f"frame_{written:05d}.png"), np.clip(comp, 0, 255).astype(np.uint8))
        else:
            # alpha_cutout: premultiply-clean BGRA PNG sequence for ProRes 4444 encoding
            alpha_u8 = (alpha * 255.0).astype(np.uint8)
            bgra = cv2.merge([frame[:, :, 0], frame[:, :, 1], frame[:, :, 2], alpha_u8])
            cv2.imwrite(str(tmp_dir / f"frame_{written:05d}.png"), bgra)

        written += 1

    cap.release()

    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    max_cov = float(np.max(coverages)) if coverages else 0.0
    if written == 0 or max_cov < min_coverage_pct:
        # Clean up temp PNGs and fail loudly rather than writing silent empty/grass matte
        for p in tmp_dir.glob("*.png"):
            p.unlink()
        tmp_dir.rmdir()
        raise RuntimeError(
            f"[ERROR] Subject coverage too low in {input_video} [{start_sec:.2f}s..]: "
            f"max={max_cov:.2f}% < threshold={min_coverage_pct:.2f}%. Select a frame window with a clear person subject."
        )

    pix_fmt = "yuva444p10le" if mode == "alpha_cutout" else "yuv420p"
    codec = "prores_ks" if out_p.suffix.lower() == ".mov" else "libx264"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:.3f}",
        "-i",
        str(tmp_dir / "frame_%05d.png"),
        "-c:v",
        codec,
    ]
    if codec == "prores_ks":
        ffmpeg_cmd += ["-profile:v", "4"]
    ffmpeg_cmd += ["-pix_fmt", pix_fmt, str(out_p)]

    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()

    return {
        "output": out_p.as_posix(),
        "mode": mode,
        "frames": written,
        "fps": round(fps, 3),
        "mean_subject_coverage_pct": round(mean_cov, 2),
        "max_subject_coverage_pct": round(max_cov, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate foreground subject using PyTorch DeepLabV3 + ProRes 4444.")
    parser.add_argument("--input", required=True, help="Source video path")
    parser.add_argument("--output", required=True, help="Output .mov (or .mp4 for opaque silhouette modes)")
    parser.add_argument(
        "--mode",
        choices=["alpha_cutout", "silhouette_black", "silhouette_live"],
        default="alpha_cutout",
        help="Compositing output mode",
    )
    parser.add_argument("--start_sec", type=float, default=0.0, help="Optional trim start in seconds")
    parser.add_argument("--duration_sec", type=float, default=None, help="Optional trim duration in seconds")
    parser.add_argument("--backbone", choices=["mobilenet", "resnet50"], default="mobilenet")
    parser.add_argument("--min_coverage_pct", type=float, default=0.5, help="Fail loudly if subject coverage below threshold")
    args = parser.parse_args()

    res = isolate_subject(
        args.input,
        args.output,
        mode=args.mode,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        backbone=args.backbone,
        min_coverage_pct=args.min_coverage_pct,
    )
    print(f"[OK] Exported {res['mode']} -> {res['output']} ({res['frames']}f @ {res['fps']}fps, cov={res['mean_subject_coverage_pct']}%)")


if __name__ == "__main__":
    main()
