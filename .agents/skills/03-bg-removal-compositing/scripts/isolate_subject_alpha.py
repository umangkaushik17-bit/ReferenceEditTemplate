"""Foreground Subject Isolation & Temporally-Smoothed Silhouette Compositing (v2.1).

Supports three compositing modes without any silent dummy fallback:
  - alpha_cutout     : ProRes 4444 (.mov, yuva444p10le) with edge-decontaminated straight alpha for V1/V2/V3 depth titles.
  - silhouette_black : Solid white (#FFFFFF) foreground subject over pure black (#000000) background.
  - silhouette_live  : Solid white (#FFFFFF) foreground subject composited over original live background.

Features:
  - Hardware auto-detection (CUDA / MPS / DirectML / multi-core CPU)
  - Supports PyTorch DeepLabV3 (`mobilenet`, `resnet50`) and ultra-fast CPU `mediapipe` segmentation
  - Motion-adaptive temporal alpha smoothing (eliminates frame-to-frame matte chatter)
  - Foreground edge color decontamination (removes background sky/foliage halos on feathered alpha edges)
  - Direct FFmpeg rawvideo stdin pipe encoding (zero gigabytes of temporary PNGs written to disk)
  - Subject presence verification (`>= min_coverage_pct`) so empty scenery shots fail loudly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCOPES_DIR = _SCRIPTS_DIR.parents[1] / "balancing-primary-color-and-skins" / "scripts"
if str(_SCOPES_DIR) not in sys.path:
    sys.path.insert(0, str(_SCOPES_DIR))

from virtual_scopes import get_compute_device  # noqa: E402


def _load_segmentation_backend(backbone: str = "mobilenet") -> dict[str, Any]:
    """Load PyTorch DeepLabV3 (GPU/CPU) or MediaPipe Selfie Segmentation."""
    dev_name, device = get_compute_device()

    if backbone == "mediapipe":
        try:
            import mediapipe as mp

            mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            return {"type": "mediapipe", "segmenter": mp_seg, "device_name": "cpu_mediapipe", "device": None}
        except Exception as exc:
            print(f"[WARN] MediaPipe unavailable ({exc}); falling back to DeepLabV3 mobilenet.")
            backbone = "mobilenet"

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

    if device is not None:
        model = model.to(device)
    model.eval()
    return {
        "type": "deeplab",
        "model": model,
        "preprocess": weights.transforms(),
        "torch": torch,
        "device_name": dev_name,
        "device": device,
    }


def _decontaminate_foreground_edges(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Replace background color bleed in semi-transparent edge pixels (0.02 < alpha < 0.85) with interior foreground color."""
    core_mask = (alpha >= 0.82).astype(np.uint8)
    edge_mask = (alpha > 0.02) & (alpha < 0.82)
    if np.count_nonzero(core_mask) < 32 or not np.any(edge_mask):
        return frame_bgr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # Propagate interior foreground colors outward into the feather boundary
    dilated_fg = cv2.dilate(frame_bgr * core_mask[..., None], kernel, iterations=2)
    dilated_w = cv2.dilate(core_mask, kernel, iterations=2)
    valid_replace = edge_mask & (dilated_w > 0)

    out = frame_bgr.copy()
    blend_w = np.clip((0.82 - alpha) / 0.80, 0.0, 1.0)[..., None]
    out_f = out.astype(np.float32)
    rep_f = dilated_fg.astype(np.float32)
    out[valid_replace] = np.clip(
        (1.0 - blend_w[valid_replace]) * out_f[valid_replace] + blend_w[valid_replace] * rep_f[valid_replace],
        0,
        255,
    ).astype(np.uint8)
    return out


def _predict_person_alpha(
    frame_bgr: np.ndarray,
    backend: dict[str, Any],
    infer_width: int = 480,
    feather_px: int = 3,
    prev_alpha: np.ndarray | None = None,
    temporal_smooth: float = 0.28,
) -> np.ndarray:
    """Return temporally stabilized soft float32 alpha matte in [0.0, 1.0] at native frame resolution."""
    orig_h, orig_w = frame_bgr.shape[:2]

    if backend["type"] == "mediapipe":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = backend["segmenter"].process(rgb)
        raw_mask = res.segmentation_mask.astype(np.float32) if res.segmentation_mask is not None else np.zeros((orig_h, orig_w), dtype=np.float32)
        alpha_full = cv2.resize(raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        from PIL import Image

        torch_mod = backend["torch"]
        model = backend["model"]
        preprocess = backend["preprocess"]
        device = backend["device"]

        scale_h = max(64, int(round(orig_h * infer_width / max(1, orig_w))))
        small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (infer_width, scale_h)), cv2.COLOR_BGR2RGB)
        inp = preprocess(Image.fromarray(small_rgb)).unsqueeze(0)
        if device is not None:
            inp = inp.to(device)

        with torch_mod.no_grad():
            logits = model(inp)["out"][0]  # (21, H, W)
            probs = torch_mod.softmax(logits, dim=0)
            person_prob = probs[15].detach().cpu().numpy()
            cls_map = logits.argmax(0).detach().cpu().numpy()
            mask = np.where((cls_map == 15) | (person_prob >= 0.48), person_prob, 0.0).astype(np.float32)

        alpha_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Motion-adaptive temporal EMA stabilization: suppresses boundary flicker on slow/steady movement
    # while allowing fast motion (>0.35 delta) to update immediately
    if prev_alpha is not None and prev_alpha.shape == alpha_full.shape and temporal_smooth > 0.0:
        diff = np.abs(alpha_full - prev_alpha)
        adaptive_w = np.where(diff < 0.32, temporal_smooth, temporal_smooth * 0.25).astype(np.float32)
        alpha_full = (1.0 - adaptive_w) * alpha_full + adaptive_w * prev_alpha

    binary = (alpha_full >= 0.42).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    k = max(1, feather_px * 2 + 1)
    soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (k, k), 0)
    return np.clip(soft, 0.0, 1.0)


def _complete_person_head_and_lcc(alpha: np.ndarray, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Keep only the largest connected person component, complete dark hair/hat in head circle,
    and return (cleaned_alpha, top_head_margin_pct).
    """
    h, w = alpha.shape[:2]
    bw = (alpha >= 0.35).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n_lbl <= 1:
        return alpha, 100.0

    best_i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    clean_a = np.where(labels == best_i, alpha, 0.0).astype(np.float32)
    bx, by, bw_box, bh_box, _ = stats[best_i]

    # Gently close small gaps inside the existing head/hat top without grabbing background trees outside the head contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_a = cv2.morphologyEx(clean_a, cv2.MORPH_CLOSE, kernel)

    ys = np.where(clean_a >= 0.40)[0]
    top_margin_pct = (float(ys.min()) / float(max(1, h))) * 100.0 if ys.size > 0 else 100.0
    return clean_a, round(top_margin_pct, 2)


def enforce_depth_sandwich_width_ratio(
    alpha: np.ndarray,
    text_width_pct: float = 25.0,
    max_occlusion_ratio: float = 0.52,
) -> np.ndarray:
    """Constrain depth-sandwich foreground alpha width so the subject stands in the center of the word
    while leaving both outer letters (e.g. 'M'/'E', 'T'/'O', 'ove' of 'Love') 100% readable on either side.
    """
    h, w = alpha.shape[:2]
    max_subj_sigma = max(0.035, (text_width_pct / 100.0) * max_occlusion_ratio * 0.5)
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys, x_idx = np.where(alpha >= 0.35)
    cx_norm = float(np.mean(x_idx)) / float(max(1, w)) if x_idx.size > 0 else 0.50
    col_envelope = np.exp(-0.5 * ((xs - cx_norm) / max_subj_sigma) ** 2).astype(np.float32)
    return alpha * col_envelope


def render_clean_white_flash_cutout(
    frame_bgr: np.ndarray,
    alpha: np.ndarray,
    flash_target: str = "subject",
) -> np.ndarray:
    """Render a clean Full-White Flash (#FAFBF2) on either the subject silhouette or the background.
    Solidifies the interior of the person mask (>= 0.36 -> 1.0 with 3px antialiased edge) so silhouettes
    are 100% opaque solid white with zero shirt/face bleed-through and zero negative inversion.
    """
    a2 = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    core = (a2 >= 0.58).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel)
    solid_a = (cv2.GaussianBlur(core, (3, 3), 0.8).astype(np.float32) / 255.0)[..., None]

    fr_f = frame_bgr.astype(np.float32)
    if flash_target == "background":
        white_bg = np.full_like(fr_f, (235.0, 247.0, 245.0))  # #F5F7EB warm cream-white
        bright_subj = np.clip(fr_f * 1.04 + 4.0, 0.0, 255.0)
        return np.clip(white_bg * (1.0 - solid_a) + bright_subj * solid_a, 0, 255).astype(np.uint8)
    # 100% opaque solid warm-white subject silhouette (#FAFBF2) with crisp antialiased edge
    white_subj = np.full_like(fr_f, (242.0, 251.0, 250.0))
    return np.clip(fr_f * (1.0 - solid_a) + white_subj * solid_a, 0, 255).astype(np.uint8)


def isolate_subject(
    input_video: str,
    output_mov: str,
    mode: str = "alpha_cutout",
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    backbone: str = "resnet50",
    min_coverage_pct: float = 0.5,
    temporal_smooth: float = 0.28,
    verify_head_safe: bool = True,
    min_head_margin_pct: float = 1.2,
) -> dict[str, Any]:
    """Segment foreground human subject and encode ProRes 4444 RGBA or silhouette transition clip via FFmpeg pipe."""
    in_p = Path(input_video)
    if not in_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    out_p = Path(output_mov)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_p = out_p.parent / f"._tmp_encode_{out_p.name}"

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

    backend = _load_segmentation_backend(backbone=backbone)
    infer_w = 512 if backend["device_name"] in ("cuda", "mps", "directml") else 448

    in_pix_fmt = "bgra" if mode == "alpha_cutout" else "bgr24"
    out_pix_fmt = "yuva444p10le" if mode == "alpha_cutout" else "yuv420p"
    codec = "prores_ks" if out_p.suffix.lower() == ".mov" else "libx264"

    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        in_pix_fmt,
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-c:v",
        codec,
    ]
    if codec == "prores_ks":
        ffmpeg_cmd += ["-profile:v", "4"]
    ffmpeg_cmd += ["-pix_fmt", out_pix_fmt, str(tmp_out_p)]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    coverages: list[float] = []
    head_margins: list[float] = []
    written = 0
    prev_alpha: np.ndarray | None = None
    proc: subprocess.Popen[bytes] | None = None

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None

        for _ in range(start_f, end_f):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            alpha = _predict_person_alpha(
                frame,
                backend,
                infer_width=infer_w,
                prev_alpha=prev_alpha,
                temporal_smooth=temporal_smooth,
            )
            alpha, top_margin_pct = _complete_person_head_and_lcc(alpha, frame)
            prev_alpha = alpha
            cov_pct = float(np.mean(alpha > 0.5) * 100.0)
            coverages.append(cov_pct)
            head_margins.append(top_margin_pct)

            if mode == "silhouette_black":
                white_fg = np.full_like(frame, 255, dtype=np.float32)
                out_bgr = (white_fg * alpha[..., None]).astype(np.uint8)
                proc.stdin.write(out_bgr.tobytes())
            elif mode == "silhouette_live":
                white_fg = np.full_like(frame, 255, dtype=np.float32)
                comp = white_fg * alpha[..., None] + frame.astype(np.float32) * (1.0 - alpha[..., None])
                out_bgr = np.clip(comp, 0, 255).astype(np.uint8)
                proc.stdin.write(out_bgr.tobytes())
            else:
                # Decontaminate edge halo colors + straight (unmultiplied) BGRA for ProRes 4444
                clean_bgr = _decontaminate_foreground_edges(frame, alpha)
                alpha_u8 = (alpha * 255.0).astype(np.uint8)
                bgra = cv2.merge([clean_bgr[:, :, 0], clean_bgr[:, :, 1], clean_bgr[:, :, 2], alpha_u8])
                proc.stdin.write(bgra.tobytes())

            written += 1

        proc.stdin.close()
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg encode failed (exit {ret}): {stderr_bytes.decode('utf-8', errors='ignore')}")
    finally:
        cap.release()
        if proc is not None and proc.poll() is None:
            proc.kill()

    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    max_cov = float(np.max(coverages)) if coverages else 0.0
    med_head_margin = float(np.median(head_margins)) if head_margins else 100.0
    head_safe = bool(med_head_margin >= min_head_margin_pct)
    if written == 0 or max_cov < min_coverage_pct or (verify_head_safe and not head_safe):
        if tmp_out_p.exists():
            tmp_out_p.unlink()
        raise RuntimeError(
            f"[ERROR] Subject cutout failed quality gate in {input_video} [{start_sec:.2f}s..]: "
            f"max_cov={max_cov:.2f}% (min={min_coverage_pct:.2f}%), "
            f"median_top_head_margin={med_head_margin:.2f}% (min={min_head_margin_pct:.2f}%, head_safe={head_safe}). "
            f"Select a frame window where the subject's full head and torso are inside the frame."
        )

    shutil.move(str(tmp_out_p), str(out_p))
    return {
        "output": out_p.as_posix(),
        "mode": mode,
        "device": backend["device_name"],
        "frames": written,
        "fps": round(fps, 3),
        "mean_subject_coverage_pct": round(mean_cov, 2),
        "max_subject_coverage_pct": round(max_cov, 2),
        "median_top_head_margin_pct": round(med_head_margin, 2),
        "head_safe": head_safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate foreground subject using GPU/CPU DeepLabV3 or MediaPipe + ProRes 4444 pipe.")
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
    parser.add_argument("--backbone", choices=["mobilenet", "resnet50", "mediapipe"], default="mobilenet")
    parser.add_argument("--min_coverage_pct", type=float, default=0.5, help="Fail loudly if subject coverage below threshold")
    parser.add_argument("--temporal_smooth", type=float, default=0.28, help="Temporal EMA smoothing weight [0..0.7]")
    args = parser.parse_args()

    res = isolate_subject(
        args.input,
        args.output,
        mode=args.mode,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        backbone=args.backbone,
        min_coverage_pct=args.min_coverage_pct,
        temporal_smooth=args.temporal_smooth,
    )
    print(
        f"[OK] Exported {res['mode']} -> {res['output']} "
        f"({res['frames']}f @ {res['fps']}fps, device={res['device']}, cov={res['mean_subject_coverage_pct']}%)",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
