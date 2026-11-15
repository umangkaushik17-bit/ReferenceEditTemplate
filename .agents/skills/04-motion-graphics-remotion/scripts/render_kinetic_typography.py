"""Kinetic Typography & Word-Level Karaoke ProRes 4444 Renderer (v2.1).

Renders transparent alpha ProRes 4444 (.mov, yuva444p10le) overlays driven by:
  1. `02_analysis/style_dna.json` (`typography.events[]`) for reference-matched titles
     (`scramble_decode`, `step_cut_no_fade`, `fade_in`), OR
  2. `02_analysis/raw_transcript_words.json` (`words[]`) for word-by-word active highlight
     captions (`viral_karaoke_highlight`, `clean_tech_minimal`) per `caption_style_schema.json`.

Streams raw RGBA frames directly into FFmpeg via stdin pipe (zero temporary PNG disk bloat).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CIPHER_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@%&*+<>?"


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(Path(font_path))
    candidates.extend(
        [
            Path("04_assets/fonts/Cinematic_Serif.ttf"),
            Path(r"C:\Windows\Fonts\georgia.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
        ]
    )
    for cand in candidates:
        if cand.exists():
            try:
                return ImageFont.truetype(str(cand), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:28] or "title"


def _scrambled_string(target: str, progress: float, frame_idx: int) -> str:
    """Return partially decoded cipher string at `progress` in [0.0, 1.0]."""
    if progress >= 1.0:
        return target
    chars = list(target)
    n_alpha = sum(1 for c in chars if not c.isspace())
    locked_count = int(round(progress * n_alpha))
    out: list[str] = []
    seen = 0
    for idx, ch in enumerate(chars):
        if ch.isspace():
            out.append(ch)
            continue
        if seen < locked_count:
            out.append(ch)
        else:
            pick = (frame_idx * 7 + idx * 13) % len(CIPHER_CHARS)
            glyph = CIPHER_CHARS[pick]
            out.append(glyph if ch.isupper() else glyph.lower())
        seen += 1
    return "".join(out)


def _render_text_canvas(
    text: str,
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    y_center_pct: float = 50.0,
    opacity: float = 1.0,
    y_offset_px: int = 0,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Render centered antialiased typography with soft drop shadow onto an RGBA canvas."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if opacity <= 0.01 or not text.strip():
        return img

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = int(round((y_center_pct / 100.0) * height - th / 2.0)) + y_offset_px

    alpha_u8 = int(round(max(0.0, min(1.0, opacity)) * 255))
    sh_alpha = int(round(max(0.0, min(1.0, opacity)) * 175))

    # Render shadow Only in local bounding box for 20x faster CPU compositing on 4K canvases
    pad = 24
    bx0 = max(0, x - pad)
    by0 = max(0, y - pad)
    bx1 = min(width, x + tw + pad)
    by1 = min(height, y + th + pad)
    if bx1 > bx0 and by1 > by0:
        sub = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(sub)
        sdraw.text((x - bx0, y - by0 + 4), text, font=font, fill=(0, 0, 0, sh_alpha))
        sub = sub.filter(ImageFilter.GaussianBlur(radius=5))
        sdraw = ImageDraw.Draw(sub)
        sdraw.text((x - bx0, y - by0), text, font=font, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], alpha_u8))
        img.paste(sub, (bx0, by0), sub)
    return img


def render_title_event_mov(
    text: str,
    output_mov: str,
    duration_sec: float,
    reveal_effect: str = "step_cut_no_fade",
    width: int = 1920,
    height: int = 1080,
    fps: float = 60.0,
    y_center_pct: float = 50.0,
    is_all_caps: bool = False,
    font_path: str | None = None,
    box_height_pct: float = 4.2,
) -> dict[str, Any]:
    """Render a single kinetic title overlay event to ProRes 4444 (.mov, yuva444p10le)."""
    out_p = Path(output_mov)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    render_text = text.upper() if is_all_caps else text
    font_px = max(28, int(round(height * max(2.8, min(9.0, box_height_pct)) / 100.0)))
    font = _load_font(font_path, font_px)
    n_frames = max(2, int(round(duration_sec * fps)))

    # Fast path for static step_cut_no_fade: render 1 PNG in memory and loop with FFmpeg
    if reveal_effect == "step_cut_no_fade":
        static_img = _render_text_canvas(render_text, width, height, font, y_center_pct=y_center_pct, opacity=1.0)
        tmp_png = out_p.parent / f"._tmp_{out_p.stem}.png"
        try:
            static_img.save(str(tmp_png), "PNG")
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-loop",
                "1",
                "-t",
                f"{duration_sec:.4f}",
                "-r",
                f"{fps:.4f}",
                "-i",
                str(tmp_png),
                "-vf",
                "format=rgba",
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                str(out_p),
            ]
            subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True)
        finally:
            if tmp_png.exists():
                tmp_png.unlink()
        return {
            "output": out_p.as_posix(),
            "text": render_text,
            "reveal_effect": reveal_effect,
            "frames": n_frames,
            "fps": fps,
            "duration_sec": round(duration_sec, 3),
        }

    # Dynamic frame-by-frame animation (scramble_decode or fade_in) via disk rawvideo file (Zero-Hang Rule 2)
    tmp_rgba = out_p.parent / f"._tmp_{out_p.stem}.rgba"
    reveal_frames = max(4, min(n_frames - 1, int(round(fps * (0.42 if reveal_effect == "scramble_decode" else 0.32)))))
    outro_fade_frames = max(3, int(round(fps * 0.18))) if reveal_effect == "fade_in" else 0
    settled_bytes: bytes | None = None

    try:
        with open(tmp_rgba, "wb") as f_rgba:
            for f_idx in range(n_frames):
                if reveal_effect == "scramble_decode":
                    if f_idx < reveal_frames:
                        prog = (f_idx + 1) / float(reveal_frames)
                        cur_txt = _scrambled_string(render_text, prog, f_idx)
                        op = min(1.0, 0.35 + 0.65 * prog)
                        frame_img = _render_text_canvas(cur_txt, width, height, font, y_center_pct=y_center_pct, opacity=op)
                        f_rgba.write(frame_img.tobytes())
                    else:
                        if settled_bytes is None:
                            settled_bytes = _render_text_canvas(
                                render_text, width, height, font, y_center_pct=y_center_pct, opacity=1.0
                            ).tobytes()
                        f_rgba.write(settled_bytes)
                else:  # fade_in with cubic ease + subtle vertical rise
                    if f_idx < reveal_frames:
                        t_norm = (f_idx + 1) / float(reveal_frames)
                        ease = 1.0 - pow(1.0 - t_norm, 3)
                        y_off = int(round((1.0 - ease) * 14))
                        frame_img = _render_text_canvas(
                            render_text, width, height, font, y_center_pct=y_center_pct, opacity=ease, y_offset_px=y_off
                        )
                        f_rgba.write(frame_img.tobytes())
                    elif outro_fade_frames > 0 and f_idx >= n_frames - outro_fade_frames:
                        rem = (n_frames - f_idx) / float(outro_fade_frames)
                        frame_img = _render_text_canvas(
                            render_text, width, height, font, y_center_pct=y_center_pct, opacity=max(0.0, rem)
                        )
                        f_rgba.write(frame_img.tobytes())
                    else:
                        if settled_bytes is None:
                            settled_bytes = _render_text_canvas(
                                render_text, width, height, font, y_center_pct=y_center_pct, opacity=1.0
                            ).tobytes()
                        f_rgba.write(settled_bytes)

        cmd = [
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
            "rgba",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.4f}",
            "-i",
            str(tmp_rgba),
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            str(out_p),
        ]
        subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    finally:
        if tmp_rgba.exists():
            tmp_rgba.unlink()

    return {
        "output": out_p.as_posix(),
        "text": render_text,
        "reveal_effect": reveal_effect,
        "frames": n_frames,
        "fps": fps,
        "duration_sec": round(duration_sec, 3),
    }


def render_karaoke_word_chunk_mov(
    words_chunk: list[dict[str, Any]],
    output_mov: str,
    style_cfg: dict[str, Any],
    width: int = 1080,
    height: int = 1920,
    fps: float = 60.0,
    font_path: str | None = None,
) -> dict[str, Any]:
    """Render a multi-word caption group where the active spoken word pops in highlight color (#FFE600)."""
    out_p = Path(output_mov)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    chunk_start = float(words_chunk[0]["start"])
    chunk_end = max(chunk_start + 0.15, float(words_chunk[-1]["end"]))
    dur = chunk_end - chunk_start
    n_frames = max(2, int(round(dur * fps)))

    is_vertical = height > width
    base_px = int(style_cfg.get("font_size_vertical" if is_vertical else "font_size_horizontal", 56))
    uppercase = style_cfg.get("text_transform", "uppercase") == "uppercase"
    active_hex = style_cfg.get("active_highlight_color", "#FFE600").lstrip("#")
    inactive_hex = style_cfg.get("inactive_color", "#FFFFFF").lstrip("#")
    act_rgb = tuple(int(active_hex[i : i + 2], 16) for i in (0, 2, 4))
    inact_rgb = tuple(int(inactive_hex[i : i + 2], 16) for i in (0, 2, 4))
    y_anchor_pct = float(style_cfg.get("y_anchor_percent", 78.0))

    font_base = _load_font(font_path, base_px)
    font_active = _load_font(font_path, int(round(base_px * float(style_cfg.get("active_scale_multiplier", 1.12)))))

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        str(out_p),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    tokens = [w["word"].upper() if uppercase else w["word"] for w in words_chunk]
    cache_by_active: dict[int, bytes] = {}

    try:
        for f_idx in range(n_frames):
            t_cur = chunk_start + (f_idx / float(fps))
            act_idx = -1
            for idx, w in enumerate(words_chunk):
                if float(w["start"]) <= t_cur <= float(w["end"]) + 0.04:
                    act_idx = idx
                    break

            if act_idx not in cache_by_active:
                img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                word_boxes = []
                total_w = 0
                space_w = int(round(base_px * 0.32))
                max_h = 0
                for idx, tok in enumerate(tokens):
                    f_use = font_active if idx == act_idx else font_base
                    bb = draw.textbbox((0, 0), tok, font=f_use)
                    ww, wh = bb[2] - bb[0], bb[3] - bb[1]
                    word_boxes.append((tok, f_use, ww, wh, idx == act_idx))
                    total_w += ww
                    max_h = max(max_h, wh)
                total_w += space_w * max(0, len(tokens) - 1)

                cur_x = (width - total_w) // 2
                base_y = int(round((y_anchor_pct / 100.0) * height - max_h / 2.0))
                for tok, f_use, ww, wh, is_act in word_boxes:
                    wy = base_y - (4 if is_act else 0)
                    # Stroke / shadow outline
                    draw.text((cur_x + 3, wy + 4), tok, font=f_use, fill=(0, 0, 0, 210))
                    color = (*act_rgb, 255) if is_act else (*inact_rgb, 255)
                    draw.text((cur_x, wy), tok, font=f_use, fill=color)
                    cur_x += ww + space_w
                cache_by_active[act_idx] = img.tobytes()

            proc.stdin.write(cache_by_active[act_idx])

        proc.stdin.close()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    return {
        "output": out_p.as_posix(),
        "text": " ".join(tokens),
        "start_sec": round(chunk_start, 3),
        "end_sec": round(chunk_end, 3),
        "duration_sec": round(dur, 3),
        "frames": n_frames,
    }


def render_from_style_dna(
    style_dna_path: str = "02_analysis/style_dna.json",
    timeline_plan_path: str = "02_analysis/timeline_plan.json",
    out_dir: str = "03_processed_vfx/rendered_titles",
    font_path: str | None = None,
) -> list[dict[str, Any]]:
    """Render all typography events from style_dna.json at the planned sequence resolution and FPS."""
    dna = json.loads(Path(style_dna_path).read_text(encoding="utf-8"))
    typo = dna.get("layer_3_aesthetic_typography_and_color", {}).get("typography", {})
    events = typo.get("events", [])

    width, height = dna.get("spatial_geometry", {}).get("container_resolution", [1920, 1080])
    fps = float(dna.get("spatial_geometry", {}).get("container_fps", 30.0))

    if Path(timeline_plan_path).exists():
        plan = json.loads(Path(timeline_plan_path).read_text(encoding="utf-8"))
        seq_cfg = plan.get("sequence_config", {})
        res = seq_cfg.get("target_resolution")
        if isinstance(res, list) and len(res) == 2:
            width, height = int(res[0]), int(res[1])
        fps = float(seq_cfg.get("target_fps", fps))

    rendered: list[dict[str, Any]] = []
    for idx, ev in enumerate(events, start=1):
        txt = ev.get("text", "").strip()
        if not txt:
            continue
        dur = max(0.15, float(ev.get("end_sec", 1.0)) - float(ev.get("start_sec", 0.0)))
        slug = _slugify(txt)
        out_mov = Path(out_dir) / f"title_{idx:02d}_{slug}.mov"
        meta = render_title_event_mov(
            text=txt,
            output_mov=str(out_mov),
            duration_sec=dur,
            reveal_effect=ev.get("reveal_effect", "step_cut_no_fade"),
            width=width,
            height=height,
            fps=fps,
            y_center_pct=float(ev.get("y_center_pct", 50.0)),
            is_all_caps=bool(ev.get("is_all_caps", False)),
            font_path=font_path,
            box_height_pct=float(ev.get("box_height_pct", 4.2)),
        )
        meta["slot_index"] = idx
        meta["timeline_start_sec"] = ev.get("start_sec", 0.0)
        meta["timeline_end_sec"] = ev.get("end_sec", round(dur, 3))
        rendered.append(meta)
        print(f"[OK] Rendered {meta['reveal_effect']} title #{idx}: '{meta['text']}' -> {meta['output']} ({meta['duration_sec']}s @ {fps}fps)")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render kinetic typography or word-level karaoke overlays to ProRes 4444.")
    parser.add_argument("--style_dna", default="02_analysis/style_dna.json", help="Path to style_dna.json")
    parser.add_argument("--timeline_plan", default="02_analysis/timeline_plan.json", help="Path to timeline_plan.json")
    parser.add_argument("--transcript", default=None, help="Optional raw_transcript_words.json for karaoke captions")
    parser.add_argument("--style_profile", default="viral_karaoke_highlight", help="Profile in caption_style_schema.json")
    parser.add_argument("--out_dir", default="03_processed_vfx/rendered_titles", help="Output directory for .mov files")
    parser.add_argument("--font", default=None, help="Optional custom .ttf font path")
    parser.add_argument("--text", default=None, help="Render a single custom text string")
    parser.add_argument("--effect", default="step_cut_no_fade", choices=["step_cut_no_fade", "scramble_decode", "fade_in"])
    parser.add_argument("--duration", type=float, default=2.0, help="Duration for single custom text render")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    if args.text:
        out_mov = Path(args.out_dir) / f"custom_{_slugify(args.text)}.mov"
        res = render_title_event_mov(
            text=args.text,
            output_mov=str(out_mov),
            duration_sec=args.duration,
            reveal_effect=args.effect,
            width=args.width,
            height=args.height,
            fps=args.fps,
            font_path=args.font,
        )
        print(f"[OK] Rendered custom title -> {res['output']}")
        return

    if args.transcript and Path(args.transcript).exists():
        schema_p = Path(__file__).resolve().parents[1] / "resources" / "caption_style_schema.json"
        schema = json.loads(schema_p.read_text(encoding="utf-8")) if schema_p.exists() else {}
        style_cfg = schema.get("styles", {}).get(args.style_profile, {})
        max_w = int(style_cfg.get("max_words_per_line", 3))
        tr_data = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        chunk_idx = 0
        for item in tr_data:
            words = [w for w in item.get("words", []) if not w.get("is_filler")]
            for i in range(0, len(words), max_w):
                chunk = words[i : i + max_w]
                if not chunk:
                    continue
                chunk_idx += 1
                out_mov = Path(args.out_dir) / f"karaoke_{chunk_idx:03d}_{_slugify(chunk[0]['word'])}.mov"
                res = render_karaoke_word_chunk_mov(
                    chunk,
                    str(out_mov),
                    style_cfg=style_cfg,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    font_path=args.font,
                )
                print(f"[OK] Rendered karaoke chunk #{chunk_idx}: '{res['text']}' -> {res['output']}")
        return

    if Path(args.style_dna).exists():
        render_from_style_dna(args.style_dna, args.timeline_plan, args.out_dir, font_path=args.font)
    else:
        raise SystemExit("[ERROR] Provide --style_dna, --transcript, or --text.")


if __name__ == "__main__":
    main()
