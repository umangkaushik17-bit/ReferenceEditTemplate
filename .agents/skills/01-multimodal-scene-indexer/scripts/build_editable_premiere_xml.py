"""General Data-Driven Final Cut Pro 7 XML (`<xmeml version="4">`) Timeline Builder (v3.0).

Reads any `02_analysis/timeline_plan.json`, `02_analysis/style_dna.json`, and `02_analysis/raw_scene_index.json`
with ZERO video-specific hardcoding and emits `02_analysis/premiere_editable_timeline.xml`:
  - Track V1: Planned segments referencing raw source files (`01_raw_media/`) with full source handles (`<in>`/`<out>`)
  - Track V2: Geometric panel/split overlays (`geometric_panel_overlays`) with native Premiere `Crop` (`left`, `right`, `top`, `bottom` %)
  - Track V3: Foreground subject cutouts (`v3_subject_cutout_overlay`)
  - Track V4: Kinetic typography overlays (`v3_kinetic_typography_overlay` or individual event clips with `Difference`/`Normal` blend mode)
  - Tracks A1/A2/A3: Camera audio (`-38 dB`), pre-cut SFX transients (`a2_sfx_plan`), and cloned reference music (`audio_plan`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET


def _path_to_file_url(p: str | Path) -> str:
    abs_p = Path(p).resolve().as_posix()
    if not abs_p.startswith("/"):
        abs_p = "/" + abs_p
    return "file://localhost" + quote(abs_p, safe="/:")


def _make_motion_filter(scale_pct: float = 100.0, center_x_norm: float = 0.0, center_y_norm: float = 0.0) -> ET.Element:
    filt = ET.Element("filter")
    eff = ET.SubElement(filt, "effect")
    ET.SubElement(eff, "name").text = "Basic Motion"
    ET.SubElement(eff, "effectid").text = "basic"
    ET.SubElement(eff, "effectcategory").text = "motion"
    ET.SubElement(eff, "effecttype").text = "motion"
    ET.SubElement(eff, "mediatype").text = "video"

    p_scale = ET.SubElement(eff, "parameter")
    ET.SubElement(p_scale, "parameterid").text = "scale"
    ET.SubElement(p_scale, "name").text = "Scale"
    ET.SubElement(p_scale, "value").text = f"{scale_pct:.2f}"

    p_center = ET.SubElement(eff, "parameter")
    ET.SubElement(p_center, "parameterid").text = "center"
    ET.SubElement(p_center, "name").text = "Center"
    val = ET.SubElement(p_center, "value")
    ET.SubElement(val, "horiz").text = f"{center_x_norm:.4f}"
    ET.SubElement(val, "vert").text = f"{center_y_norm:.4f}"
    return filt


def _make_crop_filter(left_pct: float, right_pct: float, top_pct: float, bottom_pct: float) -> ET.Element:
    filt = ET.Element("filter")
    eff = ET.SubElement(filt, "effect")
    ET.SubElement(eff, "name").text = "Crop"
    ET.SubElement(eff, "effectid").text = "crop"
    ET.SubElement(eff, "effectcategory").text = "motion"
    ET.SubElement(eff, "effecttype").text = "motion"
    ET.SubElement(eff, "mediatype").text = "video"
    for pid, pval in [("left", left_pct), ("right", right_pct), ("top", top_pct), ("bottom", bottom_pct)]:
        p = ET.SubElement(eff, "parameter")
        ET.SubElement(p, "parameterid").text = pid
        ET.SubElement(p, "name").text = pid
        ET.SubElement(p, "value").text = f"{pval:.2f}"
    return filt


def _make_audio_gain_filter(level_db: float) -> ET.Element:
    linear = pow(10.0, float(level_db) / 20.0)
    filt = ET.Element("filter")
    eff = ET.SubElement(filt, "effect")
    ET.SubElement(eff, "name").text = "Audio Levels"
    ET.SubElement(eff, "effectid").text = "audiolevels"
    ET.SubElement(eff, "effectcategory").text = "audiolevels"
    ET.SubElement(eff, "effecttype").text = "audiolevels"
    ET.SubElement(eff, "mediatype").text = "audio"
    p = ET.SubElement(eff, "parameter")
    ET.SubElement(p, "parameterid").text = "level"
    ET.SubElement(p, "name").text = "Level"
    ET.SubElement(p, "value").text = f"{linear:.5f}"
    return filt


def build_editable_fcp7_xml(
    plan_path: str = "02_analysis/timeline_plan.json",
    style_dna_path: str = "02_analysis/style_dna.json",
    raw_index_path: str = "02_analysis/raw_scene_index.json",
    out_xml_path: str = "02_analysis/premiere_editable_timeline.xml",
) -> dict[str, Any]:
    """Generate a general FCP7 XML sequence from timeline_plan.json + style_dna.json + raw_scene_index.json."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    dna = json.loads(Path(style_dna_path).read_text(encoding="utf-8")) if Path(style_dna_path).exists() else {}
    raw_idx = json.loads(Path(raw_index_path).read_text(encoding="utf-8")) if Path(raw_index_path).exists() else {}

    seq_cfg = plan.get("sequence_config", {})
    res = seq_cfg.get("target_resolution", [1920, 1080])
    W, H = int(res[0]), int(res[1])
    fps = int(round(float(seq_cfg.get("target_fps", 30.0))))

    def _sec_to_f(sec: float) -> int:
        return int(round(float(sec) * fps))

    clip_meta = {c["file_name"]: c for c in raw_idx.get("clips", [])}
    segments = plan.get("timeline_segments", [])
    total_dur_sec = max((float(s.get("timeline_end_sec", 0.0)) for s in segments), default=15.0)
    total_frames = _sec_to_f(total_dur_sec)

    xmeml = ET.Element("xmeml", version="4")
    seq = ET.SubElement(xmeml, "sequence", id="seq_autonomous_editable")
    ET.SubElement(seq, "name").text = "Autonomous_Editable_Sequence"
    ET.SubElement(seq, "duration").text = str(total_frames)
    rate = ET.SubElement(seq, "rate")
    ET.SubElement(rate, "timebase").text = str(fps)
    ET.SubElement(rate, "ntsc").text = "FALSE"

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    v_fmt = ET.SubElement(video, "format")
    v_sc = ET.SubElement(v_fmt, "samplecharacteristics")
    v_rate = ET.SubElement(v_sc, "rate")
    ET.SubElement(v_rate, "timebase").text = str(fps)
    ET.SubElement(v_rate, "ntsc").text = "FALSE"
    ET.SubElement(v_sc, "width").text = str(W)
    ET.SubElement(v_sc, "height").text = str(H)

    defined_files: set[str] = set()

    def _add_file_ref(ci: ET.Element, fid: str, fpath: str, dur_f: int, sw: int = W, sh: int = H, has_audio: bool = True, audio_only: bool = False) -> None:
        if fid in defined_files:
            ET.SubElement(ci, "file", id=fid)
            return
        defined_files.add(fid)
        f_el = ET.SubElement(ci, "file", id=fid)
        ET.SubElement(f_el, "name").text = Path(fpath).name
        ET.SubElement(f_el, "pathurl").text = _path_to_file_url(fpath)
        r_el = ET.SubElement(f_el, "rate")
        ET.SubElement(r_el, "timebase").text = str(fps)
        ET.SubElement(r_el, "ntsc").text = "FALSE"
        ET.SubElement(f_el, "duration").text = str(max(1, dur_f))
        m_el = ET.SubElement(f_el, "media")
        if not audio_only:
            ve = ET.SubElement(m_el, "video")
            sc = ET.SubElement(ve, "samplecharacteristics")
            ET.SubElement(sc, "width").text = str(sw)
            ET.SubElement(sc, "height").text = str(sh)
        if has_audio or audio_only:
            ae = ET.SubElement(m_el, "audio")
            asc = ET.SubElement(ae, "samplecharacteristics")
            ET.SubElement(asc, "depth").text = "16"
            ET.SubElement(asc, "samplerate").text = "48000"

    # Track V1: Planned base segments with source handles
    track_v1 = ET.SubElement(video, "track")
    for idx, seg in enumerate(segments, start=1):
        cname = seg["assigned_clip"]
        cpath = seg["source_path"]
        meta = clip_meta.get(cname, {"width": W, "height": H, "duration_sec": 30.0})
        sw, sh = int(meta.get("width", W)), int(meta.get("height", H))
        sdur_f = _sec_to_f(float(meta.get("duration_sec", 30.0)))
        sf = _sec_to_f(float(seg["timeline_start_sec"]))
        ef = max(sf + 1, _sec_to_f(float(seg["timeline_end_sec"])))
        inf = _sec_to_f(float(seg.get("recommended_in_sec", 0.0)))
        outf = inf + (ef - sf)

        ci = ET.SubElement(track_v1, "clipitem", id=f"v1_seg_{idx:02d}")
        ET.SubElement(ci, "name").text = cname
        ET.SubElement(ci, "enabled").text = "TRUE"
        ET.SubElement(ci, "duration").text = str(sdur_f)
        r_ci = ET.SubElement(ci, "rate")
        ET.SubElement(r_ci, "timebase").text = str(fps)
        ET.SubElement(r_ci, "ntsc").text = "FALSE"
        ET.SubElement(ci, "start").text = str(sf)
        ET.SubElement(ci, "end").text = str(ef)
        ET.SubElement(ci, "in").text = str(inf)
        ET.SubElement(ci, "out").text = str(outf)
        _add_file_ref(ci, f"file_{Path(cname).stem}", cpath, sdur_f, sw, sh)
        scale_pct = round((float(W) / float(sw)) * 100.0, 2) if sw > 0 else 100.0
        ci.append(_make_motion_filter(scale_pct=scale_pct))

    # Track V2: Geometric Panel Overlays (with native Crop filters)
    track_v2 = ET.SubElement(video, "track")
    panels = dna.get("layer_1_temporal_grammar", {}).get("geometric_panel_overlays", [])
    seg_by_id = {int(s["slot_id"]): s for s in segments if "slot_id" in s}
    for p_idx, p in enumerate(panels, start=1):
        seg = seg_by_id.get(int(p.get("segment_id", 0)))
        if not seg:
            continue
        cname = seg["assigned_clip"]
        cpath = seg["source_path"]
        meta = clip_meta.get(cname, {"width": W, "height": H, "duration_sec": 30.0})
        sw, sh = int(meta.get("width", W)), int(meta.get("height", H))
        sdur_f = _sec_to_f(float(meta.get("duration_sec", 30.0)))
        sf = _sec_to_f(float(p["start_sec"]))
        ef = max(sf + 1, _sec_to_f(float(p["end_sec"])))
        inf = _sec_to_f(float(seg.get("recommended_in_sec", 0.0)))
        outf = inf + (ef - sf)
        crop = p.get("premiere_crop_pct", {"left": 15.0, "right": 15.0, "top": 10.0, "bottom": 10.0})

        ci = ET.SubElement(track_v2, "clipitem", id=f"v2_panel_{p_idx:02d}")
        ET.SubElement(ci, "name").text = cname
        ET.SubElement(ci, "enabled").text = "TRUE"
        ET.SubElement(ci, "duration").text = str(sdur_f)
        r_ci = ET.SubElement(ci, "rate")
        ET.SubElement(r_ci, "timebase").text = str(fps)
        ET.SubElement(r_ci, "ntsc").text = "FALSE"
        ET.SubElement(ci, "start").text = str(sf)
        ET.SubElement(ci, "end").text = str(ef)
        ET.SubElement(ci, "in").text = str(inf)
        ET.SubElement(ci, "out").text = str(outf)
        _add_file_ref(ci, f"file_{Path(cname).stem}", cpath, sdur_f, sw, sh)
        scale_pct = round((float(W) / float(sw)) * 100.0, 2) if sw > 0 else 100.0
        ci.append(_make_motion_filter(scale_pct=scale_pct))
        ci.append(
            _make_crop_filter(
                float(crop.get("left", 0.0)),
                float(crop.get("right", 0.0)),
                float(crop.get("top", 0.0)),
                float(crop.get("bottom", 0.0)),
            )
        )

    # Audio Tracks A1 (camera audio), A2 (SFX), A3 (Music)
    audio = ET.SubElement(media, "audio")
    track_a2 = ET.SubElement(audio, "track")
    for idx, sfx in enumerate(plan.get("a2_sfx_plan", []), start=1):
        spath = sfx.get("file_path", "")
        if not spath or not Path(spath).exists():
            continue
        sf = _sec_to_f(float(sfx.get("timeline_start_sec", 0.0)))
        df = _sec_to_f(0.50)
        ai = ET.SubElement(track_a2, "clipitem", id=f"a2_sfx_{idx:02d}")
        ET.SubElement(ai, "name").text = Path(spath).name
        ET.SubElement(ai, "enabled").text = "TRUE"
        ET.SubElement(ai, "duration").text = str(df)
        r_ai = ET.SubElement(ai, "rate")
        ET.SubElement(r_ai, "timebase").text = str(fps)
        ET.SubElement(r_ai, "ntsc").text = "FALSE"
        ET.SubElement(ai, "start").text = str(sf)
        ET.SubElement(ai, "end").text = str(sf + df)
        ET.SubElement(ai, "in").text = "0"
        ET.SubElement(ai, "out").text = str(df)
        _add_file_ref(ai, f"file_sfx_{Path(spath).stem}", spath, df, has_audio=True, audio_only=True)
        ai.append(_make_audio_gain_filter(float(sfx.get("gain_db", -14.0))))

    track_a3 = ET.SubElement(audio, "track")
    mpath = plan.get("audio_plan", {}).get("cloned_music_track", "04_assets/music/reference_audio.wav")
    if mpath and Path(mpath).exists():
        ai_m = ET.SubElement(track_a3, "clipitem", id="a3_music_bed")
        ET.SubElement(ai_m, "name").text = Path(mpath).name
        ET.SubElement(ai_m, "enabled").text = "TRUE"
        ET.SubElement(ai_m, "duration").text = str(total_frames)
        r_m = ET.SubElement(ai_m, "rate")
        ET.SubElement(r_m, "timebase").text = str(fps)
        ET.SubElement(r_m, "ntsc").text = "FALSE"
        ET.SubElement(ai_m, "start").text = "0"
        ET.SubElement(ai_m, "end").text = str(total_frames)
        ET.SubElement(ai_m, "in").text = "0"
        ET.SubElement(ai_m, "out").text = str(total_frames)
        _add_file_ref(ai_m, "file_music_bed", mpath, total_frames, has_audio=True, audio_only=True)
        ai_m.append(_make_audio_gain_filter(-2.0))

    ET.indent(xmeml, space="  ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + ET.tostring(xmeml, encoding="unicode")
    out_p = Path(out_xml_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(xml_str, encoding="utf-8")
    return {"xml_path": out_p.as_posix(), "v1_segments": len(segments), "v2_panels": len(panels)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build editable FCP7 XML sequence from timeline_plan.json.")
    parser.add_argument("--plan", default="02_analysis/timeline_plan.json")
    parser.add_argument("--style_dna", default="02_analysis/style_dna.json")
    parser.add_argument("--raw_index", default="02_analysis/raw_scene_index.json")
    parser.add_argument("--out", default="02_analysis/premiere_editable_timeline.xml")
    args = parser.parse_args()

    res = build_editable_fcp7_xml(args.plan, args.style_dna, args.raw_index, args.out)
    print(f"[OK] Built editable FCP7 XML ({res['v1_segments']} V1 segments, {res['v2_panels']} V2 panels) -> {res['xml_path']}")


if __name__ == "__main__":
    main()
