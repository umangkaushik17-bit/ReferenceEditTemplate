---
trigger: always_on
description: Enforces Premiere Pro ExtendScript/CEP DOM invariants, frame tick math, reverse-order cutting, and Premiere 26.5.3 fast-path assembly rules.
---

# Premiere Pro MCP Execution Invariants (v3.0)

1. **Timecode & Tick Conversion Standard:**
   - Premiere Pro ExtendScript measures timeline time strictly in ticks (`254016000000` ticks per second).
   - In all calculations, lock conversions:
     `ticks = Math.round(seconds * 254016000000)`
   - Query sequence frame rate first (`23.976`, `24`, `29.97`, `30`, or `60` fps) and snap all cut timestamps to an exact frame boundary (`frame_duration = 1 / fps`) to prevent sub-frame audio pops or black-frame blips.

2. **Reverse-Chronological Mutation Rule:**
   - Whenever calling `split_clip`, `razor_all_tracks`, or ripple deletion tools on any sequence track, always sort target timestamps in **strictly descending chronological order** (`t_n > t_{n-1} > ... > t_1`).
   - Cutting or deleting from start-to-finish (`00:00:00` forward) immediately shifts the timecodes and clip array indices of all downstream media, causing catastrophic cut desync.

3. **Premiere Pro 26.5.3 Fast-Path Assembly Rule (Pillar 4):**
   - **Never call** legacy QE `set_item_in_out`, `trim_clip`, `color_correct`, or `export_frame` / `export_sequence_review_frames` over the Premiere 26.5.3 CEP bridge (`set_item_in_out` and `trim_clip` no-op on source/timeline edges in 26.x, `color_correct` fails on empty legacy QE effect catalogs, and `export_frame` opens modal dialogs that leave stale `busy_*.json` lockfiles in `%TEMP%\premiere-mcp-bridge`).
   - Instead, pre-trim and naturally color-grade the `V1` subclips (`slot_01..N.mp4`), `V2` geometric composites (`v2_geometric_composites.mov`), and `V3` kinetic typography (`v3_kinetic_graphics.mov`) in **one single Python pipeline run** (and/or generate `02_analysis/premiere_editable_timeline.xml` via `build_editable_premiere_xml.py`), and assemble the entire Premiere Pro timeline (`import_media` + `overwrite_clip` on `V1`/`V2`/`V3`/`A2`/`A3` + `save_project`) in **one single `execute` call**.

4. **Composition & Render Handoffs:**
   - Never import raw `.aep` files directly into Premiere Pro across the bridge. Render After Effects compositions and Remotion/Python animations to ProRes 4444 `.mov` with an unmultiplied alpha channel (`yuva444p10le`) and import the finished asset.
