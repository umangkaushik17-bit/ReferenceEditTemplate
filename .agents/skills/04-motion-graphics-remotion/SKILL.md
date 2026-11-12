---
name: motion-graphics-remotion
description: Generates React-based kinetic typography, word-level active highlights, and animated UI overlays rendered to ProRes 4444 on Track V3 or native Premiere MOGRTs on Track V2.
---

# Kinetic Typography & Motion Graphics Compositing (v2.0)

Renders transparent alpha ProRes 4444 motion overlays (`.mov`, `yuva444p10le`) or `.mogrt` graphics driven directly by `layer_3_aesthetic_typography_and_color.typography` in `02_analysis/style_dna.json`.

## Core Workflow Rules
1. **Dynamic Event Matching:** Never hardcode stale caption phrases or assume every reel fades in/out. Read `typography.events[]` from `style_dna.json` to map exact phrase count, casing (`is_all_caps`), vertical zone (`position_zone` / `y_center_pct`), onset/end timing, and animation mode (`reveal_effect`).
2. **Reveal Modes:**
   - `scramble_decode`: Decode characters rapidly frame-by-frame during the initial accent lock before settling cleanly.
   - `step_cut_no_fade`: Crisp `0% -> 100%` 1-frame snap appearance & disappearance (`0px` opacity fade tails).
   - `fade_in`: Smooth opacity ramp calibrated to measured `fade_frames`.
3. **Transparent ProRes 4444 Output & Track Placement:** Render standalone overlays onto Track `V3` above footage, or place typography on Track `V2` beneath a DeepLabV3-isolated Track `V3` foreground subject cutout (`03-bg-removal-compositing`) for 3-layer depth sandwiching. Follow `resources/caption_style_schema.json` styling bounds.
