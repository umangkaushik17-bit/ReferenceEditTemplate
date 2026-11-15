---
name: motion-graphics-remotion
description: Generates React/Python kinetic typography, word-level active highlights, and animated UI overlays rendered to ProRes 4444 on Track V3 or native Premiere MOGRTs on Track V2, plus synthesizes broadcast SFX in 04_assets/sfx/.
---

# Kinetic Typography, Karaoke Captions & SFX Vault Skill (v2.1)

Renders transparent alpha ProRes 4444 motion overlays (`.mov`, `yuva444p10le`) driven directly by `layer_3_aesthetic_typography_and_color.typography` in `02_analysis/style_dna.json` or word-level speech transcripts in `02_analysis/raw_transcript_words.json`.

## Scripts
1. **Render Kinetic Titles or Word-Level Karaoke Captions (`render_kinetic_typography.py`):**
   ```powershell
   # Render all reference-matched typography events from style_dna.json + timeline_plan.json:
   python .agents/skills/04-motion-graphics-remotion/scripts/render_kinetic_typography.py --style_dna 02_analysis/style_dna.json --timeline_plan 02_analysis/timeline_plan.json

   # Render word-by-word active highlight karaoke captions from raw_transcript_words.json:
   python .agents/skills/04-motion-graphics-remotion/scripts/render_kinetic_typography.py --transcript 02_analysis/raw_transcript_words.json --style_profile viral_karaoke_highlight

   # Render a single custom title overlay:
   python .agents/skills/04-motion-graphics-remotion/scripts/render_kinetic_typography.py --text "VALLEY OF FLOWERS" --effect scramble_decode --duration 2.0 --fps 60
   ```
2. **Generate Studio SFX Vault (`generate_sfx_vault.py`):**
   ```powershell
   python .agents/skills/04-motion-graphics-remotion/scripts/generate_sfx_vault.py
   ```
   Synthesizes 48kHz 16-bit stereo WAV sound effects (`whoosh_fast_01.wav`, `sub_drop_heavy_01.wav`, `pop_click_ui_01.wav`, `riser_swell_01.wav`, `shutter_snap_01.wav`) into `04_assets/sfx/` and updates `04_assets/vault_index.json` with calibrated `peak_transient_ms`.

## Core Workflow Rules
1. **Dynamic Event Matching:** Read `typography.events[]` from `style_dna.json` to map exact phrase count, casing (`is_all_caps`), vertical zone (`position_zone` / `y_center_pct`), onset/end timing, and animation mode (`reveal_effect`).
2. **Reveal Modes:**
   - `scramble_decode`: Decode characters rapidly frame-by-frame during the initial accent lock before settling cleanly.
   - `step_cut_no_fade`: Crisp `0% -> 100%` 1-frame snap appearance & disappearance (`0px` opacity fade tails).
   - `fade_in`: Smooth cubic-eased opacity ramp with subtle vertical rise.
   - `viral_karaoke_highlight`: Word-by-word active `#FFE600` highlight and `1.12x` scale pop synced to spoken word timestamps.
3. **Zero Temp-PNG Bloat:** Streams raw RGBA frames directly into FFmpeg (`prores_ks -profile:v 4 -pix_fmt yuva444p10le`) via `stdin` pipe.
