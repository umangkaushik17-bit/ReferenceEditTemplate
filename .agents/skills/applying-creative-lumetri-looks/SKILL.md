---
name: applying-creative-lumetri-looks
description: Applies stylized aesthetic color grades, clones reference video color palettes from style_dna.json, configures Track V4 adjustment layers, and performs closed-loop visual QA in Premiere Pro MCP. Trigger this skill when grading for mood, cinematic looks, or matching a reference edit's color style.
---

# Creative Lumetri Looks & Closed-Loop Color QA Skill

## When to Use This Skill
- After primary camera normalization (`balancing-primary-color-and-skins`) is complete.
- When applying a stylized aesthetic (Teal & Orange, Moody Matte, High-Retention Viral Pop, Clean Studio, Cyberpunk, Warm Golden Hour).
- When transferring the extracted color DNA (`lumetri_matrix` in `02_analysis/style_dna.json`) from a reference edit onto the timeline.

## Step-by-Step Workflow

### Step 1: Select Creative Grade Source
1. **If cloning a Reference Video**: Read `lumetri_matrix` from `02_analysis/style_dna.json`.
2. **If prompted by Aesthetic Name**: Read `.agents/skills/applying-creative-lumetri-looks/references/lumetri-look-matrices.md` and select the matching profile.

### Step 2: Apply Pass 3 Creative Grade Without Destroying Pass 1
1. Prefer placing an Adjustment Layer across **Track V4** (below Kinetic Captions if captions are moved to V5, or above V1–V3 video tracks) so a single creative grade unifies A-roll and B-roll.
2. If applying directly to clips, add a **second** `Lumetri Color` effect instance (or combine the creative delta additively with the clip's Pass 1 `recommended_primary_lumetri` values while clamping `exposure` within `±1.5` and `saturation` within `80–125`).
3. If an existing Creative Look or LUT is loaded on the Lumetri instance, set `look_intensity` via `set_lumetri` to `60` (never leave a creative LUT at `100%`).

### Step 3: Mandatory Closed-Loop Scope & Visual QA
1. Move the playhead to three checkpoints (A-roll face close-up, B-roll cutaway, and high-contrast highlight shot) and call `export_frame` into `02_analysis/qa_frames/`.
2. Run `virtual_scopes.py --frames_dir 02_analysis/qa_frames --output 02_analysis/qa_scopes.json`.
3. Verify that:
   - `p99_white` is `<= 95.0 IRE` and `p01_black` is `>= 5.0 IRE`.
   - `skin_luma_ire` sits between `55.0` and `72.0 IRE`.
4. Visually inspect the exported PNGs using multimodal vision to confirm natural skin tones and roll-off. If clipping or color banding appears, dial back `contrast` and `saturation` by `20%` and re-verify.

