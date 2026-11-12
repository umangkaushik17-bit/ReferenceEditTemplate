---
name: color-grade-lumetri
description: Applies calibrated Two-Pass Lumetri Color grading and .cube LUT application on Premiere Pro Track V4 Adjustment Layers via apply_lumetri_grade.
---

# Programmatic Lumetri Color Grading Skill

## Goal
Deliver consistent, broadcast-safe aesthetic color grading matching the reference edit without manual wheel adjustments.

## Two-Pass Grading Pipeline

### Pass 1: Per-Clip Exposure & White Balance Normalization (V1 / V2)
- Anchor blacks (`Blacks: -5 to -12`).
- Protect clipping highlights (`Highlights: -15 to -30`).
- Ensure skin tones fall along the standard vectorscope skin-tone line.

### Pass 2: Creative Look Profile on V4 Adjustment Layer
- Apply calibrated parameter presets (`resources/aesthetic_looks.json`) or the extracted `style_dna.json` color profile across an Adjustment Layer spanning Track `V4`.

