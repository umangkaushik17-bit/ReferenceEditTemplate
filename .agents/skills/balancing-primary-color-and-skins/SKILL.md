---
name: balancing-primary-color-and-skins
description: Normalizes exposure, neutralizes white balance color casts, protects skin-tone IRE levels, and matches B-roll clips to A-roll hero shots using OpenCV virtual scopes and Premiere Pro Lumetri Basic Correction. Trigger this skill before applying any creative LUT or stylized grade.
---

# Primary Color Balancing & Shot-Matching Skill

## When to Use This Skill
- Before applying any creative look or aesthetic grade in Premiere Pro.
- When A-roll talking-head clips and B-roll cutaways have mismatched lighting, exposure, or color temperature.
- When raw footage looks washed out (Log/Flat), underexposed, or has an indoor tungsten/outdoor blue cast.

## Step 1: Run Virtual Scopes Telemetry (Active Picture Only)
1. Export representative frames from each timeline clip (or point directly to clip frame folders in `02_analysis/clip_frames/`) and run:
   ```powershell
   python .agents/skills/balancing-primary-color-and-skins/scripts/virtual_scopes.py --frames_dir 02_analysis/clip_frames --output 02_analysis/color_telemetry.json --match_to <hero_clip_stem>
   ```
2. By default (`auto_crop_active=True`), `extract_active_region` automatically strips static black letterbox/pillarbox/windowbox bars before calculating IRE percentiles so black bars never drag `p01_black` or `p50_mid` down to zero. Pass `--no_crop_active` only when inspecting explicit full-frame matting plates.
3. Read `02_analysis/color_telemetry.json`:
   - `active_region`: verified cropped bounding box `[x, y, w, h]` vs `full_size`.
   - `luma_ire`: `p01_black`, `p25_shadow`, `p50_mid`, `p75_high`, `p99_white`, `dynamic_range`.
   - `rgb_parade_cast`: `r_bias`, `g_bias`, `b_bias` in neutral midtones.
   - `skin_metrics`: `skin_detected`, `skin_luma_ire` (`55-70 IRE`), YCrCb shifts.
   - `environment_metrics`: `mean_saturation_pct`, `green_fraction`, `sky_blue_fraction`, `warm_fraction`.
   - `recommended_primary_lumetri`: mathematically derived Pass 1 values (enforces highlight safety `exposure <= 0` whenever `p99_white > 94.0`).

## Step 2: Apply Pass 1 Primary Normalization (Tracks V1 & V2)
Apply each clip's `recommended_primary_lumetri` baseline via a base `Lumetri Color` effect before any global Track `V4` creative grade.

## Step 3: Pass 2 Shot-to-Shot Matching (`match_delta_to_hero`)
Designate the primary A-roll hero clip with `--match_to <hero_clip>` and apply `match_delta_to_hero` (`exposure_delta`, `temperature_delta`, `tint_delta`, `contrast_delta`, `highlights_delta`, `shadows_delta`, `saturation_delta`) across cutaways and multi-strip panels so all clips share cohesive exposure, temperature, and foliage green saturation before Track `V4` styling.
