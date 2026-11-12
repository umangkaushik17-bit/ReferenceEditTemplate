# Calibrated Lumetri Color Creative Matrices (Pass 3 Deltas)

Apply these values as **Pass 3 creative deltas** on top of a normalized Pass 1 image (where neutral midtones sit at ~48 IRE and skin sits at ~62 IRE).

| Look ID | Exposure | Contrast | Highlights | Shadows | Whites | Blacks | Saturation | Temp | Tint | Look Intensity | Target Aesthetic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean_tech_studio` | `+0.15` | `+14` | `-12` | `+6` | `+8` | `-8` | `106` | `-3` | `0` | `60` | Crisp talking-head, product demos, Apple/MKBHD studio feel |
| `cinematic_teal_orange` | `0.00` | `+24` | `-20` | `-10` | `+5` | `-14` | `112` | `-10` | `-3` | `65` | Documentary, travel B-roll, high-drama narrative |
| `viral_high_retention` | `+0.25` | `+28` | `-15` | `+12` | `+14` | `-12` | `120` | `+2` | `0` | `70` | Fast-paced Reels/Shorts/TikTok hooks, MrBeast/Hormozi pop |
| `moody_matte_film` | `-0.15` | `-8` | `-28` | `+22` | `-16` | `+16` | `86` | `+7` | `+4` | `55` | Nostalgic Kodak 2383 emulation, lifted matte blacks, soft highlights |
| `warm_golden_hour` | `+0.10` | `+16` | `-18` | `+10` | `+6` | `-6` | `110` | `+14` | `+3` | `60` | Sunset travel, lifestyle, warm motivational monologue |
| `cyber_neon_night` | `-0.10` | `+26` | `-10` | `-16` | `+12` | `-18` | `116` | `-16` | `+12` | `70` | Night b-roll, coding/gaming setups, magenta-cyan split tone |

## Skin-Protection Dampening Formula
When `skin_detected == true` in `02_analysis/color_telemetry.json` and `skin_coverage_pct > 8.0%`:
- Clamp `Temp` delta from the table above to `[-8, +8]` so faces do not turn blue or orange.
- Clamp `Saturation` delta to a maximum of `114`.

