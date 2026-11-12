---
name: reverse-engineering-style-dna
description: Deconstructs a reference sample edit into a 4-layer parametric and semantic style blueprint (temporal curves, acoustic stems, color grade, and narrative slots) saved to 02_analysis/style_dna.json, and directly extracts/clones the reference reel's music bed to 04_assets/music/ for Track A3.
---

# Reverse-Engineering Reference Style DNA (v2.0)

Deconstructs any reference video in `00_reference_style/` into `02_analysis/style_dna.json` and extracts the cloned music stem into `04_assets/music/reference_audio.wav` (auto-indexed in `04_assets/vault_index.json`).

## Usage
```powershell
python .agents/skills/reverse-engineering-style-dna/scripts/extract_style_dna.py --ref "00_reference_style/<ref>.mp4" --out "02_analysis/style_dna.json" --assets_dir "04_assets"
```
*(Omit `--ref` to auto-discover the video inside `00_reference_style/`.)*

## Extracted Blueprint Layers (`02_analysis/style_dna.json`)
1. **Spatial Geometry (`spatial_geometry`):**
   - Separates `container_resolution` (e.g. `720x1280`) from `active_picture_bounds_xywh` (e.g. `720x406` = `16:9` inside letterbox bars) and emits `recommended_landscape_adaptation` (`strip_static_letterbox`, `crop_no_stretch`, native project FPS matching).
2. **Layer 1 — Temporal & Visual Grammar (`layer_1_temporal_grammar`):**
   - **Active-Region Micro-Cuts:** Detects cuts down to `0.13s` (`~4 frames`) inside outer edge bands even inside narrow expanding slits; sets `cadence_profile` (`rapid_beat_montage` vs `lyrical_cuts`) and `allow_clip_reuse_for_high_cut_density`.
   - **Multi-Strip Opening (`multi_strip_opening`):** Detects strip count, staggered `enter_sec` timings, and exact `divider_line_px` (`0` = `borderless` vs white stroke).
   - **Dynamic Aperture (`aperture_animation`):** Tracks stepped beat-close plateaus (`stepped_close_levels`), blackout holds, `slit_unlock` timestamps synchronized against audio resurgence onsets, and enforces `scaling_policy: crop_no_stretch`.
   - **Mask & Outro Transitions (`mask_transitions`, `outro_transition`):** Distinguishes white human silhouette over pure black (`mask_on_black`) vs live background (`mask_over_live_bg`), plus outro `fade_to_black`.
3. **Layer 2 — Acoustic Architecture (`layer_2_acoustic_architecture`):**
   - BPM, beat timestamps, 50ms RMS silence dips, strong onset peaks, and `energy_resurgence_cues` aligned to visual slit/cut drops.
4. **Layer 3 — Aesthetic Color & Typography (`layer_3_aesthetic_typography_and_color`):**
   - **Real Reference `lumetri_matrix`:** Measured exclusively inside `active_picture_bounds_xywh` via `virtual_scopes.py` (never hardcoded, never polluted by black letterboxes).
   - **Visual Coherence Target:** Reference greenery/sky/warm environment distribution used to exclude off-theme clips downstream.
   - **Dynamic OCR Typography (`typography`):** Extracts title strings via Tesseract OCR, vertical position zone, casing, and `reveal_effect` (`scramble_decode` vs `step_cut_no_fade` vs `fade_in`).
5. **Layer 4 — Narrative Slots (`layer_4_narrative_slots`):**
   - Frame-indexed edit segments tagged with `in_expanding_slit` and `mask_transitions` flags ready for `select_optimal_subclips.py`.
