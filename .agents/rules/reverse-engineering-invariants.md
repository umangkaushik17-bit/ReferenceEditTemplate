---
trigger: always_on
description: Enforces active-picture geometry, native FPS matching, crop-no-stretch apertures, borderless/ruled split detection, theme coherence, DeepLabV3 subject gating, and rapid beat-cut extraction in the reverse-engineering pipeline.
---

# Reverse-Engineering & Perception Architecture Invariants

## 1. Active-Picture vs Container Geometry Rule
- Never confuse `spatial_geometry.container_resolution` with `spatial_geometry.active_picture_bounds_xywh`. When a vertical (`9:16`) reference contains letterboxed `16:9` active footage, landscape adaptations must use native `16:9` canvas dimensions (`[3840, 2160]` or `[1920, 1080]`), strip static black letterbox margins, apply `scaling_policy: crop_no_stretch`, and adopt raw camera native `recommended_sequence_fps` (e.g., `60fps`) with cuts snapped to integer frame grids.
- `virtual_scopes.py` must strip static letterbox/pillarbox/windowbox bars (`extract_active_region`) before calculating `luma_ire` (`p01..p99`) or `rgb_parade_cast`.

## 2. Visual Grammar & Transition Fidelity Rule
- **Multi-Strip Intros:** Render divider strokes **only** when `multi_strip_opening.divider_line_px > 0`. If `0` (`border_style: borderless`), strips touch seamlessly with zero white lines.
- **Stepped Apertures & Tune Unlock:** Honor discrete `aperture_animation.stepped_close_levels` (~30% / ~60% beat closes -> blackout hold) using vertical cropping/masking (`crop_no_stretch`), never vertical scale squeezing. Synchronize expanding slit unlock directly to `aperture_animation.slit_unlock.slit_open_sec` and `audio_unlock_sync`.
- **Silhouette Mask Modes:** Explicitly match `mask_transitions[].type`: `mask_on_black` (solid `#FFFFFF` subject over `#000000`) vs `mask_over_live_bg` (`#FFFFFF` subject composited directly over live footage).
- **Typography Reveal Modes:** Match `typography.events[].reveal_effect` dynamically (`scramble_decode` vs `step_cut_no_fade` vs `fade_in`) rather than defaulting all titles to fades or old cached strings.

## 3. Visual Coherence & Subject Verification Guard
- Index both `01_raw_media/a_roll/` and `01_raw_media/b_roll/` via `index_raw_media.py` before clip selection.
- When `style_dna.json` indicates outdoor/alpine consistency (`outdoor_scene_consistency_required: true`), exclude clips flagged `likely_indoor: true` in `select_optimal_subclips.py` unless zero outdoor alternatives exist.
- Never run `isolate_subject_alpha.py` on unverified shots or empty scenery. Slots with `require_person: true` (`mask_on_black` / `mask_over_live_bg` / depth-sandwich titles) must select windows with verified DeepLabV3 `person_coverage_pct >= 2.0%`.

## 4. Micro-Cut Sensitivity & Disjoint Reuse Rule
- Scene segmentation must examine the active edge region down to `>= 4 frames` (`~0.13s`) so rapid beat-montage cuts inside expanding slits are never dropped.
- When `segment_count` exceeds unique consistent raw clips (`allow_clip_reuse_for_high_cut_density: true`), allocate **disjoint non-overlapping** `[t_in, t_out]` windows across source clips via `exclude_windows`.

