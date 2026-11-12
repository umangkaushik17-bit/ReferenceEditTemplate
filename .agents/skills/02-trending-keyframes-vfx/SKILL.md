---
name: trending-keyframes-vfx
description: Applies punch-in zooms, smooth Bezier ease curves, whip-pans, and floating graphic bounces in Premiere Pro via set_clip_transform and execute_extendscript.
---

# Trending Keyframes, Aperture Framing & Bezier Curves (v2.0)

Generates keyframed camera zooms, multi-strip split layouts, and dynamic aperture framing anchored to `02_analysis/style_dna.json` and `02_analysis/timeline_plan.json`.

## Hard Architectural Invariants
1. **No Vertical Squeezing (`crop_no_stretch`):** When animating stepped close plateaus or expanding slit apertures (`aperture_animation.crop_keyframes`), always animate top/bottom matte crop boundaries (`Crop` effect or Track Matte keyframes) while holding Uniform Scale constant — **never** animate non-uniform Y-axis scale that squashes or stretches the video image.
2. **Border Fidelity (`divider_line_px`):** Check `multi_strip_opening.divider_line_px` in `style_dna.json`. When `0` (`borderless`), align strip edges pixel-to-pixel with zero white gap strokes.
3. **Frame-Snapped Curves & Audio Sync:** Snap all transform, aperture, and punch-in keyframes to exact sequence frame boundaries (`1 / recommended_sequence_fps`) and align expanding slit unlocks directly with `aperture_animation.slit_unlock.audio_unlock_sync`.
4. **Bezier Ease Profiles:** Apply cubic Bezier keyframe handles via `resources/extendscript_curves.jsx` rather than harsh linear interpolation on slow creeping zooms (`100% -> 108%`) and accent punches (`100% -> 114%..118%`).
