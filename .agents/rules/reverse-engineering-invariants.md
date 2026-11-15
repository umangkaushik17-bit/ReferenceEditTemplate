---
trigger: always_on
description: Enforces active-picture geometry, native FPS matching, crop-no-stretch apertures, frame-diff panel/sticker symmetry, Difference typography detection, head-safe subject gating, and 5-gate mathematical pre-QA.
---

# Reverse-Engineering & Zero-Retake Perception Invariants (v3.0)

## 1. Active-Picture vs Container Geometry Rule
- Never confuse `spatial_geometry.container_resolution` with `spatial_geometry.active_picture_bounds_xywh`. When a vertical (`9:16`) reference contains letterboxed `16:9` active footage, landscape adaptations must use native `16:9` canvas dimensions (`[3840, 2160]` or `[1920, 1080]`), strip static black letterbox margins, apply `scaling_policy: crop_no_stretch`, and adopt raw camera native `recommended_sequence_fps` with cuts snapped to integer frame grids.
- `virtual_scopes.py` must strip static letterbox/pillarbox/windowbox bars (`extract_active_region`) before calculating `luma_ire` (`p01..p99`) or foliage-safe `rgb_parade_cast`.

## 2. Exact Geometric Panel, Symmetry & Blend-Mode Invariants (Pillar 1)
- **Frame-Pair Geometric Panels & Stickers (`geometric_panel_overlays`):** Extract exact normalized bounding boxes `[x0_pct, y0_pct, x1_pct, y1_pct]` and `premiere_crop_pct` via frame-difference analysis (`_detect_geometric_masks_and_panels`) — never guess sticker or panel coordinates by eye.
- **Symmetry & Continuity Invariant (`continuous_1to1_screen_space`):** Lock left/right and top/bottom panel margins symmetrically when within `3.5%`. When a multi-panel overlay precedes a full-screen reveal of the same shot, enforce `continuous_1to1_screen_space: true` so the panel phase (`frame 0..k`) and full-screen phase (`frame k+1..N`) use the **exact same `(x, y)` crop and continuous frame counter** (zero position teleportation, zero time reset).
- **Multi-Band Self-Consistency:** Multi-band horizontal/vertical splits must layer over a darkened/blurred plate of the **same active shot**, never exposing leftover strips of the previous clip.
- **Difference & Solid Typography (`blend_mode` & `font_weight_class`):** Detect both `Difference`/`Exclusion` inverted text (`255 - bg`, Regular `Arial`) and `Normal` solid impact text (`Arial Black`), and reject multi-letter giant impact words (`"BUT"`) from human silhouette masks.

## 3. Head-Safe Subject Verification Guardrail (Pillar 2)
- Index both `01_raw_media/a_roll/` and `01_raw_media/b_roll/` via `index_raw_media.py` before clip selection.
- **Top-Border Head-Clip Rejection (`top_head_margin_pct >= 2.0%`):** Reject any subject cutout window in `select_optimal_subclips.py` and `isolate_subject_alpha.py` where the segmented person touches the top `2%` border of the frame (`y_min <= 0.02 * H`, indicating a clipped head). Use `deeplabv3_resnet50` with largest-connected-component filtering and circular head/hat completion (`_complete_person_head_and_lcc`).

## 4. Mandatory 5-Gate Mathematical Pre-QA Rule (Pillar 3)
- Before importing into Premiere Pro or reading visual contact sheets with LLM vision, run `.agents/skills/reverse-engineering-style-dna/scripts/verify_edit_against_reference.py` to verify all 5 automated checks (`1_cut_and_flash_timing_alignment`, `2_panel_and_sticker_bbox_symmetry`, `3_panel_to_fullscreen_continuity`, `4_cutout_head_intact_check`, `5_typography_contrast_and_size`) and generate **1 single combined QA contact sheet (`02_analysis/one_shot_qa_sheet.jpg`)**.
