# Post-Mortem & Zero-Retake Architectural Plan

Complete post-mortem of **why this edit took multiple retakes and cost ~$15+**, followed by the **architectural plan** to make future edits **one-shot (zero-retake) at ~80% lower token cost and 5x faster runtime**.

---

# Part 1: Root-Cause Diagnosis — Why Did It Take 4 Retakes & $15+?

### 1. `extract_style_dna.py` Missed 90% of the Reference's Visual Grammar on Pass 1
- **The Problem:**
  - **Typography & Blend Modes:** `extract_style_dna.py` relied on thresholding bright white pixels (`V >= 205, S <= 50`) for Tesseract OCR. Because the reference video used **`Difference` / `Invert` blend mode (`255 - bg`)** with regular `Arial` over textured rocks/trees, the automated extractor found only **1 garbled text event** out of **46 real events**.
  - **False Silhouette Masks:** The extractor misclassified the 4 giant white `"BUT"` text flashes (`0.20s`, `1.37s`, `2.53s`, `3.77s`) as human silhouette masks (`mask_over_live_bg`).
  - **Unmeasured Panel/Overlay Geometry:** The extractor had no detector for **picture-in-picture inset boxes (`7.40s`)**, **symmetrical 2-panel splits (`10.17s`)**, **3-pill stadium bars (`6.14s`, `10.95s`)**, **circle iris reveals (`11.40s`)**, or **centered sticker cards (`12.85s`)**.

### 2. Guessing Coordinates & Styles Instead of Computing Exact Frame-Diffs First
Because `style_dna.json` didn't output exact pixel boxes for those transitions, I initially approximated them by eye from small thumbnails, which directly caused every retake you caught:
- **Retake 1 (`6.14s` strips & `12.85s` cut-off sticker):**
  - At `6.14s`, I layered the 3 horizontal bands over the *previous* clip instead of a blurred/darkened plate of the *same* clip, leaving strips of the blue jersey visible in the gaps.
  - At `12.85s`, I guessed the Angry King card was `x: 34%..68%, y: 11%..89%` instead of measuring the white card's true pixel boundary (`x: 21.24%..78.68%, y: 8.08%..91.23%`).
- **Retake 2 (Typography font/weight & `5.80s` headless cutout):**
  - I initially used `Impact` / `Arial Bold` with drop shadows and cartoon outlines instead of **Regular `Arial` (`arial.ttf`) with true `Difference` (`255 - bg`) video-invert blending**.
  - At `5.80s`, `isolate_subject_alpha.py` only checked `person_coverage_pct >= 2.0%`. At `0.20s` of `20260714_111554.mp4`, your torso filled `53%` of the frame, so the check passed—even though your head was clipped off by the top border (`y = 0`) of the camera frame and `mobilenet` missed your dark bucket hat.
- **Retake 3 & 4 (`10.17s` asymmetrical panels & `10.44s` position jump):**
  - At `10.17s`, I initially shifted your position inside the panel (`10.17–10.50s`) without applying the same 4K crop to the full-screen reveal (`10.50–10.87s`), causing you to teleport horizontally from `x = 71.5%` back to `x = 50%` and resetting time by `0.33s`.

### 3. Why the Token Cost Hit ~$15+ (Context Bloat + Iterative Vision Loops + Premiere 26.5 Bugs)
1. **379 MCP Tools Loaded in Context on Every Turn:** Every tool call (`shell`, `read`, `edit`, `execute`) re-sends the entire conversation history + 379 Premiere Pro tool definitions. Turning a 5-turn workflow into a 30-turn debugging loop multiplies input token billing by **6x–8x**.
2. **12+ High-Resolution Contact Sheets Sent to Multimodal Vision (`read`):** Reading large multi-megapixel JPEG grids on every iteration adds heavy vision token charges and latency.
3. **Wasted MCP Turns on Known Premiere Pro 26.5.3 Limitations:** We spent multiple turns hitting Premiere 26.5.3 bugs (`set_item_in_out` failing to set out-points, `trim_clip` no-oping on 26.x, `color_correct` failing due to an empty legacy QE effect catalog, and `export_frame` hanging the CEP bridge with `busy_*.json` lockfiles).

---

# Part 2: The Solution Plan (Zero-Retake, Low-Cost Architecture)

To make your next edit **1-shot accurate in <3 minutes and <$2**, here is the 5-point upgrade plan to implement in the codebase:

### Pillar 1: Upgrade `extract_style_dna.py` with Exact Geometric & Blend-Mode Deconstruction
1. **Frame-Pair Geometric Mask & Panel Extractor (`_detect_geometric_masks_and_panels`):**
   - Automatically compare frames before/after every transition (`abs(frame[t] - frame[t_ref])`) to mathematically extract exact normalized bounding boxes `[x0, y0, x1, y1]`, corner radii (sharp rectangle vs. stadium pill vs. circle iris), and symmetry flags for all panels, splits, and sticker cards—**zero human guessing**.
   - Enforce a **Symmetry & Continuity Invariant**: if a multi-panel overlay precedes a full-screen reveal of the same shot (`Shot B`), automatically flag `continuous_1to1_screen_space: true` so the panel phase (`frame 0..k`) and full-screen phase (`frame k+1..N`) are guaranteed to use the **exact same `(x, y)` crop and continuous frame counter**.
2. **Difference / Knockout Typography Detector (`_detect_difference_and_solid_typography`):**
   - Detect text that uses `Difference` / `Exclusion` blend modes by comparing center ROI pixels against `255 - neighbor_bg` across consecutive frames, and measure exact cap-height in `% of frame height` + stroke-width ratio (automatically distinguishing Regular `Arial` `Difference` text vs. Heavy `Arial Black` solid text).

### Pillar 2: Add "Full-Head & Uncropped-Subject" Guardrail to `isolate_subject_alpha.py` & `select_optimal_subclips.py`
1. **Top-Border Head-Clip Rejection (`head_safe_margin_pct`):**
   - In `select_optimal_subclips.py` and `isolate_subject_alpha.py`, reject any subclip window where the segmented person mask touches the top `2%` border of the frame (`y_min <= 0.02 * H`, which means the camera cut off the top of the person's head).
2. **Face / Head Keypoint & Hat Completion Check:**
   - Run a lightweight MediaPipe face/pose head-box check (or ResNet50 + head-circle completion) during `isolate_subject_alpha.py` so dark hats/hair above the torso are always preserved and headless frames are never selected.

### Pillar 3: Automated Pre-Render Mathematical QA Script (`02_analysis/verify_edit_against_reference.py`)
Instead of rendering 8 versions and reading 12 contact sheets with LLM vision tokens ($10+ in token cost), add a **single automated Python QA script** that runs **before** importing into Premiere Pro:
- Compares every frame `t` of the rendered edit against frame `t` of the reference video across 6 automated checks:
  1. **Cut & Flash Timing Alignment:** Verifies every cut and white/black flash matches the reference within `±1 frame`.
  2. **Panel & Sticker Bounding-Box Match:** Verifies any panel/split/sticker bounding box matches the reference's `[x0, y0, x1, y1]` within `±1.5%` and checks left/right & top/bottom symmetry.
  3. **Motion & Position Continuity Across Panel-to-Fullscreen Reveals:** Computes optical-flow / centroid shift of the subject between the last panel frame (`t_open - 1`) and first full-screen frame (`t_open + 1`), failing loudly if the subject jumps by `> 3%` of screen width or if source time resets!
  4. **Cutout Head-Intact Check:** Verifies the foreground cutout's top-most alpha pixel sits cleanly inside `[5%..35%]` of its bounding height (head present, not clipped at `y = 0`).
  5. **Typography Contrast & Size Check:** Verifies every lyric event's cap-height (`% of H`), blend mode (`Difference` vs `Solid`), and readability contrast against the underlying video.
- Emits a single compact JSON pass/fail report + **1 single combined QA contact sheet** so only **1 vision read** is ever needed.

### Pillar 4: Premiere Pro 26.5.3 Fast-Path Assembly Script (1 Turn Instead of 10 Turns)
To avoid wasting MCP turns on Premiere 26.5.3's broken legacy QE calls:
- **Never call** `set_item_in_out`, `trim_clip`, `color_correct`, or `export_frame` over the Premiere 26.5.3 CEP bridge (and automatically clean any stale `%TEMP%\premiere-mcp-bridge\busy_*.json` lockfiles before connecting).
- Pre-trim and naturally color-grade the `V1` subclips (`slot_01..N.mp4`), `V2` geometric composites (`v2_geometric_composites.mov`), and `V3` kinetic typography (`v3_kinetic_graphics.mov`) in **one single Python pipeline run**, and assemble the entire Premiere Pro timeline (`import_media` + `overwrite_clip` on `A2`/`A3` first, then `V1`/`V2`/`V3` + `save_project`) in **one single `execute` call** (1 turn total!).

### Pillar 5: Isolated Subagent Reference Post-Mortem, Translucent Glass Typography & Zero-Hang Windows Runtime (v3.1)
1. **Mandatory Stage 1 Reference Post-Mortem via Isolated `subagent`:**
   - `extract_style_dna.py` now runs 1-frame strobe/blink detection (`_detect_strobe_and_blink_schedules`), per-frame intra-shot collage detection (`_detect_intra_shot_collage_and_splits`), and emits `02_analysis/ref_postmortem_sheet.jpg`.
   - A dedicated child `subagent` reads `02_analysis/ref_postmortem_sheet.jpg` in an isolated context and writes the verified slot-by-slot Post-Mortem into `02_analysis/style_dna.json` before Stage 2 begins—keeping multi-megapixel images out of the main session context.
2. **Translucent Glass Typography & Clean Full-White Flash Cutouts (Zero Negative Faces):**
   - Non-red lyric overlays (`ME`, `TO`, `THE`, `MOON`, `Where ... We`, `MY`, `AWAKE`, `I`, `FEEL`, `SO`) use a **translucent glass/screen/difference blend (`72%–78%` opacity)** so background video texture shines through the letters.
   - `isolate_subject_alpha.py` enforces largest-central-component filtering, `enforce_depth_sandwich_width_ratio` (`30%–52%` overlap so outer letters flank the subject clearly), and `render_clean_white_flash_cutout` (`#FFFFFF` flash, banning raw `255 - RGB` negative inversion on human faces).
3. **8-Gate Pixel-Level Pre-Import QA (`verify_edit_against_reference.py`):**
   - Verifies upper-border black-triangle absence (`6_upper_border_black_triangle_check`), translucent glass & no-negative-face policy (`7_translucent_glass_and_no_negative_face_check`), and 1f/2f strobe/blink cadence (`8_strobe_and_blink_cadence_check`).
4. **4 Zero-Hang Windows Runtime Rules:**
   - Stream 1 frame at a time (`cap.read()` -> ROI blend -> `vw.write()`), free PyTorch immediately after computing `384x216` masks (`del deeplab; gc.collect()`), encode intermediate streams via `MJPG` `.avi` + `ffmpeg -nostdin -preset ultrafast` (`stdin=subprocess.DEVNULL`), and terminate PyTorch scripts on Windows with `os._exit(0)`.
