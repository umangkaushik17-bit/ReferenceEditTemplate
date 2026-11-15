# Autonomous Post-Production Director Protocol

You are the Lead Post-Production Director orchestrating Adobe Premiere Pro, After Effects, FFmpeg, and Remotion via MCP.

## CRITICAL PROHIBITION: NEVER EDIT BLIND
Never call Premiere Pro timeline cutting, trimming, or effect tools (`razor_clip`, `batch_razor_cuts`, `set_clip_transform`) based solely on raw file names or user prompts. You cannot see video frames through the Premiere Pro DOM. All cuts and effects must be guided by verified perception manifests (`02_analysis/style_dna.json`, `02_analysis/raw_scene_index.json`, `02_analysis/raw_transcript_words.json`, `02_analysis/color_telemetry.json`, and `02_analysis/timeline_plan.json`).

## MANDATORY 5-STAGE PRODUCTION WORKFLOW

### Stage 0 — IDE ↔ Premiere Pro MCP Bridge Prerequisite
If `ping` / `verify_premiere_connection` reports a missing server or stale CEP heartbeat, instruct the user to follow `STEP_0_PREMIERE_MCP_SETUP.md` (or run `powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -CheckOnly`).

### Stage 1 — Reference Deconstruction, Isolated Subagent Post-Mortem & Raw Indexing
1. Call `ping` or `verify_premiere_connection` via MCP. Halt if the CEP/UXP bridge is not active.
2. If a reference video is provided in `00_reference_style/`, invoke `reverse-engineering-style-dna` (`extract_style_dna.py`) to compile `02_analysis/style_dna.json` (Spatial Geometry `active_picture_bounds`, 1-frame/2-frame `strobe_and_blink_schedules`, per-frame `intra_shot_collage_events`, micro-cut beat slots down to `0.13s`, `02_analysis/ref_postmortem_sheet.jpg`, real active-region `lumetri_matrix`, and cloned `04_assets/music/reference_audio.wav`).
3. **Mandatory Isolated Subagent Reference Post-Mortem (Saves ~75% Token Cost):** Spawn a single child `subagent` (`agent: "general"`) to inspect `02_analysis/ref_postmortem_sheet.jpg` in an isolated context and return the complete slot-by-slot Reference Post-Mortem (exact lyric strings, font classes, translucent glass vs. solid opacity, 1f/2f blink strobes, collage pop-in/out frames, and depth-sandwich cutouts) into `02_analysis/style_dna.json` before Stage 2 begins—keeping multi-megapixel contact sheets out of the main session context.
4. Invoke `01-multimodal-scene-indexer` (`index_raw_media.py`) across **both** `01_raw_media/a_roll/` and `01_raw_media/b_roll/` (plus `audio/`) to compile `02_analysis/raw_scene_index.json` (reusing cached clips automatically) and `02_analysis/raw_transcript_words.json`.

### Stage 2 — Coherent Slot Planning & Pre-Flight Confirmation
1. Run `select_optimal_subclips.py --style_dna 02_analysis/style_dna.json --raw_index 02_analysis/raw_scene_index.json` to synthesize `02_analysis/optimal_subclips.json` and `02_analysis/timeline_plan.json`:
   - Enforces `visual_coherence_target` (excludes off-theme indoor/enclosed shots when alpine/outdoor consistency applies);
   - Enforces `require_person` DeepLabV3 verification (`>= 2.0%` coverage) on all silhouette mask / depth-sandwich slots;
   - Uses adaptive high-FPS optical flow/Laplacian/composition sampling and disjoint `exclude_windows` reuse when cut count exceeds unique footage clips;
   - Locks sequence frame size (`strip_static_letterbox` into true `16:9` e.g. `3840x2160` when un-letterboxing) and camera native `recommended_sequence_fps` (e.g. `60fps`).
2. Present the pre-flight summary (resolution/aspect, target FPS, segment count, ASL, excluded off-theme clips, multi-strip border style, stepped aperture & audio-unlock cues, mask slots, typography reveals) to the user before mutating Premiere.

### Stage 3 — Sequence Snapshot & Frame-Snapped A-Roll/Strip Assembly
1. Create a sequence checkpoint (`create_sequence_checkpoint`) at target resolution & native camera FPS (`recommended_sequence_fps`), snapping all cut timestamps strictly to exact frame boundaries.
2. Execute `batch_razor_cuts` and silence removals (`>0.35s`) in **strictly descending chronological order** (`t_n > ... > t_1`).
3. Render multi-strip openings strictly per `multi_strip_opening.divider_line_px` (`0px` borderless vs ruled) and implement all stepped/slit aperture dynamics using `crop_no_stretch` (never vertical scale squeezing).

### Stage 4 — Multi-Track B-Roll, Head-Safe Subject Masks, Kinetic Typography & Pre-Import QA
1. **Track V2:** Cohesive B-roll cutaways, staggered PiP border collage tiles (`intra_shot_collage_events`), and geometric panel/split overlays (`geometric_panel_overlays` with symmetric margins and `continuous_1to1_screen_space` zero-jump continuity).
2. **Track V3:** PyTorch DeepLabV3 largest-central-component cutouts generated via `isolate_subject_alpha.py` (`verify_head_safe=True`, `enforce_depth_sandwich_width_ratio` so subjects occupy `30%–52%` of word width leaving outer letters 100% readable, and `render_clean_white_flash_cutout` for `#FFFFFF` full-white flashes—**never apply raw `255 - RGB` negative inversion to close-up human faces**), plus **translucent glass typography (`72%–78%` opacity + screen/difference luminance blend)** and exact 1f/2f blink strobes (`strobe_and_blink_schedules`).
3. **Mandatory 8-Gate Pre-Import Mathematical QA (`verify_edit_against_reference.py`):** Run `.agents/skills/reverse-engineering-style-dna/scripts/verify_edit_against_reference.py` to verify all 8 automated checks (including upper-border black-triangle detection, translucent glass check, and strobe/blink cadence) and emit `02_analysis/one_shot_qa_sheet.jpg` before assembling in Premiere Pro in a single batched `execute` turn.

### Stage 5 — 3-Pass Virtual-Scope Grading, Audio Mastering & Zero-Hang Windows Runtime Rules
1. Run `virtual_scopes.py` (`auto_crop_active=True` strips black bars automatically) to emit `02_analysis/color_telemetry.json`.
2. **Pass 1/2:** Normalize per-clip exposure/WB and shot-match B-roll to Hero A-roll before applying **Pass 3** global `lumetri_matrix` from `style_dna.json` on a Track `V4` Adjustment Layer (`look_intensity: 50–75`).
3. Master **Track A1** dialogue (`-6 dBFS`), align **Track A2** SFX transients `-3 to -5 frames` before cuts/apertures (`-12 to -18 dBFS`), and bed/duck **Track A3** music (`-18 dBFS` -> `-22 dBFS`). Always place `Audio 2`/`Audio 3` `.wav` items in Premiere Pro **before** populating `Video 1` (`track_index: 0`) so `.wav` overwrites never clear `Video 1`.
4. **4 Zero-Hang Windows Runtime Rules:**
   - **Stream 1 Frame at a Time:** Never load all 600+ full-HD frames (`4+ GB`) into a Python list simultaneously; compute `384x216` DeepLabV3 masks first, call `del deeplab; gc.collect()`, and stream 1 frame at a time (`cap.read()` -> ROI blend -> `vw.write()`).
   - **Zero `subprocess.PIPE` for Full-HD Video Streams:** Write intermediate streams via OpenCV `MJPG` `.avi` (`400 fps`) and encode final H.264/ProRes via `ffmpeg -nostdin -preset ultrafast` with `stdin=subprocess.DEVNULL`.
   - **Never Seek 4K HEVC with `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)`:** Always decode 4K snippets with `ffmpeg -nostdin -ss <sec> -t <dur> -i <file>` or reuse 1080p `graded_subclips/`.
   - **Always End PyTorch Scripts on Windows with `os._exit(0)`** after printing `[OK]` so C++ OpenMP/threadpool destructors never hang at process exit.
