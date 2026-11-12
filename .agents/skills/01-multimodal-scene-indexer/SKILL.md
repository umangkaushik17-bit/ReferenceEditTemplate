---
name: multimodal-scene-indexer
description: Indexes raw media files in 01_raw_media/ into frame-accurate scene manifests, word-level speech transcripts with silence detection (>0.35s), and optimizes peak cinematic sub-clip moments via computer vision (camera stability, focus/sharpness, composition, exposure) before editing in Premiere Pro.
---

# Multimodal Scene Indexer & Subclip Planner (v2.0)

Generates verified perception manifests across **both** `01_raw_media/a_roll/` and `01_raw_media/b_roll/` before touching the Premiere Pro timeline.

## Step 1 — Index Raw Media (`index_raw_media.py`)
```powershell
python .agents/skills/01-multimodal-scene-indexer/scripts/index_raw_media.py --raw_dir 01_raw_media --analysis_dir 02_analysis
```
Outputs `02_analysis/raw_scene_index.json` and `02_analysis/raw_transcript_words.json`:
- Probes true camera resolution, `aspect_ratio`, and authoritative **`fps`** (`recommended_sequence_fps`, e.g., `60fps`) per clip.
- Computes active-picture virtual scopes (`virtual_scopes.analyze_video_file_scopes`) and `environment` tags (`outdoor_green_landscape`, `outdoor_scenic`, `indoor_or_enclosed`, `green_fraction`) to catch dark/interior visual mismatches.
- Runs **PyTorch DeepLabV3 (`deeplabv3_mobilenet_v3_large`)** COCO class-15 person scanning to record `has_person`, `max_person_coverage_pct`, and `best_person_window_sec` across both `a_roll` and `b_roll`.
- Runs FFmpeg `silencedetect` (`> 0.35s`) on audio-bearing footage.

## Step 2 — Optimize Subclips & Synthesize Timeline Plan (`select_optimal_subclips.py`)
```powershell
python .agents/skills/01-multimodal-scene-indexer/scripts/select_optimal_subclips.py --style_dna 02_analysis/style_dna.json --raw_index 02_analysis/raw_scene_index.json --output 02_analysis/optimal_subclips.json --out_plan 02_analysis/timeline_plan.json
```
- **Visual Theme Filter:** Automatically excludes off-theme indoor/enclosed shots when `style_dna.json` specifies `outdoor_scene_consistency_required: true`.
- **Subject Verification Gate:** Slots tagged with `mask_transitions` (`require_person: true`) exclusively select windows verified by DeepLabV3 to contain `>= 2.0%` foreground human subject coverage (preventing empty-grass masks).
- **Adaptive Micro-Cut & Disjoint Reuse:** Scales optical-flow + Laplacian + composition + exposure sampling up to `>= 10 fps` for fast `< 0.6s` beat cuts and enforces non-overlapping `exclude_windows` whenever clips are reused across rapid beat montages.
- Emits both `02_analysis/optimal_subclips.json` and pre-flight `02_analysis/timeline_plan.json` (with un-letterboxed target canvas & native `recommended_sequence_fps`).
