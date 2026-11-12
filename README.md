# Autonomous Post-Production Director (Premiere Pro MCP)

An AI-orchestrated video editing and reference reverse-engineering workspace for **Adobe Premiere Pro** via the **Model Context Protocol (MCP)**.

Instead of editing blind from filenames or guessing color parameters, the system forces a perception-first pipeline: every cut, transition, subject silhouette mask, kinetic title, and 3-pass Lumetri grade is driven by telemetry manifests extracted via OpenCV, Librosa, PyTorch DeepLabV3, Tesseract OCR, and FFmpeg before touching the Premiere Pro timeline.

---

## Key Capabilities

- **4-Layer Reference Style DNA (`reverse-engineering-style-dna`):**
  - Separates container geometry vs. active picture bounds (`active_picture_bounds_xywh`) to strip letterbox/windowbox bars (`crop_no_stretch`) instead of stretching footage.
  - Detects staggered multi-strip openings (`divider_line_px: 0` borderless vs. ruled), stepped aperture closes, expanding slit unlocks synced to audio resurgence, rapid beat cuts down to ~`0.13s`, dual silhouette modes (`mask_on_black` / `mask_over_live_bg`), dynamic Tesseract OCR reveal animations, active-region Lumetri profiles, and extracts reference audio into `04_assets/music/reference_audio.wav`.
- **Multimodal Raw Media Indexing & Coherent Planning (`01-multimodal-scene-indexer`):**
  - Indexes both `01_raw_media/a_roll/` and `01_raw_media/b_roll/` (plus `audio/`), locks native sequence FPS (`recommended_sequence_fps`), classifies indoor vs. outdoor visual coherence, runs PyTorch DeepLabV3 person tracking, and selects disjoint optimal subclips (`exclude_windows`) with subject-verified mask slots.
- **Person-Verified Subject Cutouts & Silhouettes (`03-bg-removal-compositing`):**
  - Generates ProRes 4444 `.mov` alpha cutouts (`alpha_cutout`), black-background white silhouettes (`silhouette_black`), and live-background overlays (`silhouette_live`) via DeepLabV3 + FFmpeg, refusing empty scenery via `min_coverage_pct` guardrails.
- **3-Pass Closed-Loop Virtual-Scope Color Grading (`balancing-primary-color-and-skins` & `applying-creative-lumetri-looks`):**
  - Auto-crops black bars (`extract_active_region`), computes Luma IRE (`p01`–`p99`), RGB Parade balance, and YCbCr skin-tone telemetry (`02_analysis/color_telemetry.json`), balances per-clip Pass 1/Pass 2 before applying Pass 3 creative `lumetri_matrix` on Track `V4`.
- **Frame-Snapped Reverse-Chronological Premiere Execution:**
  - Enforces exact Premiere ticks (`254016000000` ticks/sec), integer frame snapping, descending cut execution (`t_n > ... > t_1`), and Track `A1`/`A2`/`A3` loudness mastering (`-6 dBFS` dialogue, `-12 to -18 dBFS` SFX `-3 to -5 frames` pre-cut, `-18 -> -22 dBFS` ducked music).

---

## Workspace Layout

```text
├── AGENTS.md                         # Master 5-stage Autonomous Post-Production Director protocol
├── STEP_0_PREMIERE_MCP_SETUP.md      # Universal Step 0 guide: any IDE <-> Premiere Pro MCP
├── setup_premiere_bridge.ps1         # Windows setup, CEP diagnostics & IDE config generator
├── setup_premiere_bridge.sh          # macOS / Linux setup & diagnostics script
├── .mcp.json                         # Workspace MCP server definition (premiere-pro-mcp)
├── requirements.txt                  # Python perception & compositing dependencies
├── .agents/
│   ├── rules/                        # Always-on guardrails (DOM, pacing, color, audio, reverse-engineering)
│   └── skills/                       # Executable Python/ExtendScript skills & SKILL.md playbooks
├── 00_reference_style/               # Drop a reference edit (.mp4/.mov) here to clone its style DNA
├── 01_raw_media/
│   ├── a_roll/                       # Primary footage clips
│   ├── b_roll/                       # Cutaway / subject / secondary clips
│   └── audio/                        # Standalone narration or music stems
├── 02_analysis/                      # Generated manifests (style_dna.json, raw_scene_index.json, timeline_plan.json, color_telemetry.json)
├── 03_processed_vfx/
│   ├── captions/                     # Subtitle / caption artifacts
│   ├── cutouts/                      # ProRes 4444 alpha cutouts & silhouette plates
│   └── rendered_titles/              # Rendered kinetic title overlays
└── 04_assets/
    ├── fonts/                        # Custom project fonts (e.g., Cinematic_Serif.ttf)
    ├── music/                        # Extracted/cloned music beds (ignored by git)
    ├── sfx/                          # Indexed sound effects library
    └── vault_index.json              # Acoustic transient & music vault index
```

---

## Prerequisites

1. **Adobe Premiere Pro** (2020–2026)
2. **Node.js `20.19+`** (`npm install -g premiere-pro-mcp`)
3. **Python `3.10+`** with packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. **System binaries on `PATH`:**
   - **FFmpeg & FFprobe** (`winget install Gyan.FFmpeg` on Windows / `brew install ffmpeg` on macOS)
   - **Tesseract OCR** (for dynamic reference title/caption extraction)

---

## Step 0 — Connect Your IDE to Premiere Pro MCP

Works with **Antigravity**, **Cursor**, **VS Code / GitHub Copilot**, **Windsurf**, **Claude Code**, **Claude Desktop**, **OpenAI Codex CLI**, or any stdio MCP client.

See [`STEP_0_PREMIERE_MCP_SETUP.md`](STEP_0_PREMIERE_MCP_SETUP.md) for full instructions, or run:

**Windows (PowerShell):**
```powershell
# Install premiere-pro-mcp + Premiere CEP connector + run diagnostics + print IDE configs:
powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -Install

# Check setup & live heartbeat only (read-only):
powershell -ExecutionPolicy Bypass -File .\setup_premiere_bridge.ps1 -CheckOnly
```

**macOS / Linux:**
```bash
chmod +x ./setup_premiere_bridge.sh
./setup_premiere_bridge.sh --install
```

Inside Premiere Pro, open **Window > Extensions > MCP for Adobe Premiere Pro** (confirm green **Running**), then ask your assistant:
> `Call ping and verify_premiere_connection. Make no changes.`

---

## How to Run a New Edit

1. Place an optional target style video in [`00_reference_style/`](00_reference_style/).
2. Place your footage in [`01_raw_media/a_roll/`](01_raw_media/a_roll/), [`01_raw_media/b_roll/`](01_raw_media/b_roll/), and [`01_raw_media/audio/`](01_raw_media/audio/).
3. Open a Premiere Pro project with a sequence active and verify the CEP MCP Bridge panel is **Running**.
4. Instruct your IDE AI agent using the workflow defined in [`AGENTS.md`](AGENTS.md) (Stages 1–5):
   - Extract reference Style DNA (`extract_style_dna.py`)
   - Index all A-roll/B-roll footage (`index_raw_media.py`)
   - Build the coherent subclip & timeline plan (`select_optimal_subclips.py`)
   - Confirm the pre-flight plan, assemble the sequence in reverse-chronological order, generate DeepLabV3 cutouts (`isolate_subject_alpha.py`), grade via `virtual_scopes.py`, and master audio (`A1`/`A2`/`A3`).
