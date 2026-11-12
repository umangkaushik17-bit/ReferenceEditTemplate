---
name: bg-removal-compositing
description: Isolates foreground subjects using local or cloud AI matting, exporting ProRes 4444 with alpha to 03_processed_vfx/cutouts/ for 3-layer V1/V2/V3 sandwich titles.
---

# Foreground Subject Matting & Silhouette Compositing (v2.0)

Uses **PyTorch DeepLabV3** (`deeplabv3_mobilenet_v3_large` / `deeplabv3_resnet50`, COCO class `15` person segmentation) + FFmpeg encoding — with zero silent dummy fallbacks — to generate foreground cutouts and silhouette transitions.

## Modes
- `alpha_cutout`: Renders transparent-alpha **ProRes 4444** (`.mov`, `yuva444p10le`) for Track `V3` foreground overlays (depth-sandwiching titles on Track `V2` behind the speaker).
- `silhouette_black`: Renders crisp `#FFFFFF` foreground subject over `#000000` background (`mask_on_black`).
- `silhouette_live`: Renders crisp `#FFFFFF` foreground subject composited directly over live footage (`mask_over_live_bg`).

## Usage
```powershell
python .agents/skills/03-bg-removal-compositing/scripts/isolate_subject_alpha.py --input "01_raw_media/b_roll/20260715_061851.mp4" --output "03_processed_vfx/cutouts/walker_cutout.mov" --mode alpha_cutout --start_sec 1.0 --duration_sec 2.0
```

## Mandatory Guardrails
- Always source `--input` and `--start_sec` from a person-verified window (`person_tracking.has_person == true`) in `02_analysis/raw_scene_index.json` or `02_analysis/optimal_subclips.json`.
- Script preserves native input `fps` (e.g. `60fps`) and **fails loudly** if maximum detected person coverage falls below `--min_coverage_pct` (default `0.5%`), preventing accidental blank or empty-grass mattes.
