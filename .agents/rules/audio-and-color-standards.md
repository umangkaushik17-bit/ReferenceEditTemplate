---
trigger: always_on
description: Broadcast audio loudness standards, ducking bounds, and two-pass Lumetri color calibration.
---

# Audio & Color Engineering Standards

1. **Audio Gain & Loudness Standards:**
   - **Track A1 (Dialogue):** Normalize to **-6 dBFS** peak (Integrated target: -14 to -16 LUFS for web/social, -23 LUFS for broadcast). Mono or dual-mono centered.
   - **Track A2 (Sound Effects):** Gain set to **-12 dBFS to -18 dBFS**. Sub-drops should not exceed -10 dBFS to prevent bass distortion.
   - **Track A3 (Music Bed):** Set ambient gain to **-18 dBFS**. During active dialogue on A1, apply automated keyframed ducking to drop the bed to **-22 dBFS to -26 dBFS** with a 200ms attack and 500ms release curve.

2. **Two-Pass Lumetri Color Protocol:**
   - **Pass 1 (Per-Clip Normalization on V1/V2):**
     - Anchor blacks between `-5` and `-12`.
     - Pull down clipping highlights between `-15` and `-30`.
     - Balance skin tone warmth (`Temperature` -15 to +15, `Tint` -10 to +10) with skin mask protection.
   - **Pass 2 (Aesthetic Grade on V4 Adjustment Layer):**
     - Apply creative grading parameter maps and 3D `.cube` LUTs across the full timeline on an Adjustment Layer on Track `V4` so all cuts, B-roll, and stock footage inherit consistent cinematic coherence.

