---
trigger: always_on
description: Retention-driven pacing standards, visual reset cadence, dead air thresholds, and J/L cut discipline.
---

# High-Retention Editing & Pacing Standards

1. **Adaptive Visual Reset Rule:**
   - For vertical short-form (9:16), never allow a static visual frame to linger for more than **2.5 to 3.5 seconds** without a visual reset.
   - For horizontal content (16:9), adapt visual reset intervals to **4.0 to 6.0 seconds** unless the `style_dna.json` specifies faster cadence.
   - A valid visual reset must be one of:
     - (a) Instant 1-frame punch-in scale jump (`100% -> 114%` to `118%`) on vocal emphasis words with +25px Y-axis repositioning to anchor eye-line.
     - (b) Smooth creeping tension zoom (`100% -> 108%`) with cubic Bezier ease handles.
     - (c) B-roll cutaway on Track `V2`.
     - (d) Pop-in graphic card, animated icon, or kinetic text highlight on Track `V3`.

2. **Dead Air & Dialogue Pacing:**
   - Strip all pauses exceeding `350ms` (`0.35s`) from the A-roll spine unless tagged as dramatic silence in `raw_transcript_words.json`.
   - Apply **J-Cuts** (audio leads video cut by 4–8 frames) and **L-Cuts** (audio trails video cut by 4–8 frames) to create seamless conversational momentum.

3. **SFX Transient Synchronization:**
   - Every whip-pan, scale jump, or graphic pop on `V2`/`V3` must have a corresponding audio transient on Track `A2`.
   - Align the audio file so its `peak_transient_ms` hits **3 to 5 frames BEFORE** the visual cut or graphic entrance frame.

