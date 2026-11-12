---
trigger: always_on
description: Enforces Premiere Pro ExtendScript/CEP DOM invariants, frame tick math, reverse-order cutting, and bridge safety limits.
---

# Premiere Pro MCP Execution Invariants

1. **Timecode & Tick Conversion Standard:**
   - Premiere Pro ExtendScript measures timeline time strictly in ticks (`254016000000` ticks per second).
   - In all calculations, lock conversions:
     `ticks = Math.round(seconds * 254016000000)`
   - Query sequence frame rate first (`23.976`, `24`, `29.97`, `30`, or `60` fps) and snap all cut timestamps to an exact frame boundary (`frame_duration = 1 / fps`) to prevent sub-frame audio pops or black-frame blips.

2. **Reverse-Chronological Mutation Rule:**
   - Whenever calling `razor_clip`, `batch_razor_cuts`, or ripple deletion tools on any sequence track, always sort target timestamps in **strictly descending chronological order** (`t_n > t_{n-1} > ... > t_1`).
   - Cutting or deleting from start-to-finish (`00:00:00` forward) immediately shifts the timecodes and clip array indices of all downstream media, causing catastrophic cut desync.

3. **Safe Bridge Polling & Batching:**
   - Never fire more than 15 individual MCP tool calls in parallel over the CEP bridge.
   - Batch contiguous edits using `batch_razor_cuts` or consolidated ExtendScript calls to avoid wedging the CEP panel.

4. **Composition & Render Handoffs:**
   - Never import raw `.aep` files directly into Premiere Pro across the bridge. Render After Effects compositions and Remotion animations to ProRes 4444 `.mov` with an unmultiplied alpha channel (or `.mogrt` packages) and import the finished asset.

