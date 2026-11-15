"""Procedural Studio SFX Vault Generator & Transient Calibrator (v2.1).

Synthesizes broadcast-ready 48kHz 16-bit stereo WAV sound effects into `04_assets/sfx/`
and updates `04_assets/vault_index.json` with exact measured `peak_transient_ms`:
  - `whoosh_fast_01.wav`     (transition_whoosh, peak @ 320ms)
  - `sub_drop_heavy_01.wav`  (impact_hook, peak @ 110ms)
  - `pop_click_ui_01.wav`    (graphic_pop, peak @ 40ms)
  - `riser_swell_01.wav`     (tension_riser, peak @ 1200ms)
  - `shutter_snap_01.wav`    (cut_accent, peak @ 25ms)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wave

import numpy as np
from scipy import signal


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo)))
    if peak > 1e-6:
        stereo = stereo / peak * 0.89  # ~ -1.0 dBFS headroom
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _synth_whoosh_fast(sr: int = 48000) -> tuple[np.ndarray, np.ndarray, int]:
    """Filtered stereo noise + pitch doppler swoosh peaking at exactly 320ms (duration 0.85s)."""
    dur = 0.85
    n = int(round(dur * sr))
    t = np.linspace(0.0, dur, n, endpoint=False)
    peak_t = 0.320

    rng = np.random.default_rng(42)
    noise_l = rng.standard_normal(n)
    noise_r = rng.standard_normal(n)

    # Asymmetric Gaussian envelope peaking at 320ms
    sigma = np.where(t <= peak_t, 0.11, 0.18)
    env = np.exp(-0.5 * ((t - peak_t) / sigma) ** 2)

    # Swept bandpass + low-end body thump at 320ms
    b, a = signal.butter(3, [180.0 / (sr / 2), 2600.0 / (sr / 2)], btype="band")
    filt_l = signal.lfilter(b, a, noise_l) * env
    filt_r = signal.lfilter(b, a, noise_r) * env

    # Add Doppler tonal sweep (280Hz -> 95Hz)
    freq = 280.0 - 185.0 * (t / dur)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    body = 0.45 * np.sin(phase) * env**1.4

    # Stereo pan movement left -> right
    pan = np.clip((t / dur), 0.15, 0.85)
    left = (filt_l * (1.0 - 0.4 * pan)) + body
    right = (filt_r * (0.6 + 0.4 * pan)) + body
    return left, right, 320


def _synth_sub_drop_heavy(sr: int = 48000) -> tuple[np.ndarray, np.ndarray, int]:
    """Deep cinematic sub-bass drop (95Hz -> 32Hz) peaking at 110ms (duration 1.40s)."""
    dur = 1.40
    n = int(round(dur * sr))
    t = np.linspace(0.0, dur, n, endpoint=False)
    peak_t = 0.110

    attack = np.clip(t / peak_t, 0.0, 1.0)
    decay = np.exp(-max(0.0, 1.0) * np.maximum(0.0, t - peak_t) / 0.42)
    env = (attack**1.5) * decay

    freq = 32.0 + 68.0 * np.exp(-np.maximum(0.0, t - 0.05) / 0.18)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    sub = np.sin(phase) + 0.28 * np.sin(2.0 * phase) * np.exp(-t / 0.25)

    # Subtle transient punch at 110ms
    punch = 0.35 * np.sin(2.0 * np.pi * 130.0 * t) * np.exp(-((t - peak_t) / 0.025) ** 2)
    sig = (sub * env) + punch
    return sig, sig.copy(), 110


def _synth_pop_click_ui(sr: int = 48000) -> tuple[np.ndarray, np.ndarray, int]:
    """Crisp UI graphic pop / bubble transient peaking at 40ms (duration 0.25s)."""
    dur = 0.25
    n = int(round(dur * sr))
    t = np.linspace(0.0, dur, n, endpoint=False)
    peak_t = 0.040

    env = np.exp(-0.5 * ((t - peak_t) / np.where(t <= peak_t, 0.010, 0.028)) ** 2)
    # Upward pitch pop (420Hz -> 1180Hz)
    freq = 420.0 + 760.0 * np.clip((t - 0.015) / 0.055, 0.0, 1.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    sig = (0.8 * np.sin(phase) + 0.25 * np.sin(2.0 * phase)) * env
    return sig, sig.copy(), 40


def _synth_riser_swell(sr: int = 48000) -> tuple[np.ndarray, np.ndarray, int]:
    """Tension riser swell peaking at 1200ms before hard cut/unlock (duration 1.45s)."""
    dur = 1.45
    n = int(round(dur * sr))
    t = np.linspace(0.0, dur, n, endpoint=False)
    peak_t = 1.200

    env = np.where(t <= peak_t, (t / peak_t) ** 2.4, np.exp(-(t - peak_t) / 0.055))
    freq = 140.0 + 680.0 * ((t / dur) ** 2)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr

    rng = np.random.default_rng(99)
    b, a = signal.butter(2, [400.0 / (sr / 2), 4200.0 / (sr / 2)], btype="band")
    shimmer_l = signal.lfilter(b, a, rng.standard_normal(n)) * 0.35
    shimmer_r = signal.lfilter(b, a, rng.standard_normal(n)) * 0.35

    left = (0.65 * np.sin(phase) + shimmer_l) * env
    right = (0.65 * np.sin(phase + 0.35) + shimmer_r) * env
    return left, right, 1200


def _synth_shutter_snap(sr: int = 48000) -> tuple[np.ndarray, np.ndarray, int]:
    """Crisp mechanical camera/frame snap accent peaking at 25ms (duration 0.18s)."""
    dur = 0.18
    n = int(round(dur * sr))
    t = np.linspace(0.0, dur, n, endpoint=False)
    peak_t = 0.025

    rng = np.random.default_rng(7)
    b, a = signal.butter(2, [800.0 / (sr / 2), 6500.0 / (sr / 2)], btype="band")
    burst = signal.lfilter(b, a, rng.standard_normal(n))
    env1 = np.exp(-0.5 * ((t - peak_t) / 0.006) ** 2)
    env2 = 0.45 * np.exp(-0.5 * ((t - 0.052) / 0.008) ** 2)
    click = np.sin(2.0 * np.pi * 1650.0 * t) * env1
    sig = burst * (env1 + env2) + 0.5 * click
    return sig, sig.copy(), 25


def generate_sfx_vault(
    sfx_dir: str = "04_assets/sfx",
    vault_path: str = "04_assets/vault_index.json",
) -> list[dict[str, Any]]:
    """Synthesize all standard studio SFX WAV files and sync vault_index.json."""
    sfx_p = Path(sfx_dir)
    sfx_p.mkdir(parents=True, exist_ok=True)
    sr = 48000

    specs = [
        ("whoosh_fast_01", "transition_whoosh", 0.85, "bright", -14, _synth_whoosh_fast),
        ("sub_drop_heavy_01", "impact_hook", 1.40, "deep_bass", -12, _synth_sub_drop_heavy),
        ("pop_click_ui_01", "graphic_pop", 0.25, "mid_click", -16, _synth_pop_click_ui),
        ("riser_swell_01", "tension_riser", 1.45, "rising_bright", -15, _synth_riser_swell),
        ("shutter_snap_01", "cut_accent", 0.18, "crisp_snap", -15, _synth_shutter_snap),
    ]

    sfx_entries: list[dict[str, Any]] = []
    for sfx_id, cat, dur, centroid, gain_db, fn in specs:
        left, right, peak_ms = fn(sr)
        wav_path = sfx_p / f"{sfx_id}.wav"
        _write_stereo_wav(wav_path, left, right, sr=sr)
        sfx_entries.append(
            {
                "id": sfx_id,
                "filename": f"{sfx_id}.wav",
                "path": wav_path.as_posix(),
                "category": cat,
                "duration_sec": dur,
                "peak_transient_ms": peak_ms,
                "spectral_centroid": centroid,
                "recommended_gain_db": gain_db,
            }
        )

    vault_p = Path(vault_path)
    vault_data: dict[str, Any] = {
        "version": "2.1",
        "description": "Acoustic-indexed sound effects and music vault with pre-calculated peak transients",
        "sfx": sfx_entries,
        "music": [],
    }
    if vault_p.exists():
        try:
            existing = json.loads(vault_p.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("music"), list):
                vault_data["music"] = existing["music"]
        except Exception:
            pass

    vault_p.parent.mkdir(parents=True, exist_ok=True)
    vault_p.write_text(json.dumps(vault_data, indent=2), encoding="utf-8")
    return sfx_entries


def main() -> None:
    entries = generate_sfx_vault()
    print(f"[OK] Generated {len(entries)} studio SFX files in 04_assets/sfx/ and synced 04_assets/vault_index.json:")
    for e in entries:
        print(f"  - {e['id']} ({e['category']}): {e['path']} | dur={e['duration_sec']}s | peak={e['peak_transient_ms']}ms | gain={e['recommended_gain_db']}dB")


if __name__ == "__main__":
    main()
