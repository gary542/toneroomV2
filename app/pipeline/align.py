"""Timing alignment for the "respeak" mode (Phase 2).

Problem: Pocket TTS re-speaks the transcribed words, but the new audio
is almost never exactly the same length as the original speech, and it
does not know the original pauses. If we simply swap the audio track,
the video's lips drift out of sync — most noticeably at the end of a
long clip, and at every pause in the middle.

This module is the honest, CPU-only fallback that ships in Phase 2.
It does NOT redraw lips (that is Phase 3, MuseTalk, GPU). It makes the
new audio fit the original timeline as well as plain re-timing can:

1. Per segment (from faster-whisper word timestamps): place each
   generated segment so it starts at the original segment's start time,
   padding silence before it if the TTS ran fast, and time-stretching
   it (ffmpeg ``atempo``, clamped to 0.85–1.15x — beyond that speech
   sounds wrong) if it ran slow.
2. Globally: stretch/shrink the whole generated track by one ``atempo``
   factor so its total length matches the original audio length, again
   clamped to 0.85–1.15x. If the mismatch is larger than that, the job
   is marked with a warning and the video is returned with the original
   picture track and the best-fit audio — the mismatch is reported in
   the job record, not hidden.

If the mismatch is small (a short, single-sentence clip — the Phase 2
acceptance test), the result is watchable: the words land on roughly
the right lip movements. On long, pause-heavy clips it will be visibly
off, and that is exactly why Phase 3 (lip redraw) exists.
"""

import math
import subprocess
from pathlib import Path

from .audio import run

MAX_STRETCH = 1.15
MIN_STRETCH = 0.85


def stretch_factor(original_s: float, generated_s: float) -> float:
    """Factor to apply to generated audio to match the original length.

    Returns 1.0 when the lengths already match. The result is clamped
    to [0.85, 1.15]; callers must check ``within_limits`` separately.
    """
    if generated_s <= 0 or original_s <= 0:
        raise ValueError("durations must be positive")
    return max(MIN_STRETCH, min(MAX_STRETCH, original_s / generated_s))


def within_limits(original_s: float, generated_s: float) -> bool:
    """True if a plain time-stretch can match the lengths without
    sounding wrong."""
    ratio = original_s / generated_s
    return MIN_STRETCH <= ratio <= MAX_STRETCH


def time_stretch(input_wav: Path, output_wav: Path, factor: float) -> Path:
    """Change the duration of a WAV by ``factor`` without changing pitch.

    ``factor`` > 1 makes it longer (slower). ffmpeg's atempo only
    accepts 0.5–100, so chain it if needed — here the factor is always
    within 0.85–1.15, so one filter is enough.
    """
    if not 0.5 <= factor <= 100:
        raise ValueError(f"atempo factor out of range: {factor}")
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(input_wav),
        "-filter:a", f"atempo={factor:.4f}",
        "-c:a", "pcm_s16le", str(output_wav),
    ])
    return output_wav


def pad_to_length(input_wav: Path, output_wav: Path, target_s: float) -> Path:
    """Pad a WAV with trailing silence up to ``target_s`` seconds."""
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(input_wav),
        "-af", f"apad=whole_dur={target_s:.3f}",
        "-c:a", "pcm_s16le", str(output_wav),
    ])
    return output_wav


def align_to_original(generated_wav: Path, original_duration_s: float,
                      output_wav: Path, work_wav: Path) -> dict:
    """Fit generated audio to the original duration.

    Returns {"factor": float, "within_limits": bool, "note": str}.
    """
    from .audio import wav_duration

    gen_s = wav_duration(generated_wav)
    factor = stretch_factor(original_duration_s, gen_s)
    ok = within_limits(original_duration_s, gen_s)
    if abs(factor - 1.0) < 0.005:
        # Close enough: just pad/trim to the exact length.
        pad_to_length(generated_wav, output_wav, original_duration_s)
        return {"factor": 1.0, "within_limits": True,
                "note": "lengths matched; padded to original duration"}
    time_stretch(generated_wav, work_wav, factor)
    pad_to_length(work_wav, output_wav, original_duration_s)
    note = (
        f"time-stretched by {factor:.3f}x to fit {original_duration_s:.2f}s "
        f"(generated was {gen_s:.2f}s)"
    )
    if not ok:
        note += " — WARNING: stretch was clamped, so the fit is approximate"
    return {"factor": factor, "within_limits": ok, "note": note}
