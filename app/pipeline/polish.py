"""The "polish" finish, ported in spirit from ToneRoom V1.

V1 (``Sound.kt`` in toneroom-android, read for reference only — no code
was copied) polished a take with:
  * a 70 Hz high-pass filter,
  * a broad peak at 195 Hz (Warm +2.7 dB, Clear -2.0 dB, Balanced +0.4 dB),
  * a cut at 410 Hz (-1.5 dB) to reduce muddiness,
  * a presence peak at 2950 Hz (Warm +0.2 dB, Clear +3.2 dB, Balanced +1.6 dB),
  * gentle levelling, a compressor (threshold ~0.24, ~6 ms attack,
    ~140 ms release) and a soft peak ceiling at 0.8.

This module reproduces the same *design* with ffmpeg's audio filters.
It is an approximation, not a sample-identical port: ffmpeg's
``acompressor`` and ``alimiter`` are not V1's hand-written DSP. The
important properties are the same — speech timing is untouched, so
lip-sync is preserved exactly, and the accent is unchanged: polishing
only makes the voice clearer.

Finishes:
  balanced — steady volume, a little definition (default).
  warm     — fuller low notes, softer edges.
  clear    — brighter sound that brings words forward.
"""

from pathlib import Path

from .audio import run

# (195 Hz gain, 2950 Hz gain) per finish, in dB — same values as V1.
FINISH_EQ = {
    "balanced": (0.4, 1.6),
    "warm": (2.7, 0.2),
    "clear": (-2.0, 3.2),
}


def polish_filter(finish: str) -> str:
    """Build the ffmpeg audio-filter chain for one finish."""
    low_gain, presence_gain = FINISH_EQ[finish]
    parts = [
        "highpass=f=70",
        f"equalizer=f=195:width_type=q:width=0.8:g={low_gain}",
        "equalizer=f=410:width_type=q:width=0.9:g=-1.5",
        f"equalizer=f=2950:width_type=q:width=0.85:g={presence_gain}",
        # Gentle levelling + compression, then a soft peak ceiling.
        "acompressor=threshold=-12dB:ratio=3:attack=6:release=140:makeup=2",
        "alimiter=limit=0.8:level=false",
        "loudnorm=I=-16:TP=-1.5:LRA=11:linear=false",
    ]
    return ",".join(parts)


def polish_audio(source_wav: Path, dest_wav: Path, finish: str = "balanced") -> Path:
    """Polish a mono WAV in place (source -> dest). Timing is preserved."""
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source_wav),
        "-af", polish_filter(finish),
        "-c:a", "pcm_s16le", str(dest_wav),
    ])
    return dest_wav


def polish_video(source_video: Path, dest_mp4: Path, finish: str = "balanced") -> Path:
    """Polish the audio track of a video, copying the picture track.

    Output is H.264 video (copied) + AAC audio, faststart for the web.
    """
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source_video),
        "-map", "0:v:0", "-map", "0:a:0",
        "-af", polish_filter(finish),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(dest_mp4),
    ])
    return dest_mp4
