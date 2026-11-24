"""Small ffmpeg/ffprobe helpers. No model code lives here.

Every function shells out to the system ``ffmpeg`` / ``ffprobe`` binaries,
which must be on PATH (``apt install ffmpeg`` on Ubuntu). Audio is handled
as mono 16-bit PCM WAV, the same interchange format V1 used on the phone.
"""

import json
import subprocess
from pathlib import Path


class FfmpegError(RuntimeError):
    pass


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise FfmpegError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.strip()[-2000:]}"
        )
    return result


def probe(path: Path) -> dict:
    """Return ffprobe's JSON for a media file."""
    result = run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)]
    )
    return json.loads(result.stdout)


def duration_seconds(path: Path) -> float:
    info = probe(path)
    return float(info["format"]["duration"])


def has_video_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "video" for s in probe(path)["streams"])


def has_audio_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path)["streams"])


def extract_audio(source: Path, dest_wav: Path, sample_rate: int = 16000) -> Path:
    """Extract the first audio track to mono PCM16 WAV at ``sample_rate``."""
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dest_wav),
    ])
    return dest_wav


def extract_audio_segment(source: Path, dest_wav: Path, start: float, length: float,
                          sample_rate: int = 16000) -> Path:
    run([
        "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(source),
        "-t", str(length), "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dest_wav),
    ])
    return dest_wav


def mux_video_audio(video: Path, audio_wav: Path, dest_mp4: Path,
                    video_codec: str = "copy") -> Path:
    """Re-mux a video with a new audio track.

    The picture track is copied untouched (``-c:v copy``), so the original
    frames, resolution and orientation are preserved. ``-shortest`` trims
    the video to the new audio if the new audio is shorter; if the new audio
    is longer, the last frame is held by the caller (see align.py) before
    this step, so the two tracks match in length.
    """
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", video_codec, "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", str(dest_mp4),
    ])
    return dest_mp4


def wav_duration(path: Path) -> float:
    return duration_seconds(path)
