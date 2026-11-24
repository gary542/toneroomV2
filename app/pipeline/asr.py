"""Speech-to-text: faster-whisper (Whisper on CTranslate2).

Chosen after the research in docs/research.md:
  * MIT licence, runs on CPU with int8 quantisation, word-level
    timestamps (needed for Phase 2 timing alignment), VAD filtering.
  * Default model "small": the README benchmark transcribes 13 minutes
    in 1m42s int8 on an 8-thread i7-12700K using ~1.5 GB RAM.
  * The model is downloaded from Hugging Face on first use; it is NOT
    bundled here. See scripts/download_models.sh.

This module is a thin, lazy wrapper: importing it must never fail when
faster-whisper is not installed, so the polish path still works on a
bare server. The failure is raised only when a job actually needs ASR,
and the message says exactly what to install.
"""

from pathlib import Path
import wave

import numpy as np

from .. import config


class ModelNotAvailable(RuntimeError):
    """Raised when a model or its runtime is not installed/configured."""


def _read_wav_mono16k(wav_path: Path) -> np.ndarray:
    """Read a mono 16 kHz PCM16 WAV as float32 samples in [-1, 1].

    The pipeline always produces this exact format (see
    app/pipeline/audio.py extract_audio), so stdlib ``wave`` is enough
    here. This deliberately avoids faster-whisper's own audio decoding,
    which goes through PyAV: sherpa-onnx 2.0.3 pulls in av 19.x, and
    faster-whisper 1.2.1 calls ``av.open(..., metadata_errors=...)``,
    which av 19 removed — so passing a file path fails with
    ``TypeError: open() got an unexpected keyword argument
    'metadata_errors'``. Passing a numpy array skips that path entirely.
    """
    with wave.open(str(wav_path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"expected mono 16-bit WAV, got {w.getnchannels()} channels, "
                f"{w.getsampwidth() * 8}-bit: {wav_path}"
            )
        if w.getframerate() != 16000:
            raise ValueError(
                f"expected 16 kHz WAV, got {w.getframerate()} Hz: {wav_path}"
            )
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav_path: Path, language: str = "en") -> list[dict]:
    """Transcribe a mono WAV. Returns segments with word timestamps.

    Each segment: {"start": float, "end": float, "text": str,
                   "words": [{"start": float, "end": float, "word": str}]}
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ModelNotAvailable(
            "faster-whisper is not installed. Install the ML requirements "
            "(`pip install -r requirements-ml.txt`) and download the model "
            "with scripts/download_models.sh, then retry."
        ) from exc

    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    segments, _info = model.transcribe(
        _read_wav_mono16k(wav_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )
    out: list[dict] = []
    for seg in segments:
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in (seg.words or [])
            ],
        })
    return out


def segments_to_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip()
