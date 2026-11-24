"""Text-to-speech: Kyutai Pocket TTS via sherpa-onnx (CPU).

Chosen after the research in docs/research.md:
  * Pocket TTS is a ~100M-parameter model that runs on CPU
    (~6x real-time on 2 cores of a MacBook Air M4, per Kyutai),
    clones a voice zero-shot from a short reference clip, and needs
    NO transcript of the reference — only the audio file.
  * Code: MIT (sherpa-onnx, Apache-2.0). Weights: CC BY 4.0, with
    Kyutai's condition that the voice is used only with the recorded
    person's consent. V2 enforces that consent at the API layer.
  * V1 used the same model, bundled as the sherpa-onnx int8 ONNX
    conversion of January 2026. V2 reuses those same model files
    (downloaded, not copied from V1's repo).

Implementation note — this uses the ``sherpa_onnx`` Python API
(``OfflineTts`` / ``OfflineTtsPocketModelConfig`` /
``GenerationConfig``), the same interface V1 used on Android.
An earlier version of this file called a ``sherpa-onnx-offline-tts``
command-line tool; that binary does NOT ship with sherpa-onnx 2.0.3
on pip (only ``sherpa-onnx-cli`` with a ``text2token`` command), so
every script/respeak job failed with "no such file". Fixed 24 Nov
2026 during the Phase 2 test.

STATUS: verified end-to-end 24 Nov 2026 (Phase 2 test) once the
Python-API rewrite below landed. See docs/plan.md Phase 2.
"""

import argparse
import wave
from pathlib import Path

import numpy as np

from .. import config

# V1 split long scripts into passages of at most 180 characters, breaking
# on sentence ends where possible. Same rule here, so one bad passage
# cannot ruin a whole generation.
CHUNK_MAX = 180
# Silence (seconds) inserted between generated passages.
CHUNK_GAP_S = 1 / 7


class ModelNotAvailable(RuntimeError):
    """Raised when the TTS runtime or model files are missing."""


POCKET_FILES = (
    "lm_flow.int8.onnx",
    "lm_main.int8.onnx",
    "encoder.onnx",
    "decoder.int8.onnx",
    "text_conditioner.onnx",
    "vocab.json",
    "token_scores.json",
)


def model_dir_ready() -> tuple[bool, str]:
    """Check the Pocket TTS model directory. Returns (ok, message)."""
    d = config.POCKET_MODEL_DIR
    missing = [f for f in POCKET_FILES if not (d / f).exists()]
    if missing:
        return False, (
            f"Pocket TTS model files missing in {d}: {', '.join(missing)}. "
            "Run: bash scripts/download_models.sh "
            "(downloads sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2, "
            "about 190 MB, CC BY 4.0)."
        )
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False, (
            "sherpa-onnx is not installed. "
            "Run: pip install -r requirements-ml.txt"
        )
    return True, ""


def split_script(text: str) -> list[str]:
    """Split text into <=180-char passages, preferring sentence ends."""
    rest = " ".join(text.split()).strip()
    out: list[str] = []
    while len(rest) > CHUNK_MAX:
        head = rest[:CHUNK_MAX]
        cut = max(head.rfind(c) for c in ".!?;")
        if cut >= 45:
            end = cut + 1
        else:
            space = head.rfind(" ")
            end = space if space > 0 else CHUNK_MAX
        out.append(rest[:end].strip())
        rest = rest[end:].strip()
    if rest:
        out.append(rest)
    return out


def _read_reference(reference_wav: Path) -> tuple[np.ndarray, int]:
    """Read the reference WAV, trimming leading/trailing silence.

    Same rule V1 used on the phone: keep a little room either side of
    the speech, require at least 2 s of active speech, cap at 12 s.
    """
    with wave.open(str(reference_wav), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"reference must be mono 16-bit WAV, got "
                f"{w.getnchannels()} channels, {w.getsampwidth() * 8}-bit"
            )
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    peak = int(np.max(np.abs(pcm))) if pcm.size else 0
    edge = max(int(peak * 0.024), 120)
    active = np.where(np.abs(pcm) > edge)[0]
    if active.size == 0 or (active[-1] - active[0]) < rate * 2:
        raise ValueError(
            "voice reference has less than 2 seconds of clear speech; "
            "record a fresh 4–12 s sample."
        )
    start = max(0, int(active[0]) - rate // 12)
    end = min(pcm.size, int(active[-1]) + rate // 10)
    trimmed = pcm[start:end]
    if trimmed.size > rate * 12:
        trimmed = trimmed[: rate * 12]
    return trimmed.astype(np.float32) / 32768.0, rate


def _make_tts():
    """Create one Pocket TTS engine (model is loaded once per job)."""
    import sherpa_onnx

    d = config.POCKET_MODEL_DIR
    pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
        lm_flow=str(d / "lm_flow.int8.onnx"),
        lm_main=str(d / "lm_main.int8.onnx"),
        encoder=str(d / "encoder.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        text_conditioner=str(d / "text_conditioner.onnx"),
        vocab_json=str(d / "vocab.json"),
        token_scores_json=str(d / "token_scores.json"),
        voice_embedding_cache_capacity=1,
    )
    tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(pocket=pocket, num_threads=2)
        )
    )
    return tts


def _write_wav(samples: np.ndarray, rate: int, dest: Path) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).round().astype(np.int16)
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return dest


def _run_pocket_chunk(tts, gen_config, text: str, out_wav: Path) -> Path:
    generated = tts.generate(text, gen_config)
    samples = np.asarray(generated.samples, dtype=np.float32)
    if samples.size < generated.sample_rate // 8:
        raise RuntimeError(
            "Pocket TTS produced almost no audio for a passage; "
            "try a shorter sentence."
        )
    return _write_wav(samples, int(generated.sample_rate), out_wav)


def _concat_wavs(parts: list[Path], dest: Path, gap_s: float = CHUNK_GAP_S) -> Path:
    """Concatenate mono WAVs, inserting a short silence gap between them."""
    if not parts:
        raise ValueError("no audio parts to join")
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
        rate = first.getframerate()
        frames = [first.readframes(first.getnframes())]
    silence = b"\x00\x00" * int(rate * gap_s)
    for part in parts[1:]:
        with wave.open(str(part), "rb") as w:
            if w.getframerate() != rate or w.getnchannels() != 1:
                raise RuntimeError(f"unexpected WAV format in {part}")
            frames.append(silence)
            frames.append(w.readframes(w.getnframes()))
    with wave.open(str(dest), "wb") as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
    return dest


def generate(text: str, reference_wav: Path, dest_wav: Path,
             work_dir: Path) -> Path:
    """Generate speech in the reference voice. Returns ``dest_wav``.

    Raises ModelNotAvailable if the runtime/model is missing, so callers
    can fail the job with a clear message instead of a fake success.
    """
    ok, message = model_dir_ready()
    if not ok:
        raise ModelNotAvailable(message)
    text = " ".join(text.split()).strip()
    if not text:
        raise ValueError("empty script text")
    work_dir.mkdir(parents=True, exist_ok=True)
    import sherpa_onnx

    ref_samples, ref_rate = _read_reference(reference_wav)
    tts = _make_tts()
    gen_config = sherpa_onnx.GenerationConfig()
    # GenerationConfig takes no constructor arguments in the Python
    # binding; set the fields on the object instead.
    gen_config.reference_audio = ref_samples.tolist()
    gen_config.reference_sample_rate = ref_rate
    gen_config.num_steps = 2
    parts: list[Path] = []
    try:
        for i, chunk in enumerate(split_script(text)):
            part = work_dir / f"tts_part_{i:02d}.wav"
            _run_pocket_chunk(tts, gen_config, chunk, part)
            parts.append(part)
        return _concat_wavs(parts, dest_wav)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pocket TTS self-test")
    parser.add_argument("--reference", required=True, help="reference WAV")
    parser.add_argument("--text", default="Hello, this is a ToneRoom test.")
    parser.add_argument("--out", default="/tmp/toneroom_tts_test.wav")
    args = parser.parse_args()
    out = generate(args.text, Path(args.reference), Path(args.out),
                   Path("/tmp/toneroom_tts_work"))
    print(f"OK: wrote {out} ({out.stat().st_size} bytes)")
