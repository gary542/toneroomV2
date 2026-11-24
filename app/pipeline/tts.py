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

Implementation note — this provider calls the ``sherpa-onnx-offline-tts``
command-line tool that ships with ``pip install sherpa-onnx``. The flags
used below are taken from the sherpa-onnx Pocket TTS documentation
(https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html), which shows the
same ``--pocket-lm-flow`` / ``--pocket-lm-main`` / ``--pocket-encoder`` /
``--pocket-decoder`` / ``--pocket-text-conditioner`` / ``--pocket-vocab-json`` /
``--pocket-token-scores-json`` / ``--reference-audio`` / ``--num-steps``
invocation. Using the CLI (rather than the Python API) keeps the exact
interface verifiable from that page.

STATUS: wired, NOT verified end-to-end in this workspace. The model
archive (~190 MB) was deliberately not downloaded here, and the
sherpa-onnx package was not installed here, so no real TTS call has
been run yet. Verify with:
    bash scripts/download_models.sh
    python -m app.pipeline.tts --self-test
and see docs/plan.md Phase 1 acceptance. The code must NOT be
presented as "TTS works" until that self-test passes.
"""

import argparse
import shutil
import subprocess
import wave
from pathlib import Path

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
    if shutil.which("sherpa-onnx-offline-tts") is None:
        return False, (
            "sherpa-onnx is not installed (no sherpa-onnx-offline-tts on PATH). "
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


def _run_pocket_chunk(text: str, reference_wav: Path, out_wav: Path) -> None:
    d = config.POCKET_MODEL_DIR
    cmd = [
        "sherpa-onnx-offline-tts",
        f"--pocket-lm-flow={d / 'lm_flow.int8.onnx'}",
        f"--pocket-lm-main={d / 'lm_main.int8.onnx'}",
        f"--pocket-encoder={d / 'encoder.onnx'}",
        f"--pocket-decoder={d / 'decoder.int8.onnx'}",
        f"--pocket-text-conditioner={d / 'text_conditioner.onnx'}",
        f"--pocket-vocab-json={d / 'vocab.json'}",
        f"--pocket-token-scores-json={d / 'token_scores.json'}",
        f"--reference-audio={reference_wav}",
        "--num-steps=2",
        f"--output-filename={out_wav}",
        text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or not out_wav.exists():
        raise RuntimeError(
            f"Pocket TTS failed ({result.returncode}): "
            f"{result.stderr.strip()[-1500:]}"
        )


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
    parts: list[Path] = []
    try:
        for i, chunk in enumerate(split_script(text)):
            part = work_dir / f"tts_part_{i:02d}.wav"
            _run_pocket_chunk(chunk, reference_wav, part)
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
