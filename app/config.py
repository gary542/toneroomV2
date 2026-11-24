"""Configuration for ToneRoom V2, read from the environment.

No secrets live in code. Copy ``.env.example`` to ``.env`` and edit it.
Only the values below are read; anything else in the environment is ignored.
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("TONEROOM_DATA_DIR", str(BASE_DIR / "data")))
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = Path(_env("TONEROOM_MODELS_DIR", str(BASE_DIR / "models")))

# Upload limits. V1 capped a take at 300 seconds; V2 keeps the same cap.
MAX_UPLOAD_MB = int(_env("TONEROOM_MAX_UPLOAD_MB", "200"))
MAX_DURATION_S = int(_env("TONEROOM_MAX_DURATION_S", "300"))
MAX_SCRIPT_CHARS = 500

# Optional API key. If set, every /v1/* request must send it as
# ``X-API-Key: <key>``. Leave unset only behind a trusted reverse proxy
# that already authenticates (see docs/deploy.md).
API_KEY = _env("TONEROOM_API_KEY", "")

# ASR: faster-whisper model size, e.g. tiny | base | small | medium | large-v3.
# "small" + int8 on CPU is the default: a good balance for a VPS.
# The model is downloaded from Hugging Face on first use (see
# scripts/download_models.sh); it is NOT bundled in this repo.
WHISPER_MODEL = _env("TONEROOM_WHISPER_MODEL", "small")
WHISPER_DEVICE = _env("TONEROOM_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _env("TONEROOM_WHISPER_COMPUTE_TYPE", "int8")

# TTS backend: "sherpa_pocket" (Kyutai Pocket TTS via sherpa-onnx, CPU)
# or "none". The Pocket TTS model files live in POCKET_MODEL_DIR and are
# NOT bundled in this repo (about 190 MB; see scripts/download_models.sh).
TTS_BACKEND = _env("TONEROOM_TTS_BACKEND", "sherpa_pocket")
POCKET_MODEL_DIR = Path(
    _env("TONEROOM_POCKET_MODEL_DIR", str(MODELS_DIR / "pocket-tts-int8-2026-01-26"))
)

# Lip-sync backend: "none" (default) or "musetalk".
# MuseTalk is a separate, GPU-only install (see docs/plan.md Phase 3).
# With "none", respeak jobs return a re-voiced video with the ORIGINAL
# picture track and a timing-aligned audio track; the lips are NOT redrawn.
LIPSYNC_BACKEND = _env("TONEROOM_LIPSYNC_BACKEND", "none")
MUSETALK_DIR = Path(_env("TONEROOM_MUSETALK_DIR", str(MODELS_DIR / "musetalk")))

# How many jobs may run at once. Keep at 1 on a small VPS: the TTS and
# lip-sync models are not safe to run concurrently on one device.
WORKERS = int(_env("TONEROOM_WORKERS", "1"))

# Retention: finished job files older than this (hours) may be deleted by
# an operator. The service itself never deletes a job automatically.
RETENTION_HOURS = int(_env("TONEROOM_RETENTION_HOURS", "168"))
