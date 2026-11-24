#!/usr/bin/env bash
# ToneRoom V2 — download the ML model files (NOT bundled in the repo).
# Run once on the machine that will do the processing:
#   bash scripts/download_models.sh
# Licences: Pocket TTS weights CC BY 4.0 (consent condition, see README);
# Whisper weights MIT. Both downloads are a few hundred MB in total.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
cd models

# 1. Kyutai Pocket TTS, sherpa-onnx int8 ONNX conversion, January 2026.
#    Same files V1 bundled on the phone (~190 MB extracted).
#    Source: https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
if [ ! -d pocket-tts-int8-2026-01-26 ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  # --no-same-owner: the archive is owned by uid 501, which cannot be
  # restored on this machine; without it tar exits non-zero even though
  # the files extract fine (fixed 24 Nov 2026, Phase 2 test).
  tar --no-same-owner -xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  # The archive extracts to sherpa-onnx-pocket-tts-int8-2026-01-26/,
  # but config.py / .env.example expect pocket-tts-int8-2026-01-26/
  # — rename it (fixed 24 Nov 2026, Phase 2 test).
  if [ -d sherpa-onnx-pocket-tts-int8-2026-01-26 ]; then
    mv sherpa-onnx-pocket-tts-int8-2026-01-26 pocket-tts-int8-2026-01-26
  fi
  echo "Pocket TTS model ready in models/pocket-tts-int8-2026-01-26"
else
  echo "Pocket TTS model already present, skipping."
fi

# 2. faster-whisper model ("small" by default; override with
#    TONEROOM_WHISPER_MODEL=base|medium|large-v3 before running).
#    Downloads from Hugging Face into the faster-whisper cache.
#    Use the venv python when present: system python3 does not have
#    faster-whisper installed (fixed 24 Nov 2026, Phase 2 test).
MODEL="${TONEROOM_WHISPER_MODEL:-small}"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -c "
from faster_whisper import WhisperModel
WhisperModel('$MODEL', device='cpu', compute_type='int8')
print('faster-whisper model $MODEL ready')
"
echo "All models downloaded."
