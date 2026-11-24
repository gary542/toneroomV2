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
  tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
  echo "Pocket TTS model ready in models/pocket-tts-int8-2026-01-26"
else
  echo "Pocket TTS model already present, skipping."
fi

# 2. faster-whisper model ("small" by default; override with
#    TONEROOM_WHISPER_MODEL=base|medium|large-v3 before running).
#    Downloads from Hugging Face into the faster-whisper cache.
MODEL="${TONEROOM_WHISPER_MODEL:-small}"
python3 -c "
from faster_whisper import WhisperModel
WhisperModel('$MODEL', device='cpu', compute_type='int8')
print('faster-whisper model $MODEL ready')
"
echo "All models downloaded."
