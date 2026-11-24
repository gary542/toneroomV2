#!/usr/bin/env bash
# Make a small synthetic test clip for the polish test:
# 3 seconds, 640x360 colour bars + a 440 Hz tone as the "voice".
# It has no real speech, so it is ONLY for the polish path and for
# checking upload/status/download plumbing — not for ASR/TTS.
# Usage: bash scripts/make_test_clip.sh [output.mp4]
set -euo pipefail
OUT="${1:-/tmp/toneroom_test_clip.mp4}"
ffmpeg -y -v error \
  -f lavfi -i "smptebars=size=640x360:rate=25:duration=3" \
  -f lavfi -i "sine=frequency=440:duration=3" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT"
echo "wrote $OUT"
