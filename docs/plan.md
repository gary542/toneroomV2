# ToneRoom V2 — phased plan

## Phase 1 — polish + script (this scaffold)

Goal: a working server that a creator can actually use for two jobs,
with the ML plumbing for the third already in place.

Done in this scaffold:
- FastAPI service, job queue, file storage, auth header, consent gate.
- `polish` end-to-end, tested (upload → polished MP4 → download).
- `script` wired to Pocket TTS via the sherpa-onnx CLI, with the
  exact model download command in `scripts/download_models.sh`.
- `respeak` wired end-to-end in code (ASR → TTS → polish → align →
  re-mux), with lip-sync honestly skipped.

Acceptance test (Phase 1):
1. `python -m pytest -v` — all tests pass.
2. Start the server, `curl /health` returns `{"status":"ok"}`.
3. Upload the synthetic clip (`bash scripts/make_test_clip.sh`),
   mode `polish` → download an MP4 that plays, with the picture
   unchanged.
4. After `bash scripts/download_models.sh`: a script job with a real
   4–12 s voice sample + a short script returns a WAV that sounds
   like the owner. **This step has NOT been run yet — it is the first
   thing to do on a machine with the models.**

## Phase 2 — respeak with timing alignment

Goal: the original idea, without lip redraw, good enough to judge
whether the voice + timing are convincing.

Work:
- Run the respeak acceptance test below on a real clip.
- Add a transcript-review step: return the transcript in
  `GET /v1/jobs/{id}` (already stored) and let the creator correct
  it before the TTS step — either as a second endpoint
  (`POST /v1/jobs/{id}/transcript`) or by re-submitting with a
  corrected script. This matters because a broad accent is exactly
  where Whisper `small` will mis-hear words.
- Tune the alignment: per-segment placement using word timestamps
  (the data is already in the transcript), instead of only the
  global `atempo` fit in `app/pipeline/align.py`.
- Try Whisper `medium` on CPU vs `small` and record the accuracy /
  speed trade-off on the same clip.
- Add a stale-job reaper (a job stuck in `running` after a crash)
  and a retention cleanup script.

Acceptance test (Phase 2):
- Test clip: Gary records one talking-head video, 20–30 seconds,
  one or two sentences, in his natural broad accent, facing the
  camera, quiet room, 25 fps, 720p. Plus a separate 4–12 s clean
  voice sample for the reference (or use the video's own audio).
- Run `respeak`. Pass if: (a) the transcript is correct, or only
  needs one or two word corrections; (b) the returned voice is
  recognisably Gary's and clearly easier to understand than the
  original; (c) the audio fits the video length and the words land
  within roughly half a second of the original lip movements for
  most of the clip; (d) the job record says `lipsync: "skipped"`.
- If (b) fails, try the Chatterbox Nano fallback before changing
  anything else (see docs/research.md).

## Phase 3 — lip-sync (MuseTalk, GPU)

Goal: the lips match the new speech, not just the timing.

Work:
- Provision a GPU worker (a GPU VPS, or a separate GPU machine that
  the VPS calls). MuseTalk 1.5 needs PyTorch 2.0.1 + CUDA and its
  weight downloads; it is not part of the CPU image.
- Implement `app/pipeline/lipsync.py` with the MuseTalk command
  already written in that file, behind `LIPSYNC_BACKEND=musetalk`.
  Keep the CPU service as the front door; the GPU worker only does
  lip-sync, so a GPU outage degrades to Phase 2 instead of taking
  the whole service down.
- Evaluate MuseTalk 1.5 vs LatentSync 1.5 on the same Phase 2 clip:
  LatentSync is sharper (512x512) but needs 8 GB VRAM (v1.5) /
  18 GB (v1.6); MuseTalk is lighter and MIT. Pick on the test clip,
  not on paper.

Acceptance test (Phase 3):
- Same test clip as Phase 2. Pass if three people who did not see
  the original watch the lip-synced version and do not notice the
  lips are wrong, and the voice is still recognisably the owner's.
- Record the GPU cost per minute of video. If it is too high for a
  creator-facing price, the Phase 2 (timing-only) version stays as
  the cheap tier.

## Later (not planned in detail yet)

- A simple upload page / Muse connector front end for the API.
- Voice-reference storage with explicit per-creator consent, so a
  returning creator does not re-upload a sample every time.
- Watermarking / labelling of AI-processed output where platforms
  require it.
- Pricing: per-minute processing cost once Phase 2/3 timings are
  measured on the real VPS.
