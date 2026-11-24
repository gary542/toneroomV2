# ToneRoom V2

A server version of ToneRoom, built from scratch for the original idea:

> A creator records a talking-head video with a broad accent that
> people outside their area struggle to understand. ToneRoom V2 takes
> that video, transcribes what was said, re-speaks the exact words in
> the owner's own voice with the accent softened and clearer, and
> returns a video where the lips still match the new speech.

## How V2 differs from V1

V1 (`gary542/toneroom-android`, **not modified, no code copied**) is an
on-device Android app that does two *separate* things:

1. **Polish** a recording — EQ/compression, keeps the original words,
   timing, accent and lip-sync.
2. **Script mode** — a 4–12 s voice sample + up to 500 characters of
   text, read aloud in the owner's voice by Kyutai Pocket TTS. A new
   narration take, not lip-synced to any video.

V1 never did those two things in one step. V2 is a new server that
does, in three modes:

| Mode | What it does | Needs ML? |
|---|---|---|
| `polish` | ffmpeg polish of the original audio; picture copied untouched. Lip-sync exact, accent unchanged. | No — works today |
| `script` | Pocket TTS reads your script in your voice; polished; returned as WAV, or as an MP4 soundtrack. | Yes — Pocket TTS |
| `respeak` | **The original idea.** Transcribe the video → re-speak the exact words in your voice → polish → time-align to the original duration → re-mux onto the original picture. Lips are redrawn only in Phase 3 (MuseTalk, GPU). | Yes — faster-whisper + Pocket TTS |

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # set TONEROOM_API_KEY before exposing it
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health        # {"status":"ok"}
```

For the ML modes, also:

```bash
.venv/bin/pip install -r requirements-ml.txt
bash scripts/download_models.sh        # ~190 MB Pocket TTS + Whisper model
```

## API

```bash
# Start a polish job (consent is mandatory — see Privacy below)
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F video=@clip.mp4 -F mode=polish -F finish=balanced -F consent=true
# {"job_id": "abc123...", "mode": "polish", "status": "queued"}

curl http://127.0.0.1:8000/v1/jobs/abc123...
curl -o out.mp4 http://127.0.0.1:8000/v1/jobs/abc123.../download
```

Full spec, data flow and failure modes: `docs/architecture.md`.
Research behind the pipeline choices: `docs/research.md`.
Phased plan and acceptance tests: `docs/plan.md`.
VPS deployment (systemd + nginx): `docs/deploy.md`.

## Running the tests

```bash
.venv/bin/python -m pytest -v
```

Tests need ffmpeg on PATH. They cover health, job creation, consent
enforcement, a real polish end-to-end (upload → download an MP4), and
the honest-failure case: a respeak job without the ML models installed
must fail with a clear message, never a fake output.

## Privacy and consent

A voice sample is personal data, and a cloned voice can be misused.
V2 only ever processes the **owner's own voice** (or a voice whose
speaker has given explicit permission), and every job requires
`consent=true` at the API layer. Uploads, references and outputs live
under `data/jobs/<id>/`, are gitignored, and are never sent anywhere
else — all processing is local to the server. Operators should delete
job files after the retention period (`TONEROOM_RETENTION_HOURS`,
default 7 days) and must not keep a creator's voice reference after
the job is done unless the creator has agreed to that in writing.
Generated audio from Chatterbox-style watermarking is not used here;
Pocket TTS outputs carry no watermark, so outputs should be labelled
as AI-processed where a platform requires it.

## Licences and attribution

V2's own code is new, written from scratch for this repo. The models
and libraries it uses keep their own licences — read them before
shipping anything commercially:

- **faster-whisper** (ASR) — MIT.
  https://github.com/SYSTRAN/faster-whisper
- **Whisper model weights** (OpenAI) — MIT.
- **Kyutai Pocket TTS** (voice cloning) — code MIT, weights
  **CC BY 4.0**, with Kyutai's condition that the voice is used only
  with the recorded person's consent.
  https://huggingface.co/kyutai/pocket-tts
- **sherpa-onnx** (Pocket TTS runtime) — Apache-2.0.
  https://github.com/k2-fsa/sherpa-onnx
- **MuseTalk 1.5** (Phase 3 lip-sync) — MIT, models stated as
  available "for any purpose, even commercially".
  https://github.com/TMElyralab/MuseTalk
- **ffmpeg** — LGPL/GPL depending on build; the Ubuntu package used
  here is the standard distro build.
- **Wav2Lip is deliberately NOT used**: its README states the
  open-source weights are for research/academic/personal purposes
  only and "any form of commercial use is strictly prohibited".
  https://github.com/Rudrabha/Wav2Lip

## Status

Scaffold, 24 Nov 2026. The `polish` path is real and tested. The
`script` and `respeak` paths are fully wired (faster-whisper +
Pocket TTS via the sherpa-onnx CLI) but **not yet verified end-to-end**,
because the model files were not downloaded in the build workspace.
Lip-sync is an honest, clearly-marked stub until Phase 3. See
`docs/plan.md` for the acceptance test for each phase.
