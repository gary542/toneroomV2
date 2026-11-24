# ToneRoom V2 — architecture

## Pipeline (text diagram)

```
                    ┌────────────────────────────┐
  POST /v1/jobs     │  FastAPI (app/main.py)     │
  video + reference │  validate: mode, consent,  │
  + mode + script   │  size, duration, script    │
                    └──────────────┬─────────────┘
                                   │ enqueue
                    ┌──────────────▼─────────────┐
                    │  Worker (app/worker.py)    │
                    │  1 thread, FIFO queue      │
                    └──────────────┬─────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        polish                script               respeak
   app/pipeline/          app/pipeline/        app/pipeline/run.py
   polish.py              tts.py + polish.py   asr.py → tts.py
   (ffmpeg only)          (Pocket TTS)         → polish.py
                                                → align.py
                                                → lipsync.py (stub)
              └────────────────────┬────────────────────┘
                                   ▼
                    data/jobs/<id>/output.mp4 | output.wav
                    data/jobs/<id>/meta.json  (status, transcript,
                                               error, lipsync)
                                   │
                    GET /v1/jobs/{id} / /download
```

## API spec

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok"}` |
| POST | `/v1/jobs` | multipart: `video` (required), `reference` (optional file), `mode` (`polish`\|`script`\|`respeak`, default `respeak`), `finish` (`balanced`\|`warm`\|`clear`), `script` (1–500 chars, required for `script`), `language` (default `en`), `consent` (must be `true`) | 202 `{"job_id","mode","status":"queued"}` |
| GET | `/v1/jobs/{id}` | — | job JSON: `id, mode, status, finish, script, language, consent, error, progress, transcript, transcript_text, output_file, output_kind, lipsync, created_at, updated_at` |
| GET | `/v1/jobs/{id}/download` | — | the finished MP4/WAV, or 409 while the job is not `done` |

Statuses: `queued → running → done | failed`. A `failed` job always
carries `error` with the exact reason. A `done` respeak job whose
timing fit was approximate carries a warning in `error` too, with
`status` still `done` — the warning is not hidden.

Auth: if `TONEROOM_API_KEY` is set, all `/v1/*` calls need header
`X-API-Key: <key>` (401 otherwise). `/health` is unauthenticated by
design, for load-balancer / uptime checks.

## Data flow and storage

- One directory per job: `data/jobs/<12-hex-id>/`.
  `input.mp4` (upload), `reference.wav` (normalised reference),
  `original.wav` (16 kHz extract), `generated_raw.wav`,
  `generated.wav`, `generated_aligned.wav`, `output.mp4`/`output.wav`,
  `meta.json` (the job record). No database.
- `data/` and `models/` are gitignored. Model files live in
  `models/` (see `scripts/download_models.sh`) and are never in git.
- The worker is a single in-process thread (`app/worker.py`), so the
  TTS/ASR models never run concurrently on one device. There is no
  Redis/Celery to operate on a small VPS.
- Limits: 200 MB upload, 300 s duration (V1's take cap), 500 script
  characters. All enforced in `app/main.py` before the job is queued.

## Failure modes

| Failure | Behaviour |
|---|---|
| No consent | 400, job never created. |
| Bad mode / finish / script length | 400 with the rule broken. |
| Oversize / over-long video | 413 / 400, job marked failed. |
| Video with no audio (respeak) | Job fails: "no audio track". |
| faster-whisper not installed | Job fails naming faster-whisper + install command. |
| Pocket TTS model missing | Job fails naming the missing files + download command. |
| Transcript empty (no speech found) | Job fails, no fake audio is produced. |
| TTS chunk fails | Job fails with the sherpa stderr tail. |
| Timing mismatch >15% | Job succeeds with a warning in `error`; `lipsync: "skipped"`. |
| Lip-sync requested but not wired | `lipsync: "skipped"`, exact MuseTalk command in `error`. |
| Worker crashes mid-job | Job stays `running` in `meta.json`; restart the service and re-check — a future version should add a stale-job reaper (noted in plan.md). |

## What is NOT in this scaffold

- No user accounts, no billing, no public upload page. The API is
  the product surface; a front end is a later decision.
- No automatic deletion (retention is an operator task; see
  `docs/deploy.md`).
- No GPU support in the shipped image. Lip-sync is a separate GPU
  worker by design (Phase 3).
