"""ToneRoom V2 — FastAPI service.

Endpoints:
  GET  /health                 -> {"status": "ok"}
  POST /v1/jobs                -> upload a video, start a job (202)
  GET  /v1/jobs/{job_id}       -> job status, transcript, errors
  GET  /v1/jobs/{job_id}/download -> finished MP4/WAV (409 until done)

Upload rules (enforced here, not just documented):
  * ``consent`` must be sent as ``true``. It means: "this is my own
    voice, or I have the speaker's explicit permission, and I agree
    the video/voice may be processed on this server and deleted after
    the retention period." Without it the job is refused (400).
  * Video: mp4/mov/webm/mkv, at most TONEROOM_MAX_UPLOAD_MB (default
    200 MB) and TONEROOM_MAX_DURATION_S (default 300 s, V1's cap).
  * ``script`` mode: ``script`` text is required, 1–500 characters,
    and a voice reference (``reference`` file, WAV/MP3) is required.
  * ``respeak`` mode: a reference is optional — the video's own audio
    is used as the reference when none is supplied.
  * ``polish`` mode: no reference, no script needed.

Auth: if TONEROOM_API_KEY is set, /v1/* requires ``X-API-Key``.
Do not expose this service to the internet without it (or without
an authenticating reverse proxy) — see docs/deploy.md.
"""

import shutil
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import config
from .jobs import MODES, FINISHES, Job
from .pipeline.audio import duration_seconds
from .worker import start_worker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    start_worker()
    yield


app = FastAPI(title="ToneRoom V2", version="0.1.0", lifespan=lifespan)

ALLOWED_VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
ALLOWED_REF = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def _check_auth(api_key: str | None) -> None:
    if config.API_KEY and api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    mode: str = Form("respeak"),
    finish: str = Form("balanced"),
    script: str = Form(""),
    language: str = Form("en"),
    consent: bool = Form(False),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> dict:
    _check_auth(x_api_key)
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(MODES)}")
    if finish not in FINISHES:
        raise HTTPException(400, f"finish must be one of {', '.join(FINISHES)}")
    if not consent:
        raise HTTPException(
            400,
            "consent is required: confirm this is your own voice, or you "
            "have the speaker's explicit permission.",
        )
    if script and len(script) > config.MAX_SCRIPT_CHARS:
        raise HTTPException(400, f"script must be at most {config.MAX_SCRIPT_CHARS} characters")
    if mode == "script" and not script.strip():
        raise HTTPException(400, "script mode needs a script (1-500 characters)")
    if mode == "script" and reference is None:
        raise HTTPException(400, "script mode needs a voice reference file")

    video_suffix = Path(video.filename or "input").suffix.lower()
    if video_suffix not in ALLOWED_VIDEO:
        raise HTTPException(400, f"video must be one of {sorted(ALLOWED_VIDEO)}")

    # Create the job record first, but only enqueue it AFTER the upload
    # has been validated — otherwise the worker could start on a file
    # that is still being written or is about to be rejected.
    from .jobs import new_job
    from .worker import enqueue

    job = new_job(mode, finish, script.strip(), language, consent)
    try:
        dest = job.dir / "input.mp4"
        size = 0
        with dest.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        413, f"video exceeds {config.MAX_UPLOAD_MB} MB limit")
                out.write(chunk)
        if duration_seconds(dest) > config.MAX_DURATION_S:
            raise HTTPException(
                400, f"video exceeds {config.MAX_DURATION_S} seconds")
        if reference is not None:
            ref_suffix = Path(reference.filename or "ref").suffix.lower()
            if ref_suffix not in ALLOWED_REF:
                raise HTTPException(400, f"reference must be one of {sorted(ALLOWED_REF)}")
            ref_dest = job.dir / f"reference{ref_suffix}"
            with ref_dest.open("wb") as out:
                shutil.copyfileobj(reference.file, out)
            # Normalise the reference to the WAV the TTS step expects.
            # Extract via a temp file: if the upload itself was a WAV,
            # the source is already named reference.wav, and ffmpeg
            # refuses to write its input file onto itself (fixed
            # 24 Nov 2026, Phase 2 test).
            from .pipeline.audio import extract_audio
            _tmp_ref = job.dir / "reference_tmp.wav"
            extract_audio(ref_dest, _tmp_ref, sample_rate=16000)
            _tmp_ref.replace(job.dir / "reference.wav")
    except HTTPException:
        job.status = "failed"
        job.error = "upload rejected (see response)"
        job.save()
        raise
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.save()
        raise HTTPException(400, str(exc)) from exc

    job.progress = "queued"
    job.save()
    enqueue(job)
    return {"job_id": job.id, "mode": job.mode, "status": job.status}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(None, alias="X-API-Key")) -> dict:
    _check_auth(x_api_key)
    job = Job.load(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@app.get("/v1/jobs/{job_id}/download")
def download_job(job_id: str, x_api_key: str | None = Header(None, alias="X-API-Key")):
    _check_auth(x_api_key)
    job = Job.load(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done" or not job.output_file:
        raise HTTPException(409, f"job is {job.status}; nothing to download yet")
    path = job.dir / job.output_file
    if not path.exists():
        raise HTTPException(404, "output file is missing")
    media = "video/mp4" if job.output_kind == "mp4" else "audio/wav"
    return FileResponse(path, media_type=media, filename=job.output_file)
