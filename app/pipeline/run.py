"""Job orchestrator: one function per mode, in plain order.

Modes:
  polish  — ffmpeg polish of the original audio; picture copied.
            Works today, no ML models needed.
  script  — Pocket TTS reads ``job.script`` in the reference voice,
            then polishes the generated audio. Output is a WAV
            (narration take, like V1's script mode) — or an MP4 if a
            video was also uploaded, with the generated audio as its
            soundtrack. Needs the Pocket TTS model (see tts.py).
  respeak — the original ToneRoom idea:
            1. extract audio from the video
            2. transcribe it with faster-whisper (word timestamps)
            3. re-speak the exact transcript with Pocket TTS in the
               owner's voice (reference = uploaded sample, or the
               video's own first 12 seconds of speech if none given)
            4. polish the new audio, then time-align it to the
               original duration (align.py)
            5. lip-sync (Phase 3, MuseTalk) — SKIPPED for now; the
               job records lipsync="skipped" honestly
            6. re-mux the original picture track with the new audio

A job that cannot run a step fails with that step's exact error in
``job.error``. Nothing here ever returns a fake success.
"""

from pathlib import Path

from .. import config
from ..jobs import Job
from . import align, asr, lipsync, polish
from .audio import (
    duration_seconds,
    extract_audio,
    extract_audio_segment,
    has_audio_stream,
    has_video_stream,
    mux_video_audio,
)
from .tts import ModelNotAvailable as TTSNotAvailable
from .tts import generate as tts_generate


def _input_paths(job: Job) -> tuple[Path, Path | None]:
    video = job.dir / "input.mp4"
    reference = job.dir / "reference.wav"
    return video, (reference if reference.exists() else None)


def _reference_for(job: Job, extracted_wav: Path) -> Path:
    """Voice reference: the uploaded sample if given, else the first
    12 seconds of the video's own speech (Pocket TTS reads at most
    ~12 s; V1 used a 4–12 s sample)."""
    _video, uploaded = _input_paths(job)
    if uploaded is not None:
        return uploaded
    ref = job.dir / "reference_from_video.wav"
    extract_audio_segment(extracted_wav, ref, start=0.0, length=12.0)
    return ref


def run_polish(job: Job) -> None:
    video, _ref = _input_paths(job)
    out = job.dir / "output.mp4"
    if has_video_stream(video):
        job.progress = "polishing audio, copying picture track"
        job.save()
        polish.polish_video(video, out, job.finish)
        job.output_file, job.output_kind = "output.mp4", "mp4"
    else:
        job.progress = "polishing audio"
        job.save()
        wav = job.dir / "original.wav"
        extract_audio(video, wav, sample_rate=48000)
        polish.polish_audio(wav, job.dir / "output.wav", job.finish)
        job.output_file, job.output_kind = "output.wav", "wav"


def run_script(job: Job) -> None:
    video, _ref = _input_paths(job)
    reference = _ref
    if reference is None:
        raise ValueError(
            "script mode needs a voice reference: upload a 4–12 s WAV "
            "of your own speech, or use respeak mode (which can take "
            "its reference from the video itself)."
        )
    job.progress = "generating voiceover with Pocket TTS"
    job.save()
    raw = job.dir / "generated_raw.wav"
    tts_generate(job.script, reference, raw, job.dir / "tts_work")
    polished = job.dir / "generated.wav"
    polish.polish_audio(raw, polished, job.finish)
    if has_video_stream(video):
        job.progress = "muxing voiceover onto video"
        job.save()
        mux_video_audio(video, polished, job.dir / "output.mp4")
        job.output_file, job.output_kind = "output.mp4", "mp4"
    else:
        polished.rename(job.dir / "output.wav")
        job.output_file, job.output_kind = "output.wav", "wav"


def run_respeak(job: Job) -> None:
    video, _ref = _input_paths(job)
    if not has_audio_stream(video):
        raise ValueError("the uploaded video has no audio track to transcribe")

    job.progress = "extracting audio"
    job.save()
    original_wav = job.dir / "original.wav"
    extract_audio(video, original_wav, sample_rate=16000)
    original_s = duration_seconds(original_wav)

    job.progress = "transcribing with faster-whisper"
    job.save()
    segments = asr.transcribe(original_wav, language=job.language)
    job.transcript = segments
    job.transcript_text = asr.segments_to_text(segments)
    job.save()
    if not job.transcript_text:
        raise ValueError("no speech was found in the video to re-speak")

    reference = _reference_for(job, original_wav)
    job.progress = "re-speaking in your voice with Pocket TTS"
    job.save()
    raw = job.dir / "generated_raw.wav"
    tts_generate(job.transcript_text, reference, raw, job.dir / "tts_work")

    job.progress = "polishing the new voice"
    job.save()
    polished = job.dir / "generated.wav"
    polish.polish_audio(raw, polished, job.finish)

    job.progress = "aligning new audio to the original timing"
    job.save()
    aligned = job.dir / "generated_aligned.wav"
    info = align.align_to_original(
        polished, original_s, aligned, job.dir / "generated_stretched.wav"
    )
    job.progress = f"timing: {info['note']}"
    job.save()
    if not info["within_limits"]:
        # Honest warning, not a failure: the video is still returned.
        job.error = (
            "Timing warning: the re-spoken audio was more than 15% "
            "different in length from the original, so the fit is "
            "approximate. Lip-sync (Phase 3) will fix this properly."
        )

    if config.LIPSYNC_BACKEND == "musetalk":
        job.progress = "redrawing lips with MuseTalk"
        job.save()
        try:
            lipsync.lipsync_video(
                video, aligned, job.dir / "output.mp4", config.MUSETALK_DIR
            )
            job.lipsync = "done"
        except lipsync.LipSyncNotAvailable as exc:
            job.lipsync = "skipped"
            job.error = str(exc)
            mux_video_audio(video, aligned, job.dir / "output.mp4")
    else:
        job.lipsync = "skipped"
        mux_video_audio(video, aligned, job.dir / "output.mp4")
    job.output_file, job.output_kind = "output.mp4", "mp4"


RUNNERS = {"polish": run_polish, "script": run_script, "respeak": run_respeak}


def run_job(job: Job) -> None:
    """Run one job to completion, recording status/error on the job."""
    job.status = "running"
    job.progress = "starting"
    job.save()
    try:
        RUNNERS[job.mode](job)
    except (TTSNotAvailable, asr.ModelNotAvailable) as exc:
        job.status = "failed"
        job.error = str(exc)
    except Exception as exc:  # keep the exact error on the job record
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    else:
        job.status = "done"
        job.progress = "done"
    job.save()
