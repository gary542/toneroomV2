"""Job records for ToneRoom V2.

Jobs are plain JSON files under ``data/jobs/<job_id>/meta.json`` plus the
uploaded and generated files in the same directory. No database is needed
at this scale, and a job can be inspected (or deleted) with ordinary files.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

MODES = ("polish", "script", "respeak")
FINISHES = ("balanced", "warm", "clear")
STATUSES = ("queued", "running", "done", "failed")


@dataclass
class Job:
    id: str
    mode: str
    status: str = "queued"
    finish: str = "balanced"
    script: str = ""
    language: str = "en"
    consent: bool = False
    error: str = ""
    progress: str = ""
    transcript: list = field(default_factory=list)  # [{start, end, text}]
    transcript_text: str = ""
    output_file: str = ""  # filename inside the job dir, e.g. output.mp4
    output_kind: str = ""  # "mp4" | "wav"
    lipsync: str = "not_requested"  # not_requested | skipped | done | failed
    created_at: str = ""
    updated_at: str = ""

    @property
    def dir(self) -> Path:
        return config.JOBS_DIR / self.id

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("dir", None)
        return d

    def save(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = self.updated_at
        self.dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: a concurrent GET must never see a half-written
        # meta.json (the worker saves while clients poll).
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(self.dir / "meta.json")

    @classmethod
    def load(cls, job_id: str) -> Optional["Job"]:
        path = config.JOBS_DIR / job_id / "meta.json"
        if not path.exists():
            return None
        # Tolerate a torn read from a writer that predates the atomic
        # save above (e.g. a job written by an older build).
        last: Exception | None = None
        for _ in range(10):
            try:
                data = json.loads(path.read_text())
                break
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                last = exc
                import time as _time

                _time.sleep(0.1)
        else:
            raise last  # type: ignore[misc]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def new_job(mode: str, finish: str, script: str, language: str, consent: bool) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12],
        mode=mode,
        finish=finish,
        script=script,
        language=language,
        consent=consent,
    )
    job.save()
    return job
