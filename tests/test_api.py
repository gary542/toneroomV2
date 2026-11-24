"""ToneRoom V2 API tests.

Run:  python -m pytest -v
Requires ffmpeg on PATH (the polish path shells out to it).
The ML models are NOT required: the respeak test below asserts that a
job needing them fails with a clear, honest error — not a fake success.
"""

import time

POLL_TIMEOUT_S = 60


def _wait_for(client, job_id, want=("done", "failed")):
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        if r.json()["status"] in want:
            return r.json()
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not finish in {POLL_TIMEOUT_S}s")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_create_job_requires_consent(client, test_clip):
    with test_clip.open("rb") as f:
        r = client.post(
            "/v1/jobs",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"mode": "polish", "consent": "false"},
        )
    assert r.status_code == 400
    assert "consent" in r.text.lower()


def test_create_job_bad_mode(client, test_clip):
    with test_clip.open("rb") as f:
        r = client.post(
            "/v1/jobs",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"mode": "teleport", "consent": "true"},
        )
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/v1/jobs/doesnotexist").status_code == 404
    assert client.get("/v1/jobs/doesnotexist/download").status_code == 404


def test_polish_end_to_end(client, test_clip):
    """The one path that must actually work with no ML models:
    upload -> polish -> download an MP4."""
    with test_clip.open("rb") as f:
        r = client.post(
            "/v1/jobs",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"mode": "polish", "finish": "balanced", "consent": "true"},
        )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = _wait_for(client, job_id)
    assert job["status"] == "done", job
    assert job["output_kind"] == "mp4"

    dl = client.get(f"/v1/jobs/{job_id}/download")
    assert dl.status_code == 200, dl.text
    assert dl.headers["content-type"] == "video/mp4"
    assert len(dl.content) > 1000
    # MP4 magic: 'ftyp' box at offset 4.
    assert dl.content[4:8] == b"ftyp"


def test_respeak_without_models_fails_honestly(client, test_clip):
    """Without faster-whisper / Pocket TTS installed, a respeak job must
    fail with a message saying what is missing — never a fake output."""
    with test_clip.open("rb") as f:
        r = client.post(
            "/v1/jobs",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"mode": "respeak", "consent": "true"},
        )
    assert r.status_code == 202, r.text
    job = _wait_for(client, r.json()["job_id"])
    assert job["status"] == "failed", job
    assert job["error"], job
    assert "faster-whisper" in job["error"] or "Pocket TTS" in job["error"]
