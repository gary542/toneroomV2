"""Test setup: point the app at a temp data dir, and build a test clip."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Make the repo root importable (tests run from the repo root or anywhere).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMP = Path(tempfile.mkdtemp(prefix="toneroom_test_"))
os.environ["TONEROOM_DATA_DIR"] = str(TMP / "data")
os.environ["TONEROOM_API_KEY"] = ""  # no auth in tests


@pytest.fixture(scope="session")
def test_clip(tmp_path_factory):
    """A 3-second synthetic MP4 (colour bars + 440 Hz tone). No speech."""
    out = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(
        ["bash", "scripts/make_test_clip.sh", str(out)],
        check=True, capture_output=True, cwd=Path(__file__).resolve().parent.parent,
    )
    return out


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
