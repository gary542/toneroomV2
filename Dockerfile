# ToneRoom V2 — CPU image (Phases 1–2). Lip-sync (Phase 3, MuseTalk)
# needs a separate GPU image; see docs/plan.md.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# ML deps are installed at deploy time, not build time, so the image
# stays small and a CPU-only host is not forced to pull torch:
#   docker compose run --rm toneroom pip install -r requirements-ml.txt
#   (or build with --build-arg INSTALL_ML=1; see docker-compose.yml)

COPY app ./app
COPY scripts ./scripts

ENV TONEROOM_DATA_DIR=/app/data
ENV TONEROOM_MODELS_DIR=/app/models
VOLUME ["/app/data", "/app/models"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
