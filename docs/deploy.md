# ToneRoom V2 — deploy notes (Hostinger VPS, Ubuntu)

Target: the same VPS as Reclaim (Ubuntu, `reclaim.agentengines.co.uk`
is the Reclaim host). ToneRoom V2 should get its own subdomain, e.g.
`toneroom.agentengines.co.uk`, and its own port — do not share
Reclaim's service or database.

## 1. On the VPS

```bash
sudo apt update && sudo apt install -y python3.12-venv ffmpeg git curl
git clone <toneroomV2 repo URL> /opt/toneroomV2   # repo to be created
cd /opt/toneroomV2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env: set TONEROOM_API_KEY to a long random value.
# Do NOT leave it empty on a public host.
```

For the ML modes, on the same host:

```bash
.venv/bin/pip install -r requirements-ml.txt
bash scripts/download_models.sh   # ~190 MB Pocket TTS + Whisper model
```

## 2. systemd unit — `/etc/systemd/system/toneroom.service`

```ini
[Unit]
Description=ToneRoom V2
After=network.target

[Service]
User=toneroom
WorkingDirectory=/opt/toneroomV2
EnvironmentFile=/opt/toneroomV2/.env
ExecStart=/opt/toneroomV2/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now toneroom
curl http://127.0.0.1:8001/health   # {"status":"ok"}
```

Port 8001 is deliberate: Reclaim uses 3000 behind its own proxy, and
ToneRoom must not collide with it. The service only listens on
localhost; nginx is the only public entry point.

## 3. nginx — reverse proxy with HTTPS

```nginx
server {
    listen 443 ssl;
    server_name toneroom.agentengines.co.uk;
    # ssl_certificate / ssl_certificate_key from certbot (see below)

    client_max_body_size 250m;   # uploads up to the 200 MB job limit

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;   # jobs are slow; polling, not streaming
    }
}
```

Then, as with Reclaim: an A record for the subdomain pointing at the
VPS IP, then `certbot --nginx -d toneroom.agentengines.co.uk`.

## 4. Firewall and retention

- Open 22/80/443 only. Port 8001 stays localhost-only.
- Retention: delete job directories older than
  `TONEROOM_RETENTION_HOURS` (default 168 h / 7 days), e.g. a weekly
  cron: `find /opt/toneroomV2/data/jobs -maxdepth 1 -type d -mtime +7
  -exec rm -rf {} +` — run it only after confirming no job is still
  `running` (check `meta.json` first in a first pass; automate the
  check before automating the delete).

## 5. Docker alternative

```bash
cp .env.example .env   # set TONEROOM_API_KEY
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

The compose file binds to 127.0.0.1 only; put nginx in front of it
as above. The image is CPU-only; Phase 3 (MuseTalk) needs a separate
GPU host — do not try to run it in this container.

## 6. Smoke test after deploy

```bash
bash scripts/make_test_clip.sh /tmp/clip.mp4
curl -X POST https://toneroom.agentengines.co.uk/v1/jobs \
  -H "X-API-Key: $TONEROOM_API_KEY" \
  -F video=@/tmp/clip.mp4 -F mode=polish -F consent=true
# poll GET /v1/jobs/<id> until done, then download it
```
