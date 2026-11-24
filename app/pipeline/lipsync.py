"""Lip-sync (Phase 3) — NOT IMPLEMENTED yet. Read this before wiring it.

Chosen backend (see docs/research.md): **MuseTalk 1.5**
(https://github.com/TMElyralab/MuseTalk), MIT licence, models stated
as available "for any purpose, even commercially". It edits the 256x256
face region of an existing video to match new audio, at 30fps+ on an
NVIDIA Tesla V100. It is GPU-only in practice and needs PyTorch 2.0.1
+ CUDA, plus its own weight downloads.

Why not Wav2Lip, the model most tutorials recommend first?
Its README states the open-source weights are for research/academic/
personal purposes only and "any form of commercial use is strictly
prohibited" (trained on LRS2). See https://github.com/Rudrabha/Wav2Lip.
Do NOT wire Wav2Lip into V2 for creator-facing use.

Why not LatentSync as the first lip-sync backend?
ByteDance LatentSync 1.6 (https://github.com/bytedance/LatentSync,
Apache-2.0 code, weights OpenRAIL++) is sharper (512x512) but needs
18 GB VRAM for v1.6 (8 GB for v1.5) — more than a typical VPS GPU,
if the VPS has one at all. It is the Phase 3 upgrade candidate after
MuseTalk proves the pipeline.

What this module does today: it raises ``NotImplementedError`` with the
exact command that will fill it in, so a respeak job can never silently
pretend its lips were redrawn. The orchestrator catches that, records
``lipsync: "skipped"`` on the job, and returns the timing-aligned video.
"""

from pathlib import Path


class LipSyncNotAvailable(NotImplementedError):
    pass


MUSETALK_COMMAND = """\
# Phase 3 — MuseTalk 1.5 (GPU only). Install once, outside this repo:
#   git clone https://github.com/TMElyralab/MuseTalk <models>/musetalk
#   cd <models>/musetalk && sh download_weights.sh   # needs a GPU host
# Then, per job (video + new audio -> lip-synced video):
#   python -m scripts.inference \\
#     --inference_config configs/inference/test.yaml \\
#     --result_dir <job_dir>/lipsync \\
#     --unet_model_path models/musetalkV15/unet.pth \\
#     --unet_config models/musetalkV15/musetalk.json \\
#     --version v15
# with video_path=<original video> and audio_path=<new audio> set in
# the inference config. See the MuseTalk README for the full setup.
"""


def lipsync_video(video: Path, audio_wav: Path, dest_mp4: Path,
                  musetalk_dir: Path) -> Path:
    """Redraw the lips of ``video`` to match ``audio_wav``.

    Not implemented in this scaffold. Raises LipSyncNotAvailable with
    the exact MuseTalk command to fill in, so callers can record the
    skip honestly instead of faking a result.
    """
    raise LipSyncNotAvailable(
        "Lip-sync is not wired in yet (Phase 3). MuseTalk 1.5 is the "
        "chosen backend and needs a GPU host. Command to fill in:\n"
        + MUSETALK_COMMAND
    )
