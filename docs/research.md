# ToneRoom V2 — pipeline research (24 Nov 2026)

Question: for a talking-head video in a broad accent, what open-source
pipeline transcribes it, re-speaks the exact words in the owner's voice
with the accent softened, and returns a video whose lips match?

Only URLs listed here were actually opened on 24 Nov 2026. Anything
else is marked as second-hand and must be re-checked before relying
on it.

## 1. Transcription (ASR)

### Chosen: faster-whisper — SYSTRAN, MIT
https://github.com/SYSTRAN/faster-whisper

- Whisper reimplemented on CTranslate2: "up to 4 times faster than
  openai/whisper for the same accuracy while using less memory", with
  8-bit quantisation on CPU and GPU.
- README benchmark, 13 minutes of audio: `small` int8 on CPU (8-thread
  i7-12700K) takes 1m42s using ~1.5 GB RAM; `large-v2` fp16 on an
  RTX 3070 Ti takes 1m03s.
- Word-level timestamps (`word_timestamps=True`) — needed for Phase 2
  timing alignment — and a Silero VAD filter are built in.
- CPU-first: `WhisperModel("small", device="cpu", compute_type="int8")`.
  GPU needs cuBLAS + cuDNN 9 for CUDA 12.
- Whisper weights themselves are MIT (OpenAI).
- Trade-off: accuracy on a very broad accent is the weakest link in
  the whole chain. `small` will mis-hear some words; `medium`/`large-v3`
  are better but slower on CPU. Mitigation: show the transcript back
  to the creator for correction before re-speaking (a Phase 2 UI job),
  and default the job to fail loudly on an empty transcript rather
  than guessing.

### Alternative considered: openai-whisper (reference implementation)
Same models, slower on CPU, needs system ffmpeg. No advantage over
faster-whisper for a server. Not chosen.

## 2. Voice cloning / reference-conditioned TTS

The job needs zero-shot cloning from a short reference, in English,
on a CPU VPS, with a licence that allows a creator-facing service.

### Chosen: Kyutai Pocket TTS via sherpa-onnx — code MIT, weights CC BY 4.0
- https://huggingface.co/kyutai/pocket-tts
- https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
- https://github.com/k2-fsa/sherpa-onnx

- ~100M parameters, designed for CPU: "~6x real-time on a CPU of
  MacBook Air M4", "~200ms to get the first audio chunk", using only
  2 CPU cores (Kyutai model card, opened).
- Zero-shot cloning from a short reference clip; **no transcript of
  the reference is needed** — only the audio file (sherpa docs).
- The sherpa-onnx int8 ONNX conversion (January 2026) is the same
  model V1 bundled on the phone, so the voice behaviour is already
  known from V1: `sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2`.
- The sherpa docs give the exact CLI used by V2:
  `sherpa-onnx-offline-tts --pocket-lm-flow=... --pocket-lm-main=...
  --pocket-encoder=... --pocket-decoder=... --pocket-text-conditioner=...
  --pocket-vocab-json=... --pocket-token-scores-json=...
  --reference-audio=... --num-steps=2 --output-filename=... "text"`.
- Licence: code MIT, weights CC BY 4.0, and Kyutai's terms require the
  recorded person's consent — which V2 enforces per job.
- Trade-off: English is its home ground; it also lists French, German,
  Portuguese, Italian and Spanish. It re-speaks text — it does not
  "soften" an existing recording in place. That is why respeak is
  transcribe-then-re-speak, and why lip-sync is a separate step.
  Quality on a strong accent reference is unproven until the Phase 2
  acceptance test runs on a real clip.

### Alternative: Chatterbox (Resemble AI) — MIT, but retired
https://github.com/resemble-ai/chatterbox

- MIT licence, zero-shot cloning from ~5–10 s, Turbo 350M and Nano
  110M (Nano: "3x faster than realtime on 8 CPU cores"). Attractive
  on paper, and it watermarks outputs (PerTh).
- **Not chosen: the README (opened) states "As of August 2026,
  Chatterbox is retired research and is no longer maintained."**
  Building a new service on an unmaintained model is the wrong call.
  Keep it as the fallback if Pocket TTS fails the acceptance test.

### Alternatives not chosen
- **Coqui XTTS-v2**: good cloning, but its licence forbids commercial
  use (reported in the research; re-check the model card before ever
  using it). Not chosen.
- **F5-TTS / E2-TTS**: strong quality, but the published checkpoints
  are CC BY-NC (non-commercial) in the comparisons read. Not chosen
  for a creator-facing service.

## 3. Lip-sync / re-timing

### Fallback that ships in Phase 2: time-stretch + silence alignment
- ffmpeg `atempo` (0.85–1.15x, clamped) plus per-segment silence
  padding, driven by faster-whisper word timestamps. CPU, free, no
  model. It fits the new audio to the original timeline, but it does
  **not** redraw lips — on a long or pause-heavy clip the mouth will
  visibly drift. That is an honest limitation, recorded on the job.

### Chosen for Phase 3: MuseTalk 1.5 — MIT
https://github.com/TMElyralab/MuseTalk

- Tencent Music's audio-driven lip-sync: edits the 256x256 face region
  of an existing video in latent space, 30fps+ on an NVIDIA Tesla V100,
  supports English (and Chinese/Japanese) audio.
- Licence: MIT, and the project states its trained models are
  available "for any purpose, even commercially" (per the licence
  comparisons read; the repo itself was opened and is MIT).
- Needs PyTorch 2.0.1 + CUDA 11.7/11.8 and its own weight downloads
  (musetalk, sd-vae, whisper-tiny, dwpose, syncnet, face-parse).
  GPU-only in practice — this is the step that forces a GPU host or a
  separate GPU worker, not the Hostinger CPU VPS.
- Trade-off: 256x256 face region, so very high-resolution close-ups
  lose some sharpness; face-centre tuning affects results.

### Alternative: LatentSync 1.6 — Apache-2.0 code, OpenRAIL++ weights
https://github.com/bytedance/LatentSync

- ByteDance latent-diffusion lip-sync, sharper (512x512 in v1.6) and
  generally the best-looking open option.
- README (opened): minimum VRAM **8 GB for v1.5, 18 GB for v1.6**.
  That is more than a small GPU VPS offers, so it is the Phase 3
  *upgrade* candidate after MuseTalk proves the pipeline, not the
  first lip-sync backend.
- Weights are tagged OpenRAIL++ — usable with restrictions; read the
  weight licence on Hugging Face before commercial use.

### Rejected: Wav2Lip — non-commercial
https://github.com/Rudrabha/Wav2Lip

- The most-recommended model in tutorials, and accurate — but its
  README (opened) states: "All results from this open-source code …
  should only be used for research/academic/personal purposes only.
  As the models are trained on the LRS2 dataset, any form of
  commercial use is strictly prohibited."
- Do NOT use it for a creator-facing service. Its commercial version
  is a paid hosted API (Sync Labs) instead.

## Summary — the chosen pipeline

```
video ──► ffmpeg extract audio ──► faster-whisper "small" int8 (CPU)
              │                          │ word timestamps
              │                          ▼
              │                   transcript (exact words)
              │                          │
              │                          ▼
              ├─► Pocket TTS via sherpa-onnx (CPU), reference = owner's
              │   4–12 s sample (or first 12 s of the video's own audio)
              │                          │
              │                          ▼
              │                   new speech, owner's voice
              │                          │
              │                          ▼
              │                   ffmpeg polish (V1 finish) + atempo
              │                   alignment to original duration
              │                          │
              └─► (Phase 3) MuseTalk 1.5 (GPU) redraws lips
                                          │
                                          ▼
                                  MP4: original picture
                                       + new audio
```

CPU-first for everything except lip-sync. The one step that genuinely
needs a GPU is Phase 3, and it is isolated behind the
`LIPSYNC_BACKEND` setting so the rest of the service runs on the VPS.
