# YuE2 on Apple Silicon (Mac) — Install Notes

Setup and working configuration for running **YuE2** (`m-a-p/YuE2-3B`) on an
Apple M3 Max Mac with MPS acceleration. End-to-end music generation is
**working** on this machine — see [Verified results](#verified-results).

## Summary

| | |
|---|---|
| Works on MPS? | **Yes** (with one CPU fallback for VAE decode, see [MPS patch](#mps-patch)) |
| Repo | `<repo>` — main branch, commit `88da114`, YuE2 v0.1.6 |
| Python env | `.venv` in the repo dir, built from `/opt/homebrew/bin/python3.12` |
| Key packages | `torch 2.10.0` (MPS build), `transformers 4.57.6`, `yue2-infer` (installed via `pip install .`) |
| Models | HF cache: `m-a-p/YuE2-3B` (6.76 GiB) and `m-a-p/YuE2-Vae` (0.49 GiB), sha256-verified |
| Test output | `runs/mps-first-song/audio.flac` — 12.0 s, 48 kHz stereo (52.9 s) |
| Validation output | `runs/final-mps-validation/audio.flac` — 29.5 s, 48 kHz stereo (99.9 s, fresh-env launcher run) |
| Launcher | `./run-yue2.sh` (wraps `.venv/bin/yue2`, forces `--device mps`) |

## Quick start (fresh clone)

```bash
git clone https://github.com/brentmsmith/YuE.git
cd YuE && git checkout apple-silicon-mps   # or clone -b apple-silicon-mps
./setup-mac.sh        # python3.12 -> .venv -> pip install . -> yue2 doctor (offline verify)
./run-yue2.sh --request examples/song.json --id my-song --config short-test-config.json
```

`setup-mac.sh` requires a `python3.12` (Homebrew `brew install python@3.12`
or pass one explicitly: `./setup-mac.sh --python /path/to/python3.12`).
The first generation downloads both checkpoints into the HF cache (a few
GiB; see below to relocate it). `run-yue2.sh` prefers `.venv/bin/yue2`
(in-repo), then `../venv/bin/yue2` (sibling layout), then `yue2` on `PATH`;
it uses `../models/hf` as `HF_HOME` when that sibling dir exists, else the
default HF cache. Run outputs land in `runs/` (gitignored).

## Install steps (as performed)

```bash
cd <parent>   # directory that will contain the YuE checkout
git clone https://github.com/multimodal-art-projection/YuE2.git   # main branch, commit 88da114
cd YuE2
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install .          # skips the NVIDIA-only `fast` extra (vllm/triton/CUDA)
# first `yue2 generate` (or `yue2 doctor`) downloads both checkpoints into the HF cache
```

Notes:

- Do **not** install the `fast` extra — it is NVIDIA-only (vllm, triton).
- `examples/generate.py` hardcodes `device="cuda"` — use the `yue2` CLI instead
  (it supports `--device mps`).
- Model download goes to the default HF cache (`~/.cache/huggingface/hub/`).
  To relocate it, set `HF_HOME` (or `HF_HUB_CACHE`) before running; the
  launcher also picks up a `../models/hf` sibling dir automatically
  (this machine keeps the cache there).
- Run outputs (`runs/...`) and the historical validation runs referenced
  below (`mps-first-song`, `final-mps-validation`, `long-mps-test`,
  `mps-flush-verify`, `cpu-smoke-mpsnoop`) are gitignored; on this machine
  they live outside the repo in a sibling `diagnostics/` dir — a fresh
  clone regenerates them under `runs/`.
- Verify the environment any time with:
  `./run-yue2.sh doctor --verify-hashes`

## Generating music

```bash
cd <repo>

# Example song (full-length, ~6 min default sampling budget — slow on MPS)
./run-yue2.sh --request examples/song.json --id my-song

# Short test (~12 s of audio, ~1 min wall time) using the sampling override
./run-yue2.sh --request examples/song.json --id my-song --config short-test-config.json
```

The launcher adds `--device mps` automatically and passes all other args
through to `yue2 generate` (see `./run-yue2.sh generate --help` for all
options: `--lyrics`, `--style`, `--seed`, `--stage`, `--budget`, `--resume`, …).

`short-test-config.json` (kept in the repo root) caps sampling for quick tests:

```json
{"abc": {"max_tokens": 768}, "semantic": {"max_tokens": 300}}
```

Outputs land in `runs/<output-root>/<id>/` (`audio.flac`, `config.json`,
`result.json`, `score.abc`, plan/semantic/latent artifacts). Without
`--output`, the output root is `runs/default` (e.g. `runs/default/<id>/`);
the verified run below used `--output runs` → `runs/mps-first-song/`.

## MPS patch

**The only source modification** is in `src/yue2/pipeline.py` (`decode()`):
the VAE decode stage is forced onto **CPU** when the pipeline device is MPS.
AR planning, semantic tokens, and NAR synthesis all still run on MPS in bf16.

Apply/inspect the patch: `git diff src/yue2/pipeline.py` (3 hunks adding
`vae_device = "cpu" if self.device.type == "mps" else self.device` and routing
`YuE2VAE.from_pretrained(...)`, `self._vae.to(vae_device)`, and
`model.decode(z.to(vae_device))` through it).

After editing `src/yue2/pipeline.py`, reinstall the package:

```bash
.venv/bin/pip install --no-deps --force-reinstall .
```

### Exact cause of the MPS VAE decode failure (experimentally verified)

PyTorch 2.10.0 MPS `F.conv1d` raises

```
NotImplementedError: Output channels > 65536 not supported at the MPS device.
```

when the conv's **output spatial length** exceeds 65536 — the message is
misleading; it is *not* about output channels in this case. Verified with
minimal probes (torch 2.10.0, Apple M3 Max):

| Probe (MPS `F.conv1d`, stride 1) | Result |
|---|---|
| out_len 65536, 512 in/out channels (33.5M output elements) | **PASS** |
| out_len 65537, **1** channel (65k output elements) | **FAIL** (same message) |
| out_len 65537, 2 channels | **FAIL** (same message) |
| out_len 65535/65536, 2 channels | PASS |
| out_channels 70000, out_len 16 | **FAIL** (same message) |
| out_channels 65536, out_len 16 | PASS |
| batch 2, out_len 65536/65537 | threshold unchanged (batch-independent) |
| `F.conv_transpose1d` out_len 200001 | **PASS** (exempt) |

So there are two real limits — **output spatial length > 65536** and
**output channels > 65536** — with the same error text; only the spatial one
matters for YuE2 (VAE max width is 1024 channels). It is *not* total tensor
size (a 65536-long, 512-channel output = 134 MB passes while a 65537-long,
1-channel output fails).

In the real YuE2-VAE decoder (64-channel latents, ×1920 total upsampling),
decoding the full 12-second test latent (300 latent frames → 575,936 samples)
fails at decoder layer `layers.4.layers.2.layers.1`
(`Conv1d 128→128, kernel 7, stride 1, padding 3`) whose input/output length is
143,984 at that point. A standalone `F.conv1d` with that exact geometry fails
identically. The largest latent tile the MPS decoder accepts is **34 frames**
(natural output length 65,216 samples); 35 frames (67,136 samples) fails.
`ConvTranspose1d` (all the block upsamplers) is exempt, but the full-rate
`Conv1d` layers (residual units + final head) always exceed 65536 for any
practical tile, so CPU decode is the only viable path.

### CPU decode is numerically safe (verified)

Same latents (`runs/mps-first-song/latent.npy`, 300 frames), decoded on both
devices over identical 34-frame slices (start/middle/end):

| Slice (frames) | max abs diff | mean abs diff | RMS diff |
|---|---|---|---|
| [0:34] | 1.013e-06 | 8.041e-08 | 1.147e-07 |
| [133:167] | 7.451e-07 | 8.753e-08 | 1.149e-07 |
| [266:300] | 6.855e-07 | 4.501e-08 | 6.631e-08 |

Worst-case CPU-vs-MPS difference ≤ **1.01e-06** (float32; far below audible).
The saved `audio.flac` reproduces from the latents via CPU decode to within
5.96e-08 (FLAC 24-bit rounding). Moving only the VAE decode to CPU does not
materially alter the audio.

### Stage device placement (verified by live instrumentation)

Monkeypatched audit run (no source changes), `PYTORCH_ENABLE_MPS_FALLBACK`
**unset** (so unsupported MPS ops raise instead of silently falling back):

| Stage | Params | Compute device | Observed |
|---|---|---|---|
| AR planning (abc) | `mps:0` bf16 | MPS | 571 tokens, 24.0 tok/s |
| AR semantic | `mps:0` bf16 | MPS | 300 tokens, 22.1 tok/s (NAR state tensors on `weight.device`) |
| NAR synthesis | `mps:0` bf16 | MPS | 32/32 steps; result moved to CPU at the end by upstream design |
| VAE decode | `cpu` fp32 | CPU | decoder params, latents, and audio all on CPU |

No stage silently falls back to CPU: with the fallback env var unset, any
unsupported MPS op would raise `NotImplementedError`; the only CPU stage is
the patched VAE decode.

## MPS-specific behavior & known limitations

- `graph_fallback_reason: "non_cuda_device"` in timing output is **expected** —
  CUDA graphs are CUDA-only; sampling runs eager on MPS.
- AR generation speed on MPS (measured): ~34.5 tok/s planning, ~22-31 tok/s
  semantic — a full 6-minute song (9000 semantic tokens) takes roughly 5-8 min
  in the semantic stage alone; prefer short tests on this machine.
- VAE decode runs on CPU (see patch above) — fast and exact (≤1e-6 vs MPS).
- bf16 on MPS works; the 3.58B-param model loads with ~6.8 GiB MPS allocated.
  Peak process RSS for a 29.5 s generation is ~5.1 GiB (plus ~14.7 GiB peak
  memory footprint reported by macOS). With other memory-hungry apps open the
  system was under swap pressure (~28.5 GB swap in use before/after the run;
  the run itself added ~45 MB and 0 page swaps).
- `yue2 doctor` and all 172 repo unit tests pass (11 skipped: CUDA/vLLM-only).
  Tests are offline (tiny synthetic models) — safe to run:
  `.venv/bin/python -m pytest tests/ -q` (pytest installed into `.venv`).
- The launcher works from a clean shell: verified with
  `env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin ./run-yue2.sh ...`
  (no activated venv or inherited env required).

## Verified results

### `runs/mps-first-song/` — 12 s smoke test

Generated with:

```bash
.venv/bin/yue2 generate --device mps --config short-test-config.json \
  --request examples/song.json --output runs --id mps-first-song
```

- Status: **complete**, 52.9 s end-to-end — plan 16.5 s (571 tokens, MPS),
  semantic 9.3 s (300 tokens, MPS, limit reached as configured), NAR 14.4 s
  (32/32 steps, MPS), VAE decode 2.7 s (CPU).
- Audio: `audio.flac`, 575,936 samples, 48 kHz stereo, 12.0 s, peak 0.627,
  RMS 0.106, all finite. No OOM, no NaNs.
- Run log with timings: `runs/mps-first-song.log`.

### `runs/final-mps-validation/` — 29.5 s validation (final validation run)

Generated through the launcher from a **clean environment**
(`env -i HOME=... PATH=/usr/bin:/bin:...`, no activated venv), 99.9 s pipeline
time / 107 s wall clock, with semantic cap 900 (the request ended naturally at
738 tokens — `truncated: false` for both abc and semantic):

```bash
printf '%s\n' '{"abc": {"max_tokens": 768}, "semantic": {"max_tokens": 900}}' > long-config.json
env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  /usr/bin/time -l ./run-yue2.sh --request examples/song.json \
  --id final-mps-validation --output runs \
  --config long-config.json
```

- Devices: AR plan MPS (bf16, 34.5 tok/s), semantic MPS (bf16, 31.1 tok/s),
  NAR MPS (bf16, 32 steps), VAE decode CPU (fp32).
- Timings: plan 16.5 s (571 tokens), semantic 23.7 s (738 tokens, natural EOS),
  NAR 40.9 s, VAE decode 9.5 s (CPU), e2e 99.9 s.
- Audio: `audio.flac`, 1,414,976 samples, 48 kHz stereo PCM_24, 29.479 s,
  peak 0.549, RMS 0.0768, all finite, **0.0000 %** of samples at/near ±1.0 —
  no clipping, not silent.
- Memory: peak RSS 5.13 GiB (5,510,447,104 B), 0 page swaps; system-wide swap
  grew ~45 MB during the run (pre-existing ~28.5 GB from other apps).
- All artifacts saved (`result.json`, `latent.npy`, `semantic.npy`,
  `score.abc`, `plan.json`, …).

### Long-song stress test (`runs/long-mps-test/`) — 2-minute generation

A 2-minute full-song run through the public pipeline API
(long-song driver + request in a local `diagnostics/validation-scratch/`
dir outside the repo, seed 831001, semantic cap 3000 tokens).
The driver wraps the public stage methods with pure-recording instrumentation
plus `torch.mps.empty_cache()` after every NAR ODE step.

**Why the empty-cache call is required on MPS.** A first attempt at this run
(partial log: `diagnostics/validation-scratch/long-run-driver-crashed.log`, kept in a local diagnostics dir outside the repo) drove the
machine into a hard memory crash and had to be killed: during NAR the MPS
*driver-allocated* memory ratcheted 10.5 GB → 38.5 GB in ~80 s while MPS
*live-allocated* stayed ~7.6-8.4 GB. The MPS allocator caches freed blocks of
varying per-step sizes and never returns them, so on a 36 GB unified-memory
machine with other apps resident the NAR stage of a multi-minute song
exhausts unified memory (swap was full: 27.2/27.6 GB, ~40 MB free pages).
Calling `torch.mps.empty_cache()` once per NAR ODE step (32 calls per chunk)
keeps driver-allocated flat: **9.24 GB for the entire NAR stage** vs 38.5 GB
without it. It only releases cached-free blocks (live tensors untouched), so
it is numerically inert.

**Results (after the fix):**
- Timings: plan 97.1 s (1174 tokens, 12.1 tok/s), semantic 401.6 s (3000
  tokens, cap reached, 7.5 tok/s; quartiles 8.9 → 7.0 tok/s — mild KV-length
  slowdown), NAR 539.3 s (32 steps; ~16.9 s/step — NAR attention cost grows
  ~quadratically with song length, vs 1.3 s/step at 738 tokens), VAE decode
  44.6 s (CPU fp32, 3 tiles), e2e 1093.7 s for **120.0 s of audio** (~9.1×
  realtime).
- This run executed while the machine was still recovering from the crash
  (heavy swap from other apps); the earlier 30 s-song validation ran plan at
  34.5 tok/s and semantic at 31 tok/s, so these numbers are a conservative
  floor, not clean-host timings.
- Memory (fixed run): MPS live 8.26 GB peak, driver 11.5 GB peak (early AR
  spike), **NAR driver flat at 9.24 GB across all 32 steps**; RSS 947 MB
  during NAR, ~7 GB during CPU fp32 VAE decode (fp32 intermediates for ~2M
  sample tiles; swap +4.5 GB transient, system free bottomed at 33%) — no
  crash, no guard trips.
- Audio: `audio.flac` 48 kHz stereo PCM_24, 5,759,936 samples, 120.000 s,
  peak 0.756, RMS 0.111, all finite, 0.0000 % at ±1.0 — no clipping, not
  silent. Semantic hit the 3000-token cap (`truncated: true`) by design.
- Full driver report: `diagnostics/validation-scratch/long-run-report.json` (local diagnostics dir, outside the repo).

**Long-song guidance for MPS (36 GB):** songs ≥ ~1 min must call
`torch.mps.empty_cache()` per NAR ODE step (or equivalent) or the MPS
allocator ratchet will exhaust unified memory during NAR. Raw `vm_stat`
"Pages free" is *not* a reliable danger signal on macOS (the kernel keeps it
low by design and reclaims inactive pages on demand); `memory_pressure -Q`'s
system-wide free percentage plus `torch.mps.driver_allocated_memory()` are
the meaningful metrics.

### MPS NAR cache flush — required on this Mac, now built into the pipeline

**Required.** Without it, the MPS (Metal) allocator caches freed per-step
NAR blocks and never returns them: driver-allocated memory ratcheted
10.5 → 38.5 GB over the NAR stage of a ~3-minute song (MPS *live*
allocated stayed ~8.4 GB), exhausting unified memory on this 36 GB
machine and crashing the system (measured during the long-song
benchmark; see the crashed-run log in `../diagnostics/validation-scratch/`, a local diagnostics dir outside the repo).

**How it is enabled.** `src/yue2/pipeline.py` `synthesize()` now wraps the
NAR `on_progress` callback (fires once per ODE step) with
`torch.mps.empty_cache()` when the pipeline device is MPS — no wrapper
script needed, `run-yue2.sh` and the plain CLI both get it automatically.
Gated on `device.type == "mps"` (no-op on CPU/CUDA, verified by a CPU
smoke run), mirrors the existing upstream `torch.cuda.empty_cache()`
calls in `close()`/`decode()`. Cost: ~0.29 s per flush (32-step NAR:
~9 s). The `.venv` site-packages copy is synced with `src/` (plain-copy
install, one-file `cp`), so the CLI picks it up without reinstalling.

**Normal-run behavior** (`run-yue2.sh ...`, default MPS): planning,
semantic and NAR run on `mps:0` bf16; VAE decode on CPU (MPS conv1d
output-length limit). NAR logs exactly two lines:
`[YuE2] MPS NAR cache flush: enabled (one empty_cache per ODE step)` at
NAR start and `[YuE2] MPS NAR driver memory: X-Y MB over N samples
(per-step cache flush)` at NAR end — flat X..Y = bounded (verified:
7363-7363 MB across a full run vs 8158 MB after a no-flush NAR and
10.5→38.5 GB ratcheting on the crashed 2-min benchmark). Normal
generation limits are unchanged (semantic max 9000 upstream default; the
3000-token cap was benchmark-driver-only).

**Verified.** (1) NAR A/B on identical plan+semantic
(`../diagnostics/validation-scratch/nar_flush_ab.py`, local diagnostics dir): upstream no-flush vs patched
per-step flush — bit-exact latents (`max|A-B| = 0.0`), and both equal the
pre-patch baseline `latent.npy` bit-for-bit, so the flush cannot change
seeds/tensors/results. (2) Launcher run `runs/mps-flush-verify/`
(`examples/song.json`, seed 831001): flush lines present, driver memory
flat 7363 MB, 27.8 s audio in 124.9 s, natural EOS, valid audio (finite,
peak 0.58, no clipping). AR stages are not run-to-run deterministic on
MPS bf16 (semantic 697 vs 738 tokens across runs at the same seed) —
expected, unrelated to the flush (A/B held inputs fixed). (3) CPU smoke
run `runs/cpu-smoke-mpsnoop/` (`--device cpu`, tiny caps): no flush
lines, clean completion. Upstream test suite: 172 passed / 11 skipped
(before the NAR patch; the patch touches only `synthesize()` runtime, no
library behavior).
