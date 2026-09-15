# Mac MPS Fixes — YuE2 (m-a-p/YuE2-3B) on Apple M3 Max

- **Upstream commit:** `88da114` ("Sync Suno v6 WildSongBench results and frontier figure")
- **YuE2 version:** `yue2-infer 0.1.6` (installed via `pip install .` into `.venv`, plain site-packages copy)
- **Modified file:** `src/yue2/pipeline.py` — this is the **only** modified source file
- **Patch file:** `MAC_MPS_FIXES.patch` (repo root, next to this file)

## What the patch does (2 changes)

1. **VAE decode → CPU on MPS.** PyTorch's MPS `conv1d` fails with
   `NotImplementedError: Output channels > 65536 not supported at the MPS
   device` whenever a conv's *output spatial length* exceeds 65536 (the error
   message is misleading — it is a length limit). The VAE decoder upsamples
   each latent tile ×1920, so any practical tile exceeds it. The patch routes
   `YuE2VAE` decode to CPU when `pipe.device.type == "mps"` (numerically
   verified: max |diff| ≤ 1e-6 vs CPU reference).

2. **`torch.mps.empty_cache()` once per NAR ODE step.** The Metal allocator
   caches freed per-step blocks and never returns them: without the flush,
   MPS *driver-allocated* memory ratcheted 10.5 → 38.5 GB during the NAR
   stage of a ~3-minute song (live allocated stayed ~8 GB), exhausting
   unified memory on a 36 GB machine and crashing macOS. With the flush
   (wired into the NAR `on_progress` callback, gated on
   `device.type == "mps"`, no-op on CPU/CUDA), driver memory stays flat
   (~7.4 GB over a full run). Cost: ~0.3 s per flush. Two log lines confirm
   it per run: `MPS NAR cache flush: enabled …` and
   `MPS NAR driver memory: X-Y MB … (per-step cache flush)`.

Note: AR generation (plan/semantic) and NAR run on `mps:0` bf16; only VAE
decode is CPU. `run-yue2.sh` (repo root) invokes the patched CLI — no
extra flags needed.

## After a future YuE update / reinstall

The `.venv` site-packages copy is a **plain copy**, not an editable install:
a reinstall (`pip install .`) overwrites it. After any update or reinstall:

1. Check whether the patch still applies (non-destructive):

   ```bash
   cd <repo>
   git stash            # only if your pipeline.py still has local changes
   git apply --check --verbose MAC_MPS_FIXES.patch
   git stash pop        # restore your working copy
   ```

   (Or against a pristine checkout: `git worktree add --detach
   ../upstream-check <commit> && git -C ../upstream-check apply --check
   --verbose ../YuE/MAC_MPS_FIXES.patch && git worktree remove
   ../upstream-check`.)

2. If it applies and upstream hasn't fixed the issues, apply it:

   ```bash
   cd <repo>
   git apply MAC_MPS_FIXES.patch
   cp src/yue2/pipeline.py .venv/lib/python3.12/site-packages/yue2/pipeline.py
   ```

   (The final `cp` syncs the venv copy — the installed package does **not**
   follow the source tree.)

3. If `git apply --check` fails (context changed), do **not** force it
   (`--3way`/fuzz). A future upstream release may already fix one or both
   issues. Re-check each condition in the new code before re-patching:

   - VAE: does `pipeline.py` still decode on `self.device` for MPS? Does
     `YuE2VAE.decode` still hit the MPS 65536-length conv1d limit (test with
     a real generate; the failure is an `NotImplementedError` at decode)?
   - NAR: is MPS driver memory still ratcheting during NAR (watch
     `torch.mps.driver_allocated_memory()` across steps; flat = fixed)?

   Then re-apply the equivalent minimal edit by hand, update the venv copy,
   and re-verify: MPS full generate (two log lines present, driver memory
   flat), CPU smoke run (no MPS lines, clean completion), and NAR A/B
   bit-exactness (see `MAC_INSTALL_NOTES.md`).

Emergency reference copy of the last verified working `pipeline.py`
(kept outside the repo, not committed):
`../validation-scratch/pipeline.py.working-m3-backup`.
