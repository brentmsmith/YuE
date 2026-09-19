#!/usr/bin/env bash
# setup-mac.sh — macOS (Apple Silicon) one-command install for YuE2.
#
# Creates .venv in the repo with python3.12, installs yue2 (skipping the
# NVIDIA-only `fast` extra: vllm/triton/CUDA), and verifies the install
# with `yue2 doctor` (offline; tiny synthetic models — no checkpoint
# download). MPS is verified by the doctor; the first `run-yue2.sh generate`
# downloads the checkpoints into the HF cache.
#
# Tested on: Apple M3 Max, macOS 15, Homebrew python@3.12 (3.12.10),
# torch 2.10.0 (MPS build). See MAC_INSTALL_NOTES.md for the full tested
# configuration, the CPU VAE decode workaround, and the NAR MPS
# cache-management fix (both already built into src/yue2/pipeline.py).
#
# Usage:
#   ./setup-mac.sh              # create .venv, install, verify
#   ./setup-mac.sh --python /path/to/python3.12   # pick the interpreter
set -euo pipefail
cd "$(dirname "$0")"

PY="${1:-}"
if [[ -n "$PY" && "$PY" != "--python" ]]; then
  echo "setup-mac.sh: unexpected argument: $1 (only --python <path> is supported)" >&2
  exit 2
fi
if [[ -n "$PY" ]]; then shift 2; fi

if [[ -z "$PY" ]]; then
  for cand in /opt/homebrew/bin/python3.12 python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
        PY="$cand"; break
      fi
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "setup-mac.sh: python3.12 not found — install Homebrew python@3.12 (brew install python@3.12)" >&2
  echo "  or pass an interpreter explicitly: ./setup-mac.sh --python /path/to/python3.12" >&2
  exit 1
fi
echo "setup-mac.sh: python: $($PY --version 2>&1)"

if [[ -x .venv/bin/yue2 ]]; then
  echo "setup-mac.sh: .venv already installed (.venv/bin/yue2 exists) — verifying only"
else
  echo "setup-mac.sh: creating .venv and installing yue2 (this can take a few minutes)"
  "$PY" -m venv .venv
  .venv/bin/pip install .
fi

echo "setup-mac.sh: verifying the install (yue2 doctor)"
./run-yue2.sh doctor
echo "setup-mac.sh: done."
echo "  generate a short test song (~1 min on an M3 Max):"
echo "    ./run-yue2.sh --request examples/song.json --id my-song --config short-test-config.json"
