#!/usr/bin/env bash
# YuE2 launcher for Apple Silicon (MPS) — Mac M3 Max setup.
#
# Wraps the installed `yue2` CLI and forces --device mps.
# The VAE decode stage automatically runs on CPU via a small patch in
# src/yue2/pipeline.py (PyTorch MPS conv1d fails when output length > 65536).
# The same patch also flushes torch.mps.empty_cache() once per NAR ODE step
# (Metal caches freed per-step blocks and never returns them; without the
# flush, NAR driver memory ratchets 10.5 -> 38.5 GB and crashes this machine).
#
# Usage:
#   ./run-yue2.sh --request examples/song.json --id my-song
#   ./run-yue2.sh --request examples/song.json --id my-song --config short-test-config.json
#   ./run-yue2.sh doctor --verify-hashes          # environment check
#   ./run-yue2.sh generate --help                 # all generation options
#
# Python env: .venv/bin/yue2 (created by ./setup-mac.sh or the install steps
# in MAC_INSTALL_NOTES.md), else ../venv/bin/yue2 (sibling of the repo, the
# layout used on this machine), else the yue2 CLI on PATH.
# Models: ../models/hf (sibling, this machine's local HF cache) if present,
# else the default HF cache (~/.cache/huggingface) — first run downloads.
#
# Any extra args are passed through to `yue2 generate` unless the first
# argument is a subcommand (doctor, generate, ...), in which case this
# script just forwards everything with --device mps added for generate.
set -euo pipefail
cd "$(dirname "$0")"

# Local model cache next to the repo (optional; fresh clones download to the
# default HF cache unless HF_HOME is set).
if [[ -z "${HF_HOME:-}" && -d ../models/hf ]]; then
  export HF_HOME="$(cd .. && pwd)/models/hf"
fi

YUE=""
for cand in .venv/bin/yue2 ../venv/bin/yue2; do
  if [[ -x "$cand" ]]; then YUE="$cand"; break; fi
done
if [[ -z "$YUE" ]]; then
  YUE="$(command -v yue2 || true)"
fi
if [[ -z "$YUE" ]]; then
  echo "error: yue2 CLI not found — create .venv with ./setup-mac.sh (or see MAC_INSTALL_NOTES.md)" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [doctor|generate|...] [args...]" >&2
  echo "   or: $0 --request examples/song.json --id my-song" >&2
  exit 2
fi

case "$1" in
  doctor|generate|batch)
    # Explicit subcommand: forward as-is, adding --device mps for generate.
    if [[ "$1" == "generate" ]]; then
      shift
      exec "$YUE" generate --device mps "$@"
    else
      exec "$YUE" "$@"
    fi
    ;;
  *)
    # Bare generation args: prepend the generate subcommand.
    exec "$YUE" generate --device mps "$@"
    ;;
esac
