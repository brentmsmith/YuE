#!/usr/bin/env bash
# make-song.sh — one-command song generation on the validated YuE2/MPS setup.
#
# Usage:
#   ./make-song.sh <request.json> [options]
#
#   ./make-song.sh song-requests/blank-template.json
#   ./make-song.sh my-song.json --id song-001-b --seed 20260914
#   ./make-song.sh song-requests/atmospheric-metal.json --max-seconds 60
#   ./make-song.sh song-requests/rnb-slow-jam.json --stage plan   # fast ABC preview, no audio
#   ./make-song.sh my-song.json --dry-run                    # show the exact command
#
# Options:
#   --id NAME         output id (default: "id" field in the request, else
#                     the request filename without .json). Output goes to
#                     runs/songs/<id>/ (override the root with SONGS_DIR).
#   --seed N          override the request's seed
#   --cot full|melody|off   override the planning mode (default: request's,
#                     else "full")
#   --cfg-scale F     override text guidance (default 1.0; 1.01 when cot=off)
#   --max-seconds N   cap the song length: semantic.max_tokens ≈ 25 × N.
#                     Upper bound only — the song still ends naturally at the
#                     lyrics; result.json "truncated" reports a hard cut.
#                     Also settable per-request as "target_seconds".
#   --config FILE     generation-config JSON (Sampling overrides). Merged;
#                     --max-seconds/target_seconds wins on semantic min/max.
#   --stage plan|audio  "plan" writes only the ABC plan (fast preview);
#                     "audio" (default) generates the full song
#   --resume          re-verify an existing completed run (no regeneration)
#   --quiet           hide YuE2 progress on stderr
#   --dry-run         print the prepared request/settings, do not generate
#
# Request JSON: real YuE2 fields (style, lyrics, cot, seed, abc, abc_path,
# cfg_scale, id, abc_sampling, semantic_sampling) plus local bookkeeping
# fields that are recorded in metadata but never sent to the model:
# target_seconds, title, genres, bpm, key, vocal, concept, variant, notes,
# rating. Keys starting with "_" are comments and ignored. Unknown keys are
# rejected with the allowed list.
#
# The script calls the existing ./run-yue2.sh launcher, so the validated MPS
# setup is preserved unchanged: MPS bf16 planning/semantic/NAR with the
# per-NAR-step torch.mps.empty_cache() flush, CPU fp32 VAE decode.
#
# Outputs (per song, never overwritten):
#   runs/songs/<id>/audio.flac        the song (48 kHz stereo)
#   runs/songs/<id>/result.json       YuE2 result (timings, truncation, hashes)
#   runs/songs/<id>/metadata.json     this workflow's bookkeeping + analytics
#   runs/songs/catalog.jsonl          one line per song for later analytics
# plus YuE2's own artifacts (score.abc, plan.json, request.json,
# config.json, semantic.npy, latent.npy, ...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Runtime lives outside the git repo: ../venv (python env), ../models/hf (HF cache).
export HF_HOME="${HF_HOME:-$(cd .. && pwd)/models/hf}"

PY="../venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || { echo "make-song.sh: no python3 found" >&2; exit 1; }

RUN="./run-yue2.sh"
[[ -x "$RUN" ]] || { echo "make-song.sh: $RUN not found (run from the YuE repo root)" >&2; exit 1; }

SONGS_DIR="${SONGS_DIR:-runs/songs}"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit "${1:-0}"; }

REQUEST=""
ID="" SEED="" COT="" CFG_SCALE="" MAX_SECONDS="" CONFIG=""
STAGE="audio" QUIET="" RESUME="" DRY_RUN=0

need_value() { [[ $# -ge 2 ]] || { echo "make-song.sh: $1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --id) need_value "$@"; ID="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --cot) need_value "$@"; COT="$2"; shift 2 ;;
    --cfg-scale) need_value "$@"; CFG_SCALE="$2"; shift 2 ;;
    --max-seconds) need_value "$@"; MAX_SECONDS="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --stage) need_value "$@"; STAGE="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    --quiet|--no-progress) QUIET="--quiet"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --*) echo "make-song.sh: unknown option: $1 (see --help)" >&2; exit 2 ;;
    *) if [[ -z "$REQUEST" ]]; then REQUEST="$1"; shift
       else echo "make-song.sh: unexpected extra argument: $1 (one request file)" >&2; exit 2; fi ;;
  esac
done

[[ -n "$REQUEST" ]] || { echo "make-song.sh: a request JSON file is required (e.g. ./make-song.sh song-requests/blank-template.json)" >&2; exit 2; }
[[ "$STAGE" == "plan" || "$STAGE" == "audio" ]] || { echo "make-song.sh: --stage must be plan or audio" >&2; exit 2; }
[[ -f "$REQUEST" ]] || { echo "make-song.sh: request file not found: $REQUEST" >&2; exit 1; }
REQUEST_ABS="$(cd "$(dirname "$REQUEST")" && pwd)/$(basename "$REQUEST")"

TMPDIR_RUN="$(mktemp -d "${TMPDIR:-/tmp}/make-song.XXXXXX")"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

# ---- Phase 1: validate the request, strip local fields, prepare config ----
# Writes the sanitized YuE2 request JSON and (if needed) a merged generation
# config JSON into $TMPDIR_RUN; prints the run plan as JSON on stdout.
prepare() {
  MS_REQ="$REQUEST_ABS" \
  MS_ID="$ID" MS_SEED="$SEED" MS_COT="$COT" MS_CFG_SCALE="$CFG_SCALE" \
  MS_MAX_SECONDS="$MAX_SECONDS" MS_CONFIG="$CONFIG" MS_STAGE="$STAGE" \
  MS_SONGS_DIR="$SONGS_DIR" MS_TMP="$TMPDIR_RUN" \
  "$PY" - <<'PREPARE_PY'
import json, os, re, sys
from pathlib import Path

YUE2_KEYS = {"style", "tags", "lyrics", "cot", "seed", "abc", "abc_path",
             "cfg_scale", "id", "abc_sampling", "semantic_sampling"}
CLI_META_KEYS = {"lang", "eval_index", "clip_id", "prompt"}   # accepted by yue2, ignored
LOCAL_KEYS = {"target_seconds", "title", "genres", "bpm", "key", "vocal",
              "concept", "variant", "notes", "rating"}
COT_MODES = {"full", "melody", "off"}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")
SECONDS_PER_TOKEN = 25.0   # verified: 300 tokens -> 12.0 s, 738 -> 29.5 s, 3000 -> 120.0 s

def die(msg):
    print(f"make-song.sh: {msg}", file=sys.stderr); sys.exit(2)

def as_number(value, field):
    if isinstance(value, str):   # CLI flags arrive as strings
        try:
            value = float(value)
        except ValueError:
            die(f"{field} must be a number, got {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        die(f"{field} must be a number")
    return float(value)

req_path = Path(os.environ["MS_REQ"])
try:
    data = json.loads(req_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    die(f"request file is not valid JSON: {req_path}: {e}")
if not isinstance(data, dict):
    die(f"request file must contain a JSON object: {req_path}")

unknown = sorted(k for k in data if k not in YUE2_KEYS and k not in CLI_META_KEYS
                 and k not in LOCAL_KEYS and not k.startswith("_"))
if unknown:
    die(f"unknown request fields {unknown}; allowed: YuE2 {sorted(YUE2_KEYS)}, "
        f"local bookkeeping {sorted(LOCAL_KEYS)}, yue2 metadata {sorted(CLI_META_KEYS)}, "
        f"any '_'-prefixed comment key")

# --- style / lyrics (required) ---
style, tags = data.get("style"), data.get("tags")
if style is None and tags is not None:
    style, tags = tags, None
if tags is not None and tags != style:
    die("style and tags disagree; pass only one")
if not isinstance(style, str) or not style.strip():
    die('style: required non-empty string — put language, genre, instruments, '
        'vocal character, mood and tempo (e.g. "... , 88 BPM") in one line')
lyrics = data.get("lyrics")
if not isinstance(lyrics, str) or not lyrics.strip():
    die('lyrics: required non-empty string — use [Verse]/[Chorus]/[Bridge] '
        'section tags, one lyric line per line, blank line between sections')

# --- planning mode / seed / guidance ---
cot = os.environ.get("MS_COT") or data.get("cot") or "full"
if cot not in COT_MODES:
    die(f"cot must be one of {sorted(COT_MODES)}, got {cot!r}")

seed_raw = os.environ.get("MS_SEED") if os.environ.get("MS_SEED") else data.get("seed", 831001)
try:
    seed = int(seed_raw)
except (TypeError, ValueError):
    die(f"seed must be an integer, got {seed_raw!r}")
if not 0 <= seed < 2**63:
    die("seed must be an integer in [0, 2**63)")

cfg_raw = os.environ.get("MS_CFG_SCALE") if os.environ.get("MS_CFG_SCALE") else data.get("cfg_scale")
if cfg_raw is not None and cfg_raw != "":
    try:
        cfg_scale = float(cfg_raw)
    except (TypeError, ValueError):
        die(f"cfg_scale must be a number, got {cfg_raw!r}")
    if not 0 <= cfg_scale <= 20:
        die("cfg_scale must be in [0, 20] (default 1.0; 1.01 when cot=off)")
else:
    cfg_scale = None

# --- external ABC (cover / reharmonization path) ---
abc, abc_path = data.get("abc"), data.get("abc_path")
if abc is not None and abc_path is not None:
    die("pass only one of abc (inline text) or abc_path (file)")
if abc_path is not None:
    p = Path(abc_path)
    if not p.is_absolute():
        p = req_path.parent / p
    try:
        abc = p.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        die(f"abc_path not readable: {e}")
    if not abc.strip():
        die("abc_path file is empty")
if abc is not None:
    if not isinstance(abc, str) or not abc.strip():
        die("abc must be non-empty ABC text")
    if cot == "off":
        die("external abc requires cot=melody or cot=full (not off)")

# --- id ---
song_id = os.environ.get("MS_ID") or data.get("id") or req_path.stem
if isinstance(song_id, str):
    song_id = song_id.strip()
if not isinstance(song_id, str) or not ID_RE.fullmatch(song_id) or song_id in {".", ".."}:
    die(f"id must be filename-safe ([A-Za-z0-9][A-Za-z0-9_.-], not . or ..): {song_id!r}")

# --- length cap -> semantic token cap (~25 tokens/second, verified) ---
ts_raw = os.environ.get("MS_MAX_SECONDS") if os.environ.get("MS_MAX_SECONDS") else data.get("target_seconds")
target_seconds = None
if ts_raw is not None and ts_raw != "":
    target_seconds = as_number(ts_raw, "target_seconds/--max-seconds")
    if not 4 <= target_seconds <= 900:
        die("target_seconds/--max-seconds must be 4..900 (min_tokens floor is 200 tokens = 8 s; "
            "the model's hard default cap is 9000 tokens ≈ 6 min)")

# --- generation config (Sampling overrides) ---
config_file = os.environ.get("MS_CONFIG") or None
config_data = {}
if config_file:
    try:
        config_data = json.loads(Path(config_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        die(f"--config file not readable JSON: {e}")
    if not isinstance(config_data, dict):
        die("--config file must contain a JSON object")

sem_sampling = data.get("semantic_sampling")
if isinstance(sem_sampling, dict) and "max_tokens" in sem_sampling and target_seconds is not None:
    die("pass only one of semantic_sampling.max_tokens and target_seconds/--max-seconds")

if target_seconds is not None:
    sem = dict(config_data.get("semantic") or {})
    sem["max_tokens"] = max(1, int(round(SECONDS_PER_TOKEN * target_seconds)))
    sem["min_tokens"] = min(200, sem["max_tokens"])
    config_data["semantic"] = sem

try:
    from yue2.protocol import GenerationConfig, Sampling, resolve_sampling
except ImportError:
    die("yue2 package not importable — run from the YuE2 source root with ../venv")
try:
    if config_data:
        GenerationConfig.from_dict(config_data)   # validates merged config early
    base = GenerationConfig()
    for name, value in (("abc_sampling", data.get("abc_sampling")),
                        ("semantic_sampling", sem_sampling)):
        if value is not None:
            resolve_sampling(value, getattr(base, name.split("_")[0]))
except (ValueError, TypeError) as e:
    die(f"invalid sampling/config: {e}")

# --- local bookkeeping fields (metadata only; never sent to the model) ---
def opt_str(field):
    v = data.get(field)
    if v is None:
        return None
    if not isinstance(v, str):
        die(f"{field} must be a string (bookkeeping only)")
    return v

genres = data.get("genres")
if genres is not None and not (isinstance(genres, list) and all(isinstance(g, str) for g in genres)):
    die("genres must be a list of strings (bookkeeping only)")
bpm = data.get("bpm")
if bpm is not None:
    bpm = as_number(bpm, "bpm")
    if 8 <= bpm <= 400 and float(bpm).is_integer():
        bpm = int(bpm)
rating = data.get("rating")
if rating is not None:
    rating = as_number(rating, "rating")
    if not 0 <= rating <= 10:
        die("rating must be in [0, 10] (bookkeeping only)")
descriptors = {"title": opt_str("title"), "genres": genres, "bpm": bpm,
               "key": opt_str("key"), "vocal": opt_str("vocal"),
               "concept": opt_str("concept"), "variant": opt_str("variant"),
               "notes": opt_str("notes") or "", "rating": rating}

# --- sanitized request for yue2 (local fields stripped) ---
sanitized = {"id": song_id, "style": style, "lyrics": lyrics, "cot": cot, "seed": seed}
if cfg_scale is not None:
    sanitized["cfg_scale"] = cfg_scale
if abc is not None:
    sanitized["abc"] = abc
for k in ("abc_sampling", "semantic_sampling"):
    if data.get(k) is not None:
        sanitized[k] = data[k]

tmp = Path(os.environ["MS_TMP"])
request_tmp = tmp / "request.json"
request_tmp.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
config_tmp = None
if config_data:
    config_tmp = tmp / "generation-config.json"
    config_tmp.write_text(json.dumps(config_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

cfg = GenerationConfig.from_dict(config_data) if config_data else GenerationConfig()
plan = {
    "id": song_id,
    "stage": os.environ.get("MS_STAGE") or "audio",
    "songs_dir": os.environ["MS_SONGS_DIR"],
    "request_file": str(req_path),
    "request_tmp": str(request_tmp),
    "config_tmp": str(config_tmp) if config_tmp else None,
    "config_source": config_file,
    "target_seconds": target_seconds,
    "semantic_max_tokens": cfg.semantic.max_tokens,
    "semantic_min_tokens": cfg.semantic.min_tokens,
    "abc_max_tokens": cfg.abc.max_tokens,
    "warnings": [],
    "effective": {"id": song_id, "cot": cot, "seed": seed, "cfg_scale": cfg_scale,
                  "target_seconds": target_seconds,
                  "semantic_max_tokens": cfg.semantic.max_tokens},
    "descriptors": descriptors,
    "original_request": data,
}
if target_seconds is None:
    plan["warnings"].append(
        f"no length cap: semantic max_tokens defaults to {cfg.semantic.max_tokens} "
        f"(~{cfg.semantic.max_tokens / SECONDS_PER_TOKEN:.0f} s upper bound); "
        "on this Mac a multi-minute song can take ~10x its length in wall time — "
        "consider target_seconds or --max-seconds")
if target_seconds is not None and target_seconds > 240:
    plan["warnings"].append(
        f"target_seconds {target_seconds:.0f} is long for this Mac; expect ~"
        f"{target_seconds * 9 / 60:.0f}+ minutes of wall time (measured ~9x realtime at 2 min)")
print(json.dumps(plan, ensure_ascii=False))
PREPARE_PY
}

PLAN="$(prepare)" || exit $?
ID="$("$PY" -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$PLAN")"
CONFIG_TMP="$("$PY" -c 'import json,sys; print(json.load(sys.stdin)["config_tmp"] or "")' <<<"$PLAN")"
"$PY" -c 'import json,sys; [print("make-song.sh: note: " + x, file=sys.stderr) for x in json.load(sys.stdin)["warnings"]]' <<<"$PLAN" >&2

OUTDIR="$SONGS_DIR/$ID"

# ---- never overwrite an existing song ----
if [[ -z "$RESUME" ]]; then
  if [[ -d "$OUTDIR" ]] && [[ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]]; then
    echo "make-song.sh: output directory is not empty: $OUTDIR" >&2
    echo "  refusing to overwrite an existing run. Either:" >&2
    echo "    - use a new id:      ./make-song.sh <request> --id ${ID}-b" >&2
    echo "    - edit the request's \"id\" field (variants: song-001-a/b/c...)" >&2
    echo "    - or re-verify the finished run: ./make-song.sh <request> --resume" >&2
    exit 1
  fi
fi
if [[ -n "$RESUME" && ! -f "$OUTDIR/result.json" ]]; then
  echo "make-song.sh: --resume needs an existing completed run: $OUTDIR/result.json not found" >&2
  exit 1
fi

# ---- build the command (run-yue2.sh adds --device mps and the MPS fixes) ----
CMD=("$RUN" generate --request "$TMPDIR_RUN/request.json" --output "$SONGS_DIR" --id "$ID" --stage "$STAGE")
[[ -n "$CONFIG_TMP" ]] && CMD+=(--config "$CONFIG_TMP")
[[ -n "$SEED" ]] && CMD+=(--seed "$SEED")
[[ -n "$COT" ]] && CMD+=(--cot "$COT")
[[ -n "$CFG_SCALE" ]] && CMD+=(--cfg-scale "$CFG_SCALE")
[[ -n "$QUIET" ]] && CMD+=("$QUIET")
[[ -n "$RESUME" ]] && CMD+=("$RESUME")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "make-song.sh: dry run — nothing generated. Prepared plan:"
  echo "  request:   $REQUEST_ABS"
  echo "  id:        $ID"
  echo "  output:    $OUTDIR"
  echo "  stage:     $STAGE"
  "$PY" -c '
import json, sys
p = json.load(sys.stdin)
e = p["effective"]
cfg = e["cfg_scale"] if e["cfg_scale"] is not None else "default"
length = " ~%ds upper bound" % (e["semantic_max_tokens"] / 25)
if e["target_seconds"] is not None:
    length = " target %.0fs, cap %d tokens (~%.0fs)" % (e["target_seconds"], e["semantic_max_tokens"], e["semantic_max_tokens"] / 25)
print("  cot:       %s   seed: %s   cfg_scale: %s" % (e["cot"], e["seed"], cfg))
print("  length:    semantic max_tokens %d%s" % (e["semantic_max_tokens"], length))
print("  abc plan:  max_tokens %d" % p["abc_max_tokens"])
d = p["descriptors"]
tags = [k + "=" + repr(d[k]) for k in ("title", "genres", "bpm", "key", "vocal", "concept", "variant") if d.get(k) is not None]
if tags:
    print("  request:   " + ", ".join(tags))
' <<<"$PLAN"
  echo "  command:   ${CMD[*]}"
  echo "(remove --dry-run to generate)"
  exit 0
fi

mkdir -p "$SONGS_DIR"
STDOUT_FILE="$TMPDIR_RUN/yue2-stdout.json"
echo "make-song.sh: generating song '$ID' -> $OUTDIR (stage: $STAGE)"
STARTED="$(date +%s)"
if "${CMD[@]}" >"$STDOUT_FILE"; then RC=0; else RC=$?; fi
WALL=$(( $(date +%s) - STARTED ))

if [[ "$RC" != 0 ]]; then
  echo "make-song.sh: generation failed (exit $RC); see the traceback above." >&2
  [[ -f "$OUTDIR/failure.json" ]] && echo "  failure record: $OUTDIR/failure.json" >&2
  echo "  the run directory was kept for inspection: $OUTDIR" >&2
  exit "$RC"
fi

# ---- Phase 2: record metadata + catalog, print where the audio is ----
MS_PLAN="$PLAN" MS_OUTDIR="$OUTDIR" MS_SONGS_DIR="$SONGS_DIR" MS_STDOUT="$STDOUT_FILE" \
MS_SCRIPT_DIR="$SCRIPT_DIR" MS_STARTED="$STARTED" MS_WALL="$WALL" \
"$PY" - <<'FINALIZE_PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

plan = json.loads(os.environ["MS_PLAN"])
outdir = Path(os.environ["MS_OUTDIR"])
songs_dir = Path(os.environ["MS_SONGS_DIR"])
stage = plan["stage"]

try:
    lines = [l for l in Path(os.environ["MS_STDOUT"]).read_text().splitlines() if l.strip()]
    result_line = json.loads(lines[-1])
except (OSError, IndexError, json.JSONDecodeError) as e:
    print(f"make-song.sh: could not parse yue2 output ({e}); run directory: {outdir}", file=sys.stderr)
    sys.exit(1)
resumed = bool(result_line.get("resumed"))

result = {}
result_path = outdir / "result.json"
if result_path.is_file():
    try:
        result = json.loads(result_path.read_text())
    except json.JSONDecodeError:
        result = {}
truncated = result.get("truncated", result_line.get("truncated"))
if isinstance(truncated, dict):
    truncated = truncated.get("semantic", False) or truncated.get("abc", False)
audio_seconds = result.get("audio_seconds")
sample_rate = result.get("sample_rate")
timing = result.get("timing", {})
wall = float(os.environ.get("MS_WALL") or 0)

meta = {
    "schema_version": 1,
    "song_id": plan["id"],
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "stage": stage,
    "status": "resumed-verified" if resumed else "complete",
    "request_file": plan["request_file"],
    "request": plan["original_request"],
    "effective": plan["effective"],
    "config_file": plan["config_source"],
    "semantic_max_tokens": plan["semantic_max_tokens"],
    "descriptors": plan["descriptors"],
    "generation": {
        "device": "mps (VAE decode on cpu)",
        "wall_seconds_started": int(os.environ["MS_STARTED"]),
        "wall_seconds": wall,
        "e2e_seconds": timing.get("e2e_seconds"),
        "truncated": truncated,
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "result": result_line,
    },
    "audio": {"full": str(outdir / "audio.flac") if stage == "audio" else None, "excerpts": []},
    # analytics placeholders — fill in later, never sent to the model:
    "notes": plan["descriptors"]["notes"],
    "rating": plan["descriptors"]["rating"],
    "metrics": {},
}

metadata_path = outdir / "metadata.json"
tmp = metadata_path.with_name(metadata_path.name + f".{os.getpid()}.tmp")
tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(tmp, metadata_path)

catalog = songs_dir / "catalog.jsonl"
row = {k: meta[k] for k in ("song_id", "created", "stage", "status", "request_file")}
row.update({
    "title": plan["descriptors"]["title"], "concept": plan["descriptors"]["concept"],
    "variant": plan["descriptors"]["variant"], "genres": plan["descriptors"]["genres"],
    "bpm": plan["descriptors"]["bpm"], "key": plan["descriptors"]["key"],
    "vocal": plan["descriptors"]["vocal"],
    "seed": plan["effective"]["seed"], "cot": plan["effective"]["cot"],
    "cfg_scale": plan["effective"]["cfg_scale"],
    "target_seconds": plan["effective"]["target_seconds"],
    "semantic_max_tokens": plan["semantic_max_tokens"],
    "audio_seconds": audio_seconds, "truncated": truncated,
    "audio": meta["audio"]["full"], "rating": plan["descriptors"]["rating"],
    "notes": plan["descriptors"]["notes"], "metrics": {},
})
with open(catalog, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

abs_outdir = Path(os.environ["MS_SCRIPT_DIR"]) / outdir
if stage == "audio":
    audio = abs_outdir / "audio.flac"
    print(f"\nmake-song.sh: song complete: {audio}")
    if audio_seconds is not None:
        print(f"  duration: {audio_seconds:.1f} s ({sample_rate or 48000} Hz stereo), wall time {wall:.0f} s")
    if truncated:
        print("  WARNING: the semantic token cap was hit — the audio is a hard cut, not a natural ending.")
        print("           Raise target_seconds/--max-seconds or shorten the lyrics for a full ending.")
    print(f"  metadata: {abs_outdir / 'metadata.json'}")
else:
    print(f"\nmake-song.sh: plan complete (ABC only, no audio): {abs_outdir}")
    print(f"  score:    {abs_outdir / 'score.abc'}")
    if truncated:
        print("  WARNING: the ABC plan was cut at the token cap — raise abc_sampling.max_tokens via --config.")
    print(f"  metadata: {abs_outdir / 'metadata.json'}")
print(f"  catalog:  {Path(os.environ['MS_SCRIPT_DIR']) / catalog}")
print(f"  full run artifacts (result.json, request.json, plan, latents): {abs_outdir}")
FINALIZE_PY
