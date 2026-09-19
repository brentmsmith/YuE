# SONG_GUIDE — making songs with YuE2 on this Mac

Practical guide to the `make-song.sh` workflow. The installation, MPS fixes,
and model files are **already done and validated** — do not reinstall or
re-benchmark. See `MAC_INSTALL_NOTES.md` / `MAC_MPS_FIXES.md` for that side.

## The exact command you normally use

```bash
cd /Users/brent/musicgen/YuE2/source
./make-song.sh song-requests/blank-template.json        # copy + edit the request first
```

That is the whole workflow: edit a small JSON request, run one command.
`make-song.sh` validates the request, calls the existing `./run-yue2.sh`
launcher unchanged (MPS bf16 generation, per-step NAR `torch.mps.empty_cache()`
flush, CPU VAE decode — all automatic, nothing to configure), refuses to
overwrite an existing song, then prints where the audio is and records
metadata.

Other invocations:

```bash
./make-song.sh my-song.json --max-seconds 60       # cap length (~60 s upper bound)
./make-song.sh my-song.json --seed 99123 --id song-001-b   # quick variant of a request
./make-song.sh my-song.json --stage plan --dry-run  # fast ABC preview, no audio, no generation
./make-song.sh my-song.json --resume               # re-verify a finished run (no regeneration)
./make-song.sh my-song.json --config my-config.json # raw Sampling overrides (advanced)
SONGS_DIR=runs/album1 ./make-song.sh my-song.json   # different output root (default runs/songs)
./run-yue2.sh generate --request file.json --id x --output runs   # the raw CLI underneath
```

## What YuE2 actually controls (and what it does not)

Verified against `src/yue2/protocol.py` / `src/yue2/cli.py` — this is the real
control surface, nothing invented:

| Control | Where | Values / default | Notes |
|---|---|---|---|
| Song/style description | `style` | one comma-separated string | language, genre, instruments, **vocal character**, mood, tempo (`... , 88 BPM`). This is the only place style/genre/vocal-description lives — there is no separate genre or vocal field. `tags` is an accepted alias. |
| Lyrics | `lyrics` | string with `[Verse]`/`[Chorus]`/`[Bridge]` section tags | one line per lyric line, blank line between sections. Length drives song length (see below). |
| Planning mode | `cot` | `full` (default) / `melody` / `off` | `full` = chord-annotated ABC plan, `melody` = melody-only plan, `off` = direct generation (no editable `score.abc`). |
| External composition | `abc` / `abc_path` | ABC text or a file next to the request | pins the melody/harmony; requires `cot: melody` or `full`. Path is resolved relative to the request file. |
| Seed | `seed` | integer in `[0, 2^63)`, default `831001` | groups variants; **not bit-exact on MPS bf16** (same seed can give 697 vs 738 semantic tokens across runs) — treat seed as "same family", not "same audio". |
| Text guidance | `cfg_scale` | number in `[0, 20]`, default `1.0` (auto `1.01` when `cot: off`) | leave default unless you know why you're moving it. |
| Sampling (advanced) | `abc_sampling`, `semantic_sampling` per request, or a `--config` JSON | see defaults below | each stage: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens`, `max_tokens`. |
| Output id | `id` (request) or `--id` | `[A-Za-z0-9._-]+` | output goes to `runs/songs/<id>/`. |
| Stage | `--stage` | `audio` (default) / `plan` | `plan` writes only `score.abc`/`plan.json` — a fast (~20–100 s) preview of melody/harmony/structure without synthesis. |
| Resume | `--resume` | flag | re-verifies a completed run's artifacts against the request; it does **not** continue or regenerate. |

**Not supported by YuE2 generation (do not look for them):** a duration
field, BPM/key fields, a genre field, reference audio, a dedicated
vocal/instrumental switch, negative prompts, or edit intervals. `lang`,
`eval_index`, `clip_id`, `prompt` are accepted-but-ignored metadata keys.
Tempo/key/mood go in the `style` string; duration is controlled indirectly
(lyrics length + semantic token cap); instrumentals via style wording or the
ABC route (below).

**Measured sampling defaults** (`src/yue2/protocol.py`, do not change
casually): semantic stage — temperature `1.0`, top_p `0.95`, top_k `100`,
repetition_penalty `1.2` over a 50-token window, `min_tokens 200`,
`max_tokens 9000`; ABC plan stage — temperature `0.7`, top_p `0.9`, top_k `30`,
repetition_penalty `1.005` over 100, `min_tokens 32`, `max_tokens 4096`.
ODE steps are fixed at 32 (midpoint); context is fixed at 24576.

## Request JSON fields

Real YuE2 fields (sent to the model): `style`, `lyrics`, `cot`, `seed`,
`cfg_scale`, `abc`, `abc_path`, `id`, `abc_sampling`, `semantic_sampling`.

Local bookkeeping fields (stripped before generation; recorded in
`metadata.json` and the catalog for later analytics): `title`, `genres`
(list), `bpm`, `key`, `vocal`, `concept`, `variant`, `notes`, `rating`
(0–10), `target_seconds` (length cap, see below). Any `"_..."` key is a
comment. `make-song.sh` rejects unknown keys with the allowed list, so a
typo in `style`/`lyrics` fails fast instead of generating a surprise.

`make-song.sh` options: `--id`, `--seed`, `--cot`, `--cfg-scale`,
`--max-seconds`, `--config`, `--stage`, `--resume`, `--quiet`, `--dry-run`.
CLI overrides beat request fields; `target_seconds` beats
`semantic_sampling.max_tokens` (pass one, not both).

## Lyrics and song structure

YuE2 reads section tags in the lyrics: `[Verse]`, `[Chorus]`, `[Bridge]`
(also seen in the model's own examples: `[Inst]`-style comments delimit
structure in ABC plans). One lyric line per line, blank line between
sections, as in `examples/song.json` and every file in `song-requests/`.
Keep lines short and singable (7–9 syllables in the reference example).
The language goes in `style` (`English, ...`, `Mandarin, ...`).

**Length control (no duration field exists — two levers):**

1. **Lyrics length drives the natural ending.** Measured on this machine:
   8 lyric lines ≈ 738 semantic tokens ≈ 29.5 s; ~25 semantic tokens per
   second of audio. The model ends the song itself (EOS) when the lyrics are
   done. Want a longer song — write more sections (verse/chorus repeats).
2. **`target_seconds` (request) or `--max-seconds` (flag)** sets
   `semantic.max_tokens ≈ 25 × N` as a hard ceiling (`min_tokens` drops to
   match below 200 if you cap under 8 s). It can cut the song mid-phrase:
   check `result.json` → `truncated.semantic` / the script's WARNING line,
   and raise it or shorten the lyrics.

The 9000-token default cap ≈ 6 minutes. On this MPS machine a ~30 s song
takes ~2 minutes wall time, a 2-minute song took ~18 minutes (NAR attention
cost grows with length) — keep `target_seconds` modest while iterating.

## Instrumental tracks

There is **no instrumental switch**: YuE2 always renders the `lyrics` you
give it. Two real options:

1. **Wordless vocal pads (used in `song-requests/vaporwave-instrumental.json`):**
   describe the sound as instrumental/wordless in `style` and use
   non-lexical lines (`(aah)`, `(ooh)`, `(mmm)`) as lyrics. Style-authentic
   for ambient/vaporwave, and the 12 s smoke run verified it generates
   cleanly (non-silent audio, plan/synthesis/decode all fine) — but the
   model may still voice the pads, so listen before trusting it.
2. **Guaranteed instrumental passages (advanced):** generate `--stage plan`,
   edit `score.abc` (rests in the `Vocal` voice, theme in the `Ins` voice —
   see `skills/yue2-music/references/abc-editing.md`), then regenerate with
   `abc_path` + `cot: melody|full`. This pins the notes, not the timbre.

## Seeds, reruns, and variants

Same request + same seed = same family of outputs, **not** bit-identical on
this machine (MPS bf16 AR stages; two runs at seed 831001 gave 697 vs 738
semantic tokens). To explore a concept:

```bash
cp song-requests/rnb-slow-jam.json my-song.json
# edit my-song.json: keep "concept": "song-001", change "id": "song-001-b", "seed": 202, tweak style/lyrics
./make-song.sh my-song.json
```

The output dir `runs/songs/<id>/` is never overwritten: a second run with the
same id aborts with a warning (use `--id song-001-c`, `--seed N`, or move the
old dir). `--resume` re-checks a finished run. To pre-hear structure cheaply
before spending minutes on synthesis: `--stage plan` (writes `score.abc`
only). For many variants, `./run-yue2.sh batch --input songs.jsonl --output
runs/songs` regenerates per line with one model load (no metadata sidecar —
`make-song.sh` is the tracked path).

## Outputs and the analytics-ready record

Per song: `runs/songs/<id>/` — `audio.flac` (48 kHz stereo),
`score.abc` (when `cot != off`), `plan.json`, `semantic.npy`, `latent.npy`,
`request.json` (the sanitized request YuE2 saved), `config.json`,
`result.json` (timings, truncation, hashes, `audio_seconds`), plus this
workflow's `metadata.json`. One line per song is appended to
`runs/songs/catalog.jsonl`.

`metadata.json` / `catalog.jsonl` carry everything needed for the planned
short-form-content analytics, extensible without schema changes:
`song_id`, `created` (date), `stage`, `status`, `request_file` + full
`request` (prompt used), `descriptors` (`genres`, `bpm`, `key`, `vocal`,
`concept`, `variant`, `title`), `effective` (`cot`, `seed`, `cfg_scale`,
`target_seconds`, `semantic_max_tokens`), `generation` (`audio_seconds`,
`truncated`, wall time, sampling), `audio.full` path, `audio.excerpts` (empty
list now — fill when excerpting later), `notes`, `rating`, `metrics` (empty
object now — add `tiktok_views`, `shorts_views`, `uses`, `likes`, `shares`,
`retention` per platform later). Variants share `concept` with distinct
`id`/`variant`/`seed`. Nothing in YuE2 is modified by this bookkeeping.

## Examples in `song-requests/`

| File | What it shows |
|---|---|
| `rnb-slow-jam.json` | atmospheric late-80s R&B slow jam, verse/chorus/bridge/chorus structure |
| `vaporwave-instrumental.json` | instrumental-style approach (wordless pads), length cap, caveat notes |
| `piano-nocturne.json` | classical-inspired piano ballad, rubato/minor-key style wording |
| `atmospheric-metal.json` | heavy atmospheric metal, clean-vs-harsh vocal contrast described in `style` |
| `blank-template.json` | copy for every new song; `_comment_*` keys explain the fields |

Run any example (each ~1–4 minutes wall time on this Mac):

```bash
./make-song.sh song-requests/rnb-slow-jam.json
```

Verified end-to-end on this machine: `runs/songs/vaporwave-instrumental-smoke/`
(12.0 s audio in 104 s wall, `--max-seconds 12`, truncation warning and
metadata/catalog all correct), plus `--resume` (identity re-verification) and
`--stage plan` (ABC-only, no audio) paths.

## MPS notes that still matter

- The two `[YuE2] MPS NAR cache flush` / `MPS NAR driver memory` log lines
  per run are expected — that is the validated protection working.
- VAE decode runs on CPU by design (MPS conv1d limit); nothing to fix.
- `runs/songs/<id>/result.json` `graph_fallback_reason:
  "non_cuda_device"` is expected on MPS.
- Long songs (≥ ~1 min) hold ~8–12 GB; close heavy apps during runs.
- If a generation dies, the partial dir is kept with `failure.json`;
  rerun into a fresh `--id` or move the broken dir aside.
