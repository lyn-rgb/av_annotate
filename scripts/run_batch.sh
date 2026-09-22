#!/usr/bin/env bash
#
# One command from a folder of videos to a folder of annotations.
#
#   scripts/run_batch.sh --data DIR --list FILE.txt --output DIR
#
#   --data DIR       where the videos are, if the list names them relatively
#   --list FILE      a text file, one video per line, '#' for comments
#   --output DIR     where the annotations go
#   --stages LIST    comma-separated, e.g. s0-preprocess,s8-asr (default: all)
#   --gpus LIST      comma-separated device indices (default: detect them)
#   --workers N      videos at once (default: one per GPU, or 1 with none)
#   --only-missing   skip videos that already have a deliverable
#   --verbose        a line per stage rather than per video
#
# This is a thin wrapper.  The work is `avannotate batch`, which is where the
# GPU detection, the pool and the progress reporting live -- a shell script
# cannot do those well, and a second implementation of them would be a second
# set of bugs.
#
# The log goes to both the terminal and $OUTPUT/batch.log, because a batch that
# runs overnight is usually read the next morning.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${PYTHON_FALLBACK:-python3}"; fi

# This script changes directory to the data root, so the package has to be
# importable from anywhere.  Setting PYTHONPATH means a checkout works without
# being installed; `pip install -e .` works too, and this is harmless alongside
# it.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Where the weights are.  download_models.sh puts them under the checkout, and
# the configs name their checkpoints by repo id -- so this is the difference
# between a run finding 26 GB of already-downloaded models and huggingface_hub
# looking in ~/.cache/huggingface, finding nothing, and either failing or
# fetching all of it again.  `${VAR:-}` so an operator who set these meant it.
export HF_HOME="${HF_HOME:-$ROOT/models/hf}"
# funasr reads emotion2vec from ModelScope's own cache, which is the other
# convention in this pipeline and the other place a path has to agree.
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/models/modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# torch's hub cache, which holds the seeded VGGish weights.  Under the checkout
# for the same reason as the two above -- a cluster's $HOME is on a quota and
# its /tmp is on the node's own disk, and neither is where a checkout that
# already has a models/ directory should be reaching.
#
# This one is load-bearing in a way the others are not.  setup_venv.sh seeds
# ``vggish-10086976.pth`` into whatever TORCH_HOME says, so if this disagreed
# with that, the seed would land somewhere S5 never looks -- and S5 does not
# degrade when it cannot find it, it asks torch.hub to download 275 MB from
# GitHub.  On a server that cannot reach GitHub that is a hard failure, and the
# weights it was fetching are ones the LoCoNet checkpoint overwrites anyway.
export TORCH_HOME="${TORCH_HOME:-$ROOT/models/torch}"

# HF_HUB_OFFLINE is deliberately NOT set here.  It is what makes a cached
# checkpoint resolve without asking the network, so on a server with no route to
# huggingface.co it is required -- but it also turns every miss into a hard
# failure rather than a download, and that is a decision rather than a default.
# Set it on such a server:
#
#     export HF_HUB_OFFLINE=1

DATA=""
LIST=""
OUTPUT=""
STAGES=""
GPUS=""
WORKERS=""
ONLY_MISSING=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --list) LIST="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --only-missing) ONLY_MISSING=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 2; }

[[ -n "$LIST" ]] || die "--list is required"
[[ -n "$OUTPUT" ]] || die "--output is required"
[[ -f "$LIST" ]] || die "no list file at $LIST"

# The videos are usually named relative to the data root, but the list is
# often written relative to itself.  Running from the data root makes the first
# convention work; the CLI resolves the second on its own.
if [[ -n "$DATA" ]]; then
    [[ -d "$DATA" ]] || die "no data directory at $DATA"
    cd "$DATA"
fi

# S7's extractor reads its checkpoint from a path relative to the *working
# directory* -- hardcoded in clearvoice, with no argument to override it.  This
# script is what chooses the working directory, so it is what has to make that
# path exist; otherwise S7 either fails or, worse, tries to download the
# checkpoint, which on a machine with no route to Hugging Face cannot succeed
# and on one that has a route would quietly fetch a second copy.
#
# Two names, because clearvoice has used both.  The build this was run against
# wants `checkpoints/` -- its own error says so, verbatim:
#     FileNotFoundError: 'checkpoints/AV_MossFormer2_TSE_16K/last_best_checkpoint.pt'
# -- while the 0.1.2 source on PyPI says `checkpoint_dir/`.  A link under each
# name is cheaper than working out which, and harmless either way.
#
# The `-L` test is the part that was learned the hard way: `ln -sfn` aimed at a
# path that is already a directory -- which is what an interrupted download
# leaves behind -- puts the link *inside* it rather than replacing it, and the
# file it was meant to provide is then still missing.  A real directory there is
# stale by definition, so it is moved aside for inspection rather than deleted.
for name in checkpoints checkpoint_dir; do
    [[ -d "$ROOT/models/clearvoice/AV_MossFormer2_TSE_16K" ]] || continue
    link="$name/AV_MossFormer2_TSE_16K"
    if [[ -d "$link" && ! -L "$link" ]]; then
        mv "$link" "$link.partial"
        printf '   moved a stale %s aside (an interrupted download, most likely)\n' "$link"
    fi
    if [[ ! -e "$link" ]]; then
        mkdir -p "$name"
        ln -s "$ROOT/models/clearvoice/AV_MossFormer2_TSE_16K" "$link"
        printf '   S7 checkpoint linked at %s/%s\n' "$(pwd)" "$link"
    fi
done

mkdir -p "$OUTPUT"
LOG="$(cd "$OUTPUT" && pwd)/batch.log"
LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"

ARGS=(batch --input "$LIST" --output "$OUTPUT")
if [[ -n "$STAGES" ]]; then
    IFS=',' read -r -a _stages <<< "$STAGES"
    for stage in "${_stages[@]}"; do ARGS+=(--stage "$stage"); done
fi
[[ -n "$GPUS" ]] && ARGS+=(--gpus "$GPUS")
[[ -n "$WORKERS" ]] && ARGS+=(--workers "$WORKERS")
(( ONLY_MISSING )) && ARGS+=(--only-missing)
(( VERBOSE )) && ARGS+=(--verbose)

printf '\033[1mavannotate batch\033[0m  log: %s\n\n' "$LOG"

# tee, and the exit code of the command rather than of tee: with `set -o
# pipefail` alone the pipeline reports the right status, but the explicit
# capture makes it obvious that a failure downstream still fails the script.
set +e
"$PYTHON" -m avannotate.cli "${ARGS[@]}" 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e

if (( status == 0 )); then
    printf '\n\033[32mdone\033[0m  deliverables in %s/work/*/s11-compose/\n' "$OUTPUT"
else
    printf '\n\033[33msome videos failed\033[0m  see %s\n' "$OUTPUT/failures.jsonl" >&2
fi
exit "$status"
