#!/usr/bin/env bash
#
# The whole corpus, one stage at a time.
#
#   scripts/run_corpus.sh --data DIR --list FILE.txt --output DIR
#
#   --data DIR       where the videos are, if the list names them relatively
#   --list FILE      a text file, one video per line, '#' for comments
#   --output DIR     where the annotations go
#   --stages LIST    comma-separated subset, run in the order given
#   --from STAGE     start here, skipping the stages that come before it
#   --gpus LIST      comma-separated device indices (default: detect them)
#   --workers N      videos at once (default: one per GPU, or 1 with none)
#   --force          re-run each stage even where its outputs are current
#   --dry-run        print the plan and stop
#
# `run_batch.sh` runs a video end to end, then the next video.  This runs S1
# over every video, then S2 over every video.  Same stages, same order within a
# video, same outputs -- the difference is *when each model is built*.
#
# Every stage builds its model inside `run()`, and `run()` is called once per
# video.  So a video-major batch over a thousand videos builds the captioner a
# thousand times, and the captioner is 16 GB read off a disk.  Stage-major
# alone would not fix that -- each video would still build its own -- which is
# what `avannotate.model_cache` is for: a single-slot per-process cache keyed by
# (stage, config).  The two together are what make the model load once per
# *stage per worker* instead of once per video.
#
# The cache holds one model, so this only works stage-major: a video-major run
# cycles through all ten stages' models inside each video and the cache would
# be a no-op.  The memory this saves is the point.  The ten models together do
# not fit on one card, so the alternative to "one at a time" is not "all at
# once", it is OOM.
#
# Resume works the same as it always did: every stage records its version and
# its input hash and skips when they are unchanged.  So re-running this script
# after it stops does only the work that is actually left, and `--from` is
# merely a convenience for skipping the check on the stages you know are done.
#
# The environment -- HF_HOME, TORCH_HOME, the clearvoice checkpoint link -- is
# set up by `run_batch.sh`, and this calls it rather than repeating it.  A
# second copy of that setup would be a second thing to get wrong, and the way
# it is wrong is silent: a stage that finds no weights does not stop, it goes
# to the network.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${PYTHON_FALLBACK:-python3}"; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA=""
LIST=""
OUTPUT=""
STAGES=""
FROM=""
GPUS=""
WORKERS=""
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --list) LIST="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 2; }

[[ -n "$LIST" ]] || die "--list is required"
[[ -n "$OUTPUT" ]] || die "--output is required"
[[ -f "$LIST" ]] || die "no list file at $LIST"
[[ -z "$DATA" || -d "$DATA" ]] || die "no data directory at $DATA"

# The stage list comes from the package rather than from a literal here.  A
# hardcoded list is a list that goes stale the first time a stage is added or
# renamed, and the way it goes stale is that the new stage simply never runs --
# which leaves every video missing a file that nothing complains about until
# something downstream reads it.
plan=()
if [[ -n "$STAGES" ]]; then
    IFS=',' read -r -a plan <<< "$STAGES"
else
    # Not `mapfile`: this has to run on the cluster's bash 5 and on a laptop's
    # bash 3.2, and `mapfile: command not found` is a poor way to find out.
    while IFS= read -r name; do
        [[ -n "$name" ]] && plan+=("$name")
    done < <("$PYTHON" -m avannotate.cli stages | awk '$1 == "implemented" {print $2}')
fi
[[ ${#plan[@]} -gt 0 ]] || die "no implemented stages; is the package importable?"

if [[ -n "$FROM" ]]; then
    for i in "${!plan[@]}"; do
        if [[ "${plan[$i]}" == "$FROM" ]]; then
            plan=("${plan[@]:$i}")
            break
        fi
    done
    [[ "${plan[0]}" == "$FROM" ]] || die "--from $FROM is not one of: ${plan[*]}"
fi

mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

printf '\033[1mavannotate — stage-major\033[0m  %d stages, %s at a time\n' \
    "${#plan[@]}" "${WORKERS:+$WORKERS videos}${WORKERS:-one per GPU}"
for i in "${!plan[@]}"; do
    printf '  %2d/%d  %s\n' "$((i + 1))" "${#plan[@]}" "${plan[$i]}"
done
printf '  logs      %s/\n' "$LOGS"
printf '  combined  %s/run.log\n\n' "$OUTPUT"

if (( DRY_RUN )); then
    printf 'dry run, nothing started\n'
    exit 0
fi

# Appended, not truncated: the whole point of this file is to outlive the
# per-stage logs of a run that was restarted.  The per-stage logs themselves
# are truncated, so a stage that is re-run replaces its own log rather than
# doubling it -- and the run.log keeps the earlier attempt.
COMBINED="$OUTPUT/run.log"
printf '\n===== run started %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$COMBINED"

declare -a ledger=()
aborted=""
aborted_stage=""
started_all=$(date +%s)

for i in "${!plan[@]}"; do
    stage="${plan[$i]}"
    log="$LOGS/$stage.log"
    printf '\033[1m===== %s (%d/%d) =====\033[0m  %s\n' \
        "$stage" "$((i + 1))" "${#plan[@]}" "$(date '+%H:%M:%S')"

    args=(--list "$LIST" --output "$OUTPUT" --stages "$stage")
    [[ -n "$DATA" ]] && args+=(--data "$DATA")
    [[ -n "$GPUS" ]] && args+=(--gpus "$GPUS")
    [[ -n "$WORKERS" ]] && args+=(--workers "$WORKERS")
    (( FORCE )) && args+=(--force)

    started=$(date +%s)
    # tee: live to the terminal, kept in the per-stage log.  `set +e` because a
    # stage where some videos failed must not stop the loop -- those videos are
    # already recorded in failures.jsonl, and stopping here would hold the
    # whole corpus at a stage because three videos could not get through it.
    set +e
    "$ROOT/scripts/run_batch.sh" "${args[@]}" 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    set -e
    elapsed=$(( $(date +%s) - started ))

    cat "$log" >> "$COMBINED"

    # Deliberately not `set -e`'s business: a nonzero status here means the
    # batch itself fell over, which is worth saying out loud and worth
    # continuing past, because the next stage's per-video skip will simply
    # decline to run for the videos that have no input.
    ok=""
    failed=""
    read -r ok failed < <(
        awk '/^videos +[0-9]+ ok, [0-9]+ failed$/ {o = $2; f = $4} END {if (o != "") print o, f}' "$log"
    ) || true

    if [[ -n "$ok" ]]; then
        ledger+=("$(printf '%-18s %6s ok %6s failed  %6ss' "$stage" "$ok" "$failed" "$elapsed")")
        printf '          %s: %s ok, %s failed in %ss\n\n' "$stage" "$ok" "$failed" "$elapsed"
        # A stage that fails on every video is not one bad video, it is a broken
        # stage -- a missing checkpoint, a dependency that is not installed, the
        # class of failure that produced `utt=0` for days.  Carrying on would
        # spend the rest of the night running stages over videos that have no
        # input, and each would report the same failure at its own stage.
        if [[ "$ok" == "0" && "$failed" != "0" ]]; then
            aborted="$stage failed for every video"
            aborted_stage="$stage"
            break
        fi
    else
        ledger+=("$(printf '%-18s %s' "$stage" "no summary (exit $status)")")
        printf '          %s: no summary, exit %s\n\n' "$stage" "$status"
        if (( status != 0 )); then
            aborted="$stage exited $status before reporting"
            aborted_stage="$stage"
            break
        fi
    fi
done

total=$(( $(date +%s) - started_all ))

printf '\033[1m===== summary =====\033[0m  %s\n' "$(date '+%H:%M:%S')"
for line in "${ledger[@]}"; do printf '  %s\n' "$line"; done
printf '  %-18s %s\n' "total" "$(printf '%dh%02dm%02ds' $((total / 3600)) $((total % 3600 / 60)) $((total % 60)))"

if [[ -n "$aborted" ]]; then
    printf '\n\033[31mstopped: %s\033[0m\n' "$aborted" >&2
    printf '  log: %s/%s.log\n' "$LOGS" "$aborted_stage" >&2
    printf '  fix it, then re-run the same command -- finished stages skip.\n' >&2
    exit 1
fi

printf '\ndeliverables in %s/work/*/s11-compose/\n' "$OUTPUT"
printf 'failures, if any, in %s/failures.jsonl (last stage to run)\n' "$OUTPUT"
