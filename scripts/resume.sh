#!/usr/bin/env bash
#
# A fresh shell after a reboot, put back together.
#
#   scripts/resume.sh [run_batch.sh's arguments...]
#
#     scripts/resume.sh --list "$A/data/examples.txt" --output "$OUT"
#
# Everything the pipeline needs is on disk under the checkout -- the venv, the
# weights, the output -- so a reboot costs the environment variables and nothing
# else, and `run_batch.sh` sets all four of those itself.  Strictly, this script
# is optional.
#
# What it adds is the checking.  A machine that came back up is not the same
# thing as a machine that is ready, and the three failures worth catching here
# are the quiet ones:
#
#   * an interpreter that no longer runs -- the venv is files, but the python it
#     points at does not have to still be where it was
#   * the VGGish seed, which `run_batch.sh` looks for under the checkout while a
#     seed written before that change sits in $HOME and is invisible.  Its
#     absence is not a slow start: the fallback is a 275 MB fetch from GitHub,
#     which a server like this one cannot make
#   * a stage whose dependencies this machine no longer satisfies
#
# It fixes the first two where it can and then hands the arguments to
# run_batch.sh, so it is a drop-in for it.
#
# What it does NOT need to do is clean up after a run the reboot interrupted.
# Every stage writes through a temporary file and records what it wrote, so a
# half-written artifact is detected as changed and re-run.  Re-running is always
# safe.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 2; }

say "checkout   $ROOT"

# --------------------------------------------------------------------------- #
# the interpreter
# --------------------------------------------------------------------------- #

[[ -x "$PYTHON" ]] || die "no interpreter at $PYTHON -- run scripts/setup_venv.sh"

# `--version` rather than `[[ -x ]]` alone: an executable file is not a working
# interpreter, and the difference shows up as a confusing failure inside the
# first stage rather than here.
version="$("$PYTHON" --version 2>&1)" \
    || die "$PYTHON exists but does not run (${version:-no output}); run scripts/setup_venv.sh"
say "python     $version"

# --------------------------------------------------------------------------- #
# the VGGish seed, at the path the run will actually look in
# --------------------------------------------------------------------------- #

export TORCH_HOME="${TORCH_HOME:-$ROOT/models/torch}"
VGGISH="$TORCH_HOME/hub/checkpoints/vggish-10086976.pth"
if [[ -f "$VGGISH" ]]; then
    say "vggish     seeded, $(( $(wc -c < "$VGGISH") / 1024 / 1024 )) MB"
else
    warn "vggish     missing at $VGGISH"
    "$PYTHON" "$ROOT/scripts/seed_vggish.py" \
        || die "could not seed it -- S5 would try a 275 MB download from GitHub"
fi

# --------------------------------------------------------------------------- #
# can this machine run the stages at all
# --------------------------------------------------------------------------- #

printf '\n'
if ! "$PYTHON" -m avannotate.cli doctor --configs-dir "$ROOT/configs"; then
    printf '\n'
    warn "the report above names what is missing, and what to run about each."
    exit 1
fi

# --------------------------------------------------------------------------- #
# hand off
# --------------------------------------------------------------------------- #

if [[ $# -eq 0 ]]; then
    printf '\n'
    say "ready.  Nothing to run -- pass run_batch.sh's arguments, for example:"
    say "  $0 --list $ROOT/data/examples.txt --output <DIR>"
    exit 0
fi

# The common case is the checkout's own corpus, whose list file names its videos
# relative to the checkout -- so a missing --data means $ROOT rather than an
# error about a video it cannot find.
have_data=0
for argument in "$@"; do
    [[ "$argument" == "--data" ]] && have_data=1
done
if (( ! have_data )); then
    set -- --data "$ROOT" "$@"
    say "no --data given; using $ROOT"
fi

printf '\n'
exec "$ROOT/scripts/run_batch.sh" "$@"
