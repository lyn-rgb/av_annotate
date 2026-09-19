#!/usr/bin/env bash
#
# One command to build the virtual environment on a GPU server and put every
# package in it.
#
#   scripts/setup_venv.sh [--python PATH] [--venv DIR] [--extras LIST] [--dev]
#                         [--index URL] [--torch-index URL] [--gpus N]
#
#   --python PATH     interpreter to build the venv from (default: the best of
#                     the ones on PATH; see "which Python" below)
#   --venv DIR        where to put it (default: ./.venv, which is what
#                     run_batch.sh and download_models.sh look for)
#   --extras LIST     comma-separated, default every runtime extra
#   --dev             also install pytest/ruff/mypy
#   --index URL       pip index (default: probed -- see lib.sh)
#   --torch-index URL install torch from a specific PyTorch index instead
#   --gpus N          how many GPUs this machine is supposed to have; checked
#                     at the end rather than assumed (default: report only)
#
# What this does NOT do: it does not fetch model weights (download_models.sh),
# does not clone the two repositories (setup_server.sh), and does not touch
# CUDA, cuDNN or the driver.  It builds an environment and checks that the
# environment can see the hardware; anything wrong below that is a machine
# problem, and the checks at the end say which.
#
# It is idempotent.  Re-running it installs what is missing and leaves the rest
# alone, so it is safe to run again after adding an extra or fixing a mirror.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The proxy and the pip index this machine should use.  See lib.sh -- which
# index is right is a property of the machine, not of this project.
# shellcheck source=lib.sh
source "$ROOT/scripts/lib.sh"

VENV="$ROOT/.venv"
PYTHON=""
EXTRAS="media,faces,diarization,asd,tse,asr,paralinguistic,caption"
WANT_DEV=0
TORCH_INDEX=""
EXPECT_GPUS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --extras) EXTRAS="$2"; shift 2 ;;
        --dev) WANT_DEV=1; shift ;;
        --index) PIP_INDEX="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        --gpus) EXPECT_GPUS="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# --------------------------------------------------------------------------- #
# which Python
# --------------------------------------------------------------------------- #
#
# 3.10 and 3.11 first, and that ordering is not arbitrary.  The project's
# requires-python is >=3.10 and its development venv is 3.10.15, so 3.10 is the
# version anything here has actually been run against.  3.11 and 3.12 are
# expected to work.  3.13 is where the compiled dependencies in this list --
# funasr, clearvoice, onnxruntime-gpu -- start not having wheels yet, and a
# source build of any of them is a much longer afternoon than passing --python.
#
# Nothing here reads the interpreter's patch version: a floor is a floor.

pick_python() {
    if [[ -n "$PYTHON" ]]; then
        [[ -x "$PYTHON" ]] || die "--python $PYTHON is not executable"
        return 0
    fi
    local candidate
    for candidate in python3.10 python3.11 python3.12; do
        if have "$candidate"; then PYTHON="$(command -v "$candidate")"; return 0; fi
    done
    if have python3; then
        PYTHON="$(command -v python3)"
        warn "none of python3.10/3.11/3.12 is on PATH; falling back to"
        warn "    $PYTHON ($("$PYTHON" -V 2>&1))"
        warn "which is a version this project has not been built against.  If"
        warn "the install fails on a compiled dependency, that is why -- install"
        warn "python3.10 or 3.11 and re-run with --python /path/to/it."
        return 0
    fi
    die "no python3 on PATH.  Install one (apt install python3.10 python3.10-venv) and re-run."
}

pick_python
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PYVER" in
    3.10|3.11|3.12) ;;
    *) warn "Python $PYVER is outside the tested range (3.10-3.12); continuing anyway" ;;
esac
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "$PYTHON is $PYVER, and this project declares requires-python >=3.10"

# Always ends up non-empty: the probe picks PyPI or a domestic mirror, and
# --index overrides both.  That is why every install below passes
# --index-url unconditionally -- the alternative is a `${PIP_INDEX:+...}`
# argument built inside an expansion, where the quotes are not quotes.
detect_pip_index
note "pip index:   $PIP_INDEX"

# --------------------------------------------------------------------------- #
# the venv
# --------------------------------------------------------------------------- #

say "virtual environment"
note "interpreter: $PYTHON ($PYVER)"
note "location:    $VENV"

if [[ -x "$VENV/bin/python" ]]; then
    note "already exists; reusing it"
else
    # `python -m venv` needs the venv module, which Debian and Ubuntu ship in a
    # separate package.  The stock error for that is a wall of text about
    # ensurepip that does not name the package, so it is named here.
    if ! "$PYTHON" -m venv "$VENV" 2>/tmp/venv_err.$$; then
        sed 's/^/   /' /tmp/venv_err.$$ >&2
        rm -f /tmp/venv_err.$$
        die "could not create the venv. On Debian/Ubuntu this is usually the missing
'venv' module, which is a separate package:
    apt-get install -y python$PYVER-venv
On RHEL/Rocky it is part of python3.  Then re-run this script."
    fi
    rm -f /tmp/venv_err.$$
    note "created"
fi

PY="$VENV/bin/python"
PIP=("$PY" -m pip)

say "pip, setuptools, wheel"
# --upgrade first: a venv created from an older interpreter ships a pip that
# cannot read the metadata of some modern wheels, and the failure it produces
# looks like the package not existing.
"${PIP[@]}" install --quiet --upgrade pip setuptools wheel \
    --index-url "$PIP_INDEX" \
    || die "could not upgrade pip; check the index ($PIP_INDEX) and the network"
note "pip $("$PY" -m pip --version | awk '{print $2}')"

# --------------------------------------------------------------------------- #
# PyTorch, before anything that depends on it
# --------------------------------------------------------------------------- #
#
# Which extras drag torch in.  Checked rather than assumed, because the
# pipeline deliberately keeps most stages off it -- S8 is CTranslate2 only, and
# downloading two and a half gigabytes of CUDA libraries for a run that never
# calls them is the kind of thing that turns a ten-minute setup into an hour.
needs_torch() {
    local extra
    for extra in ${EXTRAS//,/ }; do
        case "$extra" in
            diarization|asd|paralinguistic|tse) return 0 ;;
        esac
    done
    return 1
}

if needs_torch; then
    # Installed on its own and first, for two reasons.  On Linux the PyPI wheel
    # is the CUDA build, so `pip install torch` alone is correct and the
    # CPU-only trap is a Windows one -- but installing it explicitly means that
    # if it ever does resolve to a CPU build, that is visible here rather than
    # as a confusing "no CUDA device" three stages later.  And the packages
    # below (pyannote, funasr, clearvoice) all depend on torch, so resolving it
    # once, first, keeps pip from reconsidering it while resolving them.
    say "PyTorch"
    if [[ -n "$TORCH_INDEX" ]]; then
        note "index: $TORCH_INDEX"
        "${PIP[@]}" install --upgrade torch torchaudio --index-url "$TORCH_INDEX" \
            || die "installing torch from $TORCH_INDEX failed"
    else
        note "index: $PIP_INDEX"
        "${PIP[@]}" install --upgrade torch torchaudio \
            --index-url "$PIP_INDEX" \
            || die "installing torch failed"
    fi
else
    say "PyTorch"
    note "skipped: none of [$EXTRAS] needs it"
fi

# --------------------------------------------------------------------------- #
# the project, and every extra
# --------------------------------------------------------------------------- #

REQUIREMENT="$ROOT[$EXTRAS]"
if [[ "$WANT_DEV" == 1 ]]; then
    REQUIREMENT="$ROOT[$EXTRAS,dev]"
fi

say "the pipeline and its dependencies"
note "$REQUIREMENT"
note "this is the long step -- several gigabytes, most of it torch's CUDA"
note "libraries when they are not already present"
"${PIP[@]}" install --upgrade "$REQUIREMENT" \
    --index-url "$PIP_INDEX" \
    || die "installing the pipeline dependencies failed.
The usual cause on a machine that can reach nothing abroad is the index:
    --index URL     (currently $PIP_INDEX)
and on one that can reach nothing at all, a wheelhouse:
    --index is not enough; see setup_server.sh --from-bundle"

# --------------------------------------------------------------------------- #
# DiariZen, which is not a package
# --------------------------------------------------------------------------- #

say "DiariZen (S4)"
DIARIZEN="$ROOT/models/DiariZen"
if [[ ! -d "$DIARIZEN" ]]; then
    warn "no checkout at $DIARIZEN, so it is not installed and S4 cannot run."
    warn "It is a repository rather than a package, and it is not on PyPI:"
    warn "    scripts/setup_server.sh --stages s4-diarize"
    warn "will clone and install it when the machine can reach GitHub, and"
    warn "docs/server-setup.md covers the case where it cannot."
elif ! "$PY" -c "import diarizen" >/dev/null 2>&1; then
    note "installing from $DIARIZEN"
    # Its own order: requirements, then the package, then the vendored
    # pyannote-audio -- which is a modified copy, so installing the PyPI one
    # instead is not a substitute.
    ( cd "$DIARIZEN" && "${PIP[@]}" install -r requirements.txt \
        && "${PIP[@]}" install -e . ) \
        || warn "DiariZen's own requirements did not install; S4 may still import"
    if [[ -d "$DIARIZEN/pyannote-audio" ]]; then
        ( cd "$DIARIZEN/pyannote-audio" && "${PIP[@]}" install -e . ) \
            || warn "the vendored pyannote-audio did not install"
    fi
    "$PY" -c "from diarizen.pipelines.inference import DiariZenPipeline" >/dev/null 2>&1 \
        && note "DiariZenPipeline imports" \
        || warn "DiariZen is installed but does not import; S4 will fail at run time"
else
    note "already importable"
fi

# --------------------------------------------------------------------------- #
# can this environment see the hardware
# --------------------------------------------------------------------------- #
#
# The point of the script is not that pip exited zero -- it is that a batch run
# on four cards will not fall back to the CPU at the first stage.  So the checks
# are the ones that would catch that quietly happening.

say "checking the hardware this environment can see"

"$PY" - <<'PYEOF' || true
import sys

try:
    import torch
except ModuleNotFoundError:
    print("   torch does not import -- nothing below can be checked")
    sys.exit(0)

print(f"   torch {torch.__version__}  (built against CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("   CUDA NOT AVAILABLE.")
    print("   torch is installed but cannot see a device.  In order of how")
    print("   often it is the answer: a CPU-only wheel; a driver older than")
    print("   the CUDA this torch was built for; or no GPU exposed to this")
    print("   container (check `nvidia-smi` outside it).")
    sys.exit(0)

count = torch.cuda.device_count()
print(f"   {count} device(s):")
for index in range(count):
    free, total = torch.cuda.mem_get_info(index)
    print(f"     [{index}] {torch.cuda.get_device_name(index)}"
          f"  {total / 1024**3:.0f} GB total, {free / 1024**3:.0f} GB free")
    # The compute capability decides whether a prebuilt wheel has kernels for
    # this card at all.  Ada is 8.9, so anything built for CUDA >= 11.8 covers
    # it; a wheel built only for older architectures runs but slowly, and the
    # way that shows up is a batch that is mysteriously 5x off the estimate.
    major, minor = torch.cuda.get_device_capability(index)
    print(f"         compute capability {major}.{minor}, "
          f"kernels: {torch.cuda.get_arch_list()[:4]}{'...' if len(torch.cuda.get_arch_list()) > 4 else ''}")
PYEOF

# onnxruntime-gpu carries its own CUDA/cuDNN requirement and is the single most
# common thing to be silently unusable on an otherwise fine GPU box: it falls
# back to the CPU provider without failing, so insightface (S1, S3) runs about
# ten times slower and nothing says so.
"$PY" - <<'PYEOF' || true
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    print("   onnxruntime is not installed (S1/S3 need it: --extras faces)")
    raise SystemExit(0)

available = ort.get_available_providers()
print(f"   onnxruntime {ort.__version__}, providers: {available}")
if "CUDAExecutionProvider" not in available:
    print("   CUDAExecutionProvider is NOT available -- insightface will run")
    print("   on the CPU.  onnxruntime-gpu wants a cuDNN matching the CUDA it")
    print("   was built for; `pip install onnxruntime-gpu` without it gives a")
    print("   working install that quietly uses the CPU.")
    raise SystemExit(0)
print(f"   -> {ort.get_device()}")
PYEOF

if (( EXPECT_GPUS > 0 )); then
    found="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$found" == "$EXPECT_GPUS" ]]; then
        note "$found GPU(s), as expected"
    else
        warn "expected $EXPECT_GPUS GPUs and nvidia-smi lists ${found:-0}"
    fi
fi

# --------------------------------------------------------------------------- #
# what the pipeline itself makes of it
# --------------------------------------------------------------------------- #

say "the pipeline's own check"
HF_HOME="${HF_HOME:-$ROOT/models/hf}" "$PY" -m avannotate.cli doctor \
    --configs-dir "$ROOT/configs" 2>&1 | tail -20 || true

cat <<EOF

The environment is at:

    $VENV

and the scripts that use it (run_batch.sh, download_models.sh) look there by
default, so nothing needs activating.  To use it by hand:

    source "$VENV/bin/activate"

What is still missing is almost certainly not a package -- check the doctor
output above.  The usual remainder is model weights (download_models.sh) and
the two repository checkouts (setup_server.sh), neither of which is this
script's job.

EOF
