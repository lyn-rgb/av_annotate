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

# Put pip into a venv that was created without it.
#
# Three routes, tried in order, because each has a precondition the others do
# not.  The first is the one that matters on a cluster: the base interpreter
# was chosen precisely because it has pip, and pip can be told to act on
# another environment -- which needs pip >= 22.3, old enough now that anything
# still shipping an older one is the exception.
bootstrap_pip() {
    if "$PYTHON" -m pip --python "$VENV/bin/python" install --upgrade pip setuptools wheel \
        >/dev/null 2>&1; then
        note "pip taken from the base interpreter"
        return 0
    fi

    # The official bootstrap script.  Needs a route to PyPI, which is exactly
    # what a locked-down cluster may not have -- but on a machine that has one,
    # it is the most reliable of the three.
    local script="/tmp/get-pip.$$"
    if curl -fsSL --connect-timeout 20 --max-time 120 \
        https://bootstrap.pypa.io/get-pip.py -o "$script" 2>/dev/null; then
        if "$VENV/bin/python" "$script" >/dev/null 2>&1; then
            rm -f "$script"
            note "pip bootstrapped with get-pip.py"
            return 0
        fi
    fi
    rm -f "$script"

    # virtualenv brings its own pip wheels, so it needs no network at all.
    if have virtualenv && virtualenv -p "$PYTHON" --clear "$VENV" >/dev/null 2>&1; then
        note "pip provided by virtualenv"
        return 0
    fi
    return 1
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

# An existing venv is reused only if its interpreter actually runs.
#
# The case that matters here is a transferred one.  A .venv built on a laptop is
# a directory of Mach-O binaries; `rsync -a` preserves the executable bit, so on
# a Linux server `-x` is true and the file cannot be executed at all.  Checking
# only `-x` therefore reuses a venv that cannot run, and the failures that
# follow -- pip exiting with "exec format error", or run_batch.sh falling back to
# the system python and reporting every package missing -- read as something
# else entirely.
VENV_USABLE=0
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c 'pass' >/dev/null 2>&1; then
    VENV_USABLE=1
    note "already exists; reusing it"
elif [[ -e "$VENV" ]]; then
    # Moved aside, not deleted.  The directory is somebody's, it may contain
    # more than this project put there, and the name says what happened so it
    # is obvious later what it is and that it is safe to remove.
    ASIDE="$VENV.broken.$(date +%Y%m%d%H%M%S)"
    warn "$VENV exists, but its python does not run on this machine."
    warn "That is what a venv copied from another OS looks like: the file is"
    warn "still marked executable, so it is only obvious when you try it."
    warn "Moving it to $(basename "$ASIDE") and building a fresh one."
    mv "$VENV" "$ASIDE"
fi

if (( VENV_USABLE == 0 )); then
    VENV_ERR="/tmp/venv_err.$$"
    if "$PYTHON" -m venv "$VENV" 2>"$VENV_ERR"; then
        note "created"
    else
        printf '   the standard route failed; its output was:\n' >&2
        sed 's/^/     /' "$VENV_ERR" >&2

        # Nearly every failure here is ensurepip.  `venv` can make the
        # directory; it cannot put pip in it, because Debian, Ubuntu and most
        # HPC images ship pip's bootstrap wheels separately from the
        # interpreter -- and on a cluster there is no root to install them
        # with, so "apt-get install python3.X-venv" is advice nobody can take.
        #
        # The interpreter is not broken.  It just needs pip from somewhere
        # else, and there are three ways to get it.
        if ! grep -qi 'ensurepip\|Failing command' "$VENV_ERR"; then
            rm -f "$VENV_ERR"
            die "could not create the venv, and not for the usual reason -- the
output above is not about ensurepip.  Check the interpreter:
    $PYTHON -m venv /tmp/probe-venv"
        fi

        note ""
        note "that is ensurepip, which this machine's Python does not ship."
        note "The interpreter itself is fine, so this builds the venv without"
        note "pip and then gets pip by another route."
        rm -f "$VENV_ERR"

        # --clear rather than rm -rf: the directory was made moments ago by the
        # attempt above, and this is the venv module's own way to empty it.
        "$PYTHON" -m venv --without-pip --clear "$VENV" \
            || die "even 'venv --without-pip' failed; this interpreter cannot make
virtual environments at all.  On a cluster the usual answers are a python
module (module avail python; module load python/3.11) or conda, and then:
    scripts/setup_venv.sh --python \$(which python3.11)"

        bootstrap_pip || die "the venv exists but pip could not be put into it.
Tried, in order:
  1. $PYTHON -m pip --python <venv> install pip     (needs pip >= 22.3)
  2. curl https://bootstrap.pypa.io/get-pip.py      (needs a route to PyPI)
  3. virtualenv                                     (needs virtualenv)
At least one of those has to work before this can continue.  On a cluster with
none of them, ask for a python module that ships pip."

        note "created (without ensurepip)"
    fi
fi

note "pip $("$VENV/bin/python" -m pip --version 2>/dev/null | awk '{print $2}')"

PY="$VENV/bin/python"
PIP=("$PY" -m pip)

say "pip, setuptools, wheel"

# What pip will actually talk to, before it starts talking.  pip reads more
# than one index, and a container can leave a second one behind that no flag
# here undoes: the NGC PyTorch images ship
# `extra-index-url = https://pypi.ngc.nvidia.com`, which does not resolve
# outside NVIDIA -- so every install retries a host that will never answer, and
# the error names that host rather than the index you thought you set.
# `--index-url` on the command line overrides the *main* index and cannot
# remove an extra one; only the environment or the config file can.
note "pip will use:"
PIP_CONFIG="$("${PIP[@]}" config list 2>/dev/null || true)"
if [[ -n "$PIP_CONFIG" ]]; then
    printf '%s\n' "$PIP_CONFIG" | sed 's/^/     /'
else
    note "     (nothing configured; pip's own defaults for anything this"
    note "      script does not pass explicitly)"
fi
if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    note "     PIP_EXTRA_INDEX_URL is set in the environment:"
    note "       $PIP_EXTRA_INDEX_URL"
fi

# --upgrade first: a venv created from an older interpreter ships a pip that
# cannot read the metadata of some modern wheels, and the failure it produces
# looks like the package not existing.
"${PIP[@]}" install --quiet --upgrade pip setuptools wheel \
    --index-url "$PIP_INDEX" \
    || die "could not upgrade pip.  The index this script is using is
    $PIP_INDEX
but pip may also be reading an extra one from a config file -- the line above
lists what it will actually contact, and anything unreachable has to go from
there rather than from the command line:
    python -m pip config debug     # says which file, and which line
    export PIP_EXTRA_INDEX_URL=    # or clear it for this shell"
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
    # Pinned, and this pin is load-bearing rather than caution.
    #
    # The vendored pyannote-audio inside DiariZen does
    # `from torchaudio import AudioMetaData` at module scope.  torchaudio
    # deprecated that name in 2.8 and removed it in 2.9, so an unpinned install
    # gets 2.9 or later and S4 dies at import on a name that no longer exists --
    # while every other stage would have been perfectly happy.  The import is
    # what kills it; the one construction of it is on a training path this
    # pipeline never runs.
    #
    # 2.8 is therefore the last usable release, and it is recent enough for
    # everything else here: the 4090 is sm_89, comfortably inside its kernels,
    # and transformers has no trouble with it.  DiariZen's own constraints.txt
    # pins 2.1.1, which is further back than anything else here wants to go.
    #
    # torchvision comes along because it is pinned to torch *exactly* -- 0.23.0
    # requires torch==2.8.0 -- so leaving a newer one installed means pip
    # reports a conflict and, worse, leaves an ABI-mismatched extension module
    # that crashes on import.  Nothing in the inference path imports it (its
    # users in the LoCoNet checkout are the data loaders and the in-tree face
    # detector, none of which the adapter touches), but "nothing imports it
    # today" is not a reason to leave a broken one lying around.
    #
    # Override with TORCH_PIN/TORCHVISION_PIN if the vendored pyannote is ever
    # updated past the torchaudio rename.
    pin="${TORCH_PIN:-2.8.*}"
    tv_pin="${TORCHVISION_PIN:-0.23.0}"
    say "PyTorch"
    if [[ -n "$TORCH_INDEX" ]]; then
        note "index: $TORCH_INDEX"
    else
        note "index: $PIP_INDEX"
    fi
    note "pinned to $pin (see the comment in this script)"
    "${PIP[@]}" install --upgrade "torch==$pin" "torchaudio==$pin" \
        "torchvision==$tv_pin" \
        --index-url "${TORCH_INDEX:-$PIP_INDEX}" \
        || die "installing torch failed"
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
# cv2, and which of the two packages provides it
# --------------------------------------------------------------------------- #
#
# Two distributions both install a package called `cv2`: opencv-python, which
# links against libGL and libgtk, and opencv-python-headless, which does not.
# This project asks for the headless one -- and insightface, which S1 cannot do
# without, asks for the other.  So installing the `faces` extra gets both, and
# whichever was written last is the one `import cv2` finds.
#
# On a laptop that is invisible.  On a headless server it is
# `ImportError: libGL.so.1: cannot open shared object file` the first time S1
# touches cv2 -- and the fix people reach for, `apt-get install libgl1`, needs
# root, which a cluster does not give you.
#
# So the non-headless one goes, and the headless one is reinstalled rather than
# merely kept: uninstalling one of two packages that share a directory leaves a
# mix of both packages' files under one name.

say "cv2"
if "$PY" -m pip show opencv-python >/dev/null 2>&1; then
    note "opencv-python is here (insightface asks for it by name), and S1 needs"
    note "the headless build instead -- removing it and reinstalling headless"
    "${PIP[@]}" uninstall -y -q opencv-python >/dev/null 2>&1 || true
    "${PIP[@]}" install -q --force-reinstall --no-deps \
        --index-url "$PIP_INDEX" opencv-python-headless \
        || warn "could not reinstall opencv-python-headless; cv2 may be a mix"
    note "cv2 is now the headless build"
else
    note "only the headless build is installed"
fi

# --------------------------------------------------------------------------- #
# onnxruntime, which has cv2's problem and one of its own
# --------------------------------------------------------------------------- #
#
# `onnxruntime` and `onnxruntime-gpu` both install a package called
# `onnxruntime`, and three of this pipeline's dependencies ask for the CPU one
# by name -- insightface, faster-whisper and modelscope -- against this
# project's own `onnxruntime-gpu`.  Whichever was written last is the one that
# gets imported.  The CPU build has no CUDA provider, so insightface and
# faster-whisper's VAD quietly run on the CPU: no error, no warning that
# anything is wrong, just a stage that is about ten times slower than the card
# it is sitting next to.
#
# The second half is not obvious from the package name.  `onnxruntime-gpu`
# does not depend on CUDA or cuDNN -- they are optional extras -- so a bare
# install offers the CUDA provider only if the machine already has a matching
# CUDA and cuDNN of its own.  Asking for [cuda,cudnn] pulls them from PyPI
# instead, which is what makes this work on a cluster with no root: nothing
# here needs a system package.

# Only where it is wanted.  `asr` pulls onnxruntime too (faster-whisper uses it
# for its VAD, on a few seconds of audio at a time, which the CPU does
# perfectly well), and dragging a gigabyte of CUDA libraries in for that would
# be the same mistake as installing torch for a stage that never calls it.
if [[ ",$EXTRAS," == *",faces,"* ]] || "$PY" -m pip show onnxruntime-gpu >/dev/null 2>&1; then
    say "onnxruntime"
    if "$PY" -m pip show onnxruntime >/dev/null 2>&1; then
        note "the CPU build is installed (three dependencies ask for it by name);"
        note "removing it so the GPU one is what gets imported"
    fi
    # Both are removed before either is installed: the two packages write into
    # the same directory, so uninstalling one by name can take the other's files
    # with it and leave a mixture behind.
    "${PIP[@]}" uninstall -y -q onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    "${PIP[@]}" install -q --index-url "$PIP_INDEX" "onnxruntime-gpu[cuda,cudnn]" \
        || warn "could not install onnxruntime-gpu[cuda,cudnn]; S1/S3 will use the CPU"
fi

# --------------------------------------------------------------------------- #
# DiariZen, which is not a package
# --------------------------------------------------------------------------- #

say "DiariZen (S4)"
DIARIZEN="$ROOT/models/DiariZen"
if [[ ! -d "$DIARIZEN" ]]; then
    warn "no checkout at $DIARIZEN, so S4 cannot run yet."
    warn "DiariZen is a repository rather than a package and is not on PyPI,"
    warn "so it has to exist before this script can install it.  Two ways:"
    warn "    scripts/download_models.sh     fetches it, on a machine that"
    warn "                                   cannot reach GitHub"
    warn "    scripts/setup_server.sh --stages s4-diarize    clones it, on one"
    warn "                                   that can"
    warn "Either way, run this script again afterwards -- it is idempotent, and"
    warn "everything except DiariZen will already be installed."
elif ! "$PY" -c "import diarizen" >/dev/null 2>&1; then
    note "installing from $DIARIZEN"
    # Its own order: requirements, then the package, then the vendored
    # pyannote-audio -- which is a modified copy, so installing the PyPI one
    # instead is not a substitute.
    ( cd "$DIARIZEN" && "${PIP[@]}" install -r requirements.txt \
        && "${PIP[@]}" install -e . ) \
        || warn "DiariZen's own requirements did not install; S4 may still import"
    if [[ -d "$DIARIZEN/pyannote-audio" ]]; then
        # That setup.py opens with `from pkg_resources import ...`, and
        # pkg_resources was removed from setuptools in 81 -- so the build needs
        # a setuptools that still has it, *and* needs to be able to see it.
        #
        # Build isolation is what makes that awkward: it gives the build its own
        # freshly-installed setuptools, and a PIP_CONSTRAINT does not reach into
        # it (tried; the isolated build gets 84 and dies on the import either
        # way).  So isolation comes off for this one install, which means the
        # venv's setuptools is what runs -- hence the pin.
        #
        # setuptools<81 is still well above the >=68 this project and >=38.3
        # that setup.py checks for, so nothing else minds.
        note "pinning setuptools for it (its setup.py predates the removal)"
        "${PIP[@]}" install --quiet "setuptools<81" wheel --index-url "$PIP_INDEX" \
            || warn "could not pin setuptools; the vendored build may fail"
        ( cd "$DIARIZEN/pyannote-audio" \
          && "${PIP[@]}" install -e . --no-build-isolation --index-url "$PIP_INDEX" ) \
            || warn "the vendored pyannote-audio did not install"
    fi
    "$PY" -c "from diarizen.pipelines.inference import DiariZenPipeline" >/dev/null 2>&1 \
        && note "DiariZenPipeline imports" \
        || warn "DiariZen is installed but does not import; S4 will fail at run time"
else
    note "already importable"
fi

# --------------------------------------------------------------------------- #
# the 275 MB that does not have to be downloaded
# --------------------------------------------------------------------------- #
#
# Building LoCoNet makes the repository's in-tree torchvggish fetch VGGish's
# pretrained weights from a GitHub release -- and LoCoNet's own checkpoint then
# overwrites every one of them, because it carries the same tensors under
# "audioEncoder".  So the download is pure waste, and on a machine that cannot
# reach GitHub it is a hard failure at the first video rather than a slow start.
#
# torch's hub cache is keyed by filename, so seeding it from the checkpoint is
# enough: torchvggish asks for vggish-10086976.pth and gets this.  It is here
# rather than in setup_server.sh because this is the script that owns the
# environment, and the one place a working torch is guaranteed to exist.

say "torch's VGGish cache"
# In its own script rather than inline, so it can also be run on its own --
# which is what it is for, since the thing it replaces is a snippet people
# paste, and a pasted snippet resolves its paths against whatever directory
# they were standing in.  This one resolves against the checkout.
"$PY" "$ROOT/scripts/seed_vggish.py" \
    || warn "could not seed the VGGish cache; S5 will try a 275 MB download"

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
