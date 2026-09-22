#!/usr/bin/env bash
#
# Everything that has to be fetched before the pipeline can run.
#
# The rule this script follows: it does what can be automated and stops to say
# exactly what it cannot.  Two of the six models need a git clone and a pip
# install, which is scriptable; two need weights that only exist behind a
# manual link, which is not.  Pretending otherwise would produce a script that
# reports success and leaves a batch to fail on video one.
#
# Usage:
#   scripts/setup_server.sh [--root DIR] [--stages LIST] [--no-weights]
#
#   --root DIR      where the checkouts go (default: ./models)
#   --stages LIST   comma-separated, default every stage that needs something
#   --no-weights    skip the downloads that this script can do
#   --from-bundle DIR
#                   install from a bundle made by make_offline_bundle.sh and
#                   make no network calls at all. Use this on a server that
#                   cannot reach github.com -- which is also the reason the
#                   bundle exists.
#
# The default root is not arbitrary: the shipped configs name their model paths
# relative to themselves as "../models/<name>", so cloning anywhere else would
# leave every config pointing at nothing and the doctor reporting it forever.
# /models/ is gitignored, which is what it is there for.
#
# Idempotent: re-running it fetches nothing that is already there.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# curl does not read the macOS system proxy; this does.  See lib.sh.
# shellcheck source=lib.sh
source "$ROOT/scripts/lib.sh"
MODELS="$ROOT/models"
# Where torch's hub cache lives, which is where the VGGish seed below goes.
# Exported rather than expanded at the point of use: that step is a heredoc
# python reading TORCH_HOME from its own environment, and the value has to be
# the one run_batch.sh will look in.  Two defaults that merely happen to agree
# is the shape of the bug this is here to avoid -- they disagreed, and S5
# quietly asked GitHub for 275 MB it cannot reach.
export TORCH_HOME="${TORCH_HOME:-$MODELS/torch}"
STAGES="all"
FETCH_WEIGHTS=1
BUNDLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) MODELS="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --no-weights) FETCH_WEIGHTS=0; shift ;;
        --from-bundle) BUNDLE="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

wants() {
    [[ "$STAGES" == "all" ]] && return 0
    [[ ",$STAGES," == *",$1,"* ]]
}

have() { command -v "$1" >/dev/null 2>&1; }

# Where pip is allowed to look.  Defined before any stage runs, because a stage
# running on its own still needs them -- they lived inside the S4 block first,
# and `--stages s5-asd` died on an unbound variable.
#
# With a bundle, PyPI is taken out of the picture entirely: everything is either
# already installed or in the wheelhouse, and a silent fallback to the network
# is exactly what a locked-down server cannot do.


mkdir -p "$MODELS"

# --------------------------------------------------------------------------- #
# where the bytes come from: a checkout, or the bundle
# --------------------------------------------------------------------------- #

# Where pip is allowed to look.  Built here rather than inside a stage's block,
# because a stage running on its own still needs it: it lived inside the S4
# block at first, and `--stages s5-asd` died on an unbound variable.
#
# With a bundle, PyPI is out of the picture entirely -- everything is either
# already installed or in the wheelhouse, and a silent fallback to the network
# is exactly what a locked-down server cannot do.
PIP_EXTRA=()
if [[ -n "$BUNDLE" && -d "$BUNDLE/wheels" ]]; then
    PIP_EXTRA=(--no-index --find-links "$BUNDLE/wheels")
else
    # Which index to use is a property of the machine rather than of this
    # project, so lib.sh probes for it: PyPI where it is reachable, a domestic
    # mirror where it is not.  Without a bundle this is the only source of
    # packages, and a machine that cannot reach GitHub usually cannot reach
    # PyPI either.  `--index-url` is skipped entirely in the wheelhouse branch
    # above, where `--no-index` is the whole point.
    pip_index_args
    PIP_EXTRA=(${PIP_INDEX_ARGS[@]+"${PIP_INDEX_ARGS[@]}"})
fi

# `"${arr[@]}"` on an empty array is an error under `set -u` on bash 3.2, which
# is still what /bin/bash is on macOS.  This spelling is the portable one.
pip_install() { "$PYTHON" -m pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} "$@"; }

if [[ -n "$BUNDLE" ]]; then
    [[ -d "$BUNDLE" ]] || die "no bundle at $BUNDLE"
    [[ -f "$BUNDLE/MANIFEST.json" ]] || die "$BUNDLE has no MANIFEST.json; is it a bundle?"
    printf '\n\033[1m%s\033[0m\n' "installing from $BUNDLE -- no network calls"
    note "the bundle is not verified against its manifest; sha256sum -c if you care"
fi

# Unpack a repository tarball into place, from the bundle if there is one and
# from GitHub otherwise.  One function so the two paths cannot diverge -- the
# offline route is meant to be the same install, not a second one.
install_repo() {
    local name="$1" url="$2" dest="$3"

    if [[ -d "$dest/.git" || -f "$dest/loconet.py" || -f "$dest/setup.py" || -f "$dest/pyproject.toml" ]]; then
        note "already at $dest"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    if [[ -n "$BUNDLE" ]]; then
        local tarball="$BUNDLE/repos/$name.tar.gz"
        [[ -f "$tarball" ]] || die "$BUNDLE has no repos/$name.tar.gz"
        note "unpacking $name from the bundle"
        # codeload tarballs wrap everything in one top-level directory.
        mkdir -p "$dest"
        tar xzf "$tarball" -C "$dest" --strip-components=1
    else
        note "cloning $name from $url"
        # --recurse-submodules matters for DiariZen, and its absence is silent:
        # the vendored pyannote-audio and dscore are submodules, so a plain
        # --depth 1 clone leaves both as empty directories -- and the check for
        # "$DEST/pyannote-audio" below then succeeds on an empty directory and
        # skips the install that was the point of it.
        git clone --depth 1 --recurse-submodules --shallow-submodules "$url" "$dest" \
            || die "cloning $name failed; if this machine has no GitHub access, \
build a bundle elsewhere with scripts/make_offline_bundle.sh and pass --from-bundle"
        if [[ -d "$dest/pyannote-audio" && -z "$(ls -A "$dest/pyannote-audio")" ]]; then
            warn "$dest/pyannote-audio is an empty directory -- its submodule did"
            warn "not come down.  Fetch it with:"
            warn "    git -C $dest submodule update --init --recursive --depth 1"
        fi
    fi
}

# Put a file where a stage expects it, from the bundle if there is one.
# Returns non-zero when neither place has it, so the caller can decide whether
# that is fatal or merely a warning.
place_file() {
    local name="$1" dest="$2"
    [[ -f "$dest" ]] && { note "already at $dest"; return 0; }
    if [[ -n "$BUNDLE" && -f "$BUNDLE/models/$name" ]]; then
        mkdir -p "$(dirname "$dest")"
        cp "$BUNDLE/models/$name" "$dest"
        note "$name -> $dest ($(( $(wc -c < "$dest") / 1024 / 1024 )) MB)"
        return 0
    fi
    return 1
}

# --------------------------------------------------------------------------- #
# ffmpeg is not optional: S0 through S11 all decode through it
# --------------------------------------------------------------------------- #

say "checking the prerequisites everything needs"
for tool in ffmpeg ffprobe; do
    if have "$tool"; then
        note "$tool: $(command -v "$tool")"
    else
        die "$tool is not installed, and every stage decodes through it."
    fi
done

# The configured interpreter, not whatever `python` happens to be first on PATH:
# a server means a venv, and checking the wrong one is how a run ends up
# installing into a system environment nobody looks at again.
if "$PYTHON" -c 'import sys; print(sys.version_info[:2])' >/dev/null 2>&1; then
    note "$PYTHON: $("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') ($(command -v "$PYTHON"))"
else
    die "the interpreter $PYTHON does not run; set PYTHON=/path/to/venv/bin/python"
fi

# git is only needed to clone.  With a bundle it is never used.
if [[ -z "$BUNDLE" ]] && ! have git; then
    die "git is required to clone the two repositories; either install it or build a bundle elsewhere with scripts/make_offline_bundle.sh"
fi

# --------------------------------------------------------------------------- #
# DiariZen -- a repository, not a package
# --------------------------------------------------------------------------- #

if wants s4-diarize; then
    say "S4: DiariZen"
    DEST="$MODELS/DiariZen"
    install_repo DiariZen https://github.com/BUTSpeechFIT/DiariZen "$DEST"

    # Its own instructions, in its own order.  The vendored pyannote-audio is
    # the part that surprises people: DiariZen ships a modified copy of it and
    # installs that, not the one on PyPI.
    note "installing DiariZen's requirements"
    ( cd "$DEST" \
      && pip_install -r requirements.txt \
      && pip_install -e . ) \
        || die "installing DiariZen failed; see docs/server-setup.md"

    if [[ -d "$DEST/pyannote-audio" ]]; then
        note "installing the vendored pyannote-audio"
        # Its setup.py does `from pkg_resources import ...`, and pkg_resources
        # was removed from setuptools in 81.  So this build needs a setuptools
        # that still has it AND needs to be able to see it, which rules out
        # build isolation -- the isolated build gets its own newest setuptools
        # and a PIP_CONSTRAINT does not reach into it.  Hence the pin plus
        # --no-build-isolation.
        #
        # -c ../constraints.txt is dropped here rather than kept: it pins torch
        # to 2.1.1, and pip resolving pyannote's own torch dependency would
        # honour that and downgrade a working torch to satisfy it.  See
        # docs/server-setup.md.
        pip_install "setuptools<81" wheel \
            || warn "could not pin setuptools; the vendored build may fail"
        ( cd "$DEST/pyannote-audio" \
          && pip_install -e .[dev,testing] --no-build-isolation ) \
            || warn "the vendored pyannote-audio did not install; DiariZen may still import"
    else
        warn "no pyannote-audio directory found; DiariZen's layout may have changed"
    fi

    note "verifying the import"
    "$PYTHON" -c "from diarizen.pipelines.inference import DiariZenPipeline; print('   DiariZenPipeline imports')" \
        || die "DiariZen is cloned but does not import; the module path may have changed. \
This is what docs/server-setup.md's first checklist item is for."

    note "weights are fetched from Hugging Face on first use (CC BY-NC 4.0, non-commercial)"
fi

# --------------------------------------------------------------------------- #
# LoCoNet -- a repository, plus weights behind a Google Drive link
# --------------------------------------------------------------------------- #

if wants s5-asd; then
    say "S5: LoCoNet"
    # Nested, because configs/s5.loconet.json names ../models/loconet/... and
    # a checkout one directory away from where the config expects it is a
    # checkout the pipeline will never find.
    DEST="$MODELS/loconet/LoCoNet_ASD"
    mkdir -p "$MODELS/loconet"
    install_repo LoCoNet_ASD https://github.com/SJTUwxz/LoCoNet_ASD "$DEST"

    note "installing the packages its inference path needs"
    pip_install "$ROOT[asd]" \
        || pip_install torch opencv-python-headless resampy

    # The one thing no script can fetch: the weights sit behind a Google Drive
    # link, which needs a browser.  Stated as commands rather than as prose.
    AVAMODEL="$MODELS/loconet/loconet_AVA.model"
    if place_file loconet_AVA.model "$AVAMODEL"; then
        :
    else
        warn "the AVA weights are not in this bundle and are NOT fetched -- they"
        warn "are behind a Google Drive link, which needs a browser:"
        warn "    $DEST/README.md"
        warn "configs/s5.loconet.json expects them at $AVAMODEL"
        warn "Download them on a connected machine, put the file in the bundle's"
        warn "models/ directory, and re-run this script."
    fi

    # And the step that saves 275 MB.  Building the model makes torchvggish
    # download VGGish's pretrained weights into torch's hub cache -- and then
    # LoCoNet's own checkpoint overwrites every one of them.  Those weights are
    # already inside the checkpoint under "audioEncoder", key for key, so the
    # cache can be filled from it instead.  20 MB of local copying against a
    # 275 MB download that some servers cannot make at all.
    VGGISH_CACHE="$TORCH_HOME/hub/checkpoints/vggish-10086976.pth"
    if [[ -f "$VGGISH_CACHE" ]]; then
        note "torch's VGGish cache is already seeded"
    elif [[ -n "$BUNDLE" && -f "$BUNDLE/models/vggish-10086976.pth" ]]; then
        say "seeding torch's hub cache from the bundle"
        mkdir -p "$(dirname "$VGGISH_CACHE")"
        cp "$BUNDLE/models/vggish-10086976.pth" "$VGGISH_CACHE"
        note "$VGGISH_CACHE ($(( $(wc -c < "$VGGISH_CACHE") / 1024 / 1024 )) MB)"
    elif [[ -f "$AVAMODEL" ]]; then
        say "seeding torch's hub cache from the checkpoint (saves a 275 MB download)"
        "$PYTHON" - "$AVAMODEL" <<'PYEOF'
import os, sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
inner = "model.module.model.audioEncoder."
audio = {k[len(inner):]: v for k, v in ckpt.items() if k.startswith(inner)}
if not audio:
    raise SystemExit(
        "no audioEncoder weights in the checkpoint; the VGGish cache cannot be "
        "seeded from it and the 275 MB download will happen on first use"
    )
# No fallback: this script exports TORCH_HOME, so a second default here would
# be a second thing to keep in agreement with run_batch.sh.
target = os.path.join(
    os.environ["TORCH_HOME"],
    "hub", "checkpoints", "vggish-10086976.pth",
)
os.makedirs(os.path.dirname(target), exist_ok=True)
torch.save(audio, target)
print(f"   {target}: {os.path.getsize(target) // 1024 // 1024} MB from {len(audio)} keys")
PYEOF
    fi
fi

# --------------------------------------------------------------------------- #
# the pip-installable stages
# --------------------------------------------------------------------------- #

if wants s1-faces; then
    say "S1: insightface (also what S3 needs for identity vectors)"
    pip_install "$ROOT[faces]" || warn "installing insightface failed"

    # buffalo_l is fetched from a GitHub release on first use, and it is the
    # primary path for S1 and S3 rather than an option -- insightface has no
    # other source for it.  Placed here so the first run does not reach out.
    #
    # insightface extracts the zip in place, so a zip that is present and a
    # directory that is absent is the state it resumes from.
    BUFFALO="$MODELS/insightface/models/buffalo_l.zip"
    if [[ -d "${BUFFALO%.zip}" ]]; then
        note "buffalo_l already extracted at ${BUFFALO%.zip}"
    else
        place_file buffalo_l.zip "$BUFFALO" \
            || warn "buffalo_l is not in the bundle; insightface will try to fetch it from GitHub on first use"
    fi

    # And YuNet, the alternative detector, for the same reason.
    place_file yunet.onnx "$MODELS/yunet.onnx" \
        || note "YuNet is not in the bundle; only configs/s1.yunet.json needs it"
fi

if wants s7-tse; then
    say "S7: ClearerVoice"
    pip_install "$ROOT[tse]" || warn "installing clearvoice failed"
    note "the checkpoint is fetched from Hugging Face on first use"
fi

if wants s8-asr; then
    say "S8: faster-whisper"
    pip_install "$ROOT[asr]" || warn "installing faster-whisper failed"
    if [[ "$FETCH_WEIGHTS" == 1 ]]; then
        note "pre-fetching the checkpoint so the first run is not the one that waits"
        "$PYTHON" -m pip show faster-whisper >/dev/null 2>&1 && \
            "$PYTHON" -c "
from faster_whisper import WhisperModel
WhisperModel('${WHISPER_MODEL:-large-v3}', device='cpu', compute_type='int8')
print('   ${WHISPER_MODEL:-large-v3} is cached')
" || warn "could not pre-fetch the checkpoint; it will download on first use"
    fi
fi

if wants s9-paralinguistic; then
    say "S9: the three taggers"
    pip_install "$ROOT[paralinguistic]" || warn "installing the taggers failed"

    if [[ "$FETCH_WEIGHTS" == 1 ]]; then
        # PANNs downloads its own label list on import, into ~/panns_data, and
        # that step needs a home directory and network.  Doing it now means a
        # container running as nobody fails here rather than mid-batch.
        note "pre-fetching PANNs' label list"
        "$PYTHON" -c "from panns_inference import labels; print(f'   {len(labels)} AudioSet labels cached')" \
            || warn "could not fetch the AudioSet labels"

        # Under models/, like everything else, so configs/s9.paralinguistic.json
        # can name it with the same ../models/ convention.
        PANNS_DIR="$MODELS/panns"
        PANNS_PATH="$PANNS_DIR/Cnn14_mAP=0.431.pth"
        if [[ -f "$PANNS_PATH" ]]; then
            note "PANNs checkpoint already at $PANNS_PATH"
        elif place_file 'Cnn14_mAP=0.431.pth' "$PANNS_PATH"; then
            :
        else
            mkdir -p "$PANNS_DIR"
            note "fetching PANNs' checkpoint (Zenodo, 312 MB)"
            if curl -L -C - --fail -s -o "$PANNS_PATH" \
                'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'; then
                note "saved to $PANNS_PATH"
            else
                warn "the download failed or was interrupted; resume it with:"
                warn "    curl -L -C - -o '$PANNS_PATH' \\"
                warn "      'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'"
            fi
        fi
        warn "then set \"checkpoint\" in configs/s9.paralinguistic.json to:"
        warn "    \"../models/panns/Cnn14_mAP=0.431.pth\""
    fi
fi

if wants s10-caption; then
    say "S10: the captioning model"
    pip_install "$ROOT[caption]" || warn "installing transformers failed"
    warn "check the checkpoint's weight size against the card BEFORE choosing one:"
    warn "    30B-A3B is 62 GB bf16 and will not fit a 48 GB card;"
    warn "    its FP8 build is 32 GB but transformers cannot load it;"
    warn "    8B dense is ~16 GB and is what configs/s10.caption.json assumes."
    warn "    See docs/server-setup.md for the full table."
fi

# --------------------------------------------------------------------------- #
# what is left
# --------------------------------------------------------------------------- #

say "re-running the doctor"
"$PYTHON" -m avannotate.cli doctor || true

cat <<EOF

What is left, and none of it can be automated:

  1. LoCoNet's AVA weights -- behind a Google Drive link, which needs a browser.
     Put loconet_AVA.model in a bundle's models/ directory and re-run with
     --from-bundle, or drop it at $MODELS/loconet/loconet_AVA.model.
  2. PANNs' checkpoint -- Zenodo. Fetch it on a connected machine, or let this
     script do it if the network allows.

Check what this machine actually has, against the configs a run will use:

    "$PYTHON" -m avannotate.cli doctor \
        --configs-dir "$ROOT/configs"

EOF
