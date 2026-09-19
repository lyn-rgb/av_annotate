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
#   --root DIR      where the checkouts go (default: ./third_party)
#   --stages LIST   comma-separated, default every stage that needs something
#   --no-weights    skip the downloads that this script can do
#
# Idempotent: re-running it fetches nothing that is already there.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
STAGES="all"
FETCH_WEIGHTS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) THIRD_PARTY="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --no-weights) FETCH_WEIGHTS=0; shift ;;
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

mkdir -p "$THIRD_PARTY"

# --------------------------------------------------------------------------- #
# ffmpeg is not optional: S0 through S11 all decode through it
# --------------------------------------------------------------------------- #

say "checking the prerequisites everything needs"
for tool in ffmpeg ffprobe git python; do
    if have "$tool"; then
        note "$tool: $(command -v "$tool")"
    else
        die "$tool is not installed, and every stage needs it."
    fi
done

# --------------------------------------------------------------------------- #
# DiariZen -- a repository, not a package
# --------------------------------------------------------------------------- #

if wants s4-diarize; then
    say "S4: DiariZen"
    DEST="$THIRD_PARTY/DiariZen"
    if [[ -d "$DEST/.git" ]]; then
        note "already cloned at $DEST"
    else
        git clone --recursive https://github.com/BUTSpeechFIT/DiariZen "$DEST" \
            || die "cloning DiariZen failed; check network access to github.com"
    fi

    # Its own instructions, in its own order.  The vendored pyannote-audio is
    # the part that surprises people: DiariZen ships a modified copy of it and
    # installs that, not the one on PyPI.
    note "installing DiariZen's requirements"
    ( cd "$DEST" \
      && python -m pip install -r requirements.txt \
      && python -m pip install -e . ) \
        || die "installing DiariZen failed; see docs/server-setup.md"

    if [[ -d "$DEST/pyannote-audio" ]]; then
        note "installing the vendored pyannote-audio"
        ( cd "$DEST/pyannote-audio" \
          && python -m pip install -e .[dev,testing] -c ../constraints.txt ) \
            || warn "the vendored pyannote-audio did not install; DiariZen may still import"
    else
        warn "no pyannote-audio directory found; DiariZen's layout may have changed"
    fi

    note "verifying the import"
    python -c "from diarizen.pipelines.inference import DiariZenPipeline; print('   DiariZenPipeline imports')" \
        || die "DiariZen is cloned but does not import; the module path may have changed. \
This is what docs/server-setup.md's first checklist item is for."

    note "weights are fetched from Hugging Face on first use (CC BY-NC 4.0, non-commercial)"
fi

# --------------------------------------------------------------------------- #
# LoCoNet -- a repository, plus weights behind a Google Drive link
# --------------------------------------------------------------------------- #

if wants s5-asd; then
    say "S5: LoCoNet"
    DEST="$THIRD_PARTY/LoCoNet_ASD"
    if [[ -d "$DEST/.git" ]]; then
        note "already cloned at $DEST"
    else
        git clone https://github.com/SJTUwxz/LoCoNet_ASD "$DEST" \
            || die "cloning LoCoNet_ASD failed"
    fi

    note "installing the packages its inference path needs"
    python -m pip install 'avannotate[asd]' || python -m pip install torch opencv-python-headless

    # The one thing no script can do.  Stated as a command to run rather than
    # as a paragraph to read.
    warn "the AVA weights are NOT downloaded by this script."
    warn "Open the repository README and use its Google Drive link:"
    warn "    $DEST/README.md"
    warn "Save it as $THIRD_PARTY/loconet_AVA.model, then set in configs/s5.loconet.json:"
    warn "    \"repo\":       \"$DEST\""
    warn "    \"checkpoint\": \"$THIRD_PARTY/loconet_AVA.model\""
fi

# --------------------------------------------------------------------------- #
# the pip-installable stages
# --------------------------------------------------------------------------- #

if wants s1-faces; then
    say "S1: insightface (also what S3 needs for identity vectors)"
    python -m pip install 'avannotate[faces]' || warn "installing insightface failed"
fi

if wants s7-tse; then
    say "S7: ClearerVoice"
    python -m pip install 'avannotate[tse]' || warn "installing clearvoice failed"
    note "the checkpoint is fetched from Hugging Face on first use"
fi

if wants s8-asr; then
    say "S8: faster-whisper"
    python -m pip install 'avannotate[asr]' || warn "installing faster-whisper failed"
    if [[ "$FETCH_WEIGHTS" == 1 ]]; then
        note "pre-fetching the checkpoint so the first run is not the one that waits"
        python -m pip show faster-whisper >/dev/null 2>&1 && \
            python -c "
from faster_whisper import WhisperModel
WhisperModel('${WHISPER_MODEL:-large-v3}', device='cpu', compute_type='int8')
print('   ${WHISPER_MODEL:-large-v3} is cached')
" || warn "could not pre-fetch the checkpoint; it will download on first use"
    fi
fi

if wants s9-paralinguistic; then
    say "S9: the three taggers"
    python -m pip install 'avannotate[paralinguistic]' || warn "installing the taggers failed"

    if [[ "$FETCH_WEIGHTS" == 1 ]]; then
        # PANNs downloads its own label list on import, into ~/panns_data, and
        # that step needs a home directory and network.  Doing it now means a
        # container running as nobody fails here rather than mid-batch.
        note "pre-fetching PANNs' label list"
        python -c "from panns_inference import labels; print(f'   {len(labels)} AudioSet labels cached')" \
            || warn "could not fetch the AudioSet labels"

        PANNS_DIR="${PANNS_DIR:-$HOME/panns_data}"
        PANNS_PATH="$PANNS_DIR/Cnn14_mAP=0.431.pth"
        if [[ -f "$PANNS_PATH" ]]; then
            note "PANNs checkpoint already at $PANNS_PATH"
        else
            warn "PANNs' checkpoint is NOT downloaded by this script (Zenodo, 312 MB)."
            warn "Fetch it and set \"checkpoint\" in configs/s9.paralinguistic.json:"
            warn "    mkdir -p $PANNS_DIR && curl -L -C - -o '$PANNS_PATH' \\"
            warn "      'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'"
        fi
    fi
fi

if wants s10-caption; then
    say "S10: the captioning model"
    python -m pip install 'avannotate[caption]' || warn "installing transformers failed"
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
python -m avannotate.cli doctor || true

cat <<'EOF'

Two things are deliberately not automated, and a human has to do them once:

  1. LoCoNet's AVA weights -- a Google Drive link in third_party/LoCoNet_ASD/README.md
  2. PANNs' checkpoint     -- Zenodo; the curl command is printed above

Everything else is installed.  Re-run this script's doctor with the config a
run will use, which is what checks the two checkpoint paths:

    python -m avannotate.cli doctor \
        --config configs/s5.loconet.json \
        --config configs/s9.paralinguistic.json

EOF
