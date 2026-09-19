#!/usr/bin/env bash
#
# Every model this pipeline uses, into one folder.
#
#   scripts/download_models.sh [--root DIR] [--endpoint URL] [--only LIST]
#
#   --root DIR       where the models go (default: ./models)
#   --endpoint URL   Hugging Face endpoint (default: https://hf-mirror.com)
#   --only LIST      comma-separated stage names, e.g. s1-faces,s8-asr
#
# **Hugging Face is reached through a mirror by default.**  `HF_ENDPOINT` is
# honoured by every library that uses `huggingface_hub`, which means it redirects
# not only the downloads this script makes but also the ones ClearerVoice and
# faster-whisper make on their own.  That is the whole reason it is set here
# rather than each URL being rewritten by hand: most of the checkpoints are
# fetched by code this project does not own.
#
# Where a model exists on ModelScope it is taken from there instead --
# emotion2vec is a ModelScope model first and a Hugging Face mirror second.
#
# LoCoNet's AVA weights are on Google Drive, which a locked-down server usually
# cannot reach.  `gdown` is tried, and if that fails the file has to be carried:
# it exists nowhere else.  That is said plainly rather than as a footnote,
# because it is the one model file a server cannot get for itself.
#
# Verified: the layout it produces is the one `configs/` names, so `doctor`
# passes afterwards.  The downloads themselves are unverified here -- this
# machine has no route to any of these hosts.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ONLY="all"
PYTHON="${PYTHON:-python3}"

# The Drive file id from the repository's README.  gdown takes the id, not the
# share URL, and the URL is the only thing the README publishes.
LOCONET_GDRIVE_ID="1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) MODELS="$2"; shift 2 ;;
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Everything Hugging Face goes to the mirror, and lands under the folder this
# script was given rather than in a home directory nobody thinks about until a
# container has no home directory.
export HF_ENDPOINT="$ENDPOINT"
export HF_HOME="$MODELS/hf"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m   %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

wants() {
    [[ "$ONLY" == "all" ]] && return 0
    [[ ",$ONLY," == *",$1,"* ]]
}

mkdir -p "$MODELS" "$HF_HOME"
say "models into $MODELS"
note "Hugging Face endpoint: $ENDPOINT"
note "Hugging Face cache:    $HF_HOME"

# `hf` is the current name and `huggingface-cli` the old one; which exists
# depends on the version installed, and both do the same thing.
HF=""
for candidate in hf huggingface-cli; do
    if have "$candidate"; then HF="$candidate"; break; fi
done
if [[ -z "$HF" ]]; then
    warn "neither 'hf' nor 'huggingface-cli' is on PATH -- the Hugging Face"
    warn "checkpoints will be fetched by the stages themselves on first use,"
    warn "through the mirror set above. Install huggingface_hub to pre-fetch."
fi

FAILED=()
fetch_hf() {
    local repo="$1" why="$2"
    if [[ -z "$HF" ]]; then return 0; fi
    printf '   %-46s ' "$repo"
    if "$HF" download "$repo" >/dev/null 2>&1; then
        echo "ok   ($why)"
    else
        echo "FAIL ($why)"
        FAILED+=("$repo")
    fi
}

fetch_modelscope() {
    local repo="$1" why="$2"
    printf '   %-46s ' "$repo"
    if ! "$PYTHON" -c "import modelscope" >/dev/null 2>&1; then
        echo "skipped (modelscope not installed; $why)"
        return 0
    fi
    if "$PYTHON" -c "
from modelscope import snapshot_download
snapshot_download('$repo', cache_dir='$MODELS/modelscope')
" >/dev/null 2>&1; then
        echo "ok   ($why, from ModelScope)"
    else
        echo "FAIL ($why)"
        FAILED+=("$repo")
    fi
}

fetch_url() {
    local url="$1" target="$2" floor="$3" why="$4"
    printf '   %-46s ' "$(basename "$target")"
    if [[ -f "$target" ]] && (( $(wc -c < "$target") >= floor )); then
        echo "already there ($why)"
        return 0
    fi
    mkdir -p "$(dirname "$target")"
    if curl -fsSL -C - --retry 3 --retry-delay 2 -o "$target" "$url" 2>/dev/null \
        && (( $(wc -c < "$target") >= floor )); then
        echo "ok   ($why)"
    else
        echo "FAIL ($why)"
        rm -f "$target"
        FAILED+=("$why")
    fi
}

# --------------------------------------------------------------------------- #
# S1 -- face detection, and the identity vectors S3 needs
# --------------------------------------------------------------------------- #

if wants s1-faces; then
    say "S1: face detection"
    note "buffalo_l and YuNet live on GitHub releases; if this machine cannot"
    note "reach GitHub, carry them in a bundle instead (setup_server.sh --from-bundle)."
    fetch_url "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip" \
        "$MODELS/insightface/models/buffalo_l.zip" 10000000 "insightface identity vectors"
    fetch_url "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
        "$MODELS/yunet.onnx" 100000 "YuNet"
fi

# --------------------------------------------------------------------------- #
# S4 -- diarization
# --------------------------------------------------------------------------- #

if wants s4-diarize; then
    say "S4: diarization"
    fetch_hf "BUT-FIT/diarizen-wavlm-large-s80-md-v2" "the checkpoint"
    # Pulled by DiariZen's own code alongside the checkpoint, so it is easy to
    # miss when counting what has to be available offline.
    fetch_hf "pyannote/wespeaker-voxceleb-resnet34-LM" "its embedding model"
fi

# --------------------------------------------------------------------------- #
# S5 -- active speaker detection
# --------------------------------------------------------------------------- #

if wants s5-asd; then
    say "S5: LoCoNet"
    if [[ -f "$MODELS/loconet/loconet_AVA.model" ]]; then
        note "loconet_AVA.model already at $MODELS/loconet/"
    else
        # Google Drive, which a locked-down server usually cannot reach.  gdown
        # is tried because it costs nothing; failing that the file has to be
        # carried, and that is worth saying plainly rather than as a footnote.
        if "$PYTHON" -c "import gdown" >/dev/null 2>&1; then
            printf '   %-46s ' "loconet_AVA.model"
            if "$PYTHON" -m gdown --id "$LOCONET_GDRIVE_ID" \
                -O "$MODELS/loconet/loconet_AVA.model" >/dev/null 2>&1 \
                && [[ -s "$MODELS/loconet/loconet_AVA.model" ]]; then
                echo "ok   (from Google Drive)"
            else
                echo "FAIL (Google Drive)"
                rm -f "$MODELS/loconet/loconet_AVA.model"
                FAILED+=("loconet_AVA.model (Google Drive unreachable)")
            fi
        else
            bad "loconet_AVA.model is not here, and gdown is not installed to fetch it."
        fi
        if [[ ! -s "$MODELS/loconet/loconet_AVA.model" ]]; then
            bad "  It exists only on Google Drive -- carry it from a machine that"
            bad "  can open the repository's README link, or put it in a bundle:"
            bad "    $MODELS/loconet/loconet_AVA.model"
            FAILED+=("loconet_AVA.model (carry by hand)")
        fi
    fi
    # 275 MB from a GitHub release, and unnecessary: LoCoNet's checkpoint
    # contains these weights verbatim under "audioEncoder", and setup_server.sh
    # seeds torch's cache from it.  Fetched anyway only if the checkpoint is
    # absent.
    if [[ ! -f "$MODELS/loconet/loconet_AVA.model" ]]; then
        fetch_url "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth" \
            "$MODELS/vggish-10086976.pth" 1000000 "VGGish (also derivable from the checkpoint)"
    fi
fi

# --------------------------------------------------------------------------- #
# S7 -- target speaker extraction
# --------------------------------------------------------------------------- #

if wants s7-tse; then
    say "S7: ClearerVoice"
    note "AV_MossFormer2_TSE_16K is fetched by the clearvoice package itself on"
    note "first use, through the endpoint set above -- there is no repo id to"
    note "name here.  To pre-fetch it, run any S7 with network once."
fi

# --------------------------------------------------------------------------- #
# S8 -- speech recognition
# --------------------------------------------------------------------------- #

if wants s8-asr; then
    say "S8: faster-whisper"
    # The CTranslate2 conversion, which is what faster-whisper actually loads --
    # not openai/whisper-large-v3, which is the PyTorch original.
    fetch_hf "Systran/faster-whisper-large-v3" "large-v3, CTranslate2"
fi

# --------------------------------------------------------------------------- #
# S9 -- paralinguistic tagging
# --------------------------------------------------------------------------- #

if wants s9-paralinguistic; then
    say "S9: the three taggers"

    # ModelScope first: emotion2vec is a ModelScope model, and the Hugging Face
    # copy is the mirror of it rather than the other way round.
    fetch_modelscope "iic/emotion2vec_plus_large" "emotion"
    note "if ModelScope is not installed, emotion2vec comes from the mirror:"
    fetch_hf "emotion2vec/emotion2vec_plus_large" "emotion (Hugging Face copy)"

    fetch_hf "laion/voice-tagging-whisper" "delivery"
    # The tagger ships no processor of its own; the adapter uses whisper-small's.
    fetch_hf "openai/whisper-small" "delivery, processor only"

    fetch_url "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" \
        "$MODELS/panns/Cnn14_mAP=0.431.pth" 100000000 "PANNs events"
    note "PANNs also downloads its AudioSet label list on import, into ~/panns_data"
fi

# --------------------------------------------------------------------------- #
# S10 -- captioning
# --------------------------------------------------------------------------- #

if wants s10-caption; then
    say "S10: captioning"
    # The 8B dense checkpoint, not the 30B mixture-of-experts one: the 30B is
    # 62 GB in bf16 and does not fit a 48 GB card, and its FP8 build cannot be
    # loaded by transformers at all.  Change this line if you have the memory.
    fetch_hf "Qwen/Qwen3-VL-8B-Instruct" "captioning"
fi

# --------------------------------------------------------------------------- #
# where everything ended up
# --------------------------------------------------------------------------- #

say "what is here"
"$PYTHON" - "$MODELS" <<'PYEOF'
import os, sys
from pathlib import Path

root = Path(sys.argv[1])
total = 0
for path in sorted(root.rglob("*")):
    if path.is_file() and not path.name.startswith("."):
        total += path.stat().st_size
print(f"   {root}: {total / 1024 / 1024 / 1024:.1f} GB")
for entry in sorted(root.iterdir()):
    if entry.is_dir():
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        print(f"     {entry.name}/{' ' * max(0, 20 - len(entry.name))} {size / 1024 / 1024:.0f} MB")
PYEOF

cat <<EOF

The stages read these through HF_HOME and the configs' relative paths, so a run
needs both set as this script had them:

    export HF_HOME="$HF_HOME"
    export HF_ENDPOINT="$ENDPOINT"

Then check it took:

    "$PYTHON" -m avannotate.cli doctor

EOF

if (( ${#FAILED[@]} )); then
    echo
    bad "not everything arrived:"
    for item in "${FAILED[@]}"; do bad "  $item"; done
    exit 1
fi
echo "everything this script can fetch is here"
