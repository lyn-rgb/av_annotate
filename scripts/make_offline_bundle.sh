#!/usr/bin/env bash
#
# Everything the pipeline needs, gathered into one directory to carry to a
# server that cannot reach github.com.
#
# Run this on a machine that CAN reach the internet.  It produces a directory
# (and a tarball of it) holding every artifact whose home is GitHub, plus
# whatever else you ask for.  `setup_server.sh --from-bundle DIR` then installs
# from it without making a single network call.
#
# Why this is needed at all: the pipeline's code names three GitHub URLs, but
# the dependencies reach for more of them on their own --
#
#   insightface's buffalo_l    github.com/deepinsight/insightface/releases
#   torchvggish's VGGish       github.com/harritaylor/torchvggish/releases
#
# -- and neither is mentioned anywhere in this repository.  The second is now
# avoidable (it can be taken from the LoCoNet checkpoint), but buffalo_l is the
# primary path for S1 and S3 and has no substitute.
#
# Usage:
#   scripts/make_offline_bundle.sh [--out DIR] [--with-wheels] [--with-hf]
#
#   --out DIR        where to build it (default: ./offline-bundle)
#   --with-wheels    also run `pip download` so the server needs no PyPI either
#   --with-hf        also snapshot the Hugging Face checkpoints into the bundle
#
# Add --with-wheels and --with-hf if the server is cut off from PyPI and
# Hugging Face too; without them the bundle covers GitHub only, which is what
# "cannot reach github" usually means.
#
# A server that cannot reach GitHub usually cannot reach much else either, so
# the two weights that do NOT live on GitHub are packed when they can be had:
# PANNs' from Zenodo (fetched), and LoCoNet's from Google Drive (copied in from
# wherever you already downloaded it -- no script can fetch that one).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# curl does not read the macOS system proxy; this does.  See lib.sh.
# shellcheck source=lib.sh
source "$ROOT/scripts/lib.sh"
OUT="$ROOT/offline-bundle"
WITH_WHEELS=0
WITH_HF=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --with-wheels) WITH_WHEELS=1; shift ;;
        --with-hf) WITH_HF=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# The repositories, and the two model files whose home is a GitHub release.
DIARIZEN_REPO="https://github.com/BUTSpeechFIT/DiariZen"
LOCONET_REPO="https://github.com/SJTUwxz/LoCoNet_ASD"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
VGGISH_URL="https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"

# Not GitHub, and included anyway: a locked-down server is usually locked down
# all the way, and arriving without these means a second trip.
PANNS_URL="https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
# Where your own downloads already live, so they can be carried over rather
# than fetched again.  LoCoNet's weights are behind Google Drive and can only
# come from here.
MANUAL_DIR="$ROOT/models"

rm -rf "$OUT"
mkdir -p "$OUT/repos" "$OUT/models" "$OUT/hf"
say "building an offline bundle in $OUT"

if ! have curl; then
    die "curl is required to build the bundle"
fi

# --------------------------------------------------------------------------- #
# repositories
# --------------------------------------------------------------------------- #

# Download, resuming across attempts.
#
# `--retry` alone is not enough: a connection that closes early gives curl exit
# 18 ("transferred a partial file"), which it does not consider transient, so a
# single interrupted transfer fails the whole bundle. Building a bundle has to
# work on an ordinary flaky connection -- that is the machine it runs on --
# so this resumes with `-C -` until the file stops growing.
fetch_resumable() {
    local url="$1" target="$2"
    local attempt size previous=0
    for attempt in 1 2 3 4 5 6 7 8; do
        curl -fsSL -C - --retry 3 --retry-delay 2 --connect-timeout 30 \
            -o "$target" "$url" && return 0
        # Checked for existence before being measured: after a restart the file
        # is genuinely gone, and `wc -c < missing` is a shell redirect error
        # rather than a number.
        if [[ -f "$target" ]]; then
            size=$(wc -c < "$target")
        else
            size=0
        fi
        if (( size > 0 && size == previous )); then
            # Two attempts that moved nothing: the server is not honouring the
            # range request, so resuming will never finish.  Start over.
            note "no progress at $size bytes; restarting the download"
            rm -f "$target"
            previous=0
            continue
        fi
        previous=$size
        note "attempt $attempt stopped at $size bytes; resuming"
    done
    return 1
}

# A tarball rather than a git clone: the server does not need git, does not need
# the history, and a tarball cannot half-succeed the way a clone can.
fetch_repo() {
    local name="$1" url="$2" local_dir="${3:-}"
    say "repository: $name"

    local target="$OUT/repos/$name.tar.gz"
    rm -f "$target"

    # A checkout already on this machine is packed as it stands, and that is the
    # whole difference between a bundle that can be built here and one that
    # cannot: this script is meant to run on a machine that CAN reach the
    # internet, and when that stops being true -- a proxy subscription lapses,
    # the route is blocked -- the working checkouts are still here.  Refusing to
    # use them would mean a bundle that cannot be built on the machine holding
    # everything it needs.
    #
    # Packed as a single top-level directory, because setup_server.sh unpacks it
    # with --strip-components=1, the same shape codeload produces.
    if [[ -n "$local_dir" && -d "$local_dir" ]]; then
        note "using the checkout at $local_dir rather than downloading"
        tar czf "$target" -C "$(dirname "$local_dir")" "$(basename "$local_dir")"
        gzip -t "$target" 2>/dev/null || die "packing $local_dir produced something that is not a gzip"
        note "$(du -h "$target" | cut -f1) -> ${target#"$OUT"/}"
        echo "$name local:$local_dir $url" >> "$OUT/repos/SOURCES.txt"
        return 0
    fi

    # Recorded so a bundle can be traced back to the commit it came from --
    # `main` moves, and a bundle with no provenance is a bundle nobody can
    # reproduce.  Needs git here only, never on the server.
    # Provenance is a nicety, not a requirement: recording which commit a
    # tarball came from is worth having and is never worth failing a bundle
    # over.  `|| true` because a failing command substitution aborts the script
    # under `set -e` before the fallback below can run -- and a mirror or a
    # proxy that cannot serve git is an ordinary thing to meet.
    local sha="unknown"
    if have git; then
        sha="$(git ls-remote "$url" HEAD 2>/dev/null | awk '{print $1}' || true)"
        [[ -n "$sha" ]] || sha="unknown"
    fi
    note "commit $sha"

    fetch_resumable \
        "$(echo "$url" | sed 's|https://github.com/|https://codeload.github.com/|')/tar.gz/refs/heads/main" \
        "$target" \
        || die "could not download $url after several attempts, and there is no checkout at ${local_dir:-<none given>} to pack instead"
    # A tarball that is not a gzip is an error page, and unpacking it later
    # would fail on the server rather than here.
    gzip -t "$target" 2>/dev/null || die "$name.tar.gz is not a gzip archive -- the download is not what it should be"
    note "$(du -h "$target" | cut -f1) -> ${target#"$OUT"/}"

    echo "$name $sha $url" >> "$OUT/repos/SOURCES.txt"
    note "recorded in repos/SOURCES.txt so a bundle can be traced to a commit"
}

fetch_repo DiariZen "$DIARIZEN_REPO" "$MANUAL_DIR/DiariZen"
fetch_repo LoCoNet_ASD "$LOCONET_REPO" "$MANUAL_DIR/loconet/LoCoNet_ASD"

# --------------------------------------------------------------------------- #
# model files
# --------------------------------------------------------------------------- #

say "model files"

fetch_file() {
    local url="$1" target="$2" min_bytes="$3"
    local size
    rm -f "$OUT/$target"
    # The size floor is checked after a resumable fetch because a truncated
    # download is the failure that actually happens here: curl can exit 0 on a
    # connection that closed early, and a short model file does not fail until
    # something tries to load it.
    fetch_resumable "$url" "$OUT/$target" \
        || die "could not download $url"
    size="$(wc -c < "$OUT/$target")"
    if (( size < min_bytes )); then
        die "$target is only $size bytes (expected at least $min_bytes) -- the
download is truncated even after retrying. Check the URL, or fetch it by hand."
    fi
    if (( size >= 1048576 )); then
        note "$target: $((size / 1024 / 1024)) MB"
    else
        note "$target: $((size / 1024)) KB"
    fi
}

fetch_file "$BUFFALO_URL" "models/buffalo_l.zip" 10000000
fetch_file "$YUNET_URL" "models/yunet.onnx" 100000

# Optional, and the least necessary thing here: the LoCoNet checkpoint contains
# these weights verbatim, so the server can seed torch's cache from it instead.
# Included anyway because it is 20 MB and removes a step from the server.
if fetch_resumable "$VGGISH_URL" "$OUT/models/vggish-10086976.pth" \
    && (( $(wc -c < "$OUT/models/vggish-10086976.pth") > 1000000 )); then
    note "models/vggish-10086976.pth: $(( $(wc -c < "$OUT/models/vggish-10086976.pth") / 1024 / 1024 )) MB"
else
    warn "the VGGish weights did not download; the server can seed them from the"
    warn "LoCoNet checkpoint instead -- setup_server.sh does this automatically"
    rm -f "$OUT/models/vggish-10086976.pth"
fi

# --------------------------------------------------------------------------- #
# the weights that do not live on GitHub
# --------------------------------------------------------------------------- #

say "weights that are not on GitHub"

if fetch_resumable "$PANNS_URL" "$OUT/models/Cnn14_mAP=0.431.pth" \
    && (( $(wc -c < "$OUT/models/Cnn14_mAP=0.431.pth") > 100000000 )); then
    note "models/Cnn14_mAP=0.431.pth: $(( $(wc -c < "$OUT/models/Cnn14_mAP=0.431.pth") / 1024 / 1024 )) MB (PANNs, from Zenodo)"
else
    warn "PANNs' checkpoint did not download from Zenodo; fetch it by hand"
    rm -f "$OUT/models/Cnn14_mAP=0.431.pth"
fi

# LoCoNet's weights.  Google Drive, which needs either a browser or `gdown` --
# and a server very often has a route to neither, which is the whole reason
# this bundle exists.  Tried in order: the file if you already have it, then
# gdown, then a note saying what to do by hand.
LOCONET_GDRIVE_ID="1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK"

verify_loconet() {
    # A Google Drive download that hits a quota or a "confirm you are not a
    # robot" page returns *HTML with status 200*, and gdown saves it happily.
    # The file is then 2 KB of markup with a .model extension, and nothing
    # complains until S5 tries to load it on the server.  Checking that it is a
    # torch state dict with the keys this model has is the difference between
    # finding that here and finding it there.
    "$PYTHON" - "$1" <<'PYEOF'
import sys
try:
    import torch
except ModuleNotFoundError:
    sys.exit(0)  # cannot check without torch; not a reason to fail
path = sys.argv[1]

# Checked before torch sees it.  A Google Drive download that hits a quota or a
# "confirm you are not a robot" page is *HTML with status 200*, and gdown saves
# it without complaint -- and torch's complaint is several lines of
# weights_only essay that buries the actual reason.
with open(path, "rb") as handle:
    head = handle.read(512).lstrip()
if head[:1] == b"<" or head[:15].lower().startswith(b"<!doctype html"):
    raise SystemExit(
        "this is an HTML page, not a checkpoint -- the download returned a "
        "Google Drive quota or sign-in page instead of the file"
    )

try:
    state = torch.load(path, map_location="cpu")
except Exception as error:
    first = str(error).splitlines()[0]
    raise SystemExit(f"not a torch checkpoint: {type(error).__name__}: {first}")
if not isinstance(state, dict):
    raise SystemExit(f"expected a state dict, got {type(state).__name__}")
keys = [k for k in state if k.startswith("model.module.")]
if not keys:
    raise SystemExit(
        "no 'model.module.' keys -- this is a torch file but not the LoCoNet "
        f"release. First keys: {sorted(state)[:3]}"
    )
print(f"   verified: {len(state)} keys, all under model.module.")
PYEOF
}

for candidate in "$MANUAL_DIR/loconet/loconet_AVA.model" "$MANUAL_DIR/loconet_AVA.model"; do
    if [[ -f "$candidate" ]]; then
        cp "$candidate" "$OUT/models/loconet_AVA.model"
        note "models/loconet_AVA.model: copied from $candidate"
        break
    fi
done

if [[ ! -f "$OUT/models/loconet_AVA.model" ]]; then
    if "$PYTHON" -c "import gdown" >/dev/null 2>&1; then
        note "fetching LoCoNet's weights from Google Drive with gdown"
        "$PYTHON" -m gdown --id "$LOCONET_GDRIVE_ID" \
            -O "$OUT/models/loconet_AVA.model" >/dev/null 2>&1 || true
    else
        warn "gdown is not installed, so the Google Drive file cannot be fetched here"
        warn "    pip install gdown   (then re-run this script)"
    fi
fi

if [[ -f "$OUT/models/loconet_AVA.model" ]]; then
    note "$(( $(wc -c < "$OUT/models/loconet_AVA.model") / 1024 / 1024 )) MB"
    verify_loconet "$OUT/models/loconet_AVA.model" \
        || die "models/loconet_AVA.model is not the released checkpoint (see above).
Download it in a browser from the repository README, or delete it and re-run."
else
    warn "LoCoNet's weights are NOT in this bundle, and S5 will not run without them."
    warn "They are behind a Google Drive link in the repository's README.  On a"
    warn "machine that can open it, save the file as:"
    warn "    $MANUAL_DIR/loconet/loconet_AVA.model"
    warn "and re-run this script -- it will be copied in."
fi

# --------------------------------------------------------------------------- #
# optional: a wheelhouse, for a server with no PyPI either
# --------------------------------------------------------------------------- #

if [[ "$WITH_WHEELS" == 1 ]]; then
    say "wheels (pip download)"
    # Every extra, so the server can install any stage without reaching PyPI.
    # Some of these are large; the bundle grows to several GB.
    "$PYTHON" -m pip download \
        --dest "$OUT/wheels" \
        "$ROOT" \
        "$ROOT[faces,diarization,asd,tse,asr,paralinguistic,caption]" \
        || warn "pip download did not finish; the bundle has whatever it managed"
    note "$(find "$OUT/wheels" -name '*.whl' | wc -l | tr -d ' ') wheels, $(du -sh "$OUT/wheels" | cut -f1)"
else
    note "skipping wheels; pass --with-wheels if the server has no PyPI either"
fi

# --------------------------------------------------------------------------- #
# optional: a Hugging Face cache
# --------------------------------------------------------------------------- #

if [[ "$WITH_HF" == 1 ]]; then
    say "Hugging Face checkpoints"
    # A file listing what to snapshot, so this is explicit rather than "whatever
    # happens to be cached".  Names come from the stage defaults.
    cat > "$OUT/hf/MODELS.txt" <<'EOF'
# One repo id per line, as the stage configs name them.
BUT-FIT/diarizen-wavlm-large-s80-md-v2
pyannote/wespeaker-voxceleb-resnet34-LM
laion/voice-tagging-whisper
openai/whisper-small
iic/emotion2vec_plus_large
Qwen/Qwen3-VL-8B-Instruct
EOF
    note "set HF_HOME=$OUT/hf and run the snapshot yourself, or use:"
    note "    huggingface-cli download <repo-id> --local-dir $OUT/hf/hub"
    warn "the ClearerVoice and faster-whisper checkpoints are fetched by their own"
    warn "packages on first use and are not covered by this list"
else
    note "skipping Hugging Face; pass --with-hf if the server cannot reach it"
fi

# --------------------------------------------------------------------------- #
# the manifest, and the tarball
# --------------------------------------------------------------------------- #

say "manifest"
"$PYTHON" - "$OUT" <<'PYEOF'
import hashlib, json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
entries = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "MANIFEST.json":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    entries.append({
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
(root / "MANIFEST.json").write_text(json.dumps({
    "entries": entries,
    "count": len(entries),
    "total_bytes": sum(item["bytes"] for item in entries),
}, indent=2) + "\n", encoding="utf-8")
print(f"   {len(entries)} files, "
      f"{sum(item['bytes'] for item in entries) / 1024 / 1024:.0f} MB")
PYEOF

say "packing"
TARBALL="${OUT}.tar"
tar -cf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
note "$TARBALL ($(du -h "$TARBALL" | cut -f1))"

cat <<EOF

Carry $(basename "$TARBALL") to the server, unpack it, and install from it:

    tar xf $(basename "$TARBALL") -C /somewhere
    scripts/setup_server.sh --from-bundle /somewhere/$(basename "$OUT")

MANIFEST.json lists every file with its size and sha256, so what arrived can be
checked against what was sent.  Nothing in the bundle is fetched again on the
server.
EOF
