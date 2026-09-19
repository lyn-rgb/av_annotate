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
# **ModelScope is tried before the mirror**, because the mirror is not a way
# around a blocked network -- hf-mirror.com is itself abroad (it resolves to a
# Sakura address in Japan), so it is blocked by exactly the things that block
# huggingface.co.  ModelScope is domestic, carries most of these checkpoints
# under the same repo ids, and is reachable on precisely the networks where the
# mirror is not.  Where ModelScope has no copy the mirror is still tried.
#
# The result is written in the layout `huggingface_hub` reads rather than into
# a folder of its own, so the repo ids in `configs/` keep working unchanged.
# See the long note above `ms_fetch` for the layout and how it was verified.
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

# curl does not read the macOS system proxy; this does.  See lib.sh.
# shellcheck source=lib.sh
source "$ROOT/scripts/lib.sh"
MODELS="$ROOT/models"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ONLY="all"
# The interpreter the packages actually went into.  A venv beside the checkout
# is what this project's own docs use, and preferring the system python3 over it
# is how the script ends up reporting "modelscope is not installed" about an
# environment where it plainly is.  Set PYTHON to override.
if [[ -z "${PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
fi
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

# The directory huggingface_hub reads, which is what a pre-fetched checkpoint
# has to be written into for the stages to find it by repo id.
HUB="$HF_HOME/hub"
# ModelScope's own API.  Plain HTTPS, so it needs no package installed to use.
MS_API="https://www.modelscope.cn/api/v1/models"

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
note "interpreter:           $PYTHON"

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

# --------------------------------------------------------------------------- #
# ModelScope: the same checkpoints, from a host that is not abroad
# --------------------------------------------------------------------------- #
#
# hf-mirror.com is the right default and is *still a foreign host* -- it
# resolves to 160.16.86.14, a Sakura address in Japan, so it is blocked by
# exactly the things that block huggingface.co.  ModelScope is domestic and
# carries most of these checkpoints under the same repo ids, so it is tried
# first and the mirror is the fallback.
#
# Files are written in the layout `huggingface_hub` reads, not into a folder of
# their own.  The stages name their checkpoints by repo id
# ("Qwen/Qwen3-VL-8B-Instruct"), because that is the argument `from_pretrained`
# takes, and a plain directory would mean editing every config to name a path
# instead.  The layout:
#
#     <hub>/models--Qwen--Qwen3-VL-8B-Instruct/
#         refs/main                  the revision the snapshot is filed under
#         snapshots/<revision>/...   the files themselves
#
# The blobs/ directory and the symlinks the real cache uses are an optimisation
# -- they let several revisions share one copy of a 4 GB shard -- and are not
# required.  Verified against huggingface_hub 1.32.0 that refs/main plus
# snapshots/<rev>/<file> resolves in offline mode through both
# `try_to_load_from_cache` and `hf_hub_download`, which is the pair
# `from_pretrained` calls underneath.
#
# Resolution offline additionally needs HF_HUB_OFFLINE=1; see the note printed
# at the end.  Without it, snapshot_download still reaches for the network to
# resolve "main" however complete the cache is.

# $1 repo id to download from
# $2 "cache" for the huggingface_hub layout, "dir" for a plain directory
# $3 where to write it (the hub directory, or the directory itself)
# $4 what it is for, for the status line
# $5 the repo id to file it under, when that differs from where it came from.
#    ModelScope's mirrors are not always named like the original -- whisper-small
#    lives there as `openai-mirror/whisper-small` -- and the cache directory is
#    what the loader looks up, so it has to carry the *original* id or the
#    download lands somewhere nothing asks for.
ms_fetch() {
    local repo="$1" kind="$2" where="$3" why="$4" as="${5:-$1}"
    printf '   %-46s ' "$as"
    # Progress and the reason for a failure both go to stderr, so the status
    # line above stays on one line on stdout.
    if "$PYTHON" - "$repo" "$kind" "$where" "$MS_API" "$as" <<'PYEOF'
import fnmatch, hashlib, json, subprocess, sys, urllib.error, urllib.parse, urllib.request
from pathlib import Path

repo, kind, where, api_base, as_repo = (
    sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5],
)
API = f"{api_base}/{repo}"
CHUNK = 1 << 22
ANNOUNCE = 50 * 1024 * 1024  # big enough that watching it is worth the lines


def note(message):
    print(message, file=sys.stderr, flush=True)


def fail(message):
    note(message)
    sys.exit(1)


# Domestic, and reached directly.  Through a proxy this is both the slow way
# round and -- on the network this whole path exists for -- the broken way
# round, since the proxy is what is down.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "avannotate"})
    with opener.open(request, timeout=60) as response:
        return json.load(response)


try:
    # The branch the repo ids resolve to, which is not always "master".
    branch = get_json(API).get("Data", {}).get("Revision") or "master"
except (urllib.error.URLError, ValueError) as error:
    fail(f"not on ModelScope: {type(error).__name__}: {error}")


def walk(root=""):
    query = urllib.parse.urlencode({"Revision": branch, "Root": root})
    found = []
    for entry in get_json(f"{API}/repo/files?{query}")["Data"]["Files"]:
        if entry["Type"] == "tree":
            found.extend(walk(entry["Path"]))
        else:
            found.append(entry)
    return found


try:
    listing = sorted(walk(), key=lambda entry: entry["Path"])
except (urllib.error.URLError, KeyError, ValueError) as error:
    fail(f"could not list its files: {type(error).__name__}: {error}")

if not listing:
    fail("it lists no files")

files = listing

# Training-run leftovers, which inference never reads: laion/voice-tagging-whisper
# ships a 1.8 GB optimizer.pt beside 0.9 GB of actual weights, and on a slow link
# that is not a rounding error.  Nothing here is a weight file, so this cannot
# empty a repo of the thing it was wanted for.
TRAINING = ("optimizer*", "scheduler*", "rng_state*", "trainer_state*", "training_args*")

# Alternate weight formats: the same weights in another framework's container.
# whisper-small's flax_model.msgpack is 0.9 GB of JAX that nothing in a PyTorch
# pipeline will ever open.
ALTERNATE = ("tf_model.h5", "flax_model.msgpack", "rust_model.ot")
SAFETENSORS = ("model.safetensors", "model-*.safetensors")
BIN_WEIGHTS = ("pytorch_model.bin", "pytorch_model-*.bin")
PYTORCH_WEIGHTS = SAFETENSORS + BIN_WEIGHTS


def redundant(entry, has_pytorch, has_safetensors):
    name = Path(entry["Path"]).name
    if any(fnmatch.fnmatch(name, pattern) for pattern in TRAINING):
        return True
    # The same weights in two containers.  transformers reads safetensors in
    # preference, so the .bin is dead weight: whisper-small ships both and they
    # are 0.9 GB each.
    if has_safetensors and any(fnmatch.fnmatch(name, p) for p in BIN_WEIGHTS):
        return True
    # Only when a PyTorch file is there to take its place.  A repo that ships
    # nothing else would otherwise arrive with no weights at all, which is a
    # worse outcome than arriving large.
    return has_pytorch and name in ALTERNATE


names = [Path(entry["Path"]).name for entry in files]


def any_match(patterns):
    return any(fnmatch.fnmatch(name, p) for name in names for p in patterns)


has_pytorch = any_match(PYTORCH_WEIGHTS)
has_safetensors = any_match(SAFETENSORS)
skipped = [e for e in files if redundant(e, has_pytorch, has_safetensors)]
files = [e for e in files if not redundant(e, has_pytorch, has_safetensors)]
if not files:
    fail("everything it lists is training state or another framework's weights")
saved = sum(entry.get("Size") or 0 for entry in skipped)


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# The revision the snapshot is filed under.  ModelScope publishes no repo-level
# commit through this API, so this is a digest of the file list -- which is the
# property the cache wants it for: it changes exactly when the contents do.  A
# fixed string would leave a snapshot taken last month looking current.
#
# Of the FULL listing, deliberately, not of `files`.  Digesting the filtered
# list would tie the revision to this script's exclusion rules, so editing the
# skip list above would rename every revision and strand every byte already
# downloaded under the old name -- which is exactly what it did the first time
# this list was touched.  What the revision identifies is what ModelScope has.
manifest = hashlib.sha256()
for entry in listing:
    manifest.update(f"{entry['Path']}\0{entry.get('Sha256') or ''}\n".encode())
revision = manifest.hexdigest()[:40]

if kind == "cache":
    # The huggingface_hub layout, so the repo ids in configs/ keep resolving.
    # Named after as_repo, not repo: see the $5 note above the function.
    repo_dir = where / f"models--{as_repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
else:
    # A plain directory, for the checkpoints that something other than
    # huggingface_hub reads.  clearvoice wants one, and reads it relative to
    # the working directory, so it never sees a cache.
    repo_dir = None
    snapshot = where
snapshot.mkdir(parents=True, exist_ok=True)


def fetch(url, dest):
    """Resume a partial file; start over if resuming is not honoured."""
    resume = ["-C", "-"] if dest.exists() and dest.stat().st_size else []
    for extra in (resume, []):
        result = subprocess.run([
            "curl", "-fsSL", *extra, "--retry", "3", "--retry-delay", "2",
            "--connect-timeout", "30", "--noproxy", "*", "-o", str(dest), url,
        ])
        if result.returncode == 0:
            return True
        # A server that ignores Range fails `-C -` identically every time, so
        # the second pass is what gets past it.
    return False


fetched = kept = 0
for entry in files:
    relative = entry["Path"]
    dest = snapshot / relative
    want = entry.get("Sha256") or ""
    if dest.is_file() and want and sha256_of(dest) == want:
        kept += 1
        continue

    size = entry.get("Size") or 0
    if size >= ANNOUNCE:
        note(f"       {relative}  ({size / 1024 ** 3:.1f} GB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/repo?" + urllib.parse.urlencode(
        {"Revision": branch, "FilePath": relative}
    )
    if not fetch(url, dest):
        fail(f"could not download {relative}")

    # Checked per file rather than once at the end: a 4 GB shard that arrived
    # short is worth catching before another 4 GB is spent on top of it.
    if want and sha256_of(dest) != want:
        dest.unlink()
        fail(f"{relative} does not match the sha256 ModelScope publishes for it")
    fetched += 1

# Written last, so an interrupted run leaves its revision unreferenced: the
# next run resumes into it rather than trusting a snapshot that is not whole.
#
# No trailing newline, and that is load-bearing.  The lookup reads this file and
# uses the bytes verbatim as the snapshot directory name -- there is no strip:
#
#     with open(revision_file) as f:
#         revision = f.read()
#     if revision not in os.listdir(snapshots_dir):
#         return None
#
# so a file ending in "\n" names a directory that does not exist and every
# lookup misses, silently, with the files sitting right there.  Verified
# against huggingface_hub 1.32.0's try_to_load_from_cache, which is the
# function from_pretrained calls.
#
# There is no ref to write in the "dir" case: clearvoice looks for a checkpoint
# file by name, not for a revision.
if repo_dir is not None:
    repo_dir.joinpath("refs").mkdir(parents=True, exist_ok=True)
    repo_dir.joinpath("refs", "main").write_text(revision)

total = sum((snapshot / entry["Path"]).stat().st_size for entry in files)
summary = (f"       {len(files)} files, {total / 1024 ** 3:.1f} GB "
           f"({fetched} fetched, {kept} already had it)")
if skipped:
    summary += f"; {len(skipped)} training files skipped, {saved / 1024 ** 3:.1f} GB"
note(summary)
PYEOF
    then
        echo "ok   ($why, from ModelScope)"
        return 0
    fi
    # Deliberately not recorded in FAILED: the caller falls back to the mirror
    # and it is that attempt which decides whether this checkpoint arrived.
    return 1
}

# ModelScope where it has the repo, the Hugging Face mirror otherwise.  The
# order matters on exactly the network this is for: the mirror is foreign, so
# it is the fallback that usually fails, not the one that usually works.
fetch_checkpoint() {
    local hf_repo="$1" why="$2" ms_repo="${3:-$1}"
    # ModelScope's mirror of a repo is sometimes filed under another name, so
    # an alias is tried first when one is given.  Either way the files land
    # under hf_repo, which is the id the loaders ask for.
    if [[ "$ms_repo" != "$hf_repo" ]] \
        && ms_fetch "$ms_repo" cache "$HUB" "$why" "$hf_repo"; then
        return 0
    fi
    ms_fetch "$hf_repo" cache "$HUB" "$why" "$hf_repo" && return 0
    note "       (no copy on ModelScope; trying the Hugging Face mirror)"
    fetch_hf "$hf_repo" "$why"
}

# For the checkpoints that ModelScope's own package has to place, because
# something other than huggingface_hub is what reads them -- funasr, in
# practice.  Skipped rather than failed when the package is absent: funasr will
# fetch the same file from the same host on first use, so a missing package
# costs a wait later, not a broken run.
#
# `cache_dir` is fixed to match what the closing note tells the operator to set
# MODELSCOPE_CACHE to.  If those two disagree, the download lands somewhere
# nothing reads and the first run pays for it again.
fetch_modelscope_native() {
    local repo="$1" why="$2"
    printf '   %-46s ' "$repo"
    if ! "$PYTHON" -c "import modelscope" >/dev/null 2>&1; then
        echo "skipped (modelscope is not installed)"
        note "       funasr fetches this from ModelScope on first use, which needs"
        note "       the package anyway:  pip install modelscope"
        return 0
    fi
    if "$PYTHON" -c "
from modelscope import snapshot_download
snapshot_download('$repo', cache_dir='$MODELS/modelscope')
" >/dev/null 2>&1; then
        echo "ok   ($why, from ModelScope)"
        return 0
    fi
    echo "FAIL ($why)"
    FAILED+=("$repo")
    return 1
}

fetch_url() {
    local url="$1" target="$2" floor="$3" why="$4"
    printf '   %-46s ' "$(basename "$target")"
    if [[ -f "$target" ]] && (( $(wc -c < "$target") >= floor )); then
        echo "already there ($why)"
        return 0
    fi
    mkdir -p "$(dirname "$target")"
    # --connect-timeout, like every other fetch here: without it a host that
    # accepts nothing hangs until the kernel gives up, which on a blocked route
    # is where these downloads go to die quietly.  No --max-time, because the
    # files this fetches are large and a slow transfer is not a failed one.
    if curl -fsSL -C - --retry 3 --retry-delay 2 --connect-timeout 30 \
        -o "$target" "$url" 2>/dev/null \
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
    # Not on ModelScope under any name; this one needs the mirror, and the
    # mirror needs a network that can reach Japan.
    fetch_hf "BUT-FIT/diarizen-wavlm-large-s80-md-v2" "the checkpoint"
    # Pulled by DiariZen's own code alongside the checkpoint, so it is easy to
    # miss when counting what has to be available offline.
    fetch_checkpoint "pyannote/wespeaker-voxceleb-resnet34-LM" "its embedding model"
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
    # clearvoice fetches this itself, with
    # `snapshot_download(repo_id="alibabasglab/AV_MossFormer2_TSE_16K",
    # local_dir=checkpoint_dir)` -- and it skips the download entirely when
    # `checkpoint_dir/last_best_checkpoint` is already there.  So the shape it
    # wants is a plain directory, not the cache layout, which is why this one
    # is fetched with "dir".
    #
    # It is Alibaba's own model, so ModelScope has it under the same id.
    ms_fetch "alibabasglab/AV_MossFormer2_TSE_16K" dir \
        "$MODELS/clearvoice/AV_MossFormer2_TSE_16K" "the extraction checkpoint"
    warn "S7 will not find that on its own.  clearvoice reads its checkpoint"
    warn "from a path relative to the WORKING DIRECTORY --"
    warn "    checkpoint_dir/AV_MossFormer2_TSE_16K"
    warn "-- it hardcodes that default, and its public API takes no"
    warn "checkpoint_dir argument, so there is nowhere to tell it otherwise."
    warn "Point the run at what was fetched, from wherever the batch runs:"
    warn "    mkdir -p checkpoint_dir"
    warn "    ln -s '$MODELS/clearvoice/AV_MossFormer2_TSE_16K' \\"
    warn "          checkpoint_dir/AV_MossFormer2_TSE_16K"
fi

# --------------------------------------------------------------------------- #
# S8 -- speech recognition
# --------------------------------------------------------------------------- #

if wants s8-asr; then
    say "S8: faster-whisper"
    # The CTranslate2 conversion, which is what faster-whisper actually loads --
    # not openai/whisper-large-v3, which is the PyTorch original.
    # faster-whisper resolves the id "large-v3" to this repo through
    # snapshot_download, so the cache layout written here is what it reads.
    fetch_checkpoint "Systran/faster-whisper-large-v3" "large-v3, CTranslate2"
fi

# --------------------------------------------------------------------------- #
# S9 -- paralinguistic tagging
# --------------------------------------------------------------------------- #

if wants s9-paralinguistic; then
    say "S9: the three taggers"

    # emotion2vec is the one checkpoint that does not go through
    # huggingface_hub at all: funasr loads it by ModelScope repo id --
    # `AutoModel("iic/emotion2vec_plus_large")` -- and reads ModelScope's own
    # cache, so the layout written by fetch_checkpoint is the wrong shape for
    # it.  The modelscope package writes the right one.
    fetch_modelscope_native "iic/emotion2vec_plus_large" "emotion"

    fetch_checkpoint "laion/voice-tagging-whisper" "delivery"
    # The tagger ships no processor of its own; the adapter uses whisper-small's.
    # On ModelScope it is filed under openai-mirror/, which is what the alias
    # argument is for -- the files still land under openai/whisper-small.
    fetch_checkpoint "openai/whisper-small" "delivery, processor only" \
        "openai-mirror/whisper-small"

    # Zenodo, which is abroad like everything else that is not ModelScope.
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
    # Alibaba's own model, so it is on ModelScope under the same id -- which is
    # the difference between this being the one 16 GB download that works and
    # the one that does not.
    fetch_checkpoint "Qwen/Qwen3-VL-8B-Instruct" "captioning"
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
needs them set as this script had them:

    export HF_HOME="$HF_HOME"
    export HF_ENDPOINT="$ENDPOINT"

Then check it took:

    "$PYTHON" -m avannotate.cli doctor

**On a machine with no route abroad, add these two.**  They are not set by the
script because each of them turns something off, and turning it off is a
decision rather than a default:

    export HF_HUB_OFFLINE=1
    export MODELSCOPE_CACHE="$MODELS/modelscope"

HF_HUB_OFFLINE=1 is what makes the pre-fetched checkpoints actually get used.
Without it \`from_pretrained\` and \`snapshot_download\` still ask the network to
resolve "main" before they look in the cache, so a machine with no route out
fails on a checkpoint that is sitting right there on disk -- which reads as the
download never having happened.  Set it only once everything above says "ok";
with it set, a checkpoint this script did not fetch cannot be fetched later.

MODELSCOPE_CACHE is where funasr looks for emotion2vec, and it has to match the
path this script downloaded into.  The one above does.

EOF

if (( ${#FAILED[@]} )); then
    echo
    bad "not everything arrived:"
    for item in "${FAILED[@]}"; do bad "  $item"; done
    exit 1
fi
echo "everything this script can fetch is here"
