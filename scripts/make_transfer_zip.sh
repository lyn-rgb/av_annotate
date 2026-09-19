#!/usr/bin/env bash
#
# One archive to carry the pipeline to a server.
#
#   scripts/make_transfer_zip.sh [--out FILE] [--minimal]
#
#   --out FILE   where to write it (default: ../avannotate-transfer.zip)
#   --minimal    carry only what the server cannot fetch for itself (~170 MB
#                rather than ~26 GB); run download_models.sh there for the rest
#
# The default destination is outside the checkout on purpose.  An archive
# written into the thing it is archiving gets included in the next one, and a
# 22 GB file inside a 22 GB file is a mistake that only shows up once.
#
# --minimal is usually the right one.  Every checkpoint except two is on
# ModelScope, which download_models.sh fetches from directly on the machine
# that needs it -- and fetching 26 GB there is better than uploading 26 GB
# here, because an upload that fails at 90% costs the whole thing while a
# download that fails resumes.  What cannot be fetched is small:
#
#   models/loconet/loconet_AVA.model   Google Drive only.  Not on ModelScope
#                                      under any spelling and not on Hugging
#                                      Face either -- the HF repo that looks
#                                      like it might carry it is a model card
#                                      pointing back at the same Drive link.
#   models/loconet/LoCoNet_ASD/        no mirror anywhere; see download_models.sh
#   models/yunet.onnx                  a GitHub release asset; ModelScope has
#                                      only a LiteRT conversion of it
#
# Those three are about 170 MB together, which is the whole point.
#
# What is left out, and why each is not "necessary content":
#
#   .venv/                 built for this machine's OS and CPU; on the server it
#                          is not merely useless but actively misleading, since
#                          its interpreter is executable and will not run.
#                          setup_venv.sh builds the right one there.
#   __pycache__/, *.pyc    regenerated on first import.
#   build/, *.egg-info/    pip's leftovers from installing this project.
#   .pytest_cache/ etc.    tool caches.
#   .DS_Store              macOS folder metadata.
#   outputs/, work/        pipeline output, not input.
#   offline-bundle/        a transfer artifact in its own right.
#   models/**/.lock/       ModelScope's advisory locks, meaningless elsewhere.
#   data/README.md         despite the name this is a ZIP of data/examples/,
#                          byte-identical to the directory beside it.  The
#                          directory is the one the code reads.
#
# .git/ IS included.  It is 2.4 MB, and it is what lets someone on the server
# answer "which revision is this?" without asking.
#
# The three Qwen shards are 4.6 GB each, past the 4 GB that classic zip can
# address, so this writes Zip64 -- which every unzip from the last fifteen years
# reads, and which is checked at the end rather than assumed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PYTHON="$ROOT/.venv/bin/python"
MODE="full"
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --minimal) MODE="minimal"; shift ;;
        -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Named for what it holds, so building one does not quietly overwrite the other.
if [[ -z "$OUT" ]]; then
    if [[ "$MODE" == "minimal" ]]; then
        OUT="$(cd "$ROOT/.." && pwd)/avannotate-minimal.zip"
    else
        OUT="$(cd "$ROOT/.." && pwd)/avannotate-transfer.zip"
    fi
fi

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

say "packing $ROOT"
note "into $OUT"

rm -f "$OUT"

"$PYTHON" - "$ROOT" "$OUT" "$MODE" <<'PYEOF'
import sys, time, zipfile
from pathlib import Path

root, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

# What the server cannot fetch for itself, and therefore what --minimal carries.
# Everything else under models/ is on ModelScope and is better downloaded there:
# a download resumes, a failed upload does not.
CARRY = ("models/loconet", "models/yunet.onnx")

# Skipped by any path component, so a rule holds however deep the match is.
# `.egg-info` is matched as a suffix rather than a name because the directory is
# `avannotate.egg-info` and its *contents* are what rglob yields -- checking the
# file's own name would let PKG-INFO and SOURCES.txt through, and those carry
# absolute paths from whichever machine built them.
EXCLUDE_PARTS = {
    ".venv", "__pycache__", "build", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "outputs", "work", "offline-bundle", ".lock",
}
EXCLUDE_NAMES = {".DS_Store", "data/README.md"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


def excluded(relative: Path) -> bool:
    return any(
        part in EXCLUDE_PARTS or part.endswith(".egg-info")
        for part in relative.parts
    )

# Deflate buys nothing on a 4.6 GB safetensors shard and costs minutes; it earns
# its keep on configs and code.  So: text always deflated, and anything large
# enough to be a model weight stored as-is.
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".toml", ".sh",
                 ".yaml", ".yml", ".cfg", ".ini", ".csv", ".gitignore"}
STORE_OVER = 8 * 1024 * 1024


def carried(relative: Path) -> bool:
    text = str(relative)
    return any(text == item or text.startswith(item + "/") for item in CARRY)


def wanted(path: Path) -> bool:
    rel = path.relative_to(root)
    if excluded(rel):
        return False
    if str(rel) in EXCLUDE_NAMES or path.name in EXCLUDE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if mode == "minimal" and rel.parts[0] == "models" and not carried(rel):
        return False
    return path.is_file() and not path.is_symlink()


files = sorted(p for p in root.rglob("*") if wanted(p))
if not files:
    sys.exit("nothing to pack -- every file was excluded, which is not right")

total = sum(p.stat().st_size for p in files)
print(f"   {len(files)} files, {total / 1024**3:.2f} GB uncompressed")

started = time.time()
written = 0
next_report = 0

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
    for path in files:
        rel = path.relative_to(root)
        size = path.stat().st_size
        if path.suffix in TEXT_SUFFIXES or size < STORE_OVER:
            method = zipfile.ZIP_DEFLATED
        else:
            method = zipfile.ZIP_STORED
        archive.write(path, arcname=f"avannotate/{rel}", compress_type=method)
        written += size

        # A line per gigabyte, because 22 GB of quiet is indistinguishable from
        # a hang, and this runs for minutes.
        if written >= next_report:
            elapsed = time.time() - started
            print(f"   {written / 1024**3:5.1f} / {total / 1024**3:.1f} GB"
                  f"  ({100 * written / total:5.1f}%)  {elapsed:5.0f}s", flush=True)
            next_report = written + 1024**3

print(f"   done in {time.time() - started:.0f}s")
PYEOF

[[ -f "$OUT" ]] || die "no archive was written"

say "checking the archive"
note "$(du -h "$OUT" | cut -f1)  $OUT"

"$PYTHON" - "$OUT" <<'PYEOF'
import sys, zipfile
from pathlib import Path

out = Path(sys.argv[1])
with zipfile.ZipFile(out) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"corrupt entry: {bad}")

    names = archive.infolist()
    biggest = max(names, key=lambda i: i.file_size)
    print(f"   {len(names)} entries, integrity check passed")

    # Zip64 is the thing that would silently not work, so it is read back
    # rather than assumed: read the last byte of the largest member, which is
    # past 4 GB and therefore past what a classic zip could address.
    with archive.open(biggest) as handle:
        handle.seek(biggest.file_size - 1)
        tail = handle.read(1)
    print(f"   largest member readable at its end: {biggest.filename}")
    print(f"     {biggest.file_size / 1024**3:.2f} GB, last byte 0x{tail.hex()}")

    # What travelled, so the contents are not a mystery until the server has
    # it.  The archive has exactly one top-level directory, so the breakdown
    # worth printing is the level inside it -- that is what someone unzipping
    # actually looks at.
    roots = sorted({i.filename.split("/")[0] for i in names})
    print(f"   unpacks to: {', '.join(roots)}/")

    sizes: dict[str, int] = {}
    for item in names:
        parts = item.filename.split("/")
        key = parts[1] if len(parts) > 2 else "(files at the root)"
        sizes[key] = sizes.get(key, 0) + item.file_size
    for key, size in sorted(sizes.items(), key=lambda pair: -pair[1]):
        print(f"     {key:<24} {size / 1024**3:7.2f} GB")
PYEOF

say "what to do with it"
cat <<EOF

    scp $(basename "$OUT") server:~/
    ssh server 'unzip -q ~/$(basename "$OUT") -d ~/ && cd ~/avannotate'

EOF

if [[ "$MODE" == "minimal" ]]; then
    cat <<'EOF'
Then, on the server, in this order:

    scripts/setup_venv.sh --gpus 4      # the environment
    scripts/download_models.sh          # ~26 GB, from ModelScope
    scripts/setup_venv.sh --gpus 4      # again, for DiariZen

The third command looks redundant and is not, which is worth explaining.
DiariZen is a repository rather than a package, so setup_venv.sh installs it
from models/DiariZen -- and on a fresh server that directory does not exist
until download_models.sh puts it there.  The second run is fast: everything
else is already installed, so it only builds DiariZen.

Both scripts are idempotent, which is also what makes them safe to re-run
after a download that stopped part way through.

download_models.sh is where every weight comes from, so nothing large needed
uploading; the two things with no source anywhere -- the LoCoNet weights and
its checkout -- are in this archive, which is the only reason it is 162 MB
rather than a megabyte.

Finally:

    python -m avannotate.cli doctor --configs-dir configs

EOF
else
    cat <<'EOF'
Then, on the server:

    scripts/setup_venv.sh --gpus 4

which is the only build step -- the weights are already here, so
download_models.sh has nothing left to fetch, and setup_server.sh's clones are
unnecessary because both checkouts travelled inside this archive.

EOF
fi
