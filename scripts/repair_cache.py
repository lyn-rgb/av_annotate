#!/usr/bin/env python3
"""Point every repo's ``refs/main`` at the snapshot that holds its weights.

    python scripts/repair_cache.py [--dry-run] [repo ...]

huggingface_hub rewrites ``refs/main`` every time it resolves a revision from the
network.  On a machine whose route to the Hub is unreliable that is a trap: a run
which resolves ``main``, downloads the small files of the new revision and then
dies part way through the large ones leaves the ref naming a snapshot that has
the configs and not the weights.  Every later run resolves to *that* revision,
fails to find the weights, and reports them missing -- while they sit on disk
under the revision that worked a minute earlier::

    ref     main = b'0c351dd0...'
    snapshots  0c351dd0...:  9 files,  0.0 GB
               aea7f62f...: 17 files, 17.5 GB

The repair is arithmetic.  A snapshot that stopped part way through is smaller
than one that did not, so the ref belongs on the largest, and that is all this
does.  It reads no weights -- only file sizes -- so it is quick even on a cache
of tens of gigabytes.

Run it after any run that touched the network and failed; ``check_offline.py``
is what says whether you need to.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def total_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def hub_root() -> Path:
    """The cache the run will use, resolved the way ``check_offline`` resolves it.

    Deliberately not ``~/.cache/huggingface``: that is huggingface_hub's own
    default and *not* what a batch uses, because ``run_batch.sh`` points
    ``HF_HOME`` at the checkout.  Repairing one cache while the run reads
    another is a repair that reports success and changes nothing.
    """

    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path(__file__).resolve().parents[1] / "models" / "hf" / "hub"


def repair(repo_dir: Path, *, dry_run: bool) -> str | None:
    """Repoint one repo, or return None if there is nothing to do."""

    snapshots = [path for path in (repo_dir / "snapshots").glob("*") if path.is_dir()]
    if not snapshots:
        return None

    sizes = {path.name: total_bytes(path) for path in snapshots}
    best = max(sizes, key=lambda name: sizes[name])

    ref = repo_dir / "refs" / "main"
    current = ref.read_bytes().decode() if ref.is_file() else None
    if current == best:
        return None

    was = f"{current[:12]} ({human(sizes.get(current or '', 0))})" if current else "unset"
    if not dry_run:
        ref.parent.mkdir(parents=True, exist_ok=True)
        # ``write_bytes`` rather than ``write_text``: huggingface_hub uses this
        # file's bytes *verbatim* as the snapshot directory name, so a trailing
        # newline names a directory that does not exist and every lookup misses.
        # It is the difference between a repair and a second fault.
        ref.write_bytes(best.encode())

    return f"{repo_dir.name}: {was} -> {best[:12]} ({human(sizes[best])})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repos", nargs="*", help="only these repo ids (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="say what it would do")
    args = parser.parse_args(argv)

    root = hub_root()
    if not root.is_dir():
        print(f"no cache at {root}", file=sys.stderr)
        return 2

    wanted = (
        {f"models--{repo.replace('/', '--')}" for repo in args.repos}
        if args.repos
        else None
    )

    changed = 0
    for repo_dir in sorted(root.glob("models--*")):
        if wanted is not None and repo_dir.name not in wanted:
            continue
        line = repair(repo_dir, dry_run=args.dry_run)
        if line is not None:
            print(("would fix  " if args.dry_run else "repointed  ") + line)
            changed += 1

    if changed == 0:
        print(f"every ref under {root} already names its largest snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
