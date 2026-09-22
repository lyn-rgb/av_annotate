#!/usr/bin/env python3
"""Seed torch's hub cache with the VGGish weights LoCoNet's checkpoint carries.

Building LoCoNet makes the repository's in-tree ``torchvggish`` fetch VGGish's
pretrained weights from a GitHub release -- and LoCoNet's own checkpoint then
overwrites every one of them, because it carries the same tensors under
``model.module.model.audioEncoder.``.  So the download is waste, and on a
machine that cannot reach GitHub it is a hard failure at the first video
rather than a slow start.

torch's hub cache is keyed by filename, so putting the checkpoint's copy at
``vggish-10086976.pth`` is enough: torchvggish asks for that name and gets this.

    python scripts/seed_vggish.py [--force]

Run it after ``download_models.sh`` has put the checkpoint in place.  It is
idempotent, and it says where it looked when it cannot find something -- which
is the part a copy-pasted snippet gets wrong, since that one resolves its paths
against whatever directory you happened to be standing in.

It is also what ``setup_venv.sh`` calls, so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Where the checkpoint keeps the encoder, as the released file spells it.
CHECKPOINT_PREFIX = "model.module.model.audioEncoder."

#: The name torchvggish asks for.  Not a choice -- see the module docstring.
CACHE_FILENAME = "vggish-10086976.pth"


def repo_root() -> Path:
    """The checkout this script lives in, whatever the working directory is."""
    return Path(__file__).resolve().parents[1]


def checkpoint_path() -> Path:
    return repo_root() / "models" / "loconet" / "loconet_AVA.model"


def cache_path() -> Path:
    """Where torch's hub looks, honouring TORCH_HOME the way torch does.

    The fallback is the checkout's own ``models/torch`` rather than torch's
    ``~/.cache/torch``, and that is not cosmetic: ``run_batch.sh`` points
    TORCH_HOME there, so a seed written to ``$HOME`` would be one S5 never
    looks at -- and S5 does not degrade when it cannot find the cache, it asks
    torch.hub to download 275 MB from GitHub, which a server with no route
    there cannot do.  Every other path in this script is already relative to
    :func:`repo_root` for the same reason; this one was the exception.
    """

    home = os.environ.get("TORCH_HOME")
    base = Path(home) if home else repo_root() / "models" / "torch"
    return base / "hub" / "checkpoints" / CACHE_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a cache file that is already there",
    )
    args = parser.parse_args(argv)

    import torch  # imported here so --help works without torch installed

    target = cache_path()
    source = checkpoint_path()

    if target.is_file() and not args.force:
        print(f"already seeded: {target} ({target.stat().st_size // 1024 // 1024} MB)")
        return 0

    if not source.is_file():
        print(f"no checkpoint at {source}", file=sys.stderr)
        print(
            "That file is the LoCoNet release, and it is what these weights come\n"
            "from -- there is nothing to seed from without it.  It travels in the\n"
            "transfer archive because it has no other source; see docs/server-setup.md.",
            file=sys.stderr,
        )
        return 1

    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        print(f"{source} is not a state dict", file=sys.stderr)
        return 1

    audio = {
        key[len(CHECKPOINT_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(CHECKPOINT_PREFIX)
    }
    if not audio:
        print(f"no tensors under {CHECKPOINT_PREFIX!r} in {source}", file=sys.stderr)
        print(
            "Either this is not the LoCoNet release, or its layout has changed.\n"
            f"Keys present: {sorted(state)[:3]}",
            file=sys.stderr,
        )
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(audio, target)
    print(f"seeded {target} ({target.stat().st_size // 1024 // 1024} MB, {len(audio)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
