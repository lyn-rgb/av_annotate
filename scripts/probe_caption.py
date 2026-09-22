#!/usr/bin/env python3
"""Build S10's captioner with nothing in between, and say what the cache holds.

    python scripts/probe_caption.py

The batch catches every failure and records the message, which is right for a
corpus and useless for a diagnosis::

    OSError: We couldn't connect to 'https://hf-mirror.com' to load the files,
    and couldn't find them in the cached files.

That names a host and never names a file -- and hf_hub raises it in two
different situations that need opposite answers.  Either a file is genuinely
absent from the cache, in which case the network is irrelevant and retrying
forever will not help, or the file is there and the network was consulted
anyway, in which case it is a connection problem and nothing is missing.

This calls the same builder the stage calls, with nothing in between, so the
traceback says which ``from_pretrained`` it was.  It also lists what the cache
actually holds, because more often than not the answer is in the difference
between that list and what the model's repository has.

Reads the stage's own config file rather than taking arguments, so it cannot
drift from what a batch would do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avannotate.caption.model import build_captioner  # noqa: E402
from avannotate.stages.base import load_config_file  # noqa: E402


def cache_report(model: str) -> None:
    """What the hub cache holds for one repo, revision by revision."""

    home = os.environ.get("HF_HOME")
    if not home:
        print("   HF_HOME    not set -- huggingface_hub would use ~/.cache/huggingface")
        return

    snapshot = Path(home) / "hub" / ("models--" + model.replace("/", "--"))
    print(f"   cache      {snapshot}")
    if not snapshot.is_dir():
        print("              the repository is not in the cache at all")
        return

    refs = snapshot / "refs"
    if refs.is_dir():
        for path in sorted(refs.iterdir()):
            # The bytes, not a repr: a trailing newline here is a real failure
            # mode -- it becomes part of the directory name hf_hub looks for --
            # and repr would hide it behind an escape.
            print(f"   ref        {path.name} = {path.read_bytes()!r}")

    for revision in sorted((snapshot / "snapshots").glob("*")):
        files = sorted(path for path in revision.rglob("*") if path.is_file())
        total = sum(path.stat().st_size for path in files)
        print(f"   snapshot   {revision.name}: {len(files)} files, {total / 1e9:.1f} GB")
        for path in files:
            size = path.stat().st_size
            print(f"                {size:>12,}  {path.relative_to(revision)}")


def main() -> int:
    mapping = load_config_file(ROOT / "configs" / "s10.caption.json")
    model = str(mapping.get("model", ""))
    print(f"   model      {model}")
    cache_report(model)
    print("   building")
    print()

    # Deliberately uncaught.  The traceback is the whole point of this script:
    # it names the call that failed, which the batch's recorded message cannot.
    captioner = build_captioner(
        {
            "backend": mapping.get("backend"),
            "model": mapping.get("model"),
            "device": mapping.get("device"),
            "dtype": mapping.get("dtype"),
            "device_map": mapping.get("device_map"),
            "max_new_tokens": mapping.get("max_new_tokens"),
            "seed": mapping.get("seed"),
        }
    )
    print(f"   built      {type(captioner).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
