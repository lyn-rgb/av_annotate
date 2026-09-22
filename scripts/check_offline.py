#!/usr/bin/env python3
"""Check that everything a run needs is already on this machine.

    python scripts/check_offline.py

Run it before a batch, and especially before an offline one.  It answers one
question -- would a run find all of its weights without touching the network --
and it answers it in a second rather than after twenty minutes of downloading.

Both halves are checked, because they fail differently:

* **The Hugging Face cache**, which is how the stages name their checkpoints
  (``Qwen/Qwen3-VL-8B-Instruct`` and friends, resolved through ``HF_HOME``).
  For each repo it resolves ``refs/main`` and reports what is under the snapshot
  it points at, so a half-finished download shows up as a small file count
  rather than as a failure three stages later.
* **Everything that is not a repo id**: the LoCoNet checkout and its weights,
  insightface's face pack, YuNet, the ClearerVoice checkpoint, PANNs, the
  DiariZen checkout.  These are named by path in the configs.

Exit status is 0 when everything is present and 1 otherwise, so it can gate a
run in a shell script.

Two notes on why it reads the cache the way it does:

* ``refs/main`` is used **without stripping whitespace**, because that is what
  huggingface_hub does -- the bytes of that file are the snapshot directory
  name.  A trailing newline there means every lookup misses, silently, with the
  files sitting right there.  ``scripts/download_models.sh`` writes it without
  one for that reason.
* The presence of ``HF_HUB_OFFLINE`` does not change what this reports; it
  reports what *would* be found if it were set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

#: Repos the stages load by repo id: which stage wants each, a file that must be
#: there, and the floor the whole snapshot has to clear.
#:
#: The floor is the part that matters, and it was learned from a real miss.  A
#: probe file alone says nothing about whether the *weights* arrived: Qwen3-VL's
#: config.json is 1.5 KB, so a snapshot holding nine small files and none of the
#: four 4 GB shards passed as "ok" at 11 MB.  The stage then spent twenty
#: minutes fetching what the check had just called present.
#:
#: The floors are deliberately far below the real sizes -- they are there to
#: catch "the weights are absent", not to police a few megabytes of difference
#: between revisions.
CACHED = (
    ("Qwen/Qwen3-VL-8B-Instruct", "S10 captioning", "config.json", 10_000_000_000),
    ("Systran/faster-whisper-large-v3", "S8 ASR", "model.bin", 2_000_000_000),
    ("laion/voice-tagging-whisper", "S9 delivery", "model.safetensors", 500_000_000),
    ("openai/whisper-small", "S9 delivery processor", "config.json", 500_000_000),
    # `pytorch_model.bin`, not `configuration.json`: that name is in the copy
    # ModelScope serves, and the real Hub repo does not have it -- so probing for
    # it reported a download as broken when it was complete and S4 was running
    # off it.
    ("pyannote/wespeaker-voxceleb-resnet34-LM", "S4 embedding", "pytorch_model.bin", 10_000_000),
    ("BUT-FIT/diarizen-wavlm-large-s80-md-v2", "S4 diarization", "pytorch_model.bin", 100_000_000),
)

#: Paths inside the checkout, for the weights and code that are not repo ids.
LOCAL = (
    ("models/insightface/models/buffalo_l/det_10g.onnx", "S1/S3 face detection"),
    ("models/insightface/models/buffalo_l/w600k_r50.onnx", "S1/S3 identity vectors"),
    ("models/yunet.onnx", "S1 alternative detector"),
    ("models/loconet/loconet_AVA.model", "S5 weights"),
    # Under model/, not at the checkout root -- which is where the adapter
    # imports it from, and the path is easy to get wrong from memory.
    ("models/loconet/LoCoNet_ASD/model/loconet_encoder.py", "S5 checkout"),
    ("models/DiariZen/diarizen/pipelines/inference.py", "S4 checkout"),
    ("models/clearvoice/AV_MossFormer2_TSE_16K/last_best_checkpoint.pt", "S7 checkpoint"),
    ("models/panns/Cnn14_mAP=0.431.pth", "S9 events"),
    # Under TORCH_HOME, which run_batch.sh points at the checkout.  Listed here
    # because its absence is not a slow start: torchvggish asks torch.hub for
    # it, and the GitHub release it would fetch from is unreachable on a server
    # like this one.  It is also the one entry that is cheap to put back --
    # `scripts/seed_vggish.py` copies it out of the LoCoNet checkpoint in
    # seconds -- which is why the label says so.
    ("models/torch/hub/checkpoints/vggish-10086976.pth", "S5 audio frontend (seed_vggish.py)"),
)


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def snapshot_dir(hub: Path, repo: str) -> Path | None:
    """The snapshot huggingface_hub would resolve for this repo, or None."""
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    refs = repo_dir / "refs" / "main"
    if not refs.is_file():
        return None
    # Deliberately not stripped.  huggingface_hub uses the file's bytes verbatim
    # as the snapshot directory name, so a trailing newline names a directory
    # that does not exist and every lookup misses.
    revision = refs.read_text()
    candidate = repo_dir / "snapshots" / revision
    return candidate if candidate.is_dir() else None


def check_cache(hub: Path) -> list[str]:
    print(f"\n  Hugging Face cache: {hub}")
    if not hub.is_dir():
        print("    does not exist -- set HF_HOME, or run scripts/download_models.sh")
        return [repo for repo, _, _ in CACHED]

    missing: list[str] = []
    for repo, stage, probe, floor in CACHED:
        snapshot = snapshot_dir(hub, repo)
        if snapshot is None:
            print(f"    MISS  {repo:<44} {stage}")
            missing.append(repo)
            continue
        files = [f for f in snapshot.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        if not (snapshot / probe).is_file():
            print(f"    PART  {repo:<44} {stage}: no {probe} ({human(size)})")
            missing.append(repo)
            continue
        if size < floor:
            print(
                f"    PART  {repo:<44} {stage}: only {human(size)}, "
                f"expected at least {human(floor)}"
            )
            missing.append(repo)
            continue
        print(f"    ok    {repo:<44} {stage}  ({len(files)} files, {human(size)})")
    return missing


def check_local(root: Path) -> list[str]:
    print(f"\n  Checkout: {root}")
    missing: list[str] = []
    for relative, stage in LOCAL:
        path = root / relative
        if path.is_file():
            print(f"    ok    {relative:<52} {human(path.stat().st_size):>9}  {stage}")
        else:
            print(f"    MISS  {relative:<52} {'':>9}  {stage}")
            missing.append(relative)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the checkout (default: the one this script is in)",
    )
    parser.add_argument(
        "--hub",
        type=Path,
        default=None,
        help="the huggingface_hub cache (default: $HF_HOME/hub, else <root>/models/hf/hub)",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if args.hub is not None:
        hub = args.hub
    else:
        home = os.environ.get("HF_HOME")
        hub = (Path(home) / "hub") if home else (root / "models" / "hf" / "hub")

    print("what a run would find locally")
    if os.environ.get("HF_HUB_OFFLINE"):
        print("  (HF_HUB_OFFLINE is set)")
    else:
        print("  (HF_HUB_OFFLINE is not set -- a stage may still reach the network)")

    missing = check_local(root) + check_cache(hub)

    print()
    if missing:
        print(f"{len(missing)} missing.  A run will fail on these:")
        for item in missing:
            print(f"    {item}")
        print("\nscripts/download_models.sh fetches the ones that have a source.")
        return 1

    print("everything a run needs is present.")
    if not os.environ.get("HF_HUB_OFFLINE"):
        print(
            "Set HF_HUB_OFFLINE=1 to make the stages use these instead of asking\n"
            "the Hub to resolve `main` first -- without it a complete cache still\n"
            "costs a round trip per checkpoint, and a cluster that cannot reach\n"
            "the Hub fails on a model that is sitting right there on disk."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
