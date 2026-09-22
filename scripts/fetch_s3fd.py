#!/usr/bin/env python3
"""Put S3FD's face-detector weights where ClearerVoice looks for them.

    python scripts/fetch_s3fd.py [--from FILE] [--force]

``AV_MossFormer2_TSE_16K`` runs its own face detector -- S3FD, the one TalkNet
uses -- and fetches the weights on first use by shelling out to ``gdown``::

    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s" % (Link, PATH_WEIGHT)
    subprocess.call(cmd, shell=True, stdout=None)

Three things about that go wrong at once on a server like this one.  ``gdown``
is not installed, so the shell says ``/bin/dash: 1: gdown: not found`` and moves
on.  The id is a Google Drive file, which this network cannot reach.  And the
return value of ``subprocess.call`` is discarded, so both failures are silent
and S7 dies much later, on the missing file, with nothing in the message
pointing at a download::

    FileNotFoundError: '.../clearvoice/models/av_mossformer2_tse/
                        faceDetector/s3fd/sfd_face.pth'

That path is inside the *installed package* rather than under ``models/``, so
that is where this writes -- and it is also why ``pip install
--force-reinstall clearvoice`` undoes it.  Run this again afterwards; it is
idempotent and says so in one line.

The copy comes from the ClearerVoice Space on Hugging Face.  It is served
through Xet, and a transfer that stalls there is the documented way this fails:
``HF_HUB_DISABLE_XET=1`` falls back to the classic CDN path.  ``--from FILE``
installs a copy fetched some other way, which is the route that works when
neither path is reachable -- and the digest check below is what makes that safe
to do from anywhere.

The size and digest are the file's own, taken from the copy above.  They are
here rather than trusted to the transfer so that a redirect stub, an error page
or somebody's re-upload is caught at the download rather than inside S7.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

#: Where inside the package the weights go.  Confirmed against the traceback
#: rather than read off the repository, because the repository builds this path
#: from a bare string relative to whatever the working directory happens to be.
RELATIVE = Path("models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth")

#: The Hugging Face copy.
REPO = "mmwmm/ClearVoice"
FILENAME = "models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth"

#: Its digest, and the reason a copy from anywhere else can be trusted.
SHA256 = "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"

#: A redirect stub or an HTML error page is about a kilobyte; the weights are
#: about 90 MB.  Checked first because it gives a much better message than a
#: digest mismatch on a file that was never going to be right.
MIN_BYTES = 50_000_000


def destination() -> Path:
    """Where ClearerVoice looks, asked of the installed package.

    Derived from ``clearvoice.__file__`` rather than written down: the path
    contains the interpreter's version and the venv's layout, and both are
    properties of the machine rather than of this project.
    """

    try:
        import clearvoice
    except ModuleNotFoundError:
        raise SystemExit(
            "clearvoice is not installed. Run scripts/setup_venv.sh, or pip "
            "install clearvoice, and then run this again."
        ) from None
    return Path(clearvoice.__file__).resolve().parent / RELATIVE


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def install(source: Path, target: Path) -> int:
    if not source.is_file():
        raise SystemExit(f"no such file: {source}")

    size = source.stat().st_size
    if size < MIN_BYTES:
        raise SystemExit(
            f"{source} is {size} bytes and sfd_face.pth is about 90 MB. That is "
            "a redirect stub or an error page rather than the weights."
        )

    found = digest(source)
    if found != SHA256:
        raise SystemExit(
            f"{source} hashes to {found}, not {SHA256}. That is a different "
            "file -- a re-upload, or a download that stopped part way."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"   installed {target} ({size // 1024 // 1024} MB, digest checked)")
    return 0


def fetch(target: Path) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        raise SystemExit(
            "huggingface_hub is not installed, so this cannot fetch. Pass "
            "--from FILE with a copy you obtained another way."
        ) from None

    print(f"   fetching {REPO}/{FILENAME}")
    cached = hf_hub_download(repo_id=REPO, repo_type="space", filename=FILENAME)
    return install(Path(cached), target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="install a copy from this path instead of fetching one",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace one that is already there"
    )
    args = parser.parse_args(argv)

    target = destination()
    if target.is_file() and not args.force:
        size = target.stat().st_size
        if size >= MIN_BYTES:
            print(f"   already there: {target} ({size // 1024 // 1024} MB)")
            return 0
        print(f"   {target} is only {size} bytes; installing over it")

    if args.source is not None:
        return install(args.source, target)
    return fetch(target)


if __name__ == "__main__":
    raise SystemExit(main())
