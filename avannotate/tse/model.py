"""The extractor behind an interface, with ClearerVoice's AV model as the choice.

``AV_MossFormer2_TSE_16K`` is the only openly available *face-conditioned*
target speaker extractor, which is what this pipeline needs: the whole design
gives a person an identity in S3 and a voice timeline in S6, and the extractor
has to be told which of those to pull out.  A voice-conditioned extractor would
need a clean enrolment clip, which is exactly what the pipeline is trying to
produce -- the circle this model breaks.

Its public API is per-file::

    from clearvoice import ClearVoice

    model = ClearVoice(task="target_speaker_extraction",
                       model_names=["AV_MossFormer2_TSE_16K"])
    model(input_path="clip.mp4", online_write=True, output_path="out_dir")

It accepts ``.avi``, ``.mp4``, ``.mov`` and ``.webm``, and writes one WAV per
input.  Its own pipeline detects faces and chooses a speaker by lip motion, and
no argument overrides that -- which is why :mod:`avannotate.tse.crop_video`
exists.  Give it a video holding one face and the choice is made for it.

**Not yet run against the real package.**  Two things need confirming on the
first machine that has it: the produced file's name, which is why the adapter
finds it by scanning the output directory rather than by predicting it, and
whether ``output_path`` is taken as a directory or a file prefix.  Both are
isolated to :meth:`ClearerVoiceExtractor.extract`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

#: The model name ClearerVoice registers this checkpoint under.
MODEL_NAME = "AV_MossFormer2_TSE_16K"

#: Video containers its loader accepts.
SUPPORTED_SUFFIXES = frozenset({".avi", ".mp4", ".mov", ".webm"})


class TseError(RuntimeError):
    """The extractor could not be built or run."""


class TargetSpeakerExtractor(Protocol):
    """What the stage needs from an extractor, and nothing more."""

    name: str

    def extract(self, video: Path, output_dir: Path) -> Path:
        """Return the WAV holding the one voice present in ``video``."""
        ...


class ClearerVoiceExtractor:
    """``AV_MossFormer2_TSE_16K`` behind :class:`TargetSpeakerExtractor`."""

    name = "clearvoice-av-mossformer2"

    def __init__(self, *, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        try:
            from clearvoice import ClearVoice
        except ModuleNotFoundError as error:
            raise TseError(
                "ClearerVoice is required for this stage: pip install clearvoice. "
                "It fetches the checkpoint from Hugging Face on first use, so the "
                "first run needs network access."
            ) from error

        self.model_name = model_name
        self.device = device
        self._model = ClearVoice(
            task="target_speaker_extraction", model_names=[model_name]
        )

    def extract(self, video: Path, output_dir: Path) -> Path:
        source = Path(video)
        if not source.is_file():
            raise TseError(f"no such video: {source}")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise TseError(
                f"{source.suffix} is not a container the extractor accepts; "
                f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        before = {path.name for path in output_dir.iterdir()}
        self._call(source, output_dir)

        produced = sorted(
            path for path in output_dir.iterdir() if path.name not in before
        )
        if not produced:
            raise TseError(
                f"the extractor wrote nothing to {output_dir}. Its output naming "
                "convention is not what this adapter expects; check "
                "ClearerVoiceExtractor.extract against the installed version."
            )
        return produced[0]

    def _call(self, source: Path, output_dir: Path) -> Any:
        """The one call whose argument names are unverified.

        Kept in its own method so a version whose signature differs is a change
        here and not in the stage.
        """

        return self._model(
            input_path=str(source),
            online_write=True,
            output_path=str(output_dir),
        )


def build_extractor(config: Mapping[str, Any]) -> TargetSpeakerExtractor:
    """Construct the extractor a stage's config asks for."""

    backend = str(config.get("backend", "clearvoice"))
    if backend != "clearvoice":
        raise TseError(
            f"unknown extraction backend {backend!r}; the plan's choice is 'clearvoice'"
        )
    device = config.get("device")
    return ClearerVoiceExtractor(
        model_name=str(config.get("model", MODEL_NAME)),
        device=str(device) if device is not None else None,
    )
