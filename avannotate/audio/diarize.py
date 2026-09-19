"""The diarizer behind an interface, with DiariZen as the implementation.

The plan's choice is DiariZen: it is the strongest open-source diarizer on the
many-speaker case (1.9% DER on 5+ speakers in the multilingual benchmark), and
this corpus is exactly that.  Its weights are CC BY-NC 4.0, which the project
has confirmed is acceptable -- it is research use.

Written against DiariZen's documented interface::

    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
    result = pipeline("audio.wav")
    for turn, _, speaker in result.itertracks(yield_label=True):
        ...  # turn.start, turn.end in seconds, speaker an integer label

``result`` is a pyannote-style ``Annotation``, and its turns are allowed to
overlap -- which is the point, since simultaneous speech is the case this
pipeline exists to handle.

**Not yet run against the real package.** The interface above is taken from
DiariZen's README and model cards, not from executing it, so two things need
confirming on the first GPU machine that has it installed: that the pipeline
object accepts ``.to(device)``, and that ``itertracks(yield_label=True)`` yields
the tuple shape described.  Both are isolated to :meth:`DiariZenDiarizer.diarize`
and its constructor; nothing downstream knows about either.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from avannotate.audio.types import DiarizationResult, SpeakerTurn

#: v2 handles up to four overlapping speakers; v1 merges extra speakers into its
#: arrival-order slots.  The default is v2 because simultaneous speech is the
#: case this pipeline is built for.
DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"


class DiarizerError(RuntimeError):
    """The diarizer could not be built or run."""


class Diarizer(Protocol):
    """What the stage needs from a diarizer, and nothing more."""

    name: str

    def diarize(self, audio: Path) -> DiarizationResult: ...


def normalize_label(raw: object) -> str:
    """A stable speaker id from whatever the backend calls a speaker.

    DiariZen labels speakers with integers in its raw result and renames them to
    ``SPEAKER_00`` in the RTTM it writes.  Both spellings appear in its own
    documentation, so the id is derived from the integer when there is one and
    zero-padded, which also makes ``spk_02`` sort before ``spk_10``.
    """

    text = str(raw).strip()
    try:
        return f"spk_{int(text):02d}"
    except ValueError:
        return f"spk_{text}"


class DiariZenDiarizer:
    """DiariZen behind :class:`Diarizer`."""

    name = "diarizen"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        huggingface_token: str | None = None,
    ) -> None:
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ModuleNotFoundError as error:
            raise DiarizerError(
                "DiariZen is required for this stage. It is not on PyPI; install it "
                "from source -- https://github.com/BUTSpeechFIT/DiariZen -- which "
                "also needs a pyannote-audio checkout and dscore as its submodule. "
                "See docs/ for the server setup."
            ) from error

        load_options: dict[str, Any] = {}
        if huggingface_token is not None:
            load_options["use_auth_token"] = huggingface_token
        try:
            self._pipeline = DiariZenPipeline.from_pretrained(model, **load_options)
        except TypeError:
            # Older revisions spell the token argument differently.
            load_options.pop("use_auth_token", None)
            self._pipeline = DiariZenPipeline.from_pretrained(model, **load_options)

        self.model = model
        # Recorded rather than assumed: the summary should say where the model
        # actually ran, and a silent fallback to CPU would otherwise look like a
        # device setting that simply had no effect.
        self.device = self._place_on(device)

    def _place_on(self, device: str | None) -> str:
        if device is None:
            return "backend default"
        if not hasattr(self._pipeline, "to"):
            return "backend default (pipeline exposes no .to())"
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover - DiariZen needs torch
            return "backend default (torch unavailable)"
        self._pipeline.to(torch.device(device))
        return device

    def diarize(self, audio: Path) -> DiarizationResult:
        result = self._pipeline(str(audio))

        if not hasattr(result, "itertracks"):
            raise DiarizerError(
                f"DiariZen returned {type(result).__name__}, which has no itertracks(); "
                "the pipeline's output format is not what this adapter expects"
            )

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            label = normalize_label(speaker)
            start = float(turn.start)
            end = float(turn.end)
            if end > start:
                turns.append(SpeakerTurn(speaker=label, start=start, end=end))

        turns.sort(key=lambda item: (item.start, item.end, item.speaker))
        return DiarizationResult(
            turns=tuple(turns),
            metadata={
                "backend": self.name,
                "model": self.model,
                "device": self.device,
            },
        )


def build_diarizer(config: Mapping[str, Any]) -> Diarizer:
    """Construct the diarizer a stage's config asks for."""

    backend = str(config.get("backend", "diarizen"))
    if backend != "diarizen":
        raise DiarizerError(
            f"unknown diarization backend {backend!r}; the plan's choice is 'diarizen'"
        )
    device = config.get("device")
    token = config.get("huggingface_token")
    return DiariZenDiarizer(
        model=str(config.get("model", DEFAULT_MODEL)),
        device=str(device) if device is not None else None,
        huggingface_token=str(token) if token is not None else None,
    )
