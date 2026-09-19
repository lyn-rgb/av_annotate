"""The diarizer behind an interface, with DiariZen as the implementation.

The plan's choice is DiariZen: it is the strongest open-source diarizer on the
many-speaker case (1.9% DER on 5+ speakers in the multilingual benchmark), and
this corpus is exactly that.  Its weights are CC BY-NC 4.0, which the project
has confirmed is acceptable -- it is research use.

The call below was read off the repository's source, not its README, and four
things about it are not what a reader would assume::

    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
    result = pipeline("audio.wav")          # str, BytesIO or ProtocolFile
    for turn, _, speaker in result.itertracks(yield_label=True):
        ...  # turn.start, turn.end in seconds, speaker an integer

* **The deep import is required.**  ``diarizen/__init__.py`` and
  ``diarizen/pipelines/__init__.py`` are both empty, so there is nothing to
  re-export from a shorter path.
* **The pipeline takes a ``str``, not a ``Path``.**  ``__call__`` opens with an
  ``assert isinstance(in_wav, (str, BytesIO, ProtocolFile))``, and a ``Path``
  fails it.  A numpy array or a waveform dict fails it too -- the waveform form
  is used internally and is not part of the interface.
* **No token, no device, and ``cache_dir`` is a trap.**  The signature is
  ``from_pretrained(repo_id, cache_dir=None, rttm_out_dir=None)``.  The
  checkpoints are ungated so nothing needs a token; the device is chosen inside
  from ``torch.cuda.is_available()``; and naming a ``cache_dir`` sets
  ``local_files_only=True`` in the implementation, which works on a warm machine
  and fails on a cold one.
* **The labels are integers.**  ``Binarize`` passes ``scores.labels[k]``
  straight through, so ``speaker`` is ``0``, ``3``, ``4`` -- the ``speaker_``
  prefix in the README's example is the README's own f-string, not the API's.
  :func:`normalize_label` zero-pads them, which also makes ``spk_02`` sort
  before ``spk_10``.

``result`` is a genuine ``pyannote.core.Annotation``, and its turns are allowed
to overlap -- which is the point, since simultaneous speech is the case this
pipeline exists to handle.

**Not yet run against the real package.**  Everything above is read from the
source; nothing has been executed, so the behaviour on real audio is open.  Two
things are isolated to this module: whether ``.to(torch.device(...))`` works on
the pipeline object -- it inherits one from pyannote's ``Pipeline``, whose ``to``
rejects a bare string -- and whether ``itertracks`` yields the tuple shape above.
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
    ) -> None:
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ModuleNotFoundError as error:
            raise DiarizerError(
                "DiariZen is required for this stage. It is not on PyPI; install it "
                "from source -- https://github.com/BUTSpeechFIT/DiariZen -- which "
                "also needs a pyannote-audio checkout and dscore as its submodule. "
                "See docs/server-setup.md, or run scripts/setup_server.sh."
            ) from error

        # Called with the repo id alone, and deliberately so.  Its signature is
        # ``from_pretrained(repo_id, cache_dir=None, rttm_out_dir=None)``:
        #
        # * **No token.**  There is no token argument, and none is needed -- the
        #   DiariZen checkpoints and the wespeaker embedding model it downloads
        #   alongside them are all ungated, checked by anonymous request.
        # * **No ``cache_dir``.**  The implementation does
        #   ``local_files_only=cache_dir is not None``, so naming a cache
        #   directory switches the download *off*.  On a warm machine that works
        #   and on a cold one it fails, which is the worst way round.
        # * **No device.**  It is chosen inside from ``torch.cuda.is_available()``
        #   and corrected afterwards in ``_place_on``.
        try:
            self._pipeline = DiariZenPipeline.from_pretrained(model)
        except TypeError as error:
            raise DiarizerError(
                f"DiariZenPipeline.from_pretrained({model!r}) rejected the call: "
                f"{error}. This adapter passes the repo id alone, as the released "
                "revision's signature allows; a revision that requires more needs "
                "this call updated rather than the error swallowed."
            ) from error

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
    return DiariZenDiarizer(
        model=str(config.get("model", DEFAULT_MODEL)),
        device=str(device) if device is not None else None,
    )
