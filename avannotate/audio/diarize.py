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

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from avannotate.audio.types import DiarizationResult, SpeakerTurn

#: The classes pyannote's checkpoints reference that torch's ``weights_only``
#: loader does not allow by default.
#:
#: Enumerated from the files rather than guessed.  ``pickletools.genops`` walks
#: a checkpoint's pickle stream without executing any of it -- no unpickling, no
#: imports, no code run -- and across every checkpoint this pipeline loads,
#: these four are the whole non-default set.  DiariZen's own checkpoint needs
#: none of them (plain ``torch._utils._rebuild_tensor_v2``, allowed by default);
#: they all come from the wespeaker embedding model pyannote pulls in.
#:
#: Finding them one traceback at a time is the alternative, and it costs a
#: failed run per class.
_SAFE_GLOBALS = (
    ("torch.torch_version", "TorchVersion"),
    ("pyannote.audio.core.task", "Specifications"),
    ("pyannote.audio.core.task", "Resolution"),
    ("pyannote.audio.core.task", "Problem"),
)


def _allow_checkpoint_globals() -> None:
    """Let torch load pyannote's checkpoints again.

    PyTorch 2.6 changed ``torch.load``'s default from ``weights_only=False`` to
    ``True``, which refuses to unpickle anything not on an allowlist.  The
    pyannote checkpoints were written before that and carry a ``TorchVersion``
    stamp and three ``pyannote.audio.core.task`` enums, so the load fails with
    ``UnpicklingError: Unsupported global: ...``.

    ``add_safe_globals`` is what the error itself suggests, and a version string
    and three enums are the smallest thing that makes these files load.  The
    other option the error offers -- turning ``weights_only`` off -- would drop
    that check for every checkpoint this process ever loads, which is a much
    larger thing to give up for four benign classes.

    Process-wide and idempotent, and called at the point of use rather than at
    import time, so that importing this module does not quietly change torch's
    behaviour for everything else in the interpreter.
    """
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch is required for S4
        return
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:  # torch older than 2.6: nothing needs allowing
        return
    allowed = []
    for module_name, class_name in _SAFE_GLOBALS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # Only reachable if pyannote is absent, in which case the pipeline
            # this guards cannot be built anyway and its own error is clearer.
            continue
        found = getattr(module, class_name, None)
        if found is not None:
            allowed.append(found)
    if allowed:
        add(allowed)

#: v2 handles up to four overlapping speakers; v1 merges extra speakers into its
#: arrival-order slots.  The default is v2 because simultaneous speech is the
#: case this pipeline is built for.
DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

#: Batch size the checkpoint's own ``config.toml`` asks for.
#:
#: Chosen by its authors for the card they had, which was not a 24 GB one.  It
#: is the *starting* point rather than a commitment -- see
#: :meth:`DiariZenDiarizer._call` for what happens when it does not fit.
DEFAULT_BATCH_SIZE = 32

#: Halving stops here.  A single chunk through wavlm-large fits anywhere this
#: pipeline can otherwise run, so failing at 1 means the problem is not the
#: batch size and should be reported as itself.
MIN_BATCH_SIZE = 1


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
        batch_size: int | None = None,
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
        _allow_checkpoint_globals()
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
        #: What the pipeline will be asked to use, and what it was reduced to if
        #: that did not fit.  Starts at the checkpoint's own value.
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
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

    def _call(self, filename: str) -> Any:
        """Run the pipeline, halving the batch size when the card runs out.

        The checkpoint's ``config.toml`` asks for 32, which its authors picked
        for the card they had and which does not fit a 24 GB one.  pyannote
        catches the CUDA OOM and re-raises it as a ``MemoryError`` carrying that
        same number, so the first run of this pipeline against real video failed
        a video in S4 for no reason but the card it happened to land on::

            DiariZen failed on mix.wav: MemoryError: batch_size ( 32) is
            probably too large.

        Halving rather than committing to some smaller fixed number, because
        what fits is a property of the card *and* of whatever else is resident
        on it, and this adapter can see neither.  The size that worked is kept,
        so a batch pays for the discovery once per worker rather than once per
        video -- and on a card where 32 does fit, it never pays at all.

        Both fields have to be set: pyannote keeps the segmentation size behind
        a property and the embedding size as a plain attribute, and defaults
        each to 1, so setting one and not the other silently drops the other to
        single-chunk inference.
        """

        while True:
            self._pipeline.embedding_batch_size = self.batch_size
            self._pipeline.segmentation_batch_size = self.batch_size
            try:
                return self._pipeline(filename)
            except MemoryError as error:
                if self.batch_size <= MIN_BATCH_SIZE:
                    raise DiarizerError(
                        f"DiariZen ran out of memory on {filename} even at a batch "
                        f"size of {self.batch_size}. Something else is holding the "
                        "card, or this device cannot fit the model at all."
                    ) from error
                self.batch_size = max(MIN_BATCH_SIZE, self.batch_size // 2)
                print(
                    f"   DiariZen ran out of memory on {filename}; retrying at "
                    f"batch size {self.batch_size}"
                )

    def diarize(self, audio: Path) -> DiarizationResult:
        name = Path(audio).name
        # The call is guarded, not just the constructor.  ``DiarizerError`` is
        # documented as "could not be built *or run*", and only the build half
        # of that was true: a crash inside DiariZen arrived at the stage as
        # whatever pyannote happened to raise.  On a clip the clusterer cannot
        # make sense of -- no speech to embed, so the embeddings come back
        # degenerate -- that is a bare ``ValueError: negative dimensions are
        # not allowed`` from deep inside VBx, naming neither the diarizer nor
        # the file, and the stage cannot record a failure it does not recognise.
        try:
            result = self._call(str(audio))
        except DiarizerError:
            raise
        except Exception as error:  # noqa: BLE001 - see above
            raise DiarizerError(
                f"DiariZen failed on {name}: {type(error).__name__}: {error}"
            ) from error

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
                "batch_size": self.batch_size,
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
    batch_size = config.get("batch_size")
    return DiariZenDiarizer(
        model=str(config.get("model", DEFAULT_MODEL)),
        device=str(device) if device is not None else None,
        batch_size=int(batch_size) if batch_size is not None else None,
    )
