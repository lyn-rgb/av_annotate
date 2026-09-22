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

**Both are now answered by a real run.**  ``output_path`` is a directory, and the
extractor writes *nested* under it::

    <output_path>/AV_MossFormer2_TSE_16K/<input stem>/py_faceTracks/est_0.wav

``est_0.wav`` is the separated speech -- 16 kHz mono, as long as the crop -- and
it is the only file this stage wants; the videos beside it are intermediates, and
the ``00000.wav`` beside *those* is the track's own audio rather than the model's
output.

Two other things had to be right before that run told us anything, and both were
in the crop this adapter is handed rather than in the adapter.  The crop had no
audio track at all -- ``crop_video`` wrote ``-an`` -- and it was one frame longer
than the video it was cut from.  The extractor is audio-visual: the crop says
which face, and the sound is the thing being separated.  A silent crop leaves it
with nothing to do, and it reports that by going looking for a file it never
wrote.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

#: The model name ClearerVoice registers this checkpoint under.
MODEL_NAME = "AV_MossFormer2_TSE_16K"

#: The directory clearvoice writes under its output for each face it tracked.
#:
#: Its presence is how :meth:`ClearerVoiceExtractor.extract` tells "it looked
#: and found no face" from "it found a face and produced nothing" -- two
#: situations that look identical from the outside and need opposite answers.
_FACE_TRACK_DIR = "py_faceTracks"

#: Video containers its loader accepts.
SUPPORTED_SUFFIXES = frozenset({".avi", ".mp4", ".mov", ".webm"})


class TseError(RuntimeError):
    """The extractor could not be built or run."""


class TargetSpeakerExtractor(Protocol):
    """What the stage needs from an extractor, and nothing more."""

    name: str

    def extract(self, video: Path, output_dir: Path) -> Path | None:
        """The WAV holding the one voice in ``video``, or ``None`` if there is none.

        ``None`` means the extractor ran and found no face to condition on --
        a fact about the crop rather than a failure, and not something the
        caller should treat as one.
        """
        ...


def _pin_clearvoice_device() -> None:
    """Make clearvoice's GPU chooser agree with this process's view of the GPUs.

    clearvoice picks a card by running ``nvidia-smi`` and taking whichever has
    the most free memory.  ``nvidia-smi`` lists every physical device and
    ignores ``CUDA_VISIBLE_DEVICES``; ``torch.cuda.set_device`` does not.  In a
    batch worker -- where the pool has already restricted this process to one
    card -- the chooser therefore returns an index that means nothing here, and
    ``torch.cuda.set_device`` raises ``CUDA error: invalid device ordinal``.

    It is not a rare race.  The chooser takes the *freest* card, so it returns 0
    only when card 0 happens to be the idlest -- during a four-worker batch,
    almost never.  S7 would fail on nearly every video, and the error names a
    device rather than anything about device selection.

    Inside this process there is exactly one correct answer: device 0, because
    ``CUDA_VISIBLE_DEVICES`` has already made the assigned card the only one
    this process can see.

    Monkeypatching a library's internals is not free, and it is done here
    because there is no other way in: ``ClearVoice``'s public constructor takes
    a task and model names and nothing else, and the device is chosen inside
    the model's own ``__init__``.  Done once per process, and only for the
    chooser -- nothing else about the library is touched.
    """
    try:
        from clearvoice.networks import SpeechModel
    except (ImportError, AttributeError):  # pragma: no cover - layout changed
        return
    if getattr(SpeechModel, "_avannotate_device_pinned", False):
        return
    SpeechModel.get_free_gpu = lambda self: 0
    SpeechModel._avannotate_device_pinned = True


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
        _pin_clearvoice_device()
        self._model = ClearVoice(
            task="target_speaker_extraction", model_names=[model_name]
        )

    def extract(self, video: Path, output_dir: Path) -> Path | None:
        source = Path(video)
        if not source.is_file():
            raise TseError(f"no such video: {source}")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise TseError(
                f"{source.suffix} is not a container the extractor accepts; "
                f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # A directory of its own, created empty, for this call.
        #
        # This used to diff the shared output directory for whatever had shown
        # up in it.  That works exactly once, and then never again: clearvoice
        # writes *nested*, under ``<output_path>/AV_MossFormer2_TSE_16K/<name>/``,
        # so after one run -- successful or not -- that directory is already
        # there, nothing new appears where the diff is looking, and every later
        # call is reported as having written nothing at all.
        #
        # Which was a lie, and a misleading one: the extractor had written
        # everything, and the message sent its reader to the model's naming
        # convention instead of to the comparison.
        call_dir = Path(tempfile.mkdtemp(prefix=f"{source.stem}-", dir=output_dir))

        self._call(source, call_dir)

        # ``est_<n>.wav`` is the extracted speech, and it is the only output
        # this stage wants: the videos beside it are intermediates, and the
        # ``00000.wav`` beside *those* is the track's own audio rather than the
        # model's.  A pattern rather than a full path, because the directories
        # in between are clearvoice's business -- it names one after the input
        # file -- and predicting them would be predicting somebody else's
        # layout.
        produced = sorted(call_dir.rglob("est_*.wav"))
        if produced:
            return produced[0]

        wrote = sorted(
            str(path.relative_to(call_dir))
            for path in call_dir.rglob("*")
            if path.is_file()
        )

        # Nothing was extracted, and three different things arrive here.
        # ``py_faceTracks`` is what separates them: clearvoice writes a
        # directory of that name for each face it tracked.
        #
        # Nothing at all means it did not run the way this adapter expects, and
        # that is a fault -- a run that leaves no files behind is not a run that
        # found nothing.
        if not wrote:
            raise TseError(
                f"the extractor wrote nothing at all under {call_dir}. "
                "ClearerVoiceExtractor.extract predicts this from a real run; "
                "check it against the installed version."
            )

        # A ``py_faceTracks`` directory means it found and tracked a face, so
        # the absence of an ``est_*.wav`` beside it is a fault: it had something
        # to separate and produced nothing.
        if any(_FACE_TRACK_DIR in Path(name).parts for name in wrote):
            raise TseError(
                f"the extractor tracked a face but wrote no est_*.wav under "
                f"{call_dir}. It wrote "
                + ", ".join(wrote[:12])
                + ". ClearerVoiceExtractor.extract predicts this name from a real "
                "run; check it against the installed version."
            )

        # Neither: it ran, it left its intermediates, and it tracked no face.
        #
        # **That is a fact about the crop, not a fault.**  The crop is cut from a
        # tracked face, and a track can be a false positive -- this corpus has
        # one on purpose, which is what
        # ``test_a_background_false_positive_is_kept_and_is_distinguishable``
        # exists for.  There is nothing to separate, so there is nothing wrong:
        # the caller skips the segment.  Reported as an error, it failed the
        # whole video and took the identity's other segments down with it.
        return None

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
