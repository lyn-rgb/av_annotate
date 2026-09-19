"""The three taggers behind one call.

No single model answers the question the format asks.  Emotion classifiers have
no notion of *whispering*, which is how a line was produced rather than how the
speaker felt; voice-tagging models cover delivery but not the non-speech sounds
that replace words entirely; and a general audio tagger covers those but knows
nothing about delivery.  So this holds all three and reports each one's opinion
separately, leaving the choice to
:mod:`avannotate.paralinguistic.reduce`.

They are loaded on first use, one at a time.  A machine with only one of the
three installed can run that dimension and say so in the config, rather than
failing on an import for a model the caller never asked for.  Loading lazily
also means a video whose segments are all too short to tag never touches a
checkpoint.

Each model is asked for its own top labels and nothing else:
``top_k`` truncation happens in the stage, because how much of an answer is
worth keeping is the caller's decision and not the adapter's.

All three interfaces were read off the released artifacts rather than recalled,
and two of them are not what their documentation implies:

* ``laion/voice-tagging-whisper`` is a **generator, not a classifier**.  It has
  no ``id2label``, no classification head, and no scores -- it emits a string of
  comma-separated tags on an open vocabulary.  Loading it as a classifier
  succeeds and gives random numbers, so the mistake this avoids is a silent one.
* PANNs takes **32 kHz** input and resamples nothing, so this adapter converts.

What is *not* verified is behaviour: none of the three has been run on audio a
real pipeline produced.  See ``docs/server-setup.md``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from avannotate.paralinguistic.types import DIMENSIONS, TagScore

#: What every stage upstream produces and what two of the three models take.
SAMPLE_RATE = 16_000

#: PANNs is the exception: its mel front end is built for 32 kHz and the
#: wrapper applies no resampling, so its input has to be converted here.
PANNS_SAMPLE_RATE = 32_000


class TaggerError(RuntimeError):
    """A tagger could not be built or run."""


class Tagger(Protocol):
    """What the stage needs: one call, every dimension's answer."""

    name: str
    dimensions: tuple[str, ...]

    def tag(self, samples: NDArray[np.float32]) -> Mapping[str, tuple[TagScore, ...]]:
        """Per dimension, the labels and scores this audio produced.

        A dimension that is not being run, or that failed, is simply absent
        rather than present and empty -- the caller has no way to act on the
        difference and an empty list would read as "ran and found nothing".
        """
        ...


class _Model:
    """One lazily-built model.

    ``build`` is called at most once and the failure is remembered: a missing
    package should be one clear error naming the package, not an exception per
    segment for a thousand segments.
    """

    def __init__(self, label: str, build: Any) -> None:
        self.label = label
        self._build = build
        self._model: Any = None
        self._error: TaggerError | None = None

    def get(self) -> Any:
        if self._error is not None:
            raise self._error
        if self._model is None:
            try:
                self._model = self._build()
            except TaggerError as error:
                self._error = error
                raise
        return self._model


def _require(module: str, *, dimension: str, install: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        raise TaggerError(
            f"the {dimension} tagger needs {install}: pip install {install}. "
            "It downloads its checkpoint on first use, so the first run needs "
            "network access."
        ) from error


def _ranked(labels: Sequence[str], scores: Sequence[float], *, top: int) -> tuple[TagScore, ...]:
    """Pairs zipped, sorted by score, truncated.

    Sorted here rather than trusted from the model: the three report in
    different orders and the reduction reads the scores, but a reader looking at
    the record should see the model's best answer first without having to sort.
    """

    paired = [
        TagScore(label=str(label), score=float(score))
        for label, score in zip(labels, scores, strict=True)
    ]
    paired.sort(key=lambda item: -item.score)
    return tuple(paired[:top])


class ThreeModelTagger:
    """emotion2vec+ for affect, a voice tagger for delivery, PANNs for events."""

    name = "emotion2vec+voice-tagging+panns"

    def __init__(
        self,
        *,
        dimensions: tuple[str, ...] = DIMENSIONS,
        device: str | None = None,
        download_root: str | None = None,
        checkpoint: str | None = None,
        top: int = 20,
    ) -> None:
        self.dimensions = dimensions
        self.top = top
        self.device = device
        self.download_root = download_root
        #: PANNs' ``.pth``.  ``None`` makes the package shell out to wget, which
        #: silently does nothing where wget is missing -- so a server should
        #: pass a real path.
        self.checkpoint = checkpoint

        self._models: dict[str, _Model] = {
            "emotion": _Model("emotion2vec+", self._build_emotion),
            "delivery": _Model("voice-tagging-whisper", self._build_delivery),
            "event": _Model("panns-cnn14", self._build_event),
        }

    # -- emotion2vec+ ----------------------------------------------------- #

    def _build_emotion(self) -> Any:
        """``funasr.AutoModel(model="iic/emotion2vec_plus_large")``.

        Its ``generate`` returns one dict per input with parallel ``labels`` and
        ``scores`` lists, and the labels are bilingual with a slash -- which is
        why the vocabulary's table carries the joined form.
        """

        funasr = _require("funasr", dimension="emotion", install="funasr")
        model = funasr.AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
        return model

    def _emotion(self, samples: NDArray[np.float32]) -> tuple[TagScore, ...]:
        model = self._models["emotion"].get()
        output = model.generate(
            samples, granularity="utterance", extract_embedding=False
        )
        if not output:
            return ()
        first = output[0]
        labels = first.get("labels") or []
        scores = first.get("scores") or []
        if len(labels) != len(scores):
            raise TaggerError(
                f"emotion2vec returned {len(labels)} labels for {len(scores)} scores"
            )
        return _ranked(labels, scores, top=self.top)

    # -- voice-tagging-whisper -------------------------------------------- #

    def _build_delivery(self) -> Any:
        """A Whisper fine-tune that *generates* tags; it does not classify.

        Worth stating plainly because the model is easy to misread: its Hugging
        Face tag says ``audio-classification``, but it is a
        ``WhisperForConditionalGeneration`` with no classification head at all
        and no ``id2label`` in its config.  Loading it through
        ``WhisperForAudioClassification`` succeeds -- and attaches a **randomly
        initialised** head, so every score it returns is noise.  That failure is
        silent, which is why this comment is here rather than a note in a
        changelog.

        The processor comes from ``openai/whisper-small`` because the repository
        ships no tokenizer of its own.
        """

        transformers = _require(
            "transformers", dimension="delivery", install="transformers"
        )
        torch = _require("torch", dimension="delivery", install="torch")
        model = transformers.WhisperForConditionalGeneration.from_pretrained(
            "laion/voice-tagging-whisper", cache_dir=self.download_root
        )
        processor = transformers.WhisperProcessor.from_pretrained(
            "openai/whisper-small", cache_dir=self.download_root
        )
        model.eval()
        if self.device:
            model.to(torch.device(self.device))
        return (model, processor, torch)

    def _delivery(self, samples: NDArray[np.float32]) -> tuple[TagScore, ...]:
        model, processor, torch = self._models["delivery"].get()
        dtype = next(model.parameters()).dtype
        inputs = processor(
            samples, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        features = inputs.input_features.to(next(model.parameters()).device, dtype=dtype)
        with torch.no_grad():
            generated = model.generate(features, max_new_tokens=256)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]

        # Presence, not confidence.  Decoding is autoregressive, so there is no
        # distribution to read a probability off -- the model either named a tag
        # or it did not, and 1.0 is the honest encoding of that.  A consequence
        # worth knowing: this dimension's `min_score` admits everything the
        # model named, and exists only so a scored delivery model can be dropped
        # in without changing the schema.
        names = [part.strip() for part in text.split(",") if part.strip()]
        return tuple(TagScore(label=name, score=1.0) for name in names[: self.top])

    # -- PANNs CNN14 ------------------------------------------------------ #

    def _build_event(self) -> Any:
        """``panns_inference.AudioTagging`` with the AudioSet checkpoint.

        ``checkpoint_path=None`` makes the package download its weights with a
        shelled-out ``wget``, which **fails silently where wget is absent** and
        then fails obscurely in ``torch.load``.  A server should pre-place
        ``Cnn14_mAP=0.431.pth`` and pass an explicit path; see
        ``docs/server-setup.md``.
        """

        panns = _require("panns_inference", dimension="event", install="panns-inference")
        return panns.AudioTagging(
            checkpoint_path=self.checkpoint, device=self.device or "cpu"
        )

    def _event(self, samples: NDArray[np.float32]) -> tuple[TagScore, ...]:
        model = self._models["event"].get()
        audio = _resample_to_32k(samples)[None, :]
        # A 2-tuple: the clipwise probabilities and a 2048-dim embedding.  The
        # probabilities are already sigmoid-activated -- one per AudioSet class,
        # independent, so they sum to nothing in particular and the threshold is
        # this pipeline's to choose.
        clipwise, _ = model.inference(audio)
        names = audioset_labels()
        if clipwise.shape[-1] != len(names):
            raise TaggerError(
                f"PANNs returned {clipwise.shape[-1]} classes but the label list "
                f"has {len(names)}; the two have drifted apart"
            )
        return _ranked(names, clipwise[0].tolist(), top=self.top)

    # -- the one call ------------------------------------------------------ #

    def tag(self, samples: NDArray[np.float32]) -> Mapping[str, tuple[TagScore, ...]]:
        if len(samples) == 0:
            raise TaggerError("refusing to tag an empty array")

        reported: dict[str, tuple[TagScore, ...]] = {}
        for dimension in self.dimensions:
            method = {
                "emotion": self._emotion,
                "delivery": self._delivery,
                "event": self._event,
            }[dimension]
            reported[dimension] = method(samples)
        return reported


def _resample_to_32k(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """PANNs' mel front end is fixed at 32 kHz and it resamples nothing.

    Passing 16 kHz audio to it does not fail.  The mel filters are built for 32
    kHz, so every band lands in the wrong place and the model returns confident
    tags for a signal it never heard -- the same silent-shift failure the WAV
    layer refuses for the recognisers, and refused here by resampling rather
    than erroring because unlike those, this one has an unambiguous fix.

    librosa is used rather than something hand-rolled because ``panns-inference``
    depends on it, so it is present wherever this tagger can run at all.
    """

    try:
        import librosa
    except ModuleNotFoundError as error:
        raise TaggerError(
            "the event tagger needs librosa to resample 16 kHz audio to the "
            "32 kHz PANNs expects; it ships with panns-inference, so it should "
            "already be installed"
        ) from error

    return np.asarray(
        librosa.resample(
            np.asarray(samples, dtype=np.float32),
            orig_sr=SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
        ),
        dtype=np.float32,
    )


def audioset_labels() -> list[str]:
    """AudioSet's 527 class names, in the order PANNs' outputs are in.

    Public because a caller checking a label against the model's own taxonomy
    should be reading the same list the pipeline matched against.
    """

    try:
        from panns_inference import labels
    except (ModuleNotFoundError, ImportError) as error:
        raise TaggerError(
            "could not import AudioSet's label list from panns_inference; the "
            "package's layout has changed and audioset_labels needs updating"
        ) from error
    return [str(item) for item in labels]


def build_tagger(config: Mapping[str, Any]) -> Tagger:
    """Construct the tagger a stage's config asks for."""

    backend = str(config.get("backend", "three-model"))
    if backend != "three-model":
        raise TaggerError(
            f"unknown tagging backend {backend!r}; the plan's choice is 'three-model'"
        )
    raw = config.get("dimensions")
    dimensions = (
        tuple(str(item) for item in raw)
        if isinstance(raw, (list, tuple)) and raw
        else DIMENSIONS
    )
    device = config.get("device")
    root = config.get("download_root")
    checkpoint = config.get("checkpoint")
    return ThreeModelTagger(
        dimensions=dimensions,
        device=str(device) if device is not None else None,
        download_root=str(root) if root is not None else None,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
    )
