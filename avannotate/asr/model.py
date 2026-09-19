"""The recogniser behind an interface, with faster-whisper as the choice.

faster-whisper rather than WhisperX, which is the obvious alternative: WhisperX
adds a separate forced-alignment model per language for word timestamps, and its
default aligners are VoxPopuli weights under CC-BY-NC.  faster-whisper produces
word timestamps itself, from the decoder's cross-attention, so this stage needs
one package and one licence instead of two.  The accuracy difference is real but
it is a difference between two word-timestamp estimates, not between a
transcript and no transcript.

Whisper large-v3 is the checkpoint because the corpus is not known to be
English.  A distil or medium model would be faster and an unknown fraction of
the corpus would be transcribed worse for it.

The adapter takes samples rather than a path: the stage has already read exactly
the span it wants, and handing over a file would mean the recogniser deciding
where the audio starts -- which is the one thing this pipeline cannot delegate,
since every timestamp downstream is relative to that decision.

The interface above was read off faster-whisper 1.2.1, where it is stable and
documented; the shape of the call below is verified against that release.  What
is *not* verified is behaviour: nothing here has been run on audio a real model
produced.  See ``docs/server-setup.md``.

Two properties of that interface this stage leans on, both of which are
assertions the upstream project makes about itself rather than things that look
true:

``word.word`` keeps its leading space for spaced scripts.
    Upstream's own test asserts ``segment.text == "".join(word.word for word in
    segment.words)``, which is the join :mod:`avannotate.asr.text` performs.  It
    is why the words are joined with no separator instead of a guessed one.

An ``ndarray`` is used as-is, at 16 kHz, with no resampling and no check.
    Which makes a wrong-rate file the one input that cannot be caught from
    inside: it decodes fine and every timestamp comes out at the wrong scale.
    :func:`avannotate.asr.audio.read_source` is where that is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from avannotate.asr.text import collapse_whitespace, join_words
from avannotate.asr.types import TranscribedWord, Transcription

#: Whisper's checkpoints are all trained at 16 kHz and it resamples nothing.
SAMPLE_RATE = 16_000


class AsrError(RuntimeError):
    """The recogniser could not be built or run."""


class Transcriber(Protocol):
    """What the stage needs from a recogniser, and nothing more."""

    name: str

    def transcribe(
        self, samples: NDArray[np.float32], *, language: str | None
    ) -> Transcription:
        """Recognise ``samples`` (16 kHz mono float32).

        ``language`` is a code to force, or ``None`` to let the recogniser
        decide -- and the returned ``detected`` says which of the two happened,
        because a forced language is not evidence about anything.
        """
        ...

    def detect(self, samples: NDArray[np.float32]) -> tuple[str, float]:
        """The language of ``samples``, and the recogniser's confidence in it.

        Separate from :meth:`transcribe` because it is a different amount of
        work: language is decided from a single 30-second window before any
        decoding happens, so asking for it alone costs a forward pass over the
        mel rather than a full transcription nobody wanted the text of.
        """
        ...


def _weighted(values: list[float], weights: list[float]) -> float:
    """Duration-weighted mean, falling back to the plain one when it cannot be.

    A recogniser returns one entry per window, and its windows are not the same
    length: an unweighted mean lets a 0.2-second window count as much as a
    20-second one when deciding whether the whole clip is trustworthy.
    """

    if len(values) != len(weights):
        raise AsrError(
            f"{len(values)} values but {len(weights)} weights; they are built in "
            "step and a mismatch means the recogniser returned something new"
        )
    total = sum(weights)
    if total <= 0.0:
        # Every window was zero-length.  The plain mean is meaningless too, but
        # it is at least the recogniser's own number rather than a division.
        return sum(values) / len(values) if values else 0.0
    return (
        sum(value * weight for value, weight in zip(values, weights, strict=True)) / total
    )


class FasterWhisperTranscriber:
    """``WhisperModel`` behind :class:`Transcriber`."""

    name = "faster-whisper"

    def __init__(
        self,
        *,
        model: str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise AsrError(
                "faster-whisper is required for this stage: pip install "
                "faster-whisper. It downloads the checkpoint from Hugging Face "
                "on first use, so the first run needs network access."
            ) from error

        self.model_name = model
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self._model = WhisperModel(
            model,
            device=device or "auto",
            compute_type=compute_type or "default",
            download_root=download_root,
        )

    def detect(self, samples: NDArray[np.float32]) -> tuple[str, float]:
        if len(samples) == 0:
            raise AsrError("refusing to detect the language of an empty array")

        # A 1-D float array, per the method's own contract -- it takes no path
        # and does no decoding of its own.  The threshold argument is left at
        # its default on purpose: it is annotated Optional but dereferenced
        # unguarded, so passing None explicitly raises rather than defaulting.
        language, probability, _ = self._model.detect_language(
            np.asarray(samples, dtype=np.float32)
        )
        return str(language), float(probability)

    def transcribe(
        self, samples: NDArray[np.float32], *, language: str | None
    ) -> Transcription:
        if len(samples) == 0:
            raise AsrError("refusing to transcribe an empty array")

        # Consumed inside this call on purpose.  The returned segments are a
        # generator, and while detection, VAD and feature extraction all run
        # before it is returned, the decoding loop does not: a caller that
        # returned early would report an empty transcript for audio it never
        # decoded.  Word timings come from the decoder's cross-attention, which
        # is set up per decode, so there is no batching to be had here -- the
        # batched pipeline batches windows of one input, not several clips.
        streamed, info = self._model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=language,
            task="transcribe",
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            condition_on_previous_text=self.condition_on_previous_text,
        )

        words: list[TranscribedWord] = []
        texts: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        ratios: list[float] = []
        weights: list[float] = []

        for segment in streamed:
            texts.append(str(segment.text))
            logprobs.append(float(segment.avg_logprob))
            no_speech.append(float(segment.no_speech_prob))
            ratios.append(float(segment.compression_ratio))
            weights.append(max(0.0, float(segment.end) - float(segment.start)))
            for word in segment.words or ():
                words.append(
                    TranscribedWord(
                        # ``word.word``, not ``word.text`` -- the dataclass field
                        # is named for the thing, not the type.  Its leading
                        # space is upstream's own invariant: their test asserts
                        # segment.text == "".join(word.word for word in words),
                        # which is the same join this pipeline performs.
                        text=str(word.word),
                        start=float(word.start),
                        end=float(word.end),
                        probability=float(word.probability),
                    )
                )

        detected = language is None
        return Transcription(
            # Assembled from the words so the spacing rule lives in one place.
            # The segment texts are the fallback for a decode that produced
            # text but no word timings, which is rare and should not be lost.
            text=join_words(words) or collapse_whitespace("".join(texts)),
            words=tuple(words),
            language=str(info.language or language or ""),
            # A forced language has no probability behind it -- there was no
            # detection -- and reporting the recogniser's own number here would
            # let a guess be counted as evidence for itself.
            language_probability=float(info.language_probability or 0.0) if detected else 0.0,
            detected=detected,
            avg_logprob=_weighted(logprobs, weights),
            no_speech_prob=_weighted(no_speech, weights),
            compression_ratio=_weighted(ratios, weights),
        )


def build_transcriber(config: Mapping[str, Any]) -> Transcriber:
    """Construct the recogniser a stage's config asks for."""

    backend = str(config.get("backend", "faster-whisper"))
    if backend != "faster-whisper":
        raise AsrError(
            f"unknown ASR backend {backend!r}; the plan's choice is 'faster-whisper'"
        )
    device = config.get("device")
    compute_type = config.get("compute_type")
    download_root = config.get("download_root")
    return FasterWhisperTranscriber(
        model=str(config.get("model", "large-v3")),
        device=str(device) if device is not None else None,
        compute_type=str(compute_type) if compute_type is not None else None,
        download_root=str(download_root) if download_root is not None else None,
        beam_size=int(config.get("beam_size", 5)),
        vad_filter=bool(config.get("vad_filter", False)),
        condition_on_previous_text=bool(config.get("condition_on_previous_text", False)),
    )
