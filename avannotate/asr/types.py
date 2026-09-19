"""What a transcript is, before and after it meets a model.

Two vocabularies, kept apart on purpose.

On the left is what the pipeline knows: a speech segment S7 planned, and a
choice of which audio to read for it.  On the right is what a recogniser
reports: text, per-word times, and the numbers that say whether the text is
trustworthy.

The model adapter is the only thing that crosses between them.  Everything here
is plain data, so the routing decision and the language vote are testable
without a package that cannot be installed on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from avannotate.coercion import coerce_number

#: Which audio a segment's transcript came from.  The distinction is not
#: cosmetic: it is the record of whether the recogniser heard one voice or a
#: mixture, and it is what the QA report counts when a transcript reads wrong.
SOURCE_MIX = "mix"
SOURCE_EXTRACTED = "extracted"


@dataclass(frozen=True)
class SpeechSegment:
    """One person's speech between two silences, as S7 planned and extracted.

    ``start`` and ``end`` are video time.  ``audio`` is S7's extraction for this
    segment -- one person, and only while they were talking.
    """

    identity: str
    name: str
    start: float
    end: float
    audio: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "name": self.name,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "audio": self.audio,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SpeechSegment:
        name = str(payload.get("identity", "")).strip()
        if not name:
            raise ValueError("a segment needs an identity")
        return cls(
            identity=name,
            name=str(payload.get("name", "")).strip(),
            start=coerce_number(payload.get("start", 0.0), "start"),
            end=coerce_number(payload.get("end", 0.0), "end"),
            audio=str(payload.get("audio", "")),
        )


@dataclass(frozen=True)
class SegmentSource:
    """Where to read a segment's audio, and where that audio sits in the video.

    Two origins, because the two sources start at different places.  ``origin``
    is the video time of the first sample of the span being read -- not always
    the segment's own start, since the mix is read with context padding so the
    recogniser does not open on a clipped phoneme.  ``file_origin`` is the video
    time of the first sample of the *file*: zero for the demuxed mix, and the
    segment's start for S7's extraction, which was trimmed to the segment before
    it was written.

    Reading means seeking to ``origin - file_origin``.  Getting either origin
    wrong shifts every word in the segment -- by the padding in one direction,
    by the whole segment start in the other -- and a transcript that is offset
    by a second still reads perfectly well, which is why these two numbers
    travel with the path instead of being recomputed where the timestamps are
    offset.
    """

    segment: SpeechSegment
    path: str
    origin: float
    file_origin: float
    duration: float
    source: str
    overlapping: bool

    @property
    def seek(self) -> float:
        """Where in the file the span begins."""

        return self.origin - self.file_origin

    def to_dict(self) -> dict[str, object]:
        return {
            **self.segment.to_dict(),
            "audio_source": self.path,
            "source": self.source,
            "origin": round(self.origin, 4),
            "file_origin": round(self.file_origin, 4),
            "window": round(self.duration, 4),
            "overlapping": self.overlapping,
        }


@dataclass(frozen=True)
class TranscribedWord:
    """One word, timed from the start of the samples the recogniser was given.

    Slice time, not video time: the recogniser has never seen this video and
    knows nothing about where its slice sits in it.  The stage adds the slice's
    origin, which is the only place that number is known.

    ``text`` is the recogniser's token, kept verbatim: for a spaced script it
    carries its own leading space (``" was"``) and for Chinese it does not.
    That is the only reliable record of how the words were spaced, and
    :func:`avannotate.asr.text.join_words` needs it -- stripping the words here
    would glue Chinese into one word and pull English apart.
    """

    text: str
    start: float
    end: float
    probability: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
        }
        if self.probability is not None:
            payload["probability"] = round(self.probability, 4)
        return payload


@dataclass(frozen=True)
class Transcription:
    """What the recogniser said about one segment."""

    text: str
    words: tuple[TranscribedWord, ...]
    language: str
    language_probability: float
    #: Averaged over the recogniser's own windows by the adapter.  Whether any
    #: of them means the text should not be trusted is the stage's decision,
    #: not the adapter's -- the thresholds are configuration, and an adapter
    #: that reads configuration is an adapter that cannot be reused.
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    #: False when the language was forced rather than detected, which is what
    #: keeps a forced guess from being counted as evidence for itself.
    detected: bool = True

    @property
    def empty(self) -> bool:
        """No words survived: silence the extractor passed through, or a clip
        too short for the recogniser to commit to anything."""

        return not self.words

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "detected": self.detected,
            "avg_logprob": round(self.avg_logprob, 4),
            "no_speech_prob": round(self.no_speech_prob, 4),
            "compression_ratio": round(self.compression_ratio, 4),
        }


@dataclass(frozen=True)
class LanguageVote:
    """One candidate for the video's language, and what it won on."""

    code: str
    seconds: float
    segments: int
    probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "seconds": round(self.seconds, 3),
            "segments": self.segments,
            "probability": round(self.probability, 4),
        }
