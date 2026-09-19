"""Turning a recogniser's words into a line of text, and judging the result.

Two small problems that are both about not being silently wrong.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from avannotate.asr.types import TranscribedWord

_WHITESPACE = re.compile(r"\s+")

#: A word must be at least this covered by the segment for its transcript to
#: belong to the segment.  Half, because a word straddling the boundary has one
#: clear side and the alternative -- dropping it -- leaves a sentence missing a
#: word at the seam between one person's speech and the next.
MIN_WORD_COVERAGE = 0.5


def collapse_whitespace(text: str) -> str:
    """Runs of whitespace to a single space, then trimmed."""

    return _WHITESPACE.sub(" ", text).strip()


def join_words(words: Sequence[TranscribedWord]) -> str:
    """A segment's words as one line, spaced the way the recogniser spaced them.

    This is why :class:`TranscribedWord` keeps the recogniser's token text
    verbatim, leading space and all: for a spaced script the space is part of
    the token (``" was"``), and for Chinese and Japanese there is none, so
    joining the tokens and collapsing what results gives correct text in both
    without a rule about which scripts take spaces.  Stripping the words first
    would glue Chinese together and break English apart.

    Text is assembled here rather than taken from the recogniser's own segment
    text because the edges get trimmed afterwards: a word in the padding belongs
    to whatever was being said before this person started.
    """

    return collapse_whitespace("".join(word.text for word in words))


def offset_words(
    words: Sequence[TranscribedWord], *, origin: float
) -> tuple[TranscribedWord, ...]:
    """Move slice-time words onto the video's timeline.

    One addition, and it is the one that is easy to leave out: nothing about a
    transcript that is a second late looks wrong on its own.
    """

    return tuple(
        TranscribedWord(
            text=word.text,
            start=word.start + origin,
            end=word.end + origin,
            probability=word.probability,
        )
        for word in words
    )


def clip_to_span(
    words: Sequence[TranscribedWord], *, start: float, end: float
) -> tuple[TranscribedWord, ...]:
    """The words that belong to ``[start, end]`` and not to its padding."""

    kept: list[TranscribedWord] = []
    for word in words:
        span = max(0.0, min(word.end, end) - max(word.start, start))
        length = word.end - word.start
        # A zero-length word -- the recogniser does emit them -- is kept when
        # its own timestamp is inside, since coverage cannot decide for it.
        inside = start <= word.start <= end if length <= 0.0 else span >= MIN_WORD_COVERAGE * length
        if inside:
            kept.append(word)
    return tuple(kept)


def hallucination_flags(
    *,
    words: int,
    avg_logprob: float,
    no_speech_prob: float,
    compression_ratio: float,
    min_avg_logprob: float,
    max_no_speech_prob: float,
    max_compression_ratio: float,
) -> tuple[str, ...]:
    """Why this transcript should not be trusted on its own.

    These are the published heuristics for a recogniser inventing text --
    decoding silence into "Thank you for watching", or looping on one phrase
    until the window ends.  None of them is a gate: a video where somebody
    whispers is a video with low log-probabilities throughout, and still the
    right answer.  They are recorded so the QA report can rank videos for review
    without a human watching a thousand hours to find the bad ones.
    """

    flags: list[str] = []
    if words == 0:
        flags.append("empty")
    if no_speech_prob > max_no_speech_prob:
        flags.append("no_speech")
    if avg_logprob < min_avg_logprob:
        flags.append("low_confidence")
    if compression_ratio > max_compression_ratio:
        flags.append("repetition")
    return tuple(flags)
