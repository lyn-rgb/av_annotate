"""Deciding what language the video is in.

The deliverable carries one language per video, and it is not a decoration: it
is what a reader uses to know whether the transcript they are looking at is
supposed to be in a script they can read, and what a downstream model uses to
pick a tokenizer.

Two things make this less obvious than taking the recogniser's first answer.

**Detection is unreliable on short clips.**  Whisper identifies a language from
a single 30-second window, and a two-second utterance is far less evidence than
that.  Its guess on a short clip is not a weak version of the right answer; it
is frequently a confident wrong one, and a wrong language turns the transcript
into a translation, which reads fluently and is worth nothing.

**A video can hold more than one language.**  Someone speaking English inside a
Chinese video is ordinary.  Labelling the video by whichever language came first
would then produce a wrong transcript for every segment before that point.

So detection is aggregated by *how long* each language was spoken, not by how
often it was guessed, and only segments long enough to be evidence get a vote.
The aggregated answer is then what short segments are transcribed *with* --
forced, rather than detected -- which is what stops them being translated.
Segments that voted confidently for a minority language are left alone and
counted instead; see :func:`disagreements`.
"""

from __future__ import annotations

from collections.abc import Sequence

from avannotate.asr.types import LanguageVote, Transcription

#: A segment shorter than this is not evidence about the video's language.
#: Whisper detects from a 30 s window; a couple of seconds is a small fraction of
#: that, and measured behaviour on such clips is a confident guess rather than a
#: hedged one.
MIN_VOTE_SECONDS = 3.0

#: Below this the recogniser was not sure either, so the segment does not vote.
MIN_VOTE_PROBABILITY = 0.5


def collect_votes(
    transcriptions: Sequence[tuple[float, Transcription]],
    *,
    min_seconds: float = MIN_VOTE_SECONDS,
    min_probability: float = MIN_VOTE_PROBABILITY,
) -> tuple[LanguageVote, ...]:
    """Per-language totals over the segments that are evidence, ranked.

    Takes ``(duration, transcription)`` pairs rather than the segments, so the
    duration that counts is the speech's and not the padding's.
    """

    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    probabilities: dict[str, float] = {}
    for duration, transcription in transcriptions:
        code = transcription.language
        if not code or code == "unknown" or not transcription.detected:
            continue
        if duration < min_seconds or transcription.language_probability < min_probability:
            continue
        seconds[code] = seconds.get(code, 0.0) + duration
        counts[code] = counts.get(code, 0) + 1
        probabilities[code] = probabilities.get(code, 0.0) + (
            transcription.language_probability * duration
        )

    votes = [
        LanguageVote(
            code=code,
            seconds=round(total, 4),
            segments=counts[code],
            probability=probabilities[code] / total if total > 0 else 0.0,
        )
        for code, total in seconds.items()
    ]
    # Ranked by speech time, then by segment count, then by code: the first two
    # are the evidence and the last is only there so two equally evidenced
    # languages always resolve the same way on every run.
    votes.sort(key=lambda vote: (-vote.seconds, -vote.segments, vote.code))
    return tuple(votes)


def decide(
    votes: Sequence[LanguageVote], *, fallback: str | None = None
) -> str:
    """The video's language, or ``fallback`` when nothing voted.

    A video whose every segment was too short to vote still has a language --
    it is just not one this pipeline can establish, so it says so rather than
    picking the segment that happened to be longest.
    """

    if votes:
        return votes[0].code
    return fallback or "unknown"


def disagreements(
    transcriptions: Sequence[tuple[float, Transcription]], *, language: str
) -> tuple[tuple[str, float], ...]:
    """Segments the recogniser heard as a different language than the video's.

    Reported rather than corrected.  A five-second English sentence inside a
    Chinese video is either a real code-switch, which should be transcribed as
    English, or a detection error, which forcing Chinese would turn into
    nonsense.  Neither is distinguishable from here, so both are counted and
    the QA report shows the rate.
    """

    out: dict[str, float] = {}
    for duration, transcription in transcriptions:
        code = transcription.language
        if not code or code == language or code == "unknown":
            continue
        if not transcription.detected or duration < MIN_VOTE_SECONDS:
            continue
        out[code] = out.get(code, 0.0) + duration
    return tuple(sorted((code, round(seconds, 4)) for code, seconds in out.items()))
