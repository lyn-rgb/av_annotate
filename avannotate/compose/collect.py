"""Assembling the deliverable from what the ten stages left behind.

Each function here takes records and returns one part of the
:class:`~avannotate.schema.Annotation`.  They are pure, which is the point:
this is where a mistake becomes part of the output file, and a mistake here is
not an exception -- it is a transcript attached to the wrong person, or a person
missing from the face list, in a document that reads as finished.

Everything joins on S7's segment name.  S7 decided which spans exist and wrote
one audio file for each; S8 transcribed those files, S9 tagged them, and this
puts the three back together.  Joining on anything else -- a timestamp, an
index, a sort order -- would be a second definition of what a segment is, and
the two definitions would disagree on exactly the videos nobody checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from avannotate.coercion import coerce_number
from avannotate.faces.track import Tracklet
from avannotate.interval import Interval, total_duration
from avannotate.schema import FaceTrack, Shot, Utterance, Word
from avannotate.segment import SpeechSegment


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _weighted(
    values: Sequence[float], weights: Sequence[float]
) -> float:
    """Mean weighted by how much evidence each tracklet contributes.

    Weighted by detection count, so a two-frame sliver does not count as much as
    a track that was solid for ten seconds when the two are averaged into one
    number for the person.
    """

    total = sum(weights)
    if total <= 0.0:
        return _mean(values)
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def face_tracks(
    identity_tracks: Mapping[str, Sequence[int]],
    tracklets: Sequence[Tracklet],
    speech: Mapping[str, Sequence[Interval]],
) -> tuple[FaceTrack, ...]:
    """One entry per tracked person, whether or not they ever spoke.

    A person who never speaks is still in the deliverable: they are in the
    video, and a face list that only held talkers would make "who is in this
    video" unanswerable.  ``speaks`` is what distinguishes them, and it is
    derived rather than assumed.

    ``first_seen`` and ``last_seen`` come from the detections themselves rather
    than from the tracklets' frame bounds, because a tracklet can start before
    its first detection -- the tracker's Kalman prediction is not evidence that
    anyone was there.
    """

    by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    tracks: list[FaceTrack] = []

    for face_id in sorted(identity_tracks):
        members = [
            by_id[track_id] for track_id in identity_tracks[face_id] if track_id in by_id
        ]
        seen = [
            (detection.time, tracklet)
            for tracklet in members
            for detection in tracklet.detections
        ]
        if not seen:
            continue

        spoken = total_duration(speech.get(face_id, ()))
        weights = [float(tracklet.quality.frames) for tracklet in members]

        tracks.append(
            FaceTrack(
                face_id=face_id,
                first_seen=min(time for time, _ in seen),
                last_seen=max(time for time, _ in seen),
                speaks=spoken > 0.0,
                total_speech=spoken,
                tracklets=tuple(str(tracklet.track_id) for tracklet in members),
                quality={
                    "tracklets": float(len(members)),
                    "detections": float(len(seen)),
                    "mean_score": _weighted(
                        [tracklet.quality.mean_score for tracklet in members], weights
                    ),
                    "motion": _weighted(
                        [tracklet.quality.motion for tracklet in members], weights
                    ),
                },
            )
        )
    return tuple(tracks)


def utterances(
    segments: Sequence[SpeechSegment],
    transcripts: Mapping[str, Mapping[str, object]],
    tags: Mapping[str, str],
) -> tuple[Utterance, ...]:
    """One utterance per segment that has words in it.

    A segment whose transcript came back empty is dropped rather than emitted
    with empty text.  It has no audio worth hearing and no words to render, and
    a line like ``<F001> <S></S>`` in the script is a promise of speech that the
    pipeline cannot keep -- better counted in the report than written down.

    Ordered by start, then by the identifiers, so two runs over the same video
    produce byte-identical files.
    """

    built: list[Utterance] = []
    for segment in segments:
        record = transcripts.get(segment.name)
        if record is None:
            continue
        text = str(record.get("text", "")).strip()
        if not text:
            continue

        # The words are stripped here, unlike the ones S8 keeps, and the
        # difference is deliberate.  S8 holds the recogniser's tokens verbatim
        # because the leading space is what tells ``join_words`` how to space a
        # script -- and that job is finished by the time the text is stored on
        # the line above.  What is left is a word, and a word is "How".
        words = tuple(
            Word(
                text=str(item.get("text", "")).strip(),
                start=coerce_number(item.get("start", 0.0), "word.start"),
                end=coerce_number(item.get("end", 0.0), "word.end"),
            )
            for item in _objects(record.get("words"))
        )
        confidence = record.get("confidence")
        built.append(
            Utterance(
                face_id=segment.identity,
                start=segment.start,
                end=segment.end,
                text=text,
                tag=tags.get(segment.name),
                audio_path=segment.audio,
                words=words,
                confidence=(
                    float(confidence)
                    if isinstance(confidence, (int, float))
                    else _confidence(record)
                ),
                flags=_flags(record),
            )
        )

    built.sort(key=lambda item: (item.start, item.end, item.face_id))
    return tuple(built)


def _confidence(record: Mapping[str, object]) -> float | None:
    """What the recogniser thought of its own answer.

    The average log-probability, which is the number the hallucination
    heuristics are already read against -- so a reader comparing a flagged
    utterance to its score is comparing like with like.  ``None`` rather than
    zero when the recogniser reported nothing: zero is a terrible score and
    absent is not a score at all.
    """

    value = record.get("avg_logprob")
    return float(value) if isinstance(value, (int, float)) else None


def _flags(record: Mapping[str, object]) -> tuple[str, ...]:
    raw = record.get("flags")
    found = [str(item) for item in raw] if isinstance(raw, list) else []
    return tuple(dict.fromkeys(found))


def _objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def shots(
    boundaries: Sequence[tuple[int, float, float]], captions: Mapping[int, str]
) -> tuple[Shot, ...]:
    """S0's shot boundaries with S10's captions attached, by index.

    A shot S10 did not caption keeps the boundary and gets empty text.  Dropping
    it would silently shorten the video: the shot list is the timeline, and an
    uncaptioned shot is still a shot.
    """

    return tuple(
        Shot(
            index=index,
            start=start,
            end=end,
            caption=captions.get(index, ""),
        )
        for index, start, end in boundaries
    )
