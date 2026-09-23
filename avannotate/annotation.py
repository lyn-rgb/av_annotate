"""Render and parse the two-level annotation script.

The script is the deliverable a human reads and a parser recovers.  Both
directions live here so they cannot drift: ``render_script`` followed by
``parse_script`` must reproduce the utterance stream exactly, and
``test_annotation.py`` asserts it.  A change to the grammar that breaks the
round trip is a bug in this module, not in the caller.

Format::

    [GLOBAL]
    Two people are discussing in a spacious living room.

    [SHOT 1 0.0s-12.4s]
    A bright living room with a sofa and a coffee table.
    <F001> whispering: <S>I was late for work today<E>
    <F002> surprised: <S>What's going on?<E>

    [SHOT 2 12.4s-31.0s]
    The camera moves closer to the window.
    <F001> <S>I forgot my phone<E>

Shape rules, all load-bearing:

* An utterance belongs to the shot containing its **start**; utterances are
  never split across shots.
* The paralinguistic tag is optional.  With a tag the line reads
  ``<F001> tag: <S>...<E>``; without one, ``<F001> <S>...<E>``.  Both parse,
  because the tag group is optional and the colon is optional with it.
* Captions are visual only.  Nothing in a caption depends on the audio chain.
* ``<F000>`` carries off-screen speech (narration, a phone call).  The grammar
  requires a face tag, so off-screen speech needs an id rather than no tag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from avannotate.schema import (
    SCHEMA_VERSION,
    Annotation,
    FaceTrack,
    Language,
    Shot,
    Utterance,
    VideoMeta,
    Word,
)

#: The closing marker: ``<E>`` written, either accepted on the way back in.
#:
#: ``<E>`` is what the clone_voice manifests -- and so LTX-2's training data --
#: actually use.  The converter that rewrites this markup for LTX-2 opens with
#: ``<S>…<E>`` and carries ``</S>`` only as a tolerance, in its own words,
#: "because the two ends are written by hand on the data side and drift between
#: files".  This is not written by hand, so it writes the real one; it reads
#: both, for the same reason the converter does -- an annotation that has been
#: through somebody's editor should still parse.
_CLOSE = r"(?:</\s*S\s*>|<\s*E\s*>)"

#: The utterance grammar.  Anchored at both ends so a line either matches
#: completely or is treated as caption text -- a partially-matching line would
#: otherwise lose its tail silently.
UTTERANCE_RE = re.compile(rf"^<F(\d+)>\s*([\w-]+)?:?\s*<S>(.*?){_CLOSE}$")

#: The same grammar, unanchored, for the flat rendering where utterances sit
#: inside a paragraph of prose: ``<caption>, <F001> tag: <S>...<E>, <F002>...``.
#: Kept as a separate pattern rather than relaxing the anchored one, so the
#: two-level form stays strict about what constitutes a whole utterance line.
UTTERANCE_SEARCH_RE = re.compile(rf"<F(\d+)>\s*([\w-]+)?:?\s*<S>(.*?){_CLOSE}")

_SHOT_HEADER_RE = re.compile(r"^\[SHOT\s+(\d+)\s+([0-9.]+)s-([0-9.]+)s\]$")

_GLOBAL_HEADER = "[GLOBAL]"

#: Characters that would break the line-oriented grammar.  A transcript
#: containing any of these is rejected rather than escaped: mangling spoken
#: words is worse than dropping the segment, and the caller has the timestamp
#: and the audio to fall back on.
_UNSAFE_IN_TEXT = ("<", ">", "\n", "\r")


class AnnotationFormatError(ValueError):
    """The annotation cannot be rendered, or the script cannot be parsed."""


def _format_seconds(value: float) -> str:
    return f"{value:.1f}"


def normalize_tag(tag: str | None) -> str | None:
    """The tag's rendered form, or ``None``.

    Single source of truth: the renderer writes this and the QA round trip
    compares against it.  Without a shared normalizer the two disagree on any
    tag that is not already lowercase, and the round trip reports a mismatch for
    a script that is correct.
    """

    if tag is None:
        return None
    normalized = tag.strip().lower()
    return normalized or None


def _utterance_line(utterance: Utterance) -> str:
    """Render one utterance, validating everything the parser would depend on."""

    text = utterance.text.strip()
    if not text:
        raise AnnotationFormatError(
            f"utterance at {utterance.start:.2f}s has empty text; "
            "drop it upstream and record a flag instead of rendering a blank span"
        )
    for unsafe in _UNSAFE_IN_TEXT:
        if unsafe in text:
            raise AnnotationFormatError(
                f"utterance text for {utterance.face_id} at {utterance.start:.2f}s "
                f"contains {unsafe!r}, which the line grammar cannot carry"
            )
    if "<S>" in text or "</S>" in text or "<E>" in text:
        raise AnnotationFormatError(
            f"utterance text for {utterance.face_id} contains the span markers"
        )

    normalized = normalize_tag(utterance.tag)
    if normalized is None:
        prefix = f"<{utterance.face_id}> "
    else:
        if not re.fullmatch(r"[\w-]+", normalized):
            raise AnnotationFormatError(
                f"tag {utterance.tag!r} is outside the [\\w-]+ charset the parser can recover"
            )
        prefix = f"<{utterance.face_id}> {normalized}: "
    return f"{prefix}<S>{text}<E>"


def _sorted_utterances(annotation: Annotation) -> tuple[Utterance, ...]:
    return tuple(sorted(annotation.utterances, key=lambda item: (item.start, item.end)))


def _assign_to_shots(
    utterances: tuple[Utterance, ...], shots: tuple[Shot, ...]
) -> list[tuple[Shot | None, list[Utterance]]]:
    """Group utterances under the shot containing their start.

    Returned in shot order, and total: every utterance lands somewhere.  The rule
    is "the last shot that starts at or before the utterance", which degrades
    sensibly when the shots do not tile the timeline -- a detector reports
    boundaries, so its first shot may start after zero -- and when an utterance
    starts exactly on the final boundary.  Choosing the containing shot strictly
    would leave such utterances homeless, and a homeless group rendered without a
    header would be parsed back into whichever shot preceded it.
    """

    ordered = sorted(shots, key=lambda shot: shot.index)
    if not ordered:
        return [(None, list(utterances))]

    buckets: list[tuple[Shot | None, list[Utterance]]] = [(shot, []) for shot in ordered]
    for utterance in utterances:
        chosen = buckets[0]
        for bucket in buckets:
            shot = bucket[0]
            assert shot is not None  # buckets was built from `ordered`
            if shot.start <= utterance.start:
                chosen = bucket
            else:
                break
        chosen[1].append(utterance)
    return buckets


def render_script(annotation: Annotation) -> str:
    """Render the annotation as the two-level script.

    Raises :class:`AnnotationFormatError` rather than emitting something the
    parser would read back differently.
    """

    blocks: list[str] = []

    global_lines = [_GLOBAL_HEADER]
    if annotation.global_caption.strip():
        global_lines.append(annotation.global_caption.strip())
    blocks.append("\n".join(global_lines))

    grouped = _assign_to_shots(_sorted_utterances(annotation), annotation.shots)
    for shot, utterances in grouped:
        if shot is None and not utterances:
            continue
        lines: list[str] = []
        if shot is not None:
            header = (
                f"[SHOT {shot.index} {_format_seconds(shot.start)}s-"
                f"{_format_seconds(shot.end)}s]"
            )
            lines.append(header)
            if shot.caption.strip():
                lines.append(shot.caption.strip())
        lines.extend(_utterance_line(item) for item in utterances)
        if lines:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def render_flat(annotation: Annotation) -> str:
    """Render as one paragraph: caption first, then utterances separated by ``", "``.

    This is the shape the format was originally sketched in, and the shape an
    instruction paragraph tends to take.  It discards shot boundaries, so it is
    a rendering rather than the deliverable -- but it round-trips, because
    :func:`parse_script` reads utterances out of prose as well as off their own
    lines.
    """

    parts: list[str] = []
    if annotation.global_caption.strip():
        parts.append(annotation.global_caption.strip())
    parts.extend(_utterance_line(utterance) for utterance in _sorted_utterances(annotation))
    return ", ".join(parts) + "\n"


@dataclass(frozen=True)
class ParsedShot:
    index: int
    start: float
    end: float
    caption: str
    utterances: tuple[Utterance, ...]


@dataclass(frozen=True)
class ParsedScript:
    """What :func:`parse_script` recovers.

    Deliberately not an :class:`Annotation`: the script does not carry face
    tracks, per-word timings or language, so returning an Annotation would mean
    inventing them.  ``utterances`` is the flat stream in render order.
    """

    global_caption: str
    shots: tuple[ParsedShot, ...]
    utterances: tuple[Utterance, ...]


def _utterance_from_match(match: re.Match[str]) -> Utterance:
    digits, tag, text = match.groups()
    # The script's grammar has no field for timings, so a parsed utterance
    # carries 0.0 for both ends.  Callers that need times must join against
    # ``annotation.json``; the script alone cannot supply them.
    return Utterance(
        face_id=f"F{int(digits):03d}",
        start=0.0,
        end=0.0,
        text=text.strip(),
        tag=tag.lower() if tag else None,
    )


def _extract_utterances(line: str) -> tuple[list[Utterance], str]:
    """Pull every utterance span out of one line, returning the leftover prose.

    A line that is exactly one utterance leaves no remainder.  A paragraph with
    its utterances inline leaves the caption, with the ``", "`` separators that
    used to join them tidied off the fragment edges.
    """

    utterances: list[Utterance] = []
    gaps: list[str] = []
    cursor = 0
    for match in UTTERANCE_SEARCH_RE.finditer(line):
        gaps.append(line[cursor : match.start()])
        utterances.append(_utterance_from_match(match))
        cursor = match.end()
    gaps.append(line[cursor:])
    remainder = " ".join(gap.strip(" ,") for gap in gaps if gap.strip(" ,"))
    return utterances, remainder


class _ShotBuilder:
    """Accumulates one shot block while the parser is inside it."""

    __slots__ = ("caption_parts", "end", "index", "start", "utterances")

    def __init__(self, index: int, start: float, end: float) -> None:
        self.index = index
        self.start = start
        self.end = end
        self.caption_parts: list[str] = []
        self.utterances: list[Utterance] = []

    def build(self) -> ParsedShot:
        return ParsedShot(
            index=self.index,
            start=self.start,
            end=self.end,
            caption=" ".join(self.caption_parts).strip(),
            utterances=tuple(self.utterances),
        )


def parse_script(text: str) -> ParsedScript:
    """Recover the structure from a rendered script.

    Tolerant of blank lines, of captions spanning several lines, and of
    utterances embedded in prose (the flat rendering).  Strict about the
    utterance grammar itself: a span that does not match is left as caption
    text rather than half-consumed.
    """

    shots: list[ParsedShot] = []
    orphans: list[Utterance] = []
    global_parts: list[str] = []
    current: _ShotBuilder | None = None
    seen_global = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line == _GLOBAL_HEADER:
            seen_global = True
            continue

        header = _SHOT_HEADER_RE.match(line)
        if header is not None:
            if current is not None:
                shots.append(current.build())
            current = _ShotBuilder(
                index=int(header.group(1)),
                start=float(header.group(2)),
                end=float(header.group(3)),
            )
            continue

        found, remainder = _extract_utterances(line)
        if found:
            if current is not None:
                current.utterances.extend(found)
            else:
                orphans.extend(found)
            if not remainder:
                continue
            line = remainder

        # What is left is caption prose: a shot's lands inside that shot, and
        # the global block's lands before the first shot.
        if current is not None:
            current.caption_parts.append(line)
        elif seen_global:
            global_parts.append(line)

    if current is not None:
        shots.append(current.build())

    flat: list[Utterance] = list(orphans)
    for shot in shots:
        flat.extend(shot.utterances)

    return ParsedScript(
        global_caption=" ".join(global_parts).strip(),
        shots=tuple(shots),
        utterances=tuple(flat),
    )


def _as_objects(value: object, field: str) -> list[dict[str, object]]:
    """Narrow a JSON field to a list of objects.

    Malformed input is named rather than skipped: a silently dropped shot or
    utterance would change the render without anyone noticing.
    """

    if value is None:
        return []
    if not isinstance(value, list):
        raise AnnotationFormatError(f"{field} must be a list, got {type(value).__name__}")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise AnnotationFormatError(
                f"{field} entries must be objects, got {type(item).__name__}"
            )
        result.append(dict(item))
    return result


def _as_strings(value: object, field: str) -> tuple[str, ...]:
    """Narrow a JSON field to a list of plain strings (flags, tracklet names)."""

    if value is None:
        return ()
    if not isinstance(value, list):
        raise AnnotationFormatError(f"{field} must be a list, got {type(value).__name__}")
    return tuple(str(item) for item in value)


def _as_mapping(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AnnotationFormatError(f"{field} must be an object, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _require(mapping: dict[str, object], key: str, field: str) -> object:
    if key not in mapping:
        raise AnnotationFormatError(f"{field} is missing required key {key!r}")
    return mapping[key]


def _require_str(mapping: dict[str, object], key: str, field: str) -> str:
    return str(_require(mapping, key, field))


def _require_float(mapping: dict[str, object], key: str, field: str) -> float:
    return _coerce_float(_require(mapping, key, field), f"{field}.{key}")


def _require_int(mapping: dict[str, object], key: str, field: str) -> int:
    return int(_coerce_float(_require(mapping, key, field), f"{field}.{key}"))


def _coerce_float(value: object, field: str) -> float:
    """Reject bools explicitly: ``True`` is an ``int`` in Python, and a boolean
    where a timestamp belongs means the producer wrote the wrong thing."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise AnnotationFormatError(f"{field} must be a number, got {type(value).__name__}")
    try:
        return float(value)
    except ValueError as error:
        raise AnnotationFormatError(f"{field} is not a number: {value!r}") from error


def _optional_float(value: object, field: str) -> float | None:
    return None if value is None else _coerce_float(value, field)


def annotation_from_dict(payload: dict[str, object]) -> Annotation:
    """Rebuild an :class:`Annotation` from the JSON deliverable.

    Strict about the fields the renderer depends on and lenient about the rest,
    so a producer that adds a field does not break a consumer that ignores it.
    """

    version = payload.get("schema_version")
    if version is None:
        raise AnnotationFormatError("annotation has no schema_version")
    if version != SCHEMA_VERSION:
        raise AnnotationFormatError(
            f"annotation schema_version is {version!r}, expected {SCHEMA_VERSION!r}"
        )

    video_raw = payload.get("video")
    if not isinstance(video_raw, dict):
        raise AnnotationFormatError("annotation has no video block")
    video = VideoMeta(
        video_id=str(video_raw["video_id"]),
        path=str(video_raw["path"]),
        duration=float(video_raw["duration"]),
        fps=float(video_raw["fps"]),
        width=int(video_raw["width"]),
        height=int(video_raw["height"]),
    )

    language = None
    language_raw = payload.get("language")
    if isinstance(language_raw, dict):
        language_mapping = {str(key): item for key, item in language_raw.items()}
        language = Language(
            code=_require_str(language_mapping, "code", "language"),
            confidence=_optional_float(language_mapping.get("confidence"), "language.confidence"),
            source=str(language_mapping.get("source", "unknown")),
        )

    shots = tuple(
        Shot(
            index=_require_int(item, "index", "shots[]"),
            start=_require_float(item, "start", "shots[]"),
            end=_require_float(item, "end", "shots[]"),
            caption=str(item.get("caption", "")),
        )
        for item in _as_objects(payload.get("shots"), "shots")
    )

    utterances: list[Utterance] = []
    for item in _as_objects(payload.get("utterances"), "utterances"):
        words = tuple(
            Word(
                text=_require_str(word, "text", "utterances[].words[]"),
                start=_require_float(word, "start", "utterances[].words[]"),
                end=_require_float(word, "end", "utterances[].words[]"),
            )
            for word in _as_objects(item.get("words"), "utterances[].words")
        )
        utterances.append(
            Utterance(
                face_id=_require_str(item, "face_id", "utterances[]"),
                start=_require_float(item, "start", "utterances[]"),
                end=_require_float(item, "end", "utterances[]"),
                text=_require_str(item, "text", "utterances[]"),
                tag=None if item.get("tag") is None else str(item["tag"]),
                audio_path=None if item.get("audio_path") is None else str(item["audio_path"]),
                words=words,
                confidence=_optional_float(item.get("confidence"), "utterances[].confidence"),
                flags=_as_strings(item.get("flags"), "utterances[].flags"),
            )
        )

    tracks: list[FaceTrack] = []
    for item in _as_objects(payload.get("face_tracks"), "face_tracks"):
        quality_raw = _as_mapping(item.get("quality"), "face_tracks[].quality")
        tracks.append(
            FaceTrack(
                face_id=_require_str(item, "face_id", "face_tracks[]"),
                first_seen=_require_float(item, "first_seen", "face_tracks[]"),
                last_seen=_require_float(item, "last_seen", "face_tracks[]"),
                speaks=bool(item.get("speaks", False)),
                total_speech=(
                    _optional_float(item.get("total_speech"), "face_tracks[].total_speech") or 0.0
                ),
                tracklets=_as_strings(item.get("tracklets"), "face_tracks[].tracklets"),
                quality={
                    key: _coerce_float(value, f"face_tracks[].quality.{key}")
                    for key, value in quality_raw.items()
                },
            )
        )

    stats = _as_mapping(payload.get("stats"), "stats")
    return Annotation(
        video=video,
        utterances=tuple(utterances),
        shots=shots,
        face_tracks=tuple(tracks),
        global_caption=str(payload.get("global_caption", "")),
        language=language,
        stats=stats,
    )


def annotation_to_dict(annotation: Annotation) -> dict[str, object]:
    """The JSON deliverable.

    The inverse of :func:`annotation_from_dict` and kept beside it so the two
    cannot drift: a field the writer renames and the reader does not is caught
    by the round-trip test rather than by a consumer a month later.

    Numbers are rounded rather than written at full precision.  A float that
    came out of a division has seventeen significant digits, and a deliverable
    where every timestamp differs in the last four is a diff nobody can read --
    while four decimals of a second is a hundredth of a frame.
    """

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "video_id": annotation.video.video_id,
            "path": annotation.video.path,
            "duration": round(annotation.video.duration, 4),
            "fps": round(annotation.video.fps, 4),
            "width": annotation.video.width,
            "height": annotation.video.height,
        },
        "shots": [
            {
                "index": shot.index,
                "start": round(shot.start, 4),
                "end": round(shot.end, 4),
                "caption": shot.caption,
            }
            for shot in annotation.shots
        ],
        "utterances": [
            {
                "face_id": item.face_id,
                "start": round(item.start, 4),
                "end": round(item.end, 4),
                "text": item.text,
                "tag": normalize_tag(item.tag),
                "audio_path": item.audio_path,
                "words": [
                    {
                        "text": word.text,
                        "start": round(word.start, 4),
                        "end": round(word.end, 4),
                    }
                    for word in item.words
                ],
                "confidence": (
                    None if item.confidence is None else round(item.confidence, 4)
                ),
                "flags": list(item.flags),
            }
            for item in annotation.utterances
        ],
        "face_tracks": [
            {
                "face_id": track.face_id,
                "first_seen": round(track.first_seen, 4),
                "last_seen": round(track.last_seen, 4),
                "speaks": track.speaks,
                "total_speech": round(track.total_speech, 4),
                "tracklets": list(track.tracklets),
                "quality": {
                    key: round(value, 4) for key, value in track.quality.items()
                },
            }
            for track in annotation.face_tracks
        ],
        "global_caption": annotation.global_caption,
    }

    if annotation.language is not None:
        payload["language"] = {
            "code": annotation.language.code,
            "confidence": (
                None
                if annotation.language.confidence is None
                else round(annotation.language.confidence, 4)
            ),
            "source": annotation.language.source,
        }
    if annotation.stats:
        payload["stats"] = dict(annotation.stats)

    # Checked here rather than trusted: a producer writing a document its own
    # reader rejects would fail at the far end of a batch, on the one video
    # nobody is watching.
    annotation_from_dict(payload)
    return payload
