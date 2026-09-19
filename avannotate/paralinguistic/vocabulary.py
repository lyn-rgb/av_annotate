"""The closed set of tags, and how a model's label becomes one.

Three models each report labels from their own taxonomy, and the deliverable
carries exactly one tag per utterance, drawn from a vocabulary the script format
can parse.  This module is the whole of that reduction: what the labels mean,
which ones mean nothing, and how a label that is not in the table is handled.

Why the vocabulary is closed
----------------------------

A tag is rendered inside the utterance line and recovered by
``schema.TAG_PATTERN``, so a tag containing a space or a colon would not survive
the round trip -- it would be read back as part of the spoken text.  Rather than
validate late and drop what fails, the vocabulary is fixed here and every path
that produces a tag goes through :func:`tag_for`, which can only return a member.

Labels that mean nothing
------------------------

``neutral``, ``other`` and ``unknown`` are results, not descriptions: they say
the model found no distinguishing affect.  The format makes the tag optional
precisely so that an utterance with nothing to say about it renders without one,
and a rendered ``neutral:`` is worse than silence -- it looks like a finding.

Labels that are not in the table
--------------------------------

Dropped, and counted by the caller.  The taggers have hundreds of labels between
them and this pipeline wants a few dozen: the ones that describe how a person
delivered a line.  Mapping all of them would mean inventing distinctions the
deliverable does not use, and the count of unmapped labels is the signal that a
model's taxonomy has moved.

The tables are keyed by :func:`slug`, so a label is matched case-insensitively
and ``"Crying, sobbing"`` matches the same entry as ``crying sobbing``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from avannotate.paralinguistic.types import TagScore

#: Must match ``schema.TAG_PATTERN``.  Recompiled here rather than imported
#: because the two mean different things: that one recovers a tag from text,
#: this one proves a tag can be recovered from text.
_TAG = re.compile(r"[\w-]+")

#: Characters outside this set become hyphens in a slug.
_NOT_WORD = re.compile(r"[\W_]+", re.UNICODE)

#: A model's way of saying "nothing distinctive here".  Never rendered.
SILENT: frozenset[str] = frozenset({"neutral", "other", "unknown", "none", "unintelligible"})

EMOTION_TAGS: frozenset[str] = frozenset(
    {"angry", "disgusted", "fearful", "happy", "sad", "surprised"}
)

#: How the words were produced, as opposed to how the speaker felt.
DELIVERY_TAGS: frozenset[str] = frozenset(
    {"whispering", "shouting", "breathy", "mumbling", "singing", "laughing"}
)

#: Non-speech sounds.  These are not delivery: there may be no words at all.
EVENT_TAGS: frozenset[str] = frozenset(
    {
        "laughter",
        "crying",
        "screaming",
        "sigh",
        "cough",
        "sneeze",
        "sniff",
        "breathing",
        "throat-clearing",
    }
)

ALL_TAGS: frozenset[str] = EMOTION_TAGS | DELIVERY_TAGS | EVENT_TAGS

#: ``raw label slug -> tag`` for emotion2vec's classes.
#:
#: The released model's nine labels are **bilingual with a slash and are not
#: hardcoded in the source** -- they are read line by line from the checkpoint's
#: ``tokens.txt``, so these strings are what actually comes back.  Both halves
#: are listed: the joined form is what this model returns, and the English-only
#: form is what an English-only checkpoint would.
#:
#: Two entries are easy to get wrong and were wrong here first: the fifth label
#: is 中立 (neutral), not 中性, and the ninth is the literal string ``<unk>``
#: rather than ``unknown`` -- the model card's prose calls it "unknown" and the
#: runtime never says that.
EMOTION_LABELS: Mapping[str, str] = {
    "angry": "angry",
    "生气-angry": "angry",
    "disgusted": "disgusted",
    "厌恶-disgusted": "disgusted",
    "fearful": "fearful",
    "恐惧-fearful": "fearful",
    "happy": "happy",
    "开心-happy": "happy",
    "sad": "sad",
    "难过-sad": "sad",
    "surprised": "surprised",
    "吃惊-surprised": "surprised",
    "neutral": "neutral",
    "中立-neutral": "neutral",
    "other": "other",
    "其他-other": "other",
    "unknown": "unknown",
    "unk": "unknown",
}

#: ``raw label slug -> tag`` for the voice-tagging model.
#:
#: **This model has no label vocabulary and no scores.**  It is a Whisper
#: fine-tune that *generates* a comma-separated string of tags -- not a
#: classifier, despite its ``audio-classification`` tag on Hugging Face, and it
#: ships no ``id2label``.  Its own card counts "194 unique tags" in a sample of
#: 570 outputs and publishes only the most frequent, so there is no finite list
#: to match against and no confidence to threshold.
#:
#: The entries below are the published spellings that map onto this vocabulary.
#: Anything else the model invents -- and it does invent, its own examples
#: contain "natural-Sounding" and "natural-Suitable for Work" -- is dropped and
#: counted, which is the correct outcome for a generator whose output space is
#: open.
DELIVERY_LABELS: Mapping[str, str] = {
    "whispering": "whispering",
    "whispered": "whispering",
    "whispery-voice": "whispering",
    "whisper-talk-style": "whispering",
    "asmr-whisper-delivery": "whispering",
    "shouting": "shouting",
    "angry-shouting": "shouting",
    # A raised voice, from the model that describes delivery.  AudioSet's
    # "Screaming" is a different thing -- an event that need not be speech --
    # and the two are kept apart by being in different dimensions' tables.
    "screaming": "shouting",
    "breathy": "breathy",
    "slightly-breathy": "breathy",
    "breathy-voice": "breathy",
    "laughing-while-speaking": "laughing",
    "giggling-delivery": "laughing",
    "sing-speaking": "singing",
    "crying": "crying",
}

#: ``raw label slug -> tag`` for AudioSet's classes, as PANNs' wrapper exposes
#: them.  The other ~500 describe music, vehicles, animals and machinery and are
#: dropped by not being here.
#:
#: The spellings are AudioSet's own, taken from ``class_labels_indices.csv``:
#: 75 of the 527 contain a comma and 245 contain a space, which is why the table
#: is keyed by slug rather than by the raw string.
EVENT_LABELS: Mapping[str, str] = {
    "laughter": "laughter",
    "chuckle-chortle": "laughter",
    "giggle": "laughter",
    "snicker": "laughter",
    "belly-laugh": "laughter",
    "crying-sobbing": "crying",
    "sob": "crying",
    "whimper": "crying",
    "screaming": "screaming",
    "singing": "singing",
    "sigh": "sigh",
    "cough": "cough",
    "throat-clearing": "throat-clearing",
    "sneeze": "sneeze",
    "sniff": "sniff",
    "sniffing": "sniff",
    "breathing": "breathing",
    "wheeze": "breathing",
    "gasp": "breathing",
    "pant": "breathing",
}

#: Labels that are known, deliberately unused, and therefore not reported as
#: unmapped.
#:
#: "Unmapped" is supposed to mean *we have never seen this label* -- the signal
#: that a checkpoint's taxonomy has moved underneath the pipeline.  Without this
#: list that signal would be buried immediately: AudioSet's ``Speech`` fires on
#: every speech segment with high confidence, and a confidence-gated unmapped
#: count built only from :data:`EVENT_LABELS` would report it forever.
#:
#: What is here is either a class this pipeline deliberately leaves to another
#: dimension -- shouting and whispering are delivery, not events, and the
#: delivery model owns them -- or a description of the recording rather than of
#: the person: volume, style, and what kind of room it was.
IGNORED_LABELS: Mapping[str, frozenset[str]] = {
    "emotion": frozenset(),
    "delivery": frozenset(
        {
            # Volume and style: real descriptions, but not tags this deliverable
            # renders, and firing on most segments.
            "quiet",
            "very-quiet",
            "loud",
            "very-loud",
            "normal-loudness",
            "slightly-dynamic",
            "dynamic",
            "monotone",
            "natural-speaking",
            "casual-speaking-style",
            "formal-style",
            "narration-style-delivery",
            "newsreader-style",
            "storytelling-style",
            "monologue-style",
            "authoritative-style",
            "high-energy-delivery",
            "slow-deliberate-delivery",
            "precise-articulation",
            "neutral-articulation",
            "slightly-imprecise-articulation",
            "naturalness",
            "natural-genuine",
            "slightly-unnatural",
            "fluent",
            "halting-speech",
            "disfluent",
            "neutral-airflow",
            "modal-voice",
            "pressed-voice",
            "slack-voice",
            "rough-voice",
            "tense-voice",
            "slightly-tense-voice",
            "strained-voice",
            "strained-delivery",
            "fatigued-delivery",
            "out-of-breath-delivery",
            "gasping-delivery",
            "pleading-tone",
            "sad-speaking",
            "falling-intonation",
            "irregular-intonation",
            "ranting-style",
            "ranting-style-delivery",
            # Owned by the event dimension, which outranks this one.
            "sighing-delivery",
            # Not about the voice at all.  Kept out of the unmapped count so it
            # stays a signal about the model rather than about content.
            "suitable-for-work",
            "not-suitable-for-work",
        }
    ),
    "event": frozenset(
        {
            # Speech of some kind, which every segment is.
            "speech",
            "conversation",
            "narration-monologue",
            "speech-synthesizer",
            # Delivery, owned by the delivery dimension.
            "whispering",
            "shout",
            "yell",
            "babbling",
            "murmur",
            # Descriptions of the recording rather than of a person.
            "silence",
            "inside-small-room",
            "inside-large-room",
            "outside-urban-or-man-made",
            "outside-rural-or-natural",
            "field-recording",
            "music",
            "background-noise",
            "white-noise",
            "pink-noise",
            "noise",
            "environmental-noise",
        }
    ),
}


def ignored(label: str, *, dimension: str) -> bool:
    """Whether this label is a known one this pipeline chooses not to use."""

    return slug(label) in IGNORED_LABELS.get(dimension, frozenset())

#: Which table each dimension's labels are looked up in.  Defined after the
#: tables it names, not before: a name is resolved when the function runs, but
#: ruff reads the module top to bottom and is right to complain.
_TABLE: Mapping[str, Mapping[str, str]] = {
    "emotion": EMOTION_LABELS,
    "delivery": DELIVERY_LABELS,
    "event": EVENT_LABELS,
}


def slug(label: str) -> str:
    """A model's label in the shape a tag can take.

    Lowercased, with every run of non-word characters as a hyphen.  AudioSet's
    ``"Crying, sobbing"`` becomes ``crying-sobbing``; a bilingual label's slash
    becomes a hyphen too, which is why the tables above carry the joined form.
    """

    return _NOT_WORD.sub("-", label.strip().lower()).strip("-")


def is_tag(text: str) -> bool:
    """Whether ``text`` is a tag the script format can recover."""

    return bool(_TAG.fullmatch(text))


def tag_for(label: str, *, dimension: str) -> str | None:
    """The tag a label maps to, or ``None`` if it does not map to one.

    ``None`` covers both "the model said nothing distinctive" and "this label is
    not in the table".  The caller cannot act differently on the two -- neither
    produces a tag -- but :func:`unmapped` can, and the difference is what tells
    a reader whether a model's taxonomy has changed underneath the pipeline.
    """

    mapped = _TABLE.get(dimension, {}).get(slug(label))
    if mapped is None or mapped in SILENT:
        return None
    return mapped


def unmapped(
    scores: Sequence[TagScore], *, dimension: str, min_score: float
) -> tuple[str, ...]:
    """Labels the table has no entry for and the model was confident about.

    Reported rather than raised: a checkpoint that added a class is not a reason
    to fail a thousand-hour batch, but it is a reason for a reader to look.

    Gated on the score, because the signal is meant to be *the model is sure of
    something we have no word for*.  Every tagger returns a long tail of labels
    it barely believes and this pipeline has no use for; counting those from day
    one would bury the one they call for.  Labels in :data:`IGNORED_LABELS` are
    left out for the same reason -- known is not unknown.
    """

    table = _TABLE.get(dimension, {})
    found: list[str] = []
    for item in scores:
        if item.score < min_score:
            continue
        key = slug(item.label)
        if key in table or key in SILENT or ignored(item.label, dimension=dimension):
            continue
        found.append(item.label)
    return tuple(found)
