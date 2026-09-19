"""What the captioning model is asked.

The prompts are here, as functions, because they are the part of this stage that
can be read and argued with.  A prompt embedded in an adapter's method is a
prompt nobody reviews.

Two instructions carry most of the weight.

**Describe what is visible, not what is happening in the story.**  The
deliverable's captions are scenery: "A bright living room with a sofa and a
coffee table".  Asking a model to describe what it sees reliably produces what
it *infers* otherwise -- who is angry at whom, what they are about to do -- and
inference in a caption is indistinguishable from observation once it is
rendered.

**Use the identifiers you were given, and only those.**  The roster is the
model's to use and not to extend.  Saying so is not enough on its own -- the
stage checks the output against the same roster -- but a model that knows the
rule breaks it less often, and the check is cheap either way.
"""

from __future__ import annotations

from collections.abc import Sequence

from avannotate.caption.types import ShotSample

#: Shared by both prompts.  Kept in one place so the two cannot drift into
#: asking for the same thing in two different ways.
RULES = """Rules:
- Describe only what is visible in the images. Do not infer feelings, intentions, \
relationships, or what happened before or after.
- Do not quote or mention any dialogue or speech.
- Write one or two sentences of plain prose. No lists, no headings, no preamble.
- Do not begin with "The image" or "This frame"; describe the scene directly."""

SYSTEM = (
    "You write short, factual visual descriptions of video frames for an "
    "annotation dataset."
)


def _roster(identities: Sequence[str]) -> str:
    """The people line, or an explicit statement that there are none.

    Saying "no people are present" rather than saying nothing at all: silence
    leaves the model free to name somebody from the previous request, and it
    has no way to know that this shot is the empty one.
    """

    if not identities:
        return "No tracked people appear in these frames."
    listed = ", ".join(identities)
    return (
        f"These frames contain tracked people: {listed}. "
        f"If you refer to a person, use their identifier exactly as written "
        f"({listed}) and use no other identifier."
    )


def shot_prompt(
    shot: ShotSample, *, total: int, describe_people: bool = True
) -> str:
    """The request for one shot."""

    # ``index`` is already 1-based: it is the number the rendered script writes
    # as ``[SHOT 1 ...]``, and adding one here would name every shot wrongly.
    parts = [
        f"These are {len(shot.times)} frames from shot {shot.index} of {total}, "
        f"spanning {shot.start:.1f} to {shot.end:.1f} seconds.",
    ]
    if describe_people:
        parts.append(_roster(shot.identities))
    else:
        parts.append("Do not refer to anyone by identifier.")
    parts.append("Describe the setting, the lighting, and what the people are doing.")
    parts.append(RULES)
    return "\n".join(parts)


def global_prompt(
    *, identities: Sequence[str], frames: int, shots: int
) -> str:
    """The request for the whole video."""

    parts = [
        f"These are {frames} frames sampled from across a video, one from each of "
        f"{shots} shots.",
        _roster(identities),
        "Describe the video's overall setting in one sentence: where it takes "
        "place and who is in it.",
        RULES,
    ]
    return "\n".join(parts)
