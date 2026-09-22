"""Checking the names a caption uses, and removing the ones that are wrong.

A caption is the one deliverable a model writes in free text, and the one place
a face id appears outside the utterance grammar.  The format's parser looks for
``<F001>`` only when it is followed by ``<S>...<E>``, so a name in a caption is
invisible to it -- which is convenient for parsing and dangerous for truth: a
caption naming the wrong person is not caught by anything downstream.

What is checked is narrow and total.  The model was told who is in the shot; a
name it uses that was not on that list is a name it invented, and it comes out.
That is not a judgement about the description -- the model may be right that
somebody is in the room -- it is a statement about what can be verified.  A name
the tracker did not see is a claim nothing in this pipeline can support, and the
alternative to removing it is shipping it.

Names are written bare, as ``F001``, not as ``<F001>``.  Angle brackets in this
format belong to the utterance grammar, and a caption is not an utterance; a
future parser that sees ``<`` should be able to assume a speech line follows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: What a face id looks like in prose.  Word boundaries on both sides so that
#: ``F0012`` and ``XF001`` are not read as references -- the schema's ids are
#: always exactly three digits and always stand alone.
REFERENCE = re.compile(r"\bF\d{3}\b")

_WHITESPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Checked:
    """Text that has been through :func:`check_references`, and the audit."""

    text: str
    referenced: tuple[str, ...]
    dropped: tuple[str, ...]
    empty: bool


def referenced(text: str) -> tuple[str, ...]:
    """Every face id the text names, in the order it names them."""

    return tuple(dict.fromkeys(REFERENCE.findall(text)))


def check_references(text: str, *, allowed: tuple[str, ...]) -> Checked:
    """Remove names that are not on the roster, and say which were removed.

    Removing rather than discarding the whole caption: what survives is still
    true about everything except the name that was taken out, and a shot with a
    caption is worth more than a shot without one.  The text can end up reading
    awkwardly -- "and are sitting on the sofa" -- which is why the removal is
    recorded and flagged rather than done quietly.

    A caption that was *only* names is left empty, and ``empty`` says so: the
    caller decides whether to keep a blank line, and a blank line that looks
    like a description is worse than an obvious absence.
    """

    permitted = set(allowed)
    kept: list[str] = []
    dropped: list[str] = []

    for name in referenced(text):
        if name in permitted:
            kept.append(name)
        else:
            dropped.append(name)

    if not dropped:
        stripped = text.strip()
        return Checked(text=stripped, referenced=tuple(kept), dropped=(), empty=not stripped)

    cleaned = REFERENCE.sub(
        lambda match: match.group(0) if match.group(0) in permitted else "", text
    )
    # Collapsing runs of spaces only, not newlines: the model is asked for one
    # line, but a caption that came back as two is not this function's to join.
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    # Removing a name can leave the sentence starting with its punctuation.
    # Stripping that is not rewriting -- it is finishing the removal.
    cleaned = cleaned.strip().strip(" ,;:").strip()
    return Checked(
        text=cleaned,
        referenced=tuple(kept),
        dropped=tuple(dropped),
        empty=not cleaned,
    )


def flags_for(checked: Checked, *, known: set[str]) -> tuple[str, ...]:
    """Why a checked caption should be looked at.

    ``invented`` and ``misattributed`` are separated because they mean different
    things about the model: the first is a name that exists nowhere in the
    video, which is a hallucination, and the second is a real person named in a
    shot they are not in, which is a grounding failure.  Both are stripped, but
    a run with many of one and none of the other is telling you something.
    """

    found: list[str] = []
    if any(name not in known for name in checked.dropped):
        found.append("invented_name")
    if any(name in known for name in checked.dropped):
        found.append("misattributed_name")
    if checked.empty:
        found.append("empty")
    return tuple(found)
