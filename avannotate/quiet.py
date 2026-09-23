"""Run a chatty library without letting it into the log, keeping what it said.

Two of the model libraries in this pipeline narrate, and both narrate *per
file* rather than per process:

* ClearerVoice (S7) announces the model it is loading, says something about
  every file, and shells out to ffmpeg for the audio -- whose output inherits
  this process's and lands in the log too.  S7 calls it once per face per
  speech segment.
* The taggers (S9) draw tqdm bars, print a timing dict per call, and let
  ``transformers`` write its warnings to the console.  S9 calls them once per
  speech segment.

Either way a corpus produces more of the library's output than of ours, and it
lands on top of the progress bar -- which is the stage's own report, and the
whole reason the per-video lines were replaced by it.  Noise arriving from a
direction the rest of the pipeline had already dealt with.

**Kept rather than thrown away.**  The yielded callable hands back what was
said, so a caller can put its last lines in the error it raises: a failure
inside a library is exactly when its output is worth having, and a message with
no context is what discarding it would buy.

**Both descriptors *and* the Python wrappers over them, in that order.**  A
``print`` already sitting in Python's buffer would otherwise arrive after the
restore and appear in the log anyway; so the buffer is flushed on the way in and
again on the way out.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager

#: How much of a library's chatter to put in an error message.  The last few
#: lines are where the actual complaint is; the rest is its banner.
TAIL_LINES = 5


def tail(text: str, *, lines: int = TAIL_LINES) -> str:
    """The last few lines of what a library said, for an error message."""

    return " | ".join(text.splitlines()[-lines:])


@contextmanager
def quiet() -> Iterator[Callable[[], str]]:
    """Send everything written to stdout and stderr to a file instead.

    Yields a callable that returns what was captured, labelled by stream.  Two
    files rather than one: Python's ``stdout`` is block buffered and a
    subprocess's ``stderr`` is not, so sharing a file interleaves them in
    whatever order the buffers happened to flush -- a line printed first
    regularly lands last, and the tail of that is not the end of anything.  The
    tail is the whole point of keeping this.
    """

    with tempfile.TemporaryFile() as out_sink, tempfile.TemporaryFile() as err_sink:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        os.dup2(out_sink.fileno(), 1)
        os.dup2(err_sink.fileno(), 2)

        def said() -> str:
            sys.stdout.flush()
            sys.stderr.flush()
            parts: list[str] = []
            for label, sink in (("stdout", out_sink), ("stderr", err_sink)):
                sink.seek(0)
                text = sink.read().decode("utf-8", "replace").strip()
                if text:
                    parts.append(f"{label}: {text}")
            return " / ".join(parts)

        try:
            yield said
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
