"""One model per stage per process, instead of one per video.

Every stage builds its model inside ``run()``, and ``run()`` is called once per
video.  So a corpus of a thousand videos builds the captioner a thousand times
-- and the captioner is 16 GB read off a disk.  Nothing about a detector, a
diarizer or a captioner depends on which video it is looking at: each ``run``
builds it from its config, uses it, and throws it away.  That is the whole of
the wall clock for the large ones.

**Why the cache holds one.**  The ten stages' models come to more than a 24 GB
card holds, so a cache that never evicted would trade a slow batch for one that
cannot start -- and it would not even help, because the batch runs one video
through every stage before starting the next, so the model needed at any moment
cycles through all ten.  Holding one is exactly right for a run that goes
*stage-major* -- S1 over the whole list, then S2 over the whole list -- and
that is what :mod:`scripts/run_corpus.sh` drives.  Under a video-major run it is
a no-op rather than a regression: the same builds happen in the same order.

**The key is the config, not the stage.**  Changing a stage's threshold or its
checkpoint must build a new model, and a cache keyed on the stage name alone
would go on serving the old one -- silently, and with every score computed
against settings nobody chose.  Hashing the mapping closes that, at the cost of
a rebuild whenever the config changes, which is what a config change means.

**Scope is the process, which is right for a run and wrong for a test.**  The
suite substitutes fake ``build_*`` functions and calls ``run()`` hundreds of
times in one process, so an entry left behind by one test is a model the next
test never installed -- failing in whatever way that fake's absence causes,
several files away from the cache.  ``tests/conftest.py`` empties it between
tests, which gives each the scope a real invocation has.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

T = TypeVar("T")

#: What is currently held, and what it was built for.
_KEY: str | None = None
_MODEL: Any = None


def key_for(stage: str, config: Mapping[str, Any]) -> str:
    """A stable identity for "this stage's model, built this way".

    ``json.dumps(..., sort_keys=True)`` rather than ``repr``: two mappings that
    differ only in the order their keys were written are the same config, and
    rebuilding a 16 GB model because a dict literal was reordered would be a
    quiet way to lose everything this module is for.
    """

    return f"{stage}|{json.dumps(config, sort_keys=True, default=str)}"


def release() -> None:
    """Drop what is held and give its memory back.

    ``del`` alone is not enough for a GPU model.  The tensors go when the last
    reference does, but torch's caching allocator keeps the block, so the next
    model is built onto a card that still looks full -- and on a card where the
    next model is 16 GB, that is the difference between working and not.
    """

    global _KEY, _MODEL
    held = _MODEL is not None
    _MODEL = None
    _KEY = None

    # Nothing was held, so there is nothing to give back.  A full collection is
    # not free, and this is the first call of every stage in a stage-major run
    # -- the case where the cache is legitimately empty.  Paying for it there
    # buys nothing.
    if not held:
        return

    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - the CPU-only stages
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_for(
    stage: str,
    config: Mapping[str, Any],
    build: Callable[[Mapping[str, Any]], T],
) -> T:
    """``build(config)``, or the one already built for this stage and config.

    ``build`` keeps its exception behaviour: a build that fails raises here too,
    having released what was held, so the next call tries again rather than
    returning something half-constructed.  Callers that record a failed stage
    around their build still do.
    """

    global _KEY, _MODEL
    key = key_for(stage, config)
    if key == _KEY and _MODEL is not None:
        return cast(T, _MODEL)

    release()
    _MODEL = build(config)
    _KEY = key
    return cast(T, _MODEL)
