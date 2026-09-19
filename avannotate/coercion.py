"""Reading numbers back out of JSON without losing the type.

Every stage deserializes something, and a bare ``float(payload["x"])`` both
loses the type for a checker and says nothing about what happens when the value
is a string, a boolean, or a nested object.  One helper, in a module that
depends on nothing, so the readers can share it without one stage importing
another.
"""

from __future__ import annotations


def coerce_number(value: object, field: str) -> float:
    """Coerce a JSON scalar to a float.

    Booleans are rejected rather than accepted as 0 and 1: a ``true`` where a
    timestamp belongs means the producer wrote the wrong thing, and silently
    reading it as 1.0 hides that.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be a number, got {type(value).__name__}")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{field} is not a number: {value!r}") from error


def coerce_int(value: object, field: str) -> int:
    return int(coerce_number(value, field))
