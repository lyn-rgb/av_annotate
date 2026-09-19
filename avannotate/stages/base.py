"""Stage plumbing: contexts, artifacts, and the resume record.

A batch of 100-1000 hours will crash partway through, so every stage writes its
outputs and a record of what it wrote, and a rerun skips a stage only when that
record still describes the present inputs *and* the artifacts are still on disk
with the hashes the record claims.

The hash check is what makes this a resume mechanism rather than a hope.  A
truncated WAV from a machine that died mid-write, or a hand-edited
``shots.json``, otherwise looks exactly like a completed stage.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avannotate import __version__

STATE_FILENAME = "stage_state.json"

#: Key the CLI injects so a config can name a model or data file relative to
#: itself.  Without it a config with ``"model_path": "models/yunet.onnx"``
#: silently resolves against the working directory, and a batch launched from a
#: different directory than the one it was tested in fails to find its weights.
CONFIG_ROOT_KEY = "config_root"


class StageError(RuntimeError):
    """A stage could not complete."""


def config_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer from a config mapping, tolerating JSON's numbers.

    A config arrives as parsed JSON, so "3" and 3.0 are both plausible writings
    of the same intent.  Anything else is a mistake worth raising on rather than
    silently defaulting past.
    """

    value = mapping.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise StageError(f"config {key!r} must be a number, got a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(float(value))
        except ValueError as error:
            raise StageError(f"config {key!r} is not a number: {value!r}") from error
    raise StageError(f"config {key!r} must be a number, got {type(value).__name__}")


def config_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = mapping.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise StageError(f"config {key!r} must be a number, got a boolean")
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as error:
            raise StageError(f"config {key!r} is not a number: {value!r}") from error
    raise StageError(f"config {key!r} must be a number, got {type(value).__name__}")


def config_str(mapping: Mapping[str, Any], key: str, default: str) -> str:
    value = mapping.get(key)
    return default if value is None else str(value)


def config_optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return None if value is None else str(value)


def resolve_config_path(value: str | Path, config: Mapping[str, Any]) -> Path:
    """Resolve a path from a config against the config file's directory.

    Normalised, so two configs naming the same file the same way hash equal and
    a config moved to another directory does not silently point at a different
    model while keeping its cache key.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        root = config.get(CONFIG_ROOT_KEY)
        path = (Path(str(root)) / path) if root is not None else path
    return path.resolve()


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Stable hash of a JSON-shaped mapping.

    ``sort_keys`` matters: a stage's config hash must not change because a dict
    was rebuilt in a different insertion order, or every rerun would look like a
    config change and nothing would ever be skipped.
    """

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON atomically, so a crash cannot leave a half-written artifact
    that the resume check would then have to reject."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Artifact:
    """One file a stage produced, named relative to the video's work directory."""

    path: str
    sha256: str
    size: int

    @classmethod
    def capture(cls, root: Path, path: Path) -> Artifact:
        return cls(
            path=str(path.relative_to(root)),
            sha256=hash_file(path),
            size=path.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Artifact:
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size=int(payload["size"]),
        )


@dataclass(frozen=True)
class StageRecord:
    """What a stage did, and the inputs it did it to."""

    stage: str
    status: str  # "ok" | "failed"
    code_version: str
    config_hash: str
    input_hash: str
    artifacts: tuple[Artifact, ...] = ()
    error: str | None = None
    finished_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "error": self.error,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageRecord:
        return cls(
            stage=str(payload["stage"]),
            status=str(payload["status"]),
            code_version=str(payload.get("code_version", "")),
            config_hash=str(payload.get("config_hash", "")),
            input_hash=str(payload.get("input_hash", "")),
            artifacts=tuple(
                Artifact.from_dict(item) for item in (payload.get("artifacts") or [])
            ),
            error=payload.get("error"),
            finished_at=str(payload.get("finished_at", "")),
        )


@dataclass(frozen=True)
class StageRun:
    """What a stage invocation did, for the driver's log and the batch manifest."""

    stage: str
    skipped: bool
    reason: str
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageContext:
    """Everything a stage needs: where the source is, where to write, and with what."""

    video_id: str
    source: Path
    work_dir: Path
    config: Mapping[str, Any] = field(default_factory=dict)

    def stage_dir(self, stage: str) -> Path:
        directory = self.work_dir / stage
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def output(self, stage: str, name: str) -> Path:
        return self.stage_dir(stage) / name


class StageState:
    """The per-video resume record, read and written as one JSON file."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.path = work_dir / STATE_FILENAME
        self._records: dict[str, StageRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = read_json(self.path)
        except (json.JSONDecodeError, OSError):
            # A corrupt state file means "start over", not "crash": the artifacts
            # are still on disk, so the worst case is redoing one video's work.
            return
        for item in payload.get("stages") or []:
            record = StageRecord.from_dict(item)
            self._records[record.stage] = record

    @property
    def records(self) -> Mapping[str, StageRecord]:
        return dict(self._records)

    def record_for(self, stage: str) -> StageRecord | None:
        return self._records.get(stage)

    def reason_to_run(self, stage: str, *, code_version: str, input_hash: str) -> str | None:
        """``None`` when the stage can be skipped, otherwise why it cannot."""

        record = self._records.get(stage)
        if record is None:
            return "no previous run"
        if record.status != "ok":
            return f"previous run {record.status}"
        if record.code_version != code_version:
            return f"code changed ({record.code_version} -> {code_version})"
        if record.input_hash != input_hash:
            return "inputs changed"
        for artifact in record.artifacts:
            path = self.work_dir / artifact.path
            if not path.is_file():
                return f"missing artifact {artifact.path}"
            if path.stat().st_size != artifact.size:
                return f"artifact {artifact.path} changed size"
            if hash_file(path) != artifact.sha256:
                return f"artifact {artifact.path} changed content"
        return None

    def save(self, record: StageRecord) -> None:
        self._records[record.stage] = record
        write_json(
            self.path,
            {
                "schema_version": "avannotate-stage-state-v1",
                "package_version": __version__,
                "stages": [item.to_dict() for item in self._records.values()],
            },
        )
