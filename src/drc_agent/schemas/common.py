from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    def encode(item: Any):
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json", exclude_none=False)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, set):
            return sorted(item)
        raise TypeError(f"cannot canonicalize {type(item).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=encode)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Point(StrictModel):
    x: int
    y: int


class Vector(StrictModel):
    dx: int
    dy: int


class Edge(StrictModel):
    start: Point
    end: Point


class Box(StrictModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def ordered(self) -> "Box":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box coordinates must be ordered")
        return self

    @classmethod
    def from_sequence(cls, values: Iterable[int]) -> "Box":
        x1, y1, x2, y2 = [int(v) for v in values]
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def expand(self, amount: int) -> "Box":
        return Box(x1=self.x1 - amount, y1=self.y1 - amount,
                   x2=self.x2 + amount, y2=self.y2 + amount)

    def intersects(self, other: "Box") -> bool:
        return not (self.x2 < other.x1 or other.x2 < self.x1 or
                    self.y2 < other.y1 or other.y2 < self.y1)

    def intersection_area(self, other: "Box") -> int:
        width = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return width * height

    def gap(self, other: "Box") -> float:
        dx = max(self.x1 - other.x2, other.x1 - self.x2, 0)
        dy = max(self.y1 - other.y2, other.y1 - self.y2, 0)
        return (dx * dx + dy * dy) ** 0.5

    def union(self, other: "Box") -> "Box":
        return Box(x1=min(self.x1, other.x1), y1=min(self.y1, other.y1),
                   x2=max(self.x2, other.x2), y2=max(self.y2, other.y2))


Polygon = list[Point]


def canonical_polygon(points: Iterable[Point]) -> list[Point]:
    raw = [(p.x, p.y) for p in points]
    if len(raw) > 1 and raw[0] == raw[-1]:
        raw.pop()
    if len(raw) < 3:
        return [Point(x=x, y=y) for x, y in raw]
    area2 = sum(raw[i][0] * raw[(i + 1) % len(raw)][1] -
                raw[(i + 1) % len(raw)][0] * raw[i][1]
                for i in range(len(raw)))
    if area2 < 0:
        raw.reverse()
    variants = [raw[i:] + raw[:i] for i in range(len(raw))]
    result = min(variants)
    return [Point(x=x, y=y) for x, y in result]


class SourceSpan(StrictModel):
    path: str
    start_line: int
    start_column: int = 0
    end_line: int
    end_column: int = 0
    source_hash: str


class ArtifactRef(StrictModel):
    artifact_id: str
    path: str
    sha256: str
    media_type: str
    schema_name: str | None = None
    schema_version: str | None = None
    producer: str
    size_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return value.lower()

    @classmethod
    def from_path(cls, path: Path, *, producer: str, media_type: str,
                  artifact_id: str | None = None, schema_name: str | None = None,
                  schema_version: str | None = None) -> "ArtifactRef":
        resolved = path.resolve(strict=True)
        return cls(
            artifact_id=artifact_id or stable_hash(str(resolved))[:24],
            path=str(resolved), sha256=file_sha256(resolved),
            media_type=media_type, schema_name=schema_name,
            schema_version=schema_version, producer=producer,
            size_bytes=resolved.stat().st_size,
        )

    def verify(self, allowed_roots: Iterable[Path]) -> Path:
        path = Path(self.path)
        if any(part == ".." for part in path.parts):
            raise ValueError("artifact path traversal is forbidden")
        resolved = path.resolve(strict=True)
        # Optional artifact namespaces (for example ``results/``) need not
        # exist in every checkout.  A non-existent root cannot contain the
        # already strict-resolved artifact, so omit it without weakening the
        # containment or hash checks for roots that do exist.
        roots = [
            root.resolve(strict=True)
            for root in allowed_roots
            if root.exists()
        ]
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise ValueError(f"artifact path is outside allowed roots: {resolved}")
        if file_sha256(resolved) != self.sha256:
            raise ValueError(f"artifact hash mismatch: {resolved}")
        return resolved


class Predicate(StrictModel):
    predicate_id: str
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Provenance(StrictModel):
    source_type: str
    source_id: str
    source_hash: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EditabilityClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
