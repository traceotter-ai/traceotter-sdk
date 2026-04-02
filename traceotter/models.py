from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SpanKind = Literal["INTERNAL", "CLIENT"]
SpanStatus = Literal["UNSET", "OK", "ERROR"]


@dataclass
class OTelEvent:
    name: str
    timestamp_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class OTelSpanPayload:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    start_time_unix_nano: int
    end_time_unix_nano: int | None = None
    status_code: SpanStatus = "UNSET"
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[OTelEvent] = field(default_factory=list)
