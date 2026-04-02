from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class SchemaValidationError(ValueError):
    pass


class RawSpan(TypedDict, total=False):
    trace_id: str
    id: str
    span_id: str
    start_time: float | str
    attributes: dict[str, Any]
    context: dict[str, Any]


def _extract_trace_id(span: dict[str, Any]) -> str | None:
    if isinstance(span.get("trace_id"), str) and span["trace_id"]:
        return span["trace_id"]
    context = span.get("context")
    if isinstance(context, dict) and isinstance(context.get("trace_id"), str):
        return context["trace_id"]
    attrs = span.get("attributes")
    if isinstance(attrs, dict) and isinstance(attrs.get("trace_id"), str):
        return attrs["trace_id"]
    return None


def _extract_span_id(span: dict[str, Any]) -> str | None:
    if isinstance(span.get("span_id"), str) and span["span_id"]:
        return span["span_id"]
    if isinstance(span.get("id"), str) and span["id"]:
        return span["id"]
    context = span.get("context")
    if isinstance(context, dict) and isinstance(context.get("span_id"), str):
        return context["span_id"]
    attrs = span.get("attributes")
    if isinstance(attrs, dict) and isinstance(attrs.get("span_id"), str):
        return attrs["span_id"]
    return None


def _normalize_start_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.timestamp()
            except Exception as exc:  # noqa: BLE001
                raise SchemaValidationError("Invalid start_time format") from exc
    raise SchemaValidationError(
        "start_time is required and must be numeric or ISO-8601"
    )


def validate_span_schema(span: dict[str, Any]) -> RawSpan:
    if not isinstance(span, dict):
        raise SchemaValidationError("Span must be a dictionary")

    trace_id = _extract_trace_id(span)
    if not trace_id:
        raise SchemaValidationError("Missing trace_id")

    span_id = _extract_span_id(span)
    if not span_id:
        raise SchemaValidationError("Missing span_id")

    start_time = span.get("start_time", span.get("startTime"))
    normalized_start = _normalize_start_time(start_time)

    attributes = span.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise SchemaValidationError("attributes must be a dictionary")

    context = span.get("context") or {}
    if not isinstance(context, dict):
        raise SchemaValidationError("context must be a dictionary")

    return RawSpan(
        trace_id=trace_id,
        id=span.get("id") or span_id,
        span_id=span_id,
        start_time=normalized_start,
        attributes=attributes,
        context=context,
    )
