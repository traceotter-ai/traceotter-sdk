from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from traceotter.serializers import serialize_dict


class TraceotterOtelSpanAttributes:
    # Trace-level attributes
    TRACE_NAME = "traceotter.trace.name"
    TRACE_USER_ID = "user.id"
    TRACE_SESSION_ID = "session.id"
    TRACE_TAGS = "traceotter.trace.tags"
    TRACE_PUBLIC = "traceotter.trace.public"
    TRACE_METADATA = "traceotter.trace.metadata"
    TRACE_INPUT = "traceotter.trace.input"
    TRACE_OUTPUT = "traceotter.trace.output"

    # Observation-level attributes
    OBSERVATION_TYPE = "traceotter.observation.type"
    OBSERVATION_METADATA = "traceotter.observation.metadata"
    OBSERVATION_LEVEL = "traceotter.observation.level"
    OBSERVATION_STATUS_MESSAGE = "traceotter.observation.status_message"
    OBSERVATION_INPUT = "traceotter.observation.input"
    OBSERVATION_OUTPUT = "traceotter.observation.output"

    # Generation observation attributes
    OBSERVATION_COMPLETION_START_TIME = "traceotter.observation.completion_start_time"
    OBSERVATION_MODEL = "traceotter.observation.model.name"
    OBSERVATION_MODEL_PARAMETERS = "traceotter.observation.model.parameters"
    OBSERVATION_USAGE_DETAILS = "traceotter.observation.usage_details"
    OBSERVATION_COST_DETAILS = "traceotter.observation.cost_details"
    OBSERVATION_PROMPT_NAME = "traceotter.observation.prompt.name"
    OBSERVATION_PROMPT_VERSION = "traceotter.observation.prompt.version"

    # General
    ENVIRONMENT = "traceotter.environment"
    RELEASE = "traceotter.release"
    VERSION = "traceotter.version"

    # Internal
    AS_ROOT = "traceotter.internal.as_root"


def _serialize(obj: Any) -> str | None:
    if obj is None or isinstance(obj, str):
        return obj
    return json.dumps(obj, default=str, ensure_ascii=True)


def _flatten_and_serialize_metadata(
    metadata: Any,
    kind: str,
) -> dict[str, Any]:
    prefix = (
        TraceotterOtelSpanAttributes.OBSERVATION_METADATA
        if kind == "observation"
        else TraceotterOtelSpanAttributes.TRACE_METADATA
    )
    attributes: dict[str, Any] = {}
    if metadata is None:
        return attributes

    if not isinstance(metadata, dict):
        attributes[prefix] = _serialize(metadata)
        return attributes

    for key, value in metadata.items():
        attributes[f"{prefix}.{key}"] = (
            value if isinstance(value, (str, int, float, bool)) else _serialize(value)
        )
    return attributes


def create_trace_attributes(
    *,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    version: str | None = None,
    release: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    tags: list[str] | None = None,
    public: bool | None = None,
) -> dict[str, Any]:
    attributes = {
        TraceotterOtelSpanAttributes.TRACE_NAME: name,
        TraceotterOtelSpanAttributes.TRACE_USER_ID: user_id,
        TraceotterOtelSpanAttributes.TRACE_SESSION_ID: session_id,
        TraceotterOtelSpanAttributes.VERSION: version,
        TraceotterOtelSpanAttributes.RELEASE: release,
        TraceotterOtelSpanAttributes.TRACE_INPUT: _serialize(input),
        TraceotterOtelSpanAttributes.TRACE_OUTPUT: (
            None if output is None else serialize_dict(output)
        ),
        TraceotterOtelSpanAttributes.TRACE_TAGS: tags,
        TraceotterOtelSpanAttributes.TRACE_PUBLIC: public,
        **_flatten_and_serialize_metadata(metadata, "trace"),
    }
    return {key: value for key, value in attributes.items() if value is not None}


def create_span_attributes(
    *,
    metadata: Any = None,
    input: Any = None,
    output: Any = None,
    level: str | None = None,
    status_message: str | None = None,
    version: str | None = None,
    observation_type: str = "span",
) -> dict[str, Any]:
    attributes = {
        TraceotterOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        TraceotterOtelSpanAttributes.OBSERVATION_LEVEL: level,
        TraceotterOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        TraceotterOtelSpanAttributes.VERSION: version,
        TraceotterOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        TraceotterOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }
    return {key: value for key, value in attributes.items() if value is not None}


def create_generation_attributes(
    *,
    completion_start_time: datetime | None = None,
    metadata: Any = None,
    level: str | None = None,
    status_message: str | None = None,
    version: str | None = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    input: Any = None,
    output: Any = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    observation_type: str = "generation",
) -> dict[str, Any]:
    attributes = {
        TraceotterOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        TraceotterOtelSpanAttributes.OBSERVATION_LEVEL: level,
        TraceotterOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        TraceotterOtelSpanAttributes.VERSION: version,
        TraceotterOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        TraceotterOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        TraceotterOtelSpanAttributes.OBSERVATION_MODEL: model,
        TraceotterOtelSpanAttributes.OBSERVATION_PROMPT_NAME: prompt_name,
        TraceotterOtelSpanAttributes.OBSERVATION_PROMPT_VERSION: prompt_version,
        TraceotterOtelSpanAttributes.OBSERVATION_USAGE_DETAILS: _serialize(
            usage_details
        ),
        TraceotterOtelSpanAttributes.OBSERVATION_COST_DETAILS: _serialize(cost_details),
        TraceotterOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME: _serialize(
            completion_start_time
        ),
        TraceotterOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS: _serialize(
            model_parameters
        ),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }
    return {key: value for key, value in attributes.items() if value is not None}


def parse_trace_attributes_from_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return attributes

    if isinstance(metadata.get("traceotter_user_id"), str):
        attributes[TraceotterOtelSpanAttributes.TRACE_USER_ID] = metadata[
            "traceotter_user_id"
        ]

    if isinstance(metadata.get("traceotter_session_id"), str):
        attributes[TraceotterOtelSpanAttributes.TRACE_SESSION_ID] = metadata[
            "traceotter_session_id"
        ]

    raw_tags = metadata.get("traceotter_tags")
    if isinstance(raw_tags, list):
        attributes[TraceotterOtelSpanAttributes.TRACE_TAGS] = [
            str(tag) for tag in raw_tags
        ]

    return attributes


def strip_traceotter_keys_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    keep_traceotter_trace_attributes: bool = False,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    metadata_copy = metadata.copy()
    traceotter_metadata_keys = ["traceotter_prompt"]
    traceotter_trace_attribute_keys = [
        "traceotter_session_id",
        "traceotter_user_id",
        "traceotter_tags",
    ]

    for key in traceotter_metadata_keys:
        metadata_copy.pop(key, None)

    if not keep_traceotter_trace_attributes:
        for key in traceotter_trace_attribute_keys:
            metadata_copy.pop(key, None)

    return metadata_copy


def join_tags_and_metadata(
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    keep_traceotter_trace_attributes: bool = False,
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    if tags:
        combined["tags"] = tags
    if isinstance(metadata, dict):
        combined.update(metadata)

    return strip_traceotter_keys_from_metadata(
        combined,
        keep_traceotter_trace_attributes=keep_traceotter_trace_attributes,
    )
