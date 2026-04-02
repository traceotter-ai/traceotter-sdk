from __future__ import annotations

import json
from typing import Any


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return json.dumps({"unserializable": str(value)}, ensure_ascii=True)


def serialize_message(message: Any) -> dict[str, Any]:
    return {
        "role": getattr(message, "type", None) or getattr(message, "role", None),
        "content": getattr(message, "content", None),
        "name": getattr(message, "name", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def serialize_messages_batch(messages_batch: Any) -> list[list[dict[str, Any]]]:
    serialized: list[list[dict[str, Any]]] = []
    for group in messages_batch or []:
        serialized.append([serialize_message(message) for message in group])
    return serialized


def serialize_document(document: Any) -> dict[str, Any]:
    return {
        "page_content": getattr(document, "page_content", None),
        "metadata": getattr(document, "metadata", {}) or {},
    }


def serialize_documents(documents: Any) -> list[dict[str, Any]]:
    return [serialize_document(document) for document in documents or []]


def parse_model_name_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    model = metadata.get("ls_model_name")
    return model if isinstance(model, str) else None


def parse_model_parameters(invocation_params: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(invocation_params, dict):
        return {}

    params = dict(invocation_params)
    if params.get("_type") == "IBM watsonx.ai" and isinstance(
        params.get("params"), dict
    ):
        merged = dict(params)
        merged.update(params["params"])
        merged.pop("params", None)
        params = merged

    keys = [
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "request_timeout",
        "decoding_method",
        "min_new_tokens",
        "max_new_tokens",
        "stop_sequences",
    ]
    return {key: params.get(key) for key in keys if params.get(key) is not None}


def _parse_usage_model(usage: Any) -> dict[str, int] | None:
    if hasattr(usage, "__dict__"):
        usage = usage.__dict__
    if not isinstance(usage, dict):
        return None

    usage_model: dict[str, Any] = usage.copy()
    conversion_list = [
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("total_tokens", "total"),
        ("prompt_tokens", "input"),
        ("completion_tokens", "output"),
        ("prompt_token_count", "input"),
        ("candidates_token_count", "output"),
        ("total_token_count", "total"),
        ("inputTokenCount", "input"),
        ("outputTokenCount", "output"),
        ("totalTokenCount", "total"),
        ("input_token_count", "input"),
        ("generated_token_count", "output"),
    ]

    for source_key, target_key in conversion_list:
        if source_key in usage_model:
            value = usage_model.pop(source_key)
            if isinstance(value, list):
                value = sum(item for item in value if isinstance(item, int))
            usage_model[target_key] = value

    if isinstance(usage_model.get("input_token_details"), dict):
        details = usage_model.pop("input_token_details")
        for key, value in details.items():
            usage_model[f"input_{key}"] = value
            if isinstance(usage_model.get("input"), int) and isinstance(value, int):
                usage_model["input"] = max(0, usage_model["input"] - value)

    if isinstance(usage_model.get("output_token_details"), dict):
        details = usage_model.pop("output_token_details")
        for key, value in details.items():
            usage_model[f"output_{key}"] = value
            if isinstance(usage_model.get("output"), int) and isinstance(value, int):
                usage_model["output"] = max(0, usage_model["output"] - value)

    for input_key, prefix in [
        ("prompt_tokens_details", "input_modality_"),
        ("candidates_tokens_details", "output_modality_"),
        ("cache_tokens_details", "cached_modality_"),
    ]:
        details = usage_model.get(input_key)
        if isinstance(details, list):
            usage_model.pop(input_key, None)
            for item in details:
                if (
                    isinstance(item, dict)
                    and "modality" in item
                    and "token_count" in item
                ):
                    usage_model[f"{prefix}{item['modality']}"] = item["token_count"]

    usage_ints = {k: v for k, v in usage_model.items() if isinstance(v, int)}
    return usage_ints or None


def parse_usage(response: Any) -> dict[str, int] | None:
    llm_usage: dict[str, int] | None = None
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for key in ("token_usage", "usage"):
            if llm_output.get(key):
                llm_usage = _parse_usage_model(llm_output[key])
                if llm_usage:
                    break

    generations = getattr(response, "generations", None)
    if isinstance(generations, list):
        for generation in generations:
            if not isinstance(generation, list):
                continue
            for chunk in generation:
                generation_info = getattr(chunk, "generation_info", None)
                if isinstance(generation_info, dict) and generation_info.get(
                    "usage_metadata"
                ):
                    parsed = _parse_usage_model(generation_info["usage_metadata"])
                    if parsed:
                        llm_usage = parsed
                        break

                message = getattr(chunk, "message", None)
                response_metadata = getattr(message, "response_metadata", None)
                chunk_usage = None
                if isinstance(response_metadata, dict):
                    chunk_usage = response_metadata.get(
                        "usage"
                    ) or response_metadata.get("amazon-bedrock-invocationMetrics")
                chunk_usage = chunk_usage or getattr(message, "usage_metadata", None)
                if chunk_usage:
                    parsed = _parse_usage_model(chunk_usage)
                    if parsed:
                        llm_usage = parsed
                        break
            if llm_usage:
                break

    return llm_usage


def parse_model_from_response(response: Any) -> str | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        model_name = llm_output.get("model_name")
        if isinstance(model_name, str):
            return model_name
    return None


def to_gen_ai_usage_attributes(usage: dict[str, int] | None) -> dict[str, int]:
    if not usage:
        return {}
    attrs: dict[str, int] = {}
    if isinstance(usage.get("input"), int):
        attrs["gen_ai.usage.input_tokens"] = usage["input"]
    if isinstance(usage.get("output"), int):
        attrs["gen_ai.usage.output_tokens"] = usage["output"]
    if isinstance(usage.get("total"), int):
        attrs["gen_ai.usage.total_tokens"] = usage["total"]
    return attrs
