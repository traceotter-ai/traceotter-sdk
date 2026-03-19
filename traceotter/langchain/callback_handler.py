from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler

from traceotter.attributes import (
    TraceotterOtelSpanAttributes,
    create_generation_attributes,
    create_span_attributes,
    create_trace_attributes,
    join_tags_and_metadata,
    parse_trace_attributes_from_metadata,
)
from traceotter.client import TraceotterClient, get_client, now_ns
from traceotter.models import OTelEvent, OTelSpanPayload
from traceotter.serializers import (
    parse_model_from_response,
    parse_model_name_from_metadata,
    parse_model_parameters,
    parse_usage,
    safe_json_dumps,
    serialize_documents,
    serialize_messages_batch,
    to_gen_ai_usage_attributes,
)


@dataclass
class ActiveRun:
    run_id: str
    parent_run_id: str | None
    payload: OTelSpanPayload


LANGSMITH_TAG_HIDDEN = "langsmith:hidden"
CONTROL_FLOW_EXCEPTION_TYPES: set[type[BaseException]] = set()

try:
    from langgraph.errors import GraphBubbleUp

    CONTROL_FLOW_EXCEPTION_TYPES.add(GraphBubbleUp)
except Exception:
    pass

_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("traceotter_current_run_id", default=None)


class CallbackHandler(BaseCallbackHandler):
    def __init__(
        self, *, client: TraceotterClient | None = None, update_trace: bool = False
    ) -> None:
        self.client = client or get_client()
        self._runs: dict[str, ActiveRun] = {}
        self._context_tokens: dict[str, Token] = {}
        self._prompt_to_parent_run_map: dict[str, Any] = {}
        self._updated_completion_start_time_memo: set[str] = set()
        self.update_trace = update_trace

    @staticmethod
    def _new_trace_id() -> str:
        return uuid4().hex

    @staticmethod
    def _new_span_id() -> str:
        return uuid4().hex[:16]

    @staticmethod
    def _string_id(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _get_observation_type_from_serialized(
        self, serialized: dict[str, Any] | None, callback_type: str, **kwargs: Any
    ) -> Literal["tool", "retriever", "generation", "agent", "chain", "span"]:
        if callback_type == "tool":
            return "tool"
        if callback_type == "retriever":
            return "retriever"
        if callback_type == "llm":
            return "generation"
        if callback_type == "chain":
            if serialized and "id" in serialized:
                class_path = serialized.get("id", [])
                if isinstance(class_path, list) and any(
                    "agent" in str(part).lower() for part in class_path
                ):
                    return "agent"
            name = self.get_langchain_run_name(serialized, **kwargs)
            return "agent" if "agent" in name.lower() else "chain"
        return "span"

    def get_langchain_run_name(
        self, serialized: dict[str, Any] | None, **kwargs: Any
    ) -> str:
        if kwargs.get("name") is not None:
            return str(kwargs["name"])
        if serialized is None:
            return "<unknown>"
        if isinstance(serialized.get("name"), str):
            return serialized["name"]
        serialized_id = serialized.get("id")
        if isinstance(serialized_id, list) and serialized_id:
            return str(serialized_id[-1])
        return "<unknown>"

    def _attach_observation(self, run_id_str: str) -> None:
        token = _CURRENT_RUN_ID.set(run_id_str)
        self._context_tokens[run_id_str] = token

    def _detach_observation(self, run_id_str: str) -> None:
        token = self._context_tokens.pop(run_id_str, None)
        if token is not None:
            try:
                _CURRENT_RUN_ID.reset(token)
            except Exception:
                pass

    def _register_traceotter_prompt(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        metadata: dict[str, Any] | None,
    ) -> None:
        run_id_str = self._string_id(run_id)
        parent_run_id_str = self._string_id(parent_run_id)
        if run_id_str is None or parent_run_id_str is None:
            return

        traceotter_prompt = metadata.get("traceotter_prompt") if isinstance(metadata, dict) else None
        if traceotter_prompt:
            self._prompt_to_parent_run_map[parent_run_id_str] = traceotter_prompt
        elif parent_run_id_str in self._prompt_to_parent_run_map:
            self._prompt_to_parent_run_map[run_id_str] = self._prompt_to_parent_run_map[
                parent_run_id_str
            ]

    def _deregister_traceotter_prompt(self, run_id: Any) -> None:
        run_id_str = self._string_id(run_id)
        if run_id_str:
            self._prompt_to_parent_run_map.pop(run_id_str, None)

    def _start_run(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
        input_payload: Any = None,
    ) -> None:
        run_id_str = self._string_id(run_id)
        if run_id_str is None:
            return

        parent_run_id_str = self._string_id(parent_run_id)
        parent = self._runs.get(parent_run_id_str) if parent_run_id_str else None
        trace_id = parent.payload.trace_id if parent else self._new_trace_id()
        parent_span_id = parent.payload.span_id if parent else None

        payload = OTelSpanPayload(
            trace_id=trace_id,
            span_id=self._new_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            start_time_unix_nano=now_ns(),
            attributes=attributes or {},
        )

        payload.attributes["traceotter.schema.version"] = 1
        payload.attributes["traceotter.langchain.run_id"] = run_id_str
        if parent_run_id_str:
            payload.attributes["traceotter.langchain.parent_run_id"] = parent_run_id_str

        if input_payload is not None:
            payload.events.append(
                OTelEvent(
                    name="gen_ai.client.inference.operation.details",
                    timestamp_unix_nano=now_ns(),
                    attributes={"traceotter.input": safe_json_dumps(input_payload)},
                )
            )

        self._runs[run_id_str] = ActiveRun(
            run_id=run_id_str,
            parent_run_id=parent_run_id_str,
            payload=payload,
        )
        self._attach_observation(run_id_str)

    def _end_run(self, *, run_id: Any, output_payload: Any = None) -> None:
        run_id_str = self._string_id(run_id) or ""
        run = self._runs.pop(run_id_str, None)
        if run is None:
            return

        run.payload.end_time_unix_nano = now_ns()
        run.payload.status_code = "OK"
        if output_payload is not None:
            run.payload.events.append(
                OTelEvent(
                    name="traceotter.output",
                    timestamp_unix_nano=now_ns(),
                    attributes={"traceotter.output": safe_json_dumps(output_payload)},
                )
            )
        self.client.enqueue_span(run.payload)
        self._detach_observation(run_id_str)

    def _error_run(self, *, run_id: Any, error: BaseException) -> None:
        run_id_str = self._string_id(run_id) or ""
        run = self._runs.pop(run_id_str, None)
        if run is None:
            return

        run.payload.end_time_unix_nano = now_ns()
        run.payload.status_code = "ERROR"
        is_control_flow = any(isinstance(error, t) for t in CONTROL_FLOW_EXCEPTION_TYPES)
        run.payload.status_message = str(error) if not is_control_flow else None
        run.payload.attributes["error.type"] = type(error).__name__
        run.payload.attributes[TraceotterOtelSpanAttributes.OBSERVATION_LEVEL] = (
            "ERROR" if not is_control_flow else "DEFAULT"
        )
        if not is_control_flow:
            run.payload.attributes[
                TraceotterOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE
            ] = str(error)
        run.payload.attributes[TraceotterOtelSpanAttributes.OBSERVATION_COST_DETAILS] = (
            safe_json_dumps({"total": 0})
        )
        run.payload.events.append(
            OTelEvent(
                name="exception",
                timestamp_unix_nano=now_ns(),
                attributes={
                    "exception.type": type(error).__name__,
                    "exception.message": str(error),
                },
            )
        )
        self.client.enqueue_span(run.payload)
        self._detach_observation(run_id_str)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = self.get_langchain_run_name(serialized, **kwargs)
        self._register_traceotter_prompt(
            run_id=run_id, parent_run_id=parent_run_id, metadata=metadata
        )
        span_metadata = join_tags_and_metadata(tags=tags, metadata=metadata)
        observation_type = self._get_observation_type_from_serialized(
            serialized, "chain", **kwargs
        )
        attributes = create_span_attributes(
            metadata=span_metadata,
            input=inputs,
            observation_type=observation_type,
            level="DEBUG" if tags and LANGSMITH_TAG_HIDDEN in tags else None,
        )
        attributes.update(
            {
                "traceotter.span.type": observation_type,
            }
        )
        if parent_run_id is None:
            attributes.update(parse_trace_attributes_from_metadata(metadata))
            if self.update_trace:
                attributes.update(
                    create_trace_attributes(
                        name=name,
                        input=inputs,
                        metadata=span_metadata,
                    )
                )
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind="INTERNAL",
            attributes=attributes,
            input_payload=inputs,
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **kwargs: Any) -> None:
        if self.update_trace:
            run = self._runs.get(self._string_id(run_id) or "")
            if run:
                run.payload.attributes.update(create_trace_attributes(output=outputs))
        self._end_run(run_id=run_id, output_payload=outputs)
        self._deregister_traceotter_prompt(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._error_run(run_id=run_id, error=error)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        run_key = self._string_id(run_id) or ""
        run = self._runs.get(run_key)
        if run and run_key not in self._updated_completion_start_time_memo:
            run.payload.attributes[
                TraceotterOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME
            ] = safe_json_dumps(now_ns())
            self._updated_completion_start_time_memo.add(run_key)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.__on_llm_action(
            serialized=serialized,
            prompts={"messages": serialize_messages_batch(messages)},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            invocation_params=invocation_params,
            **kwargs,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.__on_llm_action(
            serialized=serialized,
            prompts=prompts[0] if len(prompts) == 1 else prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            invocation_params=invocation_params,
            **kwargs,
        )

    def __on_llm_action(
        self,
        *,
        serialized: dict[str, Any] | None,
        prompts: Any,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model = None
        if invocation_params:
            model = invocation_params.get("model") or invocation_params.get("model_name")
        if model is None:
            model = parse_model_name_from_metadata(metadata)
        span_metadata = join_tags_and_metadata(
            tags=tags,
            metadata=metadata,
            keep_traceotter_trace_attributes=True if parent_run_id is None else False,
        )
        name = (
            self.get_langchain_run_name(serialized or {}, **kwargs)
            if serialized
            else "generation"
        )
        model_parameters = parse_model_parameters(invocation_params)
        registered_prompt = None
        parent_run_id_str = self._string_id(parent_run_id)
        if parent_run_id_str:
            registered_prompt = self._prompt_to_parent_run_map.get(parent_run_id_str)
            if registered_prompt:
                self._deregister_traceotter_prompt(parent_run_id)
        attributes = create_generation_attributes(
            metadata=span_metadata,
            model=model if isinstance(model, str) else None,
            model_parameters=model_parameters,
            input=prompts,
            prompt_name=getattr(registered_prompt, "name", None),
            prompt_version=getattr(registered_prompt, "version", None),
            observation_type="generation",
        )
        attributes.update(
            {
                "traceotter.span.type": "generation",
                "gen_ai.operation.name": "chat",
            }
        )
        if isinstance(model, str):
            attributes["gen_ai.request.model"] = model
        if parent_run_id is None:
            attributes.update(parse_trace_attributes_from_metadata(metadata))

        payload = {
            "prompts": prompts,
            "invocation_params": invocation_params or {},
            "serialized": serialized,
        }
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind="CLIENT",
            attributes=attributes,
            input_payload=payload,
        )

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        run_key = self._string_id(run_id) or ""
        run = self._runs.get(run_key)
        if run:
            usage = parse_usage(response)
            run.payload.attributes.update(to_gen_ai_usage_attributes(usage))
            if usage:
                run.payload.attributes[
                    TraceotterOtelSpanAttributes.OBSERVATION_USAGE_DETAILS
                ] = safe_json_dumps(usage)
            model = parse_model_from_response(response)
            if model:
                run.payload.attributes["gen_ai.response.model"] = model
                run.payload.attributes[TraceotterOtelSpanAttributes.OBSERVATION_MODEL] = model

        self._end_run(run_id=run_id, output_payload=response)
        self._updated_completion_start_time_memo.discard(run_key)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._error_run(run_id=run_id, error=error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = self.get_langchain_run_name(serialized, **kwargs)
        span_metadata = join_tags_and_metadata(tags=tags, metadata=metadata)
        observation_type = self._get_observation_type_from_serialized(
            serialized, "tool", **kwargs
        )
        attributes = create_span_attributes(
            metadata=span_metadata,
            input=input_str,
            observation_type=observation_type,
            level="DEBUG" if tags and LANGSMITH_TAG_HIDDEN in tags else None,
        )
        attributes.update(
            {
                "traceotter.span.type": observation_type,
                "tool.name": name,
            }
        )
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind="INTERNAL",
            attributes=attributes,
            input_payload={"input": input_str},
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, output_payload=output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._error_run(run_id=run_id, error=error)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = self.get_langchain_run_name(serialized, **kwargs)
        span_metadata = join_tags_and_metadata(tags=tags, metadata=metadata)
        observation_type = self._get_observation_type_from_serialized(
            serialized, "retriever", **kwargs
        )
        attributes = create_span_attributes(
            metadata=span_metadata,
            input={"query": query},
            observation_type=observation_type,
            level="DEBUG" if tags and LANGSMITH_TAG_HIDDEN in tags else None,
        )
        attributes.update(
            {
                "traceotter.span.type": "retrieval",
            }
        )
        self._start_run(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            kind="INTERNAL",
            attributes=attributes,
            input_payload={"query": query},
        )

    def on_retriever_end(self, documents: list[Any], *, run_id: Any, **kwargs: Any) -> None:
        self._end_run(run_id=run_id, output_payload=serialize_documents(documents))

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._error_run(run_id=run_id, error=error)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.get(self._string_id(run_id) or "")
        if run:
            run.payload.attributes[TraceotterOtelSpanAttributes.OBSERVATION_TYPE] = "agent"
            run.payload.events.append(
                OTelEvent(
                    name="traceotter.agent.action",
                    timestamp_unix_nano=now_ns(),
                    attributes={"traceotter.output": safe_json_dumps(action)},
                )
            )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.get(self._string_id(run_id) or "")
        if run:
            run.payload.attributes[TraceotterOtelSpanAttributes.OBSERVATION_TYPE] = "agent"
            run.payload.events.append(
                OTelEvent(
                    name="traceotter.agent.finish",
                    timestamp_unix_nano=now_ns(),
                    attributes={"traceotter.output": safe_json_dumps(finish)},
                )
            )

