from __future__ import annotations

import json
import logging
import threading
import urllib.parse
from typing import Any

import httpx

from traceotter._utils.ingest_schema import RawSpan

log = logging.getLogger("traceotter")


class APIError(Exception):
    def __init__(self, status: int | str, message: Any):
        self.status = status
        self.message = message
        super().__init__(f"{status}: {message}")


class APIErrors(Exception):
    def __init__(self, errors: list[APIError]):
        self.errors = errors
        super().__init__(", ".join(str(err) for err in errors))


def grpc_target_from_base_url(base_url: str, port: int = 50051) -> str:
    parsed = urllib.parse.urlparse(
        base_url if "://" in base_url else f"https://{base_url}"
    )
    host = parsed.hostname or parsed.path or "localhost"
    return f"{host}:{port}"


def _span_id(raw: RawSpan) -> str:
    return str(raw.get("span_id") or raw.get("id") or "")


def _parent_span_id(raw: RawSpan) -> str | None:
    context = raw.get("context") or {}
    if isinstance(context, dict) and context.get("parent_span_id") is not None:
        return str(context.get("parent_span_id"))
    attrs = raw.get("attributes") or {}
    if isinstance(attrs, dict) and attrs.get("parent_span_id") is not None:
        return str(attrs.get("parent_span_id"))
    return None


def order_spans_parents_first(spans: list[RawSpan]) -> list[RawSpan]:
    if not spans:
        return []

    id_to_span: dict[str, RawSpan] = {}
    for span in spans:
        sid = _span_id(span)
        if sid and sid not in id_to_span:
            id_to_span[sid] = span
    ids = set(id_to_span.keys())
    if not ids:
        return spans

    children: dict[str, list[str]] = {}
    indegree: dict[str, int] = {sid: 0 for sid in ids}
    for sid, span in id_to_span.items():
        parent_id = _parent_span_id(span)
        if parent_id and parent_id in ids and parent_id != sid:
            children.setdefault(parent_id, []).append(sid)
            indegree[sid] += 1

    queue = [sid for sid in ids if indegree[sid] == 0]
    ordered_ids: list[str] = []
    while queue:
        node = queue.pop(0)
        ordered_ids.append(node)
        for child in children.get(node, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    for sid in ids:
        if sid not in ordered_ids:
            ordered_ids.append(sid)

    ordered = [id_to_span[sid] for sid in ordered_ids]
    seen = {id(span) for span in ordered}
    tail = [span for span in spans if id(span) not in seen]
    return ordered + tail


def build_ingest_payload(spans: list[RawSpan]) -> list[dict[str, Any]]:
    if not spans:
        return []
    ordered = order_spans_parents_first(spans)
    envelope: dict[str, Any] = {"spans": []}
    for span in ordered:
        wrapper: dict[str, Any] = {"details": span}
        name = span.get("attributes", {}).get("otel_span_name")
        if isinstance(name, str) and name:
            wrapper["name"] = name
        envelope["spans"].append(wrapper)
    return [envelope] if envelope["spans"] else []


class TraceotterHttpIngestClient:
    def __init__(
        self,
        *,
        secret_key: str,
        base_url: str,
        version: str,
        timeout_seconds: int = 5,
        grpc_target: str | None = None,
    ) -> None:
        self._secret_key = secret_key
        self._base_url = base_url
        self._version = version
        self._session = httpx.Client(timeout=timeout_seconds)
        self._timeout_seconds = timeout_seconds
        self._grpc_target = grpc_target
        self._grpc_channel: Any = None
        self._grpc_stub: Any = None
        self._grpc_lock = threading.Lock()

    def close(self) -> None:
        self._session.close()
        if self._grpc_channel is not None:
            self._grpc_channel.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._secret_key}",
            "x-api-key": self._secret_key,
            "Content-Type": "application/json",
            "x_traceotter_sdk_name": "python",
            "x_traceotter_sdk_version": self._version,
        }

    def _ingest_url(self) -> str:
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme:
            return self._base_url.rstrip("/") + "/v1/ingest"
        return "https://" + self._base_url.rstrip("/") + "/v1/ingest"

    def batch_post(self, spans: list[RawSpan]) -> None:
        if self._grpc_target:
            try:
                self._batch_post_grpc(spans)
                return
            except ImportError:
                # Gracefully degrade to HTTP if grpcio is not installed.
                log.warning("grpcio is not installed; falling back to HTTP ingest")
                self._grpc_target = None
        payload = build_ingest_payload(spans)
        if not payload:
            return
        url = self._ingest_url()
        resp = self._session.post(
            url,
            content=json.dumps(payload),
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        self._process_response(resp)

    def _get_grpc_stub(self) -> Any:
        if self._grpc_stub is not None:
            return self._grpc_stub
        with self._grpc_lock:
            if self._grpc_stub is not None:
                return self._grpc_stub
            try:
                import grpc
            except ImportError as exc:
                raise ImportError(
                    "gRPC ingest requires grpcio. Install with: pip install grpcio"
                ) from exc
            from traceotter.proto.v1 import traceotter_pb2_grpc

            self._grpc_channel = grpc.insecure_channel(self._grpc_target)
            self._grpc_stub = traceotter_pb2_grpc.TraceOtterServiceStub(
                self._grpc_channel
            )
        return self._grpc_stub

    @staticmethod
    def _struct_safe_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                k: TraceotterHttpIngestClient._struct_safe_value(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TraceotterHttpIngestClient._struct_safe_value(v) for v in value]
        return str(value)

    def _raw_span_to_grpc_span(self, raw: RawSpan) -> Any:
        from google.protobuf.json_format import ParseDict

        from traceotter.proto.v1 import traceotter_pb2

        span = traceotter_pb2.Span()
        span.trace_id = str(raw.get("trace_id", ""))
        span.span_id = str(raw.get("span_id", raw.get("id", "")))
        start_time = raw.get("start_time")
        if isinstance(start_time, (int, float)):
            span.start_time_unix = float(start_time)
        elif start_time is not None:
            span.start_time_iso = str(start_time)

        parent_id = (raw.get("context") or {}).get("parent_span_id") or (
            raw.get("attributes") or {}
        ).get("parent_span_id")
        if parent_id is not None:
            span.parent_id = str(parent_id)

        attrs = raw.get("attributes")
        if isinstance(attrs, dict) and attrs:
            ParseDict(self._struct_safe_value(attrs), span.attributes)
        return span

    def _grpc_metadata(self) -> list[tuple[str, str]]:
        metadata: list[tuple[str, str]] = [
            ("x_traceotter_sdk_name", "python"),
            ("x_traceotter_sdk_version", self._version),
            ("x-api-key", self._secret_key),
            ("authorization", f"Bearer {self._secret_key}"),
        ]
        return metadata

    def _batch_post_grpc(self, spans: list[RawSpan]) -> None:
        try:
            import grpc
        except ImportError as exc:
            raise ImportError(
                "gRPC ingest requires grpcio. Install with: pip install grpcio"
            ) from exc
        from traceotter.proto.v1 import traceotter_pb2

        if not spans:
            return

        request = traceotter_pb2.IngestSpansRequest()
        for raw in order_spans_parents_first(spans):
            request.spans.append(self._raw_span_to_grpc_span(raw))
        if not request.spans:
            return

        stub = self._get_grpc_stub()
        try:
            stub.IngestSpans(
                request,
                metadata=self._grpc_metadata(),
                timeout=self._timeout_seconds or 20,
            )
        except grpc.RpcError as exc:
            code = exc.code()
            status = code.name if hasattr(code, "name") else str(code)
            raise APIError(status, exc.details() or str(exc)) from exc

    @staticmethod
    def _process_response(resp: httpx.Response) -> None:
        if resp.status_code in (200, 201):
            return
        if resp.status_code == 207:
            try:
                payload = resp.json()
            except json.JSONDecodeError as exc:
                raise APIError(resp.status_code, "Invalid JSON response") from exc
            errors = payload.get("errors") if isinstance(payload, dict) else None
            if isinstance(errors, list) and errors:
                raise APIErrors(
                    [
                        APIError(
                            err.get("status", resp.status_code), err.get("message", err)
                        )
                        for err in errors
                        if isinstance(err, dict)
                    ]
                )
            return
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            payload = resp.text
        raise APIError(resp.status_code, payload)
