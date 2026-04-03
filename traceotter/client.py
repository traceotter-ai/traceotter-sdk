from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import Any

from traceotter._utils.ingest_schema import SchemaValidationError, validate_span_schema
from traceotter._utils.request import (
    APIError,
    APIErrors,
    TraceotterHttpIngestClient,
    grpc_target_from_base_url,
    order_spans_parents_first,
)
from traceotter._utils.serializer import safe_json_dumps
from traceotter.models import OTelEvent, OTelSpanPayload

log = logging.getLogger("traceotter")


def now_ns() -> int:
    return time.time_ns()


def _to_raw_span(span: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(span.get("trace_id") or "")
    span_id = str(span.get("span_id") or "")
    parent_span_id = span.get("parent_span_id")

    attributes = dict(span.get("attributes") or {})
    attributes.setdefault("otel_span_name", span.get("name"))
    attributes.setdefault("otel_status_code", span.get("status_code"))
    attributes.setdefault("otel_status_description", span.get("status_message"))
    if parent_span_id is not None:
        attributes.setdefault("parent_span_id", str(parent_span_id))

    raw: dict[str, Any] = {
        "trace_id": trace_id or None,
        "id": span_id or None,
        "span_id": span_id or None,
        "start_time": (
            float(span["start_time_unix_nano"]) / 1_000_000_000.0
            if span.get("start_time_unix_nano") is not None
            else None
        ),
        "attributes": attributes,
        "context": {},
    }
    name = span.get("name")
    if isinstance(name, str):
        raw["name"] = name
    return raw


class ConsoleExporter:
    """Placeholder exporter for local development."""

    def export(self, envelopes: list[dict[str, Any]]) -> None:
        print(safe_json_dumps(envelopes))


class HttpIngestExporter:
    def __init__(
        self,
        *,
        secret_key: str,
        host: str,
        timeout_seconds: int = 5,
        version: str = "0.2.6",
        max_retries: int = 3,
        public_key: str | None = None,
        grpc_target: str | None = None,
    ) -> None:
        self._client = TraceotterHttpIngestClient(
            secret_key=secret_key,
            base_url=host,
            version=version,
            timeout_seconds=timeout_seconds,
            grpc_target=grpc_target,
        )
        self._max_retries = max_retries

    def close(self) -> None:
        self._client.close()

    def export(self, envelopes: list[dict[str, Any]]) -> None:
        raw_spans = self._extract_raw_spans(envelopes)
        validated = []
        for raw in raw_spans:
            try:
                validated.append(validate_span_schema(raw))
            except SchemaValidationError:
                continue
        if not validated:
            return
        ordered = order_spans_parents_first(validated)
        self._send_with_retry(ordered)

    @staticmethod
    def _extract_raw_spans(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for envelope in envelopes:
            items = envelope.get("spans") if isinstance(envelope, dict) else None
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("details"), dict):
                    spans.append(item["details"])
        return spans

    def _send_with_retry(self, spans: list[dict[str, Any]]) -> None:
        attempt = 0
        delay = 0.5
        while True:
            try:
                self._client.batch_post(spans)
                return
            except APIErrors as err:
                log.debug("Ingest batch rejected: %s", err)
                return
            except APIError as err:
                status = int(err.status) if str(err.status).isdigit() else 0
                retryable = status == 429 or status >= 500 or status == 0
                if attempt >= self._max_retries or not retryable:
                    log.debug("Ingest failed: %s", err)
                    return
            except Exception:  # noqa: BLE001
                if attempt >= self._max_retries:
                    log.debug("Ingest failed after retries", exc_info=True)
                    return
            attempt += 1
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


class TraceotterClient:
    def __init__(
        self,
        *,
        exporter: ConsoleExporter | None = None,
        batch_size: int | None = None,
        flush_interval_seconds: float | None = None,
    ) -> None:
        env_flush_at = os.environ.get("TRACEOTTER_FLUSH_AT")
        env_flush_interval = os.environ.get("TRACEOTTER_FLUSH_INTERVAL")
        env_otel_flush_at = os.environ.get("OTEL_BSP_MAX_EXPORT_BATCH_SIZE")
        env_otel_flush_interval_ms = os.environ.get("OTEL_BSP_SCHEDULE_DELAY")

        resolved_batch_size = batch_size
        if resolved_batch_size is None:
            if env_flush_at is not None:
                resolved_batch_size = int(env_flush_at)
            elif env_otel_flush_at is not None:
                resolved_batch_size = int(env_otel_flush_at)
            else:
                resolved_batch_size = 512

        resolved_flush_interval_seconds = flush_interval_seconds
        if resolved_flush_interval_seconds is None:
            if env_flush_interval is not None:
                resolved_flush_interval_seconds = float(env_flush_interval)
            elif env_otel_flush_interval_ms is not None:
                resolved_flush_interval_seconds = (
                    float(env_otel_flush_interval_ms) / 1000.0
                )
            else:
                resolved_flush_interval_seconds = 5.0

        self._exporter = exporter or _default_exporter()
        self._batch_size = max(1, int(resolved_batch_size))
        self._flush_interval_seconds = max(0.1, float(resolved_flush_interval_seconds))

        self._queue: Queue[dict[str, Any]] = Queue()
        self._pending_by_trace: dict[str, list[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._ready_batch: list[dict[str, Any]] = []
        self._ready_lock = threading.Lock()
        self._last_flush_monotonic = time.monotonic()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def enqueue_span(self, payload: OTelSpanPayload) -> None:
        self._queue.put(self._encode(payload))

    def flush(self) -> None:
        # Drain queue first so in-flight spans become visible to this flush.
        while True:
            try:
                item = self._queue.get_nowait()
                self._buffer_or_export_complete_trace(item)
            except Empty:
                break

        self._flush_ready_batch(force=True)

        # Force-export remaining partial traces (useful for script end/shutdown).
        with self._pending_lock:
            remaining_batches = list(self._pending_by_trace.values())
            self._pending_by_trace.clear()

        for batch in remaining_batches:
            if batch:
                raw_spans = [_to_raw_span(span) for span in batch]
                self._exporter.export(
                    [{"spans": [{"details": span} for span in raw_spans]}]
                )

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._thread.join(timeout=2)
        self.flush()
        close = getattr(self._exporter, "close", None)
        if callable(close):
            close()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=self._flush_interval_seconds)
                self._buffer_or_export_complete_trace(item)
            except Empty:
                self._flush_ready_batch(force=False)

    def _buffer_or_export_complete_trace(self, span: dict[str, Any]) -> None:
        trace_id = str(span.get("trace_id"))
        with self._pending_lock:
            trace_batch = self._pending_by_trace.setdefault(trace_id, [])
            trace_batch.append(span)

            # Root span completion means the trace is complete in LangChain flow.
            is_completed_root = (
                span.get("parent_span_id") is None
                and span.get("end_time_unix_nano") is not None
            )
            if not is_completed_root:
                return

            completed_trace_batch = self._pending_by_trace.pop(trace_id, [])

        if completed_trace_batch:
            with self._ready_lock:
                self._ready_batch.extend(completed_trace_batch)
                ready_len = len(self._ready_batch)
            if ready_len >= self._batch_size:
                self._flush_ready_batch(force=True)

    def _flush_ready_batch(self, *, force: bool) -> None:
        now = time.monotonic()
        with self._ready_lock:
            if not self._ready_batch:
                self._last_flush_monotonic = now
                return

            interval_elapsed = (
                now - self._last_flush_monotonic
            ) >= self._flush_interval_seconds
            reached_batch_size = len(self._ready_batch) >= self._batch_size
            if not force and not interval_elapsed and not reached_batch_size:
                return

            batch = self._ready_batch
            self._ready_batch = []
            self._last_flush_monotonic = now

        if batch:
            raw_spans = [_to_raw_span(span) for span in batch]
            self._exporter.export(
                [{"spans": [{"details": span} for span in raw_spans]}]
            )

    @staticmethod
    def _encode(payload: OTelSpanPayload) -> dict[str, Any]:
        return {
            "trace_id": payload.trace_id,
            "span_id": payload.span_id,
            "parent_span_id": payload.parent_span_id,
            "name": payload.name,
            "kind": payload.kind,
            "start_time_unix_nano": payload.start_time_unix_nano,
            "end_time_unix_nano": payload.end_time_unix_nano,
            "status_code": payload.status_code,
            "status_message": payload.status_message,
            "attributes": payload.attributes,
            "events": [
                TraceotterClient._encode_event(event) for event in payload.events
            ],
        }

    @staticmethod
    def _encode_event(event: OTelEvent) -> dict[str, Any]:
        return {
            "name": event.name,
            "timestamp_unix_nano": event.timestamp_unix_nano,
            "attributes": event.attributes,
        }


_CLIENT_SINGLETON: TraceotterClient | None = None
_CLIENT_SINGLETON_LOCK = threading.Lock()


def _default_exporter() -> Any:
    secret_key = os.environ.get("TRACEOTTER_API_KEY")
    host = os.environ.get("TRACEOTTER_HOST", "https://api.traceotter.com")
    timeout = int(os.environ.get("TRACEOTTER_TIMEOUT", "5"))
    use_grpc = os.environ.get("TRACEOTTER_USE_GRPC", "").lower() in ("1", "true", "yes")
    grpc_port = int(os.environ.get("TRACEOTTER_GRPC_PORT", "50051"))
    grpc_target = grpc_target_from_base_url(host, grpc_port) if use_grpc else None
    if secret_key:
        return HttpIngestExporter(
            secret_key=secret_key,
            host=host,
            timeout_seconds=timeout,
            grpc_target=grpc_target,
        )
    return ConsoleExporter()


def get_client(
    *,
    exporter: ConsoleExporter | None = None,
    batch_size: int | None = None,
    flush_interval_seconds: float | None = None,
) -> TraceotterClient:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is not None:
        return _CLIENT_SINGLETON

    with _CLIENT_SINGLETON_LOCK:
        if _CLIENT_SINGLETON is None:
            _CLIENT_SINGLETON = TraceotterClient(
                exporter=exporter,
                batch_size=batch_size,
                flush_interval_seconds=flush_interval_seconds,
            )
    return _CLIENT_SINGLETON
