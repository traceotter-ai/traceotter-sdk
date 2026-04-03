import pytest

from traceotter.client import TraceotterClient, TraceotterConfigurationError


def test_missing_traceotter_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACEOTTER_API_KEY", raising=False)
    with pytest.raises(TraceotterConfigurationError):
        TraceotterClient()
