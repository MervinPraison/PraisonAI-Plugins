"""Tests for cli_backend_tracer plugin."""

import logging

from praisonai_plugins.observability.cli_backend_tracer import (
    CliBackendTracerPlugin,
    create_plugin,
)


def test_create_plugin_factory():
    assert isinstance(create_plugin(), CliBackendTracerPlugin)


def test_cli_backend_tracer_logs_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("PRAISONAI_CLI_BACKEND_DEBUG", "1")
    caplog.set_level(logging.INFO)

    plugin = CliBackendTracerPlugin()
    plugin.cli_backend_execute(
        {
            "agent_name": "assistant",
            "backend": "gemini",
            "session_id": "sess-1",
            "command": ["gemini", "-p", "hi"],
            "transport": "subprocess",
            "praisonai_llm_http": False,
            "error": None,
        }
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "CLI backend delegation" in message
    assert "gemini" in message
    assert "praisonai_llm_http=False" in message


def test_cli_backend_tracer_silent_by_default(monkeypatch, caplog):
    monkeypatch.delenv("PRAISONAI_CLI_BACKEND_DEBUG", raising=False)
    monkeypatch.delenv("LOGLEVEL", raising=False)
    caplog.set_level(logging.INFO)

    plugin = CliBackendTracerPlugin()
    plugin.cli_backend_execute({"backend": "gemini"})

    assert not caplog.records
