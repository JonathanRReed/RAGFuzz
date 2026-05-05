"""Tests for the restored FastAPI demo surface."""

from __future__ import annotations

import inspect
from importlib import import_module
from importlib import util as importlib_util

import pytest

DEMO_SPEC = importlib_util.find_spec("ragfuzz.demo")
FASTAPI_SPEC = importlib_util.find_spec("fastapi")


@pytest.mark.skipif(
    DEMO_SPEC is None,
    reason="ragfuzz.demo is not present yet, this test documents the expected app contract.",
)
@pytest.mark.skipif(
    FASTAPI_SPEC is None,
    reason="FastAPI is not installed in this environment.",
)
def test_demo_module_exposes_a_fastapi_app() -> None:
    from fastapi import FastAPI

    demo_module = import_module("ragfuzz.demo")
    app_factory = (
        getattr(demo_module, "app", None)
        or getattr(demo_module, "create_app", None)
        or getattr(demo_module, "build_app", None)
    )

    if app_factory is None:
        pytest.skip("Expected demo app factory is not exposed yet.")

    app = app_factory() if callable(app_factory) else app_factory

    assert isinstance(app, FastAPI)


@pytest.mark.skipif(
    DEMO_SPEC is None,
    reason="ragfuzz.demo is not present yet, this test documents the streaming endpoint.",
)
@pytest.mark.skipif(
    FASTAPI_SPEC is None,
    reason="FastAPI is not installed in this environment.",
)
def test_demo_run_route_uses_streaming_response() -> None:
    from fastapi.responses import StreamingResponse

    demo_module = import_module("ragfuzz.demo")
    app_factory = (
        getattr(demo_module, "app", None)
        or getattr(demo_module, "create_app", None)
        or getattr(demo_module, "build_app", None)
    )

    if app_factory is None:
        pytest.skip("Expected demo app factory is not exposed yet.")

    app = app_factory() if callable(app_factory) else app_factory
    run_route = next(
        (
            route
            for route in app.routes
            if "run" in getattr(route, "path", "")
            and "POST" in getattr(route, "methods", set())
        ),
        None,
    )

    if run_route is None:
        pytest.skip("Expected demo run route is not exposed yet.")

    response_class = getattr(run_route, "response_class", None)

    assert inspect.isclass(response_class)
    assert issubclass(response_class, StreamingResponse)


@pytest.mark.skipif(
    DEMO_SPEC is None,
    reason="ragfuzz.demo is not present yet.",
)
@pytest.mark.skipif(
    FASTAPI_SPEC is None,
    reason="FastAPI is not installed in this environment.",
)
def test_demo_headers_and_payload_do_not_expose_synthetic_secrets() -> None:
    from fastapi.testclient import TestClient

    demo_module = import_module("ragfuzz.demo")
    app_factory = getattr(demo_module, "create_app", None)

    if app_factory is None:
        pytest.skip("Expected demo app factory is not exposed.")

    client = TestClient(app_factory())
    page_response = client.get("/")
    status_response = client.get("/api/status")
    stream_response = client.get("/api/runs/demo/stream")

    assert page_response.headers["x-content-type-options"] == "nosniff"
    assert page_response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page_response.headers["content-security-policy"]
    assert status_response.headers["cache-control"] == "no-store"
    assert "sk-demo" not in stream_response.text
    assert "Bearer" not in stream_response.text


def test_demo_provider_probe_selects_available_ollama_model() -> None:
    from ragfuzz.demo.state import DemoState

    state = DemoState()
    selected = state._select_model(  # noqa: SLF001
        "ollama",
        "auto",
        ["gemma4:e2b", "nomic-embed-text:latest"],
    )

    assert selected == "gemma4:e2b"


def test_demo_model_selection_updates_provider_state() -> None:
    from ragfuzz.demo.state import DemoState

    state = DemoState()
    state._providers = [  # noqa: SLF001
        {
            "provider_id": "ollama",
            "status": "ready",
            "base_url": "http://localhost:11434/v1",
            "default_model": "gemma4:e2b",
            "models": ["gemma4:e2b", "qwen3.5:9b"],
            "models_available": 2,
        }
    ]

    provider = state.select_model("ollama", "qwen3.5:9b")

    assert provider["default_model"] == "qwen3.5:9b"
    assert state._active_provider_config()["default_model"] == "qwen3.5:9b"  # noqa: SLF001
