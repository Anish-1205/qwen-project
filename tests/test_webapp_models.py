from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import webapp
from models import MODEL_REGISTRY, GenerationResult, ModelBackend, ModelCapabilities, ModelSpec


def _spec(model_id, display_name, model_name):
    return ModelSpec(
        model_id, display_name, model_name, "test",
        ModelCapabilities(tool_schemas=True, tool_messages=True),
    )


class FakeBackend(ModelBackend):
    def __init__(self, spec, *, load_error=None):
        self._spec = spec
        self.load_error = load_error
        self.loaded = False
        self.closed = False

    @property
    def spec(self):
        return self._spec

    def load(self):
        if self.load_error:
            raise self.load_error
        self.loaded = True

    def generate(self, request):
        return GenerationResult("")

    def count_tokens(self, messages):
        return 0

    def close(self):
        self.closed = True


@pytest.fixture
def model_state(monkeypatch):
    original = {
        "status": webapp.APP_STATE.status,
        "detail": webapp.APP_STATE.detail,
        "error": webapp.APP_STATE.error,
        "logger": webapp.APP_STATE.logger,
        "model_backend": webapp.APP_STATE.model_backend,
        "active_model_id": webapp.APP_STATE.active_model_id,
        "tool_selector": webapp.APP_STATE.tool_selector,
        "active_tool_selector_id": webapp.APP_STATE.active_tool_selector_id,
        "orchestrator": webapp.APP_STATE.orchestrator,
    }
    current = FakeBackend(MODEL_REGISTRY["qwen"])
    runner = SimpleNamespace(model_backend=current)
    selector = object()
    test_logger = SimpleNamespace(info=lambda message: None, error=lambda message: None)
    webapp._set_state(
        status="ready",
        detail="Ready.",
        error=None,
        logger=test_logger,
        model_backend=current,
        active_model_id="qwen",
        orchestrator=runner,
        tool_selector=selector,
        active_tool_selector_id="needle2",
    )
    monkeypatch.setattr(webapp, "_release_model_backend", lambda backend: backend.close() if backend else None)
    yield runner, current
    webapp._set_state(**original)


def test_available_model_listing_and_active_reporting(model_state):
    payload = webapp.list_models()

    assert [(item["id"], item["display_name"]) for item in payload["models"]] == [
        ("qwen", "Qwen 2.5 3B Instruct"),
        ("smollm2", "SmolLM2 1.7B Instruct"),
    ]
    assert payload["active_model_id"] == "qwen"
    assert payload["active_tool_selector_id"] == "needle2"
    assert next(item for item in payload["models"] if item["id"] == "qwen")["active"] is True
    assert webapp.health()["active_model_id"] == "qwen"


def test_successful_model_switch_reuses_existing_harness_and_tool_selector(model_state, monkeypatch):
    runner, previous = model_state
    logs = []
    webapp._set_state(logger=SimpleNamespace(info=logs.append, error=logs.append))
    selector = webapp.APP_STATE.tool_selector
    spec = _spec("smollm2", "SmolLM2 1.7B Instruct", "test/smollm2")
    replacement = FakeBackend(spec)
    monkeypatch.setitem(MODEL_REGISTRY, "smollm2", spec)
    monkeypatch.setattr(webapp, "create_backend", lambda selected: replacement)

    payload = webapp.select_model(webapp.ModelSelectionRequest(model_id="smollm2"))

    assert payload["active_model_id"] == "smollm2"
    assert webapp.APP_STATE.orchestrator is runner
    assert runner.model_backend is replacement
    assert webapp.APP_STATE.model_backend is replacement
    assert runner.model_backend is not previous
    assert replacement.loaded
    assert previous.closed
    assert webapp.APP_STATE.tool_selector is selector
    assert webapp.APP_STATE.active_tool_selector_id == "needle2"
    assert any(
        "[Model Switch] outcome=started from_model_id=qwen to_model_id=smollm2" in message
        for message in logs
    )
    assert any(
        "[Model Switch] outcome=ready from_model_id=qwen to_model_id=smollm2" in message
        and "selector=needle2" in message
        for message in logs
    )


def test_invalid_model_id_returns_not_found(model_state):
    with pytest.raises(HTTPException) as exc_info:
        webapp.select_model(webapp.ModelSelectionRequest(model_id="not-a-model"))

    assert exc_info.value.status_code == 404
    assert "Unknown model id" in exc_info.value.detail


def test_shutdown_closes_active_backend(model_state):
    runner, current = model_state

    webapp.shutdown_event()

    assert current.closed
    assert webapp.APP_STATE.model_backend is None
    assert runner.model_backend is None


def test_failed_model_load_restores_previous_model(model_state, monkeypatch):
    runner, _ = model_state
    smol_spec = _spec("smollm2", "SmolLM2 1.7B Instruct", "test/smollm2")
    qwen_spec = _spec("qwen", "Qwen 2.5 3B Instruct", "test/qwen")
    failed = FakeBackend(smol_spec, load_error=RuntimeError("simulated load failure"))
    restored = FakeBackend(qwen_spec)

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "smollm2",
        smol_spec,
    )
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "qwen",
        qwen_spec,
    )
    monkeypatch.setattr(
        webapp, "create_backend",
        lambda spec: failed if spec.id == "smollm2" else restored,
    )

    with pytest.raises(HTTPException) as exc_info:
        webapp.select_model(webapp.ModelSelectionRequest(model_id="smollm2"))

    assert exc_info.value.status_code == 503
    assert "previous model was restored" in exc_info.value.detail
    assert webapp.APP_STATE.status == "ready"
    assert webapp.APP_STATE.active_model_id == "qwen"
    assert webapp.APP_STATE.model_backend is restored
    assert runner.model_backend is restored
    assert restored.loaded
