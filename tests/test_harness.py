from __future__ import annotations

from harness import DEFAULT_SYSTEM_PROMPT, HarnessRunner, RunRequest, RunResult
from models import GenerationResult, ModelBackend, ModelCapabilities, ModelSpec


class QueueBackend(ModelBackend):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self._spec = ModelSpec(
            "queue", "Queue", "test/queue-model", "test",
            ModelCapabilities(tool_schemas=True, tool_messages=True),
        )

    @property
    def spec(self):
        return self._spec

    def load(self):
        pass

    def generate(self, request):
        self.calls.append(request)
        return GenerationResult(self.outputs.pop(0))

    def count_tokens(self, messages):
        return sum(len(str(message.get("content", "")).split()) for message in messages)


class FakeMemory:
    last_retrieval_stats = {"facts": []}


def test_structured_run_contract_and_model_trace():
    backend = QueueBackend(["Hello from an interchangeable model."])
    runner = HarnessRunner(backend, FakeMemory(), logger=lambda message: None)
    history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    result = runner.run(RunRequest("Hello", history, turn_number=1))

    assert isinstance(result, RunResult)
    assert result.output == "Hello from an interchangeable model."
    assert result.messages[-1] == {"role": "assistant", "content": result.output}
    assert result.state.final_output == result.output
    assert result.state.model_call_count == 1
    request = backend.calls[0]
    assert request.max_new_tokens == 300
    assert request.do_sample is True
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert [event.kind for event in result.trace] == [
        "routing", "model_call", "memory_retrieval", "document_retrieval", "final_output"
    ]
    assert result.trace[1].data["model"] == "test/queue-model"


def test_tool_loop_honors_configured_max_steps_and_traces_execution():
    backend = QueueBackend([
        '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
        "The result is 4.",
    ])
    runner = HarnessRunner(
        backend,
        FakeMemory(),
        max_tool_steps=1,
        logger=lambda message: None,
    )
    history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    result = runner.run(RunRequest("Calculate 2 + 2", history, turn_number=1))

    assert result.output == "The result is 4."
    assert len(result.state.tool_ledger) == 1
    assert result.state.tool_ledger[0].tool == "calculator"
    assert result.state.tool_ledger[0].ok
    kinds = [event.kind for event in result.trace]
    assert kinds.count("model_call") == 2
    assert kinds.count("tool_call") == 1
    assert kinds.count("tool_result") == 1
    assert backend.calls[0].tools
    assert any(message.get("tool_calls") for message in backend.calls[1].messages)
    assert any(message.get("role") == "tool" for message in backend.calls[1].messages)


def test_capability_configuration_disables_tool_execution():
    backend = QueueBackend(["Tools are disabled for this runner."])
    runner = HarnessRunner(
        backend,
        FakeMemory(),
        enabled_capabilities={"memory", "documents"},
        logger=lambda message: None,
    )
    history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    result = runner.run(RunRequest("Calculate 2 + 2", history, turn_number=1))

    assert result.output == "Tools are disabled for this runner."
    assert result.state.routing_decision.intent.tool_use is False
    assert not result.state.tool_ledger
    assert backend.calls[0].tools is None


def test_invalid_tool_step_configuration_is_rejected():
    try:
        HarnessRunner(QueueBackend([]), FakeMemory(), max_tool_steps=0)
    except ValueError as exc:
        assert "max_tool_steps" in str(exc)
    else:
        raise AssertionError("expected invalid max_tool_steps to fail")
