from __future__ import annotations

from types import ModuleType
import sys

import pytest

from models import (
    GenerationRequest,
    GenerationResult,
    MODEL_REGISTRY,
    ModelBackend,
    ModelCapabilities,
    ModelSpec,
    TransformersBackend,
    create_backend,
)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeIds:
    shape = (1, 3)


class FakeOutputRow:
    def __getitem__(self, item):
        return [4, 5]


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, decoded=" adapter response "):
        self.template_calls = []
        self.decoded = decoded

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((list(messages), dict(kwargs)))
        return "rendered prompt"

    def __call__(self, prompt, return_tensors=None):
        if return_tensors:
            return FakeBatch(input_ids=FakeIds())
        return {"input_ids": [1, 2, 3]}

    def decode(self, tokens, skip_special_tokens=True):
        return self.decoded


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [FakeOutputRow()]


def _backend(model_id: str, *, decoded=" adapter response "):
    tokenizer = FakeTokenizer(decoded=decoded)
    model = FakeModel()
    backend = TransformersBackend(MODEL_REGISTRY[model_id], tokenizer=tokenizer, model=model)
    return backend, tokenizer, model


@pytest.mark.parametrize("model_id", ["qwen", "smollm2"])
def test_qwen_and_smollm2_satisfy_same_backend_contract(model_id):
    backend, _, model = _backend(model_id)
    messages = [{"role": "user", "content": "Hello"}]

    assert isinstance(backend, ModelBackend)
    assert backend.count_tokens(messages) == 3
    result = backend.generate(GenerationRequest(messages, max_new_tokens=8, do_sample=False))
    assert result == GenerationResult("adapter response", prompt_tokens=3, completion_tokens=2)
    assert backend.spec.model_name == MODEL_REGISTRY[model_id].model_name
    assert model.generate_kwargs["max_new_tokens"] == 8
    assert model.generate_kwargs["do_sample"] is False


def test_qwen_keeps_native_tool_schema_template_behavior():
    backend, tokenizer, _ = _backend("qwen")
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    backend.generate(GenerationRequest([{"role": "user", "content": "Calculate"}], tools=tools))

    rendered_messages, template_kwargs = tokenizer.template_calls[-1]
    assert rendered_messages == [{"role": "user", "content": "Calculate"}]
    assert template_kwargs["tools"] is tools


def test_smollm2_keeps_tool_template_policy_inside_backend():
    backend, tokenizer, _ = _backend("smollm2")
    tools = [{"type": "function", "function": {
        "name": "calculator",
        "description": "Calculate an expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }}]

    backend.generate(GenerationRequest([{"role": "user", "content": "Calculate"}], tools=tools))

    rendered_messages, template_kwargs = tokenizer.template_calls[-1]
    assert "calculator" in rendered_messages[0]["content"]
    assert "Never repeat a failed call" in rendered_messages[0]["content"]
    assert "Never copy schema metadata" in rendered_messages[0]["content"]
    assert "exact JSON type" in rendered_messages[0]["content"]
    assert "Arithmetic expressions are strings" in rendered_messages[0]["content"]
    assert "calculator(expression: required string)" in rendered_messages[0]["content"]
    assert '"properties"' not in rendered_messages[0]["content"]
    assert "<tool_call>" in rendered_messages[0]["content"]
    assert "tools" not in template_kwargs


def test_smollm2_normalizes_failed_tool_history_for_corrected_retry():
    repeated = '<tool_call>{"name":"calculator","arguments":{"expression":"1/0"}}</tool_call>'
    backend, tokenizer, _ = _backend("smollm2", decoded=repeated)
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Calculate 1/0."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "calculator", "arguments": {"expression": "1/0"}},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": (
            '{"ok":false,"tool":"calculator","error":{"code":"execution_error","message":"division by zero"}}'
        )},
    ]

    result = backend.generate(GenerationRequest(
        messages, tools=[{"type": "function", "function": {"name": "calculator"}}]
    ))

    rendered_messages, _ = tokenizer.template_calls[-1]
    assert rendered_messages[2]["content"] == repeated
    assert rendered_messages[3]["role"] == "user"
    assert "<tool_result" in rendered_messages[3]["content"]
    assert result.text == repeated
    assert messages[2]["content"] == ""
    assert messages[3]["role"] == "tool"


def test_smollm2_coalesces_late_system_enforcement_before_the_user_request():
    backend, tokenizer, _ = _backend("smollm2")
    messages = [
        {"role": "system", "content": "Base policy."},
        {"role": "user", "content": "Calculate 347 * 29."},
        {"role": "system", "content": "The calculator action is still pending."},
    ]
    tools = [{"type": "function", "function": {
        "name": "calculator",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }}]

    backend.generate(GenerationRequest(messages, tools=tools))

    rendered_messages, _ = tokenizer.template_calls[-1]
    assert [message["role"] for message in rendered_messages] == ["system", "user"]
    assert "Base policy." in rendered_messages[0]["content"]
    assert "calculator action is still pending" in rendered_messages[0]["content"]
    assert rendered_messages[-1]["content"] == "Calculate 347 * 29."


def test_smollm2_normalizes_tool_results_for_final_synthesis_without_protocol_reinjection():
    backend, tokenizer, _ = _backend("smollm2", decoded="Final answer")
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "function": {"name": "calculator", "arguments": {"expression": "2+2"}},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": '{"ok":true}'},
    ]

    assert backend.generate(GenerationRequest(messages)).text == "Final answer"
    rendered_messages, _ = tokenizer.template_calls[-1]
    assert rendered_messages[-1]["role"] == "user"
    assert "<tool_result" in rendered_messages[-1]["content"]
    assert "TOOL PROTOCOL FOR THIS TURN" not in rendered_messages[0]["content"]


def test_deterministic_generation_drops_sampling_only_options():
    backend, _, model = _backend("qwen")
    backend.generate(GenerationRequest(
        [{"role": "user", "content": "Hello"}],
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
    ))
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs


def test_registry_is_declarative_and_preserves_model_loading_policies():
    qwen = MODEL_REGISTRY["qwen"]
    smollm2 = MODEL_REGISTRY["smollm2"]

    assert not hasattr(qwen, "loader")
    assert qwen.backend == smollm2.backend == "transformers"
    assert qwen.load_options["quantization"] == {
        "kind": "bitsandbytes_4bit",
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quantization": True,
    }
    assert "quantization" not in smollm2.load_options
    assert qwen.capabilities.tool_schemas and smollm2.capabilities.tool_messages
    assert isinstance(create_backend(qwen), TransformersBackend)


def test_unknown_backend_is_rejected():
    spec = ModelSpec("bad", "Bad", "bad/model", "missing", ModelCapabilities())
    with pytest.raises(ValueError, match="Unknown model backend"):
        create_backend(spec)


def test_transformers_load_preserves_qwen_nf4_and_smollm2_non_quantized(monkeypatch):
    calls = {"tokenizer": [], "model": [], "bnb": []}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            calls["tokenizer"].append(name)
            return object()

    class LoadedModel:
        def eval(self):
            self.evaluated = True

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            model = LoadedModel()
            calls["model"].append((name, kwargs, model))
            return model

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls["bnb"].append(kwargs)

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForCausalLM = AutoModel
    fake_transformers.BitsAndBytesConfig = BitsAndBytesConfig
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    qwen = create_backend(MODEL_REGISTRY["qwen"])
    qwen.load()
    smollm2 = create_backend(MODEL_REGISTRY["smollm2"])
    smollm2.load()

    assert calls["bnb"] == [{
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": fake_torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }]
    qwen_kwargs = calls["model"][0][1]
    smollm_kwargs = calls["model"][1][1]
    assert qwen_kwargs["device_map"] == "auto" and qwen_kwargs["dtype"] == "auto"
    assert "quantization_config" in qwen_kwargs
    assert smollm_kwargs == {"device_map": "auto", "dtype": "auto"}
    assert calls["model"][0][2].evaluated and calls["model"][1][2].evaluated
