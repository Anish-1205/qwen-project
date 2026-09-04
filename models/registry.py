"""Explicit model choices available to the local application."""

from __future__ import annotations

from .backends import NATIVE_TOOL_POLICY, SMOLLM2_TEXT_TOOL_POLICY
from .contracts import ModelCapabilities, ModelSpec


MODEL_REGISTRY = {
    "qwen": ModelSpec(
        id="qwen",
        display_name="Qwen 2.5 3B Instruct",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        backend="transformers",
        capabilities=ModelCapabilities(tool_schemas=True, tool_messages=True),
        load_options={
            "device_map": "auto",
            "model_kwargs": {"dtype": "auto"},
            "prompt_policy": NATIVE_TOOL_POLICY,
            "quantization": {
                "kind": "bitsandbytes_4bit",
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "use_double_quantization": True,
            },
        },
    ),
    "smollm2": ModelSpec(
        id="smollm2",
        display_name="SmolLM2 1.7B Instruct",
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        backend="transformers",
        capabilities=ModelCapabilities(tool_schemas=True, tool_messages=True),
        load_options={
            "device_map": "auto",
            "model_kwargs": {"dtype": "auto"},
            "prompt_policy": SMOLLM2_TEXT_TOOL_POLICY,
        },
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model id: {model_id}") from exc
