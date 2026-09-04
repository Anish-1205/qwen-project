"""Execution backends for compatible local causal language models."""

from __future__ import annotations

import gc
import json
from typing import Mapping, Sequence

from .contracts import GenerationRequest, GenerationResult, ModelBackend, ModelSpec


NATIVE_TOOL_POLICY = "native_tools"
SMOLLM2_TEXT_TOOL_POLICY = "smollm2_text_tools"


class SmolLM2ToolPromptPolicy:
    """Translate canonical harness tool messages to SmolLM2's text protocol."""

    @staticmethod
    def _compact_tools(tools: Sequence[Mapping[str, object]]) -> str:
        """Render schemas as signatures so small models do not copy schema objects as values."""
        rendered = []
        for wrapped in tools:
            function = wrapped.get("function", wrapped)
            name = str(function.get("name", "unknown"))
            description = " ".join(str(function.get("description", "")).split())
            parameters = function.get("parameters", {})
            branches = parameters.get("oneOf") if isinstance(parameters, Mapping) else None
            variants = branches if isinstance(branches, list) and branches else [parameters]
            signatures = []
            for variant in variants:
                variant = variant if isinstance(variant, Mapping) else {}
                properties = variant.get("properties", {})
                required = set(variant.get("required", []))
                arguments = []
                for argument_name, raw_schema in properties.items():
                    schema = raw_schema if isinstance(raw_schema, Mapping) else {}
                    value_type = str(schema.get("type", "value"))
                    details = ["required" if argument_name in required else "optional", value_type]
                    if "enum" in schema:
                        details.append("allowed=" + json.dumps(schema["enum"], ensure_ascii=True))
                    if "minimum" in schema:
                        details.append(f"minimum={schema['minimum']}")
                    if "maximum" in schema:
                        details.append(f"maximum={schema['maximum']}")
                    arguments.append(f"{argument_name}: {' '.join(details)}")
                signatures.append("; ".join(arguments) if arguments else "no arguments")
            signature = " OR ".join(signatures)
            rendered.append(f"- {name}({signature}): {description}")
        return "\n".join(rendered)

    @classmethod
    def protocol_instruction(cls, tools: Sequence[Mapping[str, object]]) -> str:
        return (
            "TOOL PROTOCOL FOR THIS TURN\n"
            "Use only a tool name listed in AVAILABLE_TOOLS. Arguments must be one JSON object matching its schema.\n"
            "Argument values must be concrete values for this request. Never copy schema metadata such as type, "
            "properties, required, content, or description into arguments. If a field has type string, pass a JSON "
            "string, not an object describing a string.\n"
            "Use valid JSON literals: strings need double quotes, integers look like 7, booleans are true or false, "
            "and arrays look like [\"item\"]. Arithmetic expressions are strings: write \"347 * 29\", never the "
            "invalid bare value 347 * 29.\n"
            "When a tool is needed, output exactly one call and no prose:\n"
            '<tool_call>{"name":"exact_tool_name","arguments":{"required_field":"value"}}</tool_call>\n'
            "Before responding, verify that every required argument is present and each value has the exact JSON "
            "type required by the selected tool schema.\n"
            "After a tool result, either answer from successful results or issue one corrected call. "
            "A failed call is not a result. Never repeat a failed call with the same tool name and arguments; "
            "change the arguments or choose another available tool. Never invent tool results.\n"
            "AVAILABLE_TOOLS:\n"
            + cls._compact_tools(tools)
        )

    @classmethod
    def normalize(
        cls,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict]:
        normalized: list[dict] = []
        instruction_added = False
        for original in messages:
            message = dict(original)
            if tools and message.get("role") == "system" and not instruction_added:
                message["content"] = (
                    f"{str(message.get('content', '')).rstrip()}\n\n"
                    f"{cls.protocol_instruction(tools)}"
                )
                instruction_added = True
            tool_calls = message.pop("tool_calls", None)
            if tool_calls:
                rendered_calls = []
                for call in tool_calls:
                    function = call.get("function", {})
                    rendered_calls.append(
                        "<tool_call>"
                        + json.dumps(
                            {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", {}),
                            },
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )
                        + "</tool_call>"
                    )
                message["content"] = "\n".join(rendered_calls)
            if message.get("role") == "tool":
                name = message.get("name", "unknown")
                message = {
                    "role": "user",
                    "content": (
                        f"<tool_result name={json.dumps(name)}>"
                        f"{message.get('content', '')}</tool_result>"
                    ),
                }
            normalized.append(message)
        if tools and not instruction_added:
            normalized.insert(
                0, {"role": "system", "content": cls.protocol_instruction(tools)}
            )
        system_content = [
            str(message.get("content", "")).strip()
            for message in normalized
            if message.get("role") == "system" and str(message.get("content", "")).strip()
        ]
        non_system = [message for message in normalized if message.get("role") != "system"]
        if system_content:
            return [{"role": "system", "content": "\n\n".join(system_content)}, *non_system]
        return non_system


class TransformersBackend(ModelBackend):
    """Hugging Face Transformers execution for local causal language models."""

    def __init__(self, spec: ModelSpec, *, tokenizer=None, model=None):
        self._spec = spec
        self.tokenizer = tokenizer
        self.model = model

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        options = dict(self.spec.load_options)
        device_map = options.pop("device_map", "auto")
        quantization = options.pop("quantization", None)
        options.pop("prompt_policy", None)
        model_kwargs = dict(options.pop("model_kwargs", {}))
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"Unsupported Transformers load option(s): {unknown}")
        model_kwargs.setdefault("device_map", device_map)
        if quantization is not None:
            from transformers import BitsAndBytesConfig

            quantization_options = dict(quantization)
            kind = quantization_options.pop("kind", None)
            if kind != "bitsandbytes_4bit":
                raise ValueError(f"Unsupported Transformers quantization: {kind}")
            quant_type = quantization_options.pop("quant_type", "nf4")
            compute_dtype = quantization_options.pop("compute_dtype", "bfloat16")
            use_double_quant = quantization_options.pop("use_double_quantization", True)
            if quantization_options:
                unknown = ", ".join(sorted(quantization_options))
                raise ValueError(f"Unsupported quantization option(s): {unknown}")
            try:
                resolved_compute_dtype = getattr(torch, str(compute_dtype))
            except AttributeError as exc:
                raise ValueError(f"Unsupported torch compute dtype: {compute_dtype}") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=resolved_compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.spec.model_name, **model_kwargs)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _require_loaded(self) -> None:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(f"Model backend '{self.spec.id}' is not loaded")

    def count_tokens(self, messages: Sequence[Mapping[str, object]]) -> int:
        self._require_loaded()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return len(self.tokenizer(prompt)["input_ids"])

    def _prepare_messages(self, request: GenerationRequest) -> tuple[Sequence[Mapping[str, object]], object]:
        policy = self.spec.load_options.get("prompt_policy", NATIVE_TOOL_POLICY)
        messages = request.messages
        tools = request.tools
        if policy == SMOLLM2_TEXT_TOOL_POLICY:
            if tools or any(message.get("role") == "tool" or message.get("tool_calls") for message in messages):
                messages = SmolLM2ToolPromptPolicy.normalize(messages, tools)
            return messages, None
        if policy != NATIVE_TOOL_POLICY:
            raise ValueError(f"Unsupported Transformers prompt policy: {policy}")
        return messages, tools

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._require_loaded()
        import torch

        messages, tools = self._prepare_messages(request)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools is not None:
            template_kwargs["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        options = dict(request.options)
        options["do_sample"] = request.do_sample
        if request.do_sample:
            if request.temperature is not None:
                options["temperature"] = request.temperature
            if request.top_p is not None:
                options["top_p"] = request.top_p
        else:
            options.pop("temperature", None)
            options.pop("top_p", None)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **options,
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            completion_tokens = len(new_tokens)
        except TypeError:
            completion_tokens = None
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is a runtime dependency
            pass


BACKEND_TYPES = {"transformers": TransformersBackend}


def create_backend(spec: ModelSpec) -> ModelBackend:
    """Construct an unloaded backend for a declarative model specification."""

    try:
        backend_type = BACKEND_TYPES[spec.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown model backend: {spec.backend}") from exc
    return backend_type(spec)
