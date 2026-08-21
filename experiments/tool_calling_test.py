import json
import logging
import math
import re
import warnings
from typing import Any

import torch
from huggingface_hub.utils import logging as hf_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("torch.utils.flop_counter").disabled = True
warnings.filterwarnings("ignore", message="triton not found")
hf_logging.set_verbosity_error()

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def calculate(expression: str) -> float:
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("Empty expression")
    if "__" in expression:
        raise ValueError("Invalid expression")

    allowed_names: dict[str, Any] = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round})

    value = eval(expression, {"__builtins__": {}}, allowed_names)
    if isinstance(value, complex):
        raise ValueError("Complex results are not supported")
    return float(value)


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression and return a numeric result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A valid math expression, e.g. '47 * 89' or '0.15 * 340'.",
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def generate_text(tokenizer, model, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_new_tokens: int = 220) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = []

    tag_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    if tag_match:
        candidates.append(tag_match.group(1))

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        candidates.append(fence_match.group(1))

    brace_candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.S)
    candidates.extend(brace_candidates)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def detect_tool_call(raw_output: str) -> tuple[bool, str | None, dict[str, Any] | None]:
    text = (raw_output or "").strip()

    parsed = _extract_json_object(text)
    if not parsed:
        return False, None, None

    if parsed.get("name"):
        name = str(parsed.get("name"))
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"expression": args}
        if isinstance(args, dict):
            return True, name, args

    if parsed.get("function") and isinstance(parsed["function"], dict):
        fn = parsed["function"]
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"expression": args}
        if name and isinstance(args, dict):
            return True, str(name), args

    return False, None, None


def run_test_case(tokenizer, model, tools: list[dict[str, Any]], prompt: str, idx: int) -> None:
    print(f"\n=== Test Case {idx} ===")
    print(f"Prompt: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    raw_output = generate_text(tokenizer, model, messages, tools=tools, max_new_tokens=220)

    print("Raw model output:")
    print(raw_output)

    has_tool_call, tool_name, tool_args = detect_tool_call(raw_output)
    print(f"Tool call detected: {has_tool_call}")

    if not has_tool_call:
        print("Tool call arguments: None")
        print("Tool execution result: None")
        print("Final answer:")
        print(raw_output)
        return

    print(f"Tool call name: {tool_name}")
    print("Tool call arguments:")
    print(json.dumps(tool_args, indent=2, ensure_ascii=True))

    tool_result_payload: dict[str, Any]
    try:
        if tool_name != "calculate":
            raise ValueError(f"Unknown tool: {tool_name}")
        expression = str((tool_args or {}).get("expression", ""))
        result = calculate(expression)
        tool_result_payload = {"ok": True, "result": result}
    except Exception as exc:
        tool_result_payload = {"ok": False, "error": str(exc)}

    print("Tool execution result:")
    print(json.dumps(tool_result_payload, indent=2, ensure_ascii=True))

    tool_call_for_messages = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_args or {}, ensure_ascii=True),
        },
    }

    followup_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "", "tool_calls": [tool_call_for_messages]},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": tool_name,
            "content": json.dumps(tool_result_payload, ensure_ascii=True),
        },
    ]

    final_answer = generate_text(tokenizer, model, followup_messages, tools=tools, max_new_tokens=260)
    print("Final answer:")
    print(final_answer)


def main() -> None:
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    tools = build_tools()

    prompts = [
        "What is 47 * 89?",
        "What is the capital of France?",
        "Can you calculate 15% of 340?",
        "Calculate the square root of -4",
        "Tell me a joke",
        "What's 12 divided by 0?",
    ]

    for idx, prompt in enumerate(prompts, start=1):
        run_test_case(tokenizer, model, tools, prompt, idx)


if __name__ == "__main__":
    main()
