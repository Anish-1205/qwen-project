"""Run the two multi-tool grounding acceptances once against the local Qwen model.

External search and currency providers use fixed structured fixtures so this checks
model/orchestrator behavior independently of credentials and network variability.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT
from tools.calculator import calculator
from tools.random_tools import roll_die
from tools.manager import ToolManager
from tools.registry import ToolDefinition, ToolRegistry, build_default_registry


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_PATH = Path(__file__).with_name("live_tool_policy_acceptance.json")


class MemoryStub:
    last_retrieval_stats = {"facts": []}


class RecordingOrchestrator(ConversationOrchestrator):
    def __init__(self, *args, **kwargs):
        self.raw_generations: list[str] = []
        super().__init__(*args, **kwargs)

    def generate_reply(self, messages, **overrides):
        output = super().generate_reply(messages, **overrides)
        self.raw_generations.append(output)
        return output


def fixture_manager() -> ToolManager:
    defaults = build_default_registry()
    registry = ToolRegistry()
    registry.register(defaults.get("roll_die"))
    registry.register(defaults.get("calculator"))
    registry.register(ToolDefinition(
        "search_web",
        "Search the public web and return news context. This result is discovery context, not an authoritative exchange rate.",
        lambda query, count=3, freshness=None: {
            "query": query,
            "results": [{
                "title": "USD/EUR markets react to central-bank outlook",
                "url": "https://example.com/usd-eur-news",
                "snippet": "Currency markets remained focused on policy expectations; a market snippet mentioned 0.8617.",
            }],
        },
        defaults.get("search_web").parameters,
        provenance="discovery",
    ))
    registry.register(ToolDefinition(
        "currency_exchange",
        "Return the authoritative structured USD/EUR reference conversion for this acceptance run.",
        lambda base_currency, quote_currency, amount=None, date=None: {
            "base_currency": base_currency.upper(),
            "quote_currency": quote_currency.upper(),
            "rate": "0.85705",
            "rate_date": "2026-08-20",
            "provider": "acceptance fixture",
            "amount": str(amount),
            "converted_amount": "85.705",
        },
        defaults.get("currency_exchange").parameters,
        provenance="dedicated",
    ))
    return ToolManager(registry)


def run_case(tokenizer, model, prompt: str) -> dict:
    orchestrator = RecordingOrchestrator(
        tokenizer,
        model,
        MemoryStub(),
        tool_manager=fixture_manager(),
        reply_generation_kwargs={"max_new_tokens": 450, "do_sample": False},
        logger=lambda message: None,
    )
    reply = orchestrator.generate_tool_aware_reply(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        turn_number=1,
    )
    return {
        "prompt": prompt,
        "reply": reply,
        "raw_generations": orchestrator.raw_generations,
        "ledger": [asdict(entry) for entry in orchestrator.last_tool_ledger],
    }


def assess(case_a: dict, case_b: dict) -> None:
    ledger_a = case_a["ledger"]
    rolls = [entry for entry in ledger_a if entry["ok"] and entry["tool"] == "roll_die"]
    calculations = [entry for entry in ledger_a if entry["ok"] and entry["tool"] == "calculator"]
    roll_values = [entry["payload"]["data"]["value"] for entry in rolls]
    expected_expression = " + ".join(str(value) for value in roll_values[:2])
    expected_total = sum(roll_values[:2]) if len(roll_values) >= 2 else None
    calculator_arguments = calculations[-1]["arguments"] if calculations else {}
    calculator_used_rolls = (
        calculator_arguments.get("expression", "").replace(" ", "")
        == expected_expression.replace(" ", "")
        or (
            calculator_arguments.get("aggregate") == "sum"
            and calculator_arguments.get("values") == roll_values[:2]
        )
    )
    case_a["pass"] = bool(
        len(rolls) == 2
        and calculations
        and calculator_used_rolls
        and str(expected_total) in case_a["reply"]
    )
    case_a["assessment"] = {
        "roll_values": roll_values,
        "expected_calculator_expression": expected_expression,
        "expected_total": expected_total,
        "roll_count": len(rolls),
    }

    ledger_b = case_b["ledger"]
    successful_tools = [entry["tool"] for entry in ledger_b if entry["ok"]]
    case_b["pass"] = bool(
        "search_web" in successful_tools
        and "currency_exchange" in successful_tools
        and "85.705" in case_b["reply"]
        and ("news" in case_b["reply"].lower() or "market" in case_b["reply"].lower())
    )
    case_b["assessment"] = {
        "successful_tools": successful_tools,
        "authoritative_converted_amount": "85.705",
    }


def main() -> int:
    prior_report = None
    if OUTPUT_PATH.exists():
        prior_report = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    model.eval()
    case_a = run_case(
        tokenizer,
        model,
        "Roll a six-sided die twice, then use calculator to add the two actual returned rolls.",
    )
    case_b = run_case(
        tokenizer,
        model,
        "Search the web for USD/EUR exchange-rate news and use currency_exchange for 100 USD to EUR. Use search only for news context and the currency tool for the conversion.",
    )
    assess(case_a, case_b)
    prior_attempts = list(prior_report.get("prior_attempts", [])) if prior_report else []
    if prior_report:
        prior_attempts.append({
            "case_a": prior_report["case_a"],
            "case_b": prior_report["case_b"],
        })
    report = {
        "runs_per_case": (prior_report.get("runs_per_case", 0) if prior_report else 0) + 1,
        "prior_attempts": prior_attempts,
        "case_a": case_a,
        "case_b": case_b,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "case_a_pass": case_a["pass"],
        "case_b_pass": case_b["pass"],
        "case_a_reply": case_a["reply"],
        "case_b_reply": case_b["reply"],
    }, indent=2))
    return 0 if case_a["pass"] and case_b["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
