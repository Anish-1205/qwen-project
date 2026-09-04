from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from intent_classifier import IntentDecision
from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT
from tools import ToolCall, ToolManager, calculator
from tools.registry import ToolDefinition, ToolRegistry


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        return " ".join(str(message.get("content", "")) for message in messages)

    def __call__(self, prompt, *args, **kwargs):
        return {"input_ids": list(range(len(str(prompt).split())))}


class FakeMemory:
    last_retrieval_stats = {"facts": []}


class StaticClassifier:
    last_used_fallback = False

    def classify(self, user_input, messages):
        if user_input == "Hello!":
            return IntentDecision(False, False, False, True, False)
        return IntentDecision(False, False, False, False, True)


class QueueOrchestrator(ConversationOrchestrator):
    def __init__(self, *args, generated, **kwargs):
        self.generated = list(generated)
        self.generation_inputs = []
        self.generation_overrides = []
        super().__init__(*args, **kwargs)

    def generate_reply(self, messages, **overrides):
        self.generation_inputs.append([dict(message) for message in messages])
        self.generation_overrides.append(dict(overrides))
        if not self.generated:
            raise AssertionError("unexpected model generation")
        return self.generated.pop(0)


class ToolManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = ToolManager()

    def test_registry_exposes_only_the_approved_tool_schemas(self):
        schemas = self.manager.schemas()
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            ["roll_die", "random_number", "calculator", "currency_exchange", "search_web", "fetch_webpage", "weather", "read_file", "list_directory", "analyze_spreadsheet"],
        )
        self.assertFalse(schemas[0]["function"]["parameters"]["additionalProperties"])

    def test_tool_selection_parser_accepts_qwen_tool_call_format(self):
        output = '<tool_call>\n{"name":"calculator","arguments":{"expression":"25 * 4"}}\n</tool_call>'
        self.assertEqual(
            self.manager.parse_tool_call(output),
            ToolCall(name="calculator", arguments={"expression": "25 * 4"}),
        )
        self.assertIsNone(self.manager.parse_tool_call("A normal answer with {braces} stays normal."))

    def test_argument_validation_defaults_and_rejects_bad_arguments(self):
        with patch("tools.random_tools.random.randint", return_value=4) as randint:
            result = self.manager.execute(ToolCall("roll_die", {}))
        self.assertTrue(result.ok)
        self.assertEqual(result.result, 4)
        self.assertEqual(result.validated_arguments, {"sides": 6})
        randint.assert_called_once_with(1, 6)

        wrong_type = self.manager.execute(ToolCall("roll_die", {"sides": "six"}))
        extra = self.manager.execute(ToolCall("calculator", {"expression": "1 + 1", "code": "bad"}))
        reversed_range = self.manager.execute(ToolCall("random_number", {"minimum": 10, "maximum": 1}))
        self.assertFalse(wrong_type.ok)
        self.assertIn("integer", wrong_type.error)
        self.assertFalse(extra.ok)
        self.assertIn("unexpected", extra.error)
        self.assertFalse(reversed_range.ok)
        self.assertIn("less than or equal", reversed_range.error)

    def test_execution_and_invalid_tool_calls_return_structured_results(self):
        valid = self.manager.execute(ToolCall("calculator", {"expression": "(10 + 5) / 3"}))
        unknown = self.manager.execute(ToolCall("run_python", {"code": "print('no')"}))
        missing = self.manager.execute(ToolCall("calculator", {}))
        self.assertEqual(valid.payload()["data"], {"value": 5.0})
        self.assertEqual(unknown.payload()["error"]["code"], "unknown_tool")
        self.assertFalse(missing.ok)
        self.assertIn("missing required", missing.error)

    def test_calculator_supports_arithmetic_and_blocks_code(self):
        self.assertEqual(calculator("25 * 4"), 100)
        self.assertEqual(calculator("2 ** 8"), 256)
        self.assertAlmostEqual(calculator("(10 + 5) / 3"), 5.0)
        blocked = [
            "__import__('os').system('whoami')",
            "open('secret.txt').read()",
            "(1).__class__",
            "[1, 2, 3]",
        ]
        for expression in blocked:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculator(expression)


class ToolOrchestrationTests(unittest.TestCase):
    def _orchestrator(self, generated, *, tool_manager=None):
        return QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=generated,
            intent_classifier=StaticClassifier(),
            tool_manager=tool_manager,
            logger=lambda message: None,
        )

    @staticmethod
    def _tool_message(messages, name):
        return next(
            message for message in reversed(messages)
            if message.get("role") == "tool" and message.get("name") == name
        )

    def test_selected_tool_is_executed_and_returned_to_qwen_for_final_response(self):
        call = '<tool_call>{"name":"calculator","arguments":{"expression":"25 * 4"}}</tool_call>'
        orchestrator = self._orchestrator([call, "25 times 4 is 100."])
        history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        messages, reply, _, _ = orchestrator.process_turn(
            "What is 25 * 4?", history, turn_number=1
        )

        self.assertEqual(reply, "25 times 4 is 100.")
        self.assertTrue(orchestrator.last_tool_execution.ok)
        followup = orchestrator.generation_inputs[1]
        tool_message = self._tool_message(followup, "calculator")
        self.assertEqual(json.loads(tool_message["content"])["data"], {"value": 100})
        ledger_message = followup[-1]
        self.assertEqual(ledger_message["role"], "system")
        self.assertIn('"result_id": "result_1"', ledger_message["content"])
        self.assertIn('"provenance": "dedicated"', ledger_message["content"])
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertTrue(all("tools" in overrides for overrides in orchestrator.generation_overrides))

    def test_invalid_tool_request_is_not_executed_and_qwen_receives_the_error(self):
        call = '<tool_call>{"name":"shell","arguments":{"command":"dir"}}</tool_call>'
        valid = '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>'
        orchestrator = self._orchestrator([call, "I cannot run that tool.", valid, "The result is 4."])

        _, reply, _, _ = orchestrator.process_turn(
            "Calculate 2 + 2.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "The result is 4.")
        self.assertTrue(orchestrator.last_tool_execution.ok)
        self.assertEqual([result.ok for result in orchestrator.last_tool_executions], [False, True])
        payload = json.loads(self._tool_message(orchestrator.generation_inputs[1], "shell")["content"])
        self.assertEqual(payload["error"]["code"], "unknown_tool")

    def test_repeated_calls_have_stable_ordered_results_and_earlier_results_remain_visible(self):
        two_rolls = (
            '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
            '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
        )
        third_roll = '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
        calculator_call = '<tool_call>{"name":"calculator","arguments":{"expression":"3 + 2"}}</tool_call>'
        orchestrator = self._orchestrator([two_rolls, third_roll, calculator_call, "The total is 5."])

        with patch("tools.random_tools.random.randint", side_effect=[3, 2, 6]):
            reply = orchestrator.generate_tool_aware_reply(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "Roll a six-sided die twice, then use calculator to add the two rolls."},
                ],
                turn_number=1,
            )

        self.assertEqual(reply, "The total is 5.")
        self.assertEqual([entry.result_id for entry in orchestrator.last_tool_ledger], [
            "result_1", "result_2", "result_3", "result_4",
        ])
        self.assertEqual([entry.tool for entry in orchestrator.last_tool_ledger], [
            "roll_die", "roll_die", "roll_die", "calculator",
        ])
        self.assertEqual([entry.payload.get("data", {}).get("value") for entry in orchestrator.last_tool_ledger[:3]], [3, 2, 6])
        self.assertEqual(orchestrator.last_tool_ledger[3].arguments, {"expression": "3 + 2"})
        final_ledger = orchestrator.generation_inputs[-1][-1]["content"]
        self.assertLess(final_ledger.index('"result_id": "result_1"'), final_ledger.index('"result_id": "result_2"'))
        self.assertIn('"result_id": "result_3"', final_ledger)
        self.assertIn("later calls never replace earlier results", final_ledger)
        self.assertEqual(len({entry.context_call_id for entry in orchestrator.last_tool_ledger}), 4)

    def test_later_different_tool_in_same_generation_is_regenerated_from_ledger(self):
        stale_batch = (
            '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
            '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
            '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 3"}}</tool_call>'
        )
        grounded_calculator = '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 5"}}</tool_call>'
        orchestrator = self._orchestrator([stale_batch, grounded_calculator, "The total is 7."])

        with patch("tools.random_tools.random.randint", side_effect=[2, 5]):
            reply = orchestrator.generate_tool_aware_reply(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "Roll a six-sided die twice, then use calculator to add the rolls."},
                ],
                turn_number=1,
            )

        self.assertEqual(reply, "The total is 7.")
        self.assertEqual([entry.tool for entry in orchestrator.last_tool_ledger], [
            "roll_die", "roll_die", "calculator",
        ])
        self.assertEqual(orchestrator.last_tool_ledger[-1].arguments, {"expression": "2 + 5"})
        self.assertNotIn("2 + 3", [entry.arguments.get("expression") for entry in orchestrator.last_tool_ledger])
        second_generation_ledger = orchestrator.generation_inputs[1][-1]["content"]
        self.assertIn('"value": 2', second_generation_ledger)
        self.assertIn('"value": 5', second_generation_ledger)

    def test_malformed_attempt_is_ledgered_and_does_not_disturb_successful_result(self):
        roll = '<tool_call>{"name":"roll_die","arguments":{"sides":6}}</tool_call>'
        malformed = '<tool_call>{"name":"calculator","arguments":</tool_call>'
        calculator_call = '<tool_call>{"name":"calculator","arguments":{"expression":"4 + 1"}}</tool_call>'
        orchestrator = self._orchestrator([roll, malformed, calculator_call, "The result is 5."])

        with patch("tools.random_tools.random.randint", return_value=4):
            reply = orchestrator.generate_tool_aware_reply(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "Roll once, then use calculator to add one."},
                ],
                turn_number=1,
            )

        self.assertEqual(reply, "The result is 5.")
        self.assertEqual([entry.ok for entry in orchestrator.last_tool_ledger], [True, False, True])
        self.assertEqual(orchestrator.last_tool_ledger[0].payload["data"]["value"], 4)
        self.assertEqual(orchestrator.last_tool_ledger[1].payload["error"]["code"], "unknown_tool")
        final_ledger = orchestrator.generation_inputs[-1][-1]["content"]
        self.assertIn('"status": "failed"', final_ledger)
        self.assertIn("Failed entries are not usable results", final_ledger)

    def test_discovery_cannot_override_dedicated_structured_result(self):
        registry = ToolRegistry()
        schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        registry.register(ToolDefinition(
            "search_web", "Search.", lambda: {"snippet_rate": "0.8617"}, schema,
            provenance="discovery",
        ))
        registry.register(ToolDefinition(
            "currency_exchange", "Convert.", lambda: {"rate": "0.85705", "converted_amount": "85.705"}, schema,
            provenance="dedicated",
        ))
        calls = (
            '<tool_call>{"name":"search_web","arguments":{}}</tool_call>'
            '<tool_call>{"name":"currency_exchange","arguments":{}}</tool_call>'
        )
        grounded_currency_call = '<tool_call>{"name":"currency_exchange","arguments":{}}</tool_call>'
        orchestrator = self._orchestrator(
            [calls, grounded_currency_call, "Using the currency result, 100 USD converts to 85.705 EUR."],
            tool_manager=ToolManager(registry),
        )

        reply = orchestrator.generate_tool_aware_reply(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "Use search_web for context and currency_exchange for conversion."},
            ],
            turn_number=1,
        )

        self.assertEqual(reply, "Using the currency result, 100 USD converts to 85.705 EUR.")
        ledger = orchestrator.generation_inputs[-1][-1]["content"]
        self.assertIn('"provenance": "discovery"', ledger)
        self.assertIn('"provenance": "dedicated"', ledger)
        self.assertIn('"converted_amount": "85.705"', ledger)
        self.assertIn("must not override overlapping dedicated fields", ledger)

    def test_ordinary_conversation_remains_a_single_generation(self):
        orchestrator = self._orchestrator(["Hello! Nice to see you."])

        messages, reply, _, _ = orchestrator.process_turn(
            "Hello!",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "Hello! Nice to see you.")
        self.assertEqual(len(orchestrator.generation_inputs), 1)
        self.assertIsNone(orchestrator.last_tool_execution)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
