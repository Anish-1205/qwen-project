from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from intent_classifier import IntentDecision
from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT
from tools import ToolCall, ToolManager, calculator


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
        return IntentDecision(False, False, False, True)


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
            ["roll_die", "random_number", "calculator", "fetch_webpage", "weather", "read_file", "list_directory", "analyze_spreadsheet"],
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
    def _orchestrator(self, generated):
        return QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=generated,
            intent_classifier=StaticClassifier(),
            logger=lambda message: None,
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
        self.assertEqual(followup[-2]["role"], "assistant")
        self.assertEqual(followup[-2]["tool_calls"][0]["function"]["name"], "calculator")
        self.assertEqual(followup[-1]["role"], "tool")
        self.assertEqual(json.loads(followup[-1]["content"])["data"], {"value": 100})
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertTrue(all("tools" in overrides for overrides in orchestrator.generation_overrides))

    def test_invalid_tool_request_is_not_executed_and_qwen_receives_the_error(self):
        call = '<tool_call>{"name":"shell","arguments":{"command":"dir"}}</tool_call>'
        orchestrator = self._orchestrator([call, "I cannot run that tool."])

        _, reply, _, _ = orchestrator.process_turn(
            "Calculate 2 + 2.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "I cannot run that tool.")
        self.assertFalse(orchestrator.last_tool_execution.ok)
        payload = json.loads(orchestrator.generation_inputs[1][-1]["content"])
        self.assertEqual(payload["error"]["code"], "unknown_tool")

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
