from __future__ import annotations

import unittest

from intent_classifier import ConfidenceTier, DeterministicIntentRouter, IntentClassifier, IntentDecision
from harness import (
    ConversationOrchestrator,
    DEFAULT_SYSTEM_PROMPT,
    DocumentRetrievalResult,
    OrchestrationOutcome,
)
from tools import ToolDefinition, ToolManager, ToolRegistry
from models import GenerationResult, ModelBackend, ModelCapabilities, ModelSpec


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(str(message.get("content", "")) for message in messages)

    def __call__(self, prompt, *args, **kwargs):
        return {"input_ids": list(range(len(str(prompt).split())))}


class StubBackend(ModelBackend):
    def __init__(self, model_name="test/stub"):
        self._spec = ModelSpec(
            "stub", "Stub", model_name, "test",
            ModelCapabilities(tool_schemas=True, tool_messages=True),
        )

    @property
    def spec(self):
        return self._spec

    def load(self):
        pass

    def generate(self, request):
        raise AssertionError("unexpected backend generation")

    def count_tokens(self, messages):
        return sum(len(str(message.get("content", "")).split()) for message in messages)


class FakeMemory:
    def __init__(self):
        self.read_calls = []
        self.router_calls = []
        self.assessed_sources = []
        self.stored = []
        self.last_retrieval_stats = {"tier1_hits": 1, "tier2_promoted": 0, "tier3_scanned": 1, "facts": ["name=Anish"]}

    def get_orchestrated_context(self, query):
        self.read_calls.append(query)
        return "Background Profile Info:\n- Confirmed name: Anish."

    def build_router_messages(self, text):
        self.router_calls.append(text)
        return [{"role": "user", "content": text}]

    def normalize_entity(self, entity):
        return "user" if entity.strip().lower() == "user" else ""

    def normalize_relation(self, relation):
        return relation.strip().lower()

    def assess_fact_candidate(self, entity, relation, value, fact_sentence, *, source_text=None):
        self.assessed_sources.append(source_text)
        return True, "user", "accepted"

    def add_fact_with_resolution(self, entity, relation, value, fact_sentence):
        self.stored.append((entity, relation, value, fact_sentence))
        return True


class StaticClassifier:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def classify(self, user_input, messages):
        self.calls.append((user_input, list(messages)))
        return self.decision


class QueueNeedleSelector:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def select(self, messages, *, schemas):
        self.calls.append((list(messages), list(schemas)))
        return self.outputs.pop(0)


class QueueOrchestrator(ConversationOrchestrator):
    def __init__(self, *args, generated=None, **kwargs):
        self.generated = list(generated or [])
        self.generation_inputs = []
        if args and not isinstance(args[0], ModelBackend):
            if len(args) < 3:
                raise TypeError("legacy test construction requires tokenizer, model, and memory")
            args = (StubBackend(), args[2], *args[3:])
        super().__init__(*args, **kwargs)

    def generate_reply(self, messages, **overrides):
        self.generation_inputs.append([dict(message) for message in messages])
        if not self.generated:
            raise AssertionError("unexpected model generation")
        return self.generated.pop(0)


class IntentClassifierTests(unittest.TestCase):
    def test_strict_parse_and_controlled_repair(self):
        outputs = iter([
            "```json\n{}\n```",
            '{"memory_read":true,"memory_write":false,"document_read":false,"tool_use":false,"general_chat":false}',
        ])
        classifier = IntentClassifier(lambda messages, **kwargs: next(outputs), logger=lambda message: None)

        decision = classifier.classify("What is my name?", [])

        self.assertEqual(decision, IntentDecision(True, False, False, False))
        self.assertFalse(classifier.last_used_fallback)

    def test_malformed_output_uses_legacy_fallback(self):
        classifier = IntentClassifier(lambda messages, **kwargs: "not json", logger=lambda message: None)

        decision = classifier.classify("anything", [])

        self.assertEqual(decision, IntentDecision.legacy_fallback())
        self.assertTrue(classifier.last_used_fallback)

    def test_recent_context_excludes_system_retrieval_context(self):
        captured = []

        def generate(messages, **kwargs):
            captured.append(messages)
            return '{"memory_read":false,"memory_write":false,"document_read":true,"tool_use":false,"general_chat":false}'

        classifier = IntentClassifier(generate, logger=lambda message: None)
        history = [
            {"role": "system", "content": "SECRET RETRIEVED MEMORY AND DOCUMENT CHUNKS"},
            {"role": "user", "content": "What does the travel policy say about hotels?"},
            {"role": "assistant", "content": "The hotel limit is $220."},
        ]

        classifier.classify("What about meals?", history)
        prompt = captured[0][1]["content"]
        self.assertIn("travel policy", prompt)
        self.assertIn("What about meals?", prompt)
        self.assertNotIn("SECRET RETRIEVED", prompt)


class DeterministicIntentRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = DeterministicIntentRouter()

    def test_obvious_profile_reads_and_writes_are_complete_without_qwen(self):
        expected = {
            "My name is Anish.": (False, True, False, False),
            "I live in Hyderabad now.": (False, True, False, False),
            "I like Python.": (False, True, False, False),
            "Python is my favorite language.": (False, True, False, False),
            "I recently started coding in TypeScript.": (False, True, False, False),
            "I work in finance.": (False, True, False, False),
            "Remember that I prefer concise answers.": (False, True, False, False),
            "My coworker Rahul prefers Java.": (False, False, False, True),
            "What is my name?": (True, False, False, False),
            "Where do I live?": (True, False, False, False),
            "What programming language do I like?": (True, False, False, False),
            "What have I been coding in recently?": (True, False, False, False),
            "What do you remember about me?": (True, False, False, False),
            "What is the capital of India?": (False, False, False, True),
        }
        for message, flags in expected.items():
            with self.subTest(message=message):
                evidence = self.router.analyze(message, [])
                self.assertTrue(evidence.complete)
                self.assertEqual(
                    (evidence.memory_read, evidence.memory_write, evidence.document_read, evidence.general_chat),
                    flags,
                )

    def test_contextual_update_resolves_only_most_recent_durable_user_assertion(self):
        history = [
            {"role": "user", "content": "I like Python."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Where is Pune?"},
            {"role": "assistant", "content": "In Maharashtra."},
            {"role": "user", "content": "I live in Pune now."},
            {"role": "assistant", "content": "Got it."},
        ]
        evidence = self.router.analyze("Update your knowledge accordingly.", history)
        self.assertTrue(evidence.memory_write)
        self.assertEqual(evidence.source_for("memory_write"), "contextual_deterministic")
        self.assertEqual(
            self.router.resolve_memory_write_source("Update your knowledge accordingly.", history),
            "I live in Pune now.",
        )

    def test_question_is_a_high_confidence_non_write_even_if_qwen_disagrees(self):
        classifier = StaticClassifier(IntentDecision(True, True, True, True))
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=[],
            intent_classifier=classifier,
            logger=lambda message: None,
        )
        decision = orchestrator.classify_intent("What have I been coding in recently?", [])
        self.assertEqual(decision, IntentDecision(True, False, False, False))
        self.assertEqual(classifier.calls, [])

    def test_general_chat_remains_independent_for_mixed_requests(self):
        evidence = self.router.analyze("What is my name and explain recursion.", [])
        self.assertEqual(
            (evidence.memory_read, evidence.memory_write, evidence.document_read, evidence.general_chat),
            (True, False, False, True),
        )
        mixed = self.router.analyze(
            "Remember that I prefer concise answers and tell me what the travel policy says.",
            [],
        )
        self.assertTrue(mixed.memory_write)
        self.assertTrue(mixed.document_read)

    def test_webpage_routing_requires_current_turn_url_and_reading_action(self):
        actionable = (
            "Read https://example.com/report.",
            "Summarize https://example.org/a and https://example.net/b",
            "Compare https://example.com/one with https://example.com/two.",
        )
        non_actionable = (
            "The project website is https://example.com.",
            "Do not read https://example.com.",
            "How do webpage fetchers work?",
        )
        for message in actionable:
            with self.subTest(message=message):
                evidence = self.router.analyze(message, [])
                self.assertTrue(evidence.tool_use)
                self.assertEqual(evidence.source_for("tool_use"), "deterministic")
        for message in non_actionable:
            with self.subTest(message=message):
                evidence = self.router.analyze(message, [])
                self.assertFalse(evidence.tool_use)
                self.assertEqual(evidence.source_for("tool_use"), "deterministic")

    def test_current_turn_webpage_urls_are_conservatively_normalized(self):
        self.assertEqual(
            self.router.current_turn_webpage_urls(
                "Compare HTTPS://Example.COM/One?x=1 with https://example.org/two)."
            ),
            ("https://example.com/One?x=1", "https://example.org/two"),
        )
        self.assertEqual(self.router.current_turn_webpage_urls("Read that link."), ())

    def test_unfamiliar_standalone_question_delegates_ambiguous_flags_to_qwen(self):
        classifier = StaticClassifier(IntentDecision(False, False, True, False))
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=[],
            intent_classifier=classifier,
            logger=lambda message: None,
        )
        decision = orchestrator.classify_intent("What is the significance of the Antikythera mechanism?", [])
        self.assertEqual(decision, IntentDecision(False, False, True, False))
        self.assertEqual(len(classifier.calls), 1)

    def test_confidence_is_high_for_rules_and_medium_for_semantic_escalation(self):
        high = self.router.analyze("What is my name?", [])
        medium = self.router.analyze("What is the significance of the Antikythera mechanism?", [])

        self.assertEqual(high.confidence_for("memory_read"), ConfidenceTier.HIGH)
        self.assertEqual(medium.confidence_for("document_read"), ConfidenceTier.MEDIUM)

    def test_missing_action_target_is_very_low_confidence(self):
        evidence = self.router.analyze("Read that link.", [])

        self.assertIsNone(evidence.tool_use)
        self.assertEqual(evidence.confidence_for("tool_use"), ConfidenceTier.VERY_LOW)

    def test_changed_from_programming_assertions_route_to_memory_write(self):
        assertions = [
            "My programming language changed from Java to Go.",
            "My favorite programming language changed from Java to Go.",
            "I switched my preferred programming language from Java to Go.",
            "I changed programming languages from Java to Go.",
            "I switched from Java to Go for programming.",
            "My programming preference changed from Java to Go.",
            "My coding language changed from Java to Go.",
        ]
        for assertion in assertions:
            with self.subTest(assertion=assertion):
                evidence = self.router.analyze(assertion, [])
                self.assertTrue(evidence.complete)
                self.assertEqual(
                    (evidence.memory_read, evidence.memory_write, evidence.document_read, evidence.general_chat),
                    (False, True, False, False),
                )

        unrelated = self.router.analyze("I changed my job from programming to management.", [])
        self.assertEqual(
            (unrelated.memory_read, unrelated.memory_write, unrelated.document_read, unrelated.general_chat),
            (False, False, False, True),
        )

    def test_current_assertion_outranks_deictic_memory_suffix(self):
        assertions = [
            "I've started coding mostly in Rust lately. Remember that.",
            "I've started coding mostly in Rust lately.",
            "I code mostly in TypeScript now.",
            "I switched from Python to Rust for programming.",
            "I live in Pune now. Remember that.",
            "My preferred language is Go now. Save that.",
        ]

        for assertion in assertions:
            with self.subTest(assertion=assertion):
                evidence = self.router.analyze(assertion, [])
                self.assertTrue(evidence.memory_write)
                self.assertEqual(evidence.confidence_for("memory_write"), ConfidenceTier.HIGH)
                self.assertEqual(evidence.source_for("memory_write"), "deterministic")
                self.assertFalse(self.router.is_contextual_memory_command(assertion))

    def test_assertion_free_deictic_memory_command_resolves_prior_fact(self):
        history = [
            {"role": "user", "content": "I prefer tea."},
            {"role": "assistant", "content": "Understood."},
        ]

        evidence = self.router.analyze("Remember that.", history)

        self.assertTrue(evidence.memory_write)
        self.assertEqual(evidence.confidence_for("memory_write"), ConfidenceTier.HIGH)
        self.assertEqual(evidence.source_for("memory_write"), "contextual_deterministic")
        self.assertEqual(self.router.resolve_memory_write_source("Remember that.", history), "I prefer tea.")

    def test_mixed_assertion_and_same_turn_weather_reference_route_independently(self):
        cases = [
            ("I moved to Bangalore recently, and can you also check the weather there?", True, "Bangalore"),
            ("I live in Pune now. What's the weather there?", True, "Pune"),
            ("I'm visiting Tokyo tomorrow; check the weather there.", False, "Tokyo"),
            ("I moved to Bangalore recently.", True, None),
        ]

        for message, memory_write, location in cases:
            with self.subTest(message=message):
                evidence = self.router.analyze(message, [])
                self.assertFalse(evidence.memory_read)
                self.assertEqual(evidence.memory_write, memory_write)
                self.assertEqual(evidence.tool_use, location is not None)
                self.assertEqual(self.router.resolve_weather_reference(message, []), location)
                if memory_write:
                    self.assertEqual(evidence.confidence_for("memory_write"), ConfidenceTier.HIGH)
                if location:
                    self.assertEqual(evidence.confidence_for("tool_use"), ConfidenceTier.HIGH)

    def test_unresolved_weather_reference_is_very_low_confidence(self):
        evidence = self.router.analyze("Can you check the weather there?", [])

        self.assertIsNone(evidence.tool_use)
        self.assertEqual(evidence.confidence_for("tool_use"), ConfidenceTier.VERY_LOW)

    def test_lowercase_explicit_weather_locations_are_resolved_deterministically(self):
        cases = {
            "hows the weather in bengaluru ?": "Bengaluru",
            "what is the weather in kolkata": "Kolkata",
            "weather in new york right now": "New York",
        }

        for prompt, place in cases.items():
            with self.subTest(prompt=prompt):
                evidence = self.router.analyze(prompt, [])
                self.assertTrue(evidence.tool_use)
                self.assertEqual(evidence.confidence_for("tool_use"), ConfidenceTier.HIGH)
                self.assertEqual(self.router.resolve_weather_reference(prompt, []), place)

    def test_implicit_organization_knowledge_questions_route_to_documents(self):
        queries = [
            (
                "I am a remote employee. What equipment does the company provide me, what security requirements "
                "apply to my laptop, and who should I contact about device security?"
            ),
            "What leave allowance does my employer provide?",
            "Who handles device security at our company?",
            "What are our laptop security requirements?",
            "What equipment do we provide remote employees?",
            "Who is our device-security contact?",
        ]
        for query in queries:
            with self.subTest(query=query):
                evidence = self.router.analyze(query, [])
                self.assertTrue(evidence.complete)
                self.assertEqual(
                    (evidence.memory_read, evidence.memory_write, evidence.document_read, evidence.general_chat),
                    (False, False, True, False),
                )
                self.assertEqual(evidence.reason_for("document_read"), "organization-specific knowledge request")

        generic = self.router.analyze("How do companies typically equip remote employees?", [])
        self.assertIsNone(generic.document_read)

        public_company_questions = [
            "Is the company Tesla a good choice for my portfolio?",
            "What internal process does Python use for garbage collection?",
            "I am a remote employee. What equipment does the company Apple sell to customers?",
        ]
        for query in public_company_questions:
            with self.subTest(query=query):
                evidence = self.router.analyze(query, [])
                self.assertIsNot(evidence.document_read, True)

        mixed = self.router.analyze("I live in Pune. When was the company Apple founded?", [])
        self.assertTrue(mixed.complete)
        self.assertEqual(
            (mixed.memory_read, mixed.memory_write, mixed.document_read, mixed.general_chat),
            (False, True, False, True),
        )


class OrchestrationRoutingTests(unittest.TestCase):
    def _run(self, message, decision, *, history=None, document_result=None, router_output="None"):
        memory = FakeMemory()
        document_calls = []

        def lookup(query):
            document_calls.append(query)
            return document_result or DocumentRetrievalResult()

        generated = ["answer"] + ([router_output] if decision.memory_write else [])
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            memory,
            generated=generated,
            intent_classifier=StaticClassifier(decision),
            document_lookup=lookup,
            logger=lambda message: None,
        )
        messages = list(history or [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}])
        result = orchestrator.process_turn(message, messages, turn_number=1)
        return orchestrator, memory, document_calls, result

    def test_required_subsystem_gating_cases(self):
        cases = [
            ("What is the capital of India?", IntentDecision(False, False, False, True), 0, 0, 0),
            ("What is my name?", IntentDecision(True, False, False, False), 1, 0, 0),
            ("Do you remember where I live?", IntentDecision(True, False, False, False), 1, 0, 0),
            ("I moved to Pune.", IntentDecision(False, True, False, False), 0, 0, 1),
            ("What does the travel policy say?", IntentDecision(False, False, True, False), 0, 1, 0),
            ("Check my files for VPN instructions.", IntentDecision(False, False, True, False), 0, 1, 0),
            (
                "Remember that I prefer concise answers and tell me what the travel policy says.",
                IntentDecision(False, True, True, False),
                0,
                1,
                1,
            ),
            ("What is my name, and what is the capital of France?", IntentDecision(True, False, False, True), 1, 0, 0),
        ]
        for message, decision, memory_reads, document_reads, memory_writes in cases:
            with self.subTest(message=message):
                _, memory, documents, _ = self._run(message, decision)
                self.assertEqual(len(memory.read_calls), memory_reads)
                self.assertEqual(len(documents), document_reads)
                self.assertEqual(len(memory.router_calls), memory_writes)

    def test_contextual_followups_route_to_the_intended_subsystem(self):
        doc_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "What does the travel policy say about hotels?"},
            {"role": "assistant", "content": "The hotel limit is $220."},
        ]
        _, memory, documents, _ = self._run(
            "What about meals?",
            IntentDecision(False, False, True, False),
            history=doc_history,
        )
        self.assertFalse(memory.read_calls)
        self.assertEqual(documents, ["What about meals?"])

        memory_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "I live in Pune now."},
            {"role": "assistant", "content": "Understood."},
        ]
        _, memory, documents, _ = self._run(
            "Update your knowledge accordingly.",
            IntentDecision(False, True, False, False),
            history=memory_history,
        )
        self.assertEqual(memory.router_calls, ["I live in Pune now."])
        self.assertEqual(memory.assessed_sources, [])
        self.assertFalse(documents)

    def test_general_turn_has_no_memory_framing_in_prompt(self):
        _, _, _, result = self._run(
            "What is the capital of India?",
            IntentDecision(False, False, False, True),
        )
        messages = result[0]
        self.assertNotIn("confirmed information about the user", messages[0]["content"].lower())
        self.assertNotIn("Background Profile Info", messages[0]["content"])

    def test_malformed_classifier_fallback_cannot_turn_a_question_into_a_write(self):
        memory = FakeMemory()
        documents = []
        classifier = IntentClassifier(lambda messages, **kwargs: "invalid", logger=lambda message: None)
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            memory,
            generated=["answer"],
            intent_classifier=classifier,
            document_lookup=lambda query: documents.append(query) or DocumentRetrievalResult(),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            "What about that?",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(memory.read_calls, ["What about that?"])
        self.assertEqual(memory.router_calls, [])
        self.assertEqual(documents, ["What about that?"])

    def test_risky_very_low_route_asks_before_any_side_effect(self):
        memory = FakeMemory()
        document_calls = []
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            memory,
            generated=[],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            document_lookup=lambda query: document_calls.append(query) or DocumentRetrievalResult(),
            logger=lambda message: None,
        )

        _, reply, memory_context, _ = orchestrator.process_turn(
            "Update your knowledge accordingly.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(orchestrator.last_routing_decision.outcome, OrchestrationOutcome.ASK_USER)
        self.assertIn("what specific information", reply.lower())
        self.assertEqual(memory_context, "")
        self.assertFalse(memory.read_calls)
        self.assertFalse(memory.router_calls)
        self.assertFalse(document_calls)
        self.assertFalse(orchestrator.last_tool_executions)

    def test_inline_durable_assertion_never_reaches_semantic_router_or_clarification(self):
        classifier = StaticClassifier(IntentDecision(False, False, False, True))
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["Noted.", "None"],
            intent_classifier=classifier,
            logger=lambda message: None,
        )

        _, reply, _, _ = orchestrator.process_turn(
            "I've started coding mostly in Rust lately. Remember that.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "Noted.")
        self.assertEqual(classifier.calls, [])
        self.assertEqual(orchestrator.last_routing_decision.outcome, OrchestrationOutcome.ACT)
        write_evidence = orchestrator.last_routing_decision.evidence_for("memory_write")
        self.assertTrue(write_evidence.value)
        self.assertEqual(write_evidence.confidence, ConfidenceTier.HIGH)
        self.assertEqual(write_evidence.source, "deterministic")

    def test_same_turn_location_is_forced_into_weather_tool_arguments(self):
        observed_places = []
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "weather",
                "Get weather.",
                lambda place: observed_places.append(place) or {"location": place},
                {
                    "type": "object",
                    "properties": {"place": {"type": "string", "minLength": 1}},
                    "required": ["place"],
                    "additionalProperties": False,
                },
            )
        )
        needle = QueueNeedleSelector(["[]"])
        classifier = StaticClassifier(IntentDecision(True, False, False, True))
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["Bangalore weather returned.", "None"],
            intent_classifier=classifier,
            needle_selector=needle,
            tool_manager=ToolManager(registry),
            logger=lambda message: None,
        )

        _, reply, _, _ = orchestrator.process_turn(
            "I moved to Bangalore recently, and can you also check the weather there?",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "Bangalore weather returned.")
        self.assertEqual(classifier.calls, [])
        self.assertEqual(observed_places, ["Bangalore"])
        self.assertEqual(orchestrator.last_tool_execution.validated_arguments, {"place": "Bangalore"})
        self.assertEqual(orchestrator.last_routing_decision.outcome, OrchestrationOutcome.SELECT_TOOL)
        self.assertEqual(len(needle.calls), 1)

    def test_explicit_weather_location_bypasses_failed_initial_model_selection(self):
        observed_places = []
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                "weather",
                "Get weather.",
                lambda place: observed_places.append(place) or {"location": place},
                {
                    "type": "object",
                    "properties": {"place": {"type": "string", "minLength": 1}},
                    "required": ["place"],
                    "additionalProperties": False,
                },
            )
        )
        needle = QueueNeedleSelector(["No additional tool call is needed."])
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["It is clear in Bangalore."],
            needle_selector=needle,
            tool_manager=ToolManager(registry),
            logger=lambda message: None,
        )

        _, reply, _, _ = orchestrator.process_turn(
            "Check the weather in Bangalore.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(reply, "It is clear in Bangalore.")
        self.assertEqual(observed_places, ["Bangalore"])
        self.assertEqual(len(needle.calls), 1)

    def test_unresolved_weather_location_clarifies_without_tool_execution(self):
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=[],
            intent_classifier=StaticClassifier(IntentDecision(False, False, False, True, True)),
            logger=lambda message: None,
        )

        _, reply, _, _ = orchestrator.process_turn(
            "Can you check the weather there?",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(orchestrator.last_routing_decision.outcome, OrchestrationOutcome.ASK_USER)
        self.assertIn("location", reply.lower())
        self.assertFalse(orchestrator.last_tool_executions)

    def test_injected_needle_selects_tool_and_main_model_synthesizes(self):
        needle = QueueNeedleSelector([
            '<tool_call>{"name":"calculator","arguments":{"expression":"2 + 2"}}</tool_call>',
            "[]",
        ])
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["The result is 4."],
            needle_selector=needle,
            logger=lambda message: None,
        )

        reply = orchestrator.generate_tool_aware_reply(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "Calculate 2 + 2."},
            ],
            turn_number=1,
        )

        self.assertEqual(reply, "The result is 4.")
        self.assertEqual(len(needle.calls), 2)
        self.assertTrue(orchestrator.last_tool_execution.ok)
        self.assertEqual(orchestrator.routing_metrics["needle_selections"], 2)

    def test_document_context_is_trimmed_to_configured_budget(self):
        blocks = [f"Source: file{index}.txt, lines 1-2\n" + ("word " * 55) for index in range(3)]
        result = DocumentRetrievalResult(
            context="\n\n".join(blocks),
            routed_relevant=True,
            retrieved_count=3,
            sources=["file0.txt", "file1.txt", "file2.txt"],
        )
        memory = FakeMemory()
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            memory,
            generated=["answer"],
            intent_classifier=StaticClassifier(IntentDecision(False, False, True, False)),
            document_lookup=lambda query: result,
            compression_enabled=True,
            max_context_tokens=160,
            logger=lambda message: None,
        )

        messages, _, _, trimmed = orchestrator.process_turn(
            "Read the file.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
            maintain_history=False,
        )

        self.assertLess(len(trimmed.context), len("\n\n".join(blocks)))
        self.assertLessEqual(orchestrator.count_tokens([messages[0], {"role": "user", "content": "Read the file."}]), 160)

    def test_response_prompt_sets_adaptive_conversational_behavior(self):
        classifier = StaticClassifier(IntentDecision(False, False, False, True))
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["acknowledged"],
            intent_classifier=classifier,
            logger=lambda message: None,
        )

        prompt = orchestrator.build_system_prompt("", "")
        self.assertIn("merely sharing or updating a fact", prompt)
        self.assertIn("do not turn the mentioned topic into a tutorial", prompt)
        self.assertIn("explicitly asks for detail", prompt)
        self.assertIn("Never recap unrelated personal facts", prompt)

        memory_prompt = orchestrator.build_system_prompt(
            "Background Profile Info:\n- Confirmed location: Mumbai.",
            "",
        )
        self.assertIn("active confirmed facts", memory_prompt)
        self.assertIn("without hedging", memory_prompt)
        self.assertIn("Use only facts that directly answer", memory_prompt)

    def test_implicit_company_question_reaches_document_lookup_without_semantic_override(self):
        query = (
            "I am a remote employee. What equipment does the company provide me, what security requirements "
            "apply to my laptop, and who should I contact about device security?"
        )
        document_result = DocumentRetrievalResult(
            context="Source: company_policy.txt, lines 1-4\nThe company provides a laptop.",
            routed_relevant=True,
            retrieved_count=1,
            sources=["company_policy.txt"],
        )

        orchestrator, memory, document_calls, result = self._run(
            query,
            IntentDecision(False, False, False, True),
            document_result=document_result,
        )

        self.assertEqual(orchestrator.intent_classifier.calls, [])
        self.assertEqual(document_calls, [query])
        self.assertEqual(memory.read_calls, [])
        self.assertIn("company_policy.txt", result[0][0]["content"])

    def test_explicit_user_formulas_and_fallbacks_are_authoritative_in_response_prompt(self):
        rules = (
            "Footage = End Station - Start Station. "
            "Qty Today = Qty Plug if provided; otherwise Qty Today = Footage."
        )
        history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": rules},
            {"role": "assistant", "content": "Qty Today is not applicable when Qty Plug is empty."},
        ]
        orchestrator, _, _, _ = self._run(
            "Describe how to determine Footage and Qty Today for every activity.",
            IntentDecision(False, False, False, True),
            history=history,
        )

        response_messages = orchestrator.generation_inputs[-1]
        self.assertIn("Apply explicit user task rules", response_messages[0]["content"])
        self.assertIn("fallbacks", response_messages[0]["content"])
        self.assertIn("conflicting earlier assistant answers", response_messages[0]["content"])
        self.assertEqual(response_messages[-3]["content"], rules)
        self.assertEqual(response_messages[-1]["content"], "Describe how to determine Footage and Qty Today for every activity.")

    def test_compression_keeps_complete_turns_and_preserves_user_rules(self):
        rules = (
            "Footage = End Station - Start Station. "
            "Qty Today = Qty Plug if provided; otherwise Qty Today = Footage."
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=["Earlier greeting summarized."],
            keep_recent_turns=2,
            logger=lambda message: None,
        )
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "system", "content": "Summary of earlier conversation: The user introduced the task."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": rules},
            {"role": "assistant", "content": "Qty Today is not applicable when Qty Plug is empty."},
            {"role": "user", "content": "The stations are 100 and 250."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Calculate both values."},
        ]

        compressed = orchestrator.compress_context(messages)

        self.assertEqual(
            [message["role"] for message in compressed],
            ["system", "system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual(compressed[2]["content"], rules)
        self.assertEqual(compressed[-1]["content"], "Calculate both values.")
        summary_request = orchestrator.generation_inputs[0]
        self.assertIn("conditional rules", summary_request[0]["content"])
        self.assertIn("every fallback branch", summary_request[0]["content"])
        self.assertIn("Earlier summary:", summary_request[1]["content"])
        self.assertNotIn(rules, summary_request[1]["content"])

    def test_old_formula_rules_are_sent_to_the_summary_under_the_precedence_contract(self):
        rules = (
            "Footage = End Station - Start Station. "
            "Qty Today = Qty Plug if provided; otherwise Qty Today = Footage."
        )
        preserved_summary = (
            "The user's active rules are: Footage = End Station - Start Station; "
            "Qty Today = Qty Plug when provided, otherwise Qty Today = Footage."
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            FakeMemory(),
            generated=[preserved_summary, "Calculated."],
            compression_enabled=True,
            max_context_tokens=40,
            keep_recent_turns=1,
            intent_classifier=StaticClassifier(IntentDecision(False, False, False, True)),
            logger=lambda message: None,
        )
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": rules},
            {"role": "assistant", "content": "Qty Today is unavailable without Qty Plug."},
            {"role": "user", "content": "The activity runs from station 100 to 250."},
            {"role": "assistant", "content": "Understood."},
        ]

        compressed, _, _, _ = orchestrator.process_turn(
            "Describe how to determine both fields.",
            messages,
            turn_number=3,
        )

        summary_request = orchestrator.generation_inputs[0]
        response_request = orchestrator.generation_inputs[1]
        self.assertIn(rules, summary_request[1]["content"])
        self.assertIn("User rules", summary_request[0]["content"])
        self.assertIn("conflicting assistant interpretations", summary_request[0]["content"])
        self.assertIn(preserved_summary, compressed[1]["content"])
        self.assertIn(preserved_summary, response_request[1]["content"])
        self.assertEqual(response_request[-1]["content"], "Describe how to determine both fields.")
        self.assertEqual(
            [message["role"] for message in compressed],
            ["system", "system", "user", "assistant", "user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main()
