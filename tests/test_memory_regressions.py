from __future__ import annotations

import hashlib
import gc
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from intent_classifier import IntentDecision
from memory_core import OfflineMemoryManager
from orchestrator import ConversationOrchestrator, DEFAULT_SYSTEM_PROMPT
from tests.test_intent_and_orchestration import FakeTokenizer, QueueOrchestrator, StaticClassifier


class FakeEmbedModel:
    def encode(self, text):
        if isinstance(text, list):
            return np.asarray([self.encode(item) for item in text], dtype=np.float32)
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        return np.asarray([(byte - 127.5) / 127.5 for byte in digest[:32]], dtype=np.float32)


class GroundedSequenceOrchestrator(ConversationOrchestrator):
    def __init__(self, *args, extraction_outputs=None, **kwargs):
        self.extraction_outputs = dict(extraction_outputs or {})
        self.extraction_inputs = []
        super().__init__(*args, **kwargs)

    def generate_reply(self, messages, **overrides):
        if messages and "You extract only durable user facts" in str(messages[0].get("content", "")):
            source = str(messages[-1].get("content", ""))
            self.extraction_inputs.append(source)
            return self.extraction_outputs.get(source, "None")
        return "Acknowledged."


class MemoryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "memory.db")
        self.memory = OfflineMemoryManager(self.db_path, embed_model=FakeEmbedModel())

    def tearDown(self):
        del self.memory
        gc.collect()
        self.temp_dir.cleanup()

    def _rows(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            return conn.execute(
                "SELECT id, entity, relation, value, fact_sentence, is_active, invalidated_at FROM global_archive ORDER BY id"
            ).fetchall()

    def test_entity_contract_is_user_only_and_not_substring_based(self):
        self.assertEqual(self.memory.normalize_entity("user"), "user")
        self.assertEqual(self.memory.normalize_entity("the user"), "user")
        self.assertEqual(self.memory.normalize_entity("Anish"), "")
        self.assertEqual(self.memory.normalize_entity("user_profile"), "")

    def test_structured_duplicate_write_is_rejected_without_rewriting_history(self):
        self.assertTrue(self.memory.add_fact_with_resolution("user", "lives_in", "Pune", "The user moved to Pune."))
        self.assertFalse(
            self.memory.add_fact_with_resolution(
                "the user",
                "resides_in",
                "  PUNE  ",
                "Pune is now the user's home city.",
            )
        )

        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], 1)

    def test_canonical_alias_conflict_is_invalidated_and_history_preserved(self):
        vector = self.memory.embed_model.encode("The user's name is Anish.").tobytes()
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO global_archive (entity, relation, value, fact_sentence, embedding) VALUES (?, ?, ?, ?, ?)",
                ("user", "nameself", "Anish", "The user's name is Anish.", vector),
            )

        self.assertTrue(self.memory.add_fact_with_resolution("user", "name", "Alex", "The user's name is now Alex."))

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][5], 0)
        self.assertIsNotNone(rows[0][6])
        self.assertEqual(rows[1][3], "Alex")
        self.assertEqual(rows[1][5], 1)

    def test_semantically_similar_prose_does_not_block_a_changed_value(self):
        class ConstantEmbedModel:
            def encode(self, text):
                if isinstance(text, list):
                    return np.asarray([self.encode(item) for item in text], dtype=np.float32)
                return np.ones(8, dtype=np.float32)

        self.memory.embed_model = ConstantEmbedModel()
        self.assertTrue(
            self.memory.add_fact_with_resolution(
                "user",
                "favorite_programming_language",
                "Java",
                "The user's favorite programming language is Java.",
            )
        )
        self.assertTrue(
            self.memory.add_fact_with_resolution(
                "user",
                "favorite_programming_language",
                "Go",
                "The user's favorite programming language is Go.",
            )
        )

        self.assertEqual(
            [(row[3], row[5]) for row in self._rows()],
            [("Java", 0), ("Go", 1)],
        )

    def test_multi_fact_message_stores_every_supported_fact(self):
        router_output = "\n".join(
            [
                "user | lives_in | Mumbai | The user lives in Mumbai.",
                "user | works_in | finance | The user works in finance.",
                "user | prefers_beverage | coffee | The user prefers coffee.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Understood.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            "I live in Mumbai, work in finance, and prefer coffee.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        active = [(row[2], row[3]) for row in self._rows() if row[5] == 1]
        self.assertEqual(
            active,
            [("lives_in", "Mumbai"), ("works_in", "finance"), ("prefers_beverage", "coffee")],
        )

    def test_assertion_inside_question_is_still_extracted(self):
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Hello Anish.", "user | name | Anish | The user's name is Anish."],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, True)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            "My name is Anish, what can you tell me?",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual([(row[2], row[3]) for row in self._rows()], [("name", "Anish")])

    def test_named_people_are_not_coerced_into_user_relationships(self):
        router_output = "\n".join(
            [
                "user | lives_in | Mumbai | Anish lives in Mumbai.",
                "user | lives_in | Amsterdam | Alex lives in Amsterdam.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            "Anish lives in Mumbai and Alex lives in Amsterdam.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(self._rows(), [])

    def test_mixed_self_and_third_party_statement_stores_only_self_fact(self):
        router_output = "\n".join(
            [
                "user | lives_in | Mumbai | The user lives in Mumbai.",
                "user | lives_in | Amsterdam | Alex lives in Amsterdam.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            "I live in Mumbai and Alex lives in Amsterdam.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual([(row[2], row[3]) for row in self._rows()], [("lives_in", "Mumbai")])

    def test_exact_structured_queries_do_not_mix_unrelated_relations_or_people(self):
        self.memory.add_fact_with_resolution("user", "name", "Anish", "The user's name is Anish.")
        self.memory.add_fact_with_resolution("user", "lives_in", "Mumbai", "The user lives in Mumbai.")
        self.memory.add_fact_with_resolution("user", "works_in", "finance", "The user works in finance.")

        name_context = self.memory.get_orchestrated_context("What is my name?")
        self.assertIn("Anish", name_context)
        self.assertNotIn("Mumbai", name_context)
        self.assertNotIn("finance", name_context)

        work_context = self.memory.get_orchestrated_context("Where do I work?")
        self.assertIn("finance", work_context)
        self.assertNotIn("Mumbai", work_context)
        self.assertNotIn("Anish", work_context)

        anish_context = self.memory.get_orchestrated_context("Where does Anish live?")
        self.assertIn("Anish", anish_context)
        self.assertIn("Mumbai", anish_context)

        self.assertEqual(self.memory.get_orchestrated_context("Where does Alex live?"), "")

        reverse_context = self.memory.get_orchestrated_context("Who lives in Mumbai?")
        self.assertIn("Anish", reverse_context)
        self.assertIn("Mumbai", reverse_context)

    def test_read_only_exact_lookup_does_not_promote_or_touch_cache(self):
        self.memory.add_fact_with_resolution("user", "name", "Anish", "The user's name is Anish.")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM primary_cache")
        self.memory.retrieval_mutates_cache = False

        context = self.memory.get_orchestrated_context("What is my name?")

        self.assertIn("Anish", context)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM primary_cache").fetchone()[0], 0)

    def test_programming_qualified_broad_recall_is_scoped_but_profile_recall_stays_hybrid(self):
        self.memory.add_fact_with_resolution(
            "user",
            "favorite_programming_language",
            "Python",
            "The user's favorite programming language is Python.",
        )
        self.memory.add_fact_with_resolution(
            "user",
            "recently_codes_in",
            "TypeScript",
            "The user has recently been coding in TypeScript.",
        )
        self.memory.add_fact_with_resolution("user", "lives_in", "Pune", "The user lives in Pune.")
        self.memory.add_fact_with_resolution(
            "user", "prefers_beverage", "tea", "The user prefers tea."
        )

        context = self.memory.get_orchestrated_context(
            "What kinds of things do you remember about my programming preferences?"
        )

        self.assertIn("Python", context)
        self.assertIn("TypeScript", context)
        self.assertNotIn("Pune", context)
        self.assertNotIn("tea", context)
        self.assertEqual(self.memory.last_retrieval_stats["structured_hits"], 2)

        profile = self.memory.get_orchestrated_context(
            "What kinds of things do you remember about me?"
        )
        self.assertIn("Pune", profile)
        self.assertIn("tea", profile)
        self.assertEqual(self.memory.last_retrieval_stats["structured_hits"], 0)

    def test_favorite_and_recent_programming_languages_remain_distinct(self):
        turns = [
            (
                "I like coding in Python.",
                "user | favorite_programming_language | Python | The user's favorite programming language is Python.",
            ),
            (
                "I have also started coding a lot in TypeScript.",
                "user | recently_codes_in | TypeScript | The user has recently been coding in TypeScript.",
            ),
            (
                "Actually, Python is still my favorite.",
                "user | favorite_programming_language | Python | Python is still the user's favorite programming language.",
            ),
        ]
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        for turn_number, (user_input, router_output) in enumerate(turns, start=1):
            orchestrator = QueueOrchestrator(
                FakeTokenizer(),
                object(),
                self.memory,
                generated=["Acknowledged.", router_output],
                intent_classifier=StaticClassifier(IntentDecision(False, True, False, True)),
                logger=lambda message: None,
            )
            messages, _, _, _ = orchestrator.process_turn(
                user_input,
                messages,
                turn_number=turn_number,
            )

        self.memory.add_fact_with_resolution("user", "lives_in", "Hyderabad", "The user lives in Hyderabad.")
        active = [(row[2], row[3]) for row in self._rows() if row[5] == 1]
        self.assertEqual(
            active,
            [
                ("favorite_programming_language", "Python"),
                ("recently_codes_in", "TypeScript"),
                ("lives_in", "Hyderabad"),
            ],
        )

        favorite = self.memory.get_orchestrated_context("What is my favorite programming language?")
        self.assertIn("Python", favorite)
        self.assertNotIn("TypeScript", favorite)

        liked = self.memory.get_orchestrated_context("What programming language do I like?")
        self.assertIn("Python", liked)
        self.assertNotIn("TypeScript", liked)

        recent = self.memory.get_orchestrated_context("What have I been coding in recently?")
        self.assertIn("TypeScript", recent)
        self.assertNotIn("Python", recent)

        combined = self.memory.get_orchestrated_context(
            "Where do I live now and what language have I been coding in recently?"
        )
        self.assertIn("Hyderabad", combined)
        self.assertIn("TypeScript", combined)
        self.assertNotIn("Python", combined)

    def test_duplicate_prose_detection_does_not_merge_distinct_language_relations(self):
        sentence = "The user codes in TypeScript."
        self.assertTrue(self.memory.add_fact_with_resolution("user", "codes_in", "TypeScript", sentence))
        self.assertTrue(
            self.memory.add_fact_with_resolution(
                "user",
                "recently_codes_in",
                "TypeScript",
                sentence,
            )
        )

        active = [(row[2], row[3]) for row in self._rows() if row[5] == 1]
        self.assertEqual(active, [("codes_in", "TypeScript"), ("recently_codes_in", "TypeScript")])

    def test_same_message_preference_change_canonicalizes_and_current_value_wins(self):
        source = (
            "I live in Pune. I recently moved, and now I live in Hyderabad. "
            "My coworker Rahul still lives in Pune. My preferred programming language used to be Python, "
            "but I now prefer TypeScript."
        )
        # Deliberately emit the current value first and use plausible temporal
        # relation variants, as a small router model may do.
        router_output = "\n".join(
            [
                "user | currently_prefers_programming_language | TypeScript | The user now prefers TypeScript.",
                "user | lives_in | Pune | The user lived in Pune.",
                "user | previous_preferred_programming_language | Python | Python was previously the user's preferred programming language.",
                "user | lives_in | Hyderabad | The user now lives in Hyderabad.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        rows = self._rows()
        self.assertEqual(
            [(row[2], row[3], row[5]) for row in rows],
            [
                ("favorite_programming_language", "Python", 0),
                ("lives_in", "Pune", 0),
                ("favorite_programming_language", "TypeScript", 1),
                ("lives_in", "Hyderabad", 1),
            ],
        )
        self.assertIsNotNone(rows[0][6])

        current = self.memory.get_orchestrated_context("What is my preferred programming language?")
        self.assertIn("TypeScript", current)
        self.assertNotIn("Python", current)

    def test_short_language_value_has_token_aware_temporal_priority(self):
        source = "My preferred programming language used to be Python, but now I prefer R."
        router_output = "\n".join(
            [
                "user | changed_preference | R | The user now prefers R.",
                "user | previous_preference | Python | The user previously preferred Python.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(
            [(row[3], row[5]) for row in self._rows()],
            [("Python", 0), ("R", 1)],
        )

    def test_changed_from_transition_orders_old_before_current_even_when_router_reverses_them(self):
        source = "I changed my preferred programming language from Java to Go."
        router_output = "\n".join(
            [
                "user | favorite_programming_language | Go | The user's language is Go.",
                "user | favorite_programming_language | Java | The user's language was Java.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(
            [(row[3], row[5]) for row in self._rows()],
            [("Java", 0), ("Go", 1)],
        )

    def test_source_temporal_wording_overrides_contradictory_router_paraphrases(self):
        source = "I used to prefer Java, but now I prefer Go."
        router_output = "\n".join(
            [
                "user | favorite_programming_language | Go | Go was previously preferred.",
                "user | favorite_programming_language | Java | Java is now preferred.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(
            [(row[3], row[5]) for row in self._rows()],
            [("Java", 0), ("Go", 1)],
        )

    def test_relation_disambiguation_uses_candidate_meaning_without_collapsing_specific_labels(self):
        transition = (
            "My preferred programming language used to be Python, but I now prefer TypeScript."
        )
        self.assertEqual(
            self.memory.canonicalize_extracted_relation(
                "prefers",
                source_text=transition,
                fact_sentence="The user now prefers TypeScript.",
                value="TypeScript",
            ),
            "favorite_programming_language",
        )
        self.assertEqual(
            self.memory.canonicalize_extracted_relation(
                "codes_in",
                source_text=transition,
                fact_sentence="The user now prefers TypeScript.",
                value="TypeScript",
            ),
            "favorite_programming_language",
        )
        self.assertEqual(
            self.memory.canonicalize_extracted_relation(
                "codes_in",
                source_text="I code in Python because I like it.",
                fact_sentence="The user codes in Python.",
                value="Python",
            ),
            "codes_in",
        )
        self.assertEqual(
            self.memory.canonicalize_extracted_relation(
                "recently_codes_in",
                source_text="I currently code in TypeScript because I prefer it.",
                fact_sentence="The user currently codes in TypeScript.",
                value="TypeScript",
            ),
            "recently_codes_in",
        )
        self.assertEqual(
            self.memory.canonicalize_extracted_relation(
                "prefers_beverage",
                source_text="I prefer coffee whenever I use the Python programming language.",
                fact_sentence="The user prefers coffee.",
                value="coffee",
            ),
            "prefers_beverage",
        )

    def test_programming_relation_family_supports_generic_broad_and_history_queries(self):
        self.memory.add_fact_with_resolution(
            "user",
            "favorite_programming_language",
            "Java",
            "Java was the user's favorite programming language.",
        )
        self.memory.add_fact_with_resolution(
            "user",
            "favorite_programming_language",
            "Go",
            "Go is now the user's favorite programming language.",
        )

        generic = self.memory.get_orchestrated_context("What programming language do I code in?")
        self.assertIn("Go", generic)
        self.assertNotIn("Java", generic)
        self.assertEqual(
            self.memory.last_retrieval_stats["structured_candidates"],
            ["codes_in", "recently_codes_in", "favorite_programming_language"],
        )

        broad = self.memory.get_orchestrated_context("What do you know about my programming?")
        self.assertIn("Go", broad)
        self.assertNotIn("Java", broad)

        history = self.memory.get_orchestrated_context(
            "What do you remember about my programming language history?"
        )
        self.assertIn("Current favorite programming language: Go.", history)
        self.assertIn("Previous favorite programming language: Java.", history)
        self.assertEqual(self.memory.last_retrieval_stats["historical_considered"], 1)

        favorite_history = self.memory.get_orchestrated_context(
            "What was my favorite programming language before?"
        )
        self.assertIn("Current favorite programming language: Go.", favorite_history)
        self.assertIn("Previous favorite programming language: Java.", favorite_history)
        self.assertEqual(
            self.memory.last_retrieval_stats["structured_candidates"],
            ["favorite_programming_language"],
        )

        coding_history = self.memory.get_orchestrated_context("What did I code in before?")
        self.assertEqual(coding_history, "")
        self.assertEqual(self.memory.last_retrieval_stats["structured_candidates"], ["codes_in"])

    def test_non_programming_language_and_coding_attribute_queries_do_not_inject_a_language(self):
        self.memory.add_fact_with_resolution(
            "user",
            "favorite_programming_language",
            "Go",
            "The user's favorite programming language is Go.",
        )

        self.assertEqual(self.memory.get_orchestrated_context("What language do I speak?"), "")
        self.assertEqual(self.memory.get_orchestrated_context("What coding style do I prefer?"), "")

    def test_third_party_programming_preference_is_not_stored_as_the_users(self):
        source = "I prefer tea. My coworker Rahul prefers Java."
        router_output = "\n".join(
            [
                "user | prefers_beverage | tea | The user prefers tea.",
                "user | favorite_programming_language | Java | Rahul prefers Java.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual([(row[2], row[3]) for row in self._rows()], [("prefers_beverage", "tea")])

    def test_short_values_and_new_sentences_cannot_leak_first_person_grounding(self):
        source = (
            "I prefer tea. My coworker Rahul likes R. Rahul really likes Java. "
            "I heard Mina likes Go."
        )
        router_output = "\n".join(
            [
                "user | prefers_beverage | tea | The user prefers tea.",
                "user | favorite_programming_language | R | Rahul likes R.",
                "user | favorite_programming_language | Java | Rahul really likes Java.",
                "user | favorite_programming_language | Go | Mina likes Go.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual([(row[2], row[3]) for row in self._rows()], [("prefers_beverage", "tea")])

    def test_value_led_first_person_preference_is_grounded(self):
        cases = [
            ("Python", "Python is my favorite programming language."),
            ("Go", "Go is now my preferred programming language."),
            ("Rust", "Rust is still my favorite."),
        ]
        for value, source in cases:
            with self.subTest(source=source):
                valid, _, reason = self.memory.assess_fact_candidate(
                    "user",
                    "favorite_programming_language",
                    value,
                    source,
                    source_text=source,
                )
                self.assertTrue(valid, reason)

    def test_grounding_checks_all_clauses_when_third_party_and_user_share_a_value(self):
        source = "My coworker Rahul likes Java, but I prefer Java too."
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=[
                "Noted.",
                "user | favorite_programming_language | Java | The user prefers Java.",
            ],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual(
            [(row[2], row[3], row[5]) for row in self._rows()],
            [("favorite_programming_language", "Java", 1)],
        )

    def test_dotted_values_are_not_split_at_internal_periods(self):
        cases = [
            ("VB.NET", "I prefer VB.NET.", "The user prefers VB.NET."),
            ("Python 3.12", "I prefer Python 3.12.", "The user prefers Python 3.12."),
        ]
        for value, source, sentence in cases:
            with self.subTest(value=value):
                valid, _, reason = self.memory.assess_fact_candidate(
                    "user",
                    "favorite_programming_language",
                    value,
                    sentence,
                    source_text=source,
                )
                self.assertTrue(valid, reason)

        self.assertEqual(
            self.memory.fact_temporal_priority(
                "I used to prefer VB.NET, but now I prefer Python 3.12.",
                "VB.NET",
            ),
            0,
        )
        self.assertEqual(
            self.memory.fact_temporal_priority(
                "I used to prefer VB.NET, but now I prefer Python 3.12.",
                "Python 3.12",
            ),
            2,
        )

    def test_programming_memory_survives_when_assertion_is_not_in_live_or_compressed_history(self):
        self.memory.add_fact_with_resolution(
            "user",
            "favorite_programming_language",
            "TypeScript",
            "TypeScript is now the user's favorite programming language.",
        )
        orchestrator = GroundedSequenceOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            compression_enabled=True,
            max_context_tokens=80,
            keep_recent_turns=1,
            intent_classifier=StaticClassifier(IntentDecision(False, False, False, True)),
            logger=lambda message: None,
        )
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Tell me about recursion in detail."},
            {"role": "assistant", "content": "Recursion is a function calling itself. " * 12},
            {"role": "user", "content": "Now explain iteration."},
            {"role": "assistant", "content": "Iteration repeats a block. " * 12},
        ]
        self.assertNotIn("TypeScript", " ".join(message["content"] for message in messages))

        _, _, memory_context, _ = orchestrator.process_turn(
            "What programming language do I code in?",
            messages,
            turn_number=3,
        )

        self.assertIn("TypeScript", memory_context)

    def test_unsupported_relation_log_includes_raw_and_canonical_labels(self):
        logs = []
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=[
                "Noted.",
                "user | changed_response_style | concise | The user prefers concise responses.",
            ],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=logs.append,
        )

        orchestrator.process_turn(
            "I prefer concise responses.",
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        rejection = next(message for message in logs if "unsupported relation" in message)
        self.assertIn("raw_relation='changed_response_style'", rejection)
        self.assertIn("canonical_relation='changed_response_style'", rejection)

    def test_relation_canonicalization_does_not_mix_cues_from_separate_facts(self):
        source = "I use Python as my programming language. I prefer concise answers."
        router_output = "\n".join(
            [
                "user | codes_in | Python | The user uses Python as a programming language.",
                "user | response_style | concise | The user prefers concise answers.",
            ]
        )
        orchestrator = QueueOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            generated=["Noted.", router_output],
            intent_classifier=StaticClassifier(IntentDecision(False, True, False, False)),
            logger=lambda message: None,
        )

        orchestrator.process_turn(
            source,
            [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            turn_number=1,
        )

        self.assertEqual([(row[2], row[3]) for row in self._rows()], [("codes_in", "Python")])

    def test_hybrid_orchestration_routes_and_grounds_the_real_smoke_sequence(self):
        extraction_outputs = {
            "My name is Anish.": "user | name | Anish | The user's name is Anish.",
            "I live in Hyderabad now.": "user | lives_in | Hyderabad | The user lives in Hyderabad.",
            "I like coding in Python.": (
                "user | favorite_programming_language | Python | The user's favorite programming language is Python."
            ),
            "I recently started coding in TypeScript.": "\n".join(
                [
                    "user | recently_codes_in | TypeScript | The user recently started coding in TypeScript.",
                    "user | favorite_programming_language | TypeScript | TypeScript is the user's favorite language.",
                    "user | lives_in | Hyderabad | The user lives in Hyderabad.",
                    "user | favorite_programming_language | Python | The user's favorite language is Python.",
                ]
            ),
            "Python is still my favorite.": (
                "user | favorite_programming_language | Python | Python is still the user's favorite programming language."
            ),
            "I live in Pune now.": "\n".join(
                [
                    "user | lives_in | Pune | The user lives in Pune.",
                    "user | favorite_programming_language | Python | The user's favorite language is Python.",
                ]
            ),
        }
        document_calls = []
        read_calls = []
        original_read = self.memory.get_orchestrated_context

        def tracked_read(query):
            read_calls.append(query)
            return original_read(query)

        self.memory.get_orchestrated_context = tracked_read
        orchestrator = GroundedSequenceOrchestrator(
            FakeTokenizer(),
            object(),
            self.memory,
            extraction_outputs=extraction_outputs,
            intent_classifier=StaticClassifier(IntentDecision(False, False, False, True)),
            document_lookup=lambda query: document_calls.append(query),
            logger=lambda message: None,
        )
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        def turn(text):
            nonlocal messages
            result = orchestrator.process_turn(text, messages, turn_number=len(messages))
            messages = result[0]
            return result

        turn("My name is Anish.")
        self.assertIn("Anish", turn("What is my name?")[2])
        turn("I live in Hyderabad now.")
        self.assertIn("Hyderabad", turn("Where do I live?")[2])
        turn("I like coding in Python.")
        self.assertIn("Python", turn("What programming language do I like?")[2])

        before_recent = len(self._rows())
        turn("I recently started coding in TypeScript.")
        newly_written = self._rows()[before_recent:]
        self.assertEqual([(row[2], row[3]) for row in newly_written], [("recently_codes_in", "TypeScript")])
        recent_context = turn("What have I been coding in recently?")[2]
        self.assertIn("TypeScript", recent_context)
        self.assertNotIn("Python", recent_context)

        turn("Python is still my favorite.")
        favorite_context = turn("What is my favorite programming language?")[2]
        self.assertIn("Python", favorite_context)
        self.assertNotIn("TypeScript", favorite_context)
        self.assertIn("TypeScript", turn("What have I been coding in recently?")[2])

        reads_before_general = len(read_calls)
        turn("What is the capital of India?")
        self.assertEqual(len(read_calls), reads_before_general)
        self.assertEqual(document_calls, [])

        contextual_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "I like Python."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "I live in Pune now."},
            {"role": "assistant", "content": "Got it."},
        ]
        orchestrator.process_turn(
            "Update your knowledge accordingly.",
            contextual_history,
            turn_number=20,
        )
        self.assertEqual(orchestrator.extraction_inputs[-1], "I live in Pune now.")
        active = [(row[2], row[3]) for row in self._rows() if row[5] == 1]
        self.assertIn(("lives_in", "Pune"), active)
        self.assertIn(("favorite_programming_language", "Python"), active)
        self.assertIn(("recently_codes_in", "TypeScript"), active)


if __name__ == "__main__":
    unittest.main()
