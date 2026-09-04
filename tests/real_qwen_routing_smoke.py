"""Non-destructive real-Needle-2 smoke test for intent routing.

This module is intentionally outside unittest discovery. It loads Needle 2 but
never constructs a memory manager, document index, or SQLite connection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent_classifier import IntentClassifier, IntentDecision
from orchestrator import ConversationOrchestrator


MODEL_ID = "Cactus-Compute/needle2"


def as_dict(decision: IntentDecision) -> dict[str, bool]:
    return {
        "memory_read": decision.memory_read,
        "memory_write": decision.memory_write,
        "document_read": decision.document_read,
        "tool_use": decision.tool_use,
        "general_chat": decision.general_chat,
    }


def main() -> int:
    classifier = IntentClassifier(logger=lambda message: None)
    orchestrator = ConversationOrchestrator(
        object(),
        object(),
        memory=object(),
        intent_classifier=classifier,
        logger=lambda message: None,
    )
    semantic_classifier: IntentClassifier = classifier

    cases = [
        ("My name is Anish.", IntentDecision(False, True, False, False)),
        ("What is my name?", IntentDecision(True, False, False, False)),
        ("I live in Hyderabad now.", IntentDecision(False, True, False, False)),
        ("Where do I live?", IntentDecision(True, False, False, False)),
        ("I like coding in Python.", IntentDecision(False, True, False, False)),
        ("What programming language do I like?", IntentDecision(True, False, False, False)),
        ("I recently started coding in TypeScript.", IntentDecision(False, True, False, False)),
        ("What have I been coding in recently?", IntentDecision(True, False, False, False)),
        ("Python is still my favorite.", IntentDecision(False, True, False, False)),
        ("What is my favorite programming language?", IntentDecision(True, False, False, False)),
        ("What is the capital of India?", IntentDecision(False, False, False, True)),
    ]
    records: list[dict] = []
    failed = False
    for message, expected in cases:
        raw_semantic = semantic_classifier.classify(message, [])
        final_needle = orchestrator.classify_intent(message, [])
        passed = final_needle == expected
        failed = failed or not passed
        records.append(
            {
                "message": message,
                "raw_needle": as_dict(raw_semantic),
                "final_needle": as_dict(final_needle),
                "expected": as_dict(expected),
                "sources": dict(orchestrator.last_intent_sources),
                "passed": passed,
            }
        )

    contextual_cases = [
        (
            "What about meals?",
            [
                {"role": "user", "content": "What does the travel policy say about hotels?"},
                {"role": "assistant", "content": "The hotel limit is $220."},
            ],
            IntentDecision(False, False, True, False),
        ),
        (
            "Update your knowledge accordingly.",
            [
                {"role": "user", "content": "I live in Pune now."},
                {"role": "assistant", "content": "Got it."},
            ],
            IntentDecision(False, True, False, False),
        ),
    ]
    for message, history, expected in contextual_cases:
        raw_semantic = semantic_classifier.classify(message, history)
        final_needle = orchestrator.classify_intent(message, history)
        passed = final_needle == expected
        failed = failed or not passed
        records.append(
            {
                "message": message,
                "context": history,
                "raw_needle": as_dict(raw_semantic),
                "final_needle": as_dict(final_needle),
                "expected": as_dict(expected),
                "sources": dict(orchestrator.last_intent_sources),
                "passed": passed,
            }
        )

    print(json.dumps({"model": MODEL_ID, "passed": not failed, "cases": records}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
