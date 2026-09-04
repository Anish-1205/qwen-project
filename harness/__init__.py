"""Public contracts for the model-agnostic AI harness."""

from .runner import (
    DEFAULT_SYSTEM_PROMPT,
    DocumentRetrievalResult,
    HarnessRunner,
    ConversationOrchestrator,
    OrchestrationOutcome,
    RetrievalMetadata,
    RoutingDecision,
    RunRequest,
    RunResult,
    RunState,
    SEARCH_CONFIGURATION_REQUIRED_RESPONSE,
    TraceEvent,
    strip_first_line,
    strip_speaker_tags,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DocumentRetrievalResult",
    "HarnessRunner",
    "ConversationOrchestrator",
    "OrchestrationOutcome",
    "RetrievalMetadata",
    "RoutingDecision",
    "RunRequest",
    "RunResult",
    "RunState",
    "SEARCH_CONFIGURATION_REQUIRED_RESPONSE",
    "TraceEvent",
    "strip_first_line",
    "strip_speaker_tags",
]
