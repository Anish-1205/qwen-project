"""Classify independent turn intents with deterministic and model evidence.

The router resolves high-confidence lexical cases first and asks Qwen only for
flags that remain ambiguous, keeping routing predictable and inexpensive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class IntentDecision:
    memory_read: bool
    memory_write: bool
    document_read: bool
    general_chat: bool
    tool_use: bool = False

    @classmethod
    def legacy_fallback(cls) -> "IntentDecision":
        """Preserve the pre-classifier pipeline when classification is unavailable."""
        return cls(memory_read=True, memory_write=True, document_read=True, general_chat=True, tool_use=False)


@dataclass(frozen=True, slots=True)
class DeterministicIntentEvidence:
    """High-confidence per-flag evidence; ``None`` delegates that flag to Qwen."""

    memory_read: bool | None = None
    memory_write: bool | None = None
    document_read: bool | None = None
    general_chat: bool | None = None
    tool_use: bool | None = None
    sources: tuple[tuple[str, str], ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (self.memory_read, self.memory_write, self.document_read, self.general_chat, self.tool_use)
        )

    def source_for(self, flag: str) -> str:
        return dict(self.sources).get(flag, "semantic_llm")

    def reason_for(self, flag: str) -> str:
        return dict(self.reasons).get(flag, "semantic fallback")


class DeterministicIntentRouter:
    """Recognize only routing evidence that should not require a language model."""

    _QUESTION_START = re.compile(
        r"^\s*(?:what|where|who|which|when|why|how|do|does|did|is|are|am|can|could|would|will|have|has|tell)\b",
        re.I,
    )
    _SELF_REFERENCE = re.compile(r"\b(?:i|i'm|i've|me|my|mine|myself)\b", re.I)
    _PROFILE_CUES = re.compile(
        r"\b(?:name|called|live|reside|location|home|work|job|occupation|employed|code|coding|programming|language|favorite|favourite|prefer|like|recently|lately|beverage|drink|coffee|tea)\b",
        re.I,
    )
    _RECALL_CUES = re.compile(r"\b(?:remember|recall|know|known|told)\b", re.I)
    _DOCUMENT_CUES = re.compile(
        r"\b(?:document|documents|file|files|pdf|txt|uploaded|upload|policy|manual|guide|specification|spec|report|instructions|directory|playbook|knowledge\s+base|project\s+notes?|research\s+notes?)\b|\b[\w.-]+\.(?:pdf|txt)\b",
        re.I,
    )
    _STRONG_ORGANIZATION_REFERENCE = re.compile(
        r"\b(?:our|my)\s+(?:company|employer|organi[sz]ation|workplace)\b|\bcompany-wide\b"
        r"|\binternal\s+(?:company|employee|workplace|security|benefits?|equipment|requirements?)\b",
        re.I,
    )
    _CONTEXTUAL_ORGANIZATION_REFERENCE = re.compile(
        r"\b(?:this|the)\s+(?:company|employer|organi[sz]ation|workplace)\b",
        re.I,
    )
    _ORGANIZATION_KNOWLEDGE_CUES = re.compile(
        r"\b(?:provide|provided|equipment|benefits?|allowance|requirements?|required|security|laptop|device|contact|responsible|procedure|process|reimburse|entitled|eligible)\b",
        re.I,
    )
    _INTERNAL_POSSESSIVE_KNOWLEDGE = re.compile(
        r"\bour\s+(?:[\w-]+\s+){0,3}"
        r"(?:polic(?:y|ies)|rules?|requirements?|equipment|benefits?|allowance|limits?|"
        r"security|laptops?|devices?|contacts?|procedures?|process|reimbursement)\b",
        re.I,
    )
    _EMPLOYEE_CONTEXT = re.compile(
        r"\b(?:remote\s+employees?|employees?|staff|team\s+members?|at\s+work|"
        r"company\s+(?:laptop|device)|work\s+(?:laptop|device)|apply\s+to\s+me|"
        r"provide\s+me|who\s+should\s+i\s+contact)\b",
        re.I,
    )
    _INTERNAL_WE_REQUEST = re.compile(
        r"\bwe\b[^?.!]{0,80}\b(?:provide|require|allow|offer|reimburse)\b"
        r"[^?.!]{0,80}\b(?:employees?|staff|team\s+members?)\b",
        re.I,
    )
    _NAMED_PUBLIC_COMPANY_REFERENCE = re.compile(
        r"\b(?:the|this)\s+(?:company|employer|organi[sz]ation)\s+[A-Z][\w&.-]*\b"
    )
    _CONTEXT_REFERENCE = re.compile(
        r"\b(?:that|this|it|accordingly|the\s+other\s+one|what\s+about|the\s+thing|earlier)\b",
        re.I,
    )
    _MEMORY_ACTION = re.compile(r"\b(?:remember|update|store|save|keep|note)\b", re.I)
    _ORDINARY_REQUEST = re.compile(
        r"\b(?:explain|calculate|capital|weather|translate|write|summarize|compare|recursion|how|why)\b",
        re.I,
    )
    _CLEAR_GENERAL_CUE = re.compile(
        r"\b(?:capital|weather|calculate|arithmetic|recursion|translate|creative\s+writing|poem|joke)\b",
        re.I,
    )
    _EXPLICIT_LOCAL_PATH = re.compile(
        r"(?:[A-Za-z]:[\\/]|(?:^|\s)\.?\.?[\\/])[^\n\r\"']+|\b[\w .()-]+\.(?:txt|csv|json|docx|pdf|xlsx)\b",
        re.I,
    )
    _TOOL_ACTION = re.compile(r"\b(?:fetch|open|read|extract|inspect|parse|load|scrape|check|analy[sz]e|summari[sz]e|list|show|find|select|filter|group|aggregate|sort|limit|total|sum|mean|average|count|calculate|compute|compare|roll|pick|generate|give|get)\b", re.I)
    _URL = re.compile(r"https?://[^\s<>]+", re.I)
    _WEATHER_REQUEST = re.compile(
        r"\b(?:current\s+weather|weather\s+(?:in|at|for)|forecast\s+(?:in|at|for)|temperature\s+(?:in|at|for)|[\w'-]+'s\s+temperature)\b"
        r"|^\s*(?:weather|forecast)\s+(?!apis?\b|forecasting\b)(?:for\s+)?[\w .'-]+[?.!]*$"
        r"|\bwill\s+[\w .'-]+\s+(?:get|have)\s+(?:rain|snow)\b",
        re.I,
    )
    _DIRECTORY_REQUEST = re.compile(
        r"\b(?:list|show|find)\b[^?.!]{0,60}\b(?:files?|director(?:y|ies)|folders?)\b"
        r"|\b(?:what|which)\s+(?:\w+\s+){0,3}(?:files?|folders?|director(?:y|ies))\s+are\s+(?:here|in\s+(?:(?:this|that|the\s+current|current|local)\s+(?:folder|directory|project)|[A-Za-z]:[\\/]|\.{0,2}[\\/]))",
        re.I,
    )
    _DIRECTORY_DISCUSSION = re.compile(
        r"^\s*(?:what|which)\s+(?:files?|folders?|director(?:y|ies))\s+are\s+in\s+[^?.!]{0,80}\b(?:typical|standard|common|package|project|filesystem|hierarchy|kernel)\b"
        r"|^\s*(?:list|show|find)\b[^?.!]{0,100}\b(?:common|typical|standard|normally|usually|used\s+by)\b[^?.!]{0,100}\b(?:files?|folders?|director(?:y|ies)|package|project|filesystem|hierarchy|kernel)\b",
        re.I,
    )
    _DIRECTORY_PATH_REQUEST = re.compile(
        r"^\s*(?:please\s+)?(?:list|show|find|open)\s+(?:in\s+)?(?:~[\\/]|[\w.-]+[\\/])",
        re.I,
    )
    _SPREADSHEET_REQUEST = re.compile(r"\b(?:filter|group|aggregate|total|sum|mean|average|sort)\b[^?.!]{0,100}\b(?:csv|xlsx|spreadsheet|workbook|column|revenue)\b", re.I)
    _CALCULATOR_REQUEST = re.compile(
        r"(?:^\s*(?:please\s+)?(?:calculate|compute|evaluate|add|subtract|multiply|divide)\b"
        r"|^\s*(?:can|could|would|will)\s+you\s+(?:calculate|compute|evaluate)\b"
        r"|^\s*(?:use\s+)?(?:the\s+)?calculator\b"
        r"|\bwhat(?:\s+is|'s)\b[^?.!]{0,80}(?:\d\s*(?:[+*/%-]|\*\*)\s*\d)"
        r"|\bwhat(?:\s+is|'s)\s+(?:the\s+)?(?:sum|mean|average|minimum|maximum|count)\s+of\b)",
        re.I,
    )
    _RANDOM_REQUEST = re.compile(r"\b(?:roll\s+(?:a|the)?\s*d(?:ie|ice)|random\s+(?:number|integer)|pick\s+(?:a\s+)?number\s+between)\b", re.I)
    _NEGATED_TOOL_REQUEST = re.compile(
        r"^\s*(?:please\s+)?(?:do\s+not|don't|never)\s+"
        r"(?:fetch|open|read|extract|inspect|parse|load|scrape|check|analy[sz]e|summari[sz]e|list|show|find|select|filter|group|aggregate|sort|limit|total|sum|mean|average|count|calculate|compute|compare|roll|pick|generate|get)\b",
        re.I,
    )
    _TOOL_DISCUSSION = re.compile(
        r"^\s*how\s+(?:do|does|would)\b[^?.!]{0,120}\bwork\b"
        r"|^\s*how\s+(?:do|can|could|should|would)\s+(?:i|we)\b[^?.!]{0,80}\b(?:fetch|open|read|extract|inspect|parse|load|scrape|list|select|filter|group|aggregate|sort|limit|calculate)\b"
        r"|^\s*(?:show|tell|teach|explain)\s+(?:me\s+)?how\b[^?.!]{0,120}\bwork\b"
        r"|^\s*(?:can|could|would)\s+you\s+(?:explain|show|tell|teach)\b[^?.!]{0,80}\bhow\b[^?.!]{0,80}\b(?:fetch|open|read|extract|inspect|parse|load|scrape|list|select|filter|group|aggregate|sort|limit|calculate)\b"
        r"|^\s*should\s+(?:i|we)\b[^?.!]{0,80}\b(?:fetch|open|read|extract|inspect|parse|load|scrape|list|select|filter|group|aggregate|sort|limit|calculate)\b"
        r"|^\s*what\b[^?.!]{0,40}\b(?:filter|group|aggregate|sort|limit|calculation)\b[^?.!]{0,60}\bshould\s+(?:i|we)\s+use\b"
        r"|^\s*(?:explain|teach|show|tell)\b[^?.!]{0,80}\b(?:how\s+to|the\s+way\s+to)\b[^?.!]{0,80}\b(?:fetch|open|read|extract|inspect|parse|load|scrape|list|select|filter|group|aggregate|sort|calculate)\b"
        r"|^\s*(?:write|show|give)\b[^?.!]{0,60}\b(?:code|script)\b[^?.!]{0,60}\b(?:fetch|open|scrape|read|extract|inspect|parse|load|list|select|filter|group|aggregate|sort)\b",
        re.I,
    )

    @classmethod
    def is_question(cls, text: str) -> bool:
        normalized = (text or "").strip()
        return "?" in normalized or bool(cls._QUESTION_START.search(normalized))

    @classmethod
    def is_organization_knowledge_request(cls, text: str) -> bool:
        """Identify scoped internal-knowledge questions without requiring a filename."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        for segment in re.split(r"(?<=[.!?;])\s+", normalized):
            if not cls.is_question(segment) or cls._NAMED_PUBLIC_COMPANY_REFERENCE.search(segment):
                continue
            knowledge_cue = bool(cls._ORGANIZATION_KNOWLEDGE_CUES.search(segment))
            if (
                cls._INTERNAL_POSSESSIVE_KNOWLEDGE.search(segment)
                or cls._INTERNAL_WE_REQUEST.search(segment)
                or (cls._STRONG_ORGANIZATION_REFERENCE.search(segment) and knowledge_cue)
                or (
                    cls._CONTEXTUAL_ORGANIZATION_REFERENCE.search(segment)
                    and knowledge_cue
                    and cls._EMPLOYEE_CONTEXT.search(segment)
                )
            ):
                return True
        return False

    @staticmethod
    def is_programming_change_assertion(text: str) -> bool:
        """Recognize first-person old-to-new programming changes without naming values."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if re.search(
            r"\bmy\s+(?:friend|colleague|coworker|partner|manager|company|employer|team)\b",
            normalized,
            re.I,
        ):
            return False
        transition = bool(re.search(r"\bfrom\b[^.!?;]{1,120}\bto\b", normalized, re.I))
        programming_context = bool(
            re.search(r"\b(?:programming|coding)\s+(?:languages?|preference)\b", normalized, re.I)
            or re.search(r"\blanguages?\b[^.!?;]{0,50}\b(?:code|coding|programming)\b", normalized, re.I)
            or re.search(r"\b(?:for|when|while)\s+(?:coding|programming)\b", normalized, re.I)
        )
        first_person_change = bool(
            re.search(r"\bi\s+(?:have\s+)?(?:changed|switched)\b", normalized, re.I)
            or re.search(r"\bmy\b[^.!?;]{0,70}\b(?:changed|switched)\b", normalized, re.I)
        )
        return transition and programming_context and first_person_change

    @classmethod
    def _assertion_segments(cls, text: str) -> list[str]:
        segments = re.split(
            r"(?<=[.!?;])\s+|,\s*(?=(?:what|where|who|which|when|why|how|do|does|did|can|could|would|will|is|are|have|has)\b)",
            text or "",
            flags=re.I,
        )
        result: list[str] = []
        for segment in segments:
            cleaned = segment.strip()
            if not cleaned:
                continue
            explicit_inline_store = bool(
                cls._MEMORY_ACTION.search(cleaned)
                and re.search(r"\bthat\b", cleaned, flags=re.I)
                and cls._SELF_REFERENCE.search(cleaned)
            )
            if cls.is_question(cleaned) and not explicit_inline_store:
                continue
            result.append(cleaned)
        return result

    @classmethod
    def asserted_memory_relations(cls, text: str) -> tuple[str, ...]:
        relations: list[str] = []
        for segment in cls._assertion_segments(text):
            if re.search(r"\bmy\s+name\s+(?:is|is now|is still)\b|\b(?:i am|i'm)\s+(?:called|named)\b|\bcall\s+me\b", segment, re.I):
                relations.append("name")
            if re.search(r"\b(?:i|we)\s+(?:now\s+|currently\s+)?(?:live|reside)\b|\bi(?:'ve|\s+have)?\s+(?:just\s+|recently\s+)?moved\b|\bmy\s+(?:home|location|city)\s+is\b", segment, re.I):
                relations.append("lives_in")
            if re.search(r"\bi\s+(?:now\s+|currently\s+)?work\b|\bmy\s+(?:job|occupation|work)\s+is\b", segment, re.I):
                relations.append("works_in")
            if cls.is_programming_change_assertion(segment):
                relations.append("favorite_or_preference")
            if re.search(r"\bi\s+(?:still\s+)?(?:like|love|prefer)\b|\bmy\s+(?:favorite|favourite|preferred)\b|\bis\s+(?:still\s+)?my\s+(?:favorite|favourite|preferred)\b", segment, re.I):
                relations.append("favorite_or_preference")
            if re.search(r"\bi(?:'ve|\s+have)?\s+(?:also\s+)?(?:recently\s+|just\s+)?started\s+(?:coding|programming)|\bi\s+(?:recently|lately|currently)\s+(?:code|am\s+coding|have\s+been\s+coding)|\bi(?:'ve|\s+have)\s+been\s+coding", segment, re.I):
                relations.append("recently_codes_in")
            elif re.search(r"\bi\s+(?:code|program|develop)\b|\bi\s+(?:am|'m)\s+coding\b", segment, re.I):
                relations.append("codes_in")
            if re.search(r"\bi\s+(?:drink|prefer|like)\b.*\b(?:coffee|tea|beverage)\b|\bmy\s+(?:drink|beverage)\b", segment, re.I):
                relations.append("prefers_beverage")
            if (
                cls._MEMORY_ACTION.search(segment)
                and re.search(r"\bthat\b", segment, re.I)
                and cls._SELF_REFERENCE.search(segment)
            ):
                relations.append("explicit_memory_request")
        return tuple(dict.fromkeys(relations))

    @classmethod
    def is_contextual_memory_command(cls, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        return bool(
            cls._MEMORY_ACTION.search(normalized)
            and cls._CONTEXT_REFERENCE.search(normalized)
            and not cls.asserted_memory_relations(normalized)
        )

    @classmethod
    def resolve_memory_write_source(cls, text: str, recent_messages: Sequence[dict]) -> str | None:
        """Resolve a deictic update to one recent, durable user assertion only."""
        if not cls.is_contextual_memory_command(text):
            return text
        eligible = [message for message in recent_messages if message.get("role") == "user"]
        for message in reversed(eligible[-6:]):
            candidate = str(message.get("content", "")).strip()
            if (
                candidate
                and not cls.is_contextual_memory_command(candidate)
                and cls.asserted_memory_relations(candidate)
            ):
                return candidate
        return None

    @classmethod
    def _contextual_read_routes(cls, text: str, recent_messages: Sequence[dict]) -> tuple[str, ...]:
        """Inherit a clear subsystem topic for a deictic follow-up question."""
        if not cls.is_question(text) or not cls._CONTEXT_REFERENCE.search(text):
            return ()
        recent_users = [message for message in recent_messages if message.get("role") == "user"]
        for message in reversed(recent_users[-4:]):
            candidate = re.sub(r"\s+", " ", str(message.get("content", "")).strip())
            if not candidate or cls.is_contextual_memory_command(candidate):
                continue
            routes: list[str] = []
            if cls._DOCUMENT_CUES.search(candidate) or cls.is_organization_knowledge_request(candidate):
                routes.append("document_read")
            candidate_profile_question = bool(
                cls.is_question(candidate)
                and cls._SELF_REFERENCE.search(candidate)
                and cls._PROFILE_CUES.search(candidate)
            )
            if candidate_profile_question or cls.asserted_memory_relations(candidate):
                routes.append("memory_read")
            if not routes and not cls._CONTEXT_REFERENCE.search(candidate):
                routes.append("general_chat")
            return tuple(dict.fromkeys(routes))
        return ()

    def analyze(self, user_input: str, recent_messages: Sequence[dict]) -> DeterministicIntentEvidence:
        text = re.sub(r"\s+", " ", (user_input or "").strip())
        asserted = self.asserted_memory_relations(text)
        question = self.is_question(text)
        question_segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?;])\s+", text)
            if segment.strip() and self.is_question(segment)
        ]
        question_scope = " ".join(question_segments) if question_segments else text
        explicit_document_cue = bool(self._DOCUMENT_CUES.search(text))
        explicit_local_path = bool(self._EXPLICIT_LOCAL_PATH.search(text))
        local_directory_target = bool(
            explicit_local_path
            or re.search(r"\bhere\b|\b(?:this|that|the\s+current|current|local)\s+(?:folder|directory|project)\b", text, re.I)
        )
        organization_knowledge = self.is_organization_knowledge_request(text)
        document_cue = (explicit_document_cue and not explicit_local_path) or organization_knowledge
        directory_request = bool(self._DIRECTORY_REQUEST.search(text) or self._DIRECTORY_PATH_REQUEST.search(text))
        directory_discussion = bool(self._DIRECTORY_DISCUSSION.search(text) and not local_directory_target)
        spreadsheet_request = bool(self._SPREADSHEET_REQUEST.search(text))
        tool_request = bool(
            (self._URL.search(text) and self._TOOL_ACTION.search(text))
            or self._WEATHER_REQUEST.search(text)
            or (explicit_local_path and self._TOOL_ACTION.search(text))
            or directory_request
            or spreadsheet_request
            or self._CALCULATOR_REQUEST.search(text)
            or self._RANDOM_REQUEST.search(text)
        )
        negated_or_discussion = bool(
            self._NEGATED_TOOL_REQUEST.search(text)
            or self._TOOL_DISCUSSION.search(text)
            or directory_discussion
        )
        if negated_or_discussion:
            tool_request = False
        if directory_request or spreadsheet_request or directory_discussion:
            document_cue = organization_knowledge
        clear_general = bool(self._CLEAR_GENERAL_CUE.search(question_scope) or directory_discussion)
        contextual = bool(self._CONTEXT_REFERENCE.search(text))
        antecedent = self.resolve_memory_write_source(text, recent_messages) if self.is_contextual_memory_command(text) else None
        inherited_routes = self._contextual_read_routes(text, recent_messages)

        values: dict[str, bool | None] = {
            "memory_read": None,
            "memory_write": None,
            "document_read": None,
            "general_chat": None,
            "tool_use": None,
        }
        sources: dict[str, str] = {}
        reasons: dict[str, str] = {}

        def decide(flag: str, value: bool, reason: str, source: str = "deterministic") -> None:
            values[flag] = value
            sources[flag] = source
            reasons[flag] = reason

        profile_question = bool(
            question
            and self._SELF_REFERENCE.search(question_scope)
            and self._PROFILE_CUES.search(question_scope)
        )
        explicit_recall = bool(
            question
            and self._RECALL_CUES.search(question_scope)
            and re.search(r"\b(?:me|my|about\s+me)\b", question_scope, re.I)
        )
        if "memory_read" in inherited_routes:
            decide(
                "memory_read",
                True,
                "contextual question inherits the prior user-memory topic",
                "contextual_deterministic",
            )
        elif profile_question or explicit_recall:
            decide("memory_read", True, "direct question about supported user-profile facts")
        elif inherited_routes or asserted or antecedent or document_cue or not contextual:
            decide("memory_read", False, "no user-memory recall request")

        if asserted:
            decide("memory_write", True, "durable first-person assertion in current message")
        elif antecedent:
            decide(
                "memory_write",
                True,
                "contextual update resolved to the most recent durable user assertion",
                "contextual_deterministic",
            )
        elif question:
            decide("memory_write", False, "question contains no asserted or referenced durable fact")
        elif not contextual:
            decide("memory_write", False, "no durable first-person assertion")

        if "document_read" in inherited_routes:
            decide(
                "document_read",
                True,
                "contextual question inherits the prior local-document topic",
                "contextual_deterministic",
            )
        elif document_cue:
            reason = (
                "organization-specific knowledge request"
                if organization_knowledge and not explicit_document_cue
                else "explicit local-document or document-domain cue"
            )
            decide("document_read", True, reason)
        elif inherited_routes or asserted or profile_question or explicit_recall or antecedent or not question or clear_general:
            decide("document_read", False, "no document cue or unresolved document reference")

        mixed_general = bool(
            self._ORDINARY_REQUEST.search(text)
            and (profile_question or asserted)
        ) or bool(document_cue and re.search(r"\band\b.*\b(?:explain|calculate|capital|translate|write|compare)\b", text, re.I))
        mixed_general = mixed_general or bool(
            asserted and question and not profile_question and not document_cue
        )
        if "general_chat" in inherited_routes:
            decide(
                "general_chat",
                True,
                "contextual question inherits the prior general-chat topic",
                "contextual_deterministic",
            )
        elif inherited_routes:
            decide("general_chat", False, "contextual question is covered by the inherited subsystem topic")
        elif mixed_general:
            decide("general_chat", True, "message also contains an ordinary model request")
        elif profile_question or explicit_recall or asserted or antecedent or document_cue:
            decide("general_chat", False, "request is fully covered by routed subsystem intents")
        elif not contextual and (not question or clear_general):
            decide("general_chat", True, "ordinary conversation or general-knowledge request")

        if tool_request:
            decide("tool_use", True, "explicit request for an allowlisted utility action")
            for flag in ("memory_read", "memory_write", "document_read"):
                if values[flag] is None:
                    decide(flag, False, "utility request contains no evidence for this subsystem")
            if values["general_chat"] is None:
                decide("general_chat", False, "request is fully covered by tool use")
        elif (
            asserted or profile_question or explicit_recall or document_cue or clear_general
            or negated_or_discussion or (not question and not self._TOOL_ACTION.search(text))
        ):
            decide("tool_use", False, "no request to execute a utility tool")

        return DeterministicIntentEvidence(
            memory_read=values["memory_read"],
            memory_write=values["memory_write"],
            document_read=values["document_read"],
            general_chat=values["general_chat"],
            tool_use=values["tool_use"],
            sources=tuple(sources.items()),
            reasons=tuple(reasons.items()),
        )


class IntentClassifier:
    """Small deterministic Qwen pass that selects participating subsystems."""

    REQUIRED_KEYS = {"memory_read", "memory_write", "document_read", "tool_use", "general_chat"}

    def __init__(
        self,
        generate: Callable[..., str],
        *,
        logger: Callable[[str], None] = print,
        generation_kwargs: dict | None = None,
        recent_message_limit: int = 4,
        recent_message_chars: int = 600,
    ) -> None:
        self.generate = generate
        self.logger = logger
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 72,
            "do_sample": False,
        }
        self.recent_message_limit = recent_message_limit
        self.recent_message_chars = recent_message_chars
        self.last_used_fallback = False
        self.last_error = ""

    @classmethod
    def parse_decision(cls, raw: str) -> IntentDecision:
        candidate = (raw or "").strip()
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("classifier output is not a JSON object") from exc

        if not isinstance(payload, dict) or set(payload) != cls.REQUIRED_KEYS:
            raise ValueError("classifier output has an invalid key set")
        if any(type(payload[key]) is not bool for key in cls.REQUIRED_KEYS):
            raise ValueError("classifier flags must be JSON booleans")
        return IntentDecision(**payload)

    def _recent_context(self, messages: Sequence[dict]) -> str:
        eligible = [message for message in messages if message.get("role") in {"user", "assistant"}]
        rendered: list[str] = []
        for message in eligible[-self.recent_message_limit :]:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = str(message.get("content", "")).strip()[: self.recent_message_chars]
            if content:
                rendered.append(f"{role}: {content}")
        return "\n".join(rendered) if rendered else "(none)"

    def _classification_messages(self, user_input: str, recent_messages: Sequence[dict]) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You route the CURRENT user message for a local assistant. Return exactly one JSON object with five boolean keys: "
                    "memory_read, memory_write, document_read, tool_use, general_chat. The flags are independent. "
                    "Use recent conversation only to resolve references such as 'that' or 'what about meals'; never copy a prior turn's intent onto the current message. "
                    "memory_read means the current message asks for durable personal facts previously supplied by the speaker. "
                    "Set it even if the recent transcript appears to contain the answer; recent chat is not a substitute for confirmed memory. "
                    "memory_write means the current message supplies or updates a durable fact about the speaker, even when phrased as a casual statement, or explicitly asks "
                    "to remember/update a durable fact from the recent exchange. Facts solely about other named people are "
                    "not writable personal memory. A recall question is not a write merely because the preceding turn was an update. "
                    "document_read means local/uploaded files are needed, including a clear "
                    "follow-up to a file discussion. Organization-specific facts such as internal rules, employee benefits, "
                    "employer-provided resources, procedures, and responsible internal contacts also require document_read "
                    "even when no file is named; generic questions about companies do not. Words such as 'knowledge' or 'update' alone do not imply documents. "
                    "tool_use means the assistant must execute a registered utility: fetch a URL, get current weather, read/list a local path, analyze a spreadsheet, calculate arithmetic, or produce a die/random result. Discussion about those topics is not tool use. "
                    "general_chat means some part can be answered from ordinary conversation or model knowledge. Use recent "
                    "context only to resolve references. Examples: capital question => only general_chat; 'what is my name?' or "
                    "'where do I live?' or 'what is my favorite language?' => only memory_read; 'I moved to Pune', 'I live in Hyderabad now', "
                    "or 'I have started coding in TypeScript' => memory_write (and optionally general_chat); travel-policy question => only "
                    "document_read; name plus capital => memory_read and general_chat; remember a preference plus ask about "
                    "a policy => memory_write and document_read. Do not add prose or markdown."
                ),
            },
            {
                "role": "user",
                "content": f"Recent conversation:\n{self._recent_context(recent_messages)}\n\nCurrent message:\n{user_input}",
            },
        ]

    def classify(self, user_input: str, recent_messages: Sequence[dict]) -> IntentDecision:
        self.last_used_fallback = False
        self.last_error = ""
        messages = self._classification_messages(user_input, recent_messages)

        try:
            first_output = self.generate(messages, **self.generation_kwargs)
            return self.parse_decision(first_output)
        except Exception as first_exc:
            first_error = str(first_exc)

        try:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object with these keys and JSON boolean values only: "
                        "memory_read, memory_write, document_read, tool_use, general_chat. No markdown or prose."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The prior classification was invalid ({first_error}). Reclassify this message:\n"
                        f"{user_input}\n\nRecent context:\n{self._recent_context(recent_messages)}"
                    ),
                },
            ]
            repaired_output = self.generate(repair_messages, **self.generation_kwargs)
            return self.parse_decision(repaired_output)
        except Exception as repair_exc:
            self.last_used_fallback = True
            self.last_error = f"initial={first_error}; repair={repair_exc}"
            self.logger(f"[Intent] Warning: classifier validation failed; using legacy fallback ({self.last_error}).")
            return IntentDecision.legacy_fallback()
