"""Classify turn intents with Needle 2 structured inference.

``DeterministicIntentRouter`` remains for a few text-extraction helpers used by
the orchestration layer, but it no longer participates in intent decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlsplit, urlunsplit


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
    _WEBPAGE_ACTION = re.compile(
        r"\b(?:fetch|open|read|extract|inspect|parse|load|scrape|check|analy[sz]e|summari[sz]e|compare)\b",
        re.I,
    )
    _WEATHER_REQUEST = re.compile(
        r"\b(?:current\s+weather"
        r"|weather(?:\s+(?:be\s+)?like)?(?:\s+(?:today|now|right\s+now|currently|tomorrow))?\s+(?:in|at|for)"
        r"|forecast\s+(?:in|at|for)"
        r"|(?:current\s+)?temperature(?:\s+(?:today|now|right\s+now|currently))?\s+(?:in|at|for)"
        r"|(?:[\w'-]+\s+){0,4}[\w'-]+'s\s+(?:current\s+)?temperature"
        r"|(?:current\s+(?:weather\s+)?conditions|weather\s+conditions)"
        r"(?:\s+(?:today|now|right\s+now|currently))?\s+(?:in|at|for)"
        r"|conditions\s+(?:today|now|right\s+now|currently)\s+(?:in|at|for)"
        r"|(?:chance|probability)\s+of\s+(?:rain|snow|precipitation)"
        r"(?:\s+(?:today|tomorrow|now|right\s+now|currently))?\s+(?:in|at|for)"
        r"|precipitation\s+probability(?:\s+(?:today|tomorrow|now|right\s+now|currently))?\s+(?:in|at|for))\b"
        r"|^\s*(?:weather|forecast)\s+(?!apis?\b|forecasting\b)(?:for\s+)?[\w .'-]+[?.!]*$"
        r"|\bwill\s+[\w .'-]+\s+(?:get|have)\s+(?:rain|snow)\b"
        r"|\b(?:is|will)\s+it\s+(?:rain(?:ing)?|snow(?:ing)?)\s+(?:in|at)\b"
        r"|\bhow\s+(?:hot|cold|warm|cool)\s+is\s+it\s+(?:in|at)\b"
        r"|\b(?:is|are)\s+there\s+(?:any\s+)?(?:rain|snow|precipitation)\s+(?:in|at)\b",
        re.I,
    )
    _CURRENCY_EXCHANGE_REQUEST = re.compile(
        r"\b(?:convert|exchange)\b[^?.!]{0,120}\b(?:to|into|for)\b"
        r"|\b(?:current|latest|today(?:'s)?)\s+(?:(?:foreign|currency)\s+)?exchange\s+rates?\b"
        r"|\b(?:exchange|conversion)\s+rates?\b[^?.!]{0,100}\b(?:for|from|between)\b"
        r"|\bhow\s+much\s+(?:is|are|would)\b[^?.!]{0,100}\d[^?.!]{0,100}\b(?:in|into)\b",
        re.I,
    )
    _CURRENCY_EXCHANGE_DISCUSSION = re.compile(
        r"^\s*(?:(?:what\s+(?:is|are)|define|explain)\s+(?:an?\s+|the\s+concept\s+of\s+)?"
        r"(?:(?:foreign|currency)\s+)?exchange\s+rates?"
        r"|(?:what\s+is|define|explain)\s+(?:the\s+concept\s+of\s+)?currency\s+conversion)\s*[?.!]*$"
        r"|^\s*(?:how|why)\s+do(?:es)?\b[^?.!]{0,100}\bexchange\s+rates?\b"
        r"[^?.!]{0,80}\b(?:work|change|fluctuate|vary)\b"
        r"|^\s*how\s+(?:do|can|could|should|would)\s+(?:i|we)\s+convert\b",
        re.I,
    )
    _WEB_SEARCH_REQUEST = re.compile(
        r"\bsearch\s+(?:the\s+)?(?:web|internet|online)\s+(?:for|about)\b"
        r"|\b(?:web|internet)\s+search\s+(?:for|about)\b"
        r"|\bfind\b[^?.!]{0,80}\b(?:information|sources?|results?)\b[^?.!]{0,60}"
        r"\b(?:online|on\s+the\s+(?:web|internet))\b"
        r"|\bfind\b[^?.!]{0,80}\b(?:online|on\s+the\s+(?:web|internet))\b"
        r"|\blook\s+up\s+(?:the\s+)?(?:latest|current|recent|today(?:'s)?)\b"
        r"|\blook\s+(?:it|this|that)\s+up\s+(?:online|on\s+the\s+(?:web|internet))\b",
        re.I,
    )
    _WEB_SEARCH_DISCUSSION = re.compile(
        r"^\s*(?:(?:what\s+is|define|explain)\s+(?:an?\s+|the\s+concept\s+of\s+)?"
        r"(?:web|internet|online)\s+search"
        r"|why\s+is\s+(?:web|internet|online)\s+search\s+useful)\s*[?.!]*$"
        r"|^\s*how\s+(?:do|can|could|should|would)\s+(?:i|we)\s+search\s+"
        r"(?:the\s+)?(?:web|internet|online)\b",
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
    _RANDOM_REQUEST = re.compile(
        r"(?:^\s*(?:please\s+)?roll\b[^?.!]{0,80}\bd(?:ie|ice)\b"
        r"|^\s*(?:can|could|would|will)\s+you\s+roll\b[^?.!]{0,80}\bd(?:ie|ice)\b"
        r"|\brandom\s+(?:number|integer)\b"
        r"|\bpick\s+(?:a\s+)?number\s+between\b)",
        re.I,
    )
    _NEGATED_TOOL_REQUEST = re.compile(
        r"^\s*(?:please\s+)?(?:do\s+not|don't|never)\s+"
        r"(?:fetch|search|look\s+up|open|read|extract|inspect|parse|load|scrape|check|analy[sz]e|summari[sz]e|list|show|find|select|filter|group|aggregate|sort|limit|total|sum|mean|average|count|calculate|compute|compare|roll|pick|generate|get)\b",
        re.I,
    )
    _TOOL_DISCUSSION = re.compile(
        r"^\s*how\s+(?:do|does|would)\b[^?.!]{0,120}\bwork\b"
        r"|^\s*why\s+(?:does|do)\s+it\s+(?:rain|snow)\b"
        r"|^\s*what\s+(?:is|are)\s+(?:the\s+)?(?:precipitation\s+probability|chance\s+of\s+(?:rain|snow))\s*[?.!]*$"
        r"|^\s*what\s+(?:is|are)\s+(?:an?\s+|the\s+)?spreadsheet(?:s)?\b"
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

    @classmethod
    def is_web_search_request(cls, text: str) -> bool:
        """Identify explicit web-search actions for routing and completion enforcement."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        return bool(
            cls._WEB_SEARCH_REQUEST.search(normalized)
            and not cls._WEB_SEARCH_DISCUSSION.search(normalized)
            and not cls._NEGATED_TOOL_REQUEST.search(normalized)
        )

    @staticmethod
    def normalize_current_turn_url(value: str) -> str | None:
        """Conservatively normalize an explicit URL for current-turn authorization."""
        if not isinstance(value, str):
            return None
        candidate = value.strip().strip("\"'").rstrip(".,;:!?")
        pairs = {")": "(", "]": "[", "}": "{"}
        while candidate and candidate[-1] in pairs:
            closer = candidate[-1]
            if candidate.count(closer) <= candidate.count(pairs[closer]):
                break
            candidate = candidate[:-1]
        try:
            parsed = urlsplit(candidate)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                return None
            parsed.port
        except (TypeError, ValueError):
            return None
        return urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, "")
        )

    @classmethod
    def current_turn_webpage_urls(cls, text: str) -> tuple[str, ...]:
        """Return only URLs explicitly authorized for page reading in this user turn."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if (
            not cls._WEBPAGE_ACTION.search(normalized)
            or cls._NEGATED_TOOL_REQUEST.search(normalized)
        ):
            return ()
        urls: list[str] = []
        for match in cls._URL.finditer(normalized):
            url = cls.normalize_current_turn_url(match.group(0))
            if url and url not in urls:
                urls.append(url)
        return tuple(urls)

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
            self.current_turn_webpage_urls(text)
            or self._WEATHER_REQUEST.search(text)
            or self._CURRENCY_EXCHANGE_REQUEST.search(text)
            or self.is_web_search_request(text)
            or (explicit_local_path and self._TOOL_ACTION.search(text))
            or directory_request
            or spreadsheet_request
            or self._CALCULATOR_REQUEST.search(text)
            or self._RANDOM_REQUEST.search(text)
        )
        negated_or_discussion = bool(
            self._NEGATED_TOOL_REQUEST.search(text)
            or self._TOOL_DISCUSSION.search(text)
            or self._CURRENCY_EXCHANGE_DISCUSSION.search(text)
            or self._WEB_SEARCH_DISCUSSION.search(text)
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
    """Use Needle 2 as the sole intent classifier for every routing flag."""

    REQUIRED_KEYS = {"memory_read", "memory_write", "document_read", "tool_use", "general_chat"}

    def __init__(
        self,
        needle_agent=None,
        *,
        logger: Callable[[str], None] = print,
        recent_message_limit: int = 4,
        recent_message_chars: int = 600,
        max_new_tokens: int = 128,
    ) -> None:
        self.logger = logger
        self.recent_message_limit = recent_message_limit
        self.recent_message_chars = recent_message_chars
        self.max_new_tokens = max_new_tokens
        self._agent = needle_agent
        self.last_used_fallback = False
        self.last_error = ""

    @staticmethod
    def _schemas() -> list[dict]:
        """Expose intents as Needle-native routes instead of asking it to emit prose JSON."""
        descriptions = {
            "memory_read": "Retrieve durable personal facts previously supplied by the user, such as their name, location, job, or preferences.",
            "memory_write": "Store or update a durable personal fact supplied by the user. Questions and facts solely about other people do not use this route.",
            "document_read": "Read local/uploaded files or internal organization knowledge such as company policies, benefits, procedures, and contacts.",
            "tool_use": "Perform requested calculations, current weather or currency lookup, web search, URL fetch, local file/spreadsheet operation, or random result.",
            "general_chat": "Answer greetings, writing or explanation requests, and general-knowledge questions conversationally.",
        }
        return [
            {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
            for name, description in descriptions.items()
        ]

    def _get_agent(self):
        if self._agent is None:
            try:
                import needle
            except ImportError as exc:
                raise RuntimeError(
                    "Needle 2 is required for intent classification; install cactus-needle"
                ) from exc
            self._agent = needle.Needle(tools=self._schemas())
        return self._agent

    @classmethod
    def parse_decision(cls, calls: list[dict]) -> IntentDecision:
        if not isinstance(calls, list):
            raise ValueError("Needle function_calls is not a list")
        selected = {call.get("name") for call in calls if isinstance(call, dict)}
        unknown = selected - cls.REQUIRED_KEYS
        if unknown:
            raise ValueError(f"Needle returned unknown intent routes: {sorted(unknown)}")
        # Needle's documented off-topic contract is an empty call list. In this
        # routing toolset, off-topic means no subsystem is needed and the normal
        # conversational model should answer.
        if not selected:
            selected.add("general_chat")
        return IntentDecision(**{key: key in selected for key in cls.REQUIRED_KEYS})

    def _recent_context(self, messages: Sequence[dict]) -> str:
        eligible = [message for message in messages if message.get("role") in {"user", "assistant"}]
        rendered: list[str] = []
        for message in eligible[-self.recent_message_limit :]:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = str(message.get("content", "")).strip()[: self.recent_message_chars]
            if content:
                rendered.append(f"{role}: {content}")
        return "\n".join(rendered) if rendered else "(none)"

    def classify(self, user_input: str, recent_messages: Sequence[dict]) -> IntentDecision:
        self.last_used_fallback = False
        self.last_error = ""
        try:
            agent = self._get_agent()
            agent.reset()
            context = self._recent_context(recent_messages)
            query = user_input if context == "(none)" else f"Previous conversation:\n{context}\n\n{user_input}"
            response = agent.complete(query, max_new_tokens=self.max_new_tokens)
            calls = response.get("function_calls", []) if isinstance(response, dict) else []
            return self.parse_decision(calls)
        except Exception as exc:
            self.last_used_fallback = True
            self.last_error = str(exc)
            self.logger(f"[Intent] Warning: Needle 2 classification failed; using safe fallback ({self.last_error}).")
            return IntentDecision.legacy_fallback()
