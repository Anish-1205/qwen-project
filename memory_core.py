"""Persist, retrieve, and temporally update durable user facts in SQLite.

The memory store is intentionally separate from document retrieval and chat
sessions. Model-extracted facts pass through Python validation before storage.
"""

import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app_paths import AGENT_MEMORY_DB_PATH


@dataclass(frozen=True, slots=True)
class StructuredRelationResolution:
    """A structured memory lookup plan derived from the query's meaning."""

    candidates: tuple[str, ...] = ()
    reason: str = ""
    selection: str = "all"
    include_history: bool = False
    unsupported_reason: str | None = None


@contextmanager
def _open_memory_db(db_path: str | Path):
    resolved_path = Path(db_path).expanduser()
    if str(db_path) != ":memory:":
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

class OfflineMemoryManager:
    retrieval_mutates_cache = True
    PROGRAMMING_RELATIONS = (
        "codes_in",
        "recently_codes_in",
        "favorite_programming_language",
    )

    def __init__(self, db_path: str | Path = AGENT_MEMORY_DB_PATH, *, embed_model=None):
        self.db_path = db_path
        self.primary_cache_max = 5
        self.last_retrieval_stats = {
            "tier1_hits": 0,
            "tier2_promoted": 0,
            "tier3_scanned": 0,
            "structured_hits": 0,
            "structured_candidates": [],
            "structured_reason": "",
            "active_considered": 0,
            "historical_considered": 0,
            "facts": [],
        }
        self._init_db()
        if embed_model is None:
            # BGE runs entirely on CPU to keep your RTX 3060 VRAM completely clear for Qwen
            print("[Status] Initializing CPU Embedding Model (BAAI/bge-small-en-v1.5)...")
            self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
            print("[Status] Embedding Model loaded successfully on CPU.")
        else:
            self.embed_model = embed_model

    def _log(self, message: str):
        print(f"[Memory] {message}")

    def normalize_entity(self, entity: str) -> str:
        """Canonicalize entity labels so list markers and articles do not create split identities."""
        normalized = re.sub(r"^\s*(?:fact\s*:\s*|\d+\.\s*|[-*]\s*)+", "", entity or "", flags=re.I)
        normalized = normalized.strip().lower()
        if normalized.startswith("the "):
            normalized = normalized[4:].strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return ""
        if normalized == "user":
            return "user"
        return ""

    def normalize_value(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    def _value_pattern(self, value: str):
        normalized_value = self.normalize_value(value)
        if not normalized_value:
            return None
        return re.compile(
            rf"(?<![\w+#&-])(?<!\w\.){re.escape(normalized_value)}(?![\w+#&-]|\.(?=\w))",
            re.I,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split prose without treating dots inside values such as versions as boundaries."""
        return re.split(r"(?:[!?;]+|\.(?=\s|$))", text or "", flags=re.I)

    def normalize_relation(self, relation: str) -> str:
        """Canonicalize relation labels so malformed router output resolves to stable storage keys."""
        normalized = re.sub(r"^\s*(?:fact\s*:\s*|\d+\.\s*|[-*]\s*)+", "", relation or "", flags=re.I)
        normalized = normalized.strip().lower()
        normalized = normalized.replace("-", "_")
        normalized = re.sub(r"\s+", "_", normalized)
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)

        alias_map = {
            "nameself": "name",
            "my_name": "name",
            "user_name": "name",
            "full_name": "name",
            "name_of_user": "name",
            "resides_in": "lives_in",
            "live_in": "lives_in",
            "works_as": "works_in",
            "occupation": "works_in",
            "job_title": "works_in",
            "prefers": "prefers_beverage",
            "drink_preference": "prefers_beverage",
            "beverage_preference": "prefers_beverage",
            "favorite_language": "favorite_programming_language",
            "favourite_language": "favorite_programming_language",
            "preferred_programming_language": "favorite_programming_language",
            "programming_language_preference": "favorite_programming_language",
            "recent_programming_language": "recently_codes_in",
            "recently_coding_in": "recently_codes_in",
            "currently_codes_in": "recently_codes_in",
        }
        canonical = alias_map.get(normalized, normalized)
        if canonical != normalized:
            return canonical

        # Router models sometimes put temporal wording into a relation label even
        # though temporality belongs in the row status/fact sentence. Normalize
        # programming-specific variants by meaning, never by language value.
        programming_relation = bool(
            re.search(r"(?:programming|coding)_languages?", normalized)
            or re.search(r"languages?_(?:for_)?(?:programming|coding)", normalized)
        )
        if programming_relation:
            if re.search(r"(?:prefer|preference|favorite|favourite|changed_mind)", normalized):
                return "favorite_programming_language"
            if re.search(r"(?:recent|current|lately|started|newly|now)", normalized):
                return "recently_codes_in"
            if re.search(r"(?:code|coding|program|use|language)", normalized):
                return "codes_in"
        return canonical

    def canonicalize_extracted_relation(
        self,
        relation: str,
        *,
        source_text: str = "",
        fact_sentence: str = "",
        value: str = "",
    ) -> str:
        """Resolve a router label from candidate-local meaning, never from language-name lists."""
        raw_relation = re.sub(
            r"[^a-z0-9_]+",
            "",
            re.sub(r"\s+", "_", (relation or "").strip().lower().replace("-", "_")),
        )
        canonical = self.normalize_relation(relation)
        if canonical in {"name", "lives_in", "works_in"}:
            return canonical
        # These labels are already domain-specific. A separate programming fact
        # in the source must not turn a drink preference into a language fact.
        if raw_relation in {"prefers_beverage", "drink_preference", "beverage_preference"}:
            return "prefers_beverage"

        relation_text = re.sub(r"[_-]+", " ", (relation or "").casefold())
        localized_source: list[str] = []
        carried_programming_preference = False
        normalized_value = self.normalize_value(value)
        if source_text and normalized_value:
            value_pattern = self._value_pattern(normalized_value)
            source_clauses: list[str] = []
            for sentence in self._split_sentences(source_text):
                source_clauses.extend(
                    re.sub(r"\s+", " ", clause).strip().casefold()
                    for clause in re.split(
                        r"\b(?:and|but|while|whereas|however|because|although|though)\b|,",
                        sentence,
                        flags=re.I,
                    )
                    if clause.strip()
                )
            for index, clause in enumerate(source_clauses):
                if value_pattern is None or not value_pattern.search(clause):
                    continue
                localized_source.append(clause)
                if (
                    index > 0
                    and re.search(r"\b(?:now|currently|changed|switched)\b", clause)
                    and re.search(r"\b(?:prefer|preferred|favorite|favourite)\b", clause)
                    and re.search(r"\b(?:i|i'm|i've|me|my)\b", clause)
                ):
                    previous = source_clauses[index - 1]
                    if (
                        re.search(r"\b(?:programming|coding)\s+languages?\b", previous)
                        and re.search(r"\b(?:prefer|preferred|favorite|favourite)\b", previous)
                        and re.search(r"\b(?:i|i'm|i've|me|my)\b", previous)
                    ):
                        localized_source.append(previous)
                        carried_programming_preference = True

        source_evidence = " ".join(localized_source)
        fact_evidence = fact_sentence.casefold()
        local_evidence = " ".join((fact_evidence, source_evidence))
        evidence = " ".join((relation_text, local_evidence))
        programming_context = bool(
            canonical in self.PROGRAMMING_RELATIONS
            or re.search(r"\b(?:programming|coding)\s+languages?\b", evidence)
            or re.search(r"\blanguages?\b[^.!?]{0,60}\b(?:code|coding|programming)\b", evidence)
            or re.search(r"\b(?:code|coding|programming)\b[^.!?]{0,60}\blanguages?\b", evidence)
        )

        if raw_relation == "prefers":
            strong_programming_target = bool(
                carried_programming_preference
                or re.search(r"\b(?:favorite|favourite|preferred)\s+(?:programming|coding)\s+languages?\b", local_evidence)
                or re.search(r"\b(?:programming|coding)\s+languages?\b[^.!?]{0,80}\bprefer", local_evidence)
                or re.search(r"\bprefer\b[^.!?]{0,80}\b(?:as|for)\b[^.!?]{0,30}\b(?:programming|coding)\b", local_evidence)
            )
            if not strong_programming_target:
                return "prefers_beverage"

        if not programming_context:
            ambiguous_temporal_preference = bool(
                re.fullmatch(
                    r"(?:(?:changed|current|currently|new|old|previous|prior|former)_)?"
                    r"(?:preferred_)?preference|changed_mind(?:_about)?_?preference",
                    raw_relation,
                )
            )
            beverage_context = bool(
                re.search(r"\b(?:drink|drinks|beverage|coffee|tea)\b", local_evidence)
            )
            if ambiguous_temporal_preference and not beverage_context:
                return "favorite_programming_language"
            return canonical

        source_preference_context = bool(
            re.search(r"\b(?:prefer|preference|preferred|favorite|favourite|like|love)\b", source_evidence)
        )
        source_activity_context = bool(
            re.search(r"\b(?:code|codes|coding|program|programs|use|uses|using)\b", source_evidence)
        )
        source_recent_context = bool(
            re.search(r"\b(?:recent|recently|lately|now|current|currently|started|starting|newly|these days)\b", source_evidence)
        )
        if source_recent_context and source_activity_context:
            return "recently_codes_in"
        if source_preference_context:
            return "favorite_programming_language"
        if source_activity_context:
            return "codes_in"

        fact_preference_context = bool(
            re.search(r"\b(?:prefer|preference|preferred|favorite|favourite|like|love)\b", fact_evidence)
        )
        fact_activity_context = bool(
            re.search(r"\b(?:code|codes|coding|program|programs|use|uses|using)\b", fact_evidence)
        )
        fact_recent_context = bool(
            re.search(r"\b(?:recent|recently|lately|now|current|currently|started|starting|newly|these days)\b", fact_evidence)
        )
        if fact_recent_context and fact_activity_context:
            return "recently_codes_in"
        if fact_preference_context:
            return "favorite_programming_language"
        if fact_activity_context:
            return "codes_in"
        if canonical in self.PROGRAMMING_RELATIONS:
            return canonical
        if re.search(r"\b(?:code|coding|program|programming|use|using|language)\b", evidence):
            return "codes_in"
        return canonical

    def fact_temporal_priority(self, source_text: str, value: str, fact_sentence: str = "") -> int:
        """Order historical assertions before neutral and explicitly current assertions."""
        normalized_value = re.sub(r"\s+", " ", (value or "").strip()).casefold()
        if not normalized_value:
            return 1

        value_pattern = self._value_pattern(normalized_value)
        historical = re.compile(r"\b(?:used to|formerly|previously|before|old|prior|no longer)\b", re.I)
        current = re.compile(r"\b(?:now|currently|current|these days|switched to|changed to|as of)\b", re.I)
        transition_verb = re.compile(r"\b(?:changed|switched)\b", re.I)
        from_word = re.compile(r"\bfrom\b", re.I)
        to_word = re.compile(r"\bto\b", re.I)

        def priority_from(text: str) -> int | None:
            matching_clauses: list[str] = []
            for sentence in self._split_sentences(text):
                clauses = re.split(r"(?:,+|\b(?:and|but|while|whereas|however)\b)", sentence, flags=re.I)
                normalized_clauses = [re.sub(r"\s+", " ", clause).casefold() for clause in clauses]
                matching_clauses.extend(
                    clause for clause in normalized_clauses if value_pattern is not None and value_pattern.search(clause)
                )
            for clause in matching_clauses:
                verb_match = transition_verb.search(clause)
                from_match = from_word.search(clause, verb_match.end() if verb_match else 0)
                if verb_match is None or from_match is None or value_pattern is None:
                    continue
                to_match = to_word.search(clause, from_match.end())
                if to_match is None:
                    continue
                for value_match in value_pattern.finditer(clause):
                    if from_match.end() <= value_match.start() < to_match.start():
                        return 0
                    if value_match.start() >= to_match.end():
                        return 2
            if any(historical.search(clause) for clause in matching_clauses):
                return 0
            if any(current.search(clause) for clause in matching_clauses):
                return 2
            return None

        # The user's source is authoritative. Router prose is only a fallback
        # when the source itself contains no temporal signal for this value.
        source_priority = priority_from(source_text)
        if source_priority is not None:
            return source_priority
        fact_priority = priority_from(fact_sentence)
        return fact_priority if fact_priority is not None else 1

    def is_trusted_relation(self, relation: str) -> bool:
        normalized = self.normalize_relation(relation)
        return (
            normalized
            in {
                "name",
                "lives_in",
                "codes_in",
                "favorite_programming_language",
                "recently_codes_in",
                "works_in",
                "prefers_beverage",
            }
            or normalized.startswith("changed_mind")
        )

    def render_fact_summary(self, entity: str, relation: str, value: str, fact_sentence: str) -> str:
        normalized_relation = self.normalize_relation(relation)
        if normalized_relation == "name":
            return f"Confirmed name: {value}."
        if normalized_relation == "lives_in":
            return f"Confirmed location: {value}."
        if normalized_relation == "codes_in":
            return f"Confirmed coding language: {value}."
        if normalized_relation == "favorite_programming_language":
            return f"Confirmed favorite programming language: {value}."
        if normalized_relation == "recently_codes_in":
            return f"Confirmed recently used programming language: {value}."
        if normalized_relation == "works_in":
            return f"Confirmed work context: {value}."
        if normalized_relation == "prefers_beverage":
            return f"Confirmed beverage preference: {value}."
        if normalized_relation.startswith("changed_mind"):
            return fact_sentence if fact_sentence.endswith(".") else f"{fact_sentence}."
        return fact_sentence if fact_sentence.endswith(".") else f"{fact_sentence}."

    def is_placeholder_value(self, value: str) -> bool:
        """Reject placeholder or guessed values before they can reach storage."""
        candidate = re.sub(r"\s+", " ", (value or "").strip().lower())
        candidate = candidate.strip(" \t\n\r.,;:")
        if not candidate:
            return True
        placeholder_patterns = [
            r"^\(?unknown(?:[^a-z0-9].*)?\)?$",
            r"^\(?n/?a\)?$",
            r"^\(?unspecified\)?$",
            r"^\(?unclear\)?$",
            r"^\(?none\)?$",
            r"^\(?not sure\)?$",
            r"^\(?i don't know\)?$",
        ]
        return any(re.fullmatch(pattern, candidate) for pattern in placeholder_patterns)

    def _source_supports_user_fact(self, source_text: str, relation: str, value: str) -> bool:
        """Require a structured user fact to be grounded in a first-person source clause."""
        source = re.sub(r"\s+", " ", (source_text or "").strip())
        normalized_value = self.normalize_value(value)
        value_pattern = self._value_pattern(normalized_value)
        if not source or value_pattern is None or not value_pattern.search(source):
            return False

        relation = self.normalize_relation(relation)
        relation_cues = {
            "name": r"\b(?:name|called|call|am|i'm)\b",
            "lives_in": r"\b(?:live|lives|reside|resides|moved|move|based|located|home)\b",
            "works_in": r"\b(?:work|works|job|occupation|employed|finance)\b",
            "codes_in": r"\b(?:code|codes|coding|program|programs|programming|language|develop|use|uses|using)\b|\bwork(?:ing)?\s+(?:in|with)\b",
            "favorite_programming_language": r"\b(?:favorite|favourite|prefer|preference|preferred|like|likes|love|loves)\b",
            "recently_codes_in": r"\b(?:recent|recently|lately|now|current|currently|started|starting|just|been|these\s+days)\b",
            "prefers_beverage": r"\b(?:prefer|prefers|drink|drinks|coffee|tea|beverage)\b",
        }
        cue_pattern = relation_cues.get(relation, r"\b(?:changed|prefer|like|want|use)\b")
        for sentence in self._split_sentences(source):
            # Subject inheritance is useful across clauses in one sentence, but
            # must never leak into a new sentence about someone else.
            current_subject: str | None = None
            clauses = [
                part.strip()
                for part in re.split(
                    r"(?:,+|\b(?:and|but|while|whereas|because|although|though)\b)",
                    sentence,
                    flags=re.I,
                )
                if part.strip()
            ]
            for clause in clauses:
                clause_lower = clause.casefold()
                self_owned_profile = bool(re.match(
                    r"^my\s+(?:(?:current|new|old|previous|prior|recent|recently\s+used)\s+)*"
                    r"(?:name|home|location|city|job|occupation|work|drink|beverage|preference|"
                    r"(?:programming|coding|language)\s+(?:language|preference)|"
                    r"(?:favorite|favourite|preferred)(?:\s+(?:programming|coding))?\s+language|"
                    r"favorite|favourite|preferred)\b",
                    clause_lower,
                ))
                possessed_other = bool(
                    re.match(r"^my\b", clause_lower)
                    and not self_owned_profile
                    and re.match(
                        r"^my\s+(?:[\w'-]+\s+){1,5}?"
                        r"(?:is|was|lives?|resides?|works?|likes?|prefers?|codes?|programs?|uses?|drinks?|"
                        r"moved|changed|switched)\b",
                        clause_lower,
                    )
                )
                explicit_other = possessed_other
                value_led_self_fact = bool(re.search(
                    r"\b(?:is|was|remains|became)\s+(?:(?:now|still|currently|already)\s+)*my\s+"
                    r"(?:favorite|favourite|preferred)\b",
                    clause_lower,
                ))
                explicit_self = bool(
                    value_led_self_fact
                    or re.search(r"\b(?:i|i'm|i've|the user)\b", clause_lower)
                    or self_owned_profile
                    or re.search(r"\bcall me\b", clause_lower)
                )
                other_value_attribution = False
                governing_other = re.compile(
                    r"\b(?!(?:i|me|my|user|now|currently|recently|also|still|really|often|usually|sometimes|generally|personally)\b)"
                    r"(?:[a-z][\w'-]*|he|she|they)\s+"
                    r"(?:(?:really|often|still|also|now|currently|recently|usually|sometimes|generally|"
                    r"favorite|favourite|preferred|programming|coding|language)\s+){0,4}"
                    r"(?:is|was|lives?|resides?|works?|likes?|prefers?|codes?|programs?|uses?|drinks?|moved|changed|switched)\b",
                    re.I,
                )
                for subject_match in governing_other.finditer(clause_lower):
                    prefix = clause_lower[:subject_match.start()]
                    reporting_prefix = bool(
                        re.search(
                            r"\b(?:heard|know|knew|said|say|told|reported|mentioned|think|thought|believe|"
                            r"learned|noticed|suspect|guess|read|saw)\b",
                            prefix,
                        )
                    )
                    if (
                        value_pattern.search(clause_lower[subject_match.end():])
                        and (subject_match.start() == 0 or not explicit_self or reporting_prefix)
                    ):
                        other_value_attribution = True
                        break
                named_subject = bool(re.match(
                    r"^(?!(?:i|my|me|now|currently|recently|also|still|really|often|usually|sometimes|generally|personally)\b)"
                    r"(?:[a-z][\w'-]*|he|she|they)\s+(?:[\w'-]+\s+){0,4}?"
                    r"(?:is|am|was|lives?|resides?|works?|likes?|prefers?|codes?|programs?|uses?|drinks?|"
                    r"moved|changed|switched)\b",
                    clause_lower,
                ))
                if explicit_other or other_value_attribution:
                    current_subject = "other"
                elif explicit_self:
                    current_subject = "user"
                elif named_subject:
                    current_subject = "other"

                if (
                    current_subject == "user"
                    and value_pattern.search(clause_lower)
                    and re.search(cue_pattern, clause_lower, flags=re.I)
                ):
                    return True
        return False

    def _active_records_for_key(self, entity: str, relation: str, *, conn=None) -> list[tuple]:
        normalized_entity = self.normalize_entity(entity)
        normalized_relation = self.normalize_relation(relation)
        owns_connection = conn is None
        connection = conn or sqlite3.connect(self.db_path)
        try:
            rows = connection.execute(
                "SELECT id, entity, relation, value, fact_sentence FROM global_archive WHERE is_active = 1 ORDER BY id DESC"
            ).fetchall()
            return [
                row
                for row in rows
                if self.normalize_entity(row[1] or "") == normalized_entity
                and self.normalize_relation(row[2] or "") == normalized_relation
            ]
        finally:
            if owns_connection:
                connection.close()

    def _active_fact_exists(self, entity: str, relation: str, value: str, *, conn=None) -> bool:
        normalized_value = self.normalize_value(value)
        return any(
            self.normalize_value(row[3]) == normalized_value
            for row in self._active_records_for_key(entity, relation, conn=conn)
        )

    def assess_fact_candidate(
        self,
        entity: str,
        relation: str,
        value: str,
        fact_sentence: str,
        *,
        source_text: str | None = None,
    ):
        """Validate a router fact candidate and return a normalized entity plus a human-readable reason."""
        normalized_entity = self.normalize_entity(entity)
        normalized_relation = self.normalize_relation(relation)
        normalized_value = (value or "").strip()
        normalized_sentence = (fact_sentence or "").strip()

        if not normalized_entity:
            return False, normalized_entity, "malformed entity"
        if not normalized_relation:
            return False, normalized_entity, "missing relation"
        if not self.is_trusted_relation(normalized_relation):
            return False, normalized_entity, "unsupported relation"
        if not normalized_sentence:
            return False, normalized_entity, "missing fact sentence"
        if self.is_placeholder_value(normalized_value):
            return False, normalized_entity, "placeholder value"
        if source_text is not None and not self._source_supports_user_fact(source_text, normalized_relation, normalized_value):
            return False, normalized_entity, "fact is not grounded as a first-person user statement"
        if self._active_fact_exists(normalized_entity, normalized_relation, normalized_value):
            return False, normalized_entity, "duplicate active fact"
        if self._fact_sentence_exists(
            normalized_sentence,
            entity=normalized_entity,
            relation=normalized_relation,
            value=normalized_value,
        ):
            return False, normalized_entity, "duplicate fact"

        return True, normalized_entity, "accepted"

    def _fact_sentence_exists(
        self,
        fact_sentence: str,
        similarity_threshold: float = 0.96,
        *,
        entity: str | None = None,
        relation: str | None = None,
        value: str | None = None,
    ) -> bool:
        """Skip exact or near-duplicate prose for the same structured fact value."""
        cleaned_sentence = fact_sentence.strip()
        if not cleaned_sentence:
            return True

        with _open_memory_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT entity, relation, value, fact_sentence, embedding FROM global_archive WHERE is_active = 1"
            )
            active_rows = [
                (existing_sentence, embedding)
                for existing_entity, existing_relation, existing_value, existing_sentence, embedding in cursor.fetchall()
                if (
                    (entity is None or relation is None)
                    or (
                        self.normalize_entity(existing_entity or "") == self.normalize_entity(entity)
                        and self.normalize_relation(existing_relation or "") == self.normalize_relation(relation)
                    )
                )
                and (value is None or self.normalize_value(existing_value or "") == self.normalize_value(value))
            ]
            if not active_rows:
                return False

            if any((existing_sentence or "").strip().casefold() == cleaned_sentence.casefold() for existing_sentence, _ in active_rows):
                return True

            candidate_vector = self.embed_model.encode(cleaned_sentence)
            candidate_norm = np.linalg.norm(candidate_vector)
            for existing_sentence, emb_bytes in active_rows:
                if not emb_bytes:
                    continue
                existing_vector = np.frombuffer(emb_bytes, dtype=np.float32)
                if existing_vector.size == 0:
                    continue
                similarity = np.dot(candidate_vector, existing_vector) / (
                    candidate_norm * np.linalg.norm(existing_vector) + 1e-8
                )
                if similarity >= similarity_threshold:
                    return True

        return False

    def build_router_messages(self, sentence: str):
        """Few-shot router template for durable fact extraction from dense user text."""
        return [
            {
                "role": "system",
                "content": (
                    "You extract only durable user facts from the final user message. "
                    "The final user message is the entire and only evidence source: never copy facts or values from examples, prior knowledge, or an implied conversation. "
                    "Return zero or more lines in the format entity | relation | value | fact_sentence. "
                    "If the message contains no durable fact, return None. "
                    "The entity must be user and the source must explicitly be about the person speaking (I, me, or my). "
                    "Never turn facts about named people or other third parties into user facts. "
                    "Use exactly one of these relation labels: name, lives_in, works_in, codes_in, "
                    "favorite_programming_language, recently_codes_in, prefers_beverage. "
                    "Keep programming-language meanings distinct: use favorite_programming_language for likes, preferences, or favorites; "
                    "recently_codes_in for recent/current activity or a language the user has just started using; and codes_in only for general usage without either meaning. "
                    "Do not create a new relation label for words such as old, former, changed, current, or now. "
                    "For an old-to-new change, emit both values under the same canonical relation in temporal order, with the explicitly current value last. "
                    "Extract facts even when they appear inside compound or long sentences. "
                    "Questions asking you to recall, list, or confirm existing user facts are not new facts. "
                    "Every emitted value and its relation meaning must be explicitly entailed by the final user message. "
                    "Ignore greetings, filler, profile references, meta-commentary, hypothetical questions, and creative prompts."
                ),
            },
            {
                "role": "user",
                "content": "I now drink black coffee, not tea, and I work in Rust.",
            },
            {
                "role": "assistant",
                "content": (
                    "user | prefers_beverage | black coffee | The user now drinks black coffee, not tea.\n"
                    "user | codes_in | Rust | The user works in Rust."
                ),
            },
            {
                "role": "user",
                "content": "Hello assistant, based on my profile background, what should I do?",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "If you were a spaceship pilot, what ship would you fly?",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "Anish lives in Mumbai and Alex lives in Amsterdam.",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "I live in Mumbai and Alex lives in Amsterdam.",
            },
            {
                "role": "assistant",
                "content": "user | lives_in | Mumbai | The user lives in Mumbai.",
            },
            {
                "role": "user",
                "content": "What's my name again?",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "Tell me my name, what language I code in, and where I live right now.",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "Also, do you remember where I used to live before this?",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "Can you list everything you know about me?",
            },
            {
                "role": "assistant",
                "content": "None",
            },
            {
                "role": "user",
                "content": "My name is Anish. I moved to Seattle and I prefer tea over coffee.",
            },
            {
                "role": "assistant",
                "content": (
                    "user | name | Anish | The user's name is Anish.\n"
                    "user | lives_in | Seattle | The user moved to Seattle.\n"
                    "user | prefers_beverage | tea | The user prefers tea over coffee."
                ),
            },
            {
                "role": "user",
                "content": "I like coding in Python. I have also started coding a lot in TypeScript. Python is still my favorite.",
            },
            {
                "role": "assistant",
                "content": (
                    "user | favorite_programming_language | Python | The user's favorite programming language is Python.\n"
                    "user | recently_codes_in | TypeScript | The user has recently been coding in TypeScript."
                ),
            },
            {
                "role": "user",
                "content": "My favorite programming language was Ruby before, but now I prefer Kotlin.",
            },
            {
                "role": "assistant",
                "content": (
                    "user | favorite_programming_language | Ruby | Ruby was previously the user's favorite programming language.\n"
                    "user | favorite_programming_language | Kotlin | Kotlin is now the user's favorite programming language."
                ),
            },
            {"role": "user", "content": sentence},
        ]

    def _init_db(self):
        """Initializes SQLite database with HMO structures and temporal tracking."""
        with _open_memory_db(self.db_path) as conn:
            cursor = conn.cursor()
            # Tier 3 Global Archive with temporal invalidation tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT,
                    relation TEXT,
                    value TEXT,
                    fact_sentence TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    invalidated_at TIMESTAMP DEFAULT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)
            # Tier 1 Cache Operational Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS primary_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    archive_id INTEGER,
                    fact_sentence TEXT,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_fact_with_resolution(self, entity: str, relation: str, value: str, fact_sentence: str):
        """Temporal Contradiction Resolver: Invalidates conflicting edges before inserting a new one."""
        is_valid, normalized_entity, reason = self.assess_fact_candidate(entity, relation, value, fact_sentence)
        if not is_valid:
            self._log(f"[Memory Write] Rejected fact candidate: {reason}")
            return False

        entity = normalized_entity
        relation = self.normalize_relation(relation)
        value = value.strip()
        fact_sentence = fact_sentence.strip()

        # Calculate semantic embedding for the new fact string
        vector = self.embed_model.encode(fact_sentence).tobytes()

        with _open_memory_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Compare canonical keys so legacy aliases such as nameself/name resolve together.
            existing_records = self._active_records_for_key(entity, relation, conn=conn)
            if any(self.normalize_value(row[3]) == self.normalize_value(value) for row in existing_records):
                conn.rollback()
                self._log("[Memory Write] Rejected fact candidate: duplicate active fact")
                return False

            for archive_id, _, _, old_value, old_sentence in existing_records:
                if self.normalize_value(old_value) != self.normalize_value(value):
                    # Chronologically deprecate the stale preference state
                    self._log(f"[Conflict Resolution] Invalidating old fact: '{old_sentence}' -> New fact: '{fact_sentence}'")
                    cursor.execute("""
                        UPDATE global_archive
                        SET is_active = 0, invalidated_at = ?
                        WHERE id = ?
                    """, (timestamp, archive_id))
                    # Immediately evict from Tier 1 Cache
                    cursor.execute("DELETE FROM primary_cache WHERE archive_id = ?", (archive_id,))

            # Store the new fact into the active global archive (Tier 3)
            cursor.execute("""
                INSERT INTO global_archive (entity, relation, value, fact_sentence, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (entity, relation, value, fact_sentence, vector))
            new_archive_id = cursor.lastrowid
            conn.commit()

            self._log(f"[Memory Write] Stored in Tier 3 global archive as id={new_archive_id}.")

            # HMO Process: Push directly into Tier 1 Primary Cache
            self._promote_to_primary(new_archive_id, fact_sentence)
            return True

    def _promote_to_primary(self, archive_id: int, fact_sentence: str):
        """HMO Promotion: Adds a fact to Tier 1 primary cache table and enforces size limit (relegation)."""
        with _open_memory_db(self.db_path) as conn:
            cursor = conn.cursor()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Avoid duplicate inserts in primary cache
            cursor.execute("SELECT id FROM primary_cache WHERE archive_id = ?", (archive_id,))
            if cursor.fetchone():
                self._log(f"[Tier 1] archive_id={archive_id} already present; promotion skipped.")
                return

            cursor.execute("""
                INSERT INTO primary_cache (archive_id, fact_sentence, last_accessed)
                VALUES (?, ?, ?)
            """, (archive_id, fact_sentence, timestamp))

            cursor.execute("SELECT COUNT(*) FROM primary_cache")
            current_cache_size = cursor.fetchone()[0]
            self._log(f"[Tier 1] Promoted archive_id={archive_id}. Cache size is now {current_cache_size}/{self.primary_cache_max}.")

            # HMO Relegation Rule: Enforce context size ceilings
            if current_cache_size > self.primary_cache_max:
                cursor.execute("SELECT id, fact_sentence FROM primary_cache ORDER BY last_accessed ASC LIMIT 1")
                lru_row = cursor.fetchone()
                if lru_row:
                    lru_id, lru_fact = lru_row
                    self._log(f"[Tier 1] Relegating oldest fact to preserve newer cache entries: '{lru_fact}'")
                    cursor.execute("DELETE FROM primary_cache WHERE id = ?", (lru_id,))
            conn.commit()

    @staticmethod
    def _render_context(facts: list[str]) -> str:
        return "Background Profile Info:\n" + "\n".join([f"- {fact}" for fact in facts])

    def _active_user_names(self, archive_rows: list[tuple]) -> set[str]:
        return {
            self.normalize_value(value)
            for _, entity, relation, value, _, _ in archive_rows
            if self.normalize_entity(entity or "") == "user"
            and self.normalize_relation(relation or "") == "name"
            and self.normalize_value(value)
        }

    def _resolve_exact_lookup(
        self,
        user_query: str,
        archive_rows: list[tuple],
    ) -> StructuredRelationResolution:
        """Resolve exact relations and ordered relation-family candidates from query meaning."""
        query = re.sub(r"\s+", " ", user_query.strip().casefold())
        relations: list[str] = []
        has_user_reference = bool(re.search(r"\b(?:i|me|my|mine|myself)\b", query))

        if has_user_reference:
            programming_qualified = bool(re.search(r"\b(?:code|coding|program|programming)\b", query))
            if (
                re.search(r"\b(?:what\s+(?:kinds?\s+of\s+)?things|everything|anything)\b", query)
                and re.search(r"\b(?:remember|know|recall)\b", query)
                and not programming_qualified
            ):
                return StructuredRelationResolution(reason="broad profile recall uses hybrid retrieval")
            if re.search(r"\b(?:name|who am i|what am i called)\b", query):
                relations.append("name")
            if re.search(r"\b(?:live|lives|reside|resides|location|based)\b", query):
                relations.append("lives_in")
            if re.search(r"\b(?:work|job|occupation|employed)\b", query):
                relations.append("works_in")
            language_query = bool(re.search(r"\blanguages?\b", query))
            preference_language_query = bool(
                language_query
                and re.search(r"\b(?:favorite|favourite|prefer|preferred|preference|like)\b", query)
            )
            spoken_language_query = bool(
                language_query and re.search(r"\b(?:speak|speaking|spoken|natural|human)\b", query)
            )
            programming_attribute_query = bool(
                re.search(r"\b(?:code|coding|programming)\s+(?:style|paradigm|framework|editor|ide|tool)\b", query)
                and not language_query
            )
            programming_query = bool(
                (programming_qualified or preference_language_query)
                and not spoken_language_query
                and not programming_attribute_query
            )
            selection = "all"
            include_history = False
            reason = "exact user-profile relation question"
            if programming_query:
                history_query = bool(
                    re.search(
                        r"\b(?:history|historical|past|previous|previously|former|formerly|old|earlier|used to|before)\b"
                        r"|\bwhat\s+was\s+my\b",
                        query,
                    )
                )
                broad_programming_query = bool(
                    re.search(r"\bwhat\s+do\s+you\s+(?:know|remember|recall)\b", query)
                    or re.search(r"\b(?:everything|all)\b[^?]*\b(?:programming|coding|languages?)\b", query)
                    or (
                        re.search(r"\bwhat\s+(?:kinds?\s+of\s+)?things\b", query)
                        and re.search(r"\b(?:remember|know|recall)\b", query)
                    )
                )
                preference_query = bool(
                    re.search(r"\b(?:favorite|favourite|prefer|preferred|preference|like)\b", query)
                )
                recent_query = bool(
                    re.search(r"\b(?:recent|recently|lately|currently|started|been coding)\b", query)
                )
                activity_query = bool(
                    re.search(r"\b(?:code|coding|program|programming|use|using|used)\b", query)
                )
                if history_query:
                    include_history = True
                    if preference_query:
                        relations.append("favorite_programming_language")
                        reason = "programming-language preference history question"
                    elif recent_query:
                        relations.append("recently_codes_in")
                        reason = "recent programming-activity history question"
                    elif broad_programming_query or re.search(
                        r"\b(?:programming|coding)\s+languages?\s+history\b"
                        r"|\bhistory\b[^?]*\b(?:programming|coding)\b",
                        query,
                    ):
                        relations.extend(self.PROGRAMMING_RELATIONS)
                        reason = "broad programming-language history question"
                    elif activity_query:
                        relations.append("codes_in")
                        reason = "coding-language history question"
                    else:
                        relations.extend(self.PROGRAMMING_RELATIONS)
                        reason = "broad programming-language history question"
                elif preference_query:
                    relations.append("favorite_programming_language")
                    reason = "explicit programming-language preference question"
                elif recent_query:
                    relations.append("recently_codes_in")
                    reason = "explicit recent programming-activity question"
                elif broad_programming_query:
                    relations.extend(self.PROGRAMMING_RELATIONS)
                    reason = "broad programming-memory question"
                else:
                    relations.extend(self.PROGRAMMING_RELATIONS)
                    selection = "first_available"
                    reason = "generic programming-language question with ordered family fallback"
            if re.search(r"\b(?:beverage|drink|coffee|tea)\b", query) and re.search(r"\b(?:prefer|like|drink)\b", query):
                relations.append("prefers_beverage")
            if not relations and (spoken_language_query or programming_attribute_query or language_query):
                return StructuredRelationResolution(
                    unsupported_reason="question asks about an unsupported non-programming-language attribute"
                )
            return StructuredRelationResolution(
                candidates=tuple(dict.fromkeys(relations)),
                reason=reason,
                selection=selection,
                include_history=include_history,
            )

        named_match = re.search(
            r"\bwhere\s+(?:does|is)\s+([a-z][\w'-]*)\s+(live|reside|work|located|based)\b",
            query,
        )
        if named_match:
            subject, verb = named_match.groups()
            active_names = self._active_user_names(archive_rows)
            if len(active_names) != 1 or subject not in active_names:
                return StructuredRelationResolution(
                    unsupported_reason="named person is not the uniquely identified user"
                )
            relation = "works_in" if verb == "work" else "lives_in"
            return StructuredRelationResolution(
                candidates=("name", relation),
                reason="named subject resolved to the uniquely identified user",
            )

        reverse_match = re.search(r"\bwho\s+(?:currently\s+)?lives?\s+in\s+(.+?)(?:[?.!]|$)", query)
        if reverse_match:
            requested_value = self.normalize_value(reverse_match.group(1))
            matching_locations = {
                self.normalize_value(value)
                for _, entity, relation, value, _, _ in archive_rows
                if self.normalize_entity(entity or "") == "user"
                and self.normalize_relation(relation or "") == "lives_in"
                and self.normalize_value(value) == requested_value
            }
            if matching_locations and len(self._active_user_names(archive_rows)) == 1:
                return StructuredRelationResolution(
                    candidates=("name", "lives_in"),
                    reason="reverse location lookup resolved to the uniquely identified user",
                )
            return StructuredRelationResolution(
                unsupported_reason="reverse relationship cannot be resolved to one supported user"
            )

        return StructuredRelationResolution()

    def _render_historical_summary(self, relation: str, value: str, *, active: bool) -> str:
        labels = {
            "codes_in": "coding language",
            "recently_codes_in": "recently used programming language",
            "favorite_programming_language": "favorite programming language",
        }
        status = "Current" if active else "Previous"
        label = labels.get(self.normalize_relation(relation), "memory value")
        return f"{status} {label}: {value}."

    def _structured_context(
        self,
        resolution: StructuredRelationResolution,
        archive_rows: list[tuple],
        tier1_ids: list[int],
        all_archive_rows: list[tuple],
    ) -> str:
        selected: list[tuple] = []
        programming_candidates = [
            relation for relation in resolution.candidates if relation in self.PROGRAMMING_RELATIONS
        ]
        exact_candidates = [
            relation for relation in resolution.candidates if relation not in self.PROGRAMMING_RELATIONS
        ]

        def latest_active(wanted_relation: str) -> tuple | None:
            for row in sorted(archive_rows, key=lambda item: item[0], reverse=True):
                row_id, entity, relation, value, sentence, _ = row
                if self.normalize_entity(entity or "") != "user":
                    continue
                if self.normalize_relation(relation or "") == wanted_relation:
                    return row_id, entity, wanted_relation, value, sentence, True
            return None

        for wanted_relation in exact_candidates:
            match = latest_active(wanted_relation)
            if match:
                selected.append(match)

        if resolution.include_history:
            for wanted_relation in programming_candidates:
                for row in sorted(all_archive_rows, key=lambda item: item[0], reverse=True):
                    row_id, entity, relation, value, sentence, _, is_active, _ = row
                    if self.normalize_entity(entity or "") != "user":
                        continue
                    if self.normalize_relation(relation or "") == wanted_relation:
                        selected.append((row_id, entity, wanted_relation, value, sentence, bool(is_active)))
        elif resolution.selection == "first_available":
            for wanted_relation in programming_candidates:
                match = latest_active(wanted_relation)
                if match:
                    selected.append(match)
                    break
        else:
            for wanted_relation in programming_candidates:
                match = latest_active(wanted_relation)
                if match:
                    selected.append(match)

        facts = [
            self._render_historical_summary(relation, value, active=is_active)
            if resolution.include_history and relation in self.PROGRAMMING_RELATIONS
            else self.render_fact_summary(entity, relation, value, sentence)
            for _, entity, relation, value, sentence, is_active in selected
        ]
        active_considered = sum(
            1
            for _, entity, relation, _, _, _ in archive_rows
            if self.normalize_entity(entity or "") == "user"
            and self.normalize_relation(relation or "") in resolution.candidates
        )
        historical_considered = sum(
            1
            for _, entity, relation, _, _, _, is_active, _ in all_archive_rows
            if not is_active
            and self.normalize_entity(entity or "") == "user"
            and self.normalize_relation(relation or "") in resolution.candidates
        ) if resolution.include_history else 0
        selected_statuses = [
            f"{'active' if is_active else 'historical'}:{relation}"
            for _, _, relation, _, _, is_active in selected
        ]
        self._log(
            "[Memory Retrieval] Structured facts considered: "
            f"active={active_considered}, historical={historical_considered}; "
            f"selected={', '.join(selected_statuses) if selected_statuses else 'none'}."
        )
        if self.retrieval_mutates_cache:
            selected_ids = [row[0] for row in selected if row[5]]
            cached_ids = [row_id for row_id in selected_ids if row_id in tier1_ids]
            if cached_ids:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                with _open_memory_db(self.db_path) as conn:
                    conn.execute(
                        f"UPDATE primary_cache SET last_accessed = ? WHERE archive_id IN ({','.join(['?'] * len(cached_ids))})",
                        [timestamp] + cached_ids,
                    )
            for row_id, _, _, _, sentence, is_active in selected:
                if is_active and row_id not in tier1_ids:
                    self._promote_to_primary(row_id, sentence)

        self.last_retrieval_stats = {
            "tier1_hits": sum(1 for row in selected if row[0] in tier1_ids),
            "tier2_promoted": 0,
            "tier3_scanned": len(all_archive_rows) if resolution.include_history else len(archive_rows),
            "structured_hits": len(facts),
            "structured_candidates": list(resolution.candidates),
            "structured_reason": resolution.reason,
            "active_considered": active_considered,
            "historical_considered": historical_considered,
            "facts": facts,
        }
        return self._render_context(facts) if facts else ""

    def get_orchestrated_context(self, user_query: str) -> str:
        """Prefer exact structured user lookups, then preserve the existing Tier 1/semantic path."""
        self._log(f"[Memory Retrieval] Query: {user_query}")
        with _open_memory_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.fact_sentence, p.archive_id, g.entity, g.relation, g.value, g.fact_sentence
                FROM primary_cache p
                LEFT JOIN global_archive g ON g.id = p.archive_id
                WHERE g.is_active = 1
                ORDER BY p.last_accessed DESC, p.id DESC
            """)
            tier1_rows = cursor.fetchall()
            tier1_ids = [row[1] for row in tier1_rows]
            cursor.execute(
                "SELECT id, entity, relation, value, fact_sentence, embedding, is_active, invalidated_at "
                "FROM global_archive"
            )
            all_archive_rows = cursor.fetchall()
            archive_rows = [row[:6] for row in all_archive_rows if row[6]]

        resolution = self._resolve_exact_lookup(user_query, archive_rows)
        if resolution.candidates:
            self._log(
                "[Memory Retrieval] Using structured lookup with "
                f"candidates=[{', '.join(resolution.candidates)}], "
                f"selection={resolution.selection}, include_history={str(resolution.include_history).lower()}, "
                f"reason={resolution.reason}."
            )
            return self._structured_context(resolution, archive_rows, tier1_ids, all_archive_rows)
        if resolution.unsupported_reason:
            self._log(f"[Memory Retrieval] Structured lookup skipped: {resolution.unsupported_reason}.")
            self.last_retrieval_stats = {
                "tier1_hits": 0,
                "tier2_promoted": 0,
                "tier3_scanned": len(archive_rows),
                "structured_hits": 0,
                "structured_candidates": [],
                "structured_reason": resolution.unsupported_reason,
                "active_considered": 0,
                "historical_considered": 0,
                "facts": [],
            }
            return ""

        facts: list[str] = []
        seen_relations: set[str] = set()
        for cache_sentence, _, entity, relation, value, archive_sentence in tier1_rows:
            canonical_relation = self.normalize_relation(relation or "")
            if not self.is_trusted_relation(canonical_relation) or canonical_relation in seen_relations:
                continue
            facts.append(self.render_fact_summary(entity or "user", canonical_relation, value or "", archive_sentence or cache_sentence))
            seen_relations.add(canonical_relation)

        if tier1_ids and self.retrieval_mutates_cache:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with _open_memory_db(self.db_path) as conn:
                conn.execute(
                    f"UPDATE primary_cache SET last_accessed = ? WHERE archive_id IN ({','.join(['?'] * len(tier1_ids))})",
                    [timestamp] + tier1_ids,
                )

        query_vector = self.embed_model.encode(user_query)
        tier2_candidates = []
        for row_id, entity, relation, value, sentence, emb_bytes in archive_rows:
            if row_id in tier1_ids or not emb_bytes:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.size == 0:
                continue
            similarity = np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb) + 1e-8)
            canonical_relation = self.normalize_relation(relation or "")
            if similarity > 0.50 and self.is_trusted_relation(canonical_relation):
                tier2_candidates.append((similarity, row_id, entity, canonical_relation, value, sentence))

        tier2_candidates.sort(key=lambda item: item[0], reverse=True)
        promoted_facts: list[str] = []
        for _, row_id, entity, canonical_relation, value, sentence in tier2_candidates[:2]:
            if canonical_relation in seen_relations:
                continue
            summary = self.render_fact_summary(entity or "user", canonical_relation, value or "", sentence)
            facts.append(summary)
            promoted_facts.append(summary)
            seen_relations.add(canonical_relation)
            if self.retrieval_mutates_cache:
                self._log(f"[Tier 2] Promoting archive_id={row_id} into Tier 1: '{sentence}'")
                self._promote_to_primary(row_id, sentence)

        unique_facts = list(dict.fromkeys(facts))
        self.last_retrieval_stats = {
            "tier1_hits": len(tier1_rows),
            "tier2_promoted": len(promoted_facts),
            "tier3_scanned": len(archive_rows),
            "structured_hits": 0,
            "facts": unique_facts,
        }
        return self._render_context(unique_facts) if unique_facts else ""
