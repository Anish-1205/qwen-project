"""Backend-independent contracts for compatible local causal language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Observable model features available through its configured backend."""

    chat: bool = True
    system_messages: bool = True
    tool_schemas: bool = False
    tool_messages: bool = False
    token_counting: bool = True
    sampling: bool = True


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declarative model configuration; contains no executable loader."""

    id: str
    display_name: str
    model_name: str
    backend: str
    capabilities: ModelCapabilities
    load_options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One backend-independent chat generation request."""

    messages: Sequence[Mapping[str, object]]
    max_new_tokens: int = 300
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: Sequence[Mapping[str, object]] | None = None
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Normalized result returned by every model backend."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class ModelBackend(ABC):
    """Lifecycle and inference boundary used by the harness."""

    @property
    @abstractmethod
    def spec(self) -> ModelSpec:
        """Return the declarative specification for the active model."""

    @abstractmethod
    def load(self) -> None:
        """Load resources needed for inference."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one assistant response."""

    @abstractmethod
    def count_tokens(self, messages: Sequence[Mapping[str, object]]) -> int:
        """Return the rendered prompt-token count for chat messages."""

    def close(self) -> None:
        """Release backend resources. Backends without resources may do nothing."""
