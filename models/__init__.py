"""Backend-independent model contracts and configured local models."""

from .backends import TransformersBackend, create_backend
from .contracts import (
    GenerationRequest,
    GenerationResult,
    ModelBackend,
    ModelCapabilities,
    ModelSpec,
)
from .registry import MODEL_REGISTRY, get_model_spec

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "ModelBackend",
    "ModelCapabilities",
    "MODEL_REGISTRY",
    "ModelSpec",
    "TransformersBackend",
    "create_backend",
    "get_model_spec",
]
