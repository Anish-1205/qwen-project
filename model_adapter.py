"""Compatibility imports for the former top-level model module.

New code should import backend contracts from :mod:`models`.
"""

from models import (  # noqa: F401
    GenerationRequest,
    GenerationResult,
    ModelBackend,
    ModelCapabilities,
    ModelSpec,
    TransformersBackend,
    create_backend,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "ModelBackend",
    "ModelCapabilities",
    "ModelSpec",
    "TransformersBackend",
    "create_backend",
]
