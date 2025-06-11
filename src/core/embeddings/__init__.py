"""
Embedding generation services for philosophical research.

This module provides embedding generation capabilities optimized for
philosophical texts, including support for multiple languages and
philosophical terminology.
"""

# Conditional import for testing without Google Cloud dependencies
try:
    from .vertex_ai_service import VertexAIEmbeddingService
    _VERTEX_AI_AVAILABLE = True
except ImportError:
    _VERTEX_AI_AVAILABLE = False
    VertexAIEmbeddingService = None
from .embedding_models import EmbeddingRequest, EmbeddingResult, EmbeddingBatch

__all__ = [
    "EmbeddingRequest", 
    "EmbeddingResult",
    "EmbeddingBatch"
]

if _VERTEX_AI_AVAILABLE:
    __all__.append("VertexAIEmbeddingService")