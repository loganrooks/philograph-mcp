"""
Data models for embedding generation and management.

Defines the core data structures used throughout the embedding service
for philosophical research applications.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class EmbeddingRequest(BaseModel):
    """
    Request for generating embeddings from text content.
    
    Optimized for philosophical research with support for:
    - Multiple languages (ancient Greek, Latin, German, French, etc.)
    - Philosophical terminology and proper names
    - Citation context preservation
    """
    
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, max_length=10000)
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'de', 'grc', 'la')")
    philosophical_tradition: Optional[str] = Field(None, description="Tradition context (e.g., 'ancient', 'continental', 'analytic')")
    document_id: Optional[UUID] = Field(None, description="Associated document UUID")
    chunk_index: Optional[int] = Field(None, description="Index within document")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingResult(BaseModel):
    """
    Result of embedding generation with philosophical context.
    
    Contains the generated embedding vector along with metadata
    for philosophical research applications.
    """
    
    request_id: UUID
    embedding: List[float] = Field(..., description="3072-dimensional embedding vector")
    model_name: str = Field(..., description="Model used for generation")
    dimension: int = Field(3072, description="Embedding vector dimension")
    language_detected: Optional[str] = Field(None, description="Auto-detected language")
    philosophical_score: Optional[float] = Field(None, description="Confidence score for philosophical content")
    processing_time_ms: float
    cached: bool = Field(False, description="Whether result came from cache")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingBatch(BaseModel):
    """
    Batch processing request for multiple embeddings.
    
    Enables efficient processing of multiple philosophical texts
    with shared context and configuration.
    """
    
    id: UUID = Field(default_factory=uuid4)
    requests: List[EmbeddingRequest]
    batch_size: int = Field(50, description="Number of texts to process in parallel")
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field("normal", description="Processing priority: low, normal, high")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def total_texts(self) -> int:
        """Total number of texts in batch."""
        return len(self.requests)
        
    @property
    def estimated_tokens(self) -> int:
        """Estimated total tokens for cost calculation."""
        return sum(len(req.text.split()) for req in self.requests)


class EmbeddingBatchResult(BaseModel):
    """
    Results from batch embedding processing.
    
    Contains all individual results plus batch-level metrics
    for philosophical research workflows.
    """
    
    batch_id: UUID
    results: List[EmbeddingResult]
    total_processed: int
    successful: int
    failed: int
    total_processing_time_ms: float
    average_time_per_embedding_ms: float
    cache_hit_rate: float
    cost_estimate_usd: Optional[float] = Field(None, description="Estimated API cost")
    philosophical_content_ratio: Optional[float] = Field(None, description="Ratio of texts identified as philosophical")
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful embeddings."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful / self.total_processed) * 100.0


class EmbeddingCache(BaseModel):
    """
    Cache entry for storing embedding results.
    
    Optimized for Redis storage with philosophical research metadata.
    """
    
    text_hash: str = Field(..., description="SHA-256 hash of input text")
    embedding: List[float]
    model_name: str
    language: Optional[str] = None
    philosophical_tradition: Optional[str] = None
    created_at: datetime
    access_count: int = Field(0, description="Number of times accessed from cache")
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Cache expiration time")
    
    def to_redis_dict(self) -> Dict[str, Any]:
        """Convert to Redis-storable dictionary."""
        return {
            "text_hash": self.text_hash,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "language": self.language,
            "philosophical_tradition": self.philosophical_tradition,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }