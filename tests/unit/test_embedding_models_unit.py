"""
Unit tests for embedding data models.

Tests all embedding-related data structures without external dependencies.
Following TDD methodology for philosophical research requirements.
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import Mock

from src.core.embeddings.embedding_models import (
    EmbeddingRequest,
    EmbeddingResult, 
    EmbeddingBatch,
    EmbeddingBatchResult,
    EmbeddingCache
)


class TestEmbeddingRequest:
    """Test EmbeddingRequest model for philosophical texts."""
    
    def test_embedding_request_creation_basic(self):
        """Test basic EmbeddingRequest creation."""
        request = EmbeddingRequest(
            text="Being and Time is Heidegger's masterwork on fundamental ontology.",
            language="en",
            philosophical_tradition="continental"
        )
        
        assert isinstance(request.id, UUID)
        assert request.text == "Being and Time is Heidegger's masterwork on fundamental ontology."
        assert request.language == "en"
        assert request.philosophical_tradition == "continental"
        assert isinstance(request.created_at, datetime)
        assert request.metadata == {}
        
    def test_embedding_request_with_document_context(self):
        """Test EmbeddingRequest with document and chunk context."""
        doc_id = uuid4()
        request = EmbeddingRequest(
            text="§7. The Phenomenological Method of Investigation",
            document_id=doc_id,
            chunk_index=42,
            metadata={"section": "7", "page": "27"}
        )
        
        assert request.document_id == doc_id
        assert request.chunk_index == 42
        assert request.metadata["section"] == "7"
        assert request.metadata["page"] == "27"
        
    def test_embedding_request_ancient_languages(self):
        """Test EmbeddingRequest with ancient philosophical languages."""
        # Ancient Greek
        greek_request = EmbeddingRequest(
            text="τὸ ὂν λέγεται πολλαχῶς",
            language="grc",
            philosophical_tradition="ancient"
        )
        assert greek_request.language == "grc"
        
        # Latin
        latin_request = EmbeddingRequest(
            text="Cogito ergo sum",
            language="la", 
            philosophical_tradition="medieval"
        )
        assert latin_request.language == "la"
        
    def test_embedding_request_validation_errors(self):
        """Test EmbeddingRequest validation for invalid inputs."""
        # Empty text should fail
        with pytest.raises(ValueError):
            EmbeddingRequest(text="")
            
        # Text too long should fail
        with pytest.raises(ValueError):
            EmbeddingRequest(text="x" * 10001)


class TestEmbeddingResult:
    """Test EmbeddingResult model for philosophical research."""
    
    def test_embedding_result_creation_successful(self):
        """Test successful EmbeddingResult creation."""
        request_id = uuid4()
        embedding = [0.1] * 3072  # 3072-dimensional vector
        
        result = EmbeddingResult(
            request_id=request_id,
            embedding=embedding,
            model_name="textembedding-gecko@003",
            processing_time_ms=150.5,
            language_detected="en",
            philosophical_score=0.87
        )
        
        assert result.request_id == request_id
        assert len(result.embedding) == 3072
        assert result.model_name == "textembedding-gecko@003"
        assert result.dimension == 3072
        assert result.processing_time_ms == 150.5
        assert result.language_detected == "en"
        assert result.philosophical_score == 0.87
        assert result.cached is False
        assert result.error is None
        
    def test_embedding_result_with_error(self):
        """Test EmbeddingResult with error condition."""
        request_id = uuid4()
        
        result = EmbeddingResult(
            request_id=request_id,
            embedding=[],
            model_name="textembedding-gecko@003",
            processing_time_ms=0.0,
            error="API rate limit exceeded"
        )
        
        assert result.error == "API rate limit exceeded"
        assert len(result.embedding) == 0
        assert result.processing_time_ms == 0.0
        
    def test_embedding_result_cached_result(self):
        """Test EmbeddingResult from cache."""
        request_id = uuid4()
        embedding = [0.2] * 3072
        
        result = EmbeddingResult(
            request_id=request_id,
            embedding=embedding,
            model_name="textembedding-gecko@003",
            processing_time_ms=2.1,
            cached=True
        )
        
        assert result.cached is True
        assert result.processing_time_ms == 2.1  # Much faster due to cache


class TestEmbeddingBatch:
    """Test EmbeddingBatch model for batch processing."""
    
    def test_embedding_batch_creation(self):
        """Test EmbeddingBatch creation with multiple requests."""
        requests = [
            EmbeddingRequest(text="Being is the most universal concept."),
            EmbeddingRequest(text="Existence precedes essence."),
            EmbeddingRequest(text="The unexamined life is not worth living.")
        ]
        
        batch = EmbeddingBatch(
            requests=requests,
            batch_size=25,
            shared_context={"document": "existentialist_texts"},
            priority="high"
        )
        
        assert isinstance(batch.id, UUID)
        assert len(batch.requests) == 3
        assert batch.batch_size == 25
        assert batch.shared_context["document"] == "existentialist_texts"
        assert batch.priority == "high"
        
    def test_embedding_batch_properties(self):
        """Test EmbeddingBatch computed properties."""
        requests = [
            EmbeddingRequest(text="Short text"),
            EmbeddingRequest(text="This is a longer text with more words"),
            EmbeddingRequest(text="Medium length text here")
        ]
        
        batch = EmbeddingBatch(requests=requests)
        
        assert batch.total_texts == 3
        # Should count words across all texts
        assert batch.estimated_tokens > 0
        
    def test_embedding_batch_empty_requests(self):
        """Test EmbeddingBatch with no requests."""
        batch = EmbeddingBatch(requests=[])
        
        assert batch.total_texts == 0
        assert batch.estimated_tokens == 0


class TestEmbeddingBatchResult:
    """Test EmbeddingBatchResult model for batch processing results."""
    
    def test_embedding_batch_result_creation(self):
        """Test EmbeddingBatchResult creation with mixed success/failure."""
        batch_id = uuid4()
        
        # Create mock results
        successful_result = EmbeddingResult(
            request_id=uuid4(),
            embedding=[0.1] * 3072,
            model_name="textembedding-gecko@003",
            processing_time_ms=100.0
        )
        
        failed_result = EmbeddingResult(
            request_id=uuid4(),
            embedding=[],
            model_name="textembedding-gecko@003",
            processing_time_ms=0.0,
            error="Network timeout"
        )
        
        batch_result = EmbeddingBatchResult(
            batch_id=batch_id,
            results=[successful_result, failed_result],
            total_processed=2,
            successful=1,
            failed=1,
            total_processing_time_ms=100.0,
            average_time_per_embedding_ms=50.0,
            cache_hit_rate=0.0,
            cost_estimate_usd=0.002,
            philosophical_content_ratio=0.95
        )
        
        assert batch_result.batch_id == batch_id
        assert len(batch_result.results) == 2
        assert batch_result.total_processed == 2
        assert batch_result.successful == 1
        assert batch_result.failed == 1
        assert batch_result.success_rate == 50.0
        assert batch_result.cost_estimate_usd == 0.002
        assert batch_result.philosophical_content_ratio == 0.95
        
    def test_embedding_batch_result_success_rate_calculation(self):
        """Test success rate calculation for various scenarios."""
        batch_id = uuid4()
        
        # 100% success
        result_100 = EmbeddingBatchResult(
            batch_id=batch_id,
            results=[],
            total_processed=10,
            successful=10,
            failed=0,
            total_processing_time_ms=1000.0,
            average_time_per_embedding_ms=100.0,
            cache_hit_rate=0.2
        )
        assert result_100.success_rate == 100.0
        
        # 0% success  
        result_0 = EmbeddingBatchResult(
            batch_id=batch_id,
            results=[],
            total_processed=5,
            successful=0,
            failed=5,
            total_processing_time_ms=0.0,
            average_time_per_embedding_ms=0.0,
            cache_hit_rate=0.0
        )
        assert result_0.success_rate == 0.0
        
        # No processing
        result_none = EmbeddingBatchResult(
            batch_id=batch_id,
            results=[],
            total_processed=0,
            successful=0,
            failed=0,
            total_processing_time_ms=0.0,
            average_time_per_embedding_ms=0.0,
            cache_hit_rate=0.0
        )
        assert result_none.success_rate == 0.0


class TestEmbeddingCache:
    """Test EmbeddingCache model for Redis storage."""
    
    def test_embedding_cache_creation(self):
        """Test EmbeddingCache creation with philosophical metadata."""
        embedding = [0.3] * 3072
        cache_entry = EmbeddingCache(
            text_hash="abc123def456",
            embedding=embedding,
            model_name="textembedding-gecko@003",
            language="de",
            philosophical_tradition="continental",
            created_at=datetime.utcnow(),
            access_count=5
        )
        
        assert cache_entry.text_hash == "abc123def456"
        assert len(cache_entry.embedding) == 3072
        assert cache_entry.language == "de"
        assert cache_entry.philosophical_tradition == "continental"
        assert cache_entry.access_count == 5
        
    def test_embedding_cache_to_redis_dict(self):
        """Test conversion to Redis-storable dictionary."""
        now = datetime.utcnow()
        expires = now + timedelta(days=7)
        
        cache_entry = EmbeddingCache(
            text_hash="hash123",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            language="en",
            philosophical_tradition="analytic",
            created_at=now,
            access_count=10,
            expires_at=expires
        )
        
        redis_dict = cache_entry.to_redis_dict()
        
        assert redis_dict["text_hash"] == "hash123"
        assert redis_dict["embedding"] == [0.1, 0.2, 0.3]
        assert redis_dict["model_name"] == "test-model"
        assert redis_dict["language"] == "en"
        assert redis_dict["philosophical_tradition"] == "analytic"
        assert redis_dict["access_count"] == 10
        assert redis_dict["created_at"] == now.isoformat()
        assert redis_dict["expires_at"] == expires.isoformat()
        
    def test_embedding_cache_to_redis_dict_no_expiry(self):
        """Test Redis dict conversion without expiry."""
        cache_entry = EmbeddingCache(
            text_hash="hash456",
            embedding=[0.4, 0.5],
            model_name="test-model",
            created_at=datetime.utcnow()
        )
        
        redis_dict = cache_entry.to_redis_dict()
        assert redis_dict["expires_at"] is None