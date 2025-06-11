"""
Unit tests for Vertex AI embedding service.

Tests the Vertex AI service without external dependencies using mocking.
Following TDD methodology for philosophical research requirements.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

# Mock Google Cloud dependencies before importing
import sys
from unittest.mock import Mock

# Mock the google.cloud modules
sys.modules['google.cloud'] = Mock()
sys.modules['google.cloud.aiplatform'] = Mock()
sys.modules['vertexai.preview.language_models'] = Mock()

from src.core.embeddings.vertex_ai_service import VertexAIEmbeddingService
from src.core.embeddings.embedding_models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingBatch,
    EmbeddingCache
)


class TestVertexAIEmbeddingService:
    """Test VertexAIEmbeddingService with mocked dependencies."""
    
    @pytest.fixture
    def mock_vertex_ai_service(self):
        """Create VertexAIEmbeddingService with mocked dependencies."""
        with patch('src.core.embeddings.vertex_ai_service.aiplatform'), \
             patch('src.core.embeddings.vertex_ai_service.TextEmbeddingModel') as mock_model_class, \
             patch('src.core.embeddings.vertex_ai_service.redis') as mock_redis:
            
            # Mock the TextEmbeddingModel
            mock_model = Mock()
            mock_model_class.from_pretrained.return_value = mock_model
            
            service = VertexAIEmbeddingService(
                project_id="test-project",
                location="us-central1",
                redis_url="redis://localhost:6379"
            )
            
            # Override the model with our mock
            service.model = mock_model
            
            # Mock Redis client
            mock_redis_client = AsyncMock()
            service.redis_client = mock_redis_client
            
            return service, mock_model, mock_redis_client
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test VertexAIEmbeddingService initialization."""
        with patch('src.core.embeddings.vertex_ai_service.aiplatform') as mock_aiplatform, \
             patch('src.core.embeddings.vertex_ai_service.TextEmbeddingModel') as mock_model_class:
            
            service = VertexAIEmbeddingService(
                project_id="test-project",
                location="us-west1",
                model_name="textembedding-gecko@latest",
                cache_ttl_days=7,
                max_batch_size=25
            )
            
            # Verify initialization
            mock_aiplatform.init.assert_called_once_with(
                project="test-project",
                location="us-west1"
            )
            mock_model_class.from_pretrained.assert_called_once_with("textembedding-gecko@latest")
            
            assert service.project_id == "test-project"
            assert service.location == "us-west1"
            assert service.model_name == "textembedding-gecko@latest"
            assert service.cache_ttl_days == 7
            assert service.max_batch_size == 25
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, mock_vertex_ai_service):
        """Test successful embedding generation."""
        service, mock_model, mock_redis_client = mock_vertex_ai_service
        
        # Mock Vertex AI response
        mock_embedding_response = Mock()
        mock_embedding_response.values = [0.1] * 3072
        
        with patch.object(service, '_call_vertex_ai_single', return_value=[0.1] * 3072) as mock_api_call:
            request = EmbeddingRequest(
                text="Being and Time explores the fundamental structures of human existence.",
                language="en",
                philosophical_tradition="continental"
            )
            
            result = await service.generate_embedding(request, use_cache=False)
            
            assert isinstance(result, EmbeddingResult)
            assert result.request_id == request.id
            assert len(result.embedding) == 3072
            assert all(v == 0.1 for v in result.embedding)
            assert result.model_name == service.model_name
            assert result.cached is False
            assert result.error is None
            assert result.processing_time_ms > 0
            
            mock_api_call.assert_called_once_with(request.text)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_cache_hit(self, mock_vertex_ai_service):
        """Test embedding generation with cache hit."""
        service, mock_model, mock_redis_client = mock_vertex_ai_service
        
        # Mock cache hit
        cached_embedding = EmbeddingCache(
            text_hash="test_hash",
            embedding=[0.2] * 3072,
            model_name="textembedding-gecko@003",
            language="en",
            philosophical_tradition="continental",
            created_at=datetime.utcnow()
        )
        
        with patch.object(service, '_get_from_cache', return_value=cached_embedding):
            request = EmbeddingRequest(
                text="Existence precedes essence.",
                language="en"
            )
            
            result = await service.generate_embedding(request, use_cache=True)
            
            assert result.cached is True
            assert len(result.embedding) == 3072
            assert all(v == 0.2 for v in result.embedding)
            assert result.processing_time_ms < 100  # Should be very fast from cache
    
    @pytest.mark.asyncio
    async def test_generate_embedding_error_handling(self, mock_vertex_ai_service):
        """Test embedding generation error handling."""
        service, mock_model, mock_redis_client = mock_vertex_ai_service
        
        # Mock API failure
        with patch.object(service, '_call_vertex_ai_single', side_effect=Exception("API Error")):
            request = EmbeddingRequest(text="Test text")
            
            result = await service.generate_embedding(request, use_cache=False)
            
            assert result.error == "API Error"
            assert len(result.embedding) == 0
            assert result.cached is False
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, mock_vertex_ai_service):
        """Test batch embedding generation."""
        service, mock_model, mock_redis_client = mock_vertex_ai_service
        
        # Create batch of requests
        requests = [
            EmbeddingRequest(text="First philosophical text"),
            EmbeddingRequest(text="Second philosophical text"),
            EmbeddingRequest(text="Third philosophical text")
        ]
        batch = EmbeddingBatch(requests=requests, batch_size=2)
        
        # Mock successful results
        with patch.object(service, 'generate_embedding') as mock_generate:
            mock_results = [
                EmbeddingResult(
                    request_id=req.id,
                    embedding=[0.1] * 3072,
                    model_name=service.model_name,
                    processing_time_ms=100.0,
                    philosophical_score=0.8
                ) for req in requests
            ]
            mock_generate.side_effect = mock_results
            
            batch_result = await service.generate_batch_embeddings(batch)
            
            assert batch_result.batch_id == batch.id
            assert len(batch_result.results) == 3
            assert batch_result.total_processed == 3
            assert batch_result.successful == 3
            assert batch_result.failed == 0
            assert batch_result.success_rate == 100.0
            assert abs(batch_result.philosophical_content_ratio - 0.8) < 0.001
            
            # Verify all requests were processed
            assert mock_generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_with_failures(self, mock_vertex_ai_service):
        """Test batch embedding generation with some failures."""
        service, mock_model, mock_redis_client = mock_vertex_ai_service
        
        requests = [
            EmbeddingRequest(text="Success text"),
            EmbeddingRequest(text="Failure text")
        ]
        batch = EmbeddingBatch(requests=requests)
        
        # Mock mixed results
        success_result = EmbeddingResult(
            request_id=requests[0].id,
            embedding=[0.1] * 3072,
            model_name=service.model_name,
            processing_time_ms=100.0
        )
        failure_result = EmbeddingResult(
            request_id=requests[1].id,
            embedding=[],
            model_name=service.model_name,
            processing_time_ms=0.0,
            error="API Error"
        )
        
        with patch.object(service, 'generate_embedding', side_effect=[success_result, failure_result]):
            batch_result = await service.generate_batch_embeddings(batch)
            
            assert batch_result.total_processed == 2
            assert batch_result.successful == 1
            assert batch_result.failed == 1
            assert batch_result.success_rate == 50.0
    
    def test_hash_text(self, mock_vertex_ai_service):
        """Test text hashing for cache keys."""
        service, _, _ = mock_vertex_ai_service
        
        text1 = "Being and Time"
        text2 = "Being and Time"
        text3 = "Different text"
        
        hash1 = service._hash_text(text1)
        hash2 = service._hash_text(text2)
        hash3 = service._hash_text(text3)
        
        assert hash1 == hash2  # Same text should have same hash
        assert hash1 != hash3  # Different text should have different hash
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string
    
    def test_chunk_requests(self, mock_vertex_ai_service):
        """Test request chunking for batch processing."""
        service, _, _ = mock_vertex_ai_service
        
        requests = [EmbeddingRequest(text=f"Text {i}") for i in range(7)]
        chunks = service._chunk_requests(requests, chunk_size=3)
        
        assert len(chunks) == 3  # 7 requests with chunk_size=3 should create 3 chunks
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 1
    
    def test_detect_language(self, mock_vertex_ai_service):
        """Test language detection for philosophical texts."""
        service, _, _ = mock_vertex_ai_service
        
        # Test ancient Greek
        greek_text = "τὸ ὂν λέγεται πολλαχῶς"
        assert service._detect_language(greek_text) == "grc"
        
        # Test Latin
        latin_text = "Cogito ergo sum"
        assert service._detect_language(latin_text) == "la"
        
        # Test German
        german_text = "Das Sein und die Zeit"
        assert service._detect_language(german_text) == "de"
        
        # Test English (default)
        english_text = "Being and existence"
        assert service._detect_language(english_text) == "en"
    
    def test_assess_philosophical_content(self, mock_vertex_ai_service):
        """Test philosophical content assessment."""
        service, _, _ = mock_vertex_ai_service
        
        # Highly philosophical text
        philosophical_text = "Being, existence, consciousness, and metaphysical reality are fundamental ontological concepts."
        score1 = service._assess_philosophical_content(philosophical_text)
        assert score1 > 0.5
        
        # Non-philosophical text
        mundane_text = "The weather is nice today and I like ice cream."
        score2 = service._assess_philosophical_content(mundane_text)
        assert score2 < 0.3
        
        # Empty text
        empty_score = service._assess_philosophical_content("")
        assert empty_score == 0.0
    
    def test_estimate_cost(self, mock_vertex_ai_service):
        """Test API cost estimation."""
        service, _, _ = mock_vertex_ai_service
        
        # Test with no cache hits
        cost1 = service._estimate_cost(successful_requests=100, cache_hits=0)
        assert cost1 > 0
        
        # Test with cache hits (should be cheaper)
        cost2 = service._estimate_cost(successful_requests=100, cache_hits=50)
        assert cost2 < cost1
        
        # Test with all cache hits
        cost3 = service._estimate_cost(successful_requests=100, cache_hits=100)
        assert cost3 == 0.0
    
    def test_calculate_philosophical_ratio(self, mock_vertex_ai_service):
        """Test philosophical content ratio calculation."""
        service, _, _ = mock_vertex_ai_service
        
        results = [
            EmbeddingResult(
                request_id=uuid4(),
                embedding=[0.1] * 3072,
                model_name="test",
                processing_time_ms=100.0,
                philosophical_score=0.8
            ),
            EmbeddingResult(
                request_id=uuid4(),
                embedding=[0.1] * 3072,
                model_name="test",
                processing_time_ms=100.0,
                philosophical_score=0.6
            ),
            EmbeddingResult(
                request_id=uuid4(),
                embedding=[],
                model_name="test",
                processing_time_ms=0.0,
                error="Failed"
            )
        ]
        
        ratio = service._calculate_philosophical_ratio(results)
        assert ratio == 0.7  # (0.8 + 0.6) / 2 = 0.7 (excluding failed result)
        
        # Test empty results
        empty_ratio = service._calculate_philosophical_ratio([])
        assert empty_ratio == 0.0
    
    @pytest.mark.asyncio
    async def test_get_stats(self, mock_vertex_ai_service):
        """Test service statistics retrieval."""
        service, _, _ = mock_vertex_ai_service
        
        # Simulate some activity
        service.stats["total_requests"] = 100
        service.stats["cache_hits"] = 30
        service.stats["api_calls"] = 70
        service.stats["total_processing_time_ms"] = 7000.0
        service.stats["errors"] = 5
        
        stats = await service.get_stats()
        
        assert stats["total_requests"] == 100
        assert stats["cache_hits"] == 30
        assert stats["cache_hit_rate"] == 0.3
        assert stats["average_processing_time_ms"] == 100.0
        assert stats["error_rate"] == 0.05
    
    @pytest.mark.asyncio
    async def test_close_redis_connection(self, mock_vertex_ai_service):
        """Test Redis connection cleanup."""
        service, _, mock_redis_client = mock_vertex_ai_service
        
        await service.close()
        mock_redis_client.close.assert_called_once()