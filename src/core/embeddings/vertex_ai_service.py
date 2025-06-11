"""
Vertex AI embedding service for philosophical research.

Provides async embedding generation using Google Cloud Vertex AI
with optimizations for philosophical texts and academic research.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

import redis.asyncio as redis
from google.cloud import aiplatform
from google.cloud.aiplatform import TextEmbeddingModel
from vertexai.preview.language_models import TextEmbeddingInput

from .embedding_models import (
    EmbeddingRequest,
    EmbeddingResult, 
    EmbeddingBatch,
    EmbeddingBatchResult,
    EmbeddingCache
)

logger = logging.getLogger(__name__)


class VertexAIEmbeddingService:
    """
    Vertex AI embedding service optimized for philosophical research.
    
    Features:
    - Async batch processing for high throughput
    - Redis caching for performance optimization
    - Multi-language support for philosophical texts
    - Retry logic with exponential backoff
    - Cost tracking and optimization
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "textembedding-gecko@003",
        redis_url: Optional[str] = None,
        cache_ttl_days: int = 30,
        max_batch_size: int = 50,
        max_retries: int = 3
    ):
        """
        Initialize Vertex AI embedding service.
        
        Args:
            project_id: Google Cloud project ID
            location: Vertex AI region
            model_name: Embedding model to use
            redis_url: Redis connection URL for caching
            cache_ttl_days: Cache expiration in days
            max_batch_size: Maximum texts per batch
            max_retries: Maximum retry attempts
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.cache_ttl_days = cache_ttl_days
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        
        # Initialize Redis cache if URL provided
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
            
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_processing_time_ms": 0.0,
            "errors": 0
        }
    
    async def generate_embedding(
        self, 
        request: EmbeddingRequest,
        use_cache: bool = True
    ) -> EmbeddingResult:
        """
        Generate embedding for a single philosophical text.
        
        Args:
            request: Embedding request with text and metadata
            use_cache: Whether to use Redis cache
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        start_time = datetime.utcnow()
        self.stats["total_requests"] += 1
        
        try:
            # Check cache first
            if use_cache and self.redis_client:
                cached_result = await self._get_from_cache(request)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return EmbeddingResult(
                        request_id=request.id,
                        embedding=cached_result.embedding,
                        model_name=cached_result.model_name,
                        language_detected=cached_result.language,
                        processing_time_ms=processing_time,
                        cached=True
                    )
            
            # Generate new embedding
            embedding_vector = await self._call_vertex_ai_single(request.text)
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create result
            result = EmbeddingResult(
                request_id=request.id,
                embedding=embedding_vector,
                model_name=self.model_name,
                language_detected=self._detect_language(request.text),
                philosophical_score=self._assess_philosophical_content(request.text),
                processing_time_ms=processing_time,
                cached=False
            )
            
            # Cache result
            if use_cache and self.redis_client:
                await self._store_in_cache(request, result)
                
            self.stats["api_calls"] += 1
            self.stats["total_processing_time_ms"] += processing_time
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Embedding generation failed for request {request.id}: {e}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return EmbeddingResult(
                request_id=request.id,
                embedding=[],
                model_name=self.model_name,
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    async def generate_batch_embeddings(
        self,
        batch: EmbeddingBatch,
        use_cache: bool = True
    ) -> EmbeddingBatchResult:
        """
        Generate embeddings for a batch of philosophical texts.
        
        Args:
            batch: Batch of embedding requests
            use_cache: Whether to use Redis cache
            
        Returns:
            EmbeddingBatchResult with all individual results
        """
        start_time = datetime.utcnow()
        
        # Process in smaller chunks if batch is large
        all_results = []
        chunks = self._chunk_requests(batch.requests, batch.batch_size)
        
        for chunk in chunks:
            # Process chunk concurrently
            tasks = [
                self.generate_embedding(req, use_cache)
                for req in chunk
            ]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    error_result = EmbeddingResult(
                        request_id=chunk[i].id,
                        embedding=[],
                        model_name=self.model_name,
                        processing_time_ms=0.0,
                        error=str(result)
                    )
                    all_results.append(error_result)
                else:
                    all_results.append(result)
        
        # Calculate batch statistics
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        successful = sum(1 for r in all_results if not r.error)
        failed = len(all_results) - successful
        cache_hits = sum(1 for r in all_results if r.cached)
        
        return EmbeddingBatchResult(
            batch_id=batch.id,
            results=all_results,
            total_processed=len(all_results),
            successful=successful,
            failed=failed,
            total_processing_time_ms=total_time,
            average_time_per_embedding_ms=total_time / len(all_results) if all_results else 0.0,
            cache_hit_rate=cache_hits / len(all_results) if all_results else 0.0,
            cost_estimate_usd=self._estimate_cost(successful, cache_hits),
            philosophical_content_ratio=self._calculate_philosophical_ratio(all_results)
        )
    
    async def _call_vertex_ai_single(self, text: str) -> List[float]:
        """
        Call Vertex AI API for single text embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            3072-dimensional embedding vector
        """
        # Prepare input
        embedding_input = TextEmbeddingInput(text=text, task_type="SEMANTIC_SIMILARITY")
        
        # Call API with retry logic
        for attempt in range(self.max_retries):
            try:
                embeddings = await asyncio.to_thread(
                    self.model.get_embeddings,
                    [embedding_input]
                )
                return embeddings[0].values
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                    
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Vertex AI API attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        
        raise RuntimeError("All Vertex AI API attempts failed")
    
    async def _get_from_cache(self, request: EmbeddingRequest) -> Optional[EmbeddingCache]:
        """
        Retrieve embedding from Redis cache.
        
        Args:
            request: Embedding request
            
        Returns:
            Cached embedding or None
        """
        if not self.redis_client:
            return None
            
        try:
            text_hash = self._hash_text(request.text)
            cache_key = f"embedding:{text_hash}:{self.model_name}"
            
            cached_data = await self.redis_client.hgetall(cache_key)
            if not cached_data:
                return None
                
            # Parse cached data
            embedding = json.loads(cached_data[b"embedding"].decode())
            
            cache_entry = EmbeddingCache(
                text_hash=text_hash,
                embedding=embedding,
                model_name=cached_data[b"model_name"].decode(),
                language=cached_data.get(b"language", b"").decode() or None,
                philosophical_tradition=cached_data.get(b"philosophical_tradition", b"").decode() or None,
                created_at=datetime.fromisoformat(cached_data[b"created_at"].decode()),
                access_count=int(cached_data.get(b"access_count", 0)),
                last_accessed=datetime.utcnow()
            )
            
            # Update access tracking
            await self.redis_client.hincrby(cache_key, "access_count", 1)
            await self.redis_client.hset(cache_key, "last_accessed", datetime.utcnow().isoformat())
            
            return cache_entry
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _store_in_cache(self, request: EmbeddingRequest, result: EmbeddingResult) -> None:
        """
        Store embedding result in Redis cache.
        
        Args:
            request: Original embedding request
            result: Generated embedding result
        """
        if not self.redis_client or result.error:
            return
            
        try:
            text_hash = self._hash_text(request.text)
            cache_key = f"embedding:{text_hash}:{self.model_name}"
            expires_at = datetime.utcnow() + timedelta(days=self.cache_ttl_days)
            
            cache_entry = EmbeddingCache(
                text_hash=text_hash,
                embedding=result.embedding,
                model_name=result.model_name,
                language=request.language,
                philosophical_tradition=request.philosophical_tradition,
                created_at=result.generated_at,
                expires_at=expires_at
            )
            
            # Store in Redis
            cache_data = cache_entry.to_redis_dict()
            await self.redis_client.hset(cache_key, mapping=cache_data)
            await self.redis_client.expire(cache_key, int(timedelta(days=self.cache_ttl_days).total_seconds()))
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Generate SHA-256 hash for text caching."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _chunk_requests(self, requests: List[EmbeddingRequest], chunk_size: int) -> List[List[EmbeddingRequest]]:
        """Split requests into smaller chunks for batch processing."""
        return [requests[i:i + chunk_size] for i in range(0, len(requests), chunk_size)]
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of philosophical text.
        
        Simple heuristic-based detection for common philosophical languages.
        """
        # Ancient Greek detection
        if any(ord(char) >= 0x0370 and ord(char) <= 0x03FF for char in text):
            return "grc"
        
        # Latin detection (common philosophical terms)
        latin_terms = ["cogito", "esse", "ens", "ratio", "natura", "substantia"]
        if any(term in text.lower() for term in latin_terms):
            return "la"
            
        # German detection (common philosophical terms)
        german_terms = ["sein", "dasein", "bewusstsein", "erscheinung", "begriff"]
        if any(term in text.lower() for term in german_terms):
            return "de"
            
        # Default to English
        return "en"
    
    def _assess_philosophical_content(self, text: str) -> float:
        """
        Assess how philosophical the content is (0.0 to 1.0).
        
        Uses keyword matching for philosophical terms and concepts.
        """
        philosophical_terms = [
            "being", "existence", "essence", "consciousness", "reality", "truth",
            "knowledge", "epistemology", "metaphysics", "ontology", "ethics",
            "phenomenology", "dialectic", "transcendental", "empirical", "rational",
            "categorical", "synthetic", "analytic", "a priori", "a posteriori"
        ]
        
        text_lower = text.lower()
        matches = sum(1 for term in philosophical_terms if term in text_lower)
        
        # Simple scoring based on term density
        words = len(text.split())
        if words == 0:
            return 0.0
            
        return min(matches / words * 10, 1.0)  # Scale to 0-1 range
    
    def _estimate_cost(self, successful_requests: int, cache_hits: int) -> float:
        """
        Estimate API cost in USD.
        
        Based on Vertex AI pricing for text embedding model.
        """
        api_calls = successful_requests - cache_hits
        # Approximate cost: $0.0001 per 1000 characters
        # Assuming average 500 characters per request
        return api_calls * 0.0001 * 0.5
    
    def _calculate_philosophical_ratio(self, results: List[EmbeddingResult]) -> float:
        """
        Calculate ratio of philosophical content in batch.
        
        Args:
            results: List of embedding results
            
        Returns:
            Ratio of philosophical content (0.0 to 1.0)
        """
        if not results:
            return 0.0
            
        philosophical_scores = [
            r.philosophical_score for r in results 
            if r.philosophical_score is not None and not r.error
        ]
        
        if not philosophical_scores:
            return 0.0
            
        return sum(philosophical_scores) / len(philosophical_scores)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1),
            "average_processing_time_ms": self.stats["total_processing_time_ms"] / max(self.stats["api_calls"], 1),
            "error_rate": self.stats["errors"] / max(self.stats["total_requests"], 1)
        }
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()