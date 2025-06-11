"""
Database connection and session management for Philosophical Research RAG MCP Server.

Provides async PostgreSQL + pgvector connections with proper session handling
and connection pooling for philosophical research workloads.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from .models import Base


class DatabaseManager:
    """
    Manages async database connections and sessions for philosophical research data.
    
    Handles PostgreSQL + pgvector connections with proper pooling and session
    lifecycle management for concurrent philosophical researchers.
    """
    
    def __init__(self):
        self.engine = None
        self.async_session = None
        self._initialized = False
    
    async def initialize(self, database_url: str = None) -> None:
        """
        Initialize database connection with pgvector support.
        
        Args:
            database_url: PostgreSQL connection URL. If None, reads from environment.
        """
        if self._initialized:
            return
            
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql+asyncpg://philosopher:password@localhost:5432/phil_rag'
            )
        
        # Ensure we're using asyncpg driver for async PostgreSQL
        if not database_url.startswith('postgresql+asyncpg://'):
            if database_url.startswith('postgresql://'):
                database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
            else:
                raise ValueError("Database URL must be a PostgreSQL connection string")
        
        # Create async engine with optimized settings for philosophical research
        if 'test' in database_url:
            # Use NullPool for testing (no connection pooling parameters)
            self.engine = create_async_engine(
                database_url,
                echo=False,
                poolclass=NullPool,
            )
        else:
            # Use connection pooling for production
            self.engine = create_async_engine(
                database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=20,  # Support for concurrent researchers
                max_overflow=40,  # Handle burst loads during large document processing
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,  # Recycle connections hourly
            )
        
        # Create async session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects accessible after commit
            autoflush=True,  # Auto-flush before queries
            autocommit=False,  # Explicit transaction control
        )
        
        self._initialized = True
    
    async def create_tables(self) -> None:
        """
        Create all database tables with pgvector extensions.
        
        Note: This should be used for development/testing only.
        Production deployments should use Alembic migrations.
        """
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
            
        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """
        Drop all database tables.
        
        Warning: This will delete all data! Use only for testing.
        """
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
            
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with proper transaction handling.
        
        Yields:
            AsyncSession: Database session with automatic transaction management
            
        Example:
            async with db_manager.get_session() as session:
                document = await session.get(Document, document_id)
                # Session automatically committed on success or rolled back on error
        """
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
            
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
        self._initialized = False


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions for common operations
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session


async def init_database(database_url: str = None) -> None:
    """
    Initialize database connection.
    
    Args:
        database_url: PostgreSQL connection URL
    """
    await db_manager.initialize(database_url)


async def create_tables() -> None:
    """Create all database tables."""
    await db_manager.create_tables()


async def close_database() -> None:
    """Close database connections."""
    await db_manager.close()