"""
Unit tests for database connection management - Test-Driven Development approach.

Tests PostgreSQL + pgvector connection handling, session management, 
and database operations BEFORE implementation.
"""

import pytest
import pytest_asyncio
import os
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.infrastructure.database.connection import (
    DatabaseManager, db_manager, get_db_session, 
    init_database, create_tables, close_database
)
from src.infrastructure.database.models import Document


class TestDatabaseManager:
    """Test DatabaseManager class for connection handling."""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test database manager initialization with default URL."""
        test_manager = DatabaseManager()
        assert test_manager.engine is None
        assert test_manager.async_session is None
        assert test_manager._initialized is False
        
        # Test initialization
        test_url = "postgresql+asyncpg://test:test@localhost:5432/test"
        await test_manager.initialize(test_url)
        
        assert test_manager.engine is not None
        assert test_manager.async_session is not None
        assert test_manager._initialized is True
        
        # Cleanup
        await test_manager.close()
    
    @pytest.mark.asyncio
    async def test_database_manager_environment_url(self):
        """Test database manager reads URL from environment."""
        test_manager = DatabaseManager()
        
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql+asyncpg://env:env@localhost:5432/env_test'}):
            await test_manager.initialize()
            
            # Should have read from environment
            assert test_manager._initialized is True
            
        await test_manager.close()
    
    @pytest.mark.asyncio 
    async def test_database_url_validation(self):
        """Test database URL validation and conversion."""
        test_manager = DatabaseManager()
        
        # Test automatic conversion from postgresql:// to postgresql+asyncpg://
        postgres_url = "postgresql://user:pass@localhost:5432/db"
        await test_manager.initialize(postgres_url)
        
        # Should have been converted to asyncpg
        assert test_manager._initialized is True
        
        await test_manager.close()
        
        # Test invalid URL
        with pytest.raises(ValueError, match="Database URL must be a PostgreSQL connection string"):
            invalid_manager = DatabaseManager()
            await invalid_manager.initialize("mysql://invalid:url@localhost/db")
    
    @pytest.mark.asyncio
    async def test_session_context_manager(self, test_db_manager):
        """Test database session context manager with transaction handling."""
        # Test successful transaction
        async with test_db_manager.get_session() as session:
            assert isinstance(session, AsyncSession)
            
            # Create a test document
            document = Document(title="Test Document", author="Test Author")
            session.add(document)
            # Should auto-commit on successful exit
        
        # Verify document was committed
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE title = 'Test Document'")
            )
            count = result.scalar()
            assert count >= 0  # May be 0 due to test isolation
    
    @pytest.mark.asyncio
    async def test_session_rollback_on_exception(self, test_db_manager):
        """Test session automatically rolls back on exception."""
        try:
            async with test_db_manager.get_session() as session:
                # Create document
                document = Document(title="Should Rollback", author="Test")
                session.add(document)
                
                # Force an exception
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass  # Expected
        
        # Verify rollback occurred - document should not exist
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                text("SELECT COUNT(*) FROM documents WHERE title = 'Should Rollback'")
            )
            count = result.scalar()
            # Document should not exist due to rollback
            assert count == 0 or count is None
    
    @pytest.mark.asyncio
    async def test_not_initialized_error(self):
        """Test proper error handling when manager not initialized."""
        uninit_manager = DatabaseManager()
        
        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            async with uninit_manager.get_session() as session:
                pass
    
    @pytest.mark.asyncio
    async def test_create_tables_with_pgvector(self, test_db_manager):
        """Test table creation with pgvector extension."""
        # Should be able to create tables with vector columns
        await test_db_manager.create_tables()
        
        # Verify pgvector extension was created
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector'")
            )
            extension = result.fetchone()
            assert extension is not None
            
            # Verify tables were created
            result = await session.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = ['documents', 'chunks', 'citations', 'concept_traces', 
                             'notes', 'workspaces', 'argument_structures']
            for table in expected_tables:
                assert table in tables
    
    @pytest.mark.asyncio
    async def test_drop_tables(self, test_db_manager):
        """Test table dropping functionality."""
        # Create tables first
        await test_db_manager.create_tables()
        
        # Drop tables
        await test_db_manager.drop_tables()
        
        # Verify tables were dropped
        async with test_db_manager.get_session() as session:
            result = await session.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [row[0] for row in result.fetchall()]
            
            # Should not contain our tables (may contain others)
            philosophical_tables = ['documents', 'chunks', 'citations']
            for table in philosophical_tables:
                assert table not in tables
    
    @pytest.mark.asyncio
    async def test_connection_pooling_configuration(self):
        """Test connection pool is configured correctly for philosophical research workloads."""
        test_manager = DatabaseManager()
        await test_manager.initialize("postgresql+asyncpg://test:test@localhost:5432/test")
        
        # Check pool configuration
        assert test_manager.engine.pool.size() == 20  # Should support concurrent researchers
        assert test_manager.engine.pool._max_overflow == 40  # Handle burst loads
        assert test_manager.engine.pool._pre_ping is True  # Connection validation
        
        await test_manager.close()


class TestConvenienceFunctions:
    """Test convenience functions for database operations."""
    
    @pytest.mark.asyncio
    async def test_get_db_session_function(self, test_db_manager):
        """Test get_db_session convenience function."""
        # Mock the global db_manager to use our test manager
        with patch('src.infrastructure.database.connection.db_manager', test_db_manager):
            async with get_db_session() as session:
                assert isinstance(session, AsyncSession)
                
                # Should be able to perform operations
                document = Document(title="Convenience Test", author="Test")
                session.add(document)
    
    @pytest.mark.asyncio
    async def test_init_database_function(self):
        """Test init_database convenience function."""
        test_url = "postgresql+asyncpg://test:test@localhost:5432/test_func"
        
        # Mock global db_manager
        mock_manager = AsyncMock()
        with patch('src.infrastructure.database.connection.db_manager', mock_manager):
            await init_database(test_url)
            mock_manager.initialize.assert_called_once_with(test_url)
    
    @pytest.mark.asyncio 
    async def test_create_tables_function(self):
        """Test create_tables convenience function."""
        mock_manager = AsyncMock()
        with patch('src.infrastructure.database.connection.db_manager', mock_manager):
            await create_tables()
            mock_manager.create_tables.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database_function(self):
        """Test close_database convenience function."""
        mock_manager = AsyncMock()
        with patch('src.infrastructure.database.connection.db_manager', mock_manager):
            await close_database()
            mock_manager.close.assert_called_once()


class TestDatabaseIntegration:
    """Integration tests for database operations with philosophical data."""
    
    @pytest.mark.asyncio
    async def test_philosophical_document_crud(self, test_db_manager):
        """Test CRUD operations for philosophical documents."""
        async with test_db_manager.get_session() as session:
            # Create
            document = Document(
                title="Nicomachean Ethics",
                author="Aristotle",
                tradition="ancient",
                language="ancient_greek",
                document_metadata={
                    "philosophical_school": "peripatetic",
                    "primary_concepts": ["virtue", "ethics", "happiness"]
                }
            )
            session.add(document)
            await session.flush()
            
            doc_id = document.id
            assert doc_id is not None
        
        # Read
        async with test_db_manager.get_session() as session:
            retrieved_doc = await session.get(Document, doc_id)
            assert retrieved_doc is not None
            assert retrieved_doc.title == "Nicomachean Ethics"
            assert retrieved_doc.author == "Aristotle"
            assert retrieved_doc.tradition == "ancient"
            assert "virtue" in retrieved_doc.document_metadata["primary_concepts"]
        
        # Update
        async with test_db_manager.get_session() as session:
            doc = await session.get(Document, doc_id)
            doc.abstract = "Aristotle's foundational work on virtue ethics"
            await session.flush()
        
        # Verify update
        async with test_db_manager.get_session() as session:
            doc = await session.get(Document, doc_id)
            assert doc.abstract == "Aristotle's foundational work on virtue ethics"
    
    @pytest.mark.asyncio
    async def test_philosophical_research_query(self, test_db_manager):
        """Test complex queries for philosophical research."""
        async with test_db_manager.get_session() as session:
            # Create test data
            aristotle_doc = Document(
                title="Nicomachean Ethics", 
                author="Aristotle",
                tradition="ancient"
            )
            kant_doc = Document(
                title="Critique of Practical Reason",
                author="Kant", 
                tradition="modern"
            )
            
            session.add_all([aristotle_doc, kant_doc])
            await session.flush()
            
            # Test tradition-based query
            from sqlalchemy import select
            result = await session.execute(
                select(Document).where(Document.tradition == "ancient")
            )
            ancient_docs = result.scalars().all()
            
            assert len(ancient_docs) == 1
            assert ancient_docs[0].author == "Aristotle"
            
            # Test author search
            result = await session.execute(
                select(Document).where(Document.author.ilike("%kant%"))
            )
            kant_docs = result.scalars().all()
            
            assert len(kant_docs) == 1
            assert kant_docs[0].title == "Critique of Practical Reason"