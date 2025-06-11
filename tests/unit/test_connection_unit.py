"""
Pure unit tests for database connection management without database dependency.

Tests connection logic, session handling, and configuration using mocking
to avoid requiring a live database connection.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os
from contextlib import asynccontextmanager

from src.infrastructure.database.connection import (
    DatabaseManager, db_manager, get_db_session, 
    init_database, create_tables, close_database
)


class TestDatabaseManagerUnit:
    """Unit tests for DatabaseManager without database dependency."""
    
    def test_database_manager_initialization_state(self):
        """Test DatabaseManager initial state."""
        manager = DatabaseManager()
        
        assert manager.engine is None
        assert manager.async_session is None
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_database_manager_initialize_url_processing(self):
        """Test URL processing logic.""" 
        manager = DatabaseManager()
        
        # Mock create_async_engine to avoid actual connection
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_session:
            
            mock_engine.return_value = Mock()
            mock_session.return_value = Mock()
            
            # Test postgresql:// to postgresql+asyncpg:// conversion
            await manager.initialize("postgresql://user:pass@localhost:5432/db")
            
            # Should have been converted to asyncpg
            mock_engine.assert_called_once()
            call_args = mock_engine.call_args[0]
            assert call_args[0] == "postgresql+asyncpg://user:pass@localhost:5432/db"
            
            assert manager._initialized is True
    
    @pytest.mark.asyncio
    async def test_database_manager_test_url_handling(self):
        """Test special handling for test database URLs."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_session:
            
            mock_engine.return_value = Mock()
            mock_session.return_value = Mock()
            
            # Test URL with 'test' should use NullPool
            await manager.initialize("postgresql+asyncpg://user:pass@localhost:5432/test_db")
            
            # Should have been called with NullPool
            mock_engine.assert_called_once()
            call_kwargs = mock_engine.call_args[1]
            assert 'poolclass' in call_kwargs
            # NullPool should be specified for test databases
    
    @pytest.mark.asyncio
    async def test_database_manager_production_url_handling(self):
        """Test production database configuration."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_session:
            
            mock_engine.return_value = Mock()
            mock_session.return_value = Mock()
            
            # Production URL should use connection pooling
            await manager.initialize("postgresql+asyncpg://user:pass@localhost:5432/prod_db")
            
            mock_engine.assert_called_once()
            call_kwargs = mock_engine.call_args[1]
            assert call_kwargs.get('pool_size') == 20
            assert call_kwargs.get('max_overflow') == 40
            assert call_kwargs.get('pool_pre_ping') is True
    
    @pytest.mark.asyncio
    async def test_database_manager_invalid_url_error(self):
        """Test error handling for invalid URLs."""
        manager = DatabaseManager()
        
        # Should raise ValueError for non-PostgreSQL URLs
        with pytest.raises(ValueError, match="Database URL must be a PostgreSQL connection string"):
            await manager.initialize("mysql://user:pass@localhost:3306/db")
    
    @pytest.mark.asyncio
    async def test_database_manager_environment_url(self):
        """Test reading URL from environment variables."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_session, \
             patch.dict(os.environ, {'DATABASE_URL': 'postgresql://env:test@localhost:5432/env_db'}):
            
            mock_engine.return_value = Mock()
            mock_session.return_value = Mock()
            
            # Should read from environment when no URL provided
            await manager.initialize()
            
            mock_engine.assert_called_once()
            call_args = mock_engine.call_args[0]
            assert "env:test@localhost:5432/env_db" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_database_manager_idempotent_initialization(self):
        """Test that initialization is idempotent."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_session:
            
            mock_engine.return_value = Mock()
            mock_session.return_value = Mock()
            
            # First initialization
            await manager.initialize("postgresql://test:test@localhost:5432/test")
            
            # Second initialization should not re-initialize
            await manager.initialize("postgresql://test:test@localhost:5432/test")
            
            # Should only be called once
            mock_engine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_context_manager(self):
        """Test session context manager logic."""
        manager = DatabaseManager()
        manager._initialized = True
        
        # Mock async session
        mock_session_instance = AsyncMock()
        mock_session_factory = Mock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager.async_session = mock_session_factory
        
        async with manager.get_session() as session:
            assert session == mock_session_instance
            
        # Should have committed the session
        mock_session_instance.commit.assert_called_once()
        mock_session_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_exception_handling(self):
        """Test session rollback on exception."""
        manager = DatabaseManager()
        manager._initialized = True
        
        mock_session_instance = AsyncMock()
        mock_session_instance.commit.side_effect = Exception("Database error")
        
        mock_session_factory = Mock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager.async_session = mock_session_factory
        
        with pytest.raises(Exception, match="Database error"):
            async with manager.get_session() as session:
                pass  # commit() will raise exception
                
        # Should have attempted rollback
        mock_session_instance.rollback.assert_called_once()
        mock_session_instance.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_not_initialized_error(self):
        """Test error when getting session from uninitialized manager."""
        manager = DatabaseManager()
        
        with pytest.raises(RuntimeError, match="Database manager not initialized"):
            async with manager.get_session() as session:
                pass
    
    @pytest.mark.asyncio
    async def test_create_tables_with_pgvector(self):
        """Test table creation with pgvector extension."""
        manager = DatabaseManager()
        manager._initialized = True
        
        # Mock engine and connection
        mock_conn = AsyncMock()
        mock_engine = Mock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager.engine = mock_engine
        
        await manager.create_tables()
        
        # Should have executed pgvector extension creation
        mock_conn.execute.assert_called()
        # Should have run metadata.create_all
        mock_conn.run_sync.assert_called()
    
    @pytest.mark.asyncio
    async def test_drop_tables(self):
        """Test table dropping functionality."""
        manager = DatabaseManager()
        manager._initialized = True
        
        mock_conn = AsyncMock()
        mock_engine = Mock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        
        manager.engine = mock_engine
        
        await manager.drop_tables()
        
        # Should have run metadata.drop_all
        mock_conn.run_sync.assert_called()
    
    @pytest.mark.asyncio
    async def test_close_manager(self):
        """Test database manager cleanup."""
        manager = DatabaseManager()
        manager._initialized = True
        
        mock_engine = AsyncMock()
        manager.engine = mock_engine
        
        await manager.close()
        
        mock_engine.dispose.assert_called_once()
        assert manager._initialized is False


class TestConvenienceFunctionsUnit:
    """Unit tests for convenience functions without database dependency."""
    
    @pytest.mark.asyncio
    async def test_get_db_session_function(self):
        """Test get_db_session convenience function."""
        mock_session = AsyncMock()
        
        # Create proper async context manager mock
        @asynccontextmanager
        async def mock_get_session():
            yield mock_session
        
        # Mock the global db_manager
        with patch('src.infrastructure.database.connection.db_manager') as mock_manager:
            mock_manager.get_session = mock_get_session
            
            async with get_db_session() as session:
                assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_init_database_function(self):
        """Test init_database convenience function."""
        with patch('src.infrastructure.database.connection.db_manager') as mock_manager:
            mock_manager.initialize = AsyncMock()
            
            await init_database("postgresql://test:test@localhost:5432/test")
            
            mock_manager.initialize.assert_called_once_with("postgresql://test:test@localhost:5432/test")
    
    @pytest.mark.asyncio
    async def test_create_tables_function(self):
        """Test create_tables convenience function."""
        with patch('src.infrastructure.database.connection.db_manager') as mock_manager:
            mock_manager.create_tables = AsyncMock()
            
            await create_tables()
            
            mock_manager.create_tables.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database_function(self):
        """Test close_database convenience function."""
        with patch('src.infrastructure.database.connection.db_manager') as mock_manager:
            mock_manager.close = AsyncMock()
            
            await close_database()
            
            mock_manager.close.assert_called_once()


class TestDatabaseManagerConfiguration:
    """Test database manager configuration logic."""
    
    @pytest.mark.asyncio
    async def test_test_database_configuration(self):
        """Test configuration for test databases."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            await manager.initialize("postgresql+asyncpg://test:test@localhost:5432/phil_rag_test")
            
            # Should use NullPool for test databases
            call_kwargs = mock_engine.call_args[1]
            assert 'poolclass' in call_kwargs
            # Should NOT have pool_size for NullPool
            assert 'pool_size' not in call_kwargs
    
    @pytest.mark.asyncio
    async def test_production_database_configuration(self):
        """Test configuration for production databases."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker'):
            
            mock_engine.return_value = Mock()
            
            await manager.initialize("postgresql+asyncpg://prod:pass@localhost:5432/phil_rag_prod")
            
            # Should use connection pooling for production
            call_kwargs = mock_engine.call_args[1]
            assert call_kwargs['pool_size'] == 20
            assert call_kwargs['max_overflow'] == 40
            assert call_kwargs['pool_pre_ping'] is True
            assert call_kwargs['pool_recycle'] == 3600
    
    @pytest.mark.asyncio
    async def test_session_factory_configuration(self):
        """Test async session factory configuration."""
        manager = DatabaseManager()
        
        with patch('src.infrastructure.database.connection.create_async_engine') as mock_engine, \
             patch('src.infrastructure.database.connection.async_sessionmaker') as mock_sessionmaker:
            
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance
            mock_sessionmaker.return_value = Mock()
            
            await manager.initialize("postgresql+asyncpg://test:test@localhost:5432/test")
            
            # Verify session factory configuration
            mock_sessionmaker.assert_called_once()
            call_kwargs = mock_sessionmaker.call_args[1]
            
            assert call_kwargs['expire_on_commit'] is False
            assert call_kwargs['autoflush'] is True
            assert call_kwargs['autocommit'] is False