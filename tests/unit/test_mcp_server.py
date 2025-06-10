"""
Unit tests for MCP Server - Test-Driven Development approach.

Tests MCP protocol implementation, tool registration, and philosophical
research workflow integration BEFORE implementation.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from src.mcp_server.server import PhilosophicalRAGServer


class TestPhilosophicalRAGServer:
    """Test MCP Server initialization and basic functionality."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server creates with correct name and state."""
        server = PhilosophicalRAGServer()
        
        assert server.server.name == "philosophical-rag-mcp"
        assert server._initialized is False
        assert server.server is not None
    
    @pytest.mark.asyncio
    async def test_server_initialize_database_connection(self):
        """Test server initializes database connection on startup."""
        server = PhilosophicalRAGServer()
        
        # Mock database initialization
        with patch('src.mcp_server.server.init_database') as mock_init_db:
            mock_init_db.return_value = None
            
            # Mock tool/resource/prompt registration
            with patch.object(server, '_register_tools', new_callable=AsyncMock), \
                 patch.object(server, '_register_resources', new_callable=AsyncMock), \
                 patch.object(server, '_register_prompts', new_callable=AsyncMock):
                
                await server.initialize()
                
                mock_init_db.assert_called_once()
                assert server._initialized is True
    
    @pytest.mark.asyncio 
    async def test_server_initialize_idempotent(self):
        """Test server initialization is idempotent."""
        server = PhilosophicalRAGServer()
        
        with patch('src.mcp_server.server.init_database') as mock_init_db, \
             patch.object(server, '_register_tools', new_callable=AsyncMock) as mock_tools, \
             patch.object(server, '_register_resources', new_callable=AsyncMock), \
             patch.object(server, '_register_prompts', new_callable=AsyncMock):
            
            # First initialization
            await server.initialize()
            
            # Second initialization should not re-initialize
            await server.initialize()
            
            # Should only be called once
            mock_init_db.assert_called_once()
            mock_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_initialization_database_error(self):
        """Test server handles database initialization errors."""
        server = PhilosophicalRAGServer()
        
        with patch('src.mcp_server.server.init_database') as mock_init_db:
            mock_init_db.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await server.initialize()
            
            assert server._initialized is False


class TestMCPToolRegistration:
    """Test MCP tool registration for philosophical research."""
    
    @pytest.mark.asyncio
    async def test_register_search_tools(self):
        """Test registration of philosophical search tools."""
        server = PhilosophicalRAGServer()
        
        # Mock search tools
        mock_search_tools = [
            Mock(name="search_philosophical_texts"),
            Mock(name="trace_concept_genealogy"),
            Mock(name="find_citations")
        ]
        
        with patch('src.mcp_server.server.get_search_tools') as mock_get_tools:
            mock_get_tools.return_value = mock_search_tools
            
            await server._register_tools()
            
            # Should have registered search tools
            mock_get_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_citation_tools(self):
        """Test registration of citation management tools."""
        server = PhilosophicalRAGServer()
        
        mock_citation_tools = [
            Mock(name="extract_citations"),
            Mock(name="validate_citations"),
            Mock(name="sync_zotero")
        ]
        
        with patch('src.mcp_server.server.get_search_tools', return_value=[]), \
             patch('src.mcp_server.server.get_citation_tools') as mock_get_citations, \
             patch('src.mcp_server.server.get_analysis_tools', return_value=[]):
            
            mock_get_citations.return_value = mock_citation_tools
            
            await server._register_tools()
            
            mock_get_citations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_analysis_tools(self):
        """Test registration of philosophical analysis tools."""
        server = PhilosophicalRAGServer()
        
        mock_analysis_tools = [
            Mock(name="map_argument_structure"),
            Mock(name="analyze_influence_network"),
            Mock(name="compare_philosophers")
        ]
        
        with patch('src.mcp_server.server.get_search_tools', return_value=[]), \
             patch('src.mcp_server.server.get_citation_tools', return_value=[]), \
             patch('src.mcp_server.server.get_analysis_tools') as mock_get_analysis:
            
            mock_get_analysis.return_value = mock_analysis_tools
            
            await server._register_tools()
            
            mock_get_analysis.assert_called_once()


class TestMCPResourceRegistration:
    """Test MCP resource registration for philosophical documents."""
    
    @pytest.mark.asyncio
    async def test_register_document_resources(self):
        """Test registration of document access resources."""
        server = PhilosophicalRAGServer()
        
        mock_doc_resources = [
            Mock(uri="philosophical_texts/ancient/aristotle"),
            Mock(uri="philosophical_texts/modern/kant"),
            Mock(uri="citation_networks/virtue_ethics")
        ]
        
        with patch('src.mcp_server.server.get_document_resources') as mock_get_docs:
            mock_get_docs.return_value = mock_doc_resources
            
            await server._register_resources()
            
            mock_get_docs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_note_resources(self):
        """Test registration of research note resources."""
        server = PhilosophicalRAGServer()
        
        mock_note_resources = [
            Mock(uri="research_notes/workspace_123"),
            Mock(uri="annotations/document_456"),
            Mock(uri="concept_genealogies/being")
        ]
        
        with patch('src.mcp_server.server.get_document_resources', return_value=[]), \
             patch('src.mcp_server.server.get_note_resources') as mock_get_notes:
            
            mock_get_notes.return_value = mock_note_resources
            
            await server._register_resources()
            
            mock_get_notes.assert_called_once()


class TestMCPPromptRegistration:
    """Test MCP prompt registration for research workflows."""
    
    @pytest.mark.asyncio
    async def test_register_research_prompts(self):
        """Test registration of philosophical research workflow prompts."""
        server = PhilosophicalRAGServer()
        
        mock_prompts = [
            Mock(name="literature_review"),
            Mock(name="argument_analysis"),
            Mock(name="concept_mapping"),
            Mock(name="comparative_analysis")
        ]
        
        with patch('src.mcp_server.server.get_research_prompts') as mock_get_prompts:
            mock_get_prompts.return_value = mock_prompts
            
            await server._register_prompts()
            
            mock_get_prompts.assert_called_once()


class TestMCPServerRuntime:
    """Test MCP server runtime and protocol handling."""
    
    @pytest.mark.asyncio
    async def test_server_run_initializes_if_needed(self):
        """Test server run() initializes if not already initialized."""
        server = PhilosophicalRAGServer()
        
        with patch.object(server, 'initialize', new_callable=AsyncMock) as mock_init, \
             patch('src.mcp_server.server.stdio_server') as mock_stdio:
            
            # Mock stdio_server context manager
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock server.run to avoid actual MCP protocol
            server.server.run = AsyncMock()
            
            await server.run()
            
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_run_skips_init_if_initialized(self):
        """Test server run() skips initialization if already done."""
        server = PhilosophicalRAGServer()
        server._initialized = True  # Mark as already initialized
        
        with patch.object(server, 'initialize', new_callable=AsyncMock) as mock_init, \
             patch('src.mcp_server.server.stdio_server') as mock_stdio:
            
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            server.server.run = AsyncMock()
            
            await server.run()
            
            mock_init.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_server_handles_runtime_errors(self):
        """Test server properly handles and logs runtime errors."""
        server = PhilosophicalRAGServer()
        
        with patch.object(server, 'initialize', new_callable=AsyncMock), \
             patch('src.mcp_server.server.stdio_server') as mock_stdio:
            
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=("read", "write"))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Make server.run raise an exception
            server.server.run = AsyncMock(side_effect=Exception("MCP protocol error"))
            
            with pytest.raises(Exception, match="MCP protocol error"):
                await server.run()


class TestMCPServerIntegration:
    """Integration tests for MCP server with philosophical research workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_server_initialization_flow(self):
        """Test complete server initialization with all components."""
        server = PhilosophicalRAGServer()
        
        # Mock all dependencies
        mock_tools = [Mock(name=f"tool_{i}") for i in range(3)]
        mock_resources = [Mock(uri=f"resource_{i}") for i in range(2)]
        mock_prompts = [Mock(name=f"prompt_{i}") for i in range(4)]
        
        with patch('src.mcp_server.server.init_database') as mock_init_db, \
             patch('src.mcp_server.server.get_search_tools', return_value=mock_tools[:1]), \
             patch('src.mcp_server.server.get_citation_tools', return_value=mock_tools[1:2]), \
             patch('src.mcp_server.server.get_analysis_tools', return_value=mock_tools[2:]), \
             patch('src.mcp_server.server.get_document_resources', return_value=mock_resources[:1]), \
             patch('src.mcp_server.server.get_note_resources', return_value=mock_resources[1:]), \
             patch('src.mcp_server.server.get_research_prompts', return_value=mock_prompts):
            
            await server.initialize()
            
            # Verify all initialization steps completed
            mock_init_db.assert_called_once()
            assert server._initialized is True
    
    @pytest.mark.asyncio
    async def test_philosophical_research_workflow_support(self):
        """Test server supports complete philosophical research workflows."""
        server = PhilosophicalRAGServer()
        
        # Mock philosophical research tools that should be available
        expected_tools = {
            "search_philosophical_texts": Mock(),
            "trace_concept_genealogy": Mock(),
            "extract_citations": Mock(),
            "map_argument_structure": Mock(),
            "sync_zotero": Mock()
        }
        
        # Mock the tool getter functions to return expected tools
        search_tools = [expected_tools["search_philosophical_texts"]]
        citation_tools = [expected_tools["extract_citations"], expected_tools["sync_zotero"]]
        analysis_tools = [expected_tools["trace_concept_genealogy"], expected_tools["map_argument_structure"]]
        
        with patch('src.mcp_server.server.init_database'), \
             patch('src.mcp_server.server.get_search_tools', return_value=search_tools), \
             patch('src.mcp_server.server.get_citation_tools', return_value=citation_tools), \
             patch('src.mcp_server.server.get_analysis_tools', return_value=analysis_tools), \
             patch('src.mcp_server.server.get_document_resources', return_value=[]), \
             patch('src.mcp_server.server.get_note_resources', return_value=[]), \
             patch('src.mcp_server.server.get_research_prompts', return_value=[]):
            
            await server.initialize()
            
            # Server should have registered philosophical research capabilities
            assert server._initialized is True
            
            # Note: In actual implementation, we would verify tools are registered
            # with server.server.add_tool() calls, but that requires the actual MCP library