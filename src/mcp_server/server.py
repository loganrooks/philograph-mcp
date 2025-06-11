"""
Main MCP Server for Philosophical Research RAG System.

Implements the Model Context Protocol (MCP) to provide Claude Code with
sophisticated philosophical research capabilities including semantic search,
citation management, and genealogical analysis.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

from mcp.server import FastMCP
from mcp.types import Tool, Resource, Prompt

from ..infrastructure.database.connection import init_database
from .tools.search_tools import get_search_tools
from .tools.citation_tools import get_citation_tools  
from .tools.analysis_tools import get_analysis_tools
from .resources.document_resources import get_document_resources
from .resources.note_resources import get_note_resources
from .prompts.research_prompts import get_research_prompts

# Configure logging for philosophical research context
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhilosophicalRAGServer:
    """
    Main MCP Server for philosophical research capabilities.
    
    Provides Claude Code with tools for semantic search, citation management,
    genealogical analysis, and other scholarly research functions while
    maintaining academic rigor and citation accuracy.
    """
    
    def __init__(self):
        self.server = FastMCP("philosophical-rag-mcp")
        self._initialized = False
        
    async def initialize(self) -> None:
        """
        Initialize the MCP server with all philosophical research capabilities.
        
        Sets up database connections, registers tools/resources/prompts,
        and prepares the server for philosophical research workflows.
        """
        if self._initialized:
            return
            
        logger.info("Initializing Philosophical Research RAG MCP Server...")
        
        # Initialize database connection
        try:
            database_url = os.getenv('DATABASE_URL')
            await init_database(database_url)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        
        # Register all MCP capabilities
        await self._register_tools()
        await self._register_resources() 
        await self._register_prompts()
        
        self._initialized = True
        logger.info("Philosophical Research RAG MCP Server initialized successfully")
    
    async def _register_tools(self) -> None:
        """Register all philosophical research tools."""
        logger.info("Registering philosophical research tools...")
        
        # Search and discovery tools
        search_tools = await get_search_tools()
        for tool in search_tools:
            self.server.add_tool(tool)
            
        # Citation management tools
        citation_tools = await get_citation_tools()
        for tool in citation_tools:
            self.server.add_tool(tool)
            
        # Analysis and genealogy tools
        analysis_tools = await get_analysis_tools()
        for tool in analysis_tools:
            self.server.add_tool(tool)
            
        logger.info(f"Registered {len(search_tools + citation_tools + analysis_tools)} tools")
    
    async def _register_resources(self) -> None:
        """Register all philosophical research resources."""
        logger.info("Registering philosophical research resources...")
        
        # Document and text resources
        document_resources = await get_document_resources()
        for resource in document_resources:
            self.server.add_resource(resource)
            
        # Note and annotation resources
        note_resources = await get_note_resources() 
        for resource in note_resources:
            self.server.add_resource(resource)
            
        logger.info(f"Registered {len(document_resources + note_resources)} resources")
    
    async def _register_prompts(self) -> None:
        """Register all philosophical research workflow prompts."""
        logger.info("Registering philosophical research prompts...")
        
        # Research workflow prompts
        research_prompts = await get_research_prompts()
        for prompt in research_prompts:
            self.server.add_prompt(prompt)
            
        logger.info(f"Registered {len(research_prompts)} prompts")
    
    async def run(self) -> None:
        """
        Run the MCP server with stdio transport.
        
        This method starts the server and handles the MCP protocol communication
        with Claude Code, enabling philosophical research workflows.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info("Starting Philosophical Research RAG MCP Server...")
        
        try:
            # Run server with stdio transport for Claude Code integration
            await self.server.run_stdio_async()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("Philosophical Research RAG MCP Server shutting down...")


async def main() -> None:
    """
    Main entry point for the Philosophical Research RAG MCP Server.
    
    Initializes and runs the server with proper error handling and logging
    for philosophical research workflows.
    """
    server = PhilosophicalRAGServer()
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    # Run the server
    asyncio.run(main())