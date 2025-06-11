"""
Philosophical text search tools for MCP server.

Provides semantic search, keyword search, and concept genealogy tools
for philosophical research workflows.
"""

from typing import List, Dict, Any


async def get_search_tools() -> List[Dict[str, Any]]:
    """
    Get all search tools for philosophical research.
    
    Returns:
        List of search tool definitions for MCP protocol
    """
    # Mock implementation for unit testing
    # In production, these would be proper MCP Tool objects
    return [
        {
            "name": "search_philosophical_texts",
            "description": "Search philosophical texts using semantic similarity",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tradition": {"type": "string", "enum": ["ancient", "medieval", "modern", "contemporary"]},
                    "max_results": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "trace_concept_genealogy", 
            "description": "Trace the evolution of philosophical concepts across time and thinkers",
            "schema": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "start_philosopher": {"type": "string"},
                    "end_philosopher": {"type": "string"},
                    "tradition": {"type": "string"}
                },
                "required": ["concept"]
            }
        }
    ]