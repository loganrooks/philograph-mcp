"""
Note and annotation resource management for MCP server.

Provides access to research notes, annotations, and collaborative
research data through the MCP resource protocol.
"""

from typing import List, Dict, Any


async def get_note_resources() -> List[Dict[str, Any]]:
    """
    Get all note resources for philosophical research.
    
    Returns:
        List of note resource definitions for MCP protocol
    """
    # Mock implementation for unit testing
    # In production, these would be proper MCP Resource objects
    return [
        {
            "uri": "research_notes/personal",
            "name": "Personal Research Notes", 
            "description": "User's personal philosophical research notes",
            "mimeType": "text/markdown"
        },
        {
            "uri": "annotations/collaborative",
            "name": "Collaborative Annotations",
            "description": "Shared annotations on philosophical texts",
            "mimeType": "application/json"
        }
    ]