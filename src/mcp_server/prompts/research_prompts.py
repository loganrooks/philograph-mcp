"""
Research workflow prompts for MCP server.

Provides structured prompts for common philosophical research
tasks like literature reviews, argument analysis, and concept mapping.
"""

from typing import List, Dict, Any


async def get_research_prompts() -> List[Dict[str, Any]]:
    """
    Get all research prompts for philosophical workflows.
    
    Returns:
        List of prompt definitions for MCP protocol
    """
    # Mock implementation for unit testing  
    # In production, these would be proper MCP Prompt objects
    return [
        {
            "name": "literature_review",
            "description": "Generate a comprehensive literature review for a philosophical topic",
            "arguments": [
                {
                    "name": "topic",
                    "description": "The philosophical topic to review",
                    "required": True
                },
                {
                    "name": "tradition", 
                    "description": "Philosophical tradition to focus on",
                    "required": False
                }
            ]
        },
        {
            "name": "argument_analysis",
            "description": "Analyze the logical structure and validity of philosophical arguments", 
            "arguments": [
                {
                    "name": "argument_text",
                    "description": "The argument text to analyze",
                    "required": True
                },
                {
                    "name": "philosopher",
                    "description": "The philosopher who made this argument",
                    "required": False
                }
            ]
        },
        {
            "name": "concept_mapping",
            "description": "Map relationships between philosophical concepts",
            "arguments": [
                {
                    "name": "central_concept",
                    "description": "The main concept to map",
                    "required": True
                },
                {
                    "name": "related_concepts",
                    "description": "List of related concepts to include",
                    "required": False
                }
            ]
        }
    ]