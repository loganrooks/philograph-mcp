"""
Pytest configuration and fixtures for Philosophical Research RAG MCP Server.

Provides test database setup, sample philosophical data, and MCP protocol
testing utilities following TDD methodology.
"""

import asyncio
import os
from typing import AsyncGenerator, Dict, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.connection import DatabaseManager
from src.infrastructure.database.models import (
    Base, Document, Chunk, Citation, ConceptTrace, Note, ArgumentStructure
)


# Configure asyncio for testing
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_manager() -> AsyncGenerator[DatabaseManager, None]:
    """
    Create test database manager with isolated test database.
    
    Uses separate test database to avoid conflicts with development data.
    """
    # Use test database URL
    test_db_url = os.getenv(
        'TEST_DATABASE_URL',
        'postgresql+asyncpg://philosopher:password@localhost:5433/phil_rag_test'
    )
    
    db_manager = DatabaseManager()
    await db_manager.initialize(test_db_url)
    
    # Create tables for testing
    async with db_manager.engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    yield db_manager
    
    # Cleanup: Drop all tables after tests
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await db_manager.close()


@pytest.fixture
async def db_session(test_db_manager: DatabaseManager) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide clean database session for each test.
    
    Automatically rolls back transactions to keep tests isolated.
    """
    async with test_db_manager.get_session() as session:
        # Start transaction
        transaction = await session.begin()
        
        yield session
        
        # Always rollback to keep tests isolated
        await transaction.rollback()


@pytest.fixture
def sample_philosophical_documents() -> list[Dict[str, Any]]:
    """
    Sample philosophical documents from multiple traditions for testing.
    
    Includes various citation formats and philosophical concepts for
    comprehensive testing of domain-specific functionality.
    """
    return [
        {
            "title": "Nicomachean Ethics",
            "author": "Aristotle", 
            "publication_date": "330-01-01",  # Approximate BCE date
            "tradition": "ancient",
            "language": "ancient_greek",
            "abstract": "Aristotle's work on ethics and virtue theory, foundational to Western moral philosophy.",
            "content": """
            Every art and every inquiry, and similarly every action and pursuit, 
            is thought to aim at some good; and for this reason the good has rightly 
            been declared to be that at which all things aim. But a certain difference 
            is found among ends; some are activities, others are products apart from 
            the activities that produce them. Where there are ends apart from the actions, 
            it is the nature of the products to be better than the activities.
            
            Now, as there are many actions, arts, and sciences, their ends also are many; 
            the end of the medical art is health, that of shipbuilding a vessel, that of 
            strategy victory, that of economics wealth. But where such arts fall under 
            a single capacity—as bridle-making and the other arts concerned with the 
            equipment of horses fall under the art of riding, and this and every military 
            action under strategy, in the same way other arts fall under yet others—in 
            all of these the ends of the master arts are to be preferred to all the 
            subordinate ends.
            """
        },
        {
            "title": "Being and Time",
            "author": "Martin Heidegger",
            "publication_date": "1927-01-01", 
            "tradition": "continental",
            "language": "german",
            "abstract": "Fundamental ontology examining the meaning of Being through analysis of Dasein.",
            "content": """
            The question of the meaning of Being must be formulated. If it is a fundamental 
            question, or indeed the fundamental question, it must be made transparent in what 
            it asks. So we must first discuss (1) what is asked about in this question, 
            (2) what is interrogated, (3) what is sought.
            
            What is asked about is Being—that which determines entities as entities, that on 
            the basis of which entities are already understood, however we may discuss them 
            in detail. The Being of entities 'is' not itself an entity. If we are to 
            understand the problem of Being, our first philosophical step consists in not 
            μῦθόν τινα διηγεῖσθαι, in not 'telling a story'—that is to say, in not defining 
            entities as entities by tracing them back in their origin to some other entity, 
            as if Being had the character of some possible entity.
            """
        },
        {
            "title": "A Theory of Justice", 
            "author": "John Rawls",
            "publication_date": "1971-01-01",
            "tradition": "analytic",
            "language": "english", 
            "abstract": "Liberal political philosophy proposing justice as fairness and the original position.",
            "content": """
            Justice is the first virtue of social institutions, as truth is of systems of thought. 
            A theory however elegant and economical must be rejected or revised if it is untrue; 
            likewise laws and institutions no matter how efficient and well-arranged must be 
            reformed or abolished if they are unjust. Each person possesses an inviolability 
            founded on justice that even the welfare of society as a whole cannot override.
            
            For this reason justice denies that the loss of freedom for some is made right by 
            a greater good shared by others. It does not allow that the sacrifices imposed on 
            a few are outweighed by the larger sum of advantages enjoyed by many. Therefore in 
            a just society the liberties of equal citizenship are taken as settled; the rights 
            secured by justice are not subject to political bargaining or to the calculus of 
            social interests.
            """
        }
    ]


@pytest.fixture  
def sample_citations() -> list[Dict[str, Any]]:
    """
    Sample citations in various philosophical formats for testing.
    
    Tests citation extraction and format recognition across different
    philosophical citation styles and traditions.
    """
    return [
        {
            "format": "author_year",
            "text": "(Aristotle 1999)",
            "author": "Aristotle",
            "year": "1999",
            "page": None
        },
        {
            "format": "author_year_page", 
            "text": "(Heidegger 1962: 32)",
            "author": "Heidegger", 
            "year": "1962",
            "page": "32"
        },
        {
            "format": "classical",
            "text": "Aristotle Nicomachean Ethics 1.1.1",
            "author": "Aristotle",
            "work": "Nicomachean Ethics",
            "location": "1.1.1"
        },
        {
            "format": "classical_extended",
            "text": "Plato Republic 514a-515a", 
            "author": "Plato",
            "work": "Republic",
            "location": "514a-515a"
        }
    ]


@pytest.fixture
def sample_concepts() -> list[Dict[str, Any]]:
    """
    Sample philosophical concepts for genealogy testing.
    
    Provides concept evolution examples across philosophers and time periods
    for testing genealogical analysis functionality.
    """
    return [
        {
            "concept": "being",
            "philosopher": "Parmenides", 
            "tradition": "ancient",
            "definition": "That which is, unchanging and eternal, opposed to becoming",
            "timestamp": "500-01-01"  # Approximate BCE
        },
        {
            "concept": "being",
            "philosopher": "Aristotle",
            "tradition": "ancient", 
            "definition": "Being as substance (ousia), the fundamental reality underlying change",
            "timestamp": "330-01-01"
        },
        {
            "concept": "being",
            "philosopher": "Heidegger",
            "tradition": "continental",
            "definition": "Being as the fundamental question, distinguished from beings (ontic vs ontological)",
            "timestamp": "1927-01-01"
        }
    ]


@pytest.fixture
async def sample_database_data(
    db_session: AsyncSession, 
    sample_philosophical_documents: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Populate test database with sample philosophical research data.
    
    Creates complete test dataset with documents, chunks, citations, and concepts
    for integration testing of philosophical research workflows.
    """
    created_data = {
        "documents": [],
        "chunks": [],  
        "citations": [],
        "concepts": []
    }
    
    # Create sample documents
    for doc_data in sample_philosophical_documents:
        document = Document(
            title=doc_data["title"],
            author=doc_data["author"],
            publication_date=doc_data["publication_date"],
            tradition=doc_data["tradition"],
            language=doc_data["language"],
            abstract=doc_data["abstract"]
        )
        db_session.add(document)
        created_data["documents"].append(document)
    
    await db_session.flush()  # Get IDs without committing
    
    # Create sample chunks for each document
    for i, document in enumerate(created_data["documents"]):
        content = sample_philosophical_documents[i]["content"]
        
        # Split content into chunks (simplified for testing)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for j, paragraph in enumerate(paragraphs):
            chunk = Chunk(
                document_id=document.id,
                chunk_index=j,
                content=paragraph,
                start_char=j * 200,  # Simplified positioning
                end_char=(j + 1) * 200,
                chunk_metadata={"paragraph_index": j}
            )
            db_session.add(chunk)
            created_data["chunks"].append(chunk)
    
    return created_data


class MCPTestClient:
    """
    Test client for MCP protocol testing.
    
    Provides utilities for testing MCP tools, resources, and prompts
    with philosophical research workflows.
    """
    
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.prompts = {}
    
    def register_tool(self, tool):
        """Register MCP tool for testing."""
        self.tools[tool.name] = tool
    
    def register_resource(self, resource):
        """Register MCP resource for testing.""" 
        self.resources[resource.uri] = resource
    
    def register_prompt(self, prompt):
        """Register MCP prompt for testing."""
        self.prompts[prompt.name] = prompt
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call MCP tool with arguments for testing."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
        
        tool = self.tools[tool_name]
        return await tool.handler(**arguments)


@pytest.fixture
def mcp_test_client() -> MCPTestClient:
    """Provide MCP test client for protocol testing."""
    return MCPTestClient()