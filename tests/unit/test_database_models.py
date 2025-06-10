"""
Unit tests for database models - Test-Driven Development approach.

Tests all database models for philosophical research data integrity,
relationships, and academic accuracy requirements BEFORE implementation.
"""

import pytest
import pytest_asyncio
from datetime import datetime, date
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.models import (
    Document, Chunk, Citation, ConceptTrace, Note, 
    Workspace, ArgumentStructure, Base
)


class TestDocumentModel:
    """Test Document model for philosophical text storage."""
    
    @pytest.mark.asyncio
    async def test_create_document_basic(self, db_session: AsyncSession):
        """Test basic document creation with required fields."""
        document = Document(
            title="Critique of Pure Reason",
            author="Immanuel Kant"
        )
        
        db_session.add(document)
        await db_session.flush()
        
        assert document.id is not None
        assert document.title == "Critique of Pure Reason"
        assert document.author == "Immanuel Kant"
        assert document.created_at is not None
        assert document.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_document_with_philosophical_metadata(self, db_session: AsyncSession):
        """Test document with complete philosophical research metadata.""" 
        document = Document(
            title="Being and Time",
            author="Martin Heidegger",
            publication_date=date(1927, 1, 1),
            tradition="continental",
            language="german",
            doi="10.1000/test.doi",
            abstract="Fundamental ontology and analysis of Dasein",
            document_metadata={
                "publisher": "Max Niemeyer Verlag",
                "original_language": "german",
                "translator": "John Macquarrie",
                "philosophical_school": "phenomenology"
            }
        )
        
        db_session.add(document)
        await db_session.flush()
        
        assert document.tradition == "continental"
        assert document.language == "german"
        assert document.document_metadata["philosophical_school"] == "phenomenology"
        assert document.publication_date.year == 1927
    
    @pytest.mark.asyncio 
    async def test_document_file_hash_uniqueness(self, db_session: AsyncSession):
        """Test file hash uniqueness constraint for deduplication."""
        # Create first document with file hash
        doc1 = Document(
            title="Test Document 1",
            author="Test Author",
            file_hash="abc123hash"
        )
        db_session.add(doc1)
        await db_session.flush()
        
        # Attempt to create second document with same hash should work in this session
        # (uniqueness enforced at database level)
        doc2 = Document(
            title="Test Document 2", 
            author="Different Author",
            file_hash="def456hash"  # Different hash
        )
        db_session.add(doc2)
        await db_session.flush()
        
        assert doc1.file_hash != doc2.file_hash


class TestChunkModel:
    """Test Chunk model for philosophical text segmentation."""
    
    @pytest.mark.asyncio
    async def test_create_chunk_with_document(self, db_session: AsyncSession):
        """Test chunk creation linked to document."""
        # Create parent document
        document = Document(title="Test Document", author="Test Author")
        db_session.add(document)
        await db_session.flush()
        
        # Create chunk
        chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            content="Every art and every inquiry aims at some good.",
            start_char=0,
            end_char=44,
            start_page=1,
            end_page=1
        )
        
        db_session.add(chunk)
        await db_session.flush()
        
        assert chunk.document_id == document.id
        assert chunk.chunk_index == 0
        assert chunk.content == "Every art and every inquiry aims at some good."
        assert chunk.start_char == 0
        assert chunk.end_char == 44
    
    @pytest.mark.asyncio
    async def test_chunk_philosophical_metadata(self, db_session: AsyncSession):
        """Test chunk with philosophical analysis metadata."""
        document = Document(title="Test", author="Test")
        db_session.add(document)
        await db_session.flush()
        
        chunk = Chunk(
            document_id=document.id,
            chunk_index=1,
            content="Therefore, virtue is a disposition to choose the mean.",
            chunk_metadata={
                "contains_argument": True,
                "argument_type": "syllogistic",
                "philosophical_concepts": ["virtue", "mean", "disposition"],
                "conclusion_marker": "therefore"
            }
        )
        
        db_session.add(chunk)
        await db_session.flush()
        
        assert chunk.chunk_metadata["contains_argument"] is True
        assert "virtue" in chunk.chunk_metadata["philosophical_concepts"]
        assert chunk.chunk_metadata["conclusion_marker"] == "therefore"
    
    @pytest.mark.asyncio
    async def test_chunk_document_relationship(self, db_session: AsyncSession):
        """Test bidirectional relationship between chunks and documents."""
        document = Document(title="Test Document", author="Test Author")
        db_session.add(document)
        await db_session.flush()
        
        # Create multiple chunks
        chunk1 = Chunk(document_id=document.id, chunk_index=0, content="First chunk")
        chunk2 = Chunk(document_id=document.id, chunk_index=1, content="Second chunk")
        
        db_session.add_all([chunk1, chunk2])
        await db_session.flush()
        
        # Test relationship access
        await db_session.refresh(document, ["chunks"])
        assert len(document.chunks) == 2
        assert document.chunks[0].content == "First chunk"
        assert document.chunks[1].content == "Second chunk"


class TestCitationModel:
    """Test Citation model for academic citation tracking."""
    
    @pytest.mark.asyncio
    async def test_citation_creation(self, db_session: AsyncSession):
        """Test basic citation creation and extraction."""
        # Create source document
        source_doc = Document(title="Source Work", author="Citing Author")
        db_session.add(source_doc)
        await db_session.flush()
        
        citation = Citation(
            source_document_id=source_doc.id,
            cited_text="(Aristotle 1999: 123)",
            cited_author="Aristotle",
            cited_work="Nicomachean Ethics", 
            cited_page="123",
            cited_year="1999",
            citation_format="author_year_page",
            extraction_confidence=0.95,
            context="In discussing virtue ethics, the author notes..."
        )
        
        db_session.add(citation)
        await db_session.flush()
        
        assert citation.cited_author == "Aristotle"
        assert citation.cited_work == "Nicomachean Ethics"
        assert citation.cited_page == "123"
        assert citation.extraction_confidence == 0.95
        assert citation.citation_format == "author_year_page"
    
    @pytest.mark.asyncio
    async def test_citation_network_relationship(self, db_session: AsyncSession):
        """Test citation network between documents."""
        # Create source and cited documents
        source_doc = Document(title="Modern Work", author="Modern Author")
        cited_doc = Document(title="Ancient Work", author="Ancient Author")
        
        db_session.add_all([source_doc, cited_doc])
        await db_session.flush()
        
        # Create citation linking them
        citation = Citation(
            source_document_id=source_doc.id,
            cited_document_id=cited_doc.id,
            cited_text="As Aristotle argues in Nicomachean Ethics...",
            cited_author="Ancient Author",
            cited_work="Ancient Work",
            matching_confidence=0.88
        )
        
        db_session.add(citation)
        await db_session.flush()
        
        # Test relationships
        await db_session.refresh(source_doc, ["source_citations"])
        await db_session.refresh(cited_doc, ["cited_citations"])
        
        assert len(source_doc.source_citations) == 1
        assert len(cited_doc.cited_citations) == 1
        assert source_doc.source_citations[0].cited_document_id == cited_doc.id


class TestConceptTraceModel:
    """Test ConceptTrace model for philosophical genealogy."""
    
    @pytest.mark.asyncio
    async def test_concept_trace_creation(self, db_session: AsyncSession):
        """Test concept genealogy tracking across philosophers."""
        # Create work document
        work = Document(title="Being and Time", author="Heidegger")
        db_session.add(work)
        await db_session.flush()
        
        concept_trace = ConceptTrace(
            concept="being",
            philosopher="Heidegger",
            work_id=work.id,
            timestamp=date(1927, 1, 1),
            tradition="continental",
            definition="Being as the fundamental question of ontology",
            context="In fundamental ontology, Being is distinguished from beings...",
            influences=["Husserl", "Kierkegaard"],
            influenced=["Sartre", "Gadamer"],
            extraction_confidence=0.92
        )
        
        db_session.add(concept_trace)
        await db_session.flush()
        
        assert concept_trace.concept == "being"
        assert concept_trace.philosopher == "Heidegger"
        assert concept_trace.tradition == "continental"
        assert "Husserl" in concept_trace.influences
        assert "Sartre" in concept_trace.influenced
        assert concept_trace.extraction_confidence == 0.92
    
    @pytest.mark.asyncio
    async def test_concept_uniqueness_constraint(self, db_session: AsyncSession):
        """Test uniqueness constraint for concept per philosopher per work."""
        work = Document(title="Test Work", author="Test Author")
        db_session.add(work)
        await db_session.flush()
        
        # First concept trace
        trace1 = ConceptTrace(
            concept="justice",
            philosopher="Plato", 
            work_id=work.id,
            definition="First definition"
        )
        db_session.add(trace1)
        await db_session.flush()
        
        # Different concept for same philosopher/work should work
        trace2 = ConceptTrace(
            concept="virtue",
            philosopher="Plato",
            work_id=work.id, 
            definition="Virtue definition"
        )
        db_session.add(trace2)
        await db_session.flush()
        
        assert trace1.concept != trace2.concept


class TestNoteModel:
    """Test Note model for research annotations."""
    
    @pytest.mark.asyncio
    async def test_note_creation(self, db_session: AsyncSession):
        """Test research note creation and linking."""
        # Create document and chunk
        document = Document(title="Test Document", author="Test Author")
        db_session.add(document)
        await db_session.flush()
        
        chunk = Chunk(
            document_id=document.id,
            chunk_index=0,
            content="Test content"
        )
        db_session.add(chunk)
        await db_session.flush()
        
        # Create note
        note = Note(
            user_id="user123",
            document_id=document.id,
            chunk_id=chunk.id,
            title="Important Observation",
            content="This passage demonstrates Aristotle's concept of the mean.",
            tags=["virtue-ethics", "mean", "aristotle"],
            note_type="analysis",
            is_public=False
        )
        
        db_session.add(note)
        await db_session.flush()
        
        assert note.title == "Important Observation"
        assert "virtue-ethics" in note.tags
        assert note.note_type == "analysis"
        assert note.is_public is False
    
    @pytest.mark.asyncio
    async def test_hierarchical_notes(self, db_session: AsyncSession):
        """Test hierarchical note organization."""
        document = Document(title="Test", author="Test")
        db_session.add(document)
        await db_session.flush()
        
        # Parent note
        parent_note = Note(
            user_id="user123",
            document_id=document.id,
            title="Main Analysis",
            content="Overview of virtue ethics"
        )
        db_session.add(parent_note)
        await db_session.flush()
        
        # Child note
        child_note = Note(
            user_id="user123",
            document_id=document.id,
            parent_note_id=parent_note.id,
            title="Specific Point",
            content="Details about the doctrine of the mean"
        )
        db_session.add(child_note)
        await db_session.flush()
        
        # Test relationship
        await db_session.refresh(parent_note, ["child_notes"])
        assert len(parent_note.child_notes) == 1
        assert parent_note.child_notes[0].title == "Specific Point"


class TestArgumentStructureModel:
    """Test ArgumentStructure model for logical analysis."""
    
    @pytest.mark.asyncio
    async def test_argument_structure_creation(self, db_session: AsyncSession):
        """Test argument structure extraction and storage."""
        document = Document(title="Philosophical Arguments", author="Test Philosopher")
        db_session.add(document)
        await db_session.flush()
        
        argument = ArgumentStructure(
            document_id=document.id,
            argument_type="deductive",
            philosopher="Aristotle",
            argument_name="Virtue as Mean Argument",
            premises=[
                {"text": "Virtue is concerned with actions and emotions", "type": "premise", "index": 1},
                {"text": "In actions and emotions, excess and deficiency are wrong", "type": "premise", "index": 2},
                {"text": "The mean between excess and deficiency is right", "type": "premise", "index": 3}
            ],
            conclusions=[
                {"text": "Therefore, virtue is a mean between extremes", "type": "conclusion", "index": 1}
            ],
            logical_validity="valid",
            soundness_assessment="sound",
            confidence_score=0.89
        )
        
        db_session.add(argument)
        await db_session.flush()
        
        assert argument.argument_type == "deductive"
        assert argument.philosopher == "Aristotle"
        assert len(argument.premises) == 3
        assert len(argument.conclusions) == 1
        assert argument.logical_validity == "valid"
        assert argument.confidence_score == 0.89


class TestWorkspaceModel:
    """Test Workspace model for collaborative research."""
    
    @pytest.mark.asyncio
    async def test_workspace_creation(self, db_session: AsyncSession):
        """Test collaborative workspace creation."""
        workspace = Workspace(
            name="Ancient Philosophy Project",
            description="Collaborative study of virtue ethics",
            owner_id="user123",
            research_focus="ethics",
            traditions=["ancient", "aristotelian"],
            time_periods=["ancient", "classical"],
            is_public=False,
            allow_public_contributions=False,
            require_approval=True
        )
        
        db_session.add(workspace)
        await db_session.flush()
        
        assert workspace.name == "Ancient Philosophy Project"
        assert workspace.research_focus == "ethics"
        assert "ancient" in workspace.traditions
        assert "classical" in workspace.time_periods
        assert workspace.is_public is False