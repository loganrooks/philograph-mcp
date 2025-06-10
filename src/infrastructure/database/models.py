"""
Database models for Philosophical Research RAG MCP Server.

This module defines SQLAlchemy models for the hybrid PostgreSQL + pgvector
storage system, maintaining academic integrity and citation accuracy.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    """
    Core document storage with philosophical metadata.
    
    Represents philosophical texts, papers, books, and other scholarly materials
    with proper academic metadata and citation tracking.
    """
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False, index=True)
    author = Column(String, index=True)
    publication_date = Column(Date, index=True)
    tradition = Column(String, index=True)  # ancient, medieval, modern, contemporary
    language = Column(String, index=True)
    original_file_path = Column(String)
    file_hash = Column(String, unique=True)  # For deduplication
    doi = Column(String, unique=True, nullable=True)
    isbn = Column(String, nullable=True)
    url = Column(String, nullable=True)
    abstract = Column(Text, nullable=True)
    
    # Document metadata as JSON for flexibility (renamed to avoid SQLAlchemy conflict)
    document_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    source_citations = relationship("Citation", foreign_keys="Citation.source_document_id", back_populates="source_document")
    cited_citations = relationship("Citation", foreign_keys="Citation.cited_document_id", back_populates="cited_document")
    concept_traces = relationship("ConceptTrace", back_populates="work")
    notes = relationship("Note", back_populates="document")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title[:50]}...', author='{self.author}')>"


class Chunk(Base):
    """
    Text chunks with philosophical-aware boundaries and vector embeddings.
    
    Represents semantically meaningful segments of philosophical texts
    with embeddings for similarity search and proper source attribution.
    """
    
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Gemini embedding vector (3072 dimensions)
    embedding = Column(Vector(3072), nullable=True)
    
    # Position information for citation accuracy
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    
    # Chunk-specific metadata (argument markers, philosophical concepts, etc.)
    chunk_metadata = Column(JSON, default=dict)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    notes = relationship("Note", back_populates="chunk")
    
    # Ensure ordering within document
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),)
    
    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class Citation(Base):
    """
    Citation network relationships with confidence scoring.
    
    Tracks when one philosophical work cites another, maintaining academic
    standards for attribution and enabling influence network analysis.
    """
    
    __tablename__ = "citations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    
    # Raw citation text as extracted
    cited_text = Column(Text, nullable=False)
    cited_author = Column(String, nullable=True, index=True)
    cited_work = Column(String, nullable=True, index=True)
    cited_page = Column(String, nullable=True)  # Can be complex: "pp. 123-125", "1.2.3", etc.
    cited_year = Column(String, nullable=True)
    
    # Matched document (if we have it in our database)
    cited_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    
    # AI confidence in citation extraction and matching
    extraction_confidence = Column(Float, default=0.0)
    matching_confidence = Column(Float, default=0.0)
    
    # Citation context for validation
    context = Column(Text, nullable=True)
    
    # Citation format metadata
    citation_format = Column(String, nullable=True)  # "author_year", "classical", "footnote", etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_document_id], back_populates="source_citations")
    cited_document = relationship("Document", foreign_keys=[cited_document_id], back_populates="cited_citations")
    
    def __repr__(self) -> str:
        return f"<Citation(id={self.id}, author='{self.cited_author}', work='{self.cited_work}')>"


class ConceptTrace(Base):
    """
    Philosophical concept genealogy tracking.
    
    Tracks how philosophical concepts evolve across thinkers and time periods,
    enabling genealogical analysis and intellectual history research.
    """
    
    __tablename__ = "concept_traces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    concept = Column(String, nullable=False, index=True)
    philosopher = Column(String, nullable=False, index=True)
    work_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Temporal information for genealogy
    timestamp = Column(Date, nullable=True, index=True)
    tradition = Column(String, nullable=True, index=True)
    
    # Concept definition and usage context
    definition = Column(Text, nullable=True)
    context = Column(Text, nullable=True)
    usage_examples = Column(ARRAY(Text), default=list)
    
    # Semantic embedding for similarity analysis
    embedding = Column(Vector(3072), nullable=True)
    
    # Relationships to other concepts
    influences = Column(ARRAY(String), default=list)  # Philosopher names that influenced this usage
    influenced = Column(ARRAY(String), default=list)  # Philosophers influenced by this usage
    
    # Confidence in automated extraction
    extraction_confidence = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    work = relationship("Document", back_populates="concept_traces")
    
    # Ensure uniqueness of concept per philosopher per work
    __table_args__ = (UniqueConstraint("concept", "philosopher", "work_id", name="uq_concept_philosopher_work"),)
    
    def __repr__(self) -> str:
        return f"<ConceptTrace(concept='{self.concept}', philosopher='{self.philosopher}')>"


class Note(Base):
    """
    User annotations and research notes with hierarchical organization.
    
    Supports collaborative philosophical research with proper attribution
    and version tracking for scholarly annotations.
    """
    
    __tablename__ = "notes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # External user system
    
    # Link to document or specific chunk
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True, index=True)
    
    # Note content and organization
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    tags = Column(ARRAY(String), default=list, index=True)
    
    # Hierarchical organization
    parent_note_id = Column(UUID(as_uuid=True), ForeignKey("notes.id"), nullable=True)
    
    # Collaboration metadata
    is_public = Column(Boolean, default=False)
    workspace_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Note type for philosophical research
    note_type = Column(String, default="annotation")  # annotation, argument, objection, synthesis, etc.
    
    # Timestamps for version control
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="notes")
    chunk = relationship("Chunk", back_populates="notes")
    child_notes = relationship("Note", backref="parent_note", remote_side=[id])
    
    def __repr__(self) -> str:
        return f"<Note(id={self.id}, user_id={self.user_id}, title='{self.title}')>"


class Workspace(Base):
    """
    Collaborative research workspaces for philosophical projects.
    
    Enables team-based philosophical research with shared documents,
    notes, and analysis while maintaining proper attribution.
    """
    
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Workspace organization
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    is_public = Column(Boolean, default=False)
    
    # Research focus metadata
    research_focus = Column(String, nullable=True)  # epistemology, ethics, metaphysics, etc.
    traditions = Column(ARRAY(String), default=list)  # philosophical traditions involved
    time_periods = Column(ARRAY(String), default=list)  # historical periods covered
    
    # Collaboration settings
    allow_public_contributions = Column(Boolean, default=False)
    require_approval = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<Workspace(id={self.id}, name='{self.name}', owner_id={self.owner_id})>"


class ArgumentStructure(Base):
    """
    Extracted argument structures from philosophical texts.
    
    Represents logical argument patterns (premises, conclusions, objections)
    identified in philosophical works for analysis and teaching.
    """
    
    __tablename__ = "argument_structures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True)
    
    # Argument identification
    argument_type = Column(String, nullable=False)  # deductive, inductive, abductive, analogical
    philosopher = Column(String, nullable=True, index=True)
    argument_name = Column(String, nullable=True)  # "Ontological Argument", "Trolley Problem", etc.
    
    # Argument components as structured data
    premises = Column(JSON, default=list)  # List of premise objects with text and metadata
    conclusions = Column(JSON, default=list)  # List of conclusion objects
    objections = Column(JSON, default=list)  # List of objection objects
    supports = Column(JSON, default=list)  # Supporting evidence
    
    # Analysis metadata
    logical_validity = Column(String, nullable=True)  # valid, invalid, uncertain
    soundness_assessment = Column(String, nullable=True)  # sound, unsound, uncertain
    confidence_score = Column(Float, default=0.0)
    
    # Context and location
    context = Column(Text, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"<ArgumentStructure(id={self.id}, type='{self.argument_type}', philosopher='{self.philosopher}')>"