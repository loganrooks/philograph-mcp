# Philosophical Research RAG MCP Server: System Architecture
**Version:** 1.0
**Last Updated:** January 6, 2025

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Claude Code    │   Web Interface │    API Clients              │
│  (Primary)      │   (Future)      │    (Third-party)            │
└────────┬────────┴────────┬────────┴────────┬────────────────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                    ┌──────▼──────┐
                    │  MCP Server │
                    │  Interface  │
                    └──────┬──────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                    Application Layer                             │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│   Hybrid    │  Document    │  Citation    │   Analysis         │
│   Search    │  Processor   │  Manager     │   Engine           │
│   Engine    │              │              │                    │
└─────┬───────┴──────┬───────┴──────┬───────┴────────┬───────────┘
      │              │              │                │
┌─────┴──────────────┴──────────────┴────────────────┴───────────┐
│                      Data Layer                                 │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│ PostgreSQL  │   pgvector   │  Object      │   Cache            │
│ (Relations) │  (Vectors)   │  Storage     │   (Redis)          │
└─────────────┴──────────────┴──────────────┴────────────────────┘
```

## 2. Component Responsibilities & Dependencies

### MCP Server Interface (src/mcp_server/)
- **Purpose:** Primary interface between Claude Code and philosophical research system
- **Components:** Tools (search, analysis, citation), Resources (documents, notes), Prompts (research workflows)
- **Dependencies:** FastAPI, MCP protocol library, application layer services

### Hybrid Search Engine (src/core/search/)
- **Purpose:** Combines semantic and keyword search for optimal philosophical text retrieval
- **Components:** Vector search, full-text search, result fusion, re-ranking
- **Dependencies:** PostgreSQL + pgvector, Vertex AI embeddings, Redis cache

### Document Processor (src/core/documents/)
- **Purpose:** Ingestion, philosophical-aware chunking, and embedding generation
- **Components:** PDF/EPUB extractors, philosophical chunker, metadata extractor, citation extractor
- **Dependencies:** PyPDF, spaCy NLP, Vertex AI embeddings, PostgreSQL

### Citation Manager (src/core/citations/)
- **Purpose:** Zotero integration and enhanced AI-powered citation discovery
- **Components:** Zotero sync, citation extraction, network analysis, format conversion
- **Dependencies:** Zotero API, PostgreSQL, natural language processing

### Analysis Engine (src/core/analysis/)
- **Purpose:** Philosophical analysis tools (genealogy, argument mapping, influence networks)
- **Components:** Concept tracer, argument mapper, influence analyzer, visualization generator
- **Dependencies:** NetworkX, natural language processing, PostgreSQL, embedding service

### Database Layer (src/infrastructure/database/)
- **Purpose:** Hybrid relational/vector storage with academic integrity
- **Components:** PostgreSQL with pgvector, migration management, connection pooling
- **Dependencies:** PostgreSQL 17+, pgvector extension, SQLAlchemy async

## 3. System Invariants (Non-Negotiable Rules)

1. **Academic Integrity:** All citations must maintain precise bibliographic information and attribution
2. **MCP Protocol Compliance:** All tools, resources, and prompts must follow MCP specification exactly
3. **Stateless Operation:** MCP server maintains no session state between requests
4. **Performance Guarantees:** Search latency < 200ms for 95% of queries, document processing > 100 pages/minute
5. **Philosophical Accuracy:** Semantic similarity must preserve philosophical meaning and context
6. **Data Consistency:** Vector embeddings and relational data must remain synchronized
7. **Error Resilience:** System must gracefully degrade when individual components fail

## 4. Data Flow Architecture

### Document Ingestion Flow
```
Document Upload → File Validation → Content Extraction
    ↓                                      ↓
    ↓                              Metadata Extraction
    ↓                                      ↓
    ↓                              Citation Extraction
    ↓                                      ↓
    ↓                              Philosophical Chunking
    ↓                                      ↓
    ↓                              Embedding Generation
    ↓                                      ↓
    └──────────────────────────→ Database Storage
                                          ↓
                                   Index Building
```

### Search Flow
```
User Query → Query Analysis → Query Expansion
    ↓              ↓                ↓
    ↓         Embedding      Keyword Extraction
    ↓         Generation            ↓
    ↓              ↓                ↓
    ↓         Vector Search    Full-text Search
    ↓              ↓                ↓
    ↓              └────────┬───────┘
    ↓                       ↓
    ↓                  Result Fusion
    ↓                       ↓
    ↓                   Re-ranking
    ↓                       ↓
    └──────────────→ Result Presentation
```

### Concept Genealogy Flow
```
Concept Query → Occurrence Discovery → Temporal Ordering
    ↓                     ↓                    ↓
    ↓              Citation Analysis    Influence Mapping
    ↓                     ↓                    ↓
    ↓              Semantic Similarity   Network Building
    ↓                     ↓                    ↓
    └─────────────→ Genealogy Generation ←─────┘
```

## 5. Database Schema Design

### Core Tables
```sql
-- Documents with philosophical metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    author TEXT,
    publication_date DATE,
    tradition TEXT, -- ancient, medieval, modern, contemporary
    language TEXT,
    original_file_path TEXT,
    file_hash TEXT UNIQUE, -- For deduplication
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Text chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(3072), -- Gemini embedding dimension
    start_page INTEGER,
    end_page INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    metadata JSONB
);

-- Citation network
CREATE TABLE citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID REFERENCES documents(id),
    cited_text TEXT,
    cited_author TEXT,
    cited_work TEXT,
    cited_page TEXT,
    confidence_score FLOAT,
    matched_document_id UUID REFERENCES documents(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Concept genealogy tracking
CREATE TABLE concept_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept TEXT NOT NULL,
    philosopher TEXT NOT NULL,
    work_id UUID REFERENCES documents(id),
    timestamp DATE,
    definition TEXT,
    context TEXT,
    embedding vector(3072)
);

-- User annotations and notes
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    document_id UUID REFERENCES documents(id),
    chunk_id UUID REFERENCES chunks(id),
    content TEXT NOT NULL,
    tags TEXT[],
    parent_note_id UUID REFERENCES notes(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Performance Indexes
```sql
-- Vector similarity search
CREATE INDEX chunks_embedding_idx ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search
CREATE INDEX chunks_content_gin_idx ON chunks 
USING gin(to_tsvector('english', content));

-- Citation network queries
CREATE INDEX citations_source_idx ON citations(source_document_id);
CREATE INDEX citations_cited_idx ON citations(matched_document_id);

-- Temporal queries for genealogy
CREATE INDEX documents_date_tradition_idx ON documents(publication_date, tradition);
CREATE INDEX concept_traces_concept_time_idx ON concept_traces(concept, timestamp);
```

## 6. MCP Server Tool Architecture

### Tool Categories
```python
# Search Tools
- search_philosophical_texts: Semantic + keyword hybrid search
- trace_concept_genealogy: Track concept evolution across time
- find_citations: Discover citation networks and influences
- analyze_argument_structure: Map logical argument patterns

# Document Tools  
- upload_document: Process and index new philosophical texts
- extract_citations: Identify and link cited works
- generate_summary: Create philosophical text summaries

# Analysis Tools
- map_influence_network: Visualize intellectual influence patterns
- compare_philosophers: Cross-tradition comparative analysis
- identify_arguments: Extract and categorize argument types
```

### Resource Architecture
```python
# Dynamic Resources
- philosophical_texts/{tradition}/{author}: Access specific texts
- citation_networks/{concept}: Access citation graphs
- research_notes/{workspace}: Access collaborative notes
- concept_genealogies/{concept}: Access concept evolution data
```

### Prompt Templates
```python
# Research Workflows
- literature_review: Systematic review process
- argument_analysis: Logical structure examination  
- concept_mapping: Philosophical concept relationships
- comparative_analysis: Cross-tradition comparison
```

## 7. Integration Points

### External Services
- **Vertex AI:** Embedding generation with Gemini models
- **Zotero API:** Citation management and bibliographic sync
- **Z-Library MCP:** External text acquisition (future)
- **Claude Code:** Primary user interface via MCP protocol

### API Design Patterns
```python
# Async everywhere
async def search_texts(query: SearchQuery) -> SearchResults:
    pass

# Proper error handling
class PhilosophicalResearchError(Exception):
    pass

class CitationNotFoundError(PhilosophicalResearchError):
    pass

# Type safety
from pydantic import BaseModel

class ConceptGenealogyRequest(BaseModel):
    concept: str
    start_philosopher: Optional[str] = None
    end_philosopher: Optional[str] = None
    tradition_filter: Optional[str] = None
```

## 8. Performance Architecture

### Caching Strategy
1. **Query Cache:** Recent searches cached in Redis (24h TTL)
2. **Embedding Cache:** Frequently accessed embeddings (7d TTL)  
3. **Document Cache:** Hot documents in memory
4. **Result Cache:** Common query results (1h TTL)

### Scaling Patterns
- **Database Read Replicas:** For search query distribution
- **Embedding Service Pool:** Multiple Vertex AI connections
- **Connection Pooling:** Optimized PostgreSQL connections
- **Async Processing:** Non-blocking I/O throughout

### Monitoring Points
- Search latency percentiles
- Document processing throughput
- Embedding generation time
- Database query performance
- MCP protocol response times

## 9. Security Architecture

### Authentication & Authorization
- OAuth2 integration for academic institutions
- Role-based access control (RBAC)
- Research workspace isolation
- API key management for external integrations

### Data Protection
- AES-256 encryption at rest
- TLS 1.3 for data in transit
- Audit logging for all research activities
- GDPR compliance for personal research data

## 10. Deployment Architecture

### Container Structure
```yaml
services:
  mcp-server:
    image: phil-rag/mcp-server
    depends_on: [postgres, redis]
    
  postgres:
    image: pgvector/pgvector:pg17
    volumes: [pgdata:/var/lib/postgresql/data]
    
  redis:
    image: redis:7-alpine
    
  embedding-service:
    image: phil-rag/embedding-service
    environment: [VERTEX_AI_PROJECT]
```

### Kubernetes Scaling
- Horizontal Pod Autoscaler for MCP server pods
- Persistent volumes for PostgreSQL data
- ConfigMaps for environment-specific configuration
- Secrets for API keys and database credentials