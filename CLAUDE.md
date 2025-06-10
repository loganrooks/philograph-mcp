# 🎯 CURRENT TASK: Initialize Philosophical Research RAG MCP Server Development System
- **PLAN:** Complete Claude Code Development System Bootstrap following docs/CLAUDE_CODE_SYSTEM_BOOTSTRAP.md
- **STATUS:** Phase 1 - File Structure Setup and Template Implementation

---

## 🧠 CORE DIRECTIVES (VERIFY ON EVERY ACTION)

1. **SPARC-V-L³ Protocol:** You MUST follow the full SPARC-V-L³ cycle for all non-trivial changes as detailed in `docs/DEVELOPMENT_GUIDE.md`.
2. **pytest + MCP Protocol Testing is Non-Negotiable:** All MCP tools, resources, and prompts must have comprehensive test coverage including integration tests with Claude Code and philosophical research workflows.
3. **MCP Server + Hybrid Search Architecture Compliance:** Follow PostgreSQL + pgvector hybrid architecture with proper separation between MCP interface, search engine, document processing, and analysis components.
4. **Verification is Mandatory:** Before any database schema changes, embedding model updates, or MCP protocol modifications, run full test suite (`poetry run pytest`) and verify backward compatibility.
5. **Log All Anomalies:** Any deviation from the plan, unexpected error, or user correction MUST be logged with structured detail in `FEEDBACK_LOG.md`.
6. **Log Your Actions:** At the end of every response, you MUST append a structured entry to `ACTIVITY_LOG.md`.
7. **Self-Critique:** After completing a significant task, you MUST perform a self-analysis and log it in `SELF_ANALYSIS_LOG.md`.

---

## 🔄 CONTEXT INITIALIZATION PROTOCOL (CRITICAL)

**MANDATORY:** Execute this protocol at the start of EVERY conversation and whenever context may have been compacted/refreshed.

### Context Refresh Detection Triggers:
- Beginning of any new conversation
- When you cannot recall recent task details or decisions
- When foundational documents are not in working memory
- When switching between MCP server development and philosophical research features
- When database schema or embedding model changes are discussed

### IMMEDIATE INITIALIZATION SEQUENCE:
1. **ALWAYS READ FIRST:** 
   - `docs/DEVELOPMENT_GUIDE.md` - Core principles, patterns, and methodologies
   - `docs/ARCHITECTURE.md` - System design and component relationships  
   - `PROJECT_STATUS.md` - Current project state and progress
   - `FEEDBACK_LOG.md` - Recent lessons and workflow decisions

2. **VERIFY UNDERSTANDING:**
   - MCP Server + Hybrid Search Architecture requirements
   - pytest + MCP Protocol Testing methodology
   - SPARC-V-L³ protocol compliance
   - Philosophical research domain requirements and citation standards

3. **LOAD PROJECT CONTEXT:**
   - Current task status and priorities
   - Recent architectural decisions and patterns
   - Active issues and their resolution approaches

**NEVER SKIP THIS PROTOCOL** - Inconsistent decisions result from missing foundational context.

---

## 📚 KNOWLEDGE BASE INTERACTION PROTOCOL

You are required to read the following documents at specific trigger points:

- **WHEN:** Starting *any* new task.
  - **READ:** `docs/DEVELOPMENT_GUIDE.md` to refresh core principles.
  - **READ:** `docs/ARCHITECTURE.md` to understand the system context.
  - **READ:** `PROJECT_STATUS.md` to understand the current state.

- **WHEN:** Implementing MCP tools, resources, or prompts.
  - **READ:** `docs/phil-rag-implementation.md` for technical specifications.

- **WHEN:** Working on philosophical research features (search, genealogy, citations).
  - **READ:** `docs/phil-rag-user-stories.md` and `docs/phil-rag-executive-summary.md`.

- **WHEN:** A significant architectural decision is needed.
  - **ACTION:** Propose a new Architecture Decision Record (ADR) in `docs/decisions/`.

- **WHEN:** A task is complete.
  - **ACTION:** Update `CHANGELOG.md`, `PROJECT_STATUS.md`, and `SELF_ANALYSIS_LOG.md`.
  - **ACTION:** If systemic lesson learned, update `DEVELOPMENT_GUIDE.md`.

---

## 🛠️ PHILOSOPHICAL RESEARCH RAG MCP SERVER SPECIFIC GUIDELINES

### Project Overview
This is the Philosophical Research RAG MCP Server project - an advanced research infrastructure designed to revolutionize how philosophical scholars interact with texts, conduct genealogical analyses, and manage complex citation networks using AI technologies.

## High-Level Architecture

### Core Components
- **MCP Server Interface**: Primary interface between Claude Code and the philosophical research system
- **Hybrid Search Engine**: Combines semantic (vector) search with traditional keyword search for optimal philosophical text retrieval
- **Document Processing Pipeline**: Handles ingestion, philosophical-aware chunking, and embedding generation of academic texts
- **Citation Management System**: Integrates with Zotero and provides enhanced AI-powered citation discovery
- **Analysis Engine**: Provides philosophical analysis tools including concept genealogy tracing and argument mapping
- **Database Layer**: PostgreSQL with pgvector extension for hybrid relational/vector storage

### Technology Stack
- **Backend**: Python with FastAPI and asyncio
- **Database**: PostgreSQL 17+ with pgvector extension for vector similarity search
- **Embeddings**: Vertex AI Gemini embedding models (3072-dimensional vectors)
- **MCP Framework**: Latest Model Context Protocol specification (2025-03-26)
- **Caching**: Redis for performance optimization
- **Search**: HNSW indexing for fast vector similarity search

### Integration Points
- **Primary Interface**: Claude Code via MCP protocol
- **Citation Management**: Zotero API for two-way sync
- **External Text Sources**: Z-Library MCP integration
- **Cloud Services**: Vertex AI for embedding generation
- **Storage**: Local with optional cloud backup

## Development Commands

Since this project is in planning phase, the following commands will be implemented:

### Future Development Commands
```bash
# Install dependencies (when implemented)
poetry install

# Run development environment
docker-compose up -d  # Start PostgreSQL, Redis, MinIO services

# Run MCP server locally
python -m src.mcp_server.server

# Run tests
poetry run pytest

# Run linting and type checking
poetry run flake8 src/
poetry run mypy src/

# Database migrations
alembic upgrade head

# Performance benchmarks
poetry run pytest tests/performance/ --benchmark
```

## Philosophical Domain Context

### Text Processing Considerations
- **Philosophical Chunking**: Respect argument boundaries, section markers (§), and logical structures
- **Citation Patterns**: Handle various philosophical citation styles (Author Year), (Author Year: Page), classical references (Aristotle Nicomachean Ethics 1.1.1)
- **Concept Recognition**: Identify philosophical terms, proper names, and technical vocabulary
- **Multi-language Support**: Handle texts in multiple languages common in philosophical research

### Search and Analysis Features
- **Genealogical Tracing**: Track concept evolution across philosophers and time periods
- **Argument Mapping**: Identify premises, conclusions, and logical relationships
- **Influence Networks**: Map citations and intellectual influences between works
- **Tradition Filtering**: Search within specific philosophical traditions (ancient, medieval, modern, contemporary)

### Data Integrity Requirements
- **Citation Accuracy**: Maintain precise bibliographic information for academic standards
- **Version Control**: Track changes to annotations and interpretations
- **Attribution**: Ensure proper credit for collaborative research contributions
- **Scholarly Standards**: Maintain rigorous verification for all claims and sources

## MCP Server Configuration

When the project is implemented, it will be configured for Claude Code as:

```json
{
  "mcpServers": {
    "philosophical-rag": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "DATABASE_URL": "${DATABASE_URL}",
        "VERTEX_AI_PROJECT": "${VERTEX_AI_PROJECT}",
        "REDIS_URL": "${REDIS_URL}"
      }
    }
  }
}
```

## Performance Requirements

- **Search Latency**: Target < 200ms for 95% of philosophical queries
- **Document Processing**: 100 pages per minute throughput
- **Concurrent Users**: Support 100+ simultaneous philosophical researchers
- **Vector Index**: Billion-scale vector support for large philosophical corpora
- **Embedding Generation**: < 1 second per document page

## Security and Privacy

- **Academic Standards**: Maintain strict citation accuracy and attribution
- **Data Protection**: Encrypt sensitive research data and personal notes
- **Access Control**: Role-based permissions for collaborative research teams
- **Audit Trail**: Track all research activities for academic integrity

## Future Implementation Notes

This project represents a comprehensive philosophical research platform that will integrate cutting-edge AI with traditional scholarly methodologies. The architecture is designed to scale from individual researchers to large collaborative projects while maintaining the rigor expected in academic philosophy.

Key philosophical use cases include:
- Literature reviews across multiple philosophical traditions
- Concept genealogy tracing (e.g., "trace 'being' from Parmenides to Heidegger")
- Argument structure analysis and mapping
- Citation network analysis for influence studies
- Cross-cultural philosophical comparisons
- Collaborative annotation and knowledge building

---

## 📝 IMPORTANT INSTRUCTION REMINDERS

- **Never commit changes unless explicitly asked by user**
- **Always run linting and type checking (`poetry run flake8 src/` and `poetry run mypy src/`) before claiming completion**
- **Philosophical research requires rigorous citation accuracy - verify all bibliographic information**
- **MCP server must be stateless and handle concurrent philosophical researchers**
- **Vector embeddings must preserve philosophical meaning and context**
- **Search results must maintain academic relevance and authority**
- **Follow academic standards for all citation formats and attributions**
- **Respect philosophical text structures (arguments, sections, concepts) in all processing**
- **Maintain backward compatibility when updating MCP protocol or database schema**