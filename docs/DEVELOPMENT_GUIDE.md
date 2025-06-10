# Philosophical Research RAG MCP Server: Development & Contribution Guide
**Version:** 1.0
**Last Updated:** January 6, 2025

This document contains the core principles, workflows, and patterns for the Philosophical Research RAG MCP Server. Adherence is mandatory for all contributions.

## 1. The SPARC-V-L³ Development Protocol
Every non-trivial task must follow this cycle:
1. **S - Specification:** Fully understand the goal and requirements, including philosophical research context.
2. **P - Plan:** Create detailed, step-by-step plan with pytest + MCP Protocol Testing approach.
3. **A - Architecture:** Consult `docs/ARCHITECTURE.md` and analyze impact on hybrid search system.
4. **R - Refine:** Implement following TDD cycle with philosophical domain validation.
5. **C - Complete:** Ensure all tests pass (`poetry run pytest`).
6. **V - Verify:** Run linting (`poetry run flake8 src/`), type checking (`poetry run mypy src/`), and philosophical research workflow validation.
7. **L¹ - Log:** Update `ACTIVITY_LOG.md` with detailed record.
8. **L² - Learn:** Self-analysis in `SELF_ANALYSIS_LOG.md`.
9. **L³ - Level Up:** Update this guide if systemic lessons learned.

## 2. pytest + MCP Protocol Testing Requirements

### Test Categories Required:
- **Unit Tests:** All core components (search engine, document processor, analysis engine)
- **Integration Tests:** MCP server tools, resources, and prompts with Claude Code
- **Performance Tests:** Search latency < 200ms, document processing throughput
- **Philosophical Domain Tests:** Citation accuracy, argument mapping, concept genealogy
- **Database Tests:** PostgreSQL + pgvector operations and migrations

### Test Commands:
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run performance benchmarks
poetry run pytest tests/performance/ --benchmark

# Run specific philosophical research tests
poetry run pytest tests/philosophical/

# Run MCP protocol integration tests
poetry run pytest tests/integration/test_mcp_server.py
```

### Test Data Requirements:
- Sample philosophical texts from multiple traditions
- Citation examples in various formats
- Concept genealogy test cases
- Argument structure examples

## 3. MCP Server + Hybrid Search Architecture Patterns

### Core Architecture Principles:
- **Separation of Concerns:** MCP interface, search engine, document processing, and analysis engine are distinct layers
- **Stateless Operations:** MCP server maintains no session state
- **Async by Default:** All operations use asyncio for concurrent philosophical researchers
- **Error Resilience:** Graceful degradation when components fail

### Component Structure:
```
src/
├── mcp_server/           # MCP protocol implementation
│   ├── tools/           # Search, analysis, citation tools
│   ├── resources/       # Document and note resources
│   └── prompts/         # Research workflow prompts
├── core/                # Core business logic
│   ├── search/          # Hybrid search engine
│   ├── documents/       # Document processing
│   ├── citations/       # Citation management
│   └── analysis/        # Philosophical analysis
└── infrastructure/      # Database, caching, external APIs
```

### Database Schema Patterns:
- **Hybrid Storage:** Relational data in PostgreSQL, vectors in pgvector
- **Citation Networks:** Proper foreign key relationships for academic integrity
- **Versioning:** Track changes to annotations and interpretations
- **Performance:** HNSW indexing for vector similarity search

## 4. Version Control & Workflow

**Branching Strategy:**
- `main`: Production-ready code with complete test coverage
- `develop`: Integration branch for new features
- `feature/PHIL-123-description`: Individual philosophical research features

**Commit Message Format:**
```
feat(search): add concept genealogy tracing for Heideggerian analysis

- Implement temporal concept evolution tracking
- Add citation network influence mapping
- Include philosophical tradition filtering
- Add comprehensive test coverage for genealogy features

Resolves: PHIL-123
```

**Workflow Decision Matrix:**
- **Use PR workflow for:** All MCP protocol changes, database schema modifications, new philosophical analysis features
- **Use direct merge for:** Documentation updates, test improvements, minor bug fixes

## 5. The Triple-Log System
1. **Application Log:** Structured JSON logs in `logs/application.log` with philosophical research context
2. **`ACTIVITY_LOG.md`:** Immutable development action log
3. **`FEEDBACK_LOG.md` & `SELF_ANALYSIS_LOG.md`:** Learning and improvement logs

## 6. Verification Protocols (Prevents Critical Errors)

### High-Risk Operations Verification Protocol

1. **Database Schema Changes:**
   ```bash
   # Verify migration safety
   alembic upgrade head --sql > migration_review.sql
   # Review SQL for data loss risks
   # Run on staging environment first
   # Verify backward compatibility
   ```

2. **MCP Protocol Modifications:**
   - Verify all tools maintain input schema compatibility
   - Test with Claude Code integration
   - Ensure graceful error handling
   - Validate philosophical research workflows

### Philosophical Research Verification
- **Citation Accuracy:** Verify all bibliographic information against authoritative sources
- **Argument Mapping:** Validate logical structure recognition with known philosophical arguments
- **Concept Genealogy:** Test temporal ordering and influence relationships
- **Search Relevance:** Validate semantic similarity maintains philosophical meaning

## 7. Context Awareness Protocol

### Before Making Decisions:
1. **Project Context Assessment:**
   - What philosophical research domain is this for?
   - Which citation standards apply?
   - What philosophical traditions are involved?

2. **Technology Context Assessment:**
   - What Python version does this project use?
   - What are the existing MCP patterns in this codebase?
   - What poetry packages are available?

### Before Implementation:
1. **Pattern Analysis:** Understand existing solutions in codebase
2. **Dependency Verification:** Confirm all required Python packages and external APIs
3. **Constraint Assessment:** Academic accuracy requirements, performance targets, MCP protocol compliance

## 8. Requirement Analysis Protocol

### Before Starting Tasks:
1. **Complete Requirement Reading:**
   - Read entire task description and linked philosophical research documents
   - Read all `test_*.py` files that define expected behavior
   - Read all error messages completely

2. **Comprehension Verification:**
   - Can you explain the philosophical research requirement clearly?
   - What are the academic accuracy criteria?
   - What is the expected user workflow?

### For Philosophical Research Specific Issues:
- **Understand the Domain:** Read relevant philosophical texts and citations
- **Verify Academic Standards:** Ensure citation formats and attribution requirements
- **Consider Multiple Traditions:** Account for different philosophical approaches and languages
- **Validate with Experts:** When in doubt, flag for philosophical domain expert review

## 9. Python + FastAPI + PostgreSQL Specific Patterns

### Code Structure Patterns:
```python
# Async by default
async def search_philosophical_texts(query: str) -> List[SearchResult]:
    """Always use async for database and external API calls"""
    pass

# Proper error handling
@handle_philosophical_research_errors
async def process_document(file_path: str) -> ProcessedDocument:
    """Handle domain-specific errors gracefully"""
    pass

# Type hints everywhere
from typing import List, Dict, Optional, Union
from pydantic import BaseModel

class CitationRequest(BaseModel):
    author: str
    work: str
    page: Optional[str] = None
```

### Database Patterns:
- Use SQLAlchemy async sessions
- Implement proper transaction handling
- Use pgvector for embedding storage
- Maintain referential integrity for citations

## 10. Philosophical Research Domain Specific Guidelines

### Citation Management:
- Support multiple citation formats: (Author Year), (Author Year: Page), classical references
- Maintain bidirectional links between citing and cited works
- Preserve original context and page references
- Validate against known philosophical bibliographies

### Text Processing:
- Respect philosophical argument structures
- Handle section markers (§) and traditional divisions
- Preserve philosophical terminology and proper names
- Support multiple languages (ancient Greek, Latin, German, French, etc.)

### Search and Analysis:
- Semantic similarity must preserve philosophical meaning
- Concept genealogy requires temporal and influence validation
- Argument mapping must respect logical structure
- Results must maintain academic authority and relevance

### Collaboration Features:
- Track attribution for all contributions
- Maintain version history for interpretations
- Support collaborative annotation with proper attribution
- Ensure data integrity for shared research spaces