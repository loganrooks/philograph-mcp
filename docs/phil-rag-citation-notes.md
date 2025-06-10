# Citation and Notes Management System

## Overview

The Citation and Notes Management System is a cornerstone of the Philosophical Research RAG MCP Server, providing sophisticated tools for managing bibliographic data, tracking intellectual lineages, and organizing research insights. This system goes beyond traditional citation management by integrating AI-powered features specifically designed for philosophical research.

## Core Design Principles

### 1. **Philosophical Context Awareness**
- Understands different citation styles used in philosophy
- Recognizes classical references (e.g., Aristotle 1094a1)
- Handles multi-edition works and translations
- Preserves argumentative context around citations

### 2. **Bidirectional Citation Tracking**
- Track not just what a work cites, but what cites it
- Build comprehensive citation networks
- Identify influential works and hidden connections
- Trace intellectual lineages across time

### 3. **Hierarchical Note Organization**
- Support for complex, nested argument structures
- Link notes to specific passages, concepts, or arguments
- Enable multiple organizational schemes simultaneously
- Preserve the development of ideas over time

### 4. **AI-Enhanced Features**
- Automatic citation extraction from texts
- Smart citation matching and deduplication
- Context-aware citation suggestions
- Intelligent note categorization

## System Architecture

### Database Schema

```sql
-- Enhanced citation storage
CREATE TABLE citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Basic citation info
    citation_key TEXT UNIQUE NOT NULL, -- e.g., "aristotle_ethics_1094a"
    citation_type TEXT NOT NULL, -- book, article, chapter, classical, etc.
    
    -- Bibliographic data
    authors JSONB NOT NULL, -- Array of author objects
    title TEXT NOT NULL,
    publication_year INTEGER,
    publisher TEXT,
    journal TEXT,
    volume TEXT,
    issue TEXT,
    pages TEXT,
    doi TEXT,
    isbn TEXT,
    url TEXT,
    
    -- Philosophical specific
    original_language TEXT,
    translator TEXT,
    edition TEXT,
    standard_pagination TEXT, -- Bekker, Stephanus, etc.
    philosophical_tradition TEXT[],
    
    -- Integration
    zotero_key TEXT,
    matched_document_id UUID REFERENCES documents(id),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by UUID NOT NULL,
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(authors::text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(journal, '')), 'C')
    ) STORED
);

-- Citation relationships
CREATE TABLE citation_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    citing_work_id UUID REFERENCES citations(id),
    cited_work_id UUID REFERENCES citations(id),
    citation_context TEXT, -- The text around the citation
    page_number TEXT,
    section TEXT,
    citation_type TEXT, -- direct, indirect, critical, supportive
    confidence_score FLOAT,
    verified BOOLEAN DEFAULT FALSE,
    UNIQUE(citing_work_id, cited_work_id, page_number)
);

-- Notes system
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    
    -- Note content
    title TEXT,
    content TEXT NOT NULL,
    note_type TEXT, -- idea, critique, summary, question, connection
    
    -- Hierarchical structure
    parent_note_id UUID REFERENCES notes(id),
    position INTEGER, -- Order within parent
    
    -- Associations
    document_id UUID REFERENCES documents(id),
    chunk_id UUID REFERENCES chunks(id),
    citation_id UUID REFERENCES citations(id),
    
    -- Location specifics
    page_number INTEGER,
    paragraph_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    selected_text TEXT,
    
    -- Organization
    tags TEXT[],
    categories TEXT[],
    importance INTEGER CHECK (importance BETWEEN 1 AND 5),
    
    -- Versioning
    version INTEGER DEFAULT 1,
    previous_version_id UUID REFERENCES notes(id),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(array_to_string(tags, ' '), '')), 'C')
    ) STORED
);

-- Note relationships
CREATE TABLE note_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_note_id UUID REFERENCES notes(id),
    to_note_id UUID REFERENCES notes(id),
    relationship_type TEXT, -- supports, contradicts, extends, questions
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_citations_authors ON citations USING GIN (authors);
CREATE INDEX idx_citations_tradition ON citations USING GIN (philosophical_tradition);
CREATE INDEX idx_citations_search ON citations USING GIN (search_vector);
CREATE INDEX idx_notes_user ON notes (user_id);
CREATE INDEX idx_notes_tags ON notes USING GIN (tags);
CREATE INDEX idx_notes_search ON notes USING GIN (search_vector);
```

### Citation Processing Pipeline

```python
# src/services/citation_manager/citation_processor.py
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from fuzzywuzzy import fuzz
import asyncio

@dataclass
class ProcessedCitation:
    raw_text: str
    parsed_data: Dict[str, Any]
    confidence: float
    match_candidates: List[Dict[str, Any]]
    final_match: Optional[str] = None

class CitationProcessor:
    def __init__(self, db_session, embedding_service):
        self.db_session = db_session
        self.embedding_service = embedding_service
        self.citation_patterns = self._load_citation_patterns()
        
    def _load_citation_patterns(self):
        """Load citation patterns for different styles"""
        return {
            'modern': [
                # (Author Year: Page)
                re.compile(r'(?P<author>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?P<year>\d{4})(?::\s*(?P<page>\d+(?:-\d+)?))?\)'),
                # Author (Year, p. X)
                re.compile(r'(?P<author>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?P<year>\d{4}),\s*p\.?\s*(?P<page>\d+)\)'),
            ],
            'classical': [
                # Aristotle, Nicomachean Ethics, 1094a1-5
                re.compile(r'(?P<author>Aristotle|Plato|Aquinas|Augustine),\s*(?P<work>[^,]+),\s*(?P<ref>\d+[a-z]\d*(?:-\d+)?)'),
                # Kant, KrV A123/B456
                re.compile(r'(?P<author>Kant),\s*(?P<work>KrV|KpV|KU)\s*(?P<ref>[AB]\d+(?:/[AB]\d+)?)'),
            ],
            'footnote': [
                # Detailed footnote style
                re.compile(r'(?P<author>[^,]+),\s*"(?P<title>[^"]+),"?\s*(?:in\s+)?(?P<journal>[^,]+)?,?\s*(?P<year>\d{4})'),
            ]
        }
        
    async def process_citations(
        self, 
        text: str,
        source_document_id: str
    ) -> List[ProcessedCitation]:
        """Extract and process all citations from text"""
        # Extract raw citations
        raw_citations = self._extract_citations(text)
        
        # Process each citation
        processed = []
        for raw in raw_citations:
            # Parse citation
            parsed = self._parse_citation(raw)
            
            # Find potential matches
            candidates = await self._find_match_candidates(parsed)
            
            # Score and rank candidates
            best_match = await self._select_best_match(parsed, candidates)
            
            processed.append(ProcessedCitation(
                raw_text=raw['text'],
                parsed_data=parsed,
                confidence=raw['confidence'],
                match_candidates=candidates,
                final_match=best_match
            ))
            
        # Store relationships
        await self._store_citation_relationships(
            source_document_id,
            processed
        )
        
        return processed
        
    def _parse_citation(self, raw_citation: Dict) -> Dict[str, Any]:
        """Parse citation into structured data"""
        text = raw_citation['text']
        citation_type = raw_citation['type']
        
        parsed = {
            'raw_text': text,
            'type': citation_type
        }
        
        # Try each pattern
        for style, patterns in self.citation_patterns.items():
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    parsed.update(match.groupdict())
                    parsed['style'] = style
                    break
                    
        # Additional parsing for complex cases
        if 'author' in parsed:
            parsed['authors'] = self._parse_authors(parsed['author'])
            
        return parsed
        
    async def _find_match_candidates(
        self, 
        parsed_citation: Dict
    ) -> List[Dict[str, Any]]:
        """Find potential matches in database"""
        candidates = []
        
        # Build search query
        query = self.db_session.query(Citation)
        
        # Filter by author if available
        if 'authors' in parsed_citation:
            for author in parsed_citation['authors']:
                query = query.filter(
                    Citation.authors.contains([{'name': author}])
                )
                
        # Filter by year if available
        if 'year' in parsed_citation:
            year = int(parsed_citation['year'])
            query = query.filter(
                Citation.publication_year.between(year - 2, year + 2)
            )
            
        # Execute query
        potential_matches = await query.limit(20).all()
        
        # Score each candidate
        for match in potential_matches:
            score = self._calculate_match_score(parsed_citation, match)
            candidates.append({
                'citation': match,
                'score': score,
                'reasons': self._explain_match(parsed_citation, match)
            })
            
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
        
    def _calculate_match_score(
        self, 
        parsed: Dict, 
        candidate: Citation
    ) -> float:
        """Calculate similarity score between parsed and candidate"""
        score = 0.0
        
        # Author similarity
        if 'authors' in parsed:
            author_scores = []
            for parsed_author in parsed['authors']:
                for cand_author in candidate.authors:
                    author_score = fuzz.ratio(
                        parsed_author.lower(),
                        cand_author.get('name', '').lower()
                    ) / 100.0
                    author_scores.append(author_score)
            if author_scores:
                score += max(author_scores) * 0.4
                
        # Title similarity
        if 'title' in parsed and candidate.title:
            title_score = fuzz.partial_ratio(
                parsed['title'].lower(),
                candidate.title.lower()
            ) / 100.0
            score += title_score * 0.3
            
        # Year match
        if 'year' in parsed and candidate.publication_year:
            year_diff = abs(int(parsed['year']) - candidate.publication_year)
            year_score = max(0, 1 - (year_diff / 10))
            score += year_score * 0.2
            
        # Work/journal match
        if 'work' in parsed or 'journal' in parsed:
            work = parsed.get('work') or parsed.get('journal')
            if candidate.journal:
                work_score = fuzz.partial_ratio(
                    work.lower(),
                    candidate.journal.lower()
                ) / 100.0
                score += work_score * 0.1
                
        return score
```

### Advanced Note Management

```python
# src/services/notes/note_manager.py
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
from sqlalchemy import select, and_, or_

class NoteManager:
    def __init__(self, db_session, embedding_service, cache_manager):
        self.db_session = db_session
        self.embedding_service = embedding_service
        self.cache = cache_manager
        
    async def create_note(
        self,
        user_id: str,
        content: str,
        title: Optional[str] = None,
        note_type: str = "idea",
        associations: Dict[str, Any] = None,
        tags: List[str] = None,
        parent_id: Optional[str] = None
    ) -> Note:
        """Create a new note with associations"""
        # Generate title if not provided
        if not title:
            title = await self._generate_title(content)
            
        # Create note
        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            note_type=note_type,
            tags=tags or [],
            parent_note_id=parent_id
        )
        
        # Set associations
        if associations:
            if 'document_id' in associations:
                note.document_id = associations['document_id']
            if 'chunk_id' in associations:
                note.chunk_id = associations['chunk_id']
            if 'citation_id' in associations:
                note.citation_id = associations['citation_id']
            if 'selected_text' in associations:
                note.selected_text = associations['selected_text']
                
        # Calculate position if has parent
        if parent_id:
            note.position = await self._get_next_position(parent_id)
            
        self.db_session.add(note)
        await self.db_session.commit()
        
        # Invalidate caches
        await self.cache.invalidate(f"notes:user:{user_id}:*")
        
        return note
        
    async def _generate_title(self, content: str) -> str:
        """Generate title from content using AI"""
        # Simple version - take first line or generate summary
        lines = content.split('\n')
        if lines and len(lines[0]) < 100:
            return lines[0]
            
        # For longer content, generate summary
        # This could call Claude or another LLM
        return content[:50] + "..."
        
    async def organize_notes(
        self,
        user_id: str,
        organization_scheme: str = "hierarchical"
    ) -> Dict[str, Any]:
        """Organize notes by different schemes"""
        notes = await self._get_user_notes(user_id)
        
        if organization_scheme == "hierarchical":
            return self._build_hierarchy(notes)
        elif organization_scheme == "chronological":
            return self._organize_chronologically(notes)
        elif organization_scheme == "thematic":
            return await self._organize_thematically(notes)
        elif organization_scheme == "citation_based":
            return self._organize_by_citations(notes)
        else:
            raise ValueError(f"Unknown organization scheme: {organization_scheme}")
            
    def _build_hierarchy(self, notes: List[Note]) -> Dict[str, Any]:
        """Build hierarchical structure from notes"""
        # Create lookup maps
        note_map = {str(note.id): note for note in notes}
        children_map = {}
        roots = []
        
        # Build parent-child relationships
        for note in notes:
            if note.parent_note_id:
                parent_id = str(note.parent_note_id)
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(note)
            else:
                roots.append(note)
                
        # Build tree structure
        def build_tree(note):
            node = {
                'id': str(note.id),
                'title': note.title,
                'content': note.content,
                'type': note.note_type,
                'tags': note.tags,
                'children': []
            }
            
            # Add children
            if str(note.id) in children_map:
                children = children_map[str(note.id)]
                children.sort(key=lambda x: x.position or 0)
                for child in children:
                    node['children'].append(build_tree(child))
                    
            return node
            
        # Build forest
        forest = []
        roots.sort(key=lambda x: x.created_at)
        for root in roots:
            forest.append(build_tree(root))
            
        return {
            'type': 'hierarchical',
            'trees': forest,
            'total_notes': len(notes),
            'max_depth': self._calculate_max_depth(forest)
        }
        
    async def _organize_thematically(
        self, 
        notes: List[Note]
    ) -> Dict[str, Any]:
        """Organize notes by themes using embeddings"""
        # Generate embeddings for all notes
        contents = [f"{note.title}\n{note.content}" for note in notes]
        embeddings = await self.embedding_service.generate_embeddings(contents)
        
        # Cluster notes
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Convert to numpy array
        X = np.array(embeddings)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(X)
        
        # Organize by clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {
                    'notes': [],
                    'theme': None,
                    'keywords': []
                }
            clusters[label]['notes'].append(notes[i])
            
        # Generate theme names for each cluster
        for label, cluster in clusters.items():
            if label != -1:  # Not noise
                cluster['theme'] = await self._generate_theme_name(
                    cluster['notes']
                )
                cluster['keywords'] = self._extract_keywords(
                    cluster['notes']
                )
                
        return {
            'type': 'thematic',
            'clusters': clusters,
            'total_notes': len(notes),
            'num_themes': len([c for c in clusters if c != -1])
        }
        
    async def link_notes(
        self,
        from_note_id: str,
        to_note_id: str,
        relationship_type: str,
        description: Optional[str] = None
    ):
        """Create relationship between notes"""
        relationship = NoteRelationship(
            from_note_id=from_note_id,
            to_note_id=to_note_id,
            relationship_type=relationship_type,
            description=description
        )
        
        self.db_session.add(relationship)
        await self.db_session.commit()
        
    async def find_related_notes(
        self,
        note_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find notes related to given note"""
        # Direct relationships
        query = select(NoteRelationship).where(
            or_(
                NoteRelationship.from_note_id == note_id,
                NoteRelationship.to_note_id == note_id
            )
        )
        
        if relationship_types:
            query = query.where(
                NoteRelationship.relationship_type.in_(relationship_types)
            )
            
        relationships = await self.db_session.execute(query)
        
        # Build response
        related = []
        for rel in relationships:
            related_note_id = (
                rel.to_note_id if rel.from_note_id == note_id 
                else rel.from_note_id
            )
            related_note = await self.db_session.get(Note, related_note_id)
            
            related.append({
                'note': related_note,
                'relationship': rel.relationship_type,
                'direction': 'outgoing' if rel.from_note_id == note_id else 'incoming',
                'description': rel.description
            })
            
        return related
```

### Citation Network Analysis

```python
# src/services/citation_manager/network_analyzer.py
import networkx as nx
from typing import List, Dict, Any, Set
from collections import defaultdict
import asyncio

class CitationNetworkAnalyzer:
    def __init__(self, db_session):
        self.db_session = db_session
        
    async def build_citation_network(
        self,
        root_citations: List[str] = None,
        depth: int = 2,
        min_citations: int = 2
    ) -> nx.DiGraph:
        """Build citation network graph"""
        G = nx.DiGraph()
        
        # Start with root citations or all highly cited works
        if root_citations:
            to_process = set(root_citations)
        else:
            to_process = await self._get_highly_cited_works(min_citations)
            
        processed = set()
        
        # Build network iteratively
        for _ in range(depth):
            next_layer = set()
            
            for citation_id in to_process:
                if citation_id in processed:
                    continue
                    
                # Get citation data
                citation = await self.db_session.get(Citation, citation_id)
                if not citation:
                    continue
                    
                # Add node
                G.add_node(
                    citation_id,
                    title=citation.title,
                    authors=citation.authors,
                    year=citation.publication_year,
                    tradition=citation.philosophical_tradition
                )
                
                # Get relationships
                relationships = await self._get_citation_relationships(citation_id)
                
                for rel in relationships:
                    # Add edge
                    G.add_edge(
                        rel.citing_work_id,
                        rel.cited_work_id,
                        type=rel.citation_type,
                        context=rel.citation_context
                    )
                    
                    # Add to next layer
                    next_layer.add(rel.cited_work_id)
                    next_layer.add(rel.citing_work_id)
                    
                processed.add(citation_id)
                
            to_process = next_layer - processed
            
        return G
        
    async def analyze_influence(
        self,
        citation_id: str
    ) -> Dict[str, Any]:
        """Analyze influence of a work"""
        # Build local network
        network = await self.build_citation_network(
            root_citations=[citation_id],
            depth=3
        )
        
        # Calculate metrics
        metrics = {
            'direct_citations': network.in_degree(citation_id),
            'total_influence': len(nx.descendants(network, citation_id)),
            'citation_chain_length': self._max_chain_length(network, citation_id),
            'centrality': nx.betweenness_centrality(network).get(citation_id, 0),
            'clustering_coefficient': nx.clustering(network.to_undirected()).get(citation_id, 0)
        }
        
        # Find key citers
        citers = []
        for node in network.predecessors(citation_id):
            citation = await self.db_session.get(Citation, node)
            citers.append({
                'id': node,
                'title': citation.title,
                'authors': citation.authors,
                'influence': network.in_degree(node)
            })
            
        # Sort by influence
        citers.sort(key=lambda x: x['influence'], reverse=True)
        metrics['key_citers'] = citers[:10]
        
        # Find influenced works
        influenced = []
        for node in nx.descendants(network, citation_id):
            if network.has_edge(citation_id, node):
                citation = await self.db_session.get(Citation, node)
                influenced.append({
                    'id': node,
                    'title': citation.title,
                    'authors': citation.authors,
                    'citation_distance': nx.shortest_path_length(
                        network, citation_id, node
                    )
                })
                
        metrics['influenced_works'] = influenced
        
        return metrics
        
    async def find_missing_citations(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find frequently referenced but missing works"""
        # Get all unmatched citations
        query = """
            SELECT 
                cited_text,
                cited_author,
                cited_work,
                COUNT(*) as reference_count
            FROM citation_relationships
            WHERE matched_document_id IS NULL
            GROUP BY cited_text, cited_author, cited_work
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
            LIMIT 50
        """
        
        results = await self.db_session.execute(query)
        
        missing_works = []
        for row in results:
            missing_works.append({
                'text': row.cited_text,
                'author': row.cited_author,
                'work': row.cited_work,
                'reference_count': row.reference_count,
                'search_query': self._generate_search_query(row)
            })
            
        return missing_works
        
    def _generate_search_query(self, citation_info) -> str:
        """Generate search query for finding missing work"""
        parts = []
        
        if citation_info.cited_author:
            parts.append(citation_info.cited_author)
        if citation_info.cited_work:
            parts.append(citation_info.cited_work)
            
        return ' '.join(parts)
```

### AI-Enhanced Citation Features

```python
# src/services/citation_manager/ai_features.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

class AIcitationEnhancer:
    def __init__(self, llm_service, embedding_service, db_session):
        self.llm = llm_service
        self.embeddings = embedding_service
        self.db_session = db_session
        
    async def suggest_citations(
        self,
        context: str,
        num_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest relevant citations for given context"""
        # Extract key concepts
        concepts = await self._extract_concepts(context)
        
        # Generate search embedding
        context_embedding = await self.embeddings.generate_embeddings([context])
        
        # Search for relevant citations
        query = """
            SELECT 
                c.*,
                1 - (ce.embedding <=> %s::vector) as similarity
            FROM citations c
            JOIN citation_embeddings ce ON c.id = ce.citation_id
            WHERE 1 - (ce.embedding <=> %s::vector) > 0.7
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        results = await self.db_session.execute(
            query,
            (context_embedding[0], context_embedding[0], num_suggestions * 2)
        )
        
        # Score and filter
        suggestions = []
        for row in results:
            relevance = await self._calculate_relevance(
                context,
                concepts,
                row
            )
            
            if relevance > 0.6:
                suggestions.append({
                    'citation': row,
                    'relevance_score': relevance,
                    'reason': await self._explain_relevance(context, row)
                })
                
        # Sort by relevance
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return suggestions[:num_suggestions]
        
    async def _extract_concepts(self, text: str) -> List[str]:
        """Extract key philosophical concepts from text"""
        prompt = f"""
        Extract key philosophical concepts from this text.
        Return only the concepts as a comma-separated list.
        
        Text: {text}
        
        Concepts:
        """
        
        response = await self.llm.generate(prompt)
        concepts = [c.strip() for c in response.split(',')]
        
        return concepts
        
    async def verify_citation(
        self,
        citation_text: str,
        source_context: str
    ) -> Dict[str, Any]:
        """Verify citation accuracy and context"""
        # Find best match
        matches = await self._find_citation_matches(citation_text)
        
        if not matches:
            return {
                'verified': False,
                'confidence': 0.0,
                'issues': ['No matching citation found']
            }
            
        best_match = matches[0]
        
        # Verify context appropriateness
        verification = await self._verify_context(
            citation_text,
            source_context,
            best_match
        )
        
        return {
            'verified': verification['is_appropriate'],
            'confidence': verification['confidence'],
            'matched_citation': best_match,
            'issues': verification.get('issues', []),
            'suggestions': verification.get('suggestions', [])
        }
        
    async def _verify_context(
        self,
        citation_text: str,
        source_context: str,
        matched_citation: Dict
    ) -> Dict[str, Any]:
        """Verify citation is used appropriately in context"""
        # Get original context if available
        if matched_citation.get('document_id'):
            original_context = await self._get_original_context(
                matched_citation['document_id'],
                citation_text
            )
        else:
            original_context = None
            
        # Use LLM to verify
        prompt = f"""
        Verify if this citation is used appropriately:
        
        Citation: {citation_text}
        
        Used in context: {source_context}
        
        Original work: {matched_citation.get('title')} by {matched_citation.get('authors')}
        {f"Original context: {original_context}" if original_context else ""}
        
        Evaluate:
        1. Is the citation used accurately?
        2. Does it support the argument being made?
        3. Are there any misrepresentations?
        
        Respond in JSON format:
        {{
            "is_appropriate": boolean,
            "confidence": float (0-1),
            "issues": [list of any issues],
            "suggestions": [list of improvements]
        }}
        """
        
        response = await self.llm.generate(prompt, response_format="json")
        return response
        
    async def generate_bibliography(
        self,
        citation_ids: List[str],
        style: str = "chicago",
        group_by: Optional[str] = None
    ) -> str:
        """Generate formatted bibliography"""
        # Fetch all citations
        citations = []
        for cid in citation_ids:
            citation = await self.db_session.get(Citation, cid)
            if citation:
                citations.append(citation)
                
        # Sort appropriately
        if style in ["chicago", "mla"]:
            citations.sort(key=lambda x: x.authors[0]['name'] if x.authors else '')
        else:  # APA, etc.
            citations.sort(key=lambda x: (x.authors[0]['name'] if x.authors else '', x.publication_year or 0))
            
        # Group if requested
        if group_by == "type":
            grouped = self._group_by_type(citations)
        elif group_by == "tradition":
            grouped = self._group_by_tradition(citations)
        else:
            grouped = {"All References": citations}
            
        # Format bibliography
        bibliography = []
        for group_name, group_citations in grouped.items():
            if group_by:
                bibliography.append(f"\n## {group_name}\n")
                
            for citation in group_citations:
                formatted = self._format_citation(citation, style)
                bibliography.append(formatted)
                
        return '\n'.join(bibliography)
        
    def _format_citation(self, citation: Citation, style: str) -> str:
        """Format citation according to style"""
        if style == "chicago":
            return self._format_chicago(citation)
        elif style == "mla":
            return self._format_mla(citation)
        elif style == "apa":
            return self._format_apa(citation)
        else:
            raise ValueError(f"Unknown citation style: {style}")
            
    def _format_chicago(self, citation: Citation) -> str:
        """Format in Chicago style"""
        parts = []
        
        # Authors
        if citation.authors:
            author_names = []
            for author in citation.authors:
                name = author.get('name', '')
                if ',' in name:  # Already formatted
                    author_names.append(name)
                else:  # Need to format
                    parts = name.split()
                    if len(parts) >= 2:
                        author_names.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
                    else:
                        author_names.append(name)
                        
            if len(author_names) == 1:
                parts.append(author_names[0] + ".")
            elif len(author_names) == 2:
                parts.append(f"{author_names[0]}, and {author_names[1]}.")
            else:
                parts.append(f"{author_names[0]}, et al.")
                
        # Title
        if citation.title:
            if citation.citation_type == 'book':
                parts.append(f"*{citation.title}*.")
            else:
                parts.append(f'"{citation.title}."')
                
        # Publication info
        if citation.journal:
            parts.append(f"*{citation.journal}*")
            if citation.volume:
                parts.append(f"{citation.volume}")
                if citation.issue:
                    parts.append(f"({citation.issue})")
                    
        # Publisher for books
        if citation.citation_type == 'book' and citation.publisher:
            parts.append(f"{citation.publisher},")
            
        # Year
        if citation.publication_year:
            parts.append(f"{citation.publication_year}.")
            
        # Pages
        if citation.pages:
            parts.append(f"{citation.pages}.")
            
        return ' '.join(parts)
```

## Integration Features

### Zotero Synchronization

```python
# src/services/citation_manager/zotero_sync.py
class EnhancedZoteroSync(ZoteroIntegration):
    """Enhanced Zotero sync with philosophical features"""
    
    async def sync_with_ai_enhancement(self):
        """Sync with AI-powered enhancements"""
        # Get local and remote citations
        sync_result = await self.sync_library()
        
        # Enhance citations with AI
        for citation in sync_result['synced']:
            # Extract philosophical tradition
            if not citation.philosophical_tradition:
                tradition = await self._identify_tradition(citation)
                citation.philosophical_tradition = tradition
                
            # Generate keywords
            if not citation.tags:
                keywords = await self._generate_keywords(citation)
                citation.tags = keywords
                
            # Find related works
            related = await self._find_related_works(citation)
            citation.metadata['related_works'] = related
            
        await self.db_session.commit()
        
    async def _identify_tradition(self, citation: Citation) -> List[str]:
        """Identify philosophical tradition using AI"""
        prompt = f"""
        Identify the philosophical tradition(s) for this work:
        Title: {citation.title}
        Authors: {citation.authors}
        Abstract: {citation.metadata.get('abstract', 'N/A')}
        
        Return as comma-separated list (e.g., "phenomenology, existentialism")
        """
        
        response = await self.llm.generate(prompt)
        traditions = [t.strip() for t in response.split(',')]
        
        return traditions
```

### Note Export and Templates

```python
# src/services/notes/exporters.py
class NoteExporter:
    """Export notes in various formats"""
    
    async def export_notes(
        self,
        note_ids: List[str],
        format: str = "markdown",
        include_citations: bool = True,
        template: Optional[str] = None
    ) -> str:
        """Export notes in specified format"""
        notes = []
        for note_id in note_ids:
            note = await self.db_session.get(Note, note_id)
            if note:
                notes.append(note)
                
        if format == "markdown":
            return self._export_markdown(notes, include_citations, template)
        elif format == "latex":
            return self._export_latex(notes, include_citations, template)
        elif format == "obsidian":
            return self._export_obsidian(notes, include_citations)
        elif format == "roam":
            return self._export_roam(notes, include_citations)
        else:
            raise ValueError(f"Unknown export format: {format}")
            
    def _export_markdown(
        self,
        notes: List[Note],
        include_citations: bool,
        template: Optional[str]
    ) -> str:
        """Export as Markdown"""
        if template == "paper_outline":
            return self._paper_outline_template(notes, include_citations)
        elif template == "literature_review":
            return self._literature_review_template(notes, include_citations)
        else:
            # Default export
            output = []
            for note in notes:
                output.append(f"# {note.title}\n")
                output.append(f"*Created: {note.created_at}*\n")
                
                if note.tags:
                    output.append(f"Tags: {', '.join(note.tags)}\n")
                    
                output.append(f"\n{note.content}\n")
                
                if include_citations and note.citation_id:
                    citation = self.db_session.get(Citation, note.citation_id)
                    if citation:
                        output.append(f"\n**Reference**: {self._format_citation(citation)}\n")
                        
                output.append("\n---\n")
                
            return '\n'.join(output)
            
    def _paper_outline_template(
        self,
        notes: List[Note],
        include_citations: bool
    ) -> str:
        """Generate paper outline from notes"""
        # Organize notes by type
        intro_notes = [n for n in notes if 'introduction' in n.tags]
        argument_notes = [n for n in notes if n.note_type == 'argument']
        evidence_notes = [n for n in notes if n.note_type == 'evidence']
        conclusion_notes = [n for n in notes if 'conclusion' in n.tags]
        
        outline = []
        outline.append("# Paper Outline\n")
        
        # Introduction
        outline.append("## I. Introduction\n")
        for note in intro_notes:
            outline.append(f"- {note.content[:100]}...\n")
            
        # Main arguments
        outline.append("\n## II. Main Arguments\n")
        for i, note in enumerate(argument_notes, 1):
            outline.append(f"\n### {i}. {note.title}\n")
            outline.append(f"{note.content}\n")
            
            # Supporting evidence
            evidence = [e for e in evidence_notes if e.parent_note_id == note.id]
            if evidence:
                outline.append("\n**Supporting Evidence:**\n")
                for e in evidence:
                    outline.append(f"- {e.content[:100]}...\n")
                    if include_citations and e.citation_id:
                        citation = self.db_session.get(Citation, e.citation_id)
                        outline.append(f"  - Source: {self._format_citation_short(citation)}\n")
                        
        # Conclusion
        outline.append("\n## III. Conclusion\n")
        for note in conclusion_notes:
            outline.append(f"- {note.content[:100]}...\n")
            
        return '\n'.join(outline)
```

## User Interface Components

### Citation Search Interface

```typescript
// ui/components/CitationSearch.tsx
interface CitationSearchProps {
  onSelect: (citation: Citation) => void;
  context?: string;
}

export const CitationSearch: React.FC<CitationSearchProps> = ({ 
  onSelect, 
  context 
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Citation[]>([]);
  const [suggestions, setSuggestions] = useState<Citation[]>([]);
  const [loading, setLoading] = useState(false);
  
  // Get AI suggestions based on context
  useEffect(() => {
    if (context) {
      fetchSuggestions(context);
    }
  }, [context]);
  
  const fetchSuggestions = async (text: string) => {
    const response = await api.post('/citations/suggest', {
      context: text,
      limit: 5
    });
    setSuggestions(response.data);
  };
  
  const search = async () => {
    setLoading(true);
    const response = await api.post('/citations/search', {
      query,
      fuzzy: true,
      include_related: true
    });
    setResults(response.data);
    setLoading(false);
  };
  
  return (
    <div className="citation-search">
      <div className="search-bar">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search citations..."
          onKeyPress={(e) => e.key === 'Enter' && search()}
        />
        <button onClick={search} disabled={loading}>
          Search
        </button>
      </div>
      
      {suggestions.length > 0 && (
        <div className="suggestions">
          <h4>AI Suggestions</h4>
          {suggestions.map(citation => (
            <CitationCard
              key={citation.id}
              citation={citation}
              onClick={() => onSelect(citation)}
              showRelevance={true}
            />
          ))}
        </div>
      )}
      
      {results.length > 0 && (
        <div className="results">
          <h4>Search Results</h4>
          {results.map(citation => (
            <CitationCard
              key={citation.id}
              citation={citation}
              onClick={() => onSelect(citation)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
```

### Note Organization View

```typescript
// ui/components/NoteOrganizer.tsx
export const NoteOrganizer: React.FC = () => {
  const [notes, setNotes] = useState<Note[]>([]);
  const [view, setView] = useState<'hierarchy' | 'timeline' | 'graph'>('hierarchy');
  const [selectedNote, setSelectedNote] = useState<Note | null>(null);
  
  const renderHierarchy = () => {
    return (
      <TreeView
        data={buildTreeData(notes)}
        onSelect={setSelectedNote}
        onDragEnd={handleReorganize}
        renderNode={(node) => (
          <NoteNode
            note={node}
            onEdit={handleEdit}
            onAddChild={handleAddChild}
          />
        )}
      />
    );
  };
  
  const renderTimeline = () => {
    return (
      <Timeline
        items={notes.map(note => ({
          date: note.createdAt,
          title: note.title,
          content: note.content,
          type: note.noteType,
          onClick: () => setSelectedNote(note)
        }))}
      />
    );
  };
  
  const renderGraph = () => {
    return (
      <ForceGraph
        nodes={notes}
        links={buildNoteLinks(notes)}
        nodeLabel="title"
        onNodeClick={setSelectedNote}
        colorBy="noteType"
      />
    );
  };
  
  return (
    <div className="note-organizer">
      <div className="view-selector">
        <button onClick={() => setView('hierarchy')}>Hierarchy</button>
        <button onClick={() => setView('timeline')}>Timeline</button>
        <button onClick={() => setView('graph')}>Graph</button>
      </div>
      
      <div className="main-view">
        {view === 'hierarchy' && renderHierarchy()}
        {view === 'timeline' && renderTimeline()}
        {view === 'graph' && renderGraph()}
      </div>
      
      {selectedNote && (
        <NoteEditor
          note={selectedNote}
          onSave={handleSave}
          onClose={() => setSelectedNote(null)}
        />
      )}
    </div>
  );
};
```

## Performance Optimizations

### Citation Caching Strategy

```python
# src/services/citation_manager/cache_strategy.py
class CitationCacheStrategy:
    def __init__(self, cache_manager):
        self.cache = cache_manager
        
    async def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get citation with caching"""
        # Try cache first
        cache_key = f"citation:{citation_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            return Citation(**cached)
            
        # Fetch from database
        citation = await self.db_session.get(Citation, citation_id)
        
        if citation:
            # Cache for 1 week (citations rarely change)
            await self.cache.set(
                cache_key,
                citation.to_dict(),
                ttl=timedelta(days=7)
            )
            
        return citation
        
    async def search_citations(
        self,
        query: str,
        filters: Dict = None
    ) -> List[Citation]:
        """Search with result caching"""
        # Generate cache key
        cache_key = self.cache.make_cache_key(
            "citation_search",
            query=query,
            filters=filters
        )
        
        # Try cache
        cached = await self.cache.get(cache_key)
        if cached:
            return [Citation(**c) for c in cached]
            
        # Perform search
        results = await self._perform_search(query, filters)
        
        # Cache results for 1 hour
        await self.cache.set(
            cache_key,
            [r.to_dict() for r in results],
            ttl=timedelta(hours=1)
        )
        
        return results
```

## Best Practices and Guidelines

### Citation Management Best Practices

1. **Deduplication Strategy**
   - Use fuzzy matching for author names
   - Consider publication year variations (±1 year)
   - Check DOI, ISBN, and other unique identifiers
   - Manual verification for high-value citations

2. **Context Preservation**
   - Always store citation context (surrounding text)
   - Maintain argumentative role (support, critique, etc.)
   - Track citation genealogy (who cites whom)
   - Preserve page numbers and editions

3. **Philosophical Specifics**
   - Support standard pagination systems (Bekker, Stephanus)
   - Handle translations and multiple editions
   - Track philosophical traditions and schools
   - Maintain work relationships (commentaries, responses)

### Note-Taking Best Practices

1. **Organization Principles**
   - Use consistent tagging taxonomy
   - Maintain clear parent-child relationships
   - Regular review and reorganization
   - Archive completed research threads

2. **Linking Strategy**
   - Link notes to specific text passages
   - Create concept maps between notes
   - Track argument development over time
   - Build personal knowledge graphs

3. **Collaboration Guidelines**
   - Clear attribution for shared notes
   - Version control for important notes
   - Comment threads for discussions
   - Export formats for sharing

## Conclusion

The Citation and Notes Management System represents a paradigm shift in how philosophical researchers work with sources and develop ideas. By combining traditional scholarly rigor with AI-powered enhancements, the system enables:

- **Comprehensive Citation Networks**: Understanding not just what you've read, but how ideas connect
- **Intelligent Note Organization**: Multiple ways to view and organize thoughts
- **AI-Assisted Research**: Smart suggestions and automated extraction
- **Seamless Integration**: Works with existing tools like Zotero
- **Philosophical Focus**: Features designed specifically for philosophical research

This system transforms the often tedious aspects of research into opportunities for discovery, helping philosophers focus on what matters most: developing and refining ideas that advance human understanding.