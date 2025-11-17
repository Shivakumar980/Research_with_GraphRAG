# GraphRAG Query Engine Implementation Plan

## Current State Analysis

### ✅ What Exists
1. **Vector Store** (`backend/embeddings/vector_store.py`)
   - Qdrant integration
   - Embedding generation (OpenAI/sentence-transformers)
   - Basic vector search: `search(query, top_k)`
   - Returns: chunks with text, metadata, similarity score

2. **Knowledge Graph** (`backend/graph/graph_builder.py`)
   - Neo4j integration
   - Paper nodes, Entity nodes, Relationships
   - Graph operations: add_paper, add_entity, add_relationship
   - Graph info retrieval

3. **Basic Query Script** (`scripts/query.py`)
   - Vector search only
   - Simple CLI interface
   - Returns top-k similar chunks

### ❌ What's Missing
1. **Query Engine** (`backend/query/` - empty directory)
   - No GraphRAGQuery class
   - No graph traversal logic
   - No entity extraction from queries
   - No hybrid retrieval
   - No temporal filtering

2. **Graph Traversal**
   - No multi-hop relationship traversal
   - No entity-to-paper connections
   - No path finding

3. **Query Processing**
   - No query entity extraction
   - No query intent detection
   - No temporal query detection

---

## Implementation Plan

### Phase 1: Foundation - Graph Traversal Module

**Goal:** Build basic graph traversal capabilities

**Components to Create:**

#### 1.1 `backend/query/graph_traverser.py`
**Purpose:** Traverse Neo4j graph to find related entities and papers

**Key Methods:**
- `find_entities(query_entities: List[str])` - Find entities in graph matching query
- `traverse_from_entities(entities: List[str], max_hops: int = 2)` - Multi-hop traversal
- `get_related_papers(entities: List[str])` - Find papers connected to entities
- `get_entity_relationships(entity: str, max_hops: int)` - Get relationships from entity

**Cypher Queries Needed:**
- Find entities by name (fuzzy matching)
- Traverse relationships (1-hop, 2-hop, 3-hop)
- Find papers connected to entities
- Get relationship paths between entities

**Dependencies:**
- Neo4j driver (already in graph_builder.py)
- Reuse connection logic from GraphBuilder

---

#### 1.2 `backend/query/entity_extractor.py` (Query-side)
**Purpose:** Extract entities from user queries (different from document entity extraction)

**Key Methods:**
- `extract_entities_from_query(query: str)` - Extract entity names from query text
- `match_entities_in_graph(query_entities: List[str])` - Match query entities to graph entities
- `detect_temporal_intent(query: str)` - Detect if query is time-based

**Approach:**
- Option A: Simple keyword matching (fast, limited)
- Option B: LLM-based extraction (accurate, slower, costs)
- Option C: Hybrid (keyword + LLM for ambiguous cases)

**Recommendation:** Start with Option A, add Option B later

---

### Phase 2: Hybrid Retrieval - Combine Vector + Graph

**Goal:** Merge vector search results with graph traversal results

#### 2.1 `backend/query/hybrid_retriever.py`
**Purpose:** Combine and rank results from vector search and graph traversal

**Key Methods:**
- `retrieve(query: str, top_k: int, use_graph: bool, max_hops: int)` - Main retrieval
- `combine_results(vector_results, graph_results)` - Merge results
- `rank_results(combined_results)` - Score and rank
- `deduplicate(results)` - Remove duplicates

**Ranking Strategy:**
- Vector similarity score (0-1)
- Graph distance score (closer = higher)
- Temporal relevance (if time-based query)
- Combined weighted score: `final_score = α * vector_score + β * graph_score + γ * temporal_score`

**Default Weights:**
- α = 0.6 (vector search)
- β = 0.3 (graph traversal)
- γ = 0.1 (temporal relevance)

---

#### 2.2 `backend/query/vector_retriever.py`
**Purpose:** Wrapper around VectorStore for query engine

**Key Methods:**
- `search(query: str, top_k: int)` - Vector search
- `search_by_papers(papers: List[str], query: str)` - Search within specific papers
- `get_chunks_for_papers(papers: List[str])` - Get chunks for graph-found papers

**Note:** Can mostly wrap existing VectorStore.search()

---

### Phase 3: Main Query Engine

**Goal:** Unified GraphRAG query interface

#### 3.1 `backend/query/engine.py`
**Purpose:** Main GraphRAG query engine class

**Key Methods:**
- `query(query: str, top_k: int = 5, use_temporal: bool = False, max_hops: int = 2)` - Main query method
- `_process_query(query: str)` - Parse and extract entities
- `_vector_search(query: str, top_k: int)` - Get vector results
- `_graph_traversal(query_entities: List[str], max_hops: int)` - Get graph results
- `_temporal_filter(results, query: str)` - Apply temporal filtering
- `_combine_and_rank(vector_results, graph_results)` - Merge results

**Return Format:**
```python
{
    "query": str,
    "results": [
        {
            "text": str,
            "metadata": dict,
            "score": float,
            "source": "vector" | "graph" | "hybrid",
            "graph_context": {
                "entities": List[str],
                "relationships": List[dict],
                "path": List[str]
            }
        }
    ],
    "graph_entities": List[str],  # Entities found in query
    "temporal_info": dict  # If temporal query
}
```

---

### Phase 4: Temporal Reasoning

**Goal:** Add time-based filtering and ordering

#### 4.1 Temporal Query Detection
**In:** `backend/query/entity_extractor.py`

**Detect:**
- Time keywords: "evolution", "before", "after", "since", "until"
- Year mentions: "2017", "2020s", "recent"
- Temporal relationships: "evolved", "developed", "improved over time"

#### 4.2 Temporal Filtering
**In:** `backend/query/engine.py`

**Features:**
- Filter papers by year range
- Order results chronologically
- Show evolution timeline
- Group by time periods

**Cypher Queries:**
- Papers before/after year
- Papers in year range
- Chronological ordering
- Evolution paths (PRECEDES relationships)

---

### Phase 5: Enhanced Query Script

**Goal:** Update CLI to use GraphRAG engine

#### 5.1 Update `scripts/query.py`
**Changes:**
- Replace VectorStore with GraphRAGQuery
- Add command-line flags:
  - `--use-graph` (enable graph traversal)
  - `--max-hops N` (graph traversal depth)
  - `--temporal` (enable temporal filtering)
- Better output formatting:
  - Show vector results
  - Show graph results
  - Show combined results
  - Show graph context (entities, relationships)

---

## Implementation Phases Summary

### Phase 1: Foundation (Week 1)
- [ ] Create `backend/query/graph_traverser.py`
- [ ] Create `backend/query/entity_extractor.py` (query-side)
- [ ] Test graph traversal with sample queries
- [ ] Test entity extraction from queries

**Deliverable:** Can traverse graph and extract entities from queries

---

### Phase 2: Hybrid Retrieval (Week 1-2)
- [ ] Create `backend/query/vector_retriever.py`
- [ ] Create `backend/query/hybrid_retriever.py`
- [ ] Implement result combination logic
- [ ] Implement ranking algorithm
- [ ] Test with sample queries

**Deliverable:** Can combine vector + graph results

---

### Phase 3: Main Engine (Week 2)
- [ ] Create `backend/query/engine.py`
- [ ] Implement main query() method
- [ ] Integrate all components
- [ ] Add error handling
- [ ] Test end-to-end queries

**Deliverable:** Working GraphRAG query engine

---

### Phase 4: Temporal Reasoning (Week 2-3)
- [ ] Add temporal query detection
- [ ] Implement temporal filtering
- [ ] Add chronological ordering
- [ ] Test temporal queries

**Deliverable:** Time-aware queries

---

### Phase 5: CLI Enhancement (Week 3)
- [ ] Update `scripts/query.py`
- [ ] Add command-line flags
- [ ] Improve output formatting
- [ ] Add examples and help text

**Deliverable:** User-friendly CLI

---

## Technical Decisions

### 1. Entity Extraction from Queries

**Option A: Keyword Matching (Simple)**
- Pros: Fast, no API costs, works offline
- Cons: Limited accuracy, misses synonyms
- Implementation: Extract capitalized words, technical terms, known entity names

**Option B: LLM Extraction (Accurate)**
- Pros: High accuracy, understands context
- Cons: Slower, API costs, requires OpenAI key
- Implementation: Use GPT-4o-mini (same as document extraction)

**Recommendation:** Start with Option A, add Option B as enhancement

---

### 2. Graph Traversal Depth

**Default:** 2 hops
- 1-hop: Direct relationships
- 2-hop: Indirect relationships
- 3+ hops: Usually too noisy

**Configurable:** Allow user to set max_hops (1-3)

---

### 3. Result Ranking

**Simple Approach:**
- Vector score (0-1)
- Graph distance (inverse of hops: 1/hops)
- Combined: `score = 0.6 * vector_score + 0.4 * (1/hops)`

**Advanced Approach:**
- Weighted combination
- Temporal relevance
- Entity importance (centrality)
- Paper citation count (if available)

**Recommendation:** Start simple, enhance later

---

### 4. Temporal Filtering

**Automatic Detection:**
- Detect temporal keywords in query
- Extract year mentions
- Enable temporal mode automatically

**Manual Override:**
- Allow user to force temporal mode
- Allow year range specification

---

## Example Query Flows

### Query 1: "What is Transformer?"
**Flow:**
1. Vector search → finds chunks about Transformer
2. Entity extraction → "Transformer"
3. Graph traversal → finds Transformer node, related papers
4. Combine → merge vector chunks + graph context
5. Return → chunks + papers that mention Transformer

---

### Query 2: "How did attention evolve from 2017 to 2022?"
**Flow:**
1. Vector search → finds chunks about attention
2. Entity extraction → "attention", detect temporal intent
3. Graph traversal → find attention-related entities
4. Temporal filter → filter papers 2017-2022, order chronologically
5. Graph traversal → follow PRECEDES relationships
6. Combine → merge with temporal ordering
7. Return → evolution timeline with chunks

---

### Query 3: "What papers built on BERT?"
**Flow:**
1. Vector search → finds chunks about BERT
2. Entity extraction → "BERT"
3. Graph traversal → find BERT node, traverse BUILDS_ON relationships (reverse)
4. Find papers → papers that have BUILDS_ON → BERT
5. Combine → merge vector chunks + graph-found papers
6. Return → papers that built on BERT + relevant chunks

---

## Testing Strategy

### Unit Tests
- Entity extraction from queries
- Graph traversal logic
- Result combination
- Ranking algorithm

### Integration Tests
- End-to-end query flow
- Vector + graph integration
- Temporal filtering
- Error handling

### Example Queries to Test
1. Simple entity query: "What is LoRA?"
2. Relationship query: "What papers use Transformer?"
3. Temporal query: "How did attention evolve?"
4. Multi-entity query: "Compare BERT and LLaMA"
5. Complex query: "What improvements were made to attention mechanisms after 2020?"

---

## Dependencies

### New Dependencies (if needed)
- None! All required libraries already in requirements.txt:
  - `neo4j` - Graph database
  - `qdrant-client` - Vector store
  - `openai` - Optional for LLM entity extraction

### Existing Components to Reuse
- `VectorStore` - Vector search
- `GraphBuilder` - Neo4j connection logic
- `EntityExtractor` - Can adapt for query-side extraction

---

## Success Metrics

### Phase 1 Success
- Can extract entities from queries
- Can traverse graph from entities
- Returns related papers and entities

### Phase 2 Success
- Combines vector + graph results
- Ranks results appropriately
- No duplicate results

### Phase 3 Success
- End-to-end queries work
- Returns structured results
- Handles errors gracefully

### Phase 4 Success
- Detects temporal queries
- Filters by time correctly
- Shows evolution timeline

### Phase 5 Success
- CLI is user-friendly
- Output is clear and informative
- Examples work correctly

---

## Next Steps

1. **Review this plan**
2. **Decide on entity extraction approach** (keyword vs LLM)
3. **Start Phase 1** - Graph traversal module
4. **Iterate and test** each phase
5. **Gather feedback** and refine

---

## Questions to Resolve

1. **Entity Extraction:** Keyword matching or LLM? (Recommendation: Start with keyword)
2. **Default Weights:** Vector vs Graph? (Recommendation: 60/40)
3. **Max Hops:** Default depth? (Recommendation: 2 hops)
4. **Response Format:** Structured data or natural language? (Recommendation: Structured first)
5. **Temporal:** Automatic detection or manual flag? (Recommendation: Automatic with manual override)

---

**Status:** 📋 Plan Ready for Review

**Estimated Timeline:** 2-3 weeks for full implementation

**Priority:** Phase 1-3 are essential, Phase 4-5 are enhancements

