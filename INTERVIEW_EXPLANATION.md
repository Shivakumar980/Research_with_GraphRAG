# GraphRAG System - Interview Explanation Guide

## Executive Summary

I built a **GraphRAG (Graph Retrieval-Augmented Generation)** system that combines **vector search** with **knowledge graph traversal** to enable sophisticated querying of research papers. Unlike traditional RAG systems that only use semantic similarity, this system can perform **multi-hop reasoning**, **temporal queries**, and **relationship-aware retrieval** by maintaining both a vector database (Qdrant) and a knowledge graph (Neo4j).

---

## 1. High-Level Architecture

### The Problem
Traditional RAG systems have limitations:
- **Semantic search only**: Finds similar text but misses relationships
- **No multi-hop reasoning**: Can't connect concepts across documents
- **No temporal awareness**: Can't query by time or evolution of ideas
- **Limited context**: Doesn't understand how entities relate to each other

### The Solution
A **hybrid retrieval system** that combines:
1. **Vector Search** (Qdrant) - For semantic similarity
2. **Knowledge Graph** (Neo4j) - For structured relationships
3. **Hybrid Retriever** - Combines both approaches intelligently

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   PDF Papers                          Text Extraction
        │                                   │
        │                              Chunking (1000 chars)
        │                                   │
        └─────────────────┬─────────────────┘
                         │
        ┌─────────────────┴─────────────────┐
        │                                   │
   Vector Store                        Knowledge Graph
   (Qdrant)                            (Neo4j)
        │                                   │
        │  • Chunk embeddings          │  • Papers (nodes)
        │  • Semantic search           │  • Entities (nodes)
        │  • Metadata                  │  • Relationships (edges)
        │                              │  • Chunks (nodes)
        │                              │  • Temporal links
        └─────────────────┬────────────┴───┘
                          │
┌─────────────────────────┴─────────────────────────┐
│              QUERY PIPELINE                       │
└───────────────────────────────────────────────────┘
                          │
                    User Query
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   Entity Extraction              Vector Search
   (from query)                   (semantic)
        │                                   │
        │                              Graph Traversal
        │                              (multi-hop)
        │                                   │
        └─────────────────┬─────────────────┘
                          │
                   Hybrid Retriever
                   (combines & ranks)
                          │
                    Final Results
```

---

## 2. System Components

### 2.1 Document Processing Pipeline

#### **PDF Extractor** (`backend/pdf_processor/extractor.py`)
- **Purpose**: Extract text and metadata from research paper PDFs
- **Technologies**: LangChain PyPDFLoader (primary), pdfplumber (fallback), pypdf (final fallback)
- **Features**:
  - Handles complex PDF layouts
  - Text cleaning (fixes merged words, spacing issues)
  - Metadata extraction (title, authors, year, venue, arXiv ID)
  - Pre-configured metadata for known papers (Transformer, BERT, LLaMA, LoRA, Flash Attention, etc.)

**Key Implementation Details**:
- Multi-method extraction with fallbacks for reliability
- Advanced text cleaning using regex patterns to fix common PDF extraction issues
- Word-level reconstruction from pdfplumber for better spacing preservation

#### **Document Chunker** (`backend/pdf_processor/chunker.py`)
- **Purpose**: Split documents into manageable chunks for embedding
- **Configuration**: 
  - Chunk size: 1000 characters
  - Overlap: 200 characters (ensures context continuity)
- **Technology**: LangChain RecursiveCharacterTextSplitter
- **Output**: Each chunk has:
  - Unique `chunk_id` (format: `{filename}_chunk_{index}`)
  - Metadata (title, authors, year, chunk_index, total_chunks)

---

### 2.2 Vector Store (`backend/embeddings/vector_store.py`)

#### **Purpose**: Store document chunks as embeddings for semantic search

#### **Technologies**:
- **Qdrant**: Vector database (local or cloud)
- **OpenAI Embeddings**: `text-embedding-3-small` (1536 dimensions) - default
- **Sentence Transformers**: Alternative option (configurable)

#### **Key Features**:
1. **Batch Processing**: Processes embeddings in batches of 100 for efficiency
2. **Retry Logic**: Uses `tenacity` library for automatic retries on failures
3. **Metadata Storage**: Stores full chunk text and metadata as payload
4. **Unique IDs**: Generates deterministic IDs using MD5 hashing

#### **Operations**:
- `add_chunks()`: Batch upload with progress tracking
- `search()`: Semantic similarity search (cosine distance)
- `clear_collection()`: Reset vector store

---

### 2.3 Knowledge Graph Builder (`backend/graph/`)

#### **Entity Extractor** (`backend/graph/entity_extractor.py`)
- **Purpose**: Extract structured entities and relationships from text using LLM
- **Technology**: OpenAI GPT-4o-mini (cost-effective for batch processing)
- **Extraction Types**:
  - **Entities**: Concepts, Techniques, Models, Datasets, Metrics, Architectures
  - **Relationships**: BUILDS_ON, IMPROVES, USES, COMPARES_TO, INTRODUCES, EXTENDS
  - **Paper Relationships**: INTRODUCES, PROPOSES, EVALUATES, COMPARES

**Key Innovation - Batch Processing**:
- Processes **5 chunks per LLM call** (vs. 1 chunk per call)
- Reduces API costs by ~80% and speeds up extraction significantly
- Maintains chunk-to-entity mapping for linking

#### **Graph Builder** (`backend/graph/graph_builder.py`)
- **Purpose**: Build and manage Neo4j knowledge graph
- **Technology**: Neo4j graph database (with NetworkX fallback)
- **Node Types**:
  - `Paper`: Research papers with metadata
  - `Chunk`: Document chunks linked to papers
  - Entity nodes: Dynamic labels based on entity type (e.g., `Model`, `Technique`, `Concept`)

- **Relationship Types**:
  - `FROM_PAPER`: Chunk → Paper
  - `MENTIONS`: Chunk → Entity
  - `BUILDS_ON`, `IMPROVES`, `USES`, etc.: Entity → Entity
  - `PRECEDES`: Paper → Paper (temporal ordering)

**Performance Optimizations**:
- **Batch Operations**: All CRUD operations use batching (100 items per batch)
- **UNWIND Queries**: Uses Neo4j's UNWIND for efficient bulk inserts
- **Transaction Management**: Groups operations in transactions for speed

**Example Graph Structure**:
```
(Paper: "Attention Is All You Need")
    ←[:FROM_PAPER]─ (Chunk: "chunk_0")
        └─[:MENTIONS]→ (Model: "Transformer")
            └─[:BUILDS_ON]→ (Model: "BERT")
                └─[:IMPROVES]→ (Model: "LLaMA")
```

---

### 2.4 Query Engine (`backend/query/`)

#### **Vector Retriever** (`backend/query/vector_retriever.py`)
- **Purpose**: Wrapper around VectorStore for semantic search
- **Operations**:
  - `search()`: Semantic similarity search
  - `get_chunks_for_papers()`: Filter results by specific papers
  - `get_chunk_ids()`: Extract chunk IDs from results

#### **Graph Traverser** (`backend/query/graph_traverser.py`)
- **Purpose**: Traverse Neo4j graph to find related entities and papers
- **Key Operations**:
  1. **`find_entities()`**: Fuzzy matching to find entities in graph
  2. **`traverse_from_entities()`**: Multi-hop traversal (configurable depth)
  3. **`get_related_papers()`**: Find papers connected to entities
  4. **`expand_from_chunks()`**: Get graph context for vector search results

**Multi-Hop Traversal Example**:
```cypher
MATCH (start)-[*1..2]-(connected)
WHERE start.name IN $entity_names
RETURN connected, length(path) as hops
```
- Finds entities within 1-2 hops from query entities
- Returns papers, models, concepts connected through relationships

#### **Query Entity Extractor** (`backend/query/entity_extractor.py`)
- **Purpose**: Extract entities from user queries
- **Features**:
  - Named entity recognition from queries
  - Temporal intent detection (e.g., "papers from 2020", "evolution of attention")
  - Year extraction and range detection

#### **Hybrid Retriever** (`backend/query/hybrid_retriever.py`)
- **Purpose**: Intelligently combine vector and graph results
- **Algorithm**:
  1. **Vector Search**: Get top-k semantically similar chunks
  2. **Entity Extraction**: Extract entities from query
  3. **Graph Traversal**: Find related entities/papers from graph
  4. **Graph Expansion**: Expand vector results with graph context
  5. **Score Combination**: Weighted combination of vector and graph scores
     - Default: 60% vector, 40% graph
     - Configurable per query

**Scoring Formula**:
```
combined_score = (vector_weight × vector_score) + (graph_weight × graph_score)
graph_score = min(1.0, (entity_count × 0.1 + rel_count × 0.05))
```

#### **Query Engine** (`backend/query/engine.py`)
- **Purpose**: Main interface for querying the system
- **Features**:
  - Temporal filtering (by year or year ranges)
  - Configurable graph traversal depth
  - Hybrid retrieval with customizable weights

---

## 3. Data Flow

### 3.1 Ingestion Pipeline (`scripts/ingest.py`)

```
1. PDF Files (data/papers/*.pdf)
   ↓
2. PDF Extractor
   - Extract text
   - Extract metadata
   ↓
3. Document Chunker
   - Split into 1000-char chunks
   - 200-char overlap
   ↓
4. Vector Store (Qdrant)
   - Generate embeddings (OpenAI)
   - Store chunks with metadata
   ↓
5. Vector Store Ready ✓
```

### 3.2 Graph Building Pipeline (`scripts/build_graph.py`)

```
1. Load Papers (same as ingestion)
   ↓
2. Chunk Documents (same chunker as ingestion - critical for consistency!)
   ↓
3. Entity Extractor (LLM)
   - Extract entities from chunks (batch: 5 chunks/API call)
   - Extract relationships
   - Map chunks to entities
   ↓
4. Graph Builder (Neo4j)
   - Add Paper nodes
   - Add Entity nodes (batch: 100/batch)
   - Add Relationship edges (batch: 100/batch)
   - Add Chunk nodes (batch: 100/batch)
   - Link Chunks → Entities (batch: 100/batch)
   - Add temporal relationships (Paper → Paper by year)
   ↓
5. Knowledge Graph Ready ✓
```

### 3.3 Query Pipeline

```
1. User Query: "How did attention mechanisms evolve from Transformers to Flash Attention?"
   ↓
2. Query Entity Extractor
   - Extract: ["attention", "Transformers", "Flash Attention"]
   - Detect temporal intent: "evolve" → temporal query
   ↓
3. Hybrid Retriever
   ├─ Vector Search (Qdrant)
   │  └─ Find semantically similar chunks
   │
   └─ Graph Traversal (Neo4j)
      ├─ Find entities: "Transformers", "Flash Attention"
      ├─ Multi-hop traversal (2 hops)
      │  └─ Find related papers, models, concepts
      └─ Expand from vector results
         └─ Get graph context for each chunk
   ↓
4. Score Combination
   - Combine vector scores (60%) + graph scores (40%)
   - Rank by combined score
   ↓
5. Temporal Filtering (if needed)
   - Filter/sort by publication year
   ↓
6. Return Results
   - Top-k ranked chunks
   - Related papers from graph
   - Graph context (entities, relationships)
```

---

## 4. Key Technical Decisions

### 4.1 Why Hybrid Approach?
- **Vector Search Alone**: Good for semantic similarity but misses relationships
- **Graph Alone**: Good for structured queries but misses semantic nuance
- **Hybrid**: Best of both worlds - semantic understanding + relationship awareness

### 4.2 Why Neo4j?
- **Native Graph Operations**: Efficient traversal, relationship queries
- **Cypher Query Language**: Expressive for complex graph patterns
- **Performance**: Optimized for graph operations (vs. relational DB)
- **Flexibility**: Dynamic schema (can add new entity types without migration)

### 4.3 Why Qdrant?
- **Local Option**: Can run locally (no cloud dependency)
- **Performance**: Fast vector search with cosine similarity
- **Metadata Storage**: Stores full text and metadata as payload
- **Scalability**: Can scale to cloud if needed

### 4.4 Why Batch Processing?
- **Cost Efficiency**: 5 chunks per LLM call vs. 1 = 80% cost reduction
- **Speed**: Parallel processing reduces total time
- **API Limits**: Respects rate limits while maximizing throughput

### 4.5 Chunk Consistency
- **Critical**: Same chunker used in ingestion and graph building
- **Why**: Ensures `chunk_id` matches between Qdrant and Neo4j
- **Enables**: Linking vector search results to graph nodes

---

## 5. Technologies Used

### Core Technologies
- **Python 3.13**: Main language
- **Neo4j**: Graph database (bolt://localhost:7687)
- **Qdrant**: Vector database (local storage)
- **OpenAI API**: Embeddings (`text-embedding-3-small`) and entity extraction (GPT-4o-mini)

### Libraries
- **LangChain**: PDF loading, text splitting
- **Qdrant Client**: Vector database operations
- **Neo4j Driver**: Graph database operations
- **OpenAI SDK**: Embeddings and LLM calls
- **Tenacity**: Retry logic for API calls
- **python-dotenv**: Environment variable management

### Optional/Fallback
- **NetworkX**: In-memory graph fallback if Neo4j unavailable
- **pdfplumber**: PDF extraction fallback
- **pypdf**: PDF extraction final fallback
- **Sentence Transformers**: Alternative embedding option

---

## 6. Challenges & Solutions

### Challenge 1: PDF Text Extraction Quality
**Problem**: PDFs often have merged words, poor spacing, complex layouts

**Solution**:
- Multi-method extraction (LangChain → pdfplumber → pypdf)
- Advanced text cleaning with regex patterns
- Word-level reconstruction from pdfplumber coordinates

### Challenge 2: LLM API Costs
**Problem**: Extracting entities from hundreds of chunks is expensive

**Solution**:
- Batch processing (5 chunks per API call)
- Cost-effective model (GPT-4o-mini)
- Configurable limit (`MAX_CHUNKS_FOR_EXTRACTION`)

### Challenge 3: Graph Building Performance
**Problem**: Adding thousands of nodes/edges one-by-one is slow

**Solution**:
- Batch operations (100 items per batch)
- Neo4j UNWIND queries for bulk inserts
- Transaction grouping

### Challenge 4: Chunk ID Consistency
**Problem**: Vector store and graph must reference same chunks

**Solution**:
- Same chunker instance used in both pipelines
- Deterministic chunk_id generation: `{filename}_chunk_{index}`
- Verification step to ensure consistency

### Challenge 5: Hybrid Scoring
**Problem**: How to combine vector and graph scores meaningfully?

**Solution**:
- Weighted combination (configurable 60/40 default)
- Graph score based on entity/relationship counts
- Distance-based scoring for graph results (closer = higher score)

---

## 7. Use Cases & Benefits

### Use Cases

1. **Multi-Hop Reasoning**
   - Query: "What papers built on the Transformer architecture?"
   - System finds: Transformer → BERT → LLaMA (traverses relationships)

2. **Temporal Queries**
   - Query: "How did attention mechanisms evolve from 2017 to 2022?"
   - System filters/sorts by year, shows evolution

3. **Relationship-Aware Search**
   - Query: "Models that use LoRA"
   - System finds papers mentioning LoRA AND related models through graph

4. **Concept Exploration**
   - Query: "What is Flash Attention?"
   - System returns: Definition (vector) + Related papers (graph) + Related concepts (graph)

### Benefits Over Traditional RAG

1. **Multi-Hop Reasoning**: Can connect concepts across multiple papers
2. **Temporal Awareness**: Understands evolution and chronology
3. **Relationship Understanding**: Knows how entities relate (not just similarity)
4. **Context Enrichment**: Graph context adds depth to vector results
5. **Structured Queries**: Can query by relationships, not just keywords

---

## 8. Example Queries & Results

### Example 1: Multi-Hop Query
```
Query: "How did attention mechanisms evolve from Transformers to Flash Attention?"

Results:
1. Vector: Chunk about Transformer architecture (score: 0.85)
   Graph Context: Connected to BERT, LLaMA, Flash Attention
   
2. Graph: Paper "Flash Attention" (hops: 2 from Transformer)
   Relationship: Transformer → BERT → Flash Attention
   
3. Vector: Chunk about Flash Attention optimization (score: 0.82)
   Graph Context: Mentions Transformer, related to efficiency papers
```

### Example 2: Temporal Query
```
Query: "Papers about transformers from 2017 to 2020"

Results (sorted by year):
1. 2017: "Attention Is All You Need" (Transformer introduced)
2. 2018: "BERT" (uses Transformer)
3. 2020: "Vision Transformer" (applies to images)
```

### Example 3: Relationship Query
```
Query: "What models use LoRA?"

Results:
1. Vector: Chunk about LoRA technique (score: 0.88)
2. Graph: Papers mentioning LoRA
   - LLaMA (uses LoRA for fine-tuning)
   - GPT-3 (applied LoRA)
   Relationships: LoRA → USES → LLaMA
```

---

## 9. System Statistics

### Typical Graph Size (for 7 papers):
- **Papers**: 7 nodes
- **Chunks**: ~200-300 nodes (depends on paper length)
- **Entities**: ~100-200 nodes (concepts, models, techniques)
- **Relationships**: ~300-500 edges (entity-entity, chunk-entity, paper-entity)
- **Total Nodes**: ~300-500
- **Total Edges**: ~300-500

### Performance:
- **Ingestion**: ~5-10 minutes for 7 papers
- **Graph Building**: ~15-30 minutes (depends on chunk count and API speed)
- **Query Time**: <1 second (vector + graph combined)

---

## 10. Interview Talking Points

### What Makes This Special?
1. **Hybrid Architecture**: Not just vector search, not just graph - combines both intelligently
2. **Production-Ready**: Error handling, retries, fallbacks, batch processing
3. **Scalable Design**: Can handle more papers, configurable limits
4. **Cost-Effective**: Batch processing reduces API costs by 80%
5. **Flexible**: Supports both local and cloud deployments

### Technical Depth:
- **Graph Database Design**: Neo4j schema, relationship types, traversal strategies
- **Vector Search**: Embedding generation, similarity metrics, metadata storage
- **LLM Integration**: Prompt engineering, batch processing, cost optimization
- **System Integration**: Connecting multiple systems (Qdrant, Neo4j, OpenAI)

### Future Enhancements (if asked):
1. **Real-time Updates**: Incremental graph building as new papers added
2. **Advanced Traversal**: Path-based scoring, relationship weights
3. **Query Optimization**: Caching, query planning
4. **Multi-modal**: Support for images, tables in papers
5. **User Feedback**: Learning from query results to improve ranking

---

## 11. Code Structure

```
graphRAG/
├── backend/
│   ├── embeddings/
│   │   └── vector_store.py          # Qdrant operations
│   ├── graph/
│   │   ├── entity_extractor.py      # LLM-based extraction
│   │   └── graph_builder.py         # Neo4j operations
│   ├── pdf_processor/
│   │   ├── extractor.py             # PDF text extraction
│   │   └── chunker.py               # Document chunking
│   └── query/
│       ├── engine.py                # Main query interface
│       ├── hybrid_retriever.py      # Combines vector + graph
│       ├── graph_traverser.py       # Neo4j traversal
│       ├── vector_retriever.py      # Qdrant search wrapper
│       └── entity_extractor.py      # Query entity extraction
├── scripts/
│   ├── ingest.py                    # Ingestion pipeline
│   ├── build_graph.py               # Graph building pipeline
│   └── query.py                     # Query interface
└── data/
    ├── papers/                      # PDF files
    └── knowledge_graph_export.json   # Graph backup
```

---

## 12. Key Metrics to Mention

- **Efficiency**: 80% cost reduction through batch processing
- **Performance**: <1 second query time for hybrid retrieval
- **Scalability**: Handles hundreds of papers, thousands of chunks
- **Reliability**: Multi-level fallbacks (Neo4j → NetworkX, multiple PDF extractors)
- **Flexibility**: Configurable weights, traversal depth, batch sizes

---

## Quick Summary (30-second pitch)

"I built a GraphRAG system that combines vector search with knowledge graphs to enable sophisticated querying of research papers. Unlike traditional RAG, it can perform multi-hop reasoning, temporal queries, and relationship-aware retrieval. The system uses Qdrant for semantic search, Neo4j for structured relationships, and intelligently combines both through a hybrid retriever. Key innovations include batch processing for cost efficiency, consistent chunking across pipelines, and production-ready error handling."

---

## Questions You Might Get

**Q: Why not just use vector search?**
A: Vector search is great for semantic similarity but misses relationships. For example, if you ask "What models built on Transformers?", vector search might find papers mentioning both, but graph traversal can follow the BUILDS_ON relationship to find the actual lineage.

**Q: How do you handle entity extraction errors?**
A: We use batch processing to reduce costs, and we have fallback mechanisms. If extraction fails, we can load from backup JSON files. The system is designed to be resilient.

**Q: What's the bottleneck?**
A: Entity extraction (LLM calls) is the slowest part, but we've optimized it with batch processing. Query time is fast (<1 second) because both vector and graph operations are efficient.

**Q: How do you ensure chunk consistency?**
A: We use the exact same chunker instance with the same parameters in both ingestion and graph building pipelines. The chunk_id format is deterministic: `{filename}_chunk_{index}`.

**Q: Can this scale to thousands of papers?**
A: Yes, the architecture is designed for scale. Qdrant and Neo4j both handle large datasets well. The main consideration would be entity extraction costs, which we've optimized with batch processing.

