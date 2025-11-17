# GraphRAG: Research Paper Knowledge Graph

A GraphRAG system that builds a knowledge graph from research papers with:
- **Vector Search**: Qdrant for semantic search
- **Knowledge Graph**: Neo4j for entity-relationship modeling
- **Multi-hop Reasoning**: Traverse relationships across papers
- **Temporal Reasoning**: Query by publication dates and evolution
- **LLM Response Generation**: Generate natural language answers from retrieved results

## Papers Included

1. **Attention Is All You Need** (2017) - Transformer architecture
2. **BERT** (2018) - Bidirectional Encoder Representations
3. **LLaMA** (2023) - Large Language Model Meta AI
4. **LoRA** (2021) - Low-Rank Adaptation
5. **Flash Attention** (2022) - Efficient attention mechanism

## Architecture

```
PDF Papers
    ↓
Text Extraction & Chunking
    ↓
┌─────────────────┬──────────────────┐
│  Qdrant         │  Neo4j Graph     │
│  (Embeddings)   │  (Entities &     │
│  - Chunks       │   Relationships) │
│  - Semantic     │  - Papers        │
│    Search       │  - Authors       │
└─────────────────┴──────────────────┘
    ↓
GraphRAG Query Engine
    ├─ Multi-hop traversal
    ├─ Temporal filtering
    ├─ Hybrid retrieval
    └─ LLM Response Generation
```

## Setup

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key and Neo4j credentials
```

3. Run the pipeline:
```bash
# Step 1: Ingest documents (extract, chunk, embed)
python scripts/ingest.py

# Step 2: Build knowledge graph (extract entities, build graph)
python scripts/build_graph.py
```

## Usage

### Command Line

```bash
# Basic query with LLM-generated answer
python scripts/query.py "How did attention mechanisms evolve from Transformers to Flash Attention?"

# Query with options
python scripts/query.py "What is LoRA?" --top-k 10 --max-hops 2

# Disable LLM generation (show raw results only)
python scripts/query.py "What papers use Transformer?" --no-generate

# Temporal query
python scripts/query.py "Papers about transformers from 2017 to 2020" --temporal
```

### Python API

```python
from backend.query import GraphRAGQuery

# Initialize with response generation enabled (default)
query_engine = GraphRAGQuery(
    use_graph=True,
    max_hops=2,
    use_generator=True  # Enable LLM answer generation
)

# Query with generated answer
result = query_engine.query(
    "How did attention mechanisms evolve from Transformers to Flash Attention?",
    top_k=5,
    use_temporal=True,
    generate_answer=True
)

# Access generated answer
if result.get("generated_answer"):
    answer = result["generated_answer"]["answer"]
    confidence = result["generated_answer"]["confidence"]
    sources = result["generated_answer"]["sources"]
    print(f"Answer: {answer}")
    print(f"Confidence: {confidence}")
    print(f"Sources: {sources}")

# Access raw results
for res in result["results"]:
    print(f"Text: {res['text']}")
    print(f"Score: {res['score']}")
```

### Response Generation Features

- **Natural Language Answers**: Synthesizes information from multiple retrieved chunks
- **Source Citations**: Automatically extracts and cites source papers
- **Confidence Scores**: Calculates confidence based on result quality
- **Evolution Timeline**: For temporal queries, generates year-by-year summaries
- **Graph Context Integration**: Incorporates knowledge graph relationships into answers

## Web Interface

A modern React + Vite chat-based web interface is available in the `frontend/` folder.

### Quick Start

1. **Install frontend dependencies**:
```bash
cd frontend
npm install
```

2. **Start the API server** (in one terminal):
```bash
python frontend/api_server.py
```

3. **Start the React dev server** (in another terminal):
```bash
cd frontend
npm run dev
```

4. **Open your browser** to `http://localhost:3000` and start chatting!

See `frontend/README.md` for more details.

