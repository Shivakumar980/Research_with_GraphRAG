# GraphRAG System Architecture

## System Overview

This document provides a comprehensive architecture diagram of the GraphRAG system, showing data flow, components, and interactions.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React + Vite UI]
        Chat[Chat Interface]
    end
    
    subgraph "API Layer"
        API[Flask API Server]
    end
    
    subgraph "Query Engine"
        QE[GraphRAG Query Engine]
        ER[Entity Extractor]
        GT[Graph Traverser]
        HR[Hybrid Retriever]
        RG[Response Generator]
        VR[Vector Retriever]
    end
    
    subgraph "Data Storage"
        QD[Qdrant<br/>Vector Store]
        NEO[Neo4j<br/>Knowledge Graph]
    end
    
    subgraph "External Services"
        LLM[OpenAI GPT<br/>Embeddings & Generation]
    end
    
    subgraph "Ingestion Pipeline"
        PDF[PDF Papers]
        EXT[PDF Extractor]
        CHK[Document Chunker]
        ENT[Entity Extractor]
        GB[Graph Builder]
    end
    
    UI --> Chat
    Chat --> API
    API --> QE
    QE --> ER
    QE --> GT
    QE --> HR
    QE --> RG
    HR --> VR
    HR --> GT
    VR --> QD
    GT --> NEO
    RG --> LLM
    VR --> LLM
    
    PDF --> EXT
    EXT --> CHK
    CHK --> QD
    CHK --> ENT
    ENT --> GB
    GB --> NEO
    ENT --> LLM
    CHK --> LLM
    
    style UI fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
    style API fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
    style QE fill:#2C2C2C,stroke:#D97757,stroke-width:2px,color:#e0e0e0
    style QD fill:#4A90E2,stroke:#fff,stroke-width:2px,color:#fff
    style NEO fill:#008CC1,stroke:#fff,stroke-width:2px,color:#fff
    style LLM fill:#10A37F,stroke:#fff,stroke-width:2px,color:#fff
```


## Data Flow: Ingestion Pipeline

```mermaid
sequenceDiagram
    participant User
    participant IngestScript
    participant PDFExtractor
    participant Chunker
    participant Embedder
    participant Qdrant
    participant EntityExtractor
    participant GraphBuilder
    participant Neo4j
    
    User->>IngestScript: Run ingest.py
    IngestScript->>PDFExtractor: Extract text from PDFs
    PDFExtractor-->>IngestScript: Raw text + metadata
    IngestScript->>Chunker: Split into chunks
    Chunker-->>IngestScript: Chunks (1000 chars, 200 overlap)
    IngestScript->>Embedder: Generate embeddings
    Embedder->>OpenAI: API call
    OpenAI-->>Embedder: Embeddings
    Embedder-->>IngestScript: Embeddings
    IngestScript->>Qdrant: Store chunks + embeddings
    Qdrant-->>IngestScript: ✓ Stored
    
    User->>IngestScript: Run build_graph.py
    IngestScript->>EntityExtractor: Extract entities from chunks
    EntityExtractor->>OpenAI: LLM extraction
    OpenAI-->>EntityExtractor: Entities + relationships
    EntityExtractor-->>IngestScript: Structured data
    IngestScript->>GraphBuilder: Build knowledge graph
    GraphBuilder->>Neo4j: Create nodes & relationships
    Neo4j-->>GraphBuilder: ✓ Graph built
```

## Data Flow: Query Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant QueryEngine
    participant EntityExtractor
    participant GraphTraverser
    participant VectorRetriever
    participant HybridRetriever
    participant ResponseGenerator
    participant Qdrant
    participant Neo4j
    participant OpenAI
    
    User->>Frontend: Enter query
    Frontend->>API: POST /api/query
    API->>QueryEngine: query(query_string)
    
    QueryEngine->>EntityExtractor: Extract entities
    EntityExtractor->>OpenAI: LLM entity extraction
    OpenAI-->>EntityExtractor: Entities + temporal intent
    EntityExtractor-->>QueryEngine: Extracted entities
    
    QueryEngine->>VectorRetriever: Search similar chunks
    VectorRetriever->>Qdrant: Vector search
    Qdrant-->>VectorRetriever: Top-k chunks
    VectorRetriever-->>QueryEngine: Vector results
    
    QueryEngine->>GraphTraverser: Traverse from entities
    GraphTraverser->>Neo4j: Multi-hop traversal
    Neo4j-->>GraphTraverser: Related entities & papers
    GraphTraverser-->>QueryEngine: Graph results
    
    QueryEngine->>HybridRetriever: Combine results
    HybridRetriever->>HybridRetriever: Weighted scoring
    HybridRetriever-->>QueryEngine: Combined results
    
    QueryEngine->>ResponseGenerator: Generate answer
    ResponseGenerator->>OpenAI: Generate from context
    OpenAI-->>ResponseGenerator: Natural language answer
    ResponseGenerator-->>QueryEngine: Answer + sources
    
    QueryEngine-->>API: Results + answer
    API-->>Frontend: JSON response
    Frontend-->>User: Display answer + sources
```

## Component Details

### 1. Frontend Layer

```mermaid
graph TD
    A[React App] --> B[Header Component]
    A --> C[Chat Container]
    C --> D[ChatMessage Component]
    C --> E[ChatInput Component]
    E --> F[Query Input]
    E --> G[Options: Temporal, Max Hops]
    E --> H[Send Button]
    
    style A fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
    style C fill:#2C2C2C,stroke:#D97757,stroke-width:2px,color:#e0e0e0
```

### 2. Query Engine Architecture

```mermaid
graph TD
    A[GraphRAGQuery] --> B{Use Graph?}
    A --> C[QueryEntityExtractor]
    A --> D[HybridRetriever]
    A --> E[ResponseGenerator]
    
    C --> F[Extract Entities]
    C --> G[Detect Temporal Intent]
    C --> H[Match to Graph Entities]
    
    D --> I[VectorRetriever]
    D --> J[GraphTraverser]
    D --> K[Combine Results]
    
    I --> L[Qdrant Search]
    J --> M[Neo4j Traversal]
    
    E --> N[Generate Answer]
    E --> O[Extract Sources]
    E --> P[Calculate Confidence]
    
    style A fill:#2C2C2C,stroke:#D97757,stroke-width:3px,color:#e0e0e0
    style D fill:#2C2C2C,stroke:#D97757,stroke-width:2px,color:#e0e0e0
```

### 3. Hybrid Retrieval Process

```mermaid
graph LR
    A[User Query] --> B[Entity Extraction]
    A --> C[Vector Search]
    B --> D[Graph Traversal]
    C --> E[Top-k Chunks]
    D --> F[Related Entities]
    D --> G[Connected Papers]
    E --> H[Hybrid Combiner]
    F --> H
    G --> H
    H --> I[Weighted Scoring]
    I --> J[Final Results]
    
    style H fill:#2C2C2C,stroke:#D97757,stroke-width:2px,color:#e0e0e0
    style I fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
```

## Technology Stack

```mermaid
graph TB
    subgraph "Frontend"
        F1[React 18]
        F2[Vite]
        F3[CSS3]
    end
    
    subgraph "Backend"
        B1[Python 3.9+]
        B2[Flask]
        B3[Flask-CORS]
    end
    
    subgraph "Query Engine"
        Q1[Custom GraphRAG Engine]
        Q2[Entity Extraction]
        Q3[Graph Traversal]
        Q4[Hybrid Retrieval]
    end
    
    subgraph "Storage"
        S1[Qdrant<br/>Vector Database]
        S2[Neo4j<br/>Graph Database]
    end
    
    subgraph "AI/ML"
        A1[OpenAI API<br/>GPT-4o-mini]
        A2[OpenAI Embeddings<br/>text-embedding-3-small]
    end
    
    subgraph "Infrastructure"
        I1[Docker]
        I2[Docker Compose]
    end
    
    style F1 fill:#61DAFB,stroke:#fff,stroke-width:2px,color:#000
    style B2 fill:#000,stroke:#fff,stroke-width:2px,color:#fff
    style S1 fill:#4A90E2,stroke:#fff,stroke-width:2px,color:#fff
    style S2 fill:#008CC1,stroke:#fff,stroke-width:2px,color:#fff
    style A1 fill:#10A37F,stroke:#fff,stroke-width:2px,color:#fff
```

## Data Models

### Knowledge Graph Schema

```mermaid
erDiagram
    Paper ||--o{ AUTHORED_BY : has
    Paper ||--o{ CONTAINS : contains
    Paper ||--o{ CITES : cites
    Paper ||--o{ RELATED_TO : relates
    
    Entity ||--o{ CONTAINS : "contained in"
    Entity ||--o{ RELATED_TO : relates
    Entity ||--o{ EVOLVED_FROM : evolves
    
    Paper {
        string id
        string title
        int year
        string authors
        string abstract
    }
    
    Entity {
        string id
        string name
        string type
        string description
    }
    
    AUTHORED_BY {
        string role
    }
    
    CONTAINS {
        float relevance
    }
    
    CITES {
        string context
    }
    
    RELATED_TO {
        string relationship_type
        float strength
    }
```

### Vector Store Schema

```mermaid
classDiagram
    class Chunk {
        +string id
        +string text
        +list~float~ embedding
        +dict metadata
        +string paper_id
        +int chunk_index
        +float score
    }
    
    class Paper {
        +string id
        +string title
        +int year
        +string authors
    }
    
    Chunk --> Paper : belongs_to
```

## System Deployment

```mermaid
graph TB
    subgraph "Development Environment"
        DEV1[Local Machine]
        DEV2[Python venv]
        DEV3[Node.js + npm]
    end
    
    subgraph "Services"
        SVC1[Flask API<br/>localhost:5001]
        SVC2[React Dev Server<br/>localhost:5173]
        SVC3[Qdrant<br/>localhost:6333]
        SVC4[Neo4j<br/>localhost:7474]
    end
    
    subgraph "Docker Containers"
        DOCK1[Neo4j Container]
        DOCK2[Qdrant Container<br/>Optional]
    end
    
    DEV1 --> DEV2
    DEV1 --> DEV3
    DEV2 --> SVC1
    DEV3 --> SVC2
    SVC1 --> SVC3
    SVC1 --> SVC4
    DOCK1 --> SVC4
    DOCK2 --> SVC3
    
    style SVC1 fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
    style SVC2 fill:#D97757,stroke:#fff,stroke-width:2px,color:#fff
    style SVC3 fill:#4A90E2,stroke:#fff,stroke-width:2px,color:#fff
    style SVC4 fill:#008CC1,stroke:#fff,stroke-width:2px,color:#fff
```

## Key Features

1. **Hybrid Retrieval**: Combines vector similarity (60%) with graph relationships (40%)
2. **Multi-hop Reasoning**: Traverses up to 3 hops in the knowledge graph
3. **Temporal Filtering**: Queries by publication year or year ranges
4. **Entity Extraction**: Automatically extracts entities from queries
5. **LLM Response Generation**: Generates natural language answers from retrieved context
6. **Strict Context Adherence**: Ensures responses are based only on retrieved context
7. **Source Attribution**: Provides citations and confidence scores

## Performance Characteristics

- **Vector Search**: ~50-100ms per query
- **Graph Traversal**: ~100-200ms per query (depends on hop depth)
- **LLM Generation**: ~1-3s per response (depends on model)
- **Total Query Time**: ~1.5-3.5s end-to-end

