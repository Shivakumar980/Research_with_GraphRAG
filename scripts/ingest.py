#!/usr/bin/env python3
"""
Document Ingestion
Parse PDFs, extract text, chunk documents, and store embeddings in vector database
Run this first, then run build_graph.py to create the knowledge graph
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.pdf_processor.extractor import PDFExtractor
from backend.pdf_processor.chunker import DocumentChunker
from backend.embeddings.vector_store import VectorStore


def main():
    """Main pipeline to ingest documents into vector store"""
    
    print("="*80)
    print("Document Processing & Vector Store Builder")
    print("="*80)
    
    # Configuration
    papers_dir = project_root / "data" / "papers"
    collection_name = "research_papers"
    
    # Step 1: Extract text from PDFs
    print("\n[1/3] Extracting text from PDFs...")
    print("-" * 80)
    extractor = PDFExtractor()
    papers = []
    
    for pdf_file in papers_dir.glob("*.pdf"):
        result = extractor.process_paper(str(pdf_file))
        papers.append(result)
        print(f"  ✓ {result['metadata'].get('title', pdf_file.name)}")
        print(f"    {result['word_count']:,} words extracted")
    
    if not papers:
        print("❌ No PDF files found in data/papers/")
        return
    
    # Step 2: Chunk documents
    print(f"\n[2/3] Chunking documents...")
    print("-" * 80)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_papers(papers)
    print(f"  ✓ Created {len(chunks)} chunks from {len(papers)} papers")
    
    # Step 3: Create embeddings and store in Qdrant
    print(f"\n[3/3] Creating embeddings and storing in Qdrant...")
    print("-" * 80)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    use_openai = embedding_model.startswith("text-embedding") or embedding_model.startswith("ada")
    
    vector_store = VectorStore(
        collection_name=collection_name,
        embedding_model=embedding_model,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        use_openai=use_openai
    )
    
    # Clear all existing data from vector store
    print("  Clearing all existing data from vector store...")
    vector_store.clear_collection()
    
    # Add all chunks (fresh insert)
    vector_store.add_chunks(chunks)
    info = vector_store.get_collection_info()
    print(f"  ✓ Vector store ready: {info.get('points_count', 0)} points")
    
    print("\n" + "="*80)
    print("✅ Documents Processed and Stored Successfully!")
    print("="*80)
    print("\n📊 Summary:")
    print(f"  • Papers processed: {len(papers)}")
    print(f"  • Chunks created: {len(chunks)}")
    print(f"  • Vector store: {info.get('points_count', 0)} points in Qdrant")
    print(f"\nNext steps:")
    print(f"  1. Query the system: python scripts/query.py 'your question'")
    print(f"  2. Build knowledge graph: python scripts/build_graph.py")


if __name__ == "__main__":
    main()

