#!/usr/bin/env python3
"""
Build Knowledge Graph
Extracts entities and relationships from documents and builds Neo4j knowledge graph
Run this after ingest.py has populated the vector store
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.pdf_processor.extractor import PDFExtractor
from backend.pdf_processor.chunker import DocumentChunker
from backend.graph.entity_extractor import EntityExtractor
from backend.graph.graph_builder import GraphBuilder


def main():
    """Main pipeline to build knowledge graph"""
    
    print("="*80)
    print("Knowledge Graph Builder")
    print("="*80)
    
    # Configuration
    papers_dir = project_root / "data" / "papers"
    
    # Step 1: Load papers (to get metadata)
    print("\n[1/3] Loading paper metadata...")
    print("-" * 80)
    extractor = PDFExtractor()
    papers = []
    
    for pdf_file in papers_dir.glob("*.pdf"):
        result = extractor.process_paper(str(pdf_file))
        papers.append(result)
        print(f"  ✓ {result['metadata'].get('title', pdf_file.name)}")
    
    if not papers:
        print("❌ No PDF files found in data/papers/")
        return
    
    # Step 2: Extract entities and relationships
    print(f"\n[2/3] Extracting entities and relationships...")
    print("-" * 80)
    print("  ⚠ This step requires OpenAI API key (set OPENAI_API_KEY env var)")
    
    # Use same chunking as ingest.py to ensure chunk_id consistency
    # This ensures chunks in Neo4j match chunks in Qdrant
    print("  Chunking papers (using same chunker as ingest.py)...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    all_chunks = chunker.chunk_papers(papers)
    print(f"  ✓ Created {len(all_chunks)} chunks (matching Qdrant chunks)")
    
    # Configuration: Process all chunks or a sample
    MAX_CHUNKS_FOR_EXTRACTION = os.getenv("MAX_CHUNKS_FOR_EXTRACTION")
    if MAX_CHUNKS_FOR_EXTRACTION:
        max_chunks = int(MAX_CHUNKS_FOR_EXTRACTION)
        print(f"  ⚠ Processing {max_chunks} chunks (limited by MAX_CHUNKS_FOR_EXTRACTION env var)")
        print(f"     Remove MAX_CHUNKS_FOR_EXTRACTION from .env to process all {len(all_chunks)} chunks")
    else:
        max_chunks = len(all_chunks)
        print(f"  Processing ALL {len(all_chunks)} chunks for entity extraction...")
        print(f"  💡 This may take a while and incur API costs.")
        print(f"  💡 Tip: Set MAX_CHUNKS_FOR_EXTRACTION=50 in .env to limit for testing")
    
    # Track chunk-to-entity mapping for linking
    chunk_entity_map = {}  # {chunk_id: [entity_names]}
    
    # Process chunks (all or limited)
    chunks_to_process = all_chunks[:max_chunks] if max_chunks < len(all_chunks) else all_chunks
    
    try:
        entity_extractor = EntityExtractor()
        print(f"  Processing {len(chunks_to_process)} chunks for knowledge graph creation...")
        print(f"  Using batch extraction (5 chunks per LLM call)...")
        
        # Extract entities using batch processing (much faster!)
        extraction_result = entity_extractor.extract_from_chunks_batch(
            chunks_to_process,
            batch_size=5  # Process 5 chunks per LLM API call
        )
        
        entities = extraction_result.get("entities", [])
        relationships = extraction_result.get("relationships", [])
        chunk_entity_map = extraction_result.get("chunk_entity_map", {})
        
        print(f"  ✓ Extracted {len(entities)} entities")
        print(f"  ✓ Extracted {len(relationships)} relationships")
        print(f"  ✓ Mapped {len(chunk_entity_map)} chunks to entities")
        
        # Save extracted entities/relationships to file (backup before adding to Neo4j)
        extraction_backup_path = project_root / "data" / "extracted_entities_backup.json"
        backup_data = {
            "entities": entities,
            "relationships": relationships,
            "chunk_entity_map": chunk_entity_map,
            "extraction_timestamp": datetime.now().isoformat()
        }
        with open(extraction_backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        print(f"  ✓ Saved extraction backup to: {extraction_backup_path}")
        
    except Exception as e:
        print(f"  ⚠ Entity extraction failed: {e}")
        print("  Continuing without entity extraction...")
        entities = []
        relationships = []
        chunk_entity_map = {}
        
        # Try to load from backup if available
        extraction_backup_path = project_root / "data" / "extracted_entities_backup.json"
        if extraction_backup_path.exists():
            print(f"  💡 Attempting to load from backup: {extraction_backup_path}")
            try:
                with open(extraction_backup_path, 'r') as f:
                    backup_data = json.load(f)
                    entities = backup_data.get("entities", [])
                    relationships = backup_data.get("relationships", [])
                    chunk_entity_map = backup_data.get("chunk_entity_map", {})
                    print(f"  ✓ Loaded {len(entities)} entities and {len(relationships)} relationships from backup")
            except Exception as backup_error:
                print(f"  ⚠ Failed to load backup: {backup_error}")
    
    # Step 3: Build knowledge graph
    print(f"\n[3/3] Building knowledge graph...")
    print("-" * 80)
    
    try:
        graph_builder = GraphBuilder()
        
        # Add papers as nodes
        for paper in papers:
            graph_builder.add_paper(paper["metadata"])
        
        # Add entities and relationships (using batch operations for speed)
        if entities:
            print(f"  Adding {len(entities)} entities in batches...")
            entities_added = graph_builder.add_entities_batch(entities, batch_size=100)
            entities_failed = len(entities) - entities_added
            print(f"  ✓ Added {entities_added} entities ({entities_failed} failed)")
        
        relationships_added = 0
        relationships_failed = 0
        if relationships:
            print(f"  Adding {len(relationships)} relationships in batches...")
            relationships_added = graph_builder.add_relationships_batch(relationships, batch_size=100)
            relationships_failed = len(relationships) - relationships_added
            print(f"  ✓ Added {relationships_added} relationships ({relationships_failed} failed)")
        
        # Add chunk nodes and link to entities
        print("  Creating chunk nodes and linking to entities...")
        chunks_added = 0
        links_created = 0
        
        # Get all unique chunks that have entities
        chunks_to_add = set()
        for chunk_id in chunk_entity_map.keys():
            chunks_to_add.add(chunk_id)
        
        # Also add chunks from chunks_to_process (even if no entities)
        for chunk in chunks_to_process:
            chunk_id = chunk["metadata"].get("chunk_id", "")
            if chunk_id:
                chunks_to_add.add(chunk_id)
        
        # Find chunk metadata for each chunk_id
        chunk_metadata_map = {}
        for chunk in chunks_to_process:
            chunk_id = chunk["metadata"].get("chunk_id", "")
            if chunk_id:
                chunk_metadata_map[chunk_id] = chunk["metadata"]
        
        # Create chunk nodes (using batch operations)
        print("  Creating chunk nodes in batches...")
        chunks_to_add_list = [
            {"chunk_id": chunk_id, "metadata": chunk_metadata_map[chunk_id]}
            for chunk_id in chunks_to_add
            if chunk_id in chunk_metadata_map
        ]
        chunks_added = graph_builder.add_chunks_batch(chunks_to_add_list, batch_size=100)
        print(f"  ✓ Created {chunks_added} chunk nodes")
        
        # Now link chunks to entities (using batch operations)
        print("  Linking chunks to entities in batches...")
        chunk_entity_pairs = []
        for chunk_id in chunks_to_add:
            if chunk_id in chunk_entity_map:
                for entity_name in chunk_entity_map[chunk_id]:
                    chunk_entity_pairs.append({
                        "chunk_id": chunk_id,
                        "entity_name": entity_name
                    })
        
        links_created = graph_builder.link_chunks_to_entities_batch(chunk_entity_pairs, batch_size=100)
        links_failed = len(chunk_entity_pairs) - links_created
        print(f"  ✓ Created {links_created} chunk-entity links ({links_failed} failed)")
        
        # Add temporal relationships (papers by year)
        graph_builder.add_temporal_relationships(papers)
        
        graph_info = graph_builder.get_graph_info()
        print(f"  ✓ Graph built: {graph_info.get('nodes', 0)} nodes, {graph_info.get('edges', 0)} edges")
        
        # Export graph (graph is stored in Neo4j)
        graph_path = project_root / "data" / "knowledge_graph_export.json"
        graph_builder.save_graph(str(graph_path))
        print(f"  ✓ Graph exported to: {graph_path}")
        print(f"  ✓ Graph stored in Neo4j at: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")
        
        # Close connection
        graph_builder.close()
        
    except Exception as e:
        print(f"  ⚠ Graph building failed: {e}")
        print("  Continuing without graph...")
    
    print("\n" + "="*80)
    print("✅ Knowledge Graph Built Successfully!")
    print("="*80)
    print("\n📊 Summary:")
    print(f"  • Papers processed: {len(papers)}")
    print(f"  • Entities extracted: {len(entities)}")
    print(f"  • Relationships extracted: {len(relationships)}")
    print(f"  • Chunks linked: {len(chunk_entity_map) if 'chunk_entity_map' in locals() else 0}")
    print(f"  • Graph nodes: {graph_info.get('nodes', 0) if 'graph_info' in locals() else 0}")
    print(f"  • Graph edges: {graph_info.get('edges', 0) if 'graph_info' in locals() else 0}")
    print(f"\nNext steps:")
    print(f"  1. Query the system: python scripts/query.py 'your question'")
    print(f"  2. Visualize graph: python scripts/visualize.py")


if __name__ == "__main__":
    main()

