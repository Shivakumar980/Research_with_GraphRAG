#!/usr/bin/env python3
"""
GraphRAG Query Script
Hybrid retrieval combining vector search and graph traversal
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.query.engine import GraphRAGQuery


def main():
    parser = argparse.ArgumentParser(description="Query GraphRAG system")
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph traversal (vector search only)")
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum graph traversal depth (default: 2)")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal filtering")
    parser.add_argument("--no-generate", action="store_true", help="Disable LLM answer generation (show raw results only)")
    
    args = parser.parse_args()
    
    query = " ".join(args.query)
    
    print("="*80)
    print("GraphRAG Query")
    print("="*80)
    print(f"Query: {query}\n")
    
    # Initialize query engine
    query_engine = GraphRAGQuery(
        use_graph=not args.no_graph,
        max_hops=args.max_hops,
        use_generator=not args.no_generate
    )
    
    # Execute query
    try:
        result = query_engine.query(
            query=query,
            top_k=args.top_k,
            use_temporal=args.temporal,
            generate_answer=not args.no_generate
        )
        
        # Show query entities found
        if result.get("query_entities"):
            print(f"📋 Entities detected: {', '.join(result['query_entities'])}")
        
        # Show graph stats
        if not args.no_graph:
            print(f"🔍 Graph: {result.get('graph_entities_count', 0)} entities, "
                  f"{result.get('graph_papers_count', 0)} papers found")
        
        # Show temporal info
        if result.get("temporal_info", {}).get("is_temporal"):
            temporal = result["temporal_info"]
            print(f"📅 Temporal query detected: {', '.join(temporal.get('keywords', []))}")
            if temporal.get("years"):
                print(f"   Years: {', '.join(map(str, temporal['years']))}")
        
        # Show generated answer if available
        if result.get("generated_answer"):
            generated = result["generated_answer"]
            print(f"\n💬 Generated Answer (Confidence: {generated.get('confidence', 0):.2f}):\n")
            print(f"{generated.get('answer', 'No answer generated')}\n")
            
            # Show timeline if available
            if generated.get("timeline"):
                print("📅 Evolution Timeline:\n")
                for entry in generated["timeline"]:
                    print(f"  {entry['year']}: {entry['summary']}\n")
            
            # Show sources
            if generated.get("sources"):
                print("📚 Sources:\n")
                for i, source in enumerate(generated["sources"][:5], 1):
                    title = source.get("title", "Unknown")
                    year = source.get("year", "?")
                    print(f"  {i}. {title} ({year})")
                print()
        
        print(f"\n📊 Retrieved Results ({len(result.get('results', []))}):\n")
        
        # Display results
        for i, res in enumerate(result.get("results", []), 1):
            print(f"[{i}] Score: {res['score']:.3f} "
                  f"(Vector: {res.get('vector_score', 0):.3f}, "
                  f"Graph: {res.get('graph_score', 0):.3f})")
            print(f"    Source: {res.get('source', 'unknown')}")
            print(f"    Paper: {res['metadata'].get('title', 'Unknown')}")
            print(f"    Year: {res['metadata'].get('year', 'Unknown')}")
            
            # Show graph context if available
            graph_ctx = res.get("graph_context", {})
            if graph_ctx:
                entities = graph_ctx.get("entities", [])
                if entities:
                    entity_names = [e.get("name", "") for e in entities[:3]]
                    print(f"    Entities: {', '.join(entity_names)}")
                    if len(entities) > 3:
                        print(f"              ... and {len(entities) - 3} more")
            
            # Clean up text for better display
            text = res['text'].replace('\n', ' ').strip()
            import re
            text = re.sub(r'\s+', ' ', text)
            # Show first 300 chars
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"    Text: {preview}")
            print()
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

