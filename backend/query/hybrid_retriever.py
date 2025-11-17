"""
Hybrid Retriever
Combines vector search and graph traversal results
"""

from typing import List, Dict, Optional
from .vector_retriever import VectorRetriever
from .graph_traverser import GraphTraverser
from .entity_extractor import QueryEntityExtractor


class HybridRetriever:
    """Combines vector search and graph traversal for hybrid retrieval"""
    
    def __init__(self,
                 vector_retriever: Optional[VectorRetriever] = None,
                 graph_traverser: Optional[GraphTraverser] = None,
                 entity_extractor: Optional[QueryEntityExtractor] = None):
        """
        Initialize hybrid retriever
        
        Args:
            vector_retriever: Vector retriever instance
            graph_traverser: Graph traverser instance
            entity_extractor: Query entity extractor instance
        """
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.graph_traverser = graph_traverser or GraphTraverser()
        self.entity_extractor = entity_extractor or QueryEntityExtractor()
    
    def retrieve(self,
                 query: str,
                 top_k: int = 10,
                 use_graph: bool = True,
                 max_hops: int = 2,
                 vector_weight: float = 0.6,
                 graph_weight: float = 0.4) -> Dict:
        """
        Hybrid retrieval combining vector search and graph traversal
        
        Args:
            query: User query
            top_k: Number of results to return
            use_graph: Whether to use graph traversal
            max_hops: Maximum graph traversal depth
            vector_weight: Weight for vector search scores (0-1)
            graph_weight: Weight for graph scores (0-1)
            
        Returns:
            Dictionary with combined results
        """
        # 1. Vector search - Get semantically similar chunks
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        
        # 2. Graph search - If enabled
        graph_results = {"entities": [], "papers": [], "relationships": []}
        graph_context = {}
        query_entities = []
        
        if use_graph:
            # Extract entities from query
            query_entities = self.entity_extractor.extract_entities_from_query(query)
            
            if query_entities:
                # Find entities in graph
                found_entities = self.graph_traverser.find_entities(query_entities)
                
                if found_entities:
                    # Get entity names from graph
                    entity_names = [e.get("name", "") for e in found_entities if e.get("name")]
                    
                    # Traverse from entities
                    graph_results = self.graph_traverser.traverse_from_entities(entity_names, max_hops=max_hops)
                    
                    # Get related papers
                    related_papers = self.graph_traverser.get_related_papers(entity_names)
                    graph_results["papers"].extend(related_papers)
        
        # 3. Graph expansion - Expand from vector results
        if vector_results and use_graph:
            chunk_ids = self.vector_retriever.get_chunk_ids(vector_results)
            if chunk_ids:
                graph_context = self.graph_traverser.expand_from_chunks(chunk_ids)
        
        # 4. Combine and rank results
        combined = self.combine_results(
            vector_results,
            graph_results,
            graph_context,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )
        
        return {
            "query": query,
            "query_entities": query_entities,
            "results": combined[:top_k],
            "vector_results_count": len(vector_results),
            "graph_entities_count": len(graph_results.get("entities", [])),
            "graph_papers_count": len(graph_results.get("papers", [])),
            "graph_context": graph_context
        }
    
    def combine_results(self,
                       vector_results: List[Dict],
                       graph_results: Dict,
                       graph_context: Dict,
                       vector_weight: float = 0.6,
                       graph_weight: float = 0.4) -> List[Dict]:
        """
        Combine and rank vector and graph results
        
        Args:
            vector_results: Vector search results
            graph_results: Graph traversal results
            graph_context: Graph context from chunks
            vector_weight: Weight for vector scores
            graph_weight: Weight for graph scores
            
        Returns:
            Combined and ranked results
        """
        combined = []
        
        # Process vector results
        for result in vector_results:
            chunk_id = result["metadata"].get("chunk_id", "")
            vector_score = result.get("score", 0.0)
            
            # Get graph context for this chunk
            chunk_graph_context = graph_context.get(chunk_id, {})
            
            # Calculate graph score based on context
            graph_score = 0.0
            if chunk_graph_context:
                # Score based on number of entities and relationships
                entity_count = len(chunk_graph_context.get("entities", []))
                rel_count = len(chunk_graph_context.get("relationships", []))
                graph_score = min(1.0, (entity_count * 0.1 + rel_count * 0.05))
            
            # Combined score
            combined_score = (vector_weight * vector_score) + (graph_weight * graph_score)
            
            combined.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "score": combined_score,
                "vector_score": vector_score,
                "graph_score": graph_score,
                "source": "hybrid",
                "graph_context": chunk_graph_context
            })
        
        # Add graph-found papers (if not already in vector results)
        paper_titles = {r["metadata"].get("title", "") for r in vector_results}
        
        for paper in graph_results.get("papers", []):
            if paper.get("title") not in paper_titles:
                # Create a result entry for graph-found paper
                combined.append({
                    "text": f"Paper: {paper.get('title', '')}",
                    "metadata": {
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "filename": paper.get("id", ""),
                        "source": "graph"
                    },
                    "score": graph_weight * (1.0 / (paper.get("hops", 1) + 1)),  # Closer = higher score
                    "vector_score": 0.0,
                    "graph_score": graph_weight * (1.0 / (paper.get("hops", 1) + 1)),
                    "source": "graph",
                    "graph_context": {
                        "relationship": paper.get("relationship", ""),
                        "entity": paper.get("entity", ""),
                        "hops": paper.get("hops", 0)
                    }
                })
        
        # Sort by combined score (descending)
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate by chunk_id or title
        seen = set()
        deduplicated = []
        for result in combined:
            chunk_id = result["metadata"].get("chunk_id", "")
            title = result["metadata"].get("title", "")
            key = chunk_id or title or result.get("text", "")[:50]  # Use text snippet as fallback
            
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(result)
            # If no key, still include it (might be graph-only result)
            elif not key:
                deduplicated.append(result)
        
        return deduplicated

