"""
GraphRAG Query Engine
Main query interface combining vector search and graph traversal
"""

from typing import List, Dict, Optional
from .hybrid_retriever import HybridRetriever
from .entity_extractor import QueryEntityExtractor
from .response_generator import ResponseGenerator


class GraphRAGQuery:
    """Main GraphRAG query engine"""
    
    def __init__(self,
                 use_graph: bool = True,
                 max_hops: int = 2,
                 vector_weight: float = 0.6,
                 graph_weight: float = 0.4,
                 use_generator: bool = True,
                 generator_model: str = "gpt-4o-mini"):
        """
        Initialize GraphRAG query engine
        
        Args:
            use_graph: Whether to use graph traversal (default: True)
            max_hops: Maximum graph traversal depth (default: 2)
            vector_weight: Weight for vector search scores (default: 0.6)
            graph_weight: Weight for graph scores (default: 0.4)
            use_generator: Whether to generate natural language responses (default: True)
            generator_model: LLM model for response generation (default: gpt-4o-mini)
        """
        self.use_graph = use_graph
        self.max_hops = max_hops
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.use_generator = use_generator
        
        self.hybrid_retriever = HybridRetriever()
        self.entity_extractor = QueryEntityExtractor()
        
        # Initialize response generator if enabled
        self.response_generator = None
        if use_generator:
            try:
                self.response_generator = ResponseGenerator(model=generator_model)
            except Exception as e:
                print(f"Warning: Could not initialize response generator: {e}")
                print("  Response generation will be disabled. Set OPENAI_API_KEY to enable.")
                self.use_generator = False
    
    def query(self,
              query: str,
              top_k: int = 5,
              use_temporal: bool = False,
              max_hops: Optional[int] = None,
              generate_answer: Optional[bool] = None) -> Dict:
        """
        Query the GraphRAG system
        
        Args:
            query: User query text
            top_k: Number of results to return
            use_temporal: Whether to apply temporal filtering
            max_hops: Override default max_hops for this query
            generate_answer: Whether to generate natural language answer (None = use self.use_generator)
            
        Returns:
            Dictionary with query results and metadata (and generated answer if enabled)
        """
        max_hops = max_hops or self.max_hops
        generate_answer = generate_answer if generate_answer is not None else self.use_generator
        
        # Detect temporal intent
        temporal_info = self.entity_extractor.detect_temporal_intent(query)
        if temporal_info["is_temporal"]:
            use_temporal = True
        
        # Perform hybrid retrieval
        results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            use_graph=self.use_graph,
            max_hops=max_hops,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight
        )
        
        # Apply temporal filtering if needed
        if use_temporal and temporal_info.get("is_temporal"):
            # Only apply filter if there are actual years or year ranges
            # If just keywords without years, just sort by year but don't filter
            if temporal_info.get("years") or temporal_info.get("year_ranges"):
                results = self._apply_temporal_filter(results, temporal_info)
            else:
                # Just add temporal info, don't filter
                results["temporal_info"] = temporal_info
        
        # Generate natural language answer if enabled
        if generate_answer and self.response_generator:
            try:
                # Use evolution timeline for temporal queries
                if temporal_info.get("is_temporal") and temporal_info.get("keywords"):
                    generated = self.response_generator.generate_with_evolution_timeline(
                        query=query,
                        results=results.get("results", []),
                        graph_context=results.get("graph_context")
                    )
                else:
                    generated = self.response_generator.generate(
                        query=query,
                        results=results.get("results", []),
                        graph_context=results.get("graph_context"),
                        temporal_info=temporal_info
                    )
                
                results["generated_answer"] = generated
            except Exception as e:
                results["generated_answer"] = {
                    "answer": f"Error generating answer: {str(e)}",
                    "sources": [],
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        return results
    
    def _apply_temporal_filter(self, results: Dict, temporal_info: Dict) -> Dict:
        """
        Apply temporal filtering to results
        
        Args:
            results: Query results
            temporal_info: Temporal information from query
            
        Returns:
            Filtered results
        """
        if not temporal_info.get("is_temporal"):
            return results
        
        years = temporal_info.get("years", [])
        year_ranges = temporal_info.get("year_ranges", [])
        
        filtered_results = []
        
        for result in results.get("results", []):
            result_year = result["metadata"].get("year")
            
            # If no year in result, keep it (can't filter)
            if not result_year:
                filtered_results.append(result)
                continue
            
            # Check year ranges
            if year_ranges:
                keep = False
                for start_year, end_year in year_ranges:
                    if start_year <= result_year <= end_year:
                        keep = True
                        break
                if keep:
                    filtered_results.append(result)
            # Check specific years
            elif years:
                if result_year in years:
                    filtered_results.append(result)
            # If just temporal keywords, order by year
            else:
                filtered_results.append(result)
        
        # Sort by year if temporal
        if temporal_info.get("keywords") and not years and not year_ranges:
            filtered_results.sort(key=lambda x: x["metadata"].get("year", 0) or 0)
        
        results["results"] = filtered_results
        results["temporal_info"] = temporal_info
        
        return results

