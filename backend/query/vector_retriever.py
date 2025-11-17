"""
Vector Retriever
Wrapper around VectorStore for query engine
"""

import os
from typing import List, Dict, Optional
from backend.embeddings.vector_store import VectorStore


class VectorRetriever:
    """Wrapper around VectorStore for query engine"""
    
    def __init__(self,
                 collection_name: str = "research_papers",
                 embedding_model: Optional[str] = None,
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize vector retriever
        
        Args:
            collection_name: Qdrant collection name
            embedding_model: Embedding model name (default: from env)
            qdrant_url: Qdrant URL (default: from env)
            qdrant_api_key: Qdrant API key (default: from env)
        """
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        use_openai = embedding_model.startswith("text-embedding") or embedding_model.startswith("ada")
        
        # Get Qdrant config from env if not provided
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            use_openai=use_openai
        )
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of chunks with text, metadata, and score
        """
        return self.vector_store.search(query, top_k=top_k)
    
    def get_chunks_for_papers(self, paper_ids: List[str], query: str, top_k: int = 5) -> List[Dict]:
        """
        Get chunks from specific papers
        
        Args:
            paper_ids: List of paper IDs (filenames)
            query: Search query
            top_k: Number of results per paper
            
        Returns:
            List of chunks from specified papers
        """
        # Get all results
        all_results = self.search(query, top_k=top_k * len(paper_ids))
        
        # Filter by paper IDs
        filtered = []
        for result in all_results:
            result_paper_id = result["metadata"].get("filename", "")
            if result_paper_id in paper_ids:
                filtered.append(result)
        
        return filtered[:top_k * len(paper_ids)]
    
    def get_chunk_ids(self, results: List[Dict]) -> List[str]:
        """
        Extract chunk IDs from search results
        
        Args:
            results: Vector search results
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        for result in results:
            # Chunk ID is in metadata
            chunk_id = result["metadata"].get("chunk_id", "")
            if chunk_id:
                chunk_ids.append(chunk_id)
        
        return chunk_ids

