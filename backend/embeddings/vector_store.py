"""
Qdrant Vector Store
Stores document chunks as embeddings in Qdrant
"""

import os
import hashlib
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Try to import both embedding options
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VectorStore:
    """Manages embeddings and vector search in Qdrant"""
    
    def __init__(self, 
                 collection_name: str = "research_papers",
                 embedding_model: str = "text-embedding-3-small",
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 use_openai: bool = True):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of Qdrant collection
            embedding_model: Model name for embeddings
                          - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                          - Sentence Transformers: "sentence-transformers/all-MiniLM-L6-v2"
            qdrant_url: Qdrant server URL (None for local, or cloud URL)
            qdrant_api_key: Qdrant API key (required for cloud)
            use_openai: If True, use OpenAI embeddings; if False, use sentence-transformers
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.use_openai = use_openai
        
        # Initialize embedding model
        if use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")
            
            print(f"Using OpenAI embeddings: {embedding_model}")
            self.openai_client = OpenAI(api_key=api_key)
            
            # OpenAI embedding dimensions
            embedding_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self.embedding_dim = embedding_dims.get(embedding_model, 1536)
            self.embedding_model = None  # Not used for OpenAI
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
            print(f"Using sentence-transformers: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.openai_client = None
        
        # Initialize Qdrant client
        if qdrant_url:
            # Remote Qdrant (cloud or remote instance)
            if qdrant_api_key:
                print(f"Connecting to Qdrant Cloud/Remote: {qdrant_url}")
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
            else:
                print(f"Connecting to Qdrant (no auth): {qdrant_url}")
                self.client = QdrantClient(url=qdrant_url)
        else:
            # Local Qdrant
            print("Using local Qdrant storage")
            self.client = QdrantClient(path="./qdrant_storage")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise
    
    def clear_collection(self):
        """
        Clear all points from the collection (but keep the collection)
        """
        try:
            # Delete all points in the collection
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"all": True}
            )
            print(f"✓ Cleared all points from collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"⚠ Error clearing collection: {e}")
            # If collection doesn't exist, that's fine
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.use_openai:
            # Use OpenAI embeddings
            print(f"Generating embeddings with OpenAI ({len(texts)} texts)...")
            embeddings = []
            
            # OpenAI API has batch limit, process in chunks
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 500 == 0:
                    print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")
            
            return embeddings
        else:
            # Use sentence-transformers
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.tolist()
    
    def _generate_chunk_id(self, chunk: Dict) -> int:
        """Generate a unique ID for a chunk based on its content"""
        # Create a hash from text and metadata
        content = f"{chunk['text']}{chunk['metadata'].get('title', '')}{chunk['metadata'].get('chunk_index', 0)}"
        # Use first 8 bytes of hash as integer ID
        hash_bytes = hashlib.md5(content.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big', signed=False)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,))
    )
    def _upsert_batch(self, points: List[PointStruct], batch_num: int, total_batches: int):
        """Upload a batch of points with retry logic"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as e:
            print(f"  ⚠ Batch {batch_num}/{total_batches} failed: {type(e).__name__}")
            raise
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 50):
        """
        Add document chunks to vector store with batch processing and retry logic
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            batch_size: Number of chunks to upload per batch (default: 100)
        """
        print(f"Adding {len(chunks)} chunks to vector store...")
        print(f"Using batch size: {batch_size}")
        
        # Generate embeddings for all chunks
        texts = [chunk["text"] for chunk in chunks]
        print(f"Generating embeddings with {'OpenAI' if self.use_openai else 'Sentence Transformers'} ({len(texts)} texts)...")
        embeddings = self.generate_embeddings(texts)
        
        # Process in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        total_added = 0
        
        for batch_idx in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_idx:batch_idx + batch_size]
            batch_embeddings = embeddings[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            # Prepare points for this batch
            points = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                # Store metadata as payload
                payload = {
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
                
                # Generate unique ID
                chunk_id = self._generate_chunk_id(chunk)
                
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload batch with retry logic
            try:
                self._upsert_batch(points, batch_num, total_batches)
                total_added += len(points)
                print(f"  ✓ Batch {batch_num}/{total_batches}: Added {len(points)} chunks (Total: {total_added}/{len(chunks)})")
            except Exception as e:
                print(f"  ❌ Batch {batch_num}/{total_batches} failed after retries: {e}")
                print(f"     Skipping this batch. You may need to re-run the build.")
                # Continue with next batch instead of failing completely
                continue
        
        print(f"✓ Completed: Added {total_added}/{len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=query
            )
            query_embedding = response.data[0].embedding
        else:
            query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        search_results = []
        for result in results:
            search_results.append({
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            })
        
        return search_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "config": collection_info.config.dict()
            }
        except Exception as e:
            return {"error": str(e)}
    

