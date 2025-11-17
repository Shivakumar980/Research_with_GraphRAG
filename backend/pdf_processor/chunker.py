"""
Document Chunker
Splits documents into chunks for embedding and retrieval
"""

from typing import List, Dict

# Try new langchain-text-splitters first, fallback to old langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback to simple chunking if langchain not available
        RecursiveCharacterTextSplitter = None


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk a document into smaller pieces
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_id": f"{metadata.get('filename', 'unknown')}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunked_docs.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunked_docs
    
    def chunk_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Chunk multiple papers
        
        Args:
            papers: List of paper dictionaries with 'text' and 'metadata'
            
        Returns:
            List of all chunks from all papers
        """
        all_chunks = []
        
        for paper in papers:
            chunks = self.chunk_document(paper["text"], paper["metadata"])
            all_chunks.extend(chunks)
        
        return all_chunks

