"""
Entity and Relationship Extractor
Uses LLM to extract entities and relationships from papers
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI


class EntityExtractor:
    """Extracts entities and relationships from papers using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize entity extractor
        
        Args:
            api_key: OpenAI API key (or from env)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cost-effective model
    
    def extract_entities_and_relationships(self, paper_text: str, metadata: Dict) -> Dict:
        """
        Extract entities and relationships from a paper
        
        Args:
            paper_text: Full text of the paper
            metadata: Paper metadata (title, authors, year, etc.)
            
        Returns:
            Dictionary with entities and relationships
        """
        # Truncate text if too long (LLM context limit)
        max_chars = 15000  # Leave room for prompt
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "..."
        
        prompt = self._create_extraction_prompt(paper_text, metadata)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from research papers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {"entities": [], "relationships": []}
    
    def _create_extraction_prompt(self, text: str, metadata: Dict) -> str:
        """Create prompt for entity/relationship extraction"""
        return f"""Extract entities and relationships from this research paper.

Paper Title: {metadata.get('title', 'Unknown')}
Authors: {', '.join(metadata.get('authors', []))}
Year: {metadata.get('year', 'Unknown')}

Paper Text (excerpt):
{text}

Extract the following in JSON format:

1. **Entities** - Key concepts, techniques, models, datasets mentioned:
   - Type: "Concept", "Technique", "Model", "Dataset", "Metric", "Architecture"
   - Name: The entity name
   - Description: Brief description from context

2. **Relationships** - How entities relate to each other:
   - Type: "BUILDS_ON", "IMPROVES", "USES", "COMPARES_TO", "INTRODUCES", "EXTENDS"
   - Source: Entity name
   - Target: Entity name
   - Description: How they relate

3. **Paper Relationships** - Connect this paper to entities:
   - Type: "INTRODUCES", "PROPOSES", "EVALUATES", "COMPARES"
   - Source: Paper title
   - Target: Entity name

Return JSON in this format:
{{
  "entities": [
    {{"type": "Model", "name": "Transformer", "description": "..."}},
    ...
  ],
  "relationships": [
    {{"type": "BUILDS_ON", "source": "BERT", "target": "Transformer", "description": "..."}},
    ...
  ]
}}"""

    def extract_from_chunks_batch(self, chunks: List[Dict], batch_size: int = 5) -> Dict:
        """
        Extract entities and relationships from multiple chunks in batches
        Processes multiple chunks per LLM API call for efficiency
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process per LLM call (default: 5)
            
        Returns:
            Aggregated entities and relationships with chunk mapping
        """
        all_entities = {}
        all_relationships = []
        chunk_entity_map = {}
        
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"    Processing batch {batch_num}/{total_batches} ({batch_num * batch_size}/{len(chunks)} chunks)...")
            
            # Extract from batch of chunks in one LLM call
            result = self._extract_from_chunk_batch(batch)
            
            # Process results for each chunk in the batch
            for i, chunk in enumerate(batch):
                chunk_id = chunk["metadata"].get("chunk_id", "")
                if chunk_id:
                    chunk_entities = []
                    # Get entities for this specific chunk from batch result
                    chunk_result = result.get("chunks", [{}])[i] if i < len(result.get("chunks", [])) else {}
                    
                    for entity in chunk_result.get("entities", []):
                        entity_name = entity.get("name", "")
                        if entity_name:
                            chunk_entities.append(entity_name)
                            # Aggregate entities (deduplicate by name)
                            if entity_name not in all_entities:
                                all_entities[entity_name] = entity
                    
                    chunk_entity_map[chunk_id] = chunk_entities
                    
                    # Add relationships for this chunk
                    all_relationships.extend(chunk_result.get("relationships", []))
        
        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships,
            "chunk_entity_map": chunk_entity_map
        }
    
    def _extract_from_chunk_batch(self, chunks: List[Dict]) -> Dict:
        """
        Extract entities from a batch of chunks in one LLM call
        
        Args:
            chunks: List of chunk dictionaries (batch)
            
        Returns:
            Dictionary with entities and relationships per chunk
        """
        # Combine chunks into one prompt
        chunk_texts = []
        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id", "")
            text = chunk["text"][:5000]  # Limit per chunk to fit in context
            metadata = chunk["metadata"]
            chunk_texts.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata
            })
        
        prompt = self._create_batch_extraction_prompt(chunk_texts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from research papers. Extract entities and relationships from multiple text chunks."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000  # More tokens for batch processing
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error extracting entities from batch: {e}")
            # Fallback: return empty results for each chunk
            return {
                "chunks": [{"entities": [], "relationships": []} for _ in chunks]
            }
    
    def _create_batch_extraction_prompt(self, chunk_texts: List[Dict]) -> str:
        """Create prompt for batch extraction"""
        chunks_text = ""
        for i, chunk_info in enumerate(chunk_texts):
            metadata = chunk_info["metadata"]
            chunks_text += f"""
Chunk {i+1} (ID: {chunk_info.get('chunk_id', 'unknown')}):
Paper: {metadata.get('title', 'Unknown')}
Text: {chunk_info['text']}

---
"""
        
        return f"""Extract entities and relationships from these research paper chunks.

{chunks_text}

For EACH chunk, extract:

1. **Entities** - Key concepts, techniques, models, datasets:
   - Type: "Concept", "Technique", "Model", "Dataset", "Metric", "Architecture"
   - Name: The entity name
   - Description: Brief description

2. **Relationships** - How entities relate:
   - Type: "BUILDS_ON", "IMPROVES", "USES", "COMPARES_TO", "INTRODUCES", "EXTENDS"
   - Source: Entity name
   - Target: Entity name
   - Description: How they relate

Return JSON in this format:
{{
  "chunks": [
    {{
      "chunk_id": "chunk_id_1",
      "entities": [
        {{"type": "Model", "name": "Transformer", "description": "..."}}
      ],
      "relationships": [
        {{"type": "BUILDS_ON", "source": "BERT", "target": "Transformer", "description": "..."}}
      ]
    }},
    ...
  ]
}}"""
    
    def extract_from_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Extract entities and relationships from multiple chunks (sequential, for compatibility)
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Aggregated entities and relationships
        """
        all_entities = {}
        all_relationships = []
        
        print(f"Extracting entities from {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
            
            result = self.extract_entities_and_relationships(
                chunk["text"],
                chunk["metadata"]
            )
            
            # Aggregate entities (deduplicate by name)
            for entity in result.get("entities", []):
                entity_name = entity.get("name", "")
                if entity_name and entity_name not in all_entities:
                    all_entities[entity_name] = entity
            
            # Add relationships
            all_relationships.extend(result.get("relationships", []))
        
        return {
            "entities": list(all_entities.values()),
            "relationships": all_relationships
        }

