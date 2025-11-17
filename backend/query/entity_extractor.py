"""
Query Entity Extractor
Extracts entities from user queries to enable graph traversal
Supports both LLM-based and regex-based extraction with automatic fallback
"""

import os
import re
import json
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QueryEntityExtractor:
    """Extracts entities from user queries using LLM (with regex fallback)"""
    
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        """
        Initialize query entity extractor
        
        Args:
            use_llm: Whether to use LLM for extraction (default: True)
            api_key: OpenAI API key (or from env)
        """
        # Common technical terms that are likely entities
        self.technical_keywords = [
            "transformer", "bert", "gpt", "llama", "lora", "attention",
            "flash attention", "vision transformer", "diffusion", "ddpm",
            "self-attention", "multi-head", "encoder", "decoder"
        ]
        
        # LLM setup
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self.llm_client = None
        
        if self.use_llm:
            try:
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
                    self.model = "gpt-4o-mini"  # Cost-effective model
                else:
                    print("Warning: OPENAI_API_KEY not found. Falling back to regex-based extraction.")
                    self.use_llm = False
            except Exception as e:
                print(f"Warning: Could not initialize LLM client: {e}. Falling back to regex-based extraction.")
                self.use_llm = False
    
    def extract_entities_from_query(self, query: str, use_llm: Optional[bool] = None) -> List[str]:
        """
        Extract entity names from query text
        
        Uses LLM-based extraction if available, falls back to regex-based extraction
        
        Args:
            query: User query text
            use_llm: Override default LLM usage (None = use self.use_llm)
            
        Returns:
            List of potential entity names
        """
        use_llm = use_llm if use_llm is not None else self.use_llm
        
        # Try LLM-based extraction first
        if use_llm and self.llm_client:
            try:
                llm_entities = self._extract_entities_with_llm(query)
                if llm_entities:
                    return llm_entities
            except Exception as e:
                print(f"LLM extraction failed: {e}. Falling back to regex-based extraction.")
        
        # Fallback to regex-based extraction
        return self._extract_entities_with_regex(query)
    
    def _extract_entities_with_llm(self, query: str) -> List[str]:
        """
        Extract entities using LLM
        
        Args:
            query: User query text
            
        Returns:
            List of extracted entity names
        """
        prompt = self._create_entity_extraction_prompt(query)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting entity names from queries about research papers and technical concepts. Extract only the actual entity names (models, techniques, concepts, etc.), not general terms."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            entities = result.get("entities", [])
            
            # Ensure we return a list of strings
            if isinstance(entities, list):
                # Filter and clean entities
                cleaned = [str(e).strip() for e in entities if e and len(str(e).strip()) > 1]
                return cleaned
            
            return []
            
        except Exception as e:
            raise Exception(f"LLM extraction error: {e}")
    
    def _create_entity_extraction_prompt(self, query: str) -> str:
        """Create prompt for LLM-based entity extraction"""
        return f"""Extract entity names from this query about research papers and technical concepts.

Query: "{query}"

Extract all entity names mentioned in the query. Entities can be:
- Model names (e.g., "BERT", "GPT", "Transformer", "LLaMA")
- Techniques (e.g., "Flash Attention", "LoRA", "Self-Attention")
- Concepts (e.g., "Diffusion", "Vision Transformer")
- Architectures (e.g., "Encoder-Decoder")

Return ONLY the entity names as a JSON array. Do not include:
- General terms (e.g., "paper", "model", "technique")
- Question words (e.g., "what", "how", "why")
- Common verbs or adjectives

Example:
Query: "How does BERT improve upon Transformer architecture?"
Response: {{"entities": ["BERT", "Transformer"]}}

Query: "What is Flash Attention and how does it relate to self-attention?"
Response: {{"entities": ["Flash Attention", "self-attention"]}}

Now extract entities from the query above:"""
    
    def _extract_entities_with_regex(self, query: str) -> List[str]:
        """
        Extract entities using regex patterns (fallback method)
        
        Args:
            query: User query text
            
        Returns:
            List of potential entity names
        """
        entities = []
        query_lower = query.lower()
        
        # Method 1: Check for known technical keywords
        for keyword in self.technical_keywords:
            if keyword in query_lower:
                # Capitalize appropriately
                if " " in keyword:
                    # Multi-word: "flash attention" -> "Flash Attention"
                    entities.append(keyword.title())
                else:
                    # Single word: "bert" -> "BERT" or "Transformer"
                    entities.append(keyword.capitalize())
        
        # Method 2: Extract capitalized words (likely proper nouns/entity names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        for word in capitalized_words:
            # Filter out common words
            if word.lower() not in ["the", "and", "or", "how", "what", "when", "where", "why"]:
                entities.append(word)
        
        # Method 3: Extract quoted strings (often entity names)
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Method 4: Extract words after "of", "from", "to" (often entities)
        # e.g., "evolution of Transformer" -> "Transformer"
        patterns = [
            r'(?:of|from|to|in|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:architecture|model|technique|method|algorithm)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Deduplicate and clean
        entities = list(set(entities))
        entities = [e.strip() for e in entities if len(e.strip()) > 1]
        
        return entities
    
    def detect_temporal_intent(self, query: str) -> Dict:
        """
        Detect if query has temporal/time-based intent
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with temporal info:
            - is_temporal: bool
            - keywords: List of temporal keywords found
            - years: List of years mentioned
        """
        query_lower = query.lower()
        
        temporal_keywords = [
            "evolve", "evolution", "evolved", "develop", "development",
            "before", "after", "since", "until",
            "recent", "early", "later", "previous", "next",
            "timeline", "history", "chronological", "over time"
        ]
        
        # Extract years first (4-digit numbers)
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        years = [int(y) for y in years]
        
        # Extract year ranges (e.g., "2017 to 2022", "from 2017 to 2022")
        year_range_pattern = r'(\d{4})\s*(?:to|-|until)\s*(\d{4})'
        year_ranges = re.findall(year_range_pattern, query)
        year_ranges = [(int(start), int(end)) for start, end in year_ranges]
        
        # Only detect "to" and "from" as temporal if they're part of year ranges
        # Otherwise they're too common (e.g., "relate to", "from X to Y" without years)
        found_keywords = [kw for kw in temporal_keywords if kw in query_lower]
        
        # Add "to" and "from" only if they're in year ranges
        if year_ranges:
            if "to" not in found_keywords and "to" in query_lower:
                found_keywords.append("to")
            if "from" not in found_keywords and "from" in query_lower:
                found_keywords.append("from")
        
        is_temporal = len(found_keywords) > 0 or len(years) > 0 or len(year_ranges) > 0
        
        return {
            "is_temporal": is_temporal,
            "keywords": found_keywords,
            "years": years,
            "year_ranges": year_ranges
        }
    
    def match_entities_in_graph(self, 
                                query_entities: List[str], 
                                graph_entities: List[Dict],
                                use_llm: Optional[bool] = None) -> List[str]:
        """
        Match query entities to actual entities in graph
        
        Uses LLM to improve matching accuracy if available
        
        Args:
            query_entities: Entities extracted from query
            graph_entities: Entities found in graph (from GraphTraverser)
            use_llm: Override default LLM usage for matching (None = use self.use_llm)
            
        Returns:
            List of matched entity names (from graph)
        """
        if not query_entities or not graph_entities:
            return []
        
        use_llm = use_llm if use_llm is not None else self.use_llm
        
        # Try LLM-based matching if available
        if use_llm and self.llm_client and len(graph_entities) > 0:
            try:
                llm_matched = self._match_entities_with_llm(query_entities, graph_entities)
                if llm_matched:
                    return llm_matched
            except Exception as e:
                print(f"LLM matching failed: {e}. Falling back to regex-based matching.")
        
        # Fallback to regex-based matching
        return self._match_entities_with_regex(query_entities, graph_entities)
    
    def _match_entities_with_llm(self, 
                                  query_entities: List[str], 
                                  graph_entities: List[Dict]) -> List[str]:
        """
        Match query entities to graph entities using LLM
        
        Args:
            query_entities: Entities from query
            graph_entities: Entities from graph
            
        Returns:
            List of matched entity names
        """
        # Prepare graph entity list (limit to avoid token limits)
        graph_entity_list = graph_entities[:50]  # Limit to 50 for efficiency
        graph_entity_names = [e.get("name", "") for e in graph_entity_list if e.get("name")]
        
        prompt = f"""Match query entities to graph entities.

Query entities: {json.dumps(query_entities)}

Available graph entities: {json.dumps(graph_entity_names)}

For each query entity, find the best matching graph entity. Consider:
- Exact matches (case-insensitive)
- Partial matches (e.g., "BERT" matches "BERT-base")
- Abbreviations (e.g., "ViT" matches "Vision Transformer")
- Synonyms or variations

Return a JSON object with:
- "matches": array of matched graph entity names (only confident matches)
- "confidence": array of confidence scores (0-1) for each match

Example:
Query entities: ["BERT", "Transformer"]
Graph entities: ["BERT-base", "BERT-large", "Transformer", "GPT"]
Response: {{"matches": ["BERT-base", "Transformer"], "confidence": [0.9, 1.0]}}

Now match the entities above:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at matching entity names, considering variations, abbreviations, and synonyms."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            matches = result.get("matches", [])
            confidences = result.get("confidence", [])
            
            # Filter by confidence threshold (0.7)
            matched = []
            for i, match in enumerate(matches):
                confidence = confidences[i] if i < len(confidences) else 1.0
                if confidence >= 0.7 and match in graph_entity_names:
                    matched.append(match)
            
            return matched
            
        except Exception as e:
            raise Exception(f"LLM matching error: {e}")
    
    def _match_entities_with_regex(self, 
                                    query_entities: List[str], 
                                    graph_entities: List[Dict]) -> List[str]:
        """
        Match query entities to graph entities using regex (fallback method)
        
        Args:
            query_entities: Entities from query
            graph_entities: Entities from graph
            
        Returns:
            List of matched entity names
        """
        matched = []
        query_entities_lower = [e.lower() for e in query_entities]
        
        for graph_entity in graph_entities:
            entity_name = graph_entity.get("name", "")
            if not entity_name:
                continue
            
            entity_lower = entity_name.lower()
            
            # Check for exact or partial match
            for query_entity in query_entities_lower:
                if query_entity in entity_lower or entity_lower in query_entity:
                    if entity_name not in matched:
                        matched.append(entity_name)
        
        return matched

