"""
Response Generator
Uses LLM to generate natural language answers from retrieved results
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI


class ResponseGenerator:
    """Generates natural language responses from retrieved results using LLM"""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1000):
        """
        Initialize response generator
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            api_key: OpenAI API key (or from env)
            temperature: Temperature for generation (default: 0.3 for focused answers)
            max_tokens: Maximum tokens in response (default: 1000)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self,
                 query: str,
                 results: List[Dict],
                 graph_context: Optional[Dict] = None,
                 temporal_info: Optional[Dict] = None,
                 include_sources: bool = True) -> Dict:
        """
        Generate natural language response from retrieved results
        
        Args:
            query: Original user query
            results: List of retrieved results (chunks with text, metadata, scores)
            graph_context: Graph context information (optional)
            temporal_info: Temporal information if temporal query (optional)
            include_sources: Whether to include source citations in response
            
        Returns:
            Dictionary with:
            - answer: Generated natural language answer
            - sources: List of sources used (if include_sources=True)
            - confidence: Confidence score (0-1)
        """
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your query.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Prepare context from results
        context_text = self._prepare_context(results, graph_context, temporal_info)
        
        # Generate prompt
        prompt = self._create_generation_prompt(query, context_text, temporal_info)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract sources if requested
            sources = []
            if include_sources:
                sources = self._extract_sources(results)
            
            # Calculate confidence (based on result scores)
            confidence = self._calculate_confidence(results)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM"""
        return """You are an expert research assistant that answers questions about research papers and technical concepts.

CRITICAL: You MUST strictly generate content ONLY from the retrieved context provided. DO NOT make up, invent, or hallucinate any information that is not explicitly present in the context.

Your task is to synthesize information from retrieved document chunks and knowledge graph context to provide accurate, comprehensive answers.

Guidelines:
- STRICTLY base your answer ONLY on the provided context - do not add any information not in the context
- If information is not in the context, explicitly state "This information is not available in the retrieved context"
- DO NOT make assumptions or infer information beyond what is explicitly stated
- DO NOT use your general knowledge - only use information from the provided context
- Be precise and cite specific details when available
- For temporal queries, organize information chronologically based on what's in the context
- For relationship queries, explain how concepts connect based on the graph context provided
- Use clear, academic language appropriate for research questions
- If multiple perspectives exist in the context, mention them
- Include relevant technical details and terminology ONLY if they appear in the context"""
    
    def _prepare_context(self,
                        results: List[Dict],
                        graph_context: Optional[Dict],
                        temporal_info: Optional[Dict]) -> str:
        """
        Prepare context text from results and graph information
        
        Args:
            results: Retrieved results
            graph_context: Graph context information
            temporal_info: Temporal information
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add temporal context if available
        if temporal_info and temporal_info.get("is_temporal"):
            temporal_desc = []
            if temporal_info.get("years"):
                temporal_desc.append(f"years: {', '.join(map(str, temporal_info['years']))}")
            if temporal_info.get("year_ranges"):
                ranges = [f"{start}-{end}" for start, end in temporal_info["year_ranges"]]
                temporal_desc.append(f"ranges: {', '.join(ranges)}")
            if temporal_desc:
                context_parts.append(f"Temporal Context: {', '.join(temporal_desc)}")
        
        # Add retrieved chunks
        context_parts.append("\n=== Retrieved Document Chunks ===\n")
        for i, result in enumerate(results[:10], 1):  # Limit to top 10 for context
            metadata = result.get("metadata", {})
            text = result.get("text", "")
            score = result.get("score", 0.0)
            
            paper_title = metadata.get("title", "Unknown Paper")
            year = metadata.get("year", "Unknown")
            chunk_index = metadata.get("chunk_index", "")
            
            context_parts.append(f"[Chunk {i}]")
            context_parts.append(f"Source: {paper_title} ({year})")
            if chunk_index:
                context_parts.append(f"Section: Chunk {chunk_index}")
            context_parts.append(f"Relevance Score: {score:.3f}")
            context_parts.append(f"Content:\n{text}\n")
        
        # Add graph context if available
        if graph_context:
            context_parts.append("\n=== Knowledge Graph Context ===\n")
            
            # Collect entities from graph context
            all_entities = set()
            relationships = []
            
            for chunk_id, ctx in graph_context.items():
                entities = ctx.get("entities", [])
                for entity in entities:
                    entity_name = entity.get("name", "")
                    if entity_name:
                        all_entities.add(entity_name)
                
                rels = ctx.get("relationships", [])
                relationships.extend(rels)
            
            if all_entities:
                context_parts.append(f"Related Entities: {', '.join(sorted(list(all_entities))[:20])}")
            
            if relationships:
                # Show sample relationships
                sample_rels = relationships[:10]
                context_parts.append("\nKey Relationships:")
                for rel in sample_rels:
                    rel_type = rel.get("type", "")
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    if source and target:
                        context_parts.append(f"  - {source} --[{rel_type}]--> {target}")
        
        return "\n".join(context_parts)
    
    def _create_generation_prompt(self,
                                  query: str,
                                  context_text: str,
                                  temporal_info: Optional[Dict]) -> str:
        """
        Create prompt for response generation
        
        Args:
            query: User query
            context_text: Prepared context text
            temporal_info: Temporal information
            
        Returns:
            Complete prompt for LLM
        """
        prompt_parts = [
            f"Question: {query}\n",
            context_text,
            "\n=== CRITICAL INSTRUCTIONS ===\n",
            "STRICTLY generate content ONLY from the retrieved context provided above.",
            "DO NOT make up, invent, or hallucinate any information that is not explicitly present in the context.",
            "If the context does not contain information needed to answer the question, explicitly state that the information is not available.",
            "\n=== Answer Guidelines ===\n",
            "Based STRICTLY on the retrieved context above, provide a comprehensive answer to the question."
        ]
        
        # Add temporal-specific instructions
        if temporal_info and temporal_info.get("is_temporal"):
            prompt_parts.append("\nNote: This is a temporal query. Organize your answer chronologically based ONLY on the information in the context provided.")
        
        prompt_parts.append("\nProvide a clear, well-structured answer that:")
        prompt_parts.append("1. Directly addresses the question using ONLY information from the provided context")
        prompt_parts.append("2. Synthesizes information from multiple sources in the context when relevant")
        prompt_parts.append("3. Cites specific papers or concepts from the context when mentioning them")
        prompt_parts.append("4. Explains relationships between concepts ONLY if they appear in the graph context provided")
        prompt_parts.append("5. Explicitly acknowledges when information is incomplete or not available in the context")
        prompt_parts.append("\nREMINDER: Do not add any information, facts, or details that are not explicitly present in the retrieved context above.")
        
        return "\n".join(prompt_parts)
    
    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Extract source information from results
        
        Args:
            results: Retrieved results
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_papers = set()
        
        for result in results:
            metadata = result.get("metadata", {})
            paper_title = metadata.get("title", "")
            paper_id = metadata.get("filename", "")
            year = metadata.get("year")
            score = result.get("score", 0.0)
            
            # Deduplicate by paper
            paper_key = paper_title or paper_id
            if paper_key and paper_key not in seen_papers:
                seen_papers.add(paper_key)
                sources.append({
                    "title": paper_title,
                    "id": paper_id,
                    "year": year,
                    "relevance_score": score
                })
        
        # Sort by relevance score
        sources.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        return sources
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """
        Calculate confidence score based on result quality
        
        Args:
            results: Retrieved results with scores
            
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
        
        # Average of top 3 scores
        top_scores = [r.get("score", 0.0) for r in results[:3]]
        if top_scores:
            avg_score = sum(top_scores) / len(top_scores)
            # Normalize to 0-1 (assuming scores are already in that range)
            return min(1.0, max(0.0, avg_score))
        
        return 0.5  # Default medium confidence
    
    def generate_with_evolution_timeline(self,
                                          query: str,
                                          results: List[Dict],
                                          graph_context: Optional[Dict] = None) -> Dict:
        """
        Generate response with evolution timeline for temporal queries
        
        Args:
            query: User query
            results: Retrieved results
            graph_context: Graph context
            
        Returns:
            Dictionary with answer and timeline
        """
        # Group results by year
        results_by_year = {}
        for result in results:
            year = result.get("metadata", {}).get("year")
            if year:
                if year not in results_by_year:
                    results_by_year[year] = []
                results_by_year[year].append(result)
        
        # Sort by year
        sorted_years = sorted(results_by_year.keys())
        
        # Generate timeline sections
        timeline_parts = []
        for year in sorted_years:
            year_results = results_by_year[year]
            context_text = self._prepare_context(year_results, graph_context, None)
            
            prompt = f"""STRICTLY use ONLY the following information from {year} to provide a brief summary:

{context_text}

CRITICAL: Generate content ONLY from the context above. DO NOT make up or invent any information.

Provide 2-3 sentences summarizing the key developments in {year} based STRICTLY on the context provided above."""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant summarizing technical developments."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=200
                )
                
                timeline_parts.append({
                    "year": year,
                    "summary": response.choices[0].message.content.strip()
                })
            except Exception as e:
                timeline_parts.append({
                    "year": year,
                    "summary": f"Error generating summary: {str(e)}"
                })
        
        # Generate overall answer
        overall_answer = self.generate(query, results, graph_context, include_sources=True)
        
        return {
            **overall_answer,
            "timeline": timeline_parts
        }

