"""
Graph Traverser
Traverses Neo4j graph to find related entities and papers
"""

import os
from typing import List, Dict, Optional
from neo4j import GraphDatabase


class GraphTraverser:
    """Traverses Neo4j graph to find related entities and papers"""
    
    def __init__(self,
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize graph traverser
        
        Args:
            uri: Neo4j URI (default: from env)
            user: Neo4j username (default: from env)
            password: Neo4j password (default: from env)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Verify connection
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"Could not connect to Neo4j: {e}")
    
    def find_entities(self, query_entities: List[str], fuzzy: bool = True) -> List[Dict]:
        """
        Find entities in graph matching query entities
        
        Args:
            query_entities: List of entity names to search for
            fuzzy: If True, use case-insensitive partial matching
            
        Returns:
            List of matching entities with their properties
        """
        if not query_entities:
            return []
        
        with self.driver.session() as session:
            results = []
            for entity_name in query_entities:
                if fuzzy:
                    # Case-insensitive partial match
                    query = """
                    MATCH (e)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                       OR toLower($name) CONTAINS toLower(e.name)
                    RETURN e, labels(e) as labels
                    LIMIT 10
                    """
                else:
                    # Exact match
                    query = """
                    MATCH (e)
                    WHERE e.name = $name OR e.id = $name
                    RETURN e, labels(e) as labels
                    LIMIT 10
                    """
                
                records = session.run(query, name=entity_name)
                for record in records:
                    entity_data = dict(record["e"])
                    entity_data["labels"] = record["labels"]
                    entity_data["matched_query"] = entity_name
                    results.append(entity_data)
            
            return results
    
    def traverse_from_entities(self, entities: List[str], max_hops: int = 2) -> Dict:
        """
        Traverse graph from entities (multi-hop)
        
        Args:
            entities: List of entity names to start from
            max_hops: Maximum number of hops (1-3 recommended)
            
        Returns:
            Dictionary with:
            - entities: Related entities found
            - papers: Related papers found
            - relationships: Paths traversed
        """
        if not entities or max_hops < 1:
            return {"entities": [], "papers": [], "relationships": []}
        
        with self.driver.session() as session:
            # Find starting entities and traverse
            # Note: max_hops must be in query string (Neo4j doesn't allow params in relationship patterns)
            # Safe because max_hops is an integer we control
            query = f"""
            MATCH (start)
            WHERE start.name IN $entity_names OR start.id IN $entity_names
            MATCH path = (start)-[*1..{max_hops}]-(connected)
            WHERE connected:Paper OR connected:Model OR connected:Concept 
               OR connected:Technique OR connected:Dataset OR connected:Architecture
            RETURN DISTINCT start, connected, relationships(path) as rels, length(path) as hops
            ORDER BY hops
            LIMIT 100
            """
            
            entities_found = []
            papers_found = []
            relationships = []
            
            records = session.run(query, entity_names=entities)
            for record in records:
                start = dict(record["start"])
                connected = dict(record["connected"])
                rels = record["rels"]
                hops = record["hops"]
                
                connected_labels = list(connected.get("labels", [])) if hasattr(connected, "labels") else []
                
                if "Paper" in connected_labels:
                    papers_found.append({
                        "id": connected.get("id", ""),
                        "title": connected.get("title", ""),
                        "year": connected.get("year"),
                        "hops": hops
                    })
                else:
                    entities_found.append({
                        "name": connected.get("name", ""),
                        "type": connected_labels[0] if connected_labels else "Unknown",
                        "description": connected.get("description", ""),
                        "hops": hops
                    })
                
                # Collect relationship info
                for rel in rels:
                    rel_dict = dict(rel)
                    relationships.append({
                        "type": rel_dict.get("type", ""),
                        "source": start.get("name") or start.get("id", ""),
                        "target": connected.get("name") or connected.get("id", ""),
                        "hops": hops
                    })
            
            return {
                "entities": entities_found,
                "papers": papers_found,
                "relationships": relationships
            }
    
    def get_related_papers(self, entities: List[str]) -> List[Dict]:
        """
        Find papers connected to entities
        
        Args:
            entities: List of entity names
            
        Returns:
            List of papers with connection info
        """
        if not entities:
            return []
        
        with self.driver.session() as session:
            query = """
            MATCH (e)
            WHERE e.name IN $entity_names OR e.id IN $entity_names
            MATCH (p:Paper)-[r]-(e)
            RETURN DISTINCT p, type(r) as rel_type, e.name as entity_name
            LIMIT 50
            """
            
            papers = []
            records = session.run(query, entity_names=entities)
            for record in records:
                paper = dict(record["p"])
                papers.append({
                    "id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "relationship": record["rel_type"],
                    "entity": record["entity_name"]
                })
            
            return papers
    
    def get_entity_relationships(self, entity: str, max_hops: int = 2) -> List[Dict]:
        """
        Get relationships from a specific entity
        
        Args:
            entity: Entity name
            max_hops: Maximum traversal depth
            
        Returns:
            List of relationships
        """
        with self.driver.session() as session:
            # Note: max_hops must be in query string (Neo4j doesn't allow params in relationship patterns)
            # Safe because max_hops is an integer we control
            query = f"""
            MATCH (e)
            WHERE e.name = $entity OR e.id = $entity
            MATCH path = (e)-[*1..{max_hops}]-(connected)
            RETURN DISTINCT e, connected, relationships(path) as rels, length(path) as hops
            ORDER BY hops
            LIMIT 50
            """
            
            relationships = []
            records = session.run(query, entity=entity)
            for record in records:
                start = dict(record["e"])
                connected = dict(record["connected"])
                rels = record["rels"]
                hops = record["hops"]
                
                for rel in rels:
                    rel_dict = dict(rel)
                    relationships.append({
                        "source": start.get("name") or start.get("id", ""),
                        "target": connected.get("name") or connected.get("id", ""),
                        "type": rel_dict.get("type", ""),
                        "hops": hops
                    })
            
            return relationships
    
    def expand_from_chunks(self, chunk_ids: List[str]) -> Dict:
        """
        Get graph context for chunks (entities they mention and related nodes)
        
        Args:
            chunk_ids: List of chunk IDs from vector search results
            
        Returns:
            Dictionary with graph context for each chunk
        """
        if not chunk_ids:
            return {}
        
        with self.driver.session() as session:
            query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
            OPTIONAL MATCH (c)-[:MENTIONS]->(e)
            OPTIONAL MATCH (e)-[r*0..2]-(connected)
            RETURN c.chunk_id as chunk_id, e, r, connected
            LIMIT 200
            """
            
            chunk_context = {}
            records = session.run(query, chunk_ids=chunk_ids)
            
            for record in records:
                chunk_id = record.get("chunk_id")
                if not chunk_id:
                    continue
                
                if chunk_id not in chunk_context:
                    chunk_context[chunk_id] = {
                        "entities": [],
                        "related_nodes": [],
                        "relationships": []
                    }
                
                # Handle entity (may be None if no MENTIONS relationship)
                entity_node = record.get("e")
                if entity_node:
                    entity = dict(entity_node)
                    entity_name = entity.get("name", "")
                    if entity_name and entity_name not in [e["name"] for e in chunk_context[chunk_id]["entities"]]:
                        # Get entity type from node labels if available
                        entity_type = "Unknown"
                        if hasattr(entity_node, "__class__"):
                            # Try to get labels from Neo4j node
                            try:
                                labels = list(entity_node.labels) if hasattr(entity_node, "labels") else []
                                entity_type = labels[0] if labels else "Unknown"
                            except:
                                pass
                        
                        chunk_context[chunk_id]["entities"].append({
                            "name": entity_name,
                            "type": entity_type,
                            "description": entity.get("description", "")
                        })
                        
                        # Handle related nodes (may be None)
                        connected_node = record.get("connected")
                        if connected_node:
                            connected = dict(connected_node)
                            connected_name = connected.get("name") or connected.get("id", "")
                            if connected_name:
                                chunk_context[chunk_id]["related_nodes"].append({
                                    "name": connected_name,
                                    "type": "Unknown"  # Simplified for now
                                })
                            
                            # Handle relationships (may be None or empty)
                            rels = record.get("r")
                            if rels:
                                if not isinstance(rels, list):
                                    rels = [rels]
                                for rel in rels:
                                    if rel:
                                        try:
                                            rel_dict = dict(rel)
                                            rel_type = rel.type if hasattr(rel, "type") else rel_dict.get("type", "")
                                            chunk_context[chunk_id]["relationships"].append({
                                                "type": rel_type,
                                                "source": entity_name,
                                                "target": connected_name
                                            })
                                        except:
                                            pass
            
            return chunk_context
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()

