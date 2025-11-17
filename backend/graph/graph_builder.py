"""
Knowledge Graph Builder
Builds and manages the knowledge graph using Neo4j
"""

import os
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
from neo4j import GraphDatabase


class GraphBuilder:
    """Builds and manages knowledge graph in Neo4j"""
    
    def __init__(self, 
                 uri: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize graph builder with Neo4j connection
        
        Args:
            uri: Neo4j URI (default: from env or bolt://localhost:7687)
            user: Neo4j username (default: from env or neo4j)
            password: Neo4j password (default: from env or neo4j)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Verify connection
        try:
            self.driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Neo4j: {e}")
            print("  Using in-memory fallback (NetworkX)")
            self.driver = None
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize NetworkX fallback if Neo4j unavailable"""
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        self.use_neo4j = False
    
    def add_paper(self, metadata: Dict):
        """
        Add a paper as a node in Neo4j
        
        Args:
            metadata: Paper metadata (title, authors, year, etc.)
        """
        paper_id = metadata.get("filename", "unknown")
        
        if self.driver:
            # Use Neo4j
            with self.driver.session() as session:
                session.execute_write(
                    self._create_paper_node,
                    paper_id,
                    metadata
                )
        else:
            # Fallback to NetworkX
            self.graph.add_node(
                paper_id,
                type="Paper",
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                year=metadata.get("year"),
                venue=metadata.get("venue", ""),
                arxiv_id=metadata.get("arxiv_id", ""),
                **{k: v for k, v in metadata.items() if k not in ["title", "authors", "year", "venue", "arxiv_id", "filename"]}
            )
    
    @staticmethod
    def _create_paper_node(tx, paper_id: str, metadata: Dict):
        """Create paper node in Neo4j"""
        query = """
        MERGE (p:Paper {id: $paper_id})
        SET p.title = $title,
            p.authors = $authors,
            p.year = $year,
            p.venue = $venue,
            p.arxiv_id = $arxiv_id,
            p.created_at = datetime()
        RETURN p
        """
        tx.run(query, 
               paper_id=paper_id,
               title=metadata.get("title", ""),
               authors=metadata.get("authors", []),
               year=metadata.get("year"),
               venue=metadata.get("venue", ""),
               arxiv_id=metadata.get("arxiv_id", ""))
    
    def add_entity(self, entity: Dict):
        """
        Add an entity as a node in Neo4j
        
        Args:
            entity: Entity dict with type, name, description
        """
        entity_name = entity.get("name", "")
        if not entity_name:
            return
        
        if self.driver:
            # Use Neo4j
            try:
                with self.driver.session() as session:
                    session.execute_write(
                        self._create_entity_node,
                        entity_name,
                        entity
                    )
            except Exception as e:
                # Log error but don't fail completely - continue with next entity
                print(f"  ⚠ Failed to add entity '{entity_name}': {e}")
                raise  # Re-raise to let caller handle
        else:
            # Fallback to NetworkX
            if entity_name in self.graph:
                existing_data = self.graph.nodes[entity_name]
                existing_data.update({
                    "description": entity.get("description", existing_data.get("description", "")),
                    "type": entity.get("type", existing_data.get("type", "Concept"))
                })
            else:
                self.graph.add_node(
                    entity_name,
                    type=entity.get("type", "Concept"),
                    description=entity.get("description", ""),
                    extracted_at=datetime.now().isoformat()
                )
    
    @staticmethod
    def _create_entity_node(tx, entity_name: str, entity: Dict):
        """Create entity node in Neo4j"""
        entity_type = entity.get("type", "Concept")
        # Sanitize entity type: replace spaces and invalid chars with underscores
        # Neo4j labels cannot contain spaces unless escaped with backticks
        entity_type_clean = entity_type.replace(" ", "_").replace("-", "_")
        # Remove any other invalid characters (keep only alphanumeric and underscore)
        entity_type_clean = re.sub(r'[^a-zA-Z0-9_]', '_', entity_type_clean)
        # Ensure it starts with a letter
        if entity_type_clean and not entity_type_clean[0].isalpha():
            entity_type_clean = "Entity_" + entity_type_clean
        
        query = f"""
        MERGE (e:{entity_type_clean} {{name: $name}})
        SET e.description = $description,
            e.extracted_at = datetime()
        RETURN e
        """
        tx.run(query,
               name=entity_name,
               description=entity.get("description", ""))
    
    def add_entities_batch(self, entities: List[Dict], batch_size: int = 100):
        """
        Add multiple entities in batches (much faster than one-by-one)
        
        Args:
            entities: List of entity dictionaries
            batch_size: Number of entities per batch (default: 100)
            
        Returns:
            int: Number of entities successfully added
        """
        if not entities:
            return 0
        
        if not self.driver:
            # Fallback to NetworkX
            for entity in entities:
                self.add_entity(entity)
            return len(entities)
        
        added = 0
        
        # Group entities by type for more efficient batching
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get("type", "Concept")
            entity_type_clean = entity_type.replace(" ", "_").replace("-", "_")
            entity_type_clean = re.sub(r'[^a-zA-Z0-9_]', '_', entity_type_clean)
            if entity_type_clean and not entity_type_clean[0].isalpha():
                entity_type_clean = "Entity_" + entity_type_clean
            
            if entity_type_clean not in entities_by_type:
                entities_by_type[entity_type_clean] = []
            entities_by_type[entity_type_clean].append(entity)
        
        # Process each type group in batches
        total_batches = sum((len(type_entities) + batch_size - 1) // batch_size 
                           for type_entities in entities_by_type.values())
        batch_num = 0
        
        with self.driver.session() as session:
            for entity_type_clean, type_entities in entities_by_type.items():
                for batch_start in range(0, len(type_entities), batch_size):
                    batch = type_entities[batch_start:batch_start + batch_size]
                    batch_num += 1
                    try:
                        result = session.execute_write(
                            self._create_entities_batch,
                            batch,
                            entity_type_clean
                        )
                        added += result
                        if batch_num % 10 == 0 or batch_num == total_batches:
                            print(f"    Progress: {batch_num}/{total_batches} batches ({added} entities added)...")
                    except Exception as e:
                        if batch_num <= 5:  # Only show first 5 errors
                            print(f"    ⚠ Batch {batch_num} failed: {e}")
        
        return added
    
    @staticmethod
    def _create_entities_batch(tx, entities: List[Dict], entity_type: str):
        """Create multiple entity nodes in one transaction using UNWIND"""
        # Prepare data for UNWIND
        entity_data = []
        for entity in entities:
            entity_name = entity.get("name", "")
            if entity_name:
                entity_data.append({
                    "name": entity_name,
                    "description": entity.get("description", "") or ""
                })
        
        if not entity_data:
            return 0
        
        query = f"""
        UNWIND $entities AS entity
        MERGE (e:{entity_type} {{name: entity.name}})
        SET e.description = entity.description,
            e.extracted_at = datetime()
        RETURN count(e) as count
        """
        result = tx.run(query, entities=entity_data).single()
        return result["count"] if result else 0
    
    def add_relationship(self, relationship: Dict):
        """
        Add a relationship as an edge in Neo4j
        
        Args:
            relationship: Relationship dict with type, source, target, description
        """
        source = relationship.get("source", "")
        target = relationship.get("target", "")
        rel_type = relationship.get("type", "RELATED_TO")
        
        if not source or not target:
            return
        
        if self.driver:
            # Use Neo4j
            with self.driver.session() as session:
                session.execute_write(
                    self._create_relationship,
                    source,
                    target,
                    rel_type,
                    relationship.get("description", "")
                )
        else:
            # Fallback to NetworkX
            if source not in self.graph:
                self.graph.add_node(source, type="Unknown")
            if target not in self.graph:
                self.graph.add_node(target, type="Unknown")
            
            self.graph.add_edge(
                source,
                target,
                type=rel_type,
                description=relationship.get("description", ""),
                extracted_at=datetime.now().isoformat()
            )
    
    @staticmethod
    def _create_relationship(tx, source: str, target: str, rel_type: str, description: str):
        """Create relationship in Neo4j"""
        # Convert relationship type to valid Cypher format
        rel_type_clean = rel_type.upper().replace(" ", "_")
        
        query = f"""
        MATCH (a)
        WHERE a.id = $source OR a.name = $source
        MATCH (b)
        WHERE b.id = $target OR b.name = $target
        MERGE (a)-[r:{rel_type_clean}]->(b)
        SET r.description = $description,
            r.created_at = datetime()
        RETURN r
        """
        tx.run(query,
               source=source,
               target=target,
               description=description)
    
    def add_relationships_batch(self, relationships: List[Dict], batch_size: int = 100):
        """
        Add multiple relationships in batches (much faster than one-by-one)
        
        Args:
            relationships: List of relationship dictionaries
            batch_size: Number of relationships per batch (default: 100)
            
        Returns:
            int: Number of relationships successfully added
        """
        if not relationships:
            return 0
        
        if not self.driver:
            # Fallback to NetworkX
            for rel in relationships:
                self.add_relationship(rel)
            return len(relationships)
        
        added = 0
        
        # Group relationships by type for more efficient batching
        rels_by_type = {}
        for rel in relationships:
            rel_type = rel.get("type", "RELATED_TO")
            rel_type_clean = rel_type.upper().replace(" ", "_")
            
            if rel_type_clean not in rels_by_type:
                rels_by_type[rel_type_clean] = []
            rels_by_type[rel_type_clean].append(rel)
        
        # Process each type group in batches
        total_batches = sum((len(type_rels) + batch_size - 1) // batch_size 
                           for type_rels in rels_by_type.values())
        batch_num = 0
        
        with self.driver.session() as session:
            for rel_type_clean, type_rels in rels_by_type.items():
                for batch_start in range(0, len(type_rels), batch_size):
                    batch = type_rels[batch_start:batch_start + batch_size]
                    batch_num += 1
                    try:
                        result = session.execute_write(
                            self._create_relationships_batch,
                            batch,
                            rel_type_clean
                        )
                        added += result
                        if batch_num % 10 == 0 or batch_num == total_batches:
                            print(f"    Progress: {batch_num}/{total_batches} batches ({added} relationships added)...")
                    except Exception as e:
                        if batch_num <= 5:  # Only show first 5 errors
                            print(f"    ⚠ Relationship batch {batch_num} failed: {e}")
        
        return added
    
    @staticmethod
    def _create_relationships_batch(tx, relationships: List[Dict], rel_type: str):
        """Create multiple relationships in one transaction using UNWIND"""
        # Prepare data for UNWIND
        rel_data = []
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            description = rel.get("description", "") or ""
            
            if source and target:
                rel_data.append({
                    "source": source,
                    "target": target,
                    "description": description
                })
        
        if not rel_data:
            return 0
        
        query = f"""
        UNWIND $relationships AS rel
        MATCH (a)
        WHERE a.id = rel.source OR a.name = rel.source
        MATCH (b)
        WHERE b.id = rel.target OR b.name = rel.target
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.description = rel.description,
            r.created_at = datetime()
        RETURN count(r) as count
        """
        result = tx.run(query, relationships=rel_data).single()
        return result["count"] if result else 0
    
    def add_temporal_relationships(self, papers: List[Dict]):
        """
        Add temporal relationships between papers (by year) in Neo4j
        
        Args:
            papers: List of paper dictionaries
        """
        # Sort papers by year
        sorted_papers = sorted(
            [p for p in papers if p["metadata"].get("year")],
            key=lambda x: x["metadata"]["year"]
        )
        
        # Connect papers chronologically
        for i in range(len(sorted_papers) - 1):
            current = sorted_papers[i]["metadata"].get("filename", "")
            next_paper = sorted_papers[i + 1]["metadata"].get("filename", "")
            
            if current and next_paper:
                if self.driver:
                    with self.driver.session() as session:
                        session.execute_write(
                            self._create_temporal_relationship,
                            current,
                            next_paper
                        )
                else:
                    self.graph.add_edge(
                        current,
                        next_paper,
                        type="PRECEDES",
                        description="Published before"
                    )
    
    @staticmethod
    def _create_temporal_relationship(tx, source: str, target: str):
        """Create temporal relationship in Neo4j"""
        query = """
        MATCH (a:Paper {id: $source})
        MATCH (b:Paper {id: $target})
        MERGE (a)-[r:PRECEDES]->(b)
        SET r.description = "Published before",
            r.created_at = datetime()
        RETURN r
        """
        tx.run(query, source=source, target=target)
    
    def get_graph_info(self) -> Dict:
        """Get information about the graph from Neo4j"""
        if self.driver:
            with self.driver.session() as session:
                result = session.execute_read(self._get_neo4j_info)
                return result
        else:
            # Fallback to NetworkX
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "node_types": self._get_node_type_counts(),
                "edge_types": self._get_edge_type_counts()
            }
    
    @staticmethod
    def _get_neo4j_info(tx) -> Dict:
        """Get graph statistics from Neo4j"""
        # Count nodes
        node_query = "MATCH (n) RETURN count(n) as count"
        node_result = tx.run(node_query).single()
        node_count = node_result["count"] if node_result else 0
        
        # Count edges
        edge_query = "MATCH ()-[r]->() RETURN count(r) as count"
        edge_result = tx.run(edge_query).single()
        edge_count = edge_result["count"] if edge_result else 0
        
        # Count nodes by label
        label_query = "CALL db.labels() YIELD label CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value RETURN label, value.count as count"
        # Simplified version
        node_types = {}
        type_query = "MATCH (n) RETURN labels(n)[0] as label, count(n) as count"
        for record in tx.run(type_query):
            node_types[record["label"]] = record["count"]
        
        # Count relationships by type
        rel_types = {}
        rel_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
        for record in tx.run(rel_query):
            rel_types[record["rel_type"]] = record["count"]
        
        return {
            "nodes": node_count,
            "edges": edge_count,
            "node_types": node_types,
            "edge_types": rel_types
        }
    
    def _get_node_type_counts(self) -> Dict:
        """Count nodes by type (NetworkX fallback)"""
        counts = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _get_edge_type_counts(self) -> Dict:
        """Count edges by type (NetworkX fallback)"""
        counts = {}
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get("type", "Unknown")
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    def save_graph(self, filepath: str):
        """
        Export graph to JSON file (for backup/visualization)
        Note: Graph is stored in Neo4j, this is just an export
        
        Args:
            filepath: Path to save graph export
        """
        if self.driver:
            # Export from Neo4j
            with self.driver.session() as session:
                result = session.execute_read(self._export_neo4j_graph)
                graph_data = result
        else:
            # Export from NetworkX
            graph_data = {
                "nodes": [
                    {"id": node, **data}
                    for node, data in self.graph.nodes(data=True)
                ],
                "edges": [
                    {"source": source, "target": target, **data}
                    for source, target, data in self.graph.edges(data=True)
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    @staticmethod
    def _export_neo4j_graph(tx) -> Dict:
        """Export Neo4j graph to JSON format"""
        def convert_value(value):
            """Convert Neo4j types to JSON-serializable types"""
            from neo4j.time import DateTime
            if isinstance(value, DateTime):
                return value.iso_format()
            elif isinstance(value, (list, tuple)):
                return [convert_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            return value
        
        # Get all nodes
        nodes_query = "MATCH (n) RETURN n, labels(n) as labels"
        nodes = []
        for record in tx.run(nodes_query):
            node = dict(record["n"])
            node = convert_value(node)  # Convert DateTime objects
            node["labels"] = record["labels"]
            nodes.append(node)
        
        # Get all relationships
        edges_query = "MATCH (a)-[r]->(b) RETURN a, type(r) as type, r, b"
        edges = []
        for record in tx.run(edges_query):
            # Get node IDs
            source_id = record["a"].get("id") or record["a"].get("name", "")
            target_id = record["b"].get("id") or record["b"].get("name", "")
            rel_data = dict(record["r"])
            rel_data = convert_value(rel_data)  # Convert DateTime objects
            rel_data["type"] = record["type"]
            edges.append({
                "source": source_id,
                "target": target_id,
                **rel_data
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def add_chunk(self, chunk_id: str, metadata: Dict):
        """
        Add a chunk as a node in Neo4j
        
        Args:
            chunk_id: Unique chunk identifier (matches chunk_id in metadata)
            metadata: Chunk metadata (title, filename, chunk_index, etc.)
        """
        paper_id = metadata.get("filename", "unknown")
        
        if self.driver:
            # Use Neo4j
            with self.driver.session() as session:
                session.execute_write(
                    self._create_chunk_node,
                    chunk_id,
                    paper_id,
                    metadata
                )
        else:
            # Fallback to NetworkX
            self.graph.add_node(
                chunk_id,
                type="Chunk",
                paper_id=paper_id,
                chunk_index=metadata.get("chunk_index", 0),
                title=metadata.get("title", ""),
                **{k: v for k, v in metadata.items() if k not in ["title", "filename", "chunk_index"]}
            )
    
    @staticmethod
    def _create_chunk_node(tx, chunk_id: str, paper_id: str, metadata: Dict):
        """Create chunk node in Neo4j"""
        query = """
        MATCH (p:Paper {id: $paper_id})
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.chunk_index = $chunk_index,
            c.title = $title,
            c.total_chunks = $total_chunks,
            c.created_at = datetime()
        MERGE (c)-[:FROM_PAPER]->(p)
        RETURN c
        """
        tx.run(query,
               chunk_id=chunk_id,
               paper_id=paper_id,
               chunk_index=metadata.get("chunk_index", 0),
               title=metadata.get("title", ""),
               total_chunks=metadata.get("total_chunks", 0))
    
    def add_chunks_batch(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add multiple chunks in batches (much faster than one-by-one)
        
        Args:
            chunks: List of dicts with 'chunk_id' and 'metadata' keys
            batch_size: Number of chunks per batch (default: 100)
            
        Returns:
            int: Number of chunks successfully added
        """
        if not chunks:
            return 0
        
        if not self.driver:
            # Fallback to NetworkX
            for chunk in chunks:
                self.add_chunk(chunk["chunk_id"], chunk["metadata"])
            return len(chunks)
        
        added = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        with self.driver.session() as session:
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start:batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                try:
                    result = session.execute_write(
                        self._create_chunks_batch,
                        batch
                    )
                    added += result
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        print(f"    Progress: {batch_num}/{total_batches} batches ({added} chunks added)...")
                except Exception as e:
                    if batch_num <= 5:
                        print(f"    ⚠ Chunk batch {batch_num} failed: {e}")
        
        return added
    
    @staticmethod
    def _create_chunks_batch(tx, chunks: List[Dict]):
        """Create multiple chunk nodes in one transaction using UNWIND"""
        chunk_data = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            metadata = chunk.get("metadata", {})
            paper_id = metadata.get("filename", "unknown")
            
            if chunk_id:
                chunk_data.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "title": metadata.get("title", "") or "",
                    "total_chunks": metadata.get("total_chunks", 0)
                })
        
        if not chunk_data:
            return 0
        
        query = """
        UNWIND $chunks AS chunk
        MATCH (p:Paper {id: chunk.paper_id})
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.chunk_index = chunk.chunk_index,
            c.title = chunk.title,
            c.total_chunks = chunk.total_chunks,
            c.created_at = datetime()
        MERGE (c)-[:FROM_PAPER]->(p)
        RETURN count(c) as count
        """
        result = tx.run(query, chunks=chunk_data).single()
        return result["count"] if result else 0
    
    def link_chunk_to_entity(self, chunk_id: str, entity_name: str):
        """
        Link a chunk to an entity it mentions
        
        Args:
            chunk_id: Chunk identifier
            entity_name: Entity name
            
        Returns:
            bool: True if link was created, False if chunk or entity doesn't exist
        """
        if not chunk_id or not entity_name:
            return False
        
        if self.driver:
            # Use Neo4j
            try:
                with self.driver.session() as session:
                    result = session.execute_write(
                        self._create_chunk_entity_link,
                        chunk_id,
                        entity_name
                    )
                    return result
            except Exception as e:
                # Silently fail - chunk or entity might not exist
                return False
        else:
            # Fallback to NetworkX
            if chunk_id not in self.graph:
                self.graph.add_node(chunk_id, type="Chunk")
            if entity_name not in self.graph:
                self.graph.add_node(entity_name, type="Unknown")
            
            self.graph.add_edge(
                chunk_id,
                entity_name,
                type="MENTIONS",
                created_at=datetime.now().isoformat()
            )
            return True
    
    @staticmethod
    def _create_chunk_entity_link(tx, chunk_id: str, entity_name: str):
        """
        Create MENTIONS relationship between chunk and entity
        First verifies that both chunk and entity exist
        
        Returns:
            bool: True if link was created, False if chunk or entity doesn't exist
        """
        # First check if chunk exists
        chunk_check = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        RETURN c
        LIMIT 1
        """
        chunk_result = tx.run(chunk_check, chunk_id=chunk_id).single()
        if not chunk_result:
            return False  # Chunk doesn't exist
        
        # Check if entity exists
        entity_check = """
        MATCH (e)
        WHERE e.name = $entity_name OR e.id = $entity_name
        RETURN e
        LIMIT 1
        """
        entity_result = tx.run(entity_check, entity_name=entity_name).single()
        if not entity_result:
            return False  # Entity doesn't exist
        
        # Both exist, create the relationship
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        MATCH (e)
        WHERE e.name = $entity_name OR e.id = $entity_name
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.created_at = datetime()
        RETURN r
        """
        result = tx.run(query, chunk_id=chunk_id, entity_name=entity_name).single()
        return result is not None
    
    def link_chunks_to_entities_batch(self, chunk_entity_pairs: List[Dict], batch_size: int = 100):
        """
        Link multiple chunks to entities in batches (much faster than one-by-one)
        
        Args:
            chunk_entity_pairs: List of dicts with 'chunk_id' and 'entity_name' keys
            batch_size: Number of links per batch (default: 100)
            
        Returns:
            int: Number of links successfully created
        """
        if not chunk_entity_pairs:
            return 0
        
        if not self.driver:
            # Fallback to NetworkX
            for pair in chunk_entity_pairs:
                self.link_chunk_to_entity(pair["chunk_id"], pair["entity_name"])
            return len(chunk_entity_pairs)
        
        added = 0
        total_batches = (len(chunk_entity_pairs) + batch_size - 1) // batch_size
        
        with self.driver.session() as session:
            for batch_start in range(0, len(chunk_entity_pairs), batch_size):
                batch = chunk_entity_pairs[batch_start:batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                try:
                    result = session.execute_write(
                        self._create_chunk_entity_links_batch,
                        batch
                    )
                    added += result
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        print(f"    Progress: {batch_num}/{total_batches} batches ({added} links created)...")
                except Exception as e:
                    if batch_num <= 5:
                        print(f"    ⚠ Link batch {batch_num} failed: {e}")
        
        return added
    
    @staticmethod
    def _create_chunk_entity_links_batch(tx, chunk_entity_pairs: List[Dict]):
        """Create multiple chunk-entity links in one transaction using UNWIND"""
        link_data = []
        for pair in chunk_entity_pairs:
            chunk_id = pair.get("chunk_id", "")
            entity_name = pair.get("entity_name", "")
            if chunk_id and entity_name:
                link_data.append({
                    "chunk_id": chunk_id,
                    "entity_name": entity_name
                })
        
        if not link_data:
            return 0
        
        query = """
        UNWIND $links AS link
        MATCH (c:Chunk {chunk_id: link.chunk_id})
        MATCH (e)
        WHERE e.name = link.entity_name OR e.id = link.entity_name
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.created_at = datetime()
        RETURN count(r) as count
        """
        result = tx.run(query, links=link_data).single()
        return result["count"] if result else 0
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()

