# Neo4j Cypher Queries for Knowledge Graph

## Basic Queries

### View All Nodes
```cypher
MATCH (n) RETURN n LIMIT 50
```

### View All Papers
```cypher
MATCH (p:Paper) RETURN p.title, p.year, p.authors ORDER BY p.year
```

### View All Relationships
```cypher
MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 50
```

## Multi-Hop Queries

### Find Papers Connected Through Concepts
```cypher
MATCH (p1:Paper)-[:INTRODUCES]->(c:Concept)<-[:USES]-(p2:Paper)
RETURN p1.title, c.name, p2.title
```

### Find 2-Hop Connections
```cypher
MATCH path = (a)-[*2]->(b)
WHERE a:Paper AND b:Paper
RETURN path LIMIT 10
```

### Find All Paths Between Two Papers
```cypher
MATCH path = (p1:Paper {title: "Attention Is All You Need"})-[*..5]-(p2:Paper {title: "BERT"})
RETURN path
```

## Temporal Queries

### Papers by Year
```cypher
MATCH (p:Paper)
RETURN p.year, collect(p.title) as papers
ORDER BY p.year
```

### Evolution of Concepts Over Time
```cypher
MATCH (p:Paper)-[:INTRODUCES]->(c:Concept)
RETURN c.name, collect(p.year) as years_introduced
ORDER BY min(p.year)
```

### Papers Published Before/After a Year
```cypher
MATCH (p:Paper)
WHERE p.year < 2020
RETURN p.title, p.year
ORDER BY p.year
```

## Entity Queries

### Find All Concepts
```cypher
MATCH (c:Concept)
RETURN c.name, c.description
LIMIT 20
```

### Find Concepts Used by Multiple Papers
```cypher
MATCH (p:Paper)-[:USES]->(c:Concept)
WITH c, count(p) as paper_count
WHERE paper_count > 1
RETURN c.name, paper_count
ORDER BY paper_count DESC
```

### Find Related Concepts
```cypher
MATCH (c1:Concept)-[r]->(c2:Concept)
RETURN c1.name, type(r), c2.name
LIMIT 20
```

## Advanced Queries

### Graph Statistics
```cypher
MATCH (n)
RETURN labels(n)[0] as type, count(n) as count
ORDER BY count DESC
```

### Most Connected Nodes
```cypher
MATCH (n)-[r]-()
RETURN n, count(r) as connections
ORDER BY connections DESC
LIMIT 10
```

### Find Clusters (Papers with Common Concepts)
```cypher
MATCH (p1:Paper)-[:USES]->(c:Concept)<-[:USES]-(p2:Paper)
WHERE p1 <> p2
RETURN p1.title, p2.title, count(c) as common_concepts
ORDER BY common_concepts DESC
LIMIT 10
```

## Visualization Queries

### Full Graph (use carefully - may be large)
```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
```

### Papers and Their Concepts
```cypher
MATCH (p:Paper)-[r]->(c)
RETURN p, r, c
```

### Temporal Chain of Papers
```cypher
MATCH path = (p1:Paper)-[:PRECEDES*]->(p2:Paper)
RETURN path
LIMIT 10
```

