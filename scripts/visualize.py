#!/usr/bin/env python3
"""
Visualize Knowledge Graph
Creates interactive visualization of the Neo4j knowledge graph
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from pyvis.network import Network


def get_neo4j_connection():
    """Get Neo4j connection"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"❌ Could not connect to Neo4j: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return None


def fetch_graph_data(driver):
    """Fetch all nodes and relationships from Neo4j"""
    with driver.session() as session:
        # Get all nodes
        nodes_query = """
        MATCH (n)
        RETURN n, labels(n) as labels
        """
        nodes = []
        for record in session.run(nodes_query):
            node_data = dict(record["n"])
            labels = record["labels"]
            node_data["labels"] = labels
            nodes.append(node_data)
        
        # Get all relationships
        edges_query = """
        MATCH (a)-[r]->(b)
        RETURN a, type(r) as rel_type, r, b
        """
        edges = []
        for record in session.run(edges_query):
            source = dict(record["a"])
            target = dict(record["b"])
            rel_data = dict(record["r"])
            rel_type = record["rel_type"]
            
            # Get node identifiers
            source_id = source.get("id") or source.get("name") or str(source)
            target_id = target.get("id") or target.get("name") or str(target)
            
            edges.append({
                "source": source_id,
                "target": target_id,
                "type": rel_type,
                **rel_data
            })
        
        return nodes, edges


def create_pyvis_visualization(nodes, edges, output_file="graph_visualization.html"):
    """Create interactive HTML visualization using pyvis"""
    print(f"\nCreating interactive visualization...")
    
    # Create network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True
    )
    
    # Color mapping for node types
    color_map = {
        "Paper": "#FF6B6B",
        "Concept": "#4ECDC4",
        "Model": "#45B7D1",
        "Technique": "#96CEB4",
        "Dataset": "#FFEAA7",
        "Architecture": "#DDA15E",
        "Metric": "#BC6C25"
    }
    
    # Add nodes
    node_ids = set()
    for node in nodes:
        labels = node.get("labels", [])
        node_type = labels[0] if labels else "Unknown"
        
        # Get node identifier
        node_id = node.get("id") or node.get("name") or str(node)
        
        if node_id in node_ids:
            continue
        node_ids.add(node_id)
        
        # Get title (for tooltip)
        title = node.get("title") or node.get("name") or node_id
        if node.get("year"):
            title += f" ({node.get('year')})"
        
        # Get color
        color = color_map.get(node_type, "#95A5A6")
        
        # Add node
        net.add_node(
            node_id,
            label=node_id[:30] + "..." if len(node_id) > 30 else node_id,
            title=title,
            color=color,
            size=20 if node_type == "Paper" else 15
        )
    
    # Add edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        rel_type = edge.get("type", "RELATED_TO")
        
        if source in node_ids and target in node_ids:
            net.add_edge(
                source,
                target,
                title=rel_type,
                label=rel_type[:10],
                color="#7F8C8D"
            )
    
    # Configure physics
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 200},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04
        }
      }
    }
    """)
    
    # Save
    output_path = project_root / "data" / output_file
    net.save_graph(str(output_path))
    print(f"✓ Visualization saved to: {output_path}")
    print(f"  Open in browser: file://{output_path.absolute()}")
    
    return output_path


def create_simple_text_visualization(nodes, edges):
    """Create simple text-based visualization"""
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*80)
    
    # Count by type
    node_types = {}
    for node in nodes:
        labels = node.get("labels", [])
        node_type = labels[0] if labels else "Unknown"
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\n📊 Nodes: {len(nodes)}")
    for node_type, count in node_types.items():
        print(f"   • {node_type}: {count}")
    
    # Count relationships
    rel_types = {}
    for edge in edges:
        rel_type = edge.get("type", "Unknown")
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    print(f"\n🔗 Relationships: {len(edges)}")
    for rel_type, count in rel_types.items():
        print(f"   • {rel_type}: {count}")
    
    # Show sample papers
    print(f"\n📄 Papers:")
    papers = [n for n in nodes if "Paper" in n.get("labels", [])]
    for paper in papers[:5]:
        title = paper.get("title", paper.get("id", "Unknown"))
        year = paper.get("year", "?")
        print(f"   • {title} ({year})")
    
    # Show sample relationships
    print(f"\n🔗 Sample Relationships:")
    for edge in edges[:10]:
        print(f"   • {edge['source']} --[{edge['type']}]--> {edge['target']}")


def main():
    """Main visualization function"""
    print("="*80)
    print("Knowledge Graph Visualization")
    print("="*80)
    
    # Connect to Neo4j
    driver = get_neo4j_connection()
    if not driver:
        return
    
    try:
        # Fetch graph data
        print("\n📥 Fetching graph data from Neo4j...")
        nodes, edges = fetch_graph_data(driver)
        print(f"   ✓ Found {len(nodes)} nodes and {len(edges)} relationships")
        
        if len(nodes) == 0:
            print("\n⚠ No nodes found in graph. Build the graph first:")
            print("   python scripts/build_graph.py")
            return
        
        # Create text summary
        create_simple_text_visualization(nodes, edges)
        
        # Create interactive visualization
        try:
            import pyvis
            output_path = create_pyvis_visualization(nodes, edges)
            print(f"\n✅ Visualization complete!")
            print(f"\n💡 Tips:")
            print(f"   • Open the HTML file in your browser")
            print(f"   • Drag nodes to rearrange")
            print(f"   • Hover over nodes/edges for details")
            print(f"   • Use Neo4j Browser at http://localhost:7474 for advanced queries")
        except ImportError:
            print("\n⚠ pyvis not installed. Installing...")
            os.system("pip install pyvis")
            output_path = create_pyvis_visualization(nodes, edges)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()

