#!/bin/bash
# Neo4j Setup Script

echo "=========================================="
echo "Neo4j Setup for GraphRAG"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ docker-compose not found"
    exit 1
fi

echo "Choose setup method:"
echo "1. Docker Compose (Recommended - easiest)"
echo "2. Docker run (Simple standalone)"
echo "3. Skip (I'll set it up manually)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Neo4j with Docker Compose..."
        $COMPOSE_CMD up -d
        
        echo ""
        echo "✓ Neo4j is starting..."
        echo "  • HTTP Interface: http://localhost:7474"
        echo "  • Bolt Protocol: bolt://localhost:7687"
        echo "  • Username: neo4j"
        echo "  • Password: graphrag123"
        echo ""
        echo "⚠ IMPORTANT: Update your .env file with:"
        echo "  NEO4J_URI=bolt://localhost:7687"
        echo "  NEO4J_USER=neo4j"
        echo "  NEO4J_PASSWORD=graphrag123"
        echo ""
        echo "Waiting for Neo4j to be ready..."
        sleep 5
        echo "✓ Setup complete!"
        ;;
    2)
        echo ""
        read -p "Enter Neo4j password (default: graphrag123): " password
        password=${password:-graphrag123}
        
        echo "Starting Neo4j container..."
        docker run -d \
          --name neo4j-graphrag \
          -p 7474:7474 \
          -p 7687:7687 \
          -e NEO4J_AUTH=neo4j/$password \
          neo4j:5.15-community
        
        echo ""
        echo "✓ Neo4j container started!"
        echo "  • HTTP Interface: http://localhost:7474"
        echo "  • Bolt Protocol: bolt://localhost:7687"
        echo "  • Username: neo4j"
        echo "  • Password: $password"
        echo ""
        echo "⚠ IMPORTANT: Update your .env file with:"
        echo "  NEO4J_URI=bolt://localhost:7687"
        echo "  NEO4J_USER=neo4j"
        echo "  NEO4J_PASSWORD=$password"
        ;;
    3)
        echo ""
        echo "Manual setup instructions:"
        echo "1. Install Neo4j: https://neo4j.com/download/"
        echo "2. Or use Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest"
        echo "3. Update .env with your Neo4j credentials"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "To stop Neo4j:"
echo "  docker stop neo4j-graphrag"
echo ""
echo "To start Neo4j:"
echo "  docker start neo4j-graphrag"
echo ""

