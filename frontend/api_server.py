#!/usr/bin/env python3
"""
Flask API Server for GraphRAG Frontend
Provides REST API endpoints for the chat interface
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.query.engine import GraphRAGQuery

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize query engine (singleton)
query_engine = None

def get_query_engine():
    """Get or create query engine instance"""
    global query_engine
    if query_engine is None:
        query_engine = GraphRAGQuery(
            use_graph=True,
            max_hops=2,
            use_generator=True
        )
    return query_engine


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GraphRAG API'
    })


@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests from frontend"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query is required'
            }), 400
        
        query_text = data['query']
        use_graph = data.get('use_graph', True)
        use_generator = data.get('use_generator', True)
        max_hops = data.get('max_hops', 2)
        top_k = data.get('top_k', 5)
        use_temporal = data.get('use_temporal', False)
        
        # Get query engine
        engine = get_query_engine()
        
        # Override engine settings for this query
        original_use_graph = engine.use_graph
        original_use_generator = engine.use_generator
        original_max_hops = engine.max_hops
        
        engine.use_graph = use_graph
        engine.use_generator = use_generator
        engine.max_hops = max_hops
        
        try:
            # Execute query
            result = engine.query(
                query=query_text,
                top_k=top_k,
                use_temporal=use_temporal,
                generate_answer=use_generator
            )
            
            # Restore original settings
            engine.use_graph = original_use_graph
            engine.use_generator = original_use_generator
            engine.max_hops = original_max_hops
            
            return jsonify(result)
            
        except Exception as e:
            # Restore original settings
            engine.use_graph = original_use_graph
            engine.use_generator = original_use_generator
            engine.max_hops = original_max_hops
            raise e
            
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing your query'
        }), 500


if __name__ == '__main__':
    print("="*80)
    print("GraphRAG API Server")
    print("="*80)
    
    # Get port from environment or default to 5001 (5000 often used by AirPlay on macOS)
    port = int(os.getenv('API_PORT', 5001))
    
    print(f"Starting server on http://localhost:{port}")
    print(f"Frontend: Open frontend/index.html in your browser")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=True)

