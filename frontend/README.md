# GraphRAG Frontend

A modern React + Vite web-based chat interface for interacting with the GraphRAG system.

## Features

- 💬 **Chat Interface**: Clean, modern chat UI built with React
- 🤖 **AI-Powered Answers**: Get natural language responses generated from retrieved context
- 📊 **Source Citations**: See which papers and sources were used
- 📅 **Evolution Timeline**: For temporal queries, see year-by-year developments
- ⚙️ **Configurable Options**: Toggle graph traversal, answer generation, and traversal depth
- ⚡ **Fast Development**: Hot module replacement with Vite

## Setup

1. **Install Node.js dependencies**:
```bash
cd frontend
npm install
```

2. **Start the API server** (in a separate terminal):
```bash
python frontend/api_server.py
```

The server will start on `http://localhost:5000`

3. **Start the React development server**:
```bash
npm run dev
```

The frontend will start on `http://localhost:3000` (Vite default port)

4. **Build for production**:
```bash
npm run build
```

The built files will be in the `dist/` folder.

## Usage

1. Make sure the API server is running (`python frontend/api_server.py`)
2. Start the React dev server (`npm run dev`)
3. Open `http://localhost:3000` in your browser
4. Type your question in the input field
5. Press Enter or click the send button
6. View the generated answer with sources and confidence score
7. For temporal queries, see the evolution timeline

## Configuration Options

- **Use Graph**: Enable/disable knowledge graph traversal
- **Generate Answer**: Enable/disable LLM answer generation (shows raw results if disabled)
- **Max Hops**: Set graph traversal depth (1-3 hops)

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # App header component
│   │   ├── ChatMessage.jsx     # Message display component
│   │   └── ChatInput.jsx       # Input and options component
│   ├── App.jsx                 # Main app component
│   ├── App.css                 # App styles
│   ├── main.jsx                # React entry point
│   └── index.css               # Global styles
├── index.html                  # HTML template
├── vite.config.js              # Vite configuration
├── package.json                # Dependencies
└── api_server.py               # Flask API server
```

## API Endpoints

### `POST /api/query`

Query the GraphRAG system.

**Request Body:**
```json
{
  "query": "How did attention mechanisms evolve?",
  "use_graph": true,
  "use_generator": true,
  "max_hops": 2,
  "top_k": 5,
  "use_temporal": false
}
```

**Response:**
```json
{
  "query": "How did attention mechanisms evolve?",
  "query_entities": ["attention"],
  "results": [...],
  "generated_answer": {
    "answer": "...",
    "confidence": 0.85,
    "sources": [...],
    "timeline": [...]
  }
}
```

### `GET /api/health`

Health check endpoint.

## Troubleshooting

**CORS Errors**: Make sure `flask-cors` is installed and the API server is running.

**Connection Refused**: Ensure the API server is running on port 5000.

**No Results**: Make sure you've run `scripts/ingest.py` and `scripts/build_graph.py` first.

