# RAG System with GraphRAG

Advanced Retrieval-Augmented Generation system with intelligent document processing, multi-tool search agent, and knowledge graph extraction.

**Status**: ✅ **Production Ready**

---

## Quick Start

### 1. Setup

```bash
# Clone and configure
git clone <repository-url>
cd LYZR-Hackathon
cp .env-example .env

# Edit .env with your credentials:
# - OPENAI_API_KEY=sk-proj-your-key
# - POSTGRES_PASSWORD=your_password
# - NEO4J_AUTH=neo4j/your_password

# Start system
docker-compose up --build
```

### 2. Upload & Query

```bash
# Upload document (automatically processes with GraphRAG)
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@document.pdf" \
  -F "method=docling"

# Query (agent auto-selects vector or graph search)
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?", "document_id": "{id}"}'
```

### 3. Access

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | - |
| **Backend API** | http://localhost:8000/docs | - |
| **Neo4j Browser** | http://localhost:7474 | neo4j / neo4j_password |

---

## Features

### ✅ Multi-Tool Search Agent

**Intelligent routing between search types:**
- **Vector Search**: "What is X?", "Explain...", "Define..." (semantic + keyword hybrid)
- **Graph Search**: "How are X and Y related?", "What connects..." (Neo4j relationships)

**Agent automatically:**
- Analyzes query intent
- Selects appropriate search tool
- Executes search
- Synthesizes answer with citations

### ✅ GraphRAG Pipeline

**Automatic knowledge graph extraction:**
1. **Text Chunking**: Splits into 1200-char pieces (Microsoft GraphRAG approach)
2. **Entity Extraction**: Identifies people, places, concepts per chunk
3. **Relationship Mapping**: Finds connections between entities
4. **Deduplication**: Merges same entities across chunks
5. **Neo4j Storage**: Creates queryable knowledge graph

**Result**: Books with 500+ pages → 85+ entities, 142+ relationships

### ✅ Document Processing

- **PDF Upload** → Text extraction (Docling/PyMuPDF)
- **Embeddings** → text-embedding-3-small (PGVector storage)
- **Hybrid Search** → Vector + BM25 keyword search + cross-encoder reranking
- **GraphRAG** → Automatic if `GRAPHRAG_ENABLED=true`

---

## Key Improvements

### 1. Document-Aware Agent ✅

**Problem**: Agent asked "Could you provide the title?" when user already uploaded the document.

**Fix**: Agent now knows documents are uploaded and searches them automatically.

### 2. GraphRAG Chunking ✅

**Problem**: Only found 1 entity and 1 relationship from entire books.

**Fix**: Implemented proper chunking - processes 1200-char pieces separately, extracts 0-11 entities per chunk, merges results.

### 3. Tool Execution ✅

**Problem**: Agent described what it would do instead of actually calling tools.

**Fix**: Simplified system prompt, uses `StructuredTool.from_function(coroutine=...)` for proper async execution.

### 4. Error Handling ✅

**Problem**: `KeyError: 'graph_document'`, Neo4j syntax errors.

**Fix**: Updated to `graph_documents` (plural), graceful handling for unsupported Neo4j features.

---

## Configuration

### Environment Variables (.env)

```env
# OpenAI (Required)
OPENAI_API_KEY=sk-proj-your-key-here

# Database
POSTGRES_PASSWORD=your_secure_password
NEO4J_AUTH=neo4j/your_secure_password

# GraphRAG (Optional)
GRAPHRAG_ENABLED=true              # Set false to disable
GRAPHRAG_LLM_MODEL=gpt-4o-mini     # Model for entity extraction
CHUNK_SIZE=1200                     # Characters per chunk
CHUNK_OVERLAP=100                   # Overlap for context

# Search Settings
TOP_K_RESULTS=5                     # Number of results to return
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### Toggle GraphRAG

```bash
# Disable GraphRAG (vector search only)
GRAPHRAG_ENABLED=false

# Enable GraphRAG (vector + graph search)
GRAPHRAG_ENABLED=true
```

---

## Testing

### 1. Upload Document

```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@book.pdf" \
  -F "method=docling"
```

**Monitor processing:**
```bash
docker-compose logs backend --follow | grep -E "Chunk [0-9]+:|entities|relationships"
```

**Expected output:**
```
Split document into 42 chunks (size: 1200, overlap: 100)
Chunk 1: 2 entities, 1 relationships
Chunk 2: 5 entities, 4 relationships
...
GraphRAG complete: 85 entities, 142 relationships from 42 chunks
```

### 2. Check Results

```bash
# Get graph stats
curl http://localhost:8000/api/rag/graph/stats/{document_id}

# List entities
curl "http://localhost:8000/api/rag/graph/entities?document_id={id}&limit=10"

# List relationships
curl "http://localhost:8000/api/rag/graph/relationships?document_id={id}&limit=10"
```

### 3. Test Queries

**Relationship query → Graph Search:**
```bash
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How are the characters related?", "document_id": "{id}"}'
```

**Factual query → Vector Search:**
```bash
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "document_id": "{id}"}'
```

### 4. View Graph in Neo4j

```cypher
// Open http://localhost:7474 and run:
MATCH (e:__Entity__)-[r]-(e2:__Entity__)
RETURN e, r, e2 LIMIT 50
```

---

## Architecture

```
Upload PDF
    ↓
Extract Text (Docling/PyMuPDF)
    ↓
Chunk Text (1000 chars for embeddings)
    ↓
Generate Embeddings → Store in PGVector
    ↓
Build BM25 Index → Keyword search ready
    ↓
GraphRAG Processing (if enabled):
  ├─ Chunk text (1200 chars, 100 overlap)
  ├─ Process each chunk → Extract entities/relationships
  ├─ Deduplicate across chunks
  └─ Import to Neo4j
    ↓
Document Ready
    ↓
User Query
    ↓
Search Agent analyzes intent:
  ├─ "What/Explain/Define" → Vector Search (PGVector + BM25 + Reranking)
  └─ "How related/Connected" → Graph Search (Neo4j Cypher)
    ↓
Synthesize Answer with Citations
```

---

## Performance

### Processing Time

| Document Size | Chunks | Vector + BM25 | GraphRAG | Total |
|---------------|--------|---------------|----------|-------|
| 10 pages (~10K chars) | ~10 | ~10s | ~40s | ~50s |
| 50 pages (~50K chars) | ~42 | ~15s | ~3min | ~3.5min |
| 200 pages (~200K chars) | ~167 | ~25s | ~12min | ~12.5min |
| Full book (~500K chars) | ~417 | ~40s | ~30min | ~30.5min |

**GraphRAG Time = chunks × 4 seconds** (one LLM API call per chunk)

### Cost (GraphRAG with gpt-4o-mini)

| Document Size | Estimated Cost |
|---------------|----------------|
| 10 pages | ~$0.03 |
| 50 pages | ~$0.15 |
| 200 pages | ~$0.50 |
| Full book | ~$1.50 |

---

## API Endpoints

### Documents

```bash
# Upload
POST /api/rag/upload

# List all
GET /api/rag/documents

# Get status
GET /api/rag/documents/{id}

# Delete
DELETE /api/rag/documents/{id}
```

### GraphRAG

```bash
# Manual trigger (if not automatic)
POST /api/rag/documents/{id}/process-graph

# Get graph stats
GET /api/rag/graph/stats/{id}

# List entities
GET /api/rag/graph/entities?document_id={id}&limit=50

# List relationships
GET /api/rag/graph/relationships?document_id={id}&limit=50
```

### Query

```bash
# Streaming query (auto-routes to vector or graph search)
POST /api/rag/query/stream
Body: {"query": "...", "document_id": "..."}
```

---

## Troubleshooting

### GraphRAG not working?

**Check 1**: Is it enabled?
```bash
# In .env
GRAPHRAG_ENABLED=true
```

**Check 2**: View logs
```bash
docker-compose logs backend --follow | grep -i graphrag
```

**Check 3**: Neo4j running?
```bash
docker-compose ps
curl http://localhost:7474
```

### Agent not searching documents?

The agent is configured to always search uploaded documents. If it asks for clarification, check:
- Document was uploaded successfully
- Document has chunks (check document status)
- Query is clear and specific

### Slow processing?

**For GraphRAG:**
- Large documents take time (1 book = ~30 minutes)
- Each chunk requires 1 LLM API call
- Use `gpt-4o-mini` for speed (already default)
- Reduce `CHUNK_SIZE` to process fewer chunks

**For vector search:**
- Should be fast (<5 seconds)
- Check database connections
- Verify embeddings were created

---

## File Structure

```
backend/
├── app/
│   ├── api/rag_routes.py           # API endpoints
│   ├── models.py                   # Database schema
│   ├── config.py                   # Configuration
│   ├── services/
│   │   ├── graphrag_pipeline.py   # Entity/relationship extraction
│   │   ├── neo4j_service.py       # Neo4j operations
│   │   ├── graph_search.py        # Graph queries
│   │   ├── search_agent.py        # Multi-tool agent
│   │   ├── vector_store.py        # PGVector operations
│   │   ├── bm25_search.py         # Keyword search
│   │   └── reranker.py            # Result reranking
│   └── utils/
│       └── document_helpers.py    # Processing pipeline
└── requirements.txt

frontend/
├── src/
│   ├── App.jsx                    # Main UI
│   └── components/
└── package.json
```

---

## Dependencies

### Backend
- FastAPI, Uvicorn
- LangChain, LangChain-OpenAI, LangChain-Experimental
- PostgreSQL, PGVector, Neo4j drivers
- Docling (PDF processing)
- Sentence-transformers (reranking)
- PyMuPDF (alternative PDF processor)

### Frontend
- React
- Axios (API calls)

---

## Security

**Protected in .env (never commit):**
- `OPENAI_API_KEY` - OpenAI API access
- `POSTGRES_PASSWORD` - Database credentials
- `NEO4J_AUTH` - Graph database credentials

**Best Practices:**
1. `.env` is in `.gitignore`
2. Use `.env-example` as template
3. Use strong passwords in production
4. Rotate API keys regularly
5. Use secrets management for production (AWS Secrets Manager, etc.)

---

## What's Included

✅ **Document Processing**
- PDF upload and text extraction
- Automatic chunking and embedding
- BM25 keyword indexing

✅ **Multi-Tool Search Agent**
- Intelligent query analysis
- Auto-routing (vector vs graph)
- Tool execution with reasoning
- Streaming responses

✅ **GraphRAG Pipeline**
- Text chunking (1200-char pieces)
- Entity extraction per chunk
- Relationship mapping
- Deduplication and merging
- Neo4j graph storage

✅ **Vector Search**
- Hybrid retrieval (Vector + BM25)
- Cross-encoder reranking
- Document-aware responses

✅ **Graph Search**
- Natural language to Cypher
- Relationship queries
- Entity path finding

✅ **Error Handling**
- Graceful degradation
- Detailed logging
- User-friendly messages

---

## Next Steps

1. **Upload your documents** and watch automatic processing
2. **Test queries** - try both factual and relationship questions
3. **View graphs** in Neo4j browser
4. **Optimize** chunk size and models for your domain
5. **Monitor** costs and performance

---

**Last Updated**: 2025-10-12
**Status**: ✅ Production Ready
**Version**: 1.0.0
