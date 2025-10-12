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

## Testing & Development

### Unit Tests

**Test Coverage: 27%** (21 tests, 100% passing ✅)

```bash
# Install uv (fast Python package installer - 10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
cd backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest test/unit_tests/

# Run with coverage report
pytest test/unit_tests/ --cov=backend/app --cov-report=html

# View coverage report
open test_coverage_report/index.html
```

**High Coverage Modules:**
- `models.py`: 100%
- `config.py`: 100%
- `document_helpers.py`: 90%
- `document_processor.py`: 83%
- `vector_store.py`: 66%

**CI Pipeline:** Tests run automatically on PRs to `development` branch via GitHub Actions.

See `test/README.md` for detailed testing documentation.

---

## RAGAS Evaluation Framework

### Overview

RAGAS (RAG Assessment) is integrated to evaluate RAG system performance using standardized metrics. We tested with two different document types to assess system versatility:

#### Test 1: Harrier EV Product Brochure (19MB, 33 chunks)

| Metric | Score | Status | Description |
|--------|-------|--------|-------------|
| **Context Precision** | 91.25% | ✅ Excellent | Relevance of retrieved documents |
| **Context Recall** | 79.63% | ✅ Good | Coverage of ground truth information |
| **Retriever F1 Score** | 81.05% | ✅ Good | Balanced precision & recall |
| **Answer Correctness** | 33.11% | ⚠️ Needs Work | Similarity to expected answers |
| **Factual Correctness** | 21.22% | ⚠️ Needs Work | Factual accuracy verification |
| **Overall Score** | **56.30%** | ⚠️ Moderate | Average across all metrics |

**Last Tested:** October 12, 2025 at 15:24:39 IST
**Test Duration:** 8 minutes 36 seconds
**Questions Evaluated:** 40
**Document:** `test/public/harrier-ev-all-you-need-to-know.pdf`

#### Test 2: Dev Singh Resume (85KB, 5 chunks)

| Metric | Score | Status | Description |
|--------|-------|--------|-------------|
| **Context Precision** | 100.00% | ✅ Perfect | Relevance of retrieved documents |
| **Context Recall** | 90.91% | ✅ Excellent | Coverage of ground truth information |
| **Retriever F1 Score** | 90.91% | ✅ Excellent | Balanced precision & recall |
| **Answer Correctness** | 30.40% | ⚠️ Needs Work | Similarity to expected answers |
| **Factual Correctness** | 26.43% | ⚠️ Needs Work | Factual accuracy verification |
| **Overall Score** | **61.94%** | ⚠️ Moderate | Average across all metrics |

**Last Tested:** October 12, 2025 at 10:39:01 AM
**Test Duration:** 7 minutes
**Questions Evaluated:** 44
**Document:** `test/public/Dev-Singh-AI-Engineer.pdf`

### Key Findings

**✅ Strengths:**
- **Perfect retrieval on short documents** (100% precision on resume)
- Excellent document retrieval on long documents (91% precision)
- Strong information coverage (80-91% recall across tests)
- Hybrid search + reranking working effectively
- Graph RAG contributing to context
- **Better performance on structured documents** (resumes vs brochures)

**⚠️ Areas for Improvement:**
- Answer generation quality (30-33% across tests)
- Factual accuracy (21-26% across tests)
- Prompt engineering needed
- Consider GPT-4 for answer generation

**📊 Performance Insights:**
- Shorter documents (resumes) yield better retrieval scores
- Context precision varies by document complexity
- Answer correctness remains consistent (~30%) regardless of document type
- System excels at retrieval but needs improvement in answer generation

### Running RAGAS Tests

**Local Testing:**
```bash
# Ensure services are running
docker-compose up -d

# Install dependencies in container with uv (if not already installed)
docker exec lyzr-hackathon-backend-1 bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\"\$HOME/.cargo/bin:\$PATH\" && cd /app && uv pip install ragas pandas tqdm"

# Run RAGAS evaluation - Harrier EV test (40 questions)
docker exec lyzr-hackathon-backend-1 python -m pytest test/integration_tests/test_ragas_evaluation.py::TestRAGASEvaluation::test_ragas_evaluation_with_test_data -v -s

# Run RAGAS evaluation - Dev Singh resume test (44 questions)
docker exec lyzr-hackathon-backend-1 python -m pytest test/integration_tests/test_ragas_evaluation.py::TestRAGASEvaluation::test_ragas_evaluation_dev_singh -v -s

# Run all RAGAS tests
docker exec lyzr-hackathon-backend-1 python -m pytest test/integration_tests/test_ragas_evaluation.py -v -s
```

**CI/CD Pipeline:**
- **Unit Tests**: Run automatically on all branches (using `uv` for 10x faster installs)
- **RAGAS Tests**: Manual trigger on merge requests
- **Post-Merge**: Automatic validation on development branch
- **Package Manager**: Uses `uv` instead of `pip` for 10-100x faster dependency installation

### RAGAS Components

**1. Evaluator Service** (`backend/app/services/ragas_evaluator.py`)
```python
from app.services.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()
results = evaluator.evaluate_rag_performance(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)
```

**2. Integration Test** (`test/integration_tests/test_ragas_evaluation.py`)
- Processes test PDF document
- Queries all 40 test questions
- Evaluates with RAGAS metrics
- Outputs detailed performance report

**3. Test Data**

Two test datasets are available:

| Dataset | CSV File | PDF Document | Questions | Size |
|---------|----------|--------------|-----------|------|
| Harrier EV | `harrier_ev_detailed_qa.csv` | `harrier-ev-all-you-need-to-know.pdf` | 40 | 19MB |
| Dev Singh Resume | `dev_singh_ai_engineer_qa.csv` | `Dev-Singh-AI-Engineer.pdf` | 44 | 85KB |

**4. Results Tracking**

Each test run automatically appends results to single files for easy historical tracking:

**Files:**
- `test/results/ragas_summary.json` - JSON array of all Harrier EV test runs
- `test/results/ragas_summary.csv` - CSV with all Harrier EV test runs (one row per run)
- `test/results/ragas_dev_singh_summary.json` - JSON array of all Dev Singh test runs
- `test/results/ragas_dev_singh_summary.csv` - CSV with all Dev Singh test runs (one row per run)

**View latest results:**
```bash
# View most recent Harrier EV test (last entry in JSON array)
tail -20 test/results/ragas_summary.json

# View most recent Dev Singh test (last entry in JSON array)
tail -20 test/results/ragas_dev_singh_summary.json

# View all historical runs in CSV format
cat test/results/ragas_summary.csv
cat test/results/ragas_dev_singh_summary.csv
```

**Track trends over time:**
```python
import pandas as pd
import json

# Load all historical runs from JSON
with open('test/results/ragas_summary.json') as f:
    history = json.load(f)  # Array of test runs

# Convert to DataFrame for analysis
trends = pd.DataFrame(history)

# View metric trends over time
print(trends[['test_date', 'test_time', 'context_precision_mean', 'answer_correctness_mean']])

# Or simply load from CSV
trends = pd.read_csv('test/results/ragas_summary.csv')
print(trends[['test_date', 'context_precision_mean', 'answer_correctness_mean']])
```

### Metric Definitions

**Context Precision (91.25%)** ✅
- How relevant are retrieved documents?
- >80% is good; our system: 91.25%

**Context Recall (79.63%)** ✅
- How much ground truth info is captured?
- >70% is acceptable; our system: 79.63%

**Retriever F1 Score (81.05%)** ✅
- Balanced measure: F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Answer Correctness (33.11%)** ⚠️
- Similarity between generated and expected answers
- Target: >70% (needs improvement)

**Factual Correctness (21.22%)** ⚠️
- Atomic fact verification accuracy
- Target: >60% (needs improvement)

### Performance by Question Type

| Type | Context Precision | Context Recall | Avg Relevance |
|------|------------------|----------------|---------------|
| Technical Specs | 95% | 85% | 6.8 |
| Feature Summaries | 92% | 78% | 5.2 |
| Reasoning | 89% | 76% | 4.9 |
| Conversational | 91% | 80% | 5.5 |
| Contextual | 87% | 75% | 4.3 |

### Improvement Recommendations

**Immediate (High Priority):**
1. Optimize answer generation prompts
   - Add explicit instructions for concise answers
   - Include format examples
   - Add fact verification step

2. Implement answer post-processing
   - Verify facts against retrieved context
   - Remove speculative statements
   - Add confidence scoring

3. Enhance context utilization
   - Increase context window
   - Better passage merging
   - Prioritize highest relevance

**Medium-Term:**
- Consider GPT-4 for answer generation
- Add more gold-standard test cases
- Implement continuous evaluation
- Track metrics over time

**Long-Term:**
- Collect user feedback
- Fine-tune models on domain data
- Implement quality gates in CI/CD
- Chain-of-Thought prompting

### CI/CD Integration

**GitLab Pipeline** (`.gitlab-ci.yml`):

```yaml
# Unit Tests (all branches) - using uv for fast installs
unit_tests:
  stage: test
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - uv venv && source .venv/bin/activate
    - uv pip install -r requirements.txt
  script: pytest test/unit_tests/

# RAGAS Tests (manual on MRs)
ragas_integration_test:
  stage: integration
  only: merge_requests
  when: manual

# Post-Merge (automatic on development)
ragas_post_merge:
  stage: integration
  only: development
  when: on_success
```

**Benefits of `uv`:**
- ⚡ 10-100x faster than `pip` for dependency installation
- 🔒 More reliable dependency resolution
- 💾 Better caching mechanism
- 🎯 Compatible with existing `requirements.txt` files

**Artifacts:**
- Test results: `backend/test-results.xml`
- Retention: 1 week (MRs), 1 month (development)

### Dependencies

Added to `backend/requirements.txt`:
```
ragas==0.2.9
pandas==2.2.3
tqdm==4.67.1
```

### Troubleshooting

**OpenAI API Key not set:**
```bash
# Add to .env
OPENAI_API_KEY=sk-proj-your-key
```

**Test files not found:**
```bash
# Verify test folder mount in docker-compose.yml
volumes:
  - ./test:/app/test
```

**Database connection errors:**
```bash
# Check all services are healthy
docker-compose ps
```

---

### System Testing

#### 1. Upload Document

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

#### 2. Check Results

```bash
# Get graph stats
curl http://localhost:8000/api/rag/graph/stats/{document_id}

# List entities
curl "http://localhost:8000/api/rag/graph/entities?document_id={id}&limit=10"

# List relationships
curl "http://localhost:8000/api/rag/graph/relationships?document_id={id}&limit=10"
```

#### 3. Test Queries

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

#### 4. View Graph in Neo4j

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

---

## Entity Deduplication Feature

### Overview

This feature adds intelligent entity deduplication to the GraphRAG pipeline using BFS (Breadth-First Search) traversal and multiple similarity metrics.

### How It Works

#### Architecture

The entity deduplication pipeline runs automatically after GraphRAG processes a document and imports entities into Neo4j:

```
Document Upload → Text Extraction → GraphRAG Processing → Neo4j Import → Entity Deduplication
```

#### Key Components

1. **Graph Refinement Pipeline** (`backend/app/services/graph_refinement_pipeline.py`)
   - BFS-based entity traversal
   - Multi-metric similarity calculation
   - Automatic entity merging
   - Context preservation

2. **Similarity Metrics**
   - **String Similarity**: Uses Levenshtein distance for name matching
   - **Semantic Similarity**: Uses sentence transformers for description matching
   - **Contextual Similarity**: Analyzes shared relationships in the graph
   - **Type Matching**: Ensures entities are of the same type

3. **Merge Strategy**
   - **Auto-merge** (≥95% similarity): Automatically merges entities
   - **Suggest merge** (≥85% similarity): Logs for manual review
   - **Ignore** (<75% similarity): Skips potential duplicates

### Configuration

#### Environment Variables

The feature is controlled via the `.env` file:

```env
# Entity Resolution & Deduplication
ENABLE_ENTITY_DEDUPLICATION=true
ENTITY_SIMILARITY_THRESHOLD=0.85
```

#### Configuration Options

All configuration is handled in `backend/app/config.py`:

- `enable_entity_deduplication` (bool): Enable/disable the feature
- `entity_similarity_threshold` (float): Threshold for auto-merging (default: 0.85)

#### Advanced Configuration

You can customize the behavior by modifying `GraphRefinementConfig` in the pipeline:

```python
class GraphRefinementConfig(BaseModel):
    # Feature flags
    enable_deduplication: bool = True
    enable_entity_enhancement: bool = True

    # Similarity thresholds
    auto_merge_threshold: float = 0.95  # Auto-merge above this
    suggest_merge_threshold: float = 0.85  # Suggest to user
    min_similarity_threshold: float = 0.75  # Ignore below this

    # BFS configuration
    bfs_max_depth: int = 3  # How deep to traverse
    bfs_batch_size: int = 100  # Process in batches

    # Performance
    max_nodes_to_process: int = 10000
    use_embeddings: bool = True
    cache_embeddings: bool = True
```

### Usage

#### Automatic Processing

Entity deduplication runs automatically when you upload a document if `ENABLE_ENTITY_DEDUPLICATION=true`:

```bash
# Upload a document via API
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@document.pdf" \
  -F "method=pymupdf"
```

The pipeline will:
1. Extract text from PDF
2. Process with GraphRAG
3. Import entities and relationships to Neo4j
4. **Automatically deduplicate entities**
5. Store refined graph

#### Disabling the Feature

To disable entity deduplication:

```env
# .env
ENABLE_ENTITY_DEDUPLICATION=false
```

### How Entity Merging Works

#### Merge Process

1. **Identify Similar Entities**: BFS traversal finds potential duplicates
2. **Calculate Similarity**: Multi-metric scoring system
3. **Merge Decision**: Based on similarity thresholds
4. **Property Merging**: Combines descriptions and metadata
5. **Relationship Transfer**: Moves all relationships to merged entity
6. **Cleanup**: Removes duplicate entity

#### Example

**Before Deduplication:**
```
Entity A: "Machine Learning"
  - description: "A subset of AI"
  - relationships: 3

Entity B: "Machine learning"
  - description: "AI technique for pattern recognition"
  - relationships: 5
```

**After Deduplication:**
```
Entity A: "Machine Learning"
  - description: "A subset of AI. AI technique for pattern recognition"
  - relationships: 8 (combined)
  - merged_from: ["Machine learning"]
  - merge_confidence: 0.97
```

### Dependencies

The following packages are required (already added to `requirements.txt`):

```txt
sentence-transformers==3.3.1
python-Levenshtein==0.25.1
numpy==1.26.4
neo4j==5.26.0
```

Install them with:

```bash
pip install -r backend/requirements.txt
```

### Logging

The feature provides detailed logging during processing:

```
🔧 Starting entity deduplication for document abc-123...
🔄 Starting BFS deduplication scan...
   Found 150 entities to process
   ✅ Auto-merged: Entity_A + Entity_B (score: 0.97)
   ⚠️  Suggest merge: Entity_C + Entity_D (score: 0.88)
✓ Entity deduplication complete:
   - Entities processed: 150
   - Duplicates merged: 12
   - Suggested merges: 3
   - Time: 8.45s
```

### Performance Considerations

#### BFS Parameters

- **bfs_max_depth**: Controls how far to traverse (default: 3)
  - Lower = faster but may miss duplicates
  - Higher = more thorough but slower

- **bfs_batch_size**: Number of duplicates to process at once (default: 100)
  - Balances memory usage and processing speed

#### Optimization Tips

1. **Disable entity enhancement** if not needed:
   ```python
   enable_entity_enhancement: bool = False
   ```

2. **Adjust BFS depth** for large graphs:
   ```python
   bfs_max_depth: int = 2  # Faster, less thorough
   ```

3. **Use caching** for embeddings:
   ```python
   cache_embeddings: bool = True
   ```

### Troubleshooting Entity Deduplication

#### Common Issues

1. **Slow processing**
   - Reduce `bfs_max_depth`
   - Increase `min_similarity_threshold`
   - Disable `enable_entity_enhancement`

2. **Too many merges**
   - Increase `auto_merge_threshold`
   - Adjust weights in similarity calculation

3. **Too few merges**
   - Lower `auto_merge_threshold`
   - Check if entities have descriptions

#### Debugging

Enable debug logging in your application:

```python
import logging
logging.getLogger('app.services.graph_refinement_pipeline').setLevel(logging.DEBUG)
```

### Future Enhancements

Potential improvements:

1. **Cross-document linking**: Connect entities across multiple documents
2. **Manual merge review**: UI for reviewing suggested merges
3. **Undo functionality**: Ability to reverse merges
4. **Custom similarity metrics**: Domain-specific scoring
5. **Bulk operations**: Process all documents at once

### References

- Original implementation: `sample-files/entity_de_duplication.py`
- Integration point: `backend/app/utils/document_helpers.py:138-164`
- Pipeline implementation: `backend/app/services/graph_refinement_pipeline.py`
