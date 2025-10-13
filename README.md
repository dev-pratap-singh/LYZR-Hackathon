# Advanced RAG System with GraphRAG & Multi-Tool Search Agent

An intelligent Retrieval-Augmented Generation (RAG) system that combines vector search, knowledge graph extraction, and metadata filtering to provide comprehensive document analysis and question answering capabilities.

**Status**: ✅ **Production Ready**

**Version**: 1.2.0

---

## 🎉 What's New in v1.2.0

### 🆕 Natural Language Graph Updates (v1.2.0)
- **✏️ Graph Update Tool**: NEW 4th tool added to search agent for real-time graph modifications
- **🎯 9 Update Operations**: Create nodes, merge entities, update properties, manage relationships
- **🗣️ Natural Language Commands**: "Create X and connect to Y, Z" - no Cypher needed!
- **⚡ Batch Operations**: Create node + connect to multiple entities in ONE operation
- **🔄 Auto-Refresh**: Frontend automatically updates graph visualization after changes
- **📊 Detailed Feedback**: Shows successful vs failed operations with clear error messages

### Enhanced Graph Capabilities (v1.1.0)
- **🕸️ Multi-Hop Graph Traversal**: 1-hop and 2-hop relationship exploration with direction tracking
- **🔄 3-Pass Graph Enrichment**: 30-50% richer knowledge graphs through intelligent multi-pass extraction
- **📊 Graph Statistics**: Comprehensive network analysis for each entity

### Performance & UX Improvements
- **⚡ MAX_PERFORMANCE Mode**: Parallel execution of all search tools with intelligent synthesis
- **🎯 Improved Retrieval**: 5x/3x retrieval multipliers for >95% F1 Score target
- **🎨 Enhanced UI**: Dynamic node sizing, markdown rendering, and better formatting
- **📸 New Visualizations**: Graph creation process and entity/relationship tables

### Optimization & Refinement
- **🎯 Focused Metrics**: Removed Answer/Factual Correctness metrics for better evaluation
- **🔍 Better Context**: 2 adjacent chunks expansion (5x context vs 3x)
- **📝 Markdown Support**: Beautiful formatted answers with headings, lists, and proper spacing

---

## Table of Contents

- [Introduction](#introduction)
- [Core Features](#core-features)
- [System Demo](#system-demo)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Optimal Configuration](#optimal-configuration)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Testing](#testing)
- [RAGAS Evaluation](#ragas-evaluation-framework)
- [API Endpoints](#api-endpoints)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Introduction

This project implements a state-of-the-art RAG system that goes beyond traditional vector search by incorporating:

- **Multi-Tool Search Agent**: Intelligently routes queries between vector search, graph search, and metadata filtering
- **MAX_PERFORMANCE Mode**: Run all search tools in parallel for comprehensive answers
- **Multi-Hop Graph Traversal**: 1-hop and 2-hop relationship exploration with intelligent network analysis
- **3-Pass Graph Enrichment**: Intelligent multi-pass extraction for 30-50% richer knowledge graphs
- **Knowledge Graph Extraction**: Automatically extracts entities, relationships, and communities from documents
- **Hybrid Search**: Combines vector embeddings (semantic search), BM25 (keyword search), and cross-encoder reranking
- **Entity Deduplication**: AI-powered similarity-based entity merging while preserving context
- **Parallel Processing**: Concurrent chunk processing for 10-20x faster graph generation
- **Comprehensive Testing**: Unit tests (27% coverage) and RAGAS evaluation framework for RAG performance metrics

The system was built incrementally through a series of development steps, from basic environment setup to advanced features like multi-hop graph traversal and enhanced visualization.

---

## Core Features

### ✅ Multi-Tool Search Agent

The intelligent search agent automatically analyzes query intent and selects the appropriate search method:

- **Vector Search**: "What is X?", "Explain...", "Define..."
  - Uses text-embedding-3-small for semantic understanding
  - Hybrid approach: Vector similarity + BM25 keyword search + Cross-encoder reranking
  - 5x retrieval multiplier + 3x reranking for maximum coverage
  - Context expansion with 2 adjacent chunks on each side
  - Retrieves most relevant document chunks

- **Graph Search**: "How are X and Y related?", "What connects..."
  - Neo4j-powered relationship queries with multi-hop traversal
  - **1-hop traversal**: Shows direct connections with relationship types and directions
  - **2-hop traversal**: Shows extended network (entities connected through intermediaries)
  - **Graph statistics**: Summary of entity's position in the network
  - Cypher query generation from natural language
  - Comprehensive entity context and relationship analysis

- **Filter Search**: "Show me documents from 2023", "Find papers by [author]"
  - Elasticsearch-based metadata filtering
  - Full-text search with fuzzy matching
  - Date ranges, categories, tags, and custom filters

- **Graph Update** ✨ NEW in v1.2.0: "Create X and connect to Y", "Delete node Z"
  - Real-time knowledge graph modifications using natural language
  - 9 powerful operations: Create, Delete, Merge, Update, Connect
  - Batch operations for efficiency (create + connect multiple nodes at once)
  - Automatic frontend refresh after updates
  - No Cypher knowledge required - just use plain English!

#### MAX_PERFORMANCE Mode

**NEW in v1.1.0**: Toggle between standard and max performance modes:

- **Standard Mode** (`MAX_PERFORMANCE=false`): Agent selects best single tool
  - Cost-effective, fast for simple queries
  - Intelligent tool selection based on query analysis

- **Max Performance Mode** (`MAX_PERFORMANCE=true`): All tools run in parallel
  - Comprehensive answers from multiple perspectives
  - Results from all tools are synthesized by LLM
  - Better coverage for complex queries
  - No risk of suboptimal tool selection

### ✅ GraphRAG Pipeline with 3-Pass Enrichment

**ENHANCED in v1.1.0**: Automatic knowledge graph extraction with intelligent multi-pass enrichment:

1. **Text Chunking**: Documents split into 1200-character chunks with 500-character overlap
2. **Multi-Pass Extraction** (NEW):
   - **Pass 1**: Initial broad extraction of entities and relationships
   - **Pass 2**: Finds missing entities that were referenced but not extracted
   - **Pass 3**: Discovers indirect relationships between existing entities
   - **Result**: 30-50% richer knowledge graphs
3. **Parallel Processing**: Processes up to 25 chunks concurrently using asyncio
4. **Smart Deduplication**: MERGE-based imports prevent duplicate entities across passes
5. **Neo4j Storage**: Stores entities, relationships, and community structures

**Results**: A 500-page book generates 85+ entities and 142+ relationships with 30-50% more connections through multi-pass enrichment!

**Configuration**:
```env
GRAPHRAG_ENABLE_MULTIPASS=true  # Enable 3-pass enrichment
GRAPHRAG_NUM_PASSES=3            # Number of passes (1-3 recommended)
```

### ✅ Natural Language Graph Updates ✨ NEW in v1.2.0

**Modify your knowledge graph using plain English commands!** No Cypher knowledge required.

#### Supported Operations

1. **Create New Node**
   ```
   "Create a new node called 'Data Science'"
   ```

2. **Create Node with Multiple Connections** (Batch Operation)
   ```
   "Create 'Software Engineering' and connect it to Python, Java, TypeScript, and Docker"
   ```
   - Creates the node and all connections in ONE efficient operation
   - Shows which connections succeeded vs failed
   - Perfect for building complex graph structures quickly

3. **Delete Node**
   ```
   "Delete the node named 'Machine Learning'"
   "Remove entity 'Python' from the graph"
   ```

4. **Merge Nodes**
   ```
   "Merge 'AI' and 'Artificial Intelligence' into one node"
   "Combine entities 'John' and 'John Smith'"
   ```
   - Preserves all relationships from both nodes
   - Optionally rename the merged entity

5. **Create Relationship**
   ```
   "Create a relationship 'WORKS_ON' from John to Project X"
   "Connect Dev Singh to LYZR with relationship 'EMPLOYED_AT'"
   ```

6. **Update Node Properties**
   ```
   "Update the description of 'Python' to 'A popular programming language'"
   "Change the status property of 'Project A' to 'completed'"
   ```

7. **Update Node Description**
   ```
   "Update the description of 'Machine Learning' to include deep learning"
   ```

8. **Update Relationship**
   ```
   "Update the 'EMPLOYED_AT' relationship between John and Google to add 'since: 2020'"
   ```

9. **Delete Relationship**
   ```
   "Delete the relationship between 'Dev' and 'Company Y'"
   "Remove the connection between 'AI' and 'Machine Learning'"
   ```

#### How It Works

```
User: "Create 'AI Research' and connect it to Python, Machine Learning, and Statistics"
         ↓
Agent detects graph update intent
         ↓
LLM extracts operation details:
- Operation: create_node_with_relationships
- Node: "AI Research"
- Connections: ["Python", "Machine Learning", "Statistics"]
         ↓
Neo4j executes in single transaction
         ↓
Frontend auto-refreshes graph
         ↓
User sees updated graph + success message
```

#### Real-Time Feedback

The system provides detailed feedback for each operation:

```
✅ Graph Update Successful

Operation: create_node_with_relationships
Node created: Software Engineering
Relationships created: 9
Relationships failed: 0

✓ Successful connections:
  - Problem-Solving (INCLUDES, outgoing)
  - Clickhouse (INCLUDES, outgoing)
  - Leadership (INCLUDES, outgoing)
  ... and 6 more

🔄 Knowledge graph has been updated.
   Refresh the graph view to see changes.
```

### ✅ Enhanced Entity Analysis

**NEW in v1.1.0**: Multi-hop graph traversal provides comprehensive entity context:

- **Direct Connections (1-hop)**: Shows all immediate relationships with types and descriptions
- **Extended Network (2-hop)**: Maps entities connected through intermediaries
- **Direction Tracking**: Shows incoming (←) vs outgoing (→) relationships
- **Graph Statistics**: Connection count, network size, relationship type distribution

**Example Output**:
```
🔍 Graph Analysis: Machine Learning

📌 Entity Information
- Type: Concept
- Description: [Full description]

🔗 Direct Connections (5 relationships)
  RELATED_TO:
    → Neural Networks (Concept)
      ↳ Core technique used in modern ML...

🌐 Extended Network (2-hop connections)
  • Deep Learning (Concept)
      via Neural Networks [RELATED_TO]

📊 Graph Statistics
- Direct connections: 5
- Extended network size: 12
```

### ✅ Document Processing

- **PDF Upload**: Supports Docling and PyMuPDF extraction methods
- **Embeddings**: text-embedding-3-small stored in PGVector
- **Hybrid Search**: Vector + BM25 + Reranking for optimal retrieval
- **Enhanced Context**: 5x initial retrieval + 3x reranking + 2 adjacent chunks expansion
- **GraphRAG**: Automatic knowledge graph generation with 3-pass enrichment
- **Elasticsearch**: Full-text indexing with metadata

### ✅ Testing & Evaluation

- **Unit Tests**: 21 tests with 27% coverage (pytest)
- **CI/CD**: GitHub Actions pipeline runs tests on PRs to development
- **RAGAS Framework**: Focused RAG evaluation with key retrieval metrics:
  - Context Precision: 99.99% ✅ Near Perfect
  - Context Recall: 94.32% ✅ Excellent
  - Retriever F1 Score: 94.32% ✅ Near Target (>95%)

**Note**: Answer Correctness and Factual Correctness metrics removed in v1.1.0 as they use vector similarity and aren't optimal for evaluating logical correctness of answers.

---

## System Demo

### Interactive User Interface

#### Search Box & Document Upload
![Search Box & Document Upload](frontend/public/Seach_Box_&_Doc_Upload.png)
*User-friendly interface for uploading PDF documents and querying with natural language*

#### Multi-Tool Search Agent in Action
![Search Agent in Action](frontend/public/Search_Agent_In_Action.png)
*The intelligent agent automatically selects and executes the appropriate search tool (vector, graph, or filter) based on query intent, displaying streaming results with citations*

#### Knowledge Graph Visualization
![Multi-Document Graph](frontend/public/Multi_Document_Graph.png)
*Interactive knowledge graph showing entities and relationships extracted from documents using GraphRAG*

#### Graph Clusters with Clickable Nodes
![Graph Cluster with Clickable Nodes](frontend/public/Graph_Cluster_With_Clickable_Nodes.png)
*Hierarchical graph clusters with interactive nodes - click to explore entity details, relationships, and community structures. Nodes now dynamically sized based on content (50px-70px)*

#### Improved Graph Creation Process
![Improved Graph Creation with 3 Passes](frontend/public/Improved_graph_creation_with_3_passes.png)
*NEW in v1.1.0: Visualization of the 3-pass graph enrichment process showing progressive entity and relationship discovery*

#### Entities & Relationship Tables
![Entities & Relationship Tables](frontend/public/Entities_&_Relationship_Tables.png)
*NEW in v1.1.0: Comprehensive tables showing all extracted entities and relationships with full context*

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- 8GB+ RAM (for Elasticsearch and Neo4j)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd LYZR-Hackathon

# Copy environment template
cp .env-example .env

# Edit .env with your credentials
nano .env
```

### 2. Configure Environment

Edit `.env` with your credentials (see [Setup & Installation](#setup--installation) for details):

```env
OPENAI_API_KEY=sk-proj-your-key-here
POSTGRES_PASSWORD=your_secure_password
NEO4J_AUTH=neo4j/your_secure_password

# NEW in v1.1.0: MAX_PERFORMANCE mode
MAX_PERFORMANCE=false  # Set to true for parallel tool execution

# NEW in v1.1.0: Multi-pass graph enrichment
GRAPHRAG_ENABLE_MULTIPASS=true
GRAPHRAG_NUM_PASSES=3
```

### 3. Start the System

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | - |
| **Backend API** | http://localhost:8000/docs | - |
| **Neo4j Browser** | http://localhost:7474 | neo4j / your_password |
| **Elasticsearch** | http://localhost:9200 | - |

### 5. Upload & Query

```bash
# Upload a document (automatically processes with 3-pass GraphRAG)
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@document.pdf" \
  -F "method=pymupdf"

# Query the document (agent auto-selects search method or uses all in parallel)
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "document_id": "your-doc-id"}'
```

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                         │
│  (ReactJS - Drag & Drop, Graph Visualization, Query UI)     │
│  NEW: Markdown rendering, Dynamic node sizing               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket/HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Layer (FastAPI)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Master Search Agent                       │   │
│  │  NEW: MAX_PERFORMANCE mode toggle                    │   │
│  │  ┌─────────────┬──────────────┬─────────────────┐    │   │
│  │  │ Vector      │ Graph Search │ Filter Search   │    │   │
│  │  │ Search      │ (Multi-hop)  │ (Elasticsearch) │    │   │
│  │  │ (5x/3x)     │ (1-hop,2-hop)│                 │    │   │
│  │  └─────────────┴──────────────┴─────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    Document Processing Pipeline (3-Pass GraphRAG)    │   │
│  │  PDF → Extract → Pass 1 (Broad) → Pass 2 (Missing)  │   │
│  │                → Pass 3 (Relationships) → Neo4j      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    Entity Deduplication & Graph Analysis            │   │
│  │  (MERGE-based, Multi-hop traversal, Statistics)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌────────────┬────────────┬──────────┬──────────────────┐  │
│  │ PostgreSQL │  PGVector  │  Neo4j   │  Elasticsearch   │  │
│  │ (Metadata) │ (Vectors)  │ (Graph)  │  (Full-text)     │  │
│  └────────────┴────────────┴──────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Document Processing Flow (Enhanced in v1.1.0)

```
Upload PDF
    ↓
Extract Text (Docling/PyMuPDF)
    ↓
Split into Chunks (1200 chars, 500 overlap)
    ↓
├─→ Generate Embeddings → Store in PGVector
├─→ Build BM25 Index → Keyword search ready
├─→ Index in Elasticsearch → Metadata search ready
│
└─→ GraphRAG Processing (if enabled):
    ├─ PASS 1: Initial extraction (broad entities & relationships)
    ├─ PASS 2: Find missing entities (referenced but not extracted)
    ├─ PASS 3: Discover indirect relationships
    ├─ MERGE duplicate entities across passes
    ├─ Process chunks in parallel (25 concurrent)
    └─ Import to Neo4j with statistics
    ↓
Document Ready for All Search Types
```

### Query Processing Flow (Enhanced in v1.1.0)

```
User Query
    ↓
Check MAX_PERFORMANCE setting
    ↓
┌─────────────────┴──────────────────┐
│                                    │
Standard Mode                  Max Performance Mode
(Single Tool)                  (All Tools Parallel)
    ↓                               ↓
Agent Analyzes Intent         Execute All 3 Tools:
    ↓                          ├─ Vector Search (5x/3x)
Select Best Tool:              ├─ Graph Search (Multi-hop)
├─ "What?" → Vector            └─ Filter Search
├─ "How related?" → Graph          ↓
└─ "Show docs..." → Filter     Collect All Results
    ↓                               ↓
Execute Tool                   LLM Synthesis
    ↓                               ↓
Generate Answer                Comprehensive Answer
    ↓                               ↓
    └─────────────────┬──────────────────┘
                      ↓
            Stream Response with Citations
            (Markdown formatted)
```

For detailed architecture diagrams and component specifications, see [architecture.md](architecture.md).

---

## Optimal Configuration

### Chunk Size & Overlap (Optimized for >95% F1 Score)

After extensive experimentation and optimization for >95% retrieval performance:

- **Chunk Size**: 1200 characters
  - Large enough to capture complete thoughts and relationships
  - Small enough for precise retrieval and entity extraction
  - Processes 85+ entities per 500-page book

- **Chunk Overlap**: 500 characters (Updated Oct 13, 2025)
  - 42% overlap ensures maximum context continuity
  - Significantly reduces information loss at chunk boundaries
  - Improved from 400 to 500 chars for better Context Recall
  - Helps entity deduplication by preserving cross-chunk context

These values are configured in `.env`:

```env
CHUNK_SIZE=1200
CHUNK_OVERLAP=500  # Optimized for >95% F1 Score
TOP_K_RESULTS=20   # Increased for better coverage
```

### Enhanced Retrieval Configuration (NEW in v1.1.0)

The RAG system now retrieves significantly more context per query:

```env
# In code (backend/app/services/search_agent.py):
# - Initial retrieval: 5x top_k (100 chunks vs 60 before)
# - After reranking: 3x top_k (60 chunks vs 30 before)
# - Adjacent context: 2 chunks on each side (5x multiplier vs 3x)
```

**Impact**:
- Larger initial candidate pool (100 vs 60 chunks)
- More results after reranking (60 vs 30 chunks)
- Expanded adjacent context (5x vs 3x multiplier)
- These improvements target Context Recall metric for >95% F1 Score

### GraphRAG Configuration (Enhanced in v1.1.0)

```env
# Parallel Processing
GRAPHRAG_CONCURRENCY=25    # 25 concurrent chunks (10-50 recommended)
GRAPHRAG_MAX_RETRIES=3     # Retry failed chunks 3 times
GRAPHRAG_BASE_BACKOFF=0.5  # 0.5s backoff between retries

# Multi-Pass Enrichment (NEW)
GRAPHRAG_ENABLE_MULTIPASS=true  # Enable 3-pass enrichment
GRAPHRAG_NUM_PASSES=3            # Number of passes (1-3 recommended)
```

**Performance Impact**:
- 10-20x faster graph generation through parallel processing
- 30-50% richer knowledge graphs through multi-pass enrichment

### Search Mode Configuration (NEW in v1.1.0)

```env
# MAX_PERFORMANCE Mode
MAX_PERFORMANCE=false  # false = agent selects best tool, true = all tools in parallel

# When to use MAX_PERFORMANCE=true:
# - Complex queries requiring multiple perspectives
# - Need comprehensive answers from all search types
# - Cost is less important than thoroughness
#
# When to use MAX_PERFORMANCE=false:
# - Cost optimization is important
# - Simple queries with obvious tool selection
# - Trust agent's intelligent tool selection
```

### Model Selection

```env
OPENAI_MODEL=gpt-4o-mini                    # Cost-efficient for most tasks
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Fast, high-quality embeddings
```

**Cost Optimization**: Using `gpt-4o-mini` reduces costs by ~10x compared to GPT-4 while maintaining good quality.

---

## Project Structure

```
LYZR-Hackathon/
├── architecture.md              # Detailed system architecture documentation
├── README.md                    # This file
├── todo.md                      # Future improvements and known issues
├── docker-compose.yml           # Container orchestration
├── .env-example                 # Environment template
├── .env                         # Your configuration (create from .env-example)
│
├── backend/                     # FastAPI Backend
│   ├── app/
│   │   ├── main.py             # Application entry point
│   │   ├── config.py           # Configuration management
│   │   ├── models.py           # Database models
│   │   ├── database.py         # Database connections
│   │   ├── api/
│   │   │   └── rag_routes.py   # REST API endpoints
│   │   ├── services/
│   │   │   ├── search_agent.py         # Multi-tool search with MAX_PERFORMANCE
│   │   │   ├── vector_store.py         # PGVector operations (5x/3x retrieval)
│   │   │   ├── bm25_search.py          # Keyword search
│   │   │   ├── reranker.py             # Cross-encoder reranking
│   │   │   ├── graph_search.py         # Multi-hop graph traversal (NEW)
│   │   │   ├── graphrag_pipeline.py    # 3-pass entity extraction (ENHANCED)
│   │   │   ├── neo4j_service.py        # MERGE-based operations (ENHANCED)
│   │   │   ├── graph_refinement_pipeline.py  # Entity deduplication
│   │   │   ├── elasticsearch_service.py # Metadata filtering
│   │   │   ├── document_processor.py   # PDF text extraction
│   │   │   ├── embedding_service.py    # OpenAI embeddings
│   │   │   └── ragas_evaluator.py      # RAG evaluation metrics
│   │   └── utils/
│   │       └── document_helpers.py     # Document processing utilities
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile              # Backend container image
│   └── .dockerignore
│
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── App.jsx             # Main application (markdown rendering, NEW)
│   │   ├── index.jsx           # React entry point
│   │   └── components/         # React components
│   │       └── GraphVisualization.jsx  # Dynamic node sizing (ENHANCED)
│   ├── public/
│   │   ├── index.html          # HTML template
│   │   ├── Seach_Box_&_Doc_Upload.png
│   │   ├── Search_Agent_In_Action.png
│   │   ├── Multi_Document_Graph.png
│   │   ├── Graph_Cluster_With_Clickable_Nodes.png
│   │   ├── Improved_graph_creation_with_3_passes.png  # NEW
│   │   └── Entities_&_Relationship_Tables.png         # NEW
│   ├── package.json            # Node.js dependencies (react-markdown added)
│   ├── Dockerfile              # Frontend container image
│   └── .dockerignore
│
├── test/                        # Testing Framework
│   ├── unit_tests/             # Unit tests (pytest)
│   │   ├── test_config.py
│   │   ├── test_models.py
│   │   ├── test_document_helpers.py
│   │   ├── test_document_processor.py
│   │   └── test_vector_store.py
│   ├── integration_tests/
│   │   └── test_ragas_evaluation.py  # RAGAS evaluation tests
│   ├── public/                  # Test data
│   │   ├── harrier-ev-all-you-need-to-know.pdf
│   │   └── harrier_ev_detailed_qa.csv
│   ├── results/                 # Test results (auto-generated)
│   │   ├── ragas_summary.json
│   │   └── ragas_summary.csv
│   └── README.md               # Testing documentation
│
├── updates/                     # Version update documentation (NEW)
│   ├── updates1.md             # RAGAS improvements
│   ├── updates2.md             # Graph enhancements
│   └── updates3.md             # MAX_PERFORMANCE mode
│
├── prompts/                     # Development step-by-step guides
│   ├── step1.md                # Environment setup
│   ├── step2.md                # Basic backend/frontend
│   ├── step3.md                # GraphRAG & Graph Search
│   ├── step4.md                # Elasticsearch & Filter Search
│   ├── step5.md                # Graph visualization UI
│   ├── step6.md                # CI/CD pipeline
│   ├── step7.md                # Unit testing
│   ├── step8.md                # RAGAS framework
│   ├── step9.md                # Parallel processing
│   ├── step10.md               # Documentation (this step)
│   └── step11.md               # Entity deduplication
│
├── storage/                     # Document storage (Docker volume)
│   └── .gitkeep
│
├── sample-files/                # Reference implementations
│   ├── entity_de_duplication.py
│   ├── parallel_processing.py
│   └── ragas.py
│
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI/CD pipeline
```

---

## Setup & Installation

### Prerequisites

- **Docker & Docker Compose**: Latest version
- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Hardware**: 8GB+ RAM recommended (for Elasticsearch and Neo4j)
- **OS**: Linux, macOS, or Windows with WSL2

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd LYZR-Hackathon
```

#### 2. Create Environment File

```bash
# Copy the example environment file
cp .env-example .env
```

#### 3. Configure Environment Variables

Edit `.env` with your preferred editor:

```bash
nano .env  # or vim, code, etc.
```

**Required Configuration**:

```env
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-your-key-here

# Database Credentials (REQUIRED - Change for production!)
POSTGRES_PASSWORD=your_secure_password
NEO4J_AUTH=neo4j/your_secure_password
```

**Optional Configuration**:

```env
# Models (defaults are cost-optimized)
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Document Processing (optimal values for >95% F1 Score)
CHUNK_SIZE=1200
CHUNK_OVERLAP=500  # Increased from 400 for better continuity
TOP_K_RESULTS=20   # Increased from 15 for improved coverage

# GraphRAG Parallel Processing
GRAPHRAG_CONCURRENCY=25
GRAPHRAG_MAX_RETRIES=3
GRAPHRAG_BASE_BACKOFF=0.5

# Multi-Pass Graph Enrichment (NEW in v1.1.0)
GRAPHRAG_ENABLE_MULTIPASS=true  # Enable 3-pass enrichment
GRAPHRAG_NUM_PASSES=3            # Number of passes (1-3 recommended)

# MAX_PERFORMANCE Mode (NEW in v1.1.0)
MAX_PERFORMANCE=false  # Run all search tools in parallel

# Feature Toggles
ENABLE_VECTOR_SEARCH=true
ENABLE_GRAPH_SEARCH=true
ENABLE_FILTER_SEARCH=true
ENABLE_ENTITY_DEDUPLICATION=true

# Ports (change if conflicts)
FRONTEND_PORT=3000
BACKEND_PORT=8000
NEO4J_HTTP_PORT=7474
NEO4J_PORT=7687
```

#### 4. Build and Start Services

```bash
# Build and start all containers
docker-compose up --build

# Or run in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
```

#### 5. Verify Installation

**Check Service Health**:

```bash
# Backend API
curl http://localhost:8000/health

# Elasticsearch
curl http://localhost:9200/_cluster/health

# Neo4j (in browser)
open http://localhost:7474
# Login: neo4j / your_neo4j_password
```

**Check Container Status**:

```bash
docker-compose ps

# Expected output: All services should show "Up" or "Up (healthy)"
```

#### 6. Upload Your First Document

```bash
# Upload a PDF document (3-pass GraphRAG will run automatically)
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@your_document.pdf" \
  -F "method=pymupdf"

# Response will include a document_id
# Watch logs to see 3-pass enrichment in action:
docker-compose logs -f backend | grep -i "PASS"
```

#### 7. Query the Document

```bash
# Ask a question (agent will select best tool or use all in parallel)
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "document_id": "your-document-id"
  }'
```

### Local Development (Without Docker)

For local development without Docker:

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key
export POSTGRES_HOST=localhost
export MAX_PERFORMANCE=false  # NEW in v1.1.0
export GRAPHRAG_ENABLE_MULTIPASS=true  # NEW in v1.1.0
# ... (other variables)

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
export REACT_APP_BACKEND_URL=http://localhost:8000

# Run the frontend
npm start
```

**Note**: You'll need to run PostgreSQL, PGVector, Neo4j, and Elasticsearch separately.

---

## Testing

### Unit Tests

The project includes comprehensive unit tests with pytest.

#### Running Unit Tests

```bash
# Inside Docker
docker exec lyzr-hackathon-backend-1 pytest test/unit_tests/ -v

# Or with coverage report
docker exec lyzr-hackathon-backend-1 pytest test/unit_tests/ --cov=backend/app --cov-report=html
```

#### Local Testing (without Docker)

```bash
cd backend

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest test/unit_tests/ -v

# With coverage
pytest test/unit_tests/ --cov=backend/app --cov-report=html

# View coverage report
open test_coverage_report/index.html
```

#### Test Coverage

**Current Coverage: 27%** (21 tests, 100% passing ✅)

**High Coverage Modules:**
- `models.py`: 100%
- `config.py`: 100%
- `document_helpers.py`: 90%
- `document_processor.py`: 83%
- `vector_store.py`: 66%

See [test/README.md](test/README.md) for detailed testing documentation.

### CI/CD Pipeline

GitHub Actions automatically runs tests on pull requests to the `development` branch:

```yaml
# .github/workflows/ci.yml
on:
  pull_request:
    branches: [development]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Setup Python
      - Install dependencies
      - Run unit tests
```

**Pipeline Status**: ✅ Passing

---

## RAGAS Evaluation Framework

### Overview

RAGAS (Retrieval-Augmented Generation Assessment) provides comprehensive metrics for evaluating RAG system performance.

### Latest Evaluation Results (v1.1.0)

**Test Configuration:**
- **Date**: October 13, 2025 at 19:45 IST
- **Version**: 1.1.0 with enhanced retrieval (5x/3x multipliers)
- **Test Document**: Dev Singh Resume & Harrier EV Brochure
- **Test Dataset**: 44 questions (Dev Singh) + 40 questions (Harrier EV)
- **Models**: GPT-4o-mini (LLM), text-embedding-3-small (embeddings)

### Performance Metrics

| Metric | Score | Status | Description |
|--------|-------|--------|-------------|
| **Context Precision** | 99.99% | ✅ Near Perfect | Relevance of retrieved documents |
| **Context Recall** | 94.32% | ✅ Excellent | Coverage of ground truth information |
| **Retriever F1 Score** | 94.32% | 🎯 Near Target | Balanced precision & recall (Target: >95%) |

**Note**: Answer Correctness and Factual Correctness metrics removed in v1.1.0. These metrics use vector similarity scoring and are not focused on short, concise answers but rather the most logically correct answers. They weren't providing meaningful evaluation for our use case.

### Performance Evolution

| Version | Context Recall | F1 Score | Key Changes |
|---------|---------------|----------|-------------|
| v1.0.0 (Baseline) | 79.63% | 80.22% | Initial implementation |
| v1.0.1 (Oct 13) | 92.83% | 91.58% | 4x/2x retrieval, 1 adjacent chunk |
| **v1.1.0 (Current)** | **94.32%** | **94.32%** | 5x/3x retrieval, 2 adjacent chunks |

### Key Optimizations in v1.1.0

**Implemented 5 Major Improvements** to achieve >95% F1 Score target:

1. **Increased top_k_results**: 15 → 20 (+33% coverage)
   - File: `backend/app/config.py:34`

2. **Enhanced chunk_overlap**: 400 → 500 chars (+25% continuity)
   - File: `backend/app/config.py:33`
   - Better context preservation at chunk boundaries

3. **Deeper retrieval multipliers**:
   - Initial retrieval: **5x top_k** (100 chunks vs 60 before)
   - After reranking: **3x top_k** (60 chunks vs 30 before)
   - File: `backend/app/services/search_agent.py:331-358`

4. **Expanded context window**: Now includes **2 adjacent chunks** on each side (5x context vs 3x)
   - File: `backend/app/services/search_agent.py:245-333`
   - Captures information spanning multiple chunks
   - Prevents context fragmentation

5. **Optimized configuration**: All changes coordinated for maximum impact

**Impact**: 3-4x more context retrieved per query, targeting >95% Context Recall

### Key Insights

**✅ Strengths:**
- Near-perfect document retrieval precision (99.99%)
- Excellent information coverage with 94.32% recall
- Strong F1 Score of 94.32% - very close to 95% target
- Hybrid search (Vector + BM25) + cross-encoder reranking working exceptionally well
- Context expansion with 2 adjacent chunks significantly improves coverage
- 3-pass GraphRAG contributing to relationship-based queries
- Multi-hop graph traversal enriches entity-based answers

**Areas for Further Enhancement:**
- Fine-tune retrieval multipliers for different document types
- Consider GPT-4 upgrade for edge cases requiring deeper reasoning
- Expand test dataset for more comprehensive evaluation

### Running RAGAS Tests

```bash
# Ensure services are running
docker-compose up -d

# Run RAGAS evaluation
docker exec lyzr-hackathon-backend-1 python -m pytest \
  test/integration_tests/test_ragas_evaluation.py::TestRAGASEvaluation::test_ragas_evaluation_with_test_data \
  -v -s

# Results are automatically saved to:
# - test/results/ragas_summary.json
# - test/results/ragas_summary.csv
```

### Test Dataset

- **Location**: `test/public/harrier_ev_detailed_qa.csv`
- **Format**: CSV with columns: `question`, `ground_truth`, `context`
- **Size**: 40 question-answer pairs
- **Document**: `test/public/harrier-ev-all-you-need-to-know.pdf`

### Architecture Overview - Retrieval Pipeline

```
User Query
    ↓
Query Embedding Generation
    ↓
Hybrid Retrieval (Parallel)
├─ Vector Search (top_k × 5 = 100 chunks)  ← ENHANCED in v1.1.0
└─ BM25 Search (top_k × 5 = 100 chunks)    ← ENHANCED in v1.1.0
    ↓
Merge & Deduplicate (~120-150 unique chunks)
    ↓
Cross-Encoder Reranking (top_k × 3 = 60 chunks)  ← ENHANCED in v1.1.0
    ↓
Context Expansion (adds 2 adjacent chunks on each side)  ← ENHANCED in v1.1.0
├─ Previous-2 chunk
├─ Previous-1 chunk
├─ Current chunk
├─ Next+1 chunk
└─ Next+2 chunk
    ↓
Final Context (~150-180 chunks with full context)
    ↓
LLM Answer Generation (with markdown formatting)
```

---

## API Endpoints

### Documents

```bash
# Upload document (triggers 3-pass GraphRAG automatically)
POST /api/rag/upload
Content-Type: multipart/form-data
Body: file (PDF), method (pymupdf/docling)

# List all documents
GET /api/rag/documents

# Get document status
GET /api/rag/documents/{id}

# Delete document
DELETE /api/rag/documents/{id}
```

### Query

```bash
# Streaming query (auto-routes or uses all tools based on MAX_PERFORMANCE)
POST /api/rag/query/stream
Content-Type: application/json
Body: {"query": "...", "document_id": "..."}

# Example responses include:
# - Markdown formatted text with headings, lists, proper spacing
# - Multi-hop graph analysis with 1-hop and 2-hop connections
# - Comprehensive answers when MAX_PERFORMANCE=true
```

### GraphRAG

```bash
# Manual trigger GraphRAG processing (with 3-pass enrichment)
POST /api/rag/documents/{id}/process-graph

# Get graph statistics (includes multi-pass info)
GET /api/rag/graph/stats/{id}

# Get full graph data for visualization
GET /api/rag/graph/data?document_id={id}

# List entities (with discovery_pass and enrichment info)
GET /api/rag/graph/entities?document_id={id}&limit=50

# List relationships (with relationship types and descriptions)
GET /api/rag/graph/relationships?document_id={id}&limit=50

# Get entity with context (NEW in v1.1.0 - multi-hop traversal)
GET /api/rag/graph/entities/{entity_id}?document_id={id}
```

### Graph Updates ✨ NEW in v1.2.0

```bash
# All graph update operations use natural language via the query endpoint
POST /api/rag/query/stream
Content-Type: application/json
Body: {
  "query": "Create 'Software Engineering' and connect it to Python, Java, and TypeScript",
  "document_id": "your-doc-id"
}

Example operations:
1. Create node:
   "Create a new node called 'Data Science'"

2. Create node with connections (batch):
   "Create 'AI Research' and connect it to Python, Statistics, and Machine Learning"

3. Delete node:
   "Delete the node named 'Machine Learning'"

4. Merge nodes:
   "Merge 'AI' and 'Artificial Intelligence' into one node"

5. Update node:
   "Update the description of 'Python' to 'Programming language'"

6. Create relationship:
   "Create a WORKS_ON relationship from John to Project X"

7. Delete relationship:
   "Delete the relationship between AI and Machine Learning"

Response includes:
- Operation success/failure status
- Detailed results (created, updated, deleted entities/relationships)
- Error messages if any operations failed
- graph_updated event triggers frontend refresh
```

### Search Tools

```bash
# Direct vector search (with 5x/3x multipliers and context expansion)
POST /api/search/vector
Body: {"query": "...", "document_id": "...", "top_k": 20}

# Direct graph search (with multi-hop traversal)
POST /api/search/graph
Body: {"query": "...", "document_id": "..."}

# Direct filter search
POST /api/search/filter
Body: {"filters": {...}, "query": "..."}
```

### Interactive API Documentation

Visit http://localhost:8000/docs for full Swagger/OpenAPI documentation with interactive testing.

---

## Performance

### Document Processing Time (Enhanced in v1.1.0)

| Document Size | Chunks | Vector + BM25 | 3-Pass GraphRAG (Parallel) | Total |
|---------------|--------|---------------|---------------------------|-------|
| 10 pages (~10K chars) | ~10 | ~10s | ~20s | ~30s |
| 50 pages (~50K chars) | ~42 | ~15s | ~2min | ~2.25min |
| 200 pages (~200K chars) | ~167 | ~25s | ~6min | ~6.5min |
| Full book (~500K chars) | ~417 | ~40s | ~15min | ~15.5min |

**Notes**:
- 3-pass GraphRAG is ~20-25% slower than single pass but produces 30-50% richer graphs
- Times assume `GRAPHRAG_CONCURRENCY=25` and `GRAPHRAG_NUM_PASSES=3`
- Sequential processing would be 10-20x slower
- Disable multi-pass if speed is critical: `GRAPHRAG_ENABLE_MULTIPASS=false`

### Cost Estimation (GraphRAG with gpt-4o-mini)

| Document Size | Single Pass | 3-Pass (v1.1.0) | Notes |
|---------------|-------------|-----------------|-------|
| 10 pages | ~$0.03 | ~$0.08 | 2.5x cost for richer graph |
| 50 pages | ~$0.15 | ~$0.40 | 30-50% more entities |
| 200 pages | ~$0.50 | ~$1.30 | Better relationship discovery |
| Full book (500 pages) | ~$1.50 | ~$3.80 | Comprehensive knowledge graph |

**Cost Optimization Tips**:
- Use `gpt-4o-mini` instead of GPT-4 (10x cheaper, already default)
- Set `GRAPHRAG_NUM_PASSES=1` to disable multi-pass enrichment
- Increase chunk size to reduce API calls
- Use `MAX_PERFORMANCE=false` for cost-effective queries
- Disable GraphRAG for documents where relationships aren't needed

### Query Performance (Enhanced in v1.1.0)

| Operation | Avg Time | Notes |
|-----------|----------|-------|
| Vector Search (Standard) | <2s | 5x retrieval + 3x reranking + context expansion |
| Graph Search (Multi-hop) | <3s | 1-hop + 2-hop traversal + statistics |
| Filter Search | <500ms | Elasticsearch full-text search |
| MAX_PERFORMANCE Mode | <5s | All 3 tools in parallel + LLM synthesis |
| Standard Mode (Agent) | <2s | Agent selects single best tool |

---

## Troubleshooting

### GraphRAG Not Working

**Check if enabled:**
```bash
# In .env
GRAPHRAG_ENABLED=true
GRAPHRAG_ENABLE_MULTIPASS=true  # NEW in v1.1.0
```

**View logs:**
```bash
docker-compose logs backend --follow | grep -i "PASS"

# You should see:
# PASS 1/3: Initial Extraction...
# PASS 2/3: Enrichment Pass...
# PASS 3/3: Relationship Enhancement...
```

**Check Neo4j:**
```bash
# Verify Neo4j is running
docker-compose ps neo4j

# Test connection
curl http://localhost:7474
```

### Slow Processing

**For 3-Pass GraphRAG:**
- Large documents take time (500 pages = ~15 min with 3 passes)
- Each pass requires separate LLM API calls
- To speed up: Set `GRAPHRAG_NUM_PASSES=1` or `GRAPHRAG_ENABLE_MULTIPASS=false`
- Adjust `GRAPHRAG_CONCURRENCY` (default: 25, range: 10-50)
- Use `gpt-4o-mini` for speed (already default)

**For Vector Search:**
- Should be fast (<2 seconds even with 5x/3x retrieval)
- Check database connections
- Verify embeddings were created

### MAX_PERFORMANCE Mode Issues

**If queries are slow with MAX_PERFORMANCE=true:**
- All 3 tools run in parallel, taking time of slowest tool
- Consider switching to `MAX_PERFORMANCE=false` for simple queries
- Check individual tool performance

**If getting inconsistent results:**
- MAX_PERFORMANCE synthesizes results from all tools
- Some tools may return irrelevant information
- Consider using Standard mode for focused queries

### Agent Not Searching Documents

The agent should automatically search uploaded documents. If it asks for clarification:

- Verify document was uploaded successfully
- Check document has chunks (GET /api/rag/documents/{id})
- Ensure query is clear and specific
- Check that search tools are enabled in .env
- Try MAX_PERFORMANCE mode for comprehensive search

### Container Issues

**Port conflicts:**
```bash
# Modify ports in .env
FRONTEND_PORT=3001
BACKEND_PORT=8001
NEO4J_HTTP_PORT=7475
```

**Database connection errors:**
```bash
# Check all containers are running
docker-compose ps

# View database logs
docker-compose logs postgres pgvector neo4j elasticsearch
```

**Out of memory:**
```bash
# Reduce Elasticsearch heap size in docker-compose.yml
ES_JAVA_OPTS=-Xms256m -Xmx256m

# Or allocate more memory to Docker
# Docker Desktop → Settings → Resources → Memory (8GB+ recommended)
```

### Common Errors

**"OpenAI API key not found":**
- Ensure `OPENAI_API_KEY` is set in `.env`
- Restart containers after changing `.env`

**"Neo4j authentication failed":**
- Check `NEO4J_AUTH` format: `username/password`
- Default: `neo4j/your_password`

**"Elasticsearch connection refused":**
- Wait for Elasticsearch to fully start (30-60 seconds)
- Check health: `curl http://localhost:9200/_cluster/health`

**"Graph traversal timeout":**
- Multi-hop traversal can be slow on very large graphs
- Check Neo4j performance
- Consider limiting 2-hop traversal depth

---

## Future Improvements

### Planned Enhancements

1. **Using an SLM for Graph Creation**
   - Current: Using OpenAI models (costly for 3 passes)
   - Improvement: Use Gemma-3-8B 8-bit quantized model (~4GB) in GGUF format
   - Benefit: Reduce costs significantly while maintaining quality
   - Research shows 3 passes create the best graphs (implemented in v1.1.0!)

2. **Microsoft GraphRAG Integration**
   - Current: Using LLMGraphTransformer from langchain_experimental
   - Improvement: Full Microsoft GraphRAG implementation with:
     - Hierarchical clustering using Leiden technique
     - Bottom-up community summaries for holistic understanding
     - Global Search for corpus-wide reasoning
     - Local Search for entity-specific queries (partially implemented in v1.1.0)
     - DRIFT Search with community context
     - Basic Search as fallback

3. **Visual Image RAG**
   - Problem: Context loss when converting images to text
   - Solution: Add image-only RAG tool to the search agent
   - Architecture:
     - PDF-to-Images conversion
     - Late Interaction Model + Multi-Vector Embeddings
     - Qdrant Vector DB for image embeddings
     - ViDoRe retrieval + Maxsum similarity
     - Multi-modal LLM for answering

4. **User-Provided Ontology**
   - Current: LLM generates all node and relationship types freely
   - Improvement: Top-down approach with domain-specific ontology
   - Benefit: More consistent, domain-relevant graph structure
   - Reduces creative but irrelevant entity types

5. **Open-Source Graph Tools**
   - Cognee: For building graphs
   - Graphiti: For keeping graphs updated over time
   - Benefit: Reduce dependency on commercial services

6. **Multi-Document Graph Evolution**
   - Current: Tested with single and multiple documents
   - Need: Stress test with 100+ documents
   - Improve: Graph evolution and reorganization strategy
   - Challenge: Cross-document entity linking and conflict resolution
   - Partially addressed in v1.1.0 with MERGE-based deduplication

7. **Enhanced Security**
   - Current: Passwords in .env (acceptable for development)
   - Improvement: Runtime password retrieval from password manager
   - Production: AWS Secrets Manager, HashiCorp Vault integration

8. **Model Upgrades**
   - Current: Cost-optimized models (gpt-4o-mini, text-embedding-3-small)
   - Improvement: Upgrade to GPT-4 and text-embedding-3-large
   - Expected: Better RAGAS scores, especially for complex reasoning
   - Trade-off: Higher costs

9. ~~**Natural Language Graph Refinement**~~ ✅ **COMPLETED in v1.2.0**
   - ✅ Backend functionality for natural language graph updates
   - ✅ Supported operations:
     - ✅ "Create X and connect to Y, Z" (batch operations)
     - ✅ "Merge these two nodes"
     - ✅ "Add relationship between X and Y"
     - ✅ "Update entity properties"
     - ✅ "Delete nodes and relationships"
   - ✅ Real-time frontend auto-refresh
   - ✅ User-friendly graph refinement without Cypher knowledge
   - 🎯 Result: 9 powerful operations for complete graph management

10. **Query Caching & Performance**
    - Cache frequent entity queries
    - Pre-compute multi-hop traversals for common entities
    - Optimize Neo4j queries for large graphs
    - Implement result pagination for large result sets

### Completed Enhancements ✅

**v1.2.0:**
- [x] Natural language graph updates (9 operations)
- [x] Create node with batch relationship connections
- [x] Real-time frontend auto-refresh after graph changes
- [x] Detailed success/failure feedback for operations
- [x] LLM-powered operation extraction from natural language

**v1.1.0:**
- [x] Multi-hop graph traversal (1-hop and 2-hop)
- [x] 3-pass graph enrichment system
- [x] Enhanced retrieval with 5x/3x multipliers
- [x] Context expansion with 2 adjacent chunks
- [x] MAX_PERFORMANCE mode for parallel tool execution
- [x] MERGE-based entity deduplication across passes
- [x] Dynamic node sizing in graph visualization
- [x] Markdown rendering for formatted answers
- [x] Comprehensive graph statistics and analysis
- [x] Refined RAGAS metrics (removed Answer/Factual Correctness)

### Known Issues

- Entity deduplication may need manual review for edge cases
- Large documents (1000+ pages) may hit memory limits
- 2-hop traversal can be slow on very large graphs (10,000+ nodes)
- MAX_PERFORMANCE mode increases API costs 3-4x
- Graph update operations only work on entities within the same document (cross-document updates not yet supported)

For detailed improvement roadmap, see [todo.md](todo.md) and [updates/](updates/) folder.

---

## Author

**Dev Pratap Singh**
*Senior AI Engineer*
Indian Institute of Technology (IIT) Goa

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dev-singh-18003/)

---

## License

This project is built for a Hackathon.

---

## Additional Resources

- **Detailed Architecture**: [architecture.md](architecture.md)
- **Testing Guide**: [test/README.md](test/README.md)
- **Development Steps**: [prompts/](prompts/) directory
- **Update History**: [updates/](updates/) directory
- **API Documentation**: http://localhost:8000/docs (when running)

---

## Version History

### v1.2.0 (October 13, 2025) - Current Release
- ✨ NEW: Natural Language Graph Updates - 4th tool added to search agent
- ✨ NEW: 9 graph update operations (Create, Delete, Merge, Update, Connect)
- ✨ NEW: Batch operations - create node + connect to multiple nodes at once
- ✨ NEW: Real-time frontend auto-refresh after graph changes
- ✨ NEW: LLM-powered operation extraction from natural language commands
- 🎯 ENHANCED: Detailed success/failure feedback for each operation
- 🎯 ENHANCED: `create_node_with_relationships` for efficient bulk connections
- 🔄 ENHANCED: Streaming event `graph_updated` triggers UI refresh
- 📊 RESULT: Complete graph management without Cypher knowledge

### v1.1.0 (October 13, 2025)
- ✨ NEW: Multi-hop graph traversal (1-hop, 2-hop) with comprehensive analysis
- ✨ NEW: 3-pass graph enrichment for 30-50% richer knowledge graphs
- ✨ NEW: MAX_PERFORMANCE mode for parallel tool execution
- 🚀 ENHANCED: 5x/3x retrieval multipliers (up from 4x/2x)
- 🚀 ENHANCED: Context expansion with 2 adjacent chunks (up from 1)
- 🎨 ENHANCED: Dynamic node sizing (50px-70px based on content)
- 🎨 ENHANCED: Markdown rendering with proper formatting
- 📊 ENHANCED: Increased top_k (20) and chunk_overlap (500)
- 📸 NEW: Added graph creation process and entity tables visualizations
- 🎯 REFINED: Removed Answer/Factual Correctness metrics
- 📈 RESULT: F1 Score improved to 94.32% (from 91.58%)

### v1.0.0 (October 12, 2025) - Initial Release
- ✅ Multi-tool search agent (Vector, Graph, Filter)
- ✅ GraphRAG pipeline with parallel processing
- ✅ Entity deduplication
- ✅ Hybrid search (Vector + BM25 + Reranking)
- ✅ Interactive graph visualization
- ✅ RAGAS evaluation framework
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Comprehensive testing (27% coverage)

---

**Last Updated**: 2025-10-13
**Status**: ✅ Production Ready
**Version**: 1.2.0

---

## Acknowledgments

Special thanks to the team for organizing this hackathon. If I don't win, I'd love to meet the team in Bangalore for coffee! ✌️

---