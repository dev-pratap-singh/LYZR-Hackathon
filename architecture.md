# RAG System Architecture

## Overview

This document outlines the architecture for an advanced Retrieval-Augmented Generation (RAG) system that combines vector search, graph-based retrieval, and filtered search capabilities to provide comprehensive document querying and analysis.

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                         │
│  (ReactJS - Drag & Drop, Graph Visualization, Query UI)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket/HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend Layer (FastAPI)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Master Search Agent                        │  │
│  │  ┌─────────────┬──────────────┬─────────────────┐   │  │
│  │  │ Vector      │ Graph Search │ Filter Search   │   │  │
│  │  │ Search      │ (Cypher/     │ (ElasticSearch) │   │  │
│  │  │             │  Gremlin)    │                 │   │  │
│  │  └─────────────┴──────────────┴─────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         Document Processing Pipeline                 │ │
│  │  PDF → Docling → .txt → GraphRAG → Graph DB          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Entity Resolution & Deduplication            │  │
│  │  (BFS Graph Traversal Agent)                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌────────────┬────────────┬──────────┬──────────────────┐ │
│  │ PostgreSQL │  PGVector  │  Neo4j   │  ElasticSearch   │ │
│  │ (User DB)  │ (Vectors)  │ (Graph)  │  (Filter Search) │ │
│  └────────────┴────────────┴──────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│        Container Volume / S3 (Production)                    │
└─────────────────────────────────────────────────────────────┘
```

### High-Level Components

- **Frontend**: ReactJS-based interface
- **Backend**: FastAPI server
- **Databases**: PostgreSQL, PGVector, Neo4J, ElasticSearch
- **Storage**: Container Volume (Development)
- **CI/CD**: Automated testing and deployment pipeline

---

## Data Flow

### Document Ingestion Pipeline

1. **Upload**: User uploads PDF documents for the data corpus
2. **Storage**: 
   - Development: Store in Container Volume (accessible to all containers)
3. **Database Entry**: Store query data and document entries with unique IDs in user database
4. **Processing**: Convert PDFs to text markup files using Docling (IBM)

### Query Processing Pipeline

1. User submits query via frontend
2. Master Search Agent processes the query
3. Agent selects appropriate search tool(s):
   - Vector Search
   - Graph Search (Cypher/Gremlin generation)
   - Filter Search
   - Combination of above methods
4. Results aggregated and streamed back to user via WebSockets

#### Master Search Agent Decision Flow

```
START: New Query
    ↓
┌───────────────────────────────┐
│ Does it ask                   │
│ "what/explain/define"?        │
└───────────────────────────────┘
         ↓ YES                ↓ NO
    vector_search              │
                               ↓
                    ┌──────────────────────────┐
                    │ Does it ask about        │
                    │ relationships/           │
                    │ connections?             │
                    └──────────────────────────┘
                         ↓ YES            ↓ NO
                    graph_search           │
                                          ↓
                            ┌──────────────────────────┐
                            │ Does it mention          │
                            │ dates/categories/        │
                            │ authors/filters?         │
                            └──────────────────────────┘
                                 ↓ YES            ↓ NO
                            filter_search          │
                                                  ↓
                                    ┌──────────────────────┐
                                    │ Is it complex/       │
                                    │ multi-faceted?       │
                                    └──────────────────────┘
                                         ↓ YES        ↓ NO
                                    Multiple Tools    Re-analyze
```

---

## Component Details

### Frontend Features

**User Interface Components:**
- Drag-and-drop box for PDF uploads
- Visual graph display
- Graph structure data viewer (nodes/edges/metadata/clusters/hierarchy/summary)
- Interactive graph visualization with natural language ontology refinement recommendations
- Query input box
- Streaming answer display box
- Reasoning chain visualization (retrieved docs → chain of thought → final answer)

**Technology Stack:**
- ReactJS
- WebSocket integration for real-time streaming
- Graph visualization (Gephi integration)

### Backend Architecture

**API Framework:**
- FastAPI for REST endpoints and WebSocket connections

**Core Pipelines:**

1. **Data Ingestion Pipeline**
   - Use Docling to convert PDFs to .txt markup files
   - Save files to container volume
   - Update user table with document metadata

2. **Graph Generation Pipeline**
   - Process .txt files using GraphRAG
   - Generate: nodes, edges, metadata, clusters, hierarchy, summary
   - Pre-process GraphRAG output for Neo4J storage

3. **Graph Optimization Pipeline**
   - Entity Resolution & De-duplication
   - AI Agent performs BFS graph traversal
   - Collect and merge similar nodes
   - Combine edges while preserving context
   - Store cleaned graph in Neo4J

4. **Query Processing Pipeline**
   - Master Search Agent with multi-tool capabilities
   - WebSocket-based streaming responses
   - Reasoning chain generation and display

**Search Tools:**

1. **Semantic Search Tool**
   - Process: .txt → chunks → OpenAI Embeddings → PGVector storage
   - Langchain implementation with:
     - Top-k retrieval
     - Re-ranker
     - BM25
     - Hybrid (keyword + vector) search

2. **Graph Search Tool**
   - Query Neo4J using Langchain
   - Cypher/Gremlin query generation
   - Versioned ontology support
   - Optional: Export graph via GraphRAG settings.yml

3. **Filter Search Tool**
   - ElasticSearch on Docling-generated .txt files
   - Structured query support

### Database Schema

**PostgreSQL (User Database)**
- User authentication and profile data
- Document entry metadata
- Unique document IDs and query logs

**PGVector (Embeddings Database)**
- Document chunks
- OpenAI embeddings
- Vector similarity search indices

**Neo4J (Graph Database)**
- Nodes and edges from GraphRAG
- Entity metadata
- Cluster information
- Hierarchical relationships
- Summary data

---

## Container Architecture

### Container Services

1. **Frontend Container**
   - Technology: ReactJS
   - Purpose: User interface and visualization

2. **Backend Container**
   - Technology: FastAPI
   - Purpose: API server and business logic

3. **PostgreSQL Container**
   - Purpose: User and metadata storage

4. **PGVector Container**
   - Purpose: Vector embeddings storage

5. **Neo4J Container**
   - Purpose: Graph database

6. **Storage Volume**
   - Development: Shared container volume
   - Purpose: PDF and processed file storage

---

## Implementation Details

### Streaming Response Implementation

```python
from openai import OpenAI
client = OpenAI()

with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You're a helpful QA assistant."},
        {"role": "user", "content": f"Answer based on: {context}\n\nQuestion: {query}"}
    ],
) as stream:
    for event in stream:
        if event.type == "message.delta":
            print(event.delta, end="", flush=True)
```

### Graph Visualization

- **Tool**: Gephi
- **Data Source**: Neo4J database
- **Purpose**: Interactive graph exploration and visualization

### Ontology Management

- Versioned ontology support
- Natural language refinement suggestions from frontend
- AI-assisted entity resolution

---

## Testing & Quality Assurance

### Testing Strategy

1. **Unit Tests**
   - Component-level testing
   - Function validation

2. **Integration Tests**
   - End-to-end pipeline testing
   - API endpoint validation
   - Database interaction tests

3. **Retrieval Testing**
   - RAGAS metrics for performance evaluation
   - Ground-truth preparation using LLM
   - Retrieval quality assessment

### CI/CD Pipeline

- Automated testing on commits
- Continuous integration
- Deployment automation

---

## Additional Resources

### Integration References

**GraphRAG + Neo4j + Langchain/LlamaIndex:**
- [Microsoft GraphRAG with Neo4j](https://neo4j.com/blog/developer/microsoft-graphrag-neo4j/?utm_source=chatgpt.com)

---

## Future Enhancements

1. Migration from Container Volume to S3 for production
2. Advanced ontology versioning system
3. Enhanced entity resolution algorithms
4. Additional search tool integrations
5. Performance optimization and caching strategies