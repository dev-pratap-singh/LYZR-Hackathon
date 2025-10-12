"""
Multi-Tool Search Agent with Chain of Thought Reasoning
Using LangChain Agent Framework with 3 Tools: Vector, Graph, Filter Search
"""
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_search import BM25SearchService
from app.services.reranker import RerankerService
from app.services.graph_search import graph_search_service
from app.services.graphrag_pipeline import GraphRAGPipeline
from app.services.neo4j_service import neo4j_service
from app.services.elasticsearch_service import elasticsearch_service

logger = logging.getLogger(__name__)


# ===== Enhanced Streaming Callback =====
class EnhancedStreamingCallback(AsyncCallbackHandler):
    """Enhanced callback handler for agent tool execution and reasoning"""

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.current_tool = None
        self.reasoning_steps = []
        self.start_time = datetime.now()

    async def emit_event(self, event_type: str, content: str, metadata: Dict = None):
        """Emit a streaming event"""
        await self.events.put({
            "type": event_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts"""
        await self.emit_event(
            "thinking",
            "🧠 Agent is analyzing your query...",
            {"prompts_count": len(prompts)}
        )

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when a tool starts execution"""
        tool_name = serialized.get("name", "Unknown")
        self.current_tool = tool_name

        tool_icons = {
            "vector_search": "📚",
            "graph_search": "🕸️",
            "filter_search": "🔍"
        }

        icon = tool_icons.get(tool_name, "🔧")

        await self.emit_event(
            "tool_start",
            f"{icon} Executing **{tool_name}**\n📝 Input: {input_str[:200]}{'...' if len(input_str) > 200 else ''}",
            {
                "tool_name": tool_name,
                "input": input_str,
                "start_time": datetime.now().isoformat()
            }
        )

    async def on_tool_end(self, output: str, **kwargs):
        """Called when a tool finishes execution"""
        await self.emit_event(
            "tool_end",
            f"✅ **{self.current_tool}** completed\n📊 Retrieved {len(output)} chars of context",
            {
                "tool_name": self.current_tool,
                "output_preview": output,  # Send complete output
                "output_length": len(output)
            }
        )

    async def on_tool_error(self, error: Exception, **kwargs):
        """Called when a tool encounters an error"""
        await self.emit_event(
            "error",
            f"❌ Tool error in **{self.current_tool}**: {str(error)}",
            {"error": str(error), "tool": self.current_tool}
        )

    async def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when agent takes an action (chooses a tool)"""
        self.reasoning_steps.append(action.log)
        await self.emit_event(
            "reasoning",
            f"🤔 **Agent's Reasoning:**\n{action.log}",
            {
                "tool": action.tool,
                "tool_input": str(action.tool_input)[:200]
            }
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when agent finishes"""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        await self.emit_event(
            "final_answer",
            finish.return_values.get("output", ""),
            {
                "reasoning_steps": len(self.reasoning_steps),
                "elapsed_time": elapsed,
                "return_values": finish.return_values
            }
        )


# ===== Multi-Tool Search Agent =====
class SearchAgent:
    """Advanced Search Agent with 3 tools: Vector Search, Graph Search, Filter Search"""

    def __init__(self):
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            openai_api_key=settings.openai_api_key,
            streaming=True
        )

        # Initialize search services
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.bm25_search = BM25SearchService()
        self.reranker = RerankerService()
        self.graph_search = graph_search_service

        # Store current document_id for tools to access
        self.current_document_id = None

        # Agent components
        self.agent_executor = None
        self.tools = []

        # GraphRAG pipeline for processing documents
        self.graphrag_pipeline = GraphRAGPipeline()
        self.graph_processing_started = False

        logger.info("Multi-Tool Search Agent initialized")

    async def ensure_graph_processed(self, db_session):
        """
        Ensure all documents with text_filepath have been processed for the knowledge graph.
        This is called once per agent lifecycle on first query.

        Args:
            db_session: SQLAlchemy database session
        """
        if self.graph_processing_started:
            return

        self.graph_processing_started = True

        try:
            # Import here to avoid circular dependency
            from app.models import Document

            # Find all documents that have been processed but not graph-processed
            documents = db_session.query(Document).filter(
                Document.is_processed == True,
                Document.text_filepath != None,
                Document.graph_processed == False
            ).all()

            if not documents:
                logger.info("No documents need graph processing")
                return

            logger.info(f"Found {len(documents)} documents to process for knowledge graph")

            for document in documents:
                try:
                    logger.info(f"Processing knowledge graph for document {document.id}")

                    # Read extracted text
                    with open(document.text_filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()

                    # Process with GraphRAG using actual document ID
                    graph_result = await self.graphrag_pipeline.process_document(
                        text_content=text_content,
                        document_id=str(document.id)
                    )

                    # Import graph documents into Neo4j with actual document ID
                    await neo4j_service.create_constraints()
                    await neo4j_service.import_graph_documents(
                        graph_documents=graph_result["graph_documents"],
                        document_id=str(document.id)
                    )

                    # Update document record
                    document.graph_processed = True
                    document.graph_entities_count = graph_result["entities_count"]
                    document.graph_relationships_count = graph_result["relationships_count"]
                    db_session.commit()

                    logger.info(
                        f"Graph processed for {document.id}: "
                        f"{graph_result['entities_count']} entities, "
                        f"{graph_result['relationships_count']} relationships"
                    )

                except Exception as e:
                    logger.error(f"Error processing graph for document {document.id}: {e}")
                    db_session.rollback()
                    continue

            logger.info("Graph processing complete for all documents")

        except Exception as e:
            logger.error(f"Error in ensure_graph_processed: {e}")

    async def vector_search_sync(self, query: str) -> str:
        """
        Execute vector search with hybrid retrieval (Vector + BM25) and reranking
        This is called by the LangChain tool - must be synchronous but we handle async internally
        """
        try:
            logger.info(f"Vector search tool called with query: {query[:100]}")

            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Perform hybrid search (Vector + BM25)
            vector_results = await self.vector_store.vector_search(
                query_embedding=query_embedding,
                document_id=self.current_document_id,
                top_k=settings.top_k_results * 2
            )

            bm25_results = await self.bm25_search.search(
                query=query,
                document_id=self.current_document_id,
                top_k=settings.top_k_results * 2
            )

            if not vector_results and not bm25_results:
                return "❌ No relevant documents found. Please ensure documents are uploaded and processed."

            # Merge and deduplicate
            seen_ids = set()
            combined_results = []

            for result in vector_results + bm25_results:
                chunk_id = result['id']
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_results.append(result)

            # Rerank results
            search_results = self.reranker.rerank(
                query=query,
                results=combined_results,
                top_k=settings.top_k_results
            )

            # Format results for LLM context
            context_parts = []
            for idx, result in enumerate(search_results, 1):
                rerank_score = result.get('rerank_score', 0)
                passage = f"[Passage {idx}] (Relevance Score: {rerank_score:.3f})\n{result['chunk_text']}"
                context_parts.append(passage)

            result_text = "\n\n".join(context_parts)
            logger.info(f"Vector search completed: {len(search_results)} results, {len(result_text)} chars")
            return result_text

        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return f"❌ Vector search error: {str(e)}"

    async def graph_search_sync(self, query: str) -> str:
        """
        Execute graph search for relationship queries
        This is called by the LangChain tool - must be synchronous but we handle async internally
        """
        try:
            logger.info(f"Graph search tool called with query: {query[:100]}")

            graph_results = await self.graph_search.search_relationships(query, self.current_document_id)

            if not graph_results or "No entities found" in graph_results:
                return "❌ No graph data available. Please ensure the document has been processed with GraphRAG."

            logger.info(f"Graph search completed: {len(graph_results)} chars")
            return graph_results

        except Exception as e:
            logger.error(f"Graph search error: {e}", exc_info=True)
            return f"❌ Graph search error: {str(e)}"

    async def filter_search_sync(self, query: str) -> str:
        """
        Execute metadata-based filter search using Elasticsearch

        This method extracts filter criteria from natural language queries and
        searches documents based on metadata (dates, authors, categories, tags, etc.)
        """
        try:
            logger.info(f"Filter search tool called with query: {query[:100]}")

            # Use LLM to extract filter criteria from natural language
            filter_extraction_prompt = f"""Extract filter criteria from this query. Return a JSON object with these fields (use null if not mentioned):
- author: string (author name)
- document_type: string (pdf, txt, etc.)
- categories: array of strings
- tags: array of strings
- date_from: ISO date string (YYYY-MM-DD)
- date_to: ISO date string (YYYY-MM-DD)
- search_text: string (any text to search in content)

Query: {query}

Return ONLY valid JSON, no explanation."""

            extraction_response = await self.llm.ainvoke(filter_extraction_prompt)

            # Parse the extracted filters
            import json
            try:
                filters = json.loads(extraction_response.content.strip())
            except json.JSONDecodeError:
                # Fallback: use query as text search
                filters = {"search_text": query}

            # Extract search text and filters
            search_text = filters.pop("search_text", "") or query

            # Clean filters (remove null values)
            clean_filters = {k: v for k, v in filters.items() if v is not None}

            # Perform Elasticsearch search
            results = await elasticsearch_service.search(
                query=search_text,
                filters=clean_filters,
                size=10
            )

            if results["total"] == 0:
                return f"❌ No documents found matching the filters.\n\nQuery: {query}\nFilters extracted: {clean_filters}"

            # Format results for LLM context
            context_parts = [
                f"🔍 **Filter Search Results** (Found {results['total']} documents)\n",
                f"**Query**: {query}\n",
                f"**Filters Applied**: {json.dumps(clean_filters, indent=2)}\n\n"
            ]

            for idx, doc in enumerate(results["documents"], 1):
                doc_info = [
                    f"**[Document {idx}]**",
                    f"- **Filename**: {doc['filename']}",
                    f"- **Score**: {doc['score']:.3f}"
                ]

                if doc.get("author"):
                    doc_info.append(f"- **Author**: {doc['author']}")
                if doc.get("document_type"):
                    doc_info.append(f"- **Type**: {doc['document_type']}")
                if doc.get("categories"):
                    doc_info.append(f"- **Categories**: {', '.join(doc['categories'])}")
                if doc.get("tags"):
                    doc_info.append(f"- **Tags**: {', '.join(doc['tags'])}")
                if doc.get("uploaded_at"):
                    doc_info.append(f"- **Uploaded**: {doc['uploaded_at']}")

                # Add highlights if available
                if doc.get("highlights"):
                    doc_info.append(f"- **Highlights**: {' ... '.join(doc['highlights'])}")

                # Add content preview
                doc_info.append(f"- **Content Preview**: {doc['content_preview']}...")

                context_parts.append("\n".join(doc_info))

            result_text = "\n\n".join(context_parts)
            logger.info(f"Filter search completed: {results['total']} results")
            return result_text

        except Exception as e:
            logger.error(f"Filter search error: {e}", exc_info=True)
            return f"❌ Filter search error: {str(e)}"

    def _setup_tools(self):
        """Setup the 3 retrieval tools with comprehensive descriptions"""

        self.tools = [
            StructuredTool.from_function(
                coroutine=self.vector_search_sync,
                name="vector_search",
                description="""Semantic search using vector embeddings with hybrid retrieval (Vector + BM25) and cross-encoder reranking.

**WHEN TO USE (Selection Guidelines):**
✓ "What is...", "Explain...", "Define...", "Describe..." queries
✓ "Summarize...", "Tell me about..." requests
✓ General factual questions and information lookup
✓ Finding similar or relevant content
✓ Broad topic exploration without specific relationships
✓ Content-based retrieval
✓ When you need document passages containing information

**DO NOT USE for:**
✗ Relationship queries ("how are X and Y related")
✗ Date/metadata filtering ("documents from 2023")
✗ Connection analysis ("what connects X to Y")

**Input:** Natural language query string
**Returns:** Top relevant document passages with relevance scores

**Example queries:**
- "What is quantum computing?"
- "Explain machine learning algorithms"
- "Define blockchain technology"
- "Summarize the key concepts in this document"
- "Tell me about neural networks"
"""
            ),
            StructuredTool.from_function(
                coroutine=self.graph_search_sync,
                name="graph_search",
                description="""Graph-based search for relationships and connections using Neo4j knowledge graph.

**WHEN TO USE (Selection Guidelines):**
✓ "How are X and Y related?", "What's the relationship between..."
✓ "What connects X and Y?", "Connection between..."
✓ "Path from X to Y", "How does X lead to Y?"
✓ Multi-hop reasoning requiring entity relationships
✓ Network analysis and entity traversal
✓ Complex reasoning across multiple relationship steps
✓ "How are these concepts linked?", "What influences what?"
✓ Finding indirect connections between entities

**DO NOT USE for:**
✗ Simple definitions ("what is X")
✗ Metadata filtering ("documents from 2023")
✗ Broad searches without relationship focus
✗ Content retrieval without entity relationships

**Input:** Query about relationships or connections between entities
**Returns:** Graph traversal results showing entity connections and relationship paths

**Example queries:**
- "How are quantum computing and cryptography related?"
- "What connects machine learning to neural networks?"
- "What's the relationship between blockchain and AI?"
- "How does concept A influence concept B?"
- "What are the connections between these entities?"
"""
            ),
            StructuredTool.from_function(
                coroutine=self.filter_search_sync,
                name="filter_search",
                description="""Metadata-based filtering for precise attribute queries using Elasticsearch.

**WHEN TO USE (Selection Guidelines):**
✓ "Documents from [year/date]", "Files dated...", "Published in..."
✓ "Category = [category]", "Type = [type]", "Tagged as..."
✓ "Author = [name]", "Source = [source]", "Written by..."
✓ "Show me only...", "Filter by...", "Where [attribute]..."
✓ Queries with specific metadata constraints
✓ Attribute-based filtering (date, category, author, tags, properties)
✓ Precise criteria matching
✓ Full-text search within metadata-filtered documents

**DO NOT USE for:**
✗ Pure semantic searches without filters (use vector_search)
✗ Relationship queries ("how are X and Y related" - use graph_search)

**Input:** Query with filter criteria or metadata constraints
**Returns:** Documents matching the specified filters with relevance scores and highlights

**Example queries:**
- "Show me documents about AI from 2023"
- "Papers in the 'quantum computing' category"
- "Articles by author John Smith"
- "Documents tagged as 'research' published after 2022"
- "Find PDFs uploaded last month"
- "Search for documents in category 'machine learning'"
"""
            )
        ]

        logger.info(f"Setup {len(self.tools)} tools: {', '.join([t.name for t in self.tools])}")

    def _setup_agent(self):
        """Setup LangChain agent with comprehensive system prompt"""

        system_prompt = """You are an AI assistant that helps users find information from their UPLOADED DOCUMENTS.

**CRITICAL RULES:**

1. The user has uploaded documents. You MUST search these documents using your tools.
2. For EVERY question, you MUST call one of your search tools BEFORE answering.
3. NEVER answer from your own knowledge - ALWAYS search the uploaded documents first.
4. DO NOT just describe what you will do - ACTUALLY CALL THE TOOL.

**Tool Selection:**

Use **vector_search** for:
- Any "what/who/where/when/why" question about the document
- Definitions, explanations, summaries
- Finding information, facts, or content
- DEFAULT choice when unsure

Use **graph_search** for:
- "How are X and Y related?"
- "What's the relationship between..."
- Connection or relationship questions

Use **filter_search** for:
- Date/metadata filtering ("documents from 2023")
- Category/tag filtering ("papers in category X")
- Author filtering ("documents by author Y")

**IMPORTANT:**
- You have access to {tool_names}
- You MUST use these tools - they are not optional
- Call the appropriate tool IMMEDIATELY, don't just talk about calling it
- After getting tool results, synthesize a clear answer citing the passages

**Process:**
1. Identify the question type
2. Select the right tool (default: vector_search)
3. CALL THE TOOL (don't just describe it)
4. Answer based on the tool's results"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Add tool names to prompt
        tool_names = ", ".join([t.name for t in self.tools])
        prompt = prompt.partial(tool_names=tool_names)

        # Create agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        logger.info("Agent executor initialized with max_iterations=5")

    async def process_query_with_streaming(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process query using the multi-tool agent with streaming

        Args:
            query: User's question
            document_id: Optional document ID to search within

        Yields:
            SSE-formatted event strings
        """
        try:
            # Store document_id so tools can access it
            self.current_document_id = document_id
            logger.info(f"Processing query with document_id: {document_id}")

            # Setup tools and agent if not already done
            if not self.tools:
                self._setup_tools()
                self._setup_agent()

            # Create callback handler
            callback = EnhancedStreamingCallback()

            # Initial event
            yield f"data: {json.dumps({'type': 'thinking', 'content': '🔍 Analyzing your query and selecting appropriate tools...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Start agent execution as background task
            task = asyncio.create_task(
                self.agent_executor.ainvoke(
                    {"input": query},
                    {"callbacks": [callback]}
                )
            )

            # Stream events from callback
            while True:
                try:
                    # Wait for next event with timeout
                    event = await asyncio.wait_for(
                        callback.events.get(),
                        timeout=0.5
                    )

                    # Stream event to client
                    yield f"data: {json.dumps(event)}\n\n"

                    # If this is the final answer, break
                    if event["type"] == "final_answer":
                        break

                except asyncio.TimeoutError:
                    # Check if task is done
                    if task.done():
                        break
                    # Send keepalive
                    yield ": keepalive\n\n"

            # Get final result
            result = await task

            # Send metadata event
            metadata_event = {
                "type": "metadata",
                "content": "Query processing complete",
                "metadata": {
                    "has_intermediate_steps": "intermediate_steps" in result,
                    "tools_used": [
                        step[0].tool for step in result.get("intermediate_steps", [])
                    ] if "intermediate_steps" in result else []
                },
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"

        except Exception as e:
            logger.error(f"Error in query processing: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "content": f"❌ Error processing query: {str(e)}",
                "metadata": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    async def simple_query(self, query: str, document_id: Optional[str] = None) -> Dict:
        """
        Non-streaming query processing (for testing)

        Args:
            query: User question
            document_id: Optional document filter

        Returns:
            Response dictionary
        """
        try:
            # Store document_id so tools can access it
            self.current_document_id = document_id

            # Setup tools and agent if not already done
            if not self.tools:
                self._setup_tools()
                self._setup_agent()

            result = await self.agent_executor.ainvoke(
                {"input": query},
                {"callbacks": []}
            )

            return {
                "answer": result.get("output", "No answer generated"),
                "intermediate_steps": result.get("intermediate_steps", []),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error in simple query: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "query": query
            }
