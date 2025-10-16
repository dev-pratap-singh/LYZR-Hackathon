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
from app.services.memory import MemoryManager

logger = logging.getLogger(__name__)


# ===== Formatting Helper =====
def format_markdown_output(text: str) -> str:
    """
    Post-process markdown output to ensure proper spacing and formatting

    Args:
        text: Raw markdown text from LLM

    Returns:
        Properly formatted markdown with consistent spacing
    """
    import re

    # Ensure double newlines after headings (## or ###)
    text = re.sub(r'(^|\n)(#{1,3}\s+[^\n]+)\n(?!\n)', r'\1\2\n\n', text, flags=re.MULTILINE)

    # Ensure blank lines before and after horizontal rules
    text = re.sub(r'(?<!\n)\n---\n(?!\n)', r'\n\n---\n\n', text)
    text = re.sub(r'\n---\n(?!\n)', r'\n---\n\n', text)
    text = re.sub(r'(?<!\n)\n---\n', r'\n\n---\n', text)

    # Ensure blank line before lists (- at start of line)
    text = re.sub(r'([^\n])\n(-\s)', r'\1\n\n\2', text)

    # Ensure blank line after lists (list followed by non-list content)
    text = re.sub(r'(-\s[^\n]+)\n([^-\n\s])', r'\1\n\n\2', text)

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Ensure spacing around bold subsection headers within content
    text = re.sub(r'([^\n])\n(\*\*[^*]+\*\*:?)\n([^\n])', r'\1\n\n\2\n\n\3', text)

    return text.strip()


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
            "filter_search": "🔍",
            "graph_update": "✏️"
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

        # If this was a graph update, emit a special event to notify frontend to refresh graph
        if self.current_tool == "graph_update" and "✅" in output:
            await self.emit_event(
                "graph_updated",
                "🔄 Knowledge graph has been updated. Refresh the graph view to see changes.",
                {
                    "tool_name": "graph_update",
                    "update_successful": True,
                    "output_preview": output[:500]
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

        # Format the output for better readability
        formatted_output = format_markdown_output(finish.return_values.get("output", ""))

        await self.emit_event(
            "final_answer",
            formatted_output,
            {
                "reasoning_steps": len(self.reasoning_steps),
                "elapsed_time": elapsed,
                "return_values": finish.return_values
            }
        )


# ===== Multi-Tool Search Agent =====
class SearchAgent:
    """Advanced Search Agent with 3 tools: Vector Search, Graph Search, Filter Search"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize SearchAgent with optional user-provided OpenAI API key.

        Args:
            openai_api_key: Optional OpenAI API key. If not provided, uses settings.openai_api_key
        """
        # Use provided key or fallback to settings
        self.api_key = openai_api_key or settings.openai_api_key
        self._llm = None  # Lazy initialization

        # Initialize search services with the same API key
        self.embedding_service = EmbeddingService(openai_api_key=self.api_key)
        self.vector_store = VectorStoreService()
        self.bm25_search = BM25SearchService()
        self.reranker = RerankerService()
        self.graph_search = graph_search_service

        # Store current document_id for tools to access
        self.current_document_id = None

        # Agent components
        self.agent_executor = None
        self.tools = []

        # GraphRAG pipeline for processing documents with user-provided API key
        self.graphrag_pipeline = GraphRAGPipeline(openai_api_key=self.api_key)
        self.graph_processing_started = False

        # Initialize Memory Manager
        self.memory_manager = None
        if settings.memory_enabled:
            try:
                self.memory_manager = MemoryManager(
                    db_host=settings.memory_db_host,
                    db_port=settings.memory_db_port,
                    db_name=settings.memory_db_name,
                    db_user=settings.memory_db_user,
                    db_password=settings.memory_db_password,
                    model_name=settings.memory_model,
                    openai_api_key=self.api_key,
                    session_id=settings.memory_session_id
                )
                logger.info("Memory Manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Memory Manager: {e}")
                self.memory_manager = None

        logger.info("Multi-Tool Search Agent initialized")

    @property
    def llm(self):
        """Lazy initialization of LLM client"""
        if self._llm is None:
            if not self.api_key:
                error_msg = (
                    "⚠️  OPENAI_API_KEY is not configured. "
                    "AI-powered search features require an OpenAI API key. "
                    "Please set the OPENAI_API_KEY environment variable or provide an API key when creating the SearchAgent."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Initializing OpenAI LLM client for search agent")
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,  # Zero temperature for more deterministic, factual responses
                openai_api_key=self.api_key,
                streaming=True,
                timeout=300,  # 5 minute timeout for OpenAI API calls
                request_timeout=300  # 5 minute request timeout
            )
        return self._llm

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

                    # Update document record - refresh to avoid stale data
                    try:
                        db_session.refresh(document)
                        document.graph_processed = True
                        document.graph_entities_count = graph_result["entities_count"]
                        document.graph_relationships_count = graph_result["relationships_count"]
                        db_session.commit()
                    except Exception as commit_error:
                        logger.warning(f"Could not update document {document.id}: {commit_error}")
                        db_session.rollback()
                        # Document was already updated by another process, which is fine

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
            try:
                db_session.rollback()
            except:
                pass  # Session might already be closed

    async def _expand_context_with_adjacent_chunks(self, search_results: List[Dict]) -> List[Dict]:
        """
        Expand retrieved chunks by including 2 adjacent chunks on each side for better context coverage
        This improves Context Recall by ensuring information spanning multiple chunks is captured

        Args:
            search_results: List of retrieved chunks

        Returns:
            Expanded list of chunks with adjacent context included
        """
        try:
            expanded_results = []
            seen_indices = set()

            # Group chunks by document to fetch adjacent chunks efficiently
            chunks_by_doc = {}
            for result in search_results:
                doc_id = result.get('document_id')
                chunk_idx = result.get('chunk_index')
                if doc_id and chunk_idx is not None:
                    if doc_id not in chunks_by_doc:
                        chunks_by_doc[doc_id] = []
                    chunks_by_doc[doc_id].append((chunk_idx, result))

            # For each document, fetch adjacent chunks (2 on each side)
            for doc_id, chunks in chunks_by_doc.items():
                # Get all chunk indices for this document
                all_document_chunks = await self.vector_store.get_document_chunks(doc_id)
                max_index = len(all_document_chunks) - 1

                for chunk_idx, original_result in chunks:
                    # Add 2 previous chunks if they exist
                    if chunk_idx > 1 and (chunk_idx - 2) not in seen_indices:
                        prev_chunk_2 = all_document_chunks[chunk_idx - 2]
                        expanded_results.append({
                            **prev_chunk_2,
                            'is_expanded': True,
                            'expansion_type': 'previous-2',
                            'rerank_score': original_result.get('rerank_score', 0) * 0.7  # Further reduced score
                        })
                        seen_indices.add(chunk_idx - 2)

                    if chunk_idx > 0 and (chunk_idx - 1) not in seen_indices:
                        prev_chunk = all_document_chunks[chunk_idx - 1]
                        expanded_results.append({
                            **prev_chunk,
                            'is_expanded': True,
                            'expansion_type': 'previous-1',
                            'rerank_score': original_result.get('rerank_score', 0) * 0.8  # Reduced score
                        })
                        seen_indices.add(chunk_idx - 1)

                    # Add the original chunk
                    if chunk_idx not in seen_indices:
                        expanded_results.append(original_result)
                        seen_indices.add(chunk_idx)

                    # Add 2 next chunks if they exist
                    if chunk_idx < max_index and (chunk_idx + 1) not in seen_indices:
                        next_chunk = all_document_chunks[chunk_idx + 1]
                        expanded_results.append({
                            **next_chunk,
                            'is_expanded': True,
                            'expansion_type': 'next+1',
                            'rerank_score': original_result.get('rerank_score', 0) * 0.8  # Reduced score
                        })
                        seen_indices.add(chunk_idx + 1)

                    if chunk_idx < max_index - 1 and (chunk_idx + 2) not in seen_indices:
                        next_chunk_2 = all_document_chunks[chunk_idx + 2]
                        expanded_results.append({
                            **next_chunk_2,
                            'is_expanded': True,
                            'expansion_type': 'next+2',
                            'rerank_score': original_result.get('rerank_score', 0) * 0.7  # Further reduced score
                        })
                        seen_indices.add(chunk_idx + 2)

            # Sort by chunk index to maintain document flow
            expanded_results.sort(key=lambda x: (x.get('document_id', ''), x.get('chunk_index', 0)))

            logger.info(f"Context expansion: {len(search_results)} → {len(expanded_results)} chunks (2 adjacent on each side)")
            return expanded_results

        except Exception as e:
            logger.error(f"Error in context expansion: {e}", exc_info=True)
            # Return original results if expansion fails
            return search_results

    async def vector_search_sync(self, query: str) -> str:
        """
        Execute vector search with hybrid retrieval (Vector + BM25) and reranking
        This is called by the LangChain tool - must be synchronous but we handle async internally
        """
        try:
            logger.info(f"Vector search tool called with query: {query[:100]}")

            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Perform hybrid search (Vector + BM25) with increased retrieval depth
            # Retrieve 5x top_k to ensure comprehensive coverage for better Context Recall
            vector_results = await self.vector_store.vector_search(
                query_embedding=query_embedding,
                document_id=self.current_document_id,
                top_k=settings.top_k_results * 5  # Increased from 4x to 5x for better recall
            )

            bm25_results = await self.bm25_search.search(
                query=query,
                document_id=self.current_document_id,
                top_k=settings.top_k_results * 5  # Increased from 4x to 5x for better recall
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

            # Rerank results - return 3x top_k for better Context Recall
            # This ensures we include more relevant chunks that cover ground truth information
            search_results = self.reranker.rerank(
                query=query,
                results=combined_results,
                top_k=settings.top_k_results * 3  # Increased from 2x to 3x for maximum recall
            )

            # Expand context with adjacent chunks for better coverage
            expanded_results = await self._expand_context_with_adjacent_chunks(search_results)

            # Format results for LLM context
            context_parts = []
            for idx, result in enumerate(expanded_results, 1):
                rerank_score = result.get('rerank_score', 0)
                expansion_marker = ""
                if result.get('is_expanded'):
                    expansion_marker = f" [Context: {result.get('expansion_type')}]"
                passage = f"[Passage {idx}] (Relevance Score: {rerank_score:.3f}){expansion_marker}\n{result['chunk_text']}"
                context_parts.append(passage)

            result_text = "\n\n".join(context_parts)
            logger.info(f"Vector search completed: {len(expanded_results)} results (expanded), {len(result_text)} chars")
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

    async def graph_update_sync(self, query: str) -> str:
        """
        Execute graph update operations using natural language

        This method parses natural language update commands and performs
        operations on the knowledge graph (delete nodes, merge nodes, update properties, etc.)
        """
        try:
            logger.info(f"Graph update tool called with query: {query[:100]}")

            # Use LLM to extract update operation from natural language
            update_extraction_prompt = f"""Parse this graph update query and return a JSON object with the operation details.

Supported operations:
1. create_node: {{"operation": "create_node", "node_id": "entity name", "node_type": "Entity", "description": "description"}}
2. create_node_with_relationships: {{"operation": "create_node_with_relationships", "node_id": "new entity", "node_type": "Entity", "description": "description", "relationships": [{{"target_id": "existing entity", "relationship_type": "TYPE", "direction": "outgoing"}}]}}
3. delete_node: {{"operation": "delete_node", "node_id": "entity name"}}
4. merge_nodes: {{"operation": "merge_nodes", "node_id1": "first entity", "node_id2": "second entity", "new_node_id": "optional new name"}}
5. create_relationship: {{"operation": "create_relationship", "source_id": "source entity", "target_id": "target entity", "relationship_type": "TYPE", "properties": {{"key": "value"}}}}
6. update_node_property: {{"operation": "update_node_property", "node_id": "entity name", "property_name": "property", "property_value": "new value"}}
7. update_node_description: {{"operation": "update_node_description", "node_id": "entity name", "description": "new description"}}
8. update_relationship: {{"operation": "update_relationship", "source_id": "source", "target_id": "target", "relationship_type": "TYPE", "properties": {{"key": "value"}}}}
9. delete_relationship: {{"operation": "delete_relationship", "source_id": "source", "target_id": "target", "relationship_type": "TYPE"}}

IMPORTANT:
- If the query wants to create a NEW node and connect it to MULTIPLE existing nodes, use "create_node_with_relationships" operation
- The "relationships" array should contain ALL the connections to make
- Each relationship needs: target_id, relationship_type, and direction ("outgoing" or "incoming")
- direction="outgoing" means: new_node -> target
- direction="incoming" means: target -> new_node

Example for creating "Software Engineering" and connecting to multiple nodes:
{{
  "operation": "create_node_with_relationships",
  "node_id": "Software Engineering",
  "node_type": "Concept",
  "description": "Software Engineering discipline",
  "relationships": [
    {{"target_id": "Problem-Solving", "relationship_type": "REQUIRES", "direction": "outgoing"}},
    {{"target_id": "Python", "relationship_type": "USES", "direction": "outgoing"}},
    {{"target_id": "Clickhouse", "relationship_type": "USES", "direction": "outgoing"}}
  ]
}}

Query: {query}

Return ONLY valid JSON with the operation details, no explanation."""

            extraction_response = await self.llm.ainvoke(update_extraction_prompt)

            # Parse the extracted operation
            import json
            try:
                operation = json.loads(extraction_response.content.strip())
            except json.JSONDecodeError:
                return f"❌ Could not parse update operation from query: {query}"

            operation_type = operation.get("operation")
            if not operation_type:
                return f"❌ No valid operation found in query: {query}"

            # Execute the operation using neo4j_service
            result = None

            if operation_type == "create_node":
                result = await neo4j_service.create_node(
                    node_id=operation.get("node_id"),
                    node_type=operation.get("node_type", "Entity"),
                    description=operation.get("description", ""),
                    properties=operation.get("properties", {}),
                    document_id=self.current_document_id
                )

            elif operation_type == "create_node_with_relationships":
                result = await neo4j_service.create_node_with_relationships(
                    node_id=operation.get("node_id"),
                    node_type=operation.get("node_type", "Entity"),
                    description=operation.get("description", ""),
                    relationships=operation.get("relationships", []),
                    document_id=self.current_document_id
                )

            elif operation_type == "delete_node":
                result = await neo4j_service.delete_node(
                    node_id=operation.get("node_id"),
                    document_id=self.current_document_id
                )

            elif operation_type == "merge_nodes":
                result = await neo4j_service.merge_nodes(
                    node_id1=operation.get("node_id1"),
                    node_id2=operation.get("node_id2"),
                    new_node_id=operation.get("new_node_id"),
                    document_id=self.current_document_id
                )

            elif operation_type == "create_relationship":
                result = await neo4j_service.create_relationship(
                    source_id=operation.get("source_id"),
                    target_id=operation.get("target_id"),
                    relationship_type=operation.get("relationship_type"),
                    properties=operation.get("properties", {}),
                    document_id=self.current_document_id
                )

            elif operation_type == "update_node_property":
                result = await neo4j_service.update_node_property(
                    node_id=operation.get("node_id"),
                    property_name=operation.get("property_name"),
                    property_value=operation.get("property_value"),
                    document_id=self.current_document_id
                )

            elif operation_type == "update_node_description":
                result = await neo4j_service.update_node_description(
                    node_id=operation.get("node_id"),
                    description=operation.get("description"),
                    document_id=self.current_document_id
                )

            elif operation_type == "update_relationship":
                result = await neo4j_service.update_relationship(
                    source_id=operation.get("source_id"),
                    target_id=operation.get("target_id"),
                    relationship_type=operation.get("relationship_type"),
                    properties=operation.get("properties", {}),
                    document_id=self.current_document_id
                )

            elif operation_type == "delete_relationship":
                result = await neo4j_service.delete_relationship(
                    source_id=operation.get("source_id"),
                    target_id=operation.get("target_id"),
                    relationship_type=operation.get("relationship_type"),
                    document_id=self.current_document_id
                )

            else:
                return f"❌ Unsupported operation: {operation_type}"

            # Format the result
            if result and result.get("success"):
                formatted_result = f"✅ **Graph Update Successful**\n\n"
                formatted_result += f"**Operation**: {operation_type}\n"
                formatted_result += f"**Details**:\n"
                formatted_result += json.dumps(result, indent=2)
                logger.info(f"Graph update completed successfully: {operation_type}")
                return formatted_result
            else:
                error_msg = result.get("error", "Unknown error") if result else "Operation failed"
                formatted_result = f"❌ **Graph Update Failed**\n\n"
                formatted_result += f"**Operation**: {operation_type}\n"
                formatted_result += f"**Error**: {error_msg}\n"
                logger.error(f"Graph update failed: {error_msg}")
                return formatted_result

        except Exception as e:
            logger.error(f"Graph update error: {e}", exc_info=True)
            return f"❌ Graph update error: {str(e)}"

    def _setup_tools(self):
        """Setup the 4 tools: vector search, graph search, filter search, and graph update"""

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
✗ Graph updates (use graph_update)

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
                description="""Graph-based search for entities, relationships and connections using Neo4j knowledge graph with 1-hop and 2-hop traversal.

**WHEN TO USE (Selection Guidelines):**
✓ "Tell me about X", "Who is X", "What is X" - for entity-focused queries
✓ "How are X and Y related?", "What's the relationship between..."
✓ "What connects X and Y?", "Connection between..."
✓ "Path from X to Y", "How does X lead to Y?"
✓ Multi-hop reasoning requiring entity relationships
✓ Network analysis and entity traversal
✓ Complex reasoning across multiple relationship steps
✓ "How are these concepts linked?", "What influences what?"
✓ Finding indirect connections between entities
✓ When user explicitly asks to "use graph search"
✓ Exploring entity networks with direct and extended connections

**CAPABILITIES:**
- Finds entities and their complete network context
- 1-hop traversal: Direct connections and relationships
- 2-hop traversal: Extended network through intermediaries
- Relationship type analysis with descriptions
- Graph statistics and comprehensive entity profiles

**DO NOT USE for:**
✗ Metadata filtering ("documents from 2023") - use filter_search
✗ Pure content retrieval without entity focus - use vector_search
✗ Graph updates (use graph_update)

**Input:** Query about entities, relationships, or connections
**Returns:** Comprehensive graph analysis with:
- Entity information (ID, type, description)
- Direct connections (1-hop) with relationship types
- Extended network (2-hop) showing indirect connections
- Graph statistics and relationship analysis

**Example queries:**
- "Tell me about Dev Pratap Singh" (entity profile with network)
- "Who is John Smith" (entity information with connections)
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
✗ Graph updates (use graph_update)

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
            ),
            StructuredTool.from_function(
                coroutine=self.graph_update_sync,
                name="graph_update",
                description="""Update, modify, create, or delete nodes and relationships in the knowledge graph using natural language commands.

**WHEN TO USE (Selection Guidelines):**
✓ "Create a new node X", "Add entity Y"
✓ "Create X and connect it to Y, Z, A", "Add node X and link to multiple nodes"
✓ "Delete node X", "Remove entity Y"
✓ "Merge nodes X and Y", "Combine entities A and B"
✓ "Connect X to Y", "Create relationship between A and B"
✓ "Update node X property/description", "Change entity Y's [property] to [value]"
✓ "Update the relationship between X and Y"
✓ "Delete the relationship/edge between A and B"
✓ ANY query asking to create, modify, update, change, delete, merge, or connect graph elements

**SUPPORTED OPERATIONS:**
1. Create Node: Add a new entity to the graph
2. Create Node with Relationships: Add a new entity and connect it to multiple existing nodes (EFFICIENT FOR BULK CONNECTIONS)
3. Delete Node: Remove an entity from the graph
4. Merge Nodes: Combine two entities into one, preserving relationships
5. Create Relationship: Add a new connection between two entities
6. Update Node Property: Change a specific property value
7. Update Node Description: Modify the description of an entity
8. Update Relationship: Modify relationship properties
9. Delete Relationship: Remove a connection between entities

**DO NOT USE for:**
✗ Reading/searching graph data (use graph_search)
✗ Content search (use vector_search)
✗ Metadata filtering (use filter_search)

**Input:** Natural language command describing the update operation
**Returns:** Success/failure status with details of the graph update

**Example queries:**
- "Create a new node called 'Software Engineering'"
- "Create 'Software Engineering' and connect it to Python, Java, and C++"
- "Add a new entity 'Machine Learning' and link it to AI, Python, and Data Science"
- "Delete the node named 'Machine Learning'"
- "Merge 'AI' and 'Artificial Intelligence' into one node"
- "Create a relationship 'WORKS_ON' from John to Project X"
- "Update the description of 'Python' to 'A popular programming language'"
- "Change the 'status' property of 'Project A' to 'completed'"
- "Delete the relationship between 'Dev' and 'Company Y'"
"""
            )
        ]

        logger.info(f"Setup {len(self.tools)} tools: {', '.join([t.name for t in self.tools])}")

    def _setup_agent(self):
        """Setup LangChain agent with comprehensive system prompt"""

        system_prompt = """You are an AI assistant that helps users find information from their UPLOADED DOCUMENTS and manage the knowledge graph.

**CRITICAL RULES:**

1. The user has uploaded documents. You MUST search these documents using your tools.
2. For EVERY question, you MUST call one of your tools BEFORE answering.
3. NEVER answer from your own knowledge - ALWAYS search the uploaded documents first.
4. DO NOT just describe what you will do - ACTUALLY CALL THE TOOL.

**Tool Selection:**

**CRITICAL: When user explicitly requests a specific tool, YOU MUST USE THAT TOOL!**
- If user says "use filter search", "filter search", "use graph search", "graph search", "use vector search", "vector search" → USE THAT SPECIFIC TOOL
- User's explicit tool preference ALWAYS overrides default selection logic

Use **graph_update** for:
- "Delete node/entity X", "Remove X from the graph"
- "Merge X and Y", "Combine entities A and B"
- "Connect X to Y", "Create relationship between A and B"
- "Update/Change X's description/property", "Modify entity Y"
- "Delete the relationship between X and Y"
- ANY query asking to modify, update, change, delete, merge, or connect graph elements

Use **graph_search** for:
- "Tell me about X", "Who is X", "What is X" - when X is a person or entity
- "How are X and Y related?"
- "What's the relationship between..."
- Connection or relationship questions
- ANY query where user says "use graph search" or "graph search only"
- Reading/exploring the knowledge graph

Use **vector_search** for:
- General "what/why/how" questions about concepts or topics
- Definitions, explanations, summaries of general concepts
- Finding information, facts, or content from documents
- ANY query where user says "use vector search" or "vector search only"
- DEFAULT choice when unsure AND user hasn't specified a tool

Use **filter_search** for:
- Date/metadata filtering ("documents from 2023")
- Category/tag filtering ("papers in category X")
- Author filtering ("documents by author Y")
- ANY query where user says "use filter search" or "filter search only"
- When user explicitly requests metadata-based filtering

**IMPORTANT:**
- You have access to {tool_names}
- You MUST use these tools - they are not optional
- Call the appropriate tool IMMEDIATELY, don't just talk about calling it
- After getting tool results, synthesize a clear answer citing the passages
- For graph updates, ALWAYS report the result to the user clearly

**ANSWER FORMATTING RULES:**

**FOR ALL QUERIES (including graph search):**

**CRITICAL FORMATTING REQUIREMENTS - MUST FOLLOW EXACTLY:**

1. **After Every Heading**: ALWAYS add TWO blank lines after ## or ### headings
   - Example: "## Heading\\n\\nContent starts here"
   - NOT: "## Heading Content starts here"

2. **Horizontal Rules**: ALWAYS surround --- with blank lines
   - Example: "\\n\\n---\\n\\n"
   - Add blank line BEFORE and AFTER every ---

3. **Before Every List**: Add blank line before starting bullets
   - Example: "Introduction text\\n\\n- First bullet"
   - NOT: "Introduction text\\n- First bullet"

4. **Between Sections**: Add TWO blank lines between major content blocks
   - Between paragraphs and lists
   - Between different subsections
   - Between list and next heading

5. **List Formatting**:
   - Use single dash (-) for bullets
   - Keep sub-bullets indented with 2 spaces
   - Add blank line after list ends before next content

6. **Section Structure**:
   - ## for main sections (followed by 2 blank lines)
   - ### for subsections (followed by 2 blank lines)
   - --- for major breaks (surrounded by blank lines)

**CONTENT GUIDELINES:**
1. **Synthesize and summarize** the retrieved information into a clear, readable response
2. Use natural, conversational language - avoid technical jargon when possible
3. Be comprehensive but concise - include key details without overwhelming the user
4. Focus on answering the user's actual question directly
5. When presenting entity information:
   - Start with a brief summary paragraph about the entity
   - Group related information logically (Education, Work Experience, Skills, etc.)
   - Highlight key facts and relationships
   - Use bullet points or short paragraphs, not raw technical output
6. Make it scannable - use formatting to help users find information quickly

"""

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

    async def _execute_all_tools_parallel(self, query: str) -> Dict[str, str]:
        """
        Execute all search tools in parallel for max performance mode

        Args:
            query: User's question

        Returns:
            Dictionary with results from all tools
        """
        try:
            logger.info(f"Max Performance Mode: Executing all search tools in parallel for query: {query[:100]}")

            # Execute all three search tools in parallel
            vector_task = asyncio.create_task(self.vector_search_sync(query))
            graph_task = asyncio.create_task(self.graph_search_sync(query))
            filter_task = asyncio.create_task(self.filter_search_sync(query))

            # Wait for all tasks to complete
            vector_result, graph_result, filter_result = await asyncio.gather(
                vector_task, graph_task, filter_task,
                return_exceptions=True
            )

            # Handle any exceptions
            results = {
                "vector_search": str(vector_result) if not isinstance(vector_result, Exception) else f"❌ Error: {str(vector_result)}",
                "graph_search": str(graph_result) if not isinstance(graph_result, Exception) else f"❌ Error: {str(graph_result)}",
                "filter_search": str(filter_result) if not isinstance(filter_result, Exception) else f"❌ Error: {str(filter_result)}"
            }

            logger.info(f"Parallel execution completed. Results: Vector={len(results['vector_search'])} chars, Graph={len(results['graph_search'])} chars, Filter={len(results['filter_search'])} chars")
            return results

        except Exception as e:
            logger.error(f"Error in parallel tool execution: {e}", exc_info=True)
            return {
                "vector_search": f"❌ Error: {str(e)}",
                "graph_search": f"❌ Error: {str(e)}",
                "filter_search": f"❌ Error: {str(e)}"
            }

    async def _synthesize_parallel_results(self, query: str, results: Dict[str, str]) -> str:
        """
        Synthesize results from all search tools into a comprehensive answer

        Args:
            query: Original user query
            results: Dictionary containing results from all search tools

        Returns:
            Synthesized answer combining all search results
        """
        try:
            logger.info("Synthesizing results from all search tools")

            # Build a comprehensive context from all tools
            synthesis_prompt = f"""You are an AI assistant synthesizing information from multiple search sources to answer a user's question.

**User's Question:** {query}

**SEARCH RESULTS FROM ALL TOOLS:**

---
**1. VECTOR SEARCH RESULTS (Semantic Content Search):**
{results['vector_search']}

---
**2. GRAPH SEARCH RESULTS (Entity & Relationship Network):**
{results['graph_search']}

---
**3. FILTER SEARCH RESULTS (Metadata-Based Search):**
{results['filter_search']}

---

**YOUR TASK:**
Synthesize a comprehensive, well-structured answer to the user's question using information from ALL THREE search sources above.

**SYNTHESIS GUIDELINES:**

**CRITICAL FORMATTING REQUIREMENTS - MUST FOLLOW EXACTLY:**

1. **After Every Heading**: ALWAYS add TWO blank lines after ## or ### headings
   - Example: "## Heading\\n\\nContent starts here"
   - NOT: "## Heading Content starts here"

2. **Horizontal Rules**: ALWAYS surround --- with blank lines
   - Example: "\\n\\n---\\n\\n"
   - Add blank line BEFORE and AFTER every ---

3. **Before Every List**: Add blank line before starting bullets
   - Example: "Introduction text\\n\\n- First bullet"

4. **Between Sections**: Add TWO blank lines between major content blocks

5. **Section Structure**:
   - ## for main sections (followed by 2 blank lines)
   - ### for subsections (followed by 2 blank lines)
   - --- for major breaks (surrounded by blank lines)

**CONTENT GUIDELINES:**
1. **Combine Insights**: Integrate relevant information from vector, graph, and filter search results
2. **Prioritize Quality**: If some search tools returned errors or no results, focus on the successful ones
3. **Be Comprehensive**: Include all relevant details from the search results
4. **Maintain Context**: Reference which search method provided specific information when relevant
5. **Natural Language**: Write in a conversational, easy-to-understand style
6. **Address the Question**: Directly answer what the user asked

**ANSWER FORMAT:**
- Start with a direct answer or summary
- Use sections/headers (##, ###) to organize different aspects
- Add proper spacing (blank lines) between sections
- Include supporting details and context
- If relevant, mention connections and relationships from graph search
- Cite specific passages or entities when appropriate

Generate your comprehensive answer now:"""

            # Use LLM to synthesize the results
            response = await self.llm.ainvoke(synthesis_prompt)
            synthesized_answer = response.content.strip()

            # Format for better readability
            formatted_answer = format_markdown_output(synthesized_answer)

            logger.info(f"Synthesis completed: {len(formatted_answer)} chars")
            return formatted_answer

        except Exception as e:
            logger.error(f"Error in result synthesis: {e}", exc_info=True)
            # Fallback: return concatenated results
            fallback = f"""## Search Results for: {query}

### Vector Search Results:
{results['vector_search']}

### Graph Search Results:
{results['graph_search']}

### Filter Search Results:
{results['filter_search']}

*Note: Automatic synthesis failed. Showing raw results from all search tools.*"""
            return fallback

    async def _check_memory_for_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if the query can be answered from conversation memory

        This implements the memory-first strategy:
        1. Search memory for relevant context
        2. Get conversation history
        3. Use LLM to decide if memory has the answer
        4. Return answer if found, None otherwise

        Args:
            query: User's query

        Returns:
            Dictionary with answer and metadata if found, None otherwise
        """
        if not self.memory_manager:
            return None

        try:
            logger.info("Checking memory for potential answer...")

            # Get conversation history and relevant memory items
            conversation_history = self.memory_manager.generate_conversation_summary(limit=10)
            relevant_memories = self.memory_manager.search_memory_semantic(
                query=query,
                limit=50,  # IMPROVED: Increased from 15 to 50 to search more chunks (45% of 112 chunks)
                content_types=['user_query', 'assistant_response', 'large_query_chunk']  # Added large_query_chunk
            )

            # If no relevant memories, skip memory check
            if not relevant_memories or conversation_history == "No previous conversation history.":
                logger.info("No relevant memory found, will search documents")
                return None

            # Build context from relevant memories
            memory_context_parts = []
            has_large_chunks = False

            for idx, mem in enumerate(relevant_memories, 1):
                content_type_label = "Question"
                if mem['content_type'] == 'assistant_response':
                    content_type_label = "Answer"
                elif mem['content_type'] == 'large_query_chunk':
                    content_type_label = "Document Chunk"
                    has_large_chunks = True

                relevance = mem.get('relevance_score', 0)
                memory_context_parts.append(
                    f"[Memory {idx}] ({content_type_label}, Relevance: {relevance:.2f})\n{mem['full_content']}"
                )

            memory_context = "\n\n".join(memory_context_parts)

            # Ask LLM if memory can answer the question
            # Adapt prompt based on whether we have large document chunks
            if has_large_chunks:
                memory_check_prompt = f"""You are a helpful AI assistant with access to document chunks and conversation history.

**Current Question:** {query}

**Available Context:**

{memory_context}

**Your Task:**
Analyze if you can answer the current question based ONLY on the context provided above (document chunks, conversation history, etc.).

**Instructions:**
1. If the answer is clearly present in the provided context, respond with:
   ANSWER_FOUND: <your detailed answer>

2. If the question requires information NOT in the provided context, respond with:
   NEED_SEARCH: <brief explanation why>

**Examples:**
- Question about facts that appear in Document Chunks → ANSWER_FOUND (provide answer citing which chunks)
- Question about something NOT mentioned in any context → NEED_SEARCH
- Question with partial information in context → ANSWER_FOUND (use available information)

**Important:**
- Prioritize information from Document Chunks as they contain the source material
- Use conversation history for additional context
- Cite which memory/chunk contains the information
- Be thorough - check ALL provided chunks before saying NEED_SEARCH

**Your Response:**"""
            else:
                memory_check_prompt = f"""You are a helpful AI assistant with access to conversation history.

**Current Question:** {query}

**Conversation History:**
{conversation_history}

**Relevant Memory Context:**
{memory_context}

**Your Task:**
Analyze if you can answer the current question based ONLY on the conversation history and memory context above.

**Instructions:**
1. If the answer is clearly present in the conversation history/memory, respond with:
   ANSWER_FOUND: <your detailed answer>

2. If the question requires information NOT in the conversation history (new information from documents), respond with:
   NEED_SEARCH: <brief explanation why>

Examples:
- "What fake painting..." when conversation history discusses paintings → ANSWER_FOUND
- "Who invested..." when previous Q&A mentions investments → ANSWER_FOUND
- "What is quantum computing?" without prior discussion → NEED_SEARCH
- "Tell me about X" where X wasn't discussed before → NEED_SEARCH

**Your Response:**"""

            # Get LLM decision
            response = await self.llm.ainvoke(memory_check_prompt)
            response_text = response.content.strip()

            logger.info(f"Memory check response: {response_text[:200]}")

            # Parse response
            if response_text.startswith("ANSWER_FOUND:"):
                answer = response_text.replace("ANSWER_FOUND:", "").strip()
                logger.info("✓ Answer found in memory, skipping document search")
                return {
                    'answer': answer,
                    'source': 'memory',
                    'relevant_memories': relevant_memories,
                    'confidence': 'high'
                }
            else:
                logger.info("✗ Memory check: Need to search documents")
                return None

        except Exception as e:
            logger.error(f"Error checking memory: {e}", exc_info=True)
            return None

    async def _track_and_emit_memory_state(self, input_tokens: int, output_tokens: int) -> Optional[Dict]:
        """
        Track token usage and get memory state for streaming

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Returns:
            Memory state dictionary for streaming, or None if memory is disabled
        """
        if not self.memory_manager:
            return None

        try:
            # Track token usage
            token_stats = self.memory_manager.track_token_usage(input_tokens, output_tokens)

            # Get memory state
            memory_state = self.memory_manager.export_memory_state_json()

            return {
                'token_stats': token_stats,
                'memory_state': memory_state
            }
        except Exception as e:
            logger.error(f"Error tracking memory state: {e}")
            return None

    async def _process_large_query_with_memory(self, query: str) -> str:
        """
        Process extremely large queries by chunking them into memory and searching

        This handles cases where the user provides a massive context (e.g., an entire book)
        along with a question. We:
        1. Extract the question from the query
        2. Chunk the large context into memory
        3. Search through memory to find relevant chunks
        4. Answer based on retrieved chunks

        Args:
            query: Large query containing question + context

        Returns:
            Answer based on memory search
        """
        try:
            logger.info("Processing large query with memory chunking")

            # Emit progress event
            yield f"data: {json.dumps({'type': 'thinking', 'content': '📝 Detected large query. Chunking into memory for processing...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Split query into question and passage
            # Detect patterns like "Answer X? Passage: Y" or "Question: X Context: Y"
            question = ""
            passage = query

            # Try to extract question
            for separator in ["? Passage:", "? Context:", "?\n\n", "?\n"]:
                if separator in query:
                    parts = query.split(separator, 1)
                    question = parts[0].strip() + "?"
                    passage = parts[1].strip() if len(parts) > 1 else ""
                    break

            # If no clear question found, look for "Answer this question"
            if not question or len(question) < 10:
                import re
                match = re.search(r'Answer this question[^?]*\?([^?]+\?)', query, re.IGNORECASE)
                if match:
                    question = match.group(1).strip()
                    # Rest is the passage
                    passage = re.sub(r'Answer this question[^?]*\?[^?]+\?', '', query, flags=re.IGNORECASE).strip()

            if not question:
                question = "What information is contained in this text?"

            logger.info(f"Extracted question: {question[:100]}")

            # Emit question extraction
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'❓ Question extracted: {question}', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Chunk the large passage into memory (chunk size: 1000 tokens)
            chunk_size = 1000
            tokens = self.memory_manager.encoding.encode(passage)
            total_chunks = (len(tokens) + chunk_size - 1) // chunk_size

            logger.info(f"Chunking {len(tokens)} tokens into {total_chunks} chunks")
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'✂️ Chunking {len(tokens):,} tokens into {total_chunks} chunks...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Store chunks in memory
            chunk_ids = []
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.memory_manager.encoding.decode(chunk_tokens)

                item_id = self.memory_manager.add_memory_item(
                    content=chunk_text,
                    content_type="large_query_chunk",
                    priority=5,
                    metadata={
                        "chunk_index": len(chunk_ids),
                        "total_chunks": total_chunks,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                chunk_ids.append(item_id)

            logger.info(f"Stored {len(chunk_ids)} chunks in memory")
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'💾 Stored {len(chunk_ids)} chunks in memory', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Now create embeddings for each chunk and search
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'🔍 Searching through {len(chunk_ids)} chunks for relevant information...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Generate embedding for the question
            question_embedding = await self.embedding_service.generate_query_embedding(question)

            # Retrieve all chunks and compute similarity
            cursor = self.memory_manager._get_cursor()
            cursor.execute("""
                SELECT id, full_content, token_count
                FROM memory_items
                WHERE session_id = %s AND content_type = 'large_query_chunk'
                ORDER BY metadata->>'chunk_index'
            """, (self.memory_manager.session_id,))

            chunks = cursor.fetchall()
            cursor.close()

            # Compute similarity for each chunk
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            chunk_scores = []
            for chunk in chunks:
                chunk_embedding = await self.embedding_service.generate_query_embedding(chunk['full_content'])
                similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
                chunk_scores.append({
                    'id': chunk['id'],
                    'content': chunk['full_content'],
                    'score': similarity,
                    'tokens': chunk['token_count']
                })

            # Sort by similarity and take top chunks
            chunk_scores.sort(key=lambda x: x['score'], reverse=True)
            top_chunks = chunk_scores[:10]  # Top 10 most relevant chunks

            logger.info(f"Retrieved top {len(top_chunks)} relevant chunks")
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'📚 Found {len(top_chunks)} relevant chunks. Generating answer...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Build context from top chunks
            context_parts = []
            for idx, chunk_data in enumerate(top_chunks, 1):
                context_parts.append(f"[Chunk {idx}] (Relevance: {chunk_data['score']:.3f})\n{chunk_data['content']}")

            context = "\n\n".join(context_parts)

            # Use LLM to answer the question based on context
            answer_prompt = f"""Answer the following question based ONLY on the provided context chunks.

Question: {question}

Context:
{context}

Instructions:
- Answer the question directly and concisely
- Only use information from the provided context
- If the answer is not in the context, say "I cannot find this information in the provided text"
- Cite which chunk(s) contain the answer

Answer:"""

            response = await self.llm.ainvoke(answer_prompt)
            answer = response.content.strip()

            # Format the final answer
            formatted_answer = format_markdown_output(answer)

            # Emit final answer
            final_event = {
                "type": "final_answer",
                "content": formatted_answer,
                "metadata": {
                    "mode": "large_query_memory_search",
                    "chunks_processed": len(chunk_ids),
                    "chunks_used": len(top_chunks),
                    "question": question
                },
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_event)}\n\n"

            # Track tokens
            input_tokens = self.memory_manager.count_tokens(answer_prompt)
            output_tokens = self.memory_manager.count_tokens(answer)
            memory_data = await self._track_and_emit_memory_state(input_tokens, output_tokens)

            if memory_data:
                memory_event = {
                    "type": "memory_state",
                    "content": "Memory state updated",
                    "metadata": memory_data,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(memory_event)}\n\n"

        except Exception as e:
            logger.error(f"Error in large query processing: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "content": f"❌ Error processing large query: {str(e)}",
                "metadata": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"

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
            # Check if this is a very large query (> 10K tokens) that should use memory chunking
            if self.memory_manager:
                query_tokens = self.memory_manager.count_tokens(query)
                logger.info(f"Query size: {query_tokens} tokens")

                # If query is extremely large (> 10,000 tokens), use memory-based chunking
                if query_tokens > 10000:
                    logger.info(f"Large query detected ({query_tokens} tokens). Using memory chunking approach.")
                    async for event in self._process_large_query_with_memory(query):
                        yield event
                    return  # Exit early, large query handled

                # For normal-sized queries, store in memory as usual
                try:
                    self.memory_manager.add_memory_item(
                        content=query,
                        content_type="user_query",
                        priority=8,
                        metadata={"document_id": document_id, "timestamp": datetime.now().isoformat()}
                    )
                    logger.info(f"Stored query in memory: {query_tokens} tokens")
                except Exception as e:
                    logger.error(f"Error storing query in memory: {e}")

            # Store document_id so tools can access it
            self.current_document_id = document_id
            logger.info(f"Processing query with document_id: {document_id}, MAX_PERFORMANCE={settings.max_performance}")

            # ===== MEMORY-FIRST STRATEGY: Check memory before searching documents =====
            if self.memory_manager:
                try:
                    # Emit memory check event
                    memory_check_event = {
                        "type": "thinking",
                        "content": "🧠 Checking conversation memory for relevant context...",
                        "metadata": {"stage": "memory_check"},
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(memory_check_event)}\n\n"

                    # Check if memory can answer the question
                    memory_result = await self._check_memory_for_answer(query)

                    if memory_result:
                        # Answer found in memory!
                        logger.info("✓ Answering from conversation memory")

                        # Emit success event
                        memory_found_event = {
                            "type": "tool_end",
                            "content": "✅ **Memory Search** completed\n📊 Found answer in conversation history",
                            "metadata": {
                                "tool_name": "memory_search",
                                "source": "conversation_memory",
                                "confidence": memory_result.get('confidence', 'high')
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(memory_found_event)}\n\n"

                        # Format the answer
                        formatted_answer = format_markdown_output(memory_result['answer'])

                        # Store response in memory
                        try:
                            self.memory_manager.add_memory_item(
                                content=formatted_answer,
                                content_type="assistant_response",
                                priority=8,
                                metadata={
                                    "source": "memory",
                                    "query": query,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error storing response in memory: {e}")

                        # Emit final answer from memory
                        final_event = {
                            "type": "final_answer",
                            "content": formatted_answer,
                            "metadata": {
                                "source": "conversation_memory",
                                "relevant_memories": len(memory_result.get('relevant_memories', [])),
                                "confidence": memory_result.get('confidence', 'high')
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(final_event)}\n\n"

                        # Track tokens
                        input_tokens = self.memory_manager.count_tokens(query) + 500
                        output_tokens = self.memory_manager.count_tokens(formatted_answer)
                        memory_data = await self._track_and_emit_memory_state(input_tokens, output_tokens)

                        if memory_data:
                            memory_state_event = {
                                "type": "memory_state",
                                "content": "Memory state updated",
                                "metadata": memory_data,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield f"data: {json.dumps(memory_state_event)}\n\n"

                        # Return early - question answered from memory
                        return

                    else:
                        # Memory doesn't have the answer, proceed to document search
                        memory_skip_event = {
                            "type": "thinking",
                            "content": "💡 No relevant answer in memory. Searching documents...",
                            "metadata": {"stage": "memory_check_complete"},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(memory_skip_event)}\n\n"

                except Exception as e:
                    logger.error(f"Error in memory check: {e}", exc_info=True)
                    # Continue with document search even if memory check fails

            # Check if no document was provided
            if document_id is None:
                warning_event = {
                    "type": "thinking",
                    "content": "⚠️  **No document provided by user.** Checking for existing data in the system...",
                    "metadata": {"has_document": False},
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(warning_event)}\n\n"

                # Check for any existing data in databases
                has_vector_data = False
                has_graph_data = False

                try:
                    # Check vector store for any documents
                    vector_count = await self.vector_store.get_embedding_count()
                    has_vector_data = vector_count > 0
                    logger.info(f"Vector store has {vector_count} chunks")
                except Exception as e:
                    logger.error(f"Error checking vector store: {e}")

                try:
                    # Check graph database for any nodes
                    graph_count = await neo4j_service.get_total_nodes_count()
                    has_graph_data = graph_count > 0
                    logger.info(f"Graph database has {graph_count} nodes")
                except Exception as e:
                    logger.error(f"Error checking graph database: {e}")

                # Emit status about available data
                if not has_vector_data and not has_graph_data:
                    no_data_event = {
                        "type": "thinking",
                        "content": "ℹ️  **No documents or graph data found in the system.**\n\nI'll answer your question using my general knowledge instead of searching uploaded documents.",
                        "metadata": {"has_vector_data": False, "has_graph_data": False},
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(no_data_event)}\n\n"
                else:
                    data_status = []
                    if has_vector_data:
                        data_status.append("uploaded documents")
                    if has_graph_data:
                        data_status.append("knowledge graph")

                    available_data_event = {
                        "type": "thinking",
                        "content": f"✅ Found existing data in: {', '.join(data_status)}.\n\nI'll search through this data to answer your question.",
                        "metadata": {"has_vector_data": has_vector_data, "has_graph_data": has_graph_data},
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(available_data_event)}\n\n"

            # Setup tools if not already done
            if not self.tools:
                self._setup_tools()

            # Check if MAX_PERFORMANCE mode is enabled
            if settings.max_performance:
                # MAX PERFORMANCE MODE: Execute all tools in parallel
                logger.info("Using MAX PERFORMANCE mode - executing all search tools in parallel")

                # Initial event
                yield f"data: {json.dumps({'type': 'thinking', 'content': '🚀 MAX PERFORMANCE MODE: Executing all search tools in parallel...', 'metadata': {'mode': 'max_performance'}, 'timestamp': datetime.now().isoformat()})}\n\n"

                # Emit tool start events for all tools
                tool_start_time = datetime.now().isoformat()
                for tool_name, icon in [("vector_search", "📚"), ("graph_search", "🕸️"), ("filter_search", "🔍")]:
                    yield f"data: {json.dumps({'type': 'tool_start', 'content': f'{icon} Executing **{tool_name}** in parallel...', 'metadata': {'tool_name': tool_name, 'start_time': tool_start_time}, 'timestamp': datetime.now().isoformat()})}\n\n"

                # Execute all tools in parallel
                parallel_results = await self._execute_all_tools_parallel(query)

                # Emit tool completion events
                for tool_name in ["vector_search", "graph_search", "filter_search"]:
                    result_preview = parallel_results[tool_name][:200] + "..." if len(parallel_results[tool_name]) > 200 else parallel_results[tool_name]
                    yield f"data: {json.dumps({'type': 'tool_end', 'content': f'✅ **{tool_name}** completed', 'metadata': {'tool_name': tool_name, 'output_length': len(parallel_results[tool_name])}, 'timestamp': datetime.now().isoformat()})}\n\n"

                # Synthesizing event
                yield f"data: {json.dumps({'type': 'thinking', 'content': '🧠 Synthesizing results from all search tools...', 'metadata': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

                # Synthesize results from all tools
                final_answer = await self._synthesize_parallel_results(query, parallel_results)

                # Store response in memory
                if self.memory_manager:
                    try:
                        self.memory_manager.add_memory_item(
                            content=final_answer,
                            content_type="assistant_response",
                            priority=8,
                            metadata={
                                "source": "document_search",
                                "mode": "max_performance",
                                "tools_used": ["vector_search", "graph_search", "filter_search"],
                                "query": query,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        logger.info("Stored response in memory")
                    except Exception as e:
                        logger.error(f"Error storing response in memory: {e}")

                # Emit final answer
                final_event = {
                    "type": "final_answer",
                    "content": final_answer,
                    "metadata": {
                        "mode": "max_performance",
                        "tools_used": ["vector_search", "graph_search", "filter_search"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(final_event)}\n\n"

                # Send metadata event
                metadata_event = {
                    "type": "metadata",
                    "content": "Query processing complete",
                    "metadata": {
                        "mode": "max_performance",
                        "tools_used": ["vector_search", "graph_search", "filter_search"]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(metadata_event)}\n\n"

                # Send memory state event if memory manager is available
                if self.memory_manager:
                    try:
                        # Estimate tokens (rough approximation for streaming mode)
                        input_tokens = self.memory_manager.count_tokens(query) + 1000  # Query + system prompts
                        output_tokens = self.memory_manager.count_tokens(final_answer)

                        memory_data = await self._track_and_emit_memory_state(input_tokens, output_tokens)
                        if memory_data:
                            memory_event = {
                                "type": "memory_state",
                                "content": "Memory state updated",
                                "metadata": memory_data,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield f"data: {json.dumps(memory_event)}\n\n"
                    except Exception as e:
                        logger.error(f"Error emitting memory state: {e}")

            else:
                # STANDARD MODE: Agent-based tool selection
                logger.info("Using STANDARD mode - agent will select the best tool")

                # Setup agent if not already done
                if not self.agent_executor:
                    self._setup_agent()

                # Create callback handler
                callback = EnhancedStreamingCallback()

                # Initial event
                yield f"data: {json.dumps({'type': 'thinking', 'content': '🔍 Analyzing your query and selecting appropriate tools...', 'metadata': {'mode': 'standard'}, 'timestamp': datetime.now().isoformat()})}\n\n"

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

                # Store response in memory
                if self.memory_manager:
                    try:
                        final_output = result.get("output", "")
                        if final_output:
                            self.memory_manager.add_memory_item(
                                content=final_output,
                                content_type="assistant_response",
                                priority=8,
                                metadata={
                                    "source": "document_search",
                                    "mode": "standard",
                                    "tools_used": [
                                        step[0].tool for step in result.get("intermediate_steps", [])
                                    ] if "intermediate_steps" in result else [],
                                    "query": query,
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                            logger.info("Stored response in memory")
                    except Exception as e:
                        logger.error(f"Error storing response in memory: {e}")

                # Send metadata event
                metadata_event = {
                    "type": "metadata",
                    "content": "Query processing complete",
                    "metadata": {
                        "mode": "standard",
                        "has_intermediate_steps": "intermediate_steps" in result,
                        "tools_used": [
                            step[0].tool for step in result.get("intermediate_steps", [])
                        ] if "intermediate_steps" in result else []
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(metadata_event)}\n\n"

                # Send memory state event if memory manager is available
                if self.memory_manager:
                    try:
                        # Get the final answer from the result
                        final_output = result.get("output", "")

                        # Estimate tokens
                        input_tokens = self.memory_manager.count_tokens(query) + 1000  # Query + system prompts
                        output_tokens = self.memory_manager.count_tokens(final_output)

                        memory_data = await self._track_and_emit_memory_state(input_tokens, output_tokens)
                        if memory_data:
                            memory_event = {
                                "type": "memory_state",
                                "content": "Memory state updated",
                                "metadata": memory_data,
                                "timestamp": datetime.now().isoformat()
                            }
                            yield f"data: {json.dumps(memory_event)}\n\n"
                    except Exception as e:
                        logger.error(f"Error emitting memory state: {e}")

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

            # Setup tools if not already done
            if not self.tools:
                self._setup_tools()

            # Check if MAX_PERFORMANCE mode is enabled
            if settings.max_performance:
                logger.info("Simple query using MAX PERFORMANCE mode")

                # Execute all tools in parallel
                parallel_results = await self._execute_all_tools_parallel(query)

                # Synthesize results
                final_answer = await self._synthesize_parallel_results(query, parallel_results)

                return {
                    "answer": final_answer,
                    "mode": "max_performance",
                    "tools_used": ["vector_search", "graph_search", "filter_search"],
                    "query": query
                }

            else:
                logger.info("Simple query using STANDARD mode")

                # Setup agent if not already done
                if not self.agent_executor:
                    self._setup_agent()

                result = await self.agent_executor.ainvoke(
                    {"input": query},
                    {"callbacks": []}
                )

                # Format the answer for better readability
                formatted_answer = format_markdown_output(result.get("output", "No answer generated"))

                return {
                    "answer": formatted_answer,
                    "mode": "standard",
                    "intermediate_steps": result.get("intermediate_steps", []),
                    "query": query
                }

        except Exception as e:
            logger.error(f"Error in simple query: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "query": query
            }
