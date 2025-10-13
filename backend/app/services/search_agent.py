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

    def __init__(self):
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,  # Zero temperature for more deterministic, factual responses
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

Use **graph_search** for:
- "Tell me about X", "Who is X", "What is X" - when X is a person or entity
- "How are X and Y related?"
- "What's the relationship between..."
- Connection or relationship questions
- ANY query where user says "use graph search" or "graph search only"

Use **vector_search** for:
- General "what/why/how" questions about concepts or topics
- Definitions, explanations, summaries of general concepts
- Finding information, facts, or content from documents
- DEFAULT choice when unsure AND user hasn't specified a tool

Use **filter_search** for:
- Date/metadata filtering ("documents from 2023")
- Category/tag filtering ("papers in category X")
- Author filtering ("documents by author Y")

**IMPORTANT:**
- You have access to {tool_names}
- You MUST use these tools - they are not optional
- Call the appropriate tool IMMEDIATELY, don't just talk about calling it
- After getting tool results, synthesize a clear answer citing the passages

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
            logger.info(f"Processing query with document_id: {document_id}, MAX_PERFORMANCE={settings.max_performance}")

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
