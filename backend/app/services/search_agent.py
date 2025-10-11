"""
Search Agent with Chain of Thought Reasoning
Simplified version for Step 2 - Vector Search only
"""
import logging
from typing import AsyncGenerator, Dict, List, Optional
from datetime import datetime
import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from openai import AsyncOpenAI

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.bm25_search import BM25SearchService
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)


class SearchAgent:
    """Agent for processing queries with vector search and CoT reasoning"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            openai_api_key=settings.openai_api_key,
            streaming=True
        )
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.bm25_search = BM25SearchService()
        self.reranker = RerankerService()

    async def emit_event(
        self,
        event_type: str,
        content: str,
        metadata: Dict = None
    ) -> Dict:
        """Create a streaming event"""
        return {
            "type": event_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }

    async def process_query_with_streaming(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process query with vector search and stream results

        Args:
            query: User's question
            document_id: Optional document ID to search within

        Yields:
            SSE-formatted event strings
        """
        try:
            # Step 1: Analyze query
            yield f"data: {json.dumps(await self.emit_event('thinking', '🔍 Analyzing your question...'))}\n\n"

            # Step 2: Generate query embedding
            yield f"data: {json.dumps(await self.emit_event('tool_start', '📊 Generating query embedding...', {'tool': 'embedding_service'}))}\n\n"

            query_embedding = await self.embedding_service.generate_query_embedding(query)

            yield f"data: {json.dumps(await self.emit_event('tool_end', '✓ Query embedding generated', {'embedding_dim': len(query_embedding)}))}\n\n"

            # Step 3: Hybrid Search (Vector + BM25)
            yield f"data: {json.dumps(await self.emit_event('tool_start', '🔎 Performing hybrid search (Vector + BM25)...', {'tool': 'hybrid_search'}))}\n\n"

            # Vector search
            vector_results = await self.vector_store.vector_search(
                query_embedding=query_embedding,
                document_id=document_id,
                top_k=settings.top_k_results * 2  # Get more candidates for reranking
            )

            # BM25 keyword search
            bm25_results = await self.bm25_search.search(
                query=query,
                document_id=document_id,
                top_k=settings.top_k_results * 2  # Get more candidates
            )

            if not vector_results and not bm25_results:
                yield f"data: {json.dumps(await self.emit_event('error', '❌ No relevant content found. Please upload a document first.'))}\n\n"
                return

            yield f"data: {json.dumps(await self.emit_event('tool_end', f'✓ Retrieved {len(vector_results)} vector + {len(bm25_results)} keyword results', {'vector_count': len(vector_results), 'bm25_count': len(bm25_results)}))}\n\n"

            # Step 4: Merge and deduplicate results
            yield f"data: {json.dumps(await self.emit_event('tool_start', '🔄 Merging and deduplicating results...', {'tool': 'merge'}))}\n\n"

            # Combine results (deduplicate by chunk ID)
            seen_ids = set()
            combined_results = []

            for result in vector_results + bm25_results:
                chunk_id = result['id']
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_results.append(result)

            yield f"data: {json.dumps(await self.emit_event('tool_end', f'✓ Combined to {len(combined_results)} unique results', {'unique_count': len(combined_results)}))}\n\n"

            # Step 5: Rerank results
            yield f"data: {json.dumps(await self.emit_event('tool_start', '🎯 Reranking with cross-encoder...', {'tool': 'reranker'}))}\n\n"

            search_results = self.reranker.rerank(
                query=query,
                results=combined_results,
                top_k=settings.top_k_results
            )

            yield f"data: {json.dumps(await self.emit_event('tool_end', f'✓ Reranked to top {len(search_results)} results', {'final_count': len(search_results), 'top_rerank_score': search_results[0].get('rerank_score', 0) if search_results else 0}))}\n\n"

            # Step 4: CoT Reasoning - Analyze results
            analysis_msg = "💭 **ANALYSIS**: Examining retrieved content...\n- Found relevant passages from the document\n- Preparing to synthesize answer"
            yield f"data: {json.dumps(await self.emit_event('reasoning', analysis_msg))}\n\n"

            # Format context from search results
            context_parts = []
            for idx, result in enumerate(search_results, 1):
                rerank_score = result.get('rerank_score', 0)
                passage_text = f"[Passage {idx}] (Rerank Score: {rerank_score:.3f})\n{result['chunk_text']}"
                context_parts.append(passage_text)

            context = "\n\n".join(context_parts)

            # Step 5: Generate answer with reasoning
            yield f"data: {json.dumps(await self.emit_event('reasoning', '🎯 **STRATEGY**: Using retrieved passages to answer your question'))}\n\n"

            # Create prompt with CoT
            system_prompt = """You are an AI assistant that answers questions based on provided context.

**Your Task:**
1. Analyze the question carefully
2. Review the provided passages
3. Synthesize a clear, accurate answer
4. Cite specific passages when relevant
5. If the context doesn't contain enough information, say so clearly

**Response Format:**
- Provide a direct answer first
- Explain your reasoning
- Reference specific passages when making claims
- Be concise but thorough"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""**Question:** {query}

**Retrieved Context:**
{context}

Please answer the question based on the context above. Show your reasoning.""")
            ]

            # Stream the LLM response
            yield f"data: {json.dumps(await self.emit_event('reasoning', '✍️ Generating answer...'))}\n\n"

            full_response = ""
            async for chunk in self.stream_llm_response(messages):
                full_response += chunk
                # Stream tokens to frontend (optional - can be enabled for token-by-token streaming)
                # yield f"data: {json.dumps(await self.emit_event('token', chunk))}\n\n"

            # Send final answer
            yield f"data: {json.dumps(await self.emit_event('final_answer', full_response, {'sources_count': len(search_results), 'query': query}))}\n\n"

            # Send metadata
            yield f"data: {json.dumps(await self.emit_event('metadata', 'Query processing complete', {'total_steps': 5, 'results_used': len(search_results)}))}\n\n"

        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            yield f"data: {json.dumps(await self.emit_event('error', f'Error processing query: {str(e)}'))}\n\n"

    async def stream_llm_response(self, messages: List) -> AsyncGenerator[str, None]:
        """Stream LLM response"""
        try:
            # Convert LangChain messages to OpenAI format
            # LangChain uses "human", "ai", "system" but OpenAI expects "user", "assistant", "system"
            role_mapping = {
                "human": "user",
                "ai": "assistant",
                "system": "system"
            }

            openai_messages = []
            for m in messages:
                role = role_mapping.get(m.type, m.type)
                openai_messages.append({"role": role, "content": m.content})

            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=openai_messages,
                temperature=0.3,
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            raise

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
            # Generate embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)

            # Vector search
            search_results = await self.vector_store.vector_search(
                query_embedding=query_embedding,
                document_id=document_id
            )

            if not search_results:
                return {
                    "answer": "No relevant content found. Please upload a document first.",
                    "sources": []
                }

            # Build context
            context = "\n\n".join([
                f"[Passage {i+1}]\n{r['chunk_text']}"
                for i, r in enumerate(search_results)
            ])

            # Generate answer
            messages = [
                SystemMessage(content="You are a helpful AI assistant. Answer questions based on the provided context."),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{context}\n\nAnswer:")
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "answer": response.content,
                "sources": search_results,
                "query": query
            }

        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "query": query
            }
