"""
GraphRAG Pipeline for Entity and Relationship Extraction
Based on Microsoft GraphRAG approach using LangChain
Chunks text into smaller pieces and extracts entities/relationships from each chunk
"""
import logging
import asyncio
import random
from typing import List, Dict, Any, Tuple, Optional
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

from app.config import settings

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Pipeline for extracting knowledge graph from text using GraphRAG"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 400):
        """
        Initialize GraphRAG Pipeline

        Args:
            chunk_size: Size of text chunks for entity extraction (default: 1200)
            chunk_overlap: Overlap between chunks to capture relationships (default: 400)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.llm = ChatOpenAI(
            model=settings.graphrag_llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        # Initialize text splitter for chunking large documents
        # Similar to Microsoft GraphRAG approach
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

        # Initialize LLM Graph Transformer with enhanced configuration
        # Extracts entities with descriptions and relationships with descriptions
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            node_properties=["description"],
            relationship_properties=["description"],
            # Allow more diverse entity types
            allowed_nodes=[],  # Empty list means all node types are allowed
            allowed_relationships=[]  # Empty list means all relationship types are allowed
        )

        # Semaphore for controlling concurrent chunk processing
        self.semaphore = asyncio.Semaphore(settings.graphrag_concurrency)
        self.max_retries = settings.graphrag_max_retries
        self.base_backoff = settings.graphrag_base_backoff

        logger.info(
            f"GraphRAG Pipeline initialized with model: {settings.graphrag_llm_model}, "
            f"chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, "
            f"concurrency: {settings.graphrag_concurrency}, "
            f"max_retries: {settings.graphrag_max_retries}"
        )

    async def extract_graph_from_text(
        self,
        text: str,
        document_id: str,
        chunk_index: int = 0
    ) -> GraphDocument:
        """
        Extract entities and relationships from a text chunk

        Args:
            text: Text content to process
            document_id: Document identifier
            chunk_index: Index of the chunk in the document

        Returns:
            GraphDocument containing nodes and relationships
        """
        try:
            # Create LangChain document
            doc = Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "chunk_index": chunk_index
                }
            )

            # Extract graph using LLM
            graph_documents = await self.llm_transformer.aconvert_to_graph_documents([doc])

            if graph_documents:
                return graph_documents[0]
            else:
                logger.warning(f"No graph extracted from document {document_id}, chunk {chunk_index}")
                return GraphDocument(nodes=[], relationships=[], source=doc)

        except Exception as e:
            logger.error(f"Error extracting graph from text: {e}")
            raise

    async def _extract_with_retry(
        self,
        text: str,
        document_id: str,
        chunk_index: int
    ) -> GraphDocument:
        """
        Extract graph from text with retry logic and exponential backoff

        Args:
            text: Text content to process
            document_id: Document identifier
            chunk_index: Index of the chunk

        Returns:
            GraphDocument containing nodes and relationships
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.extract_graph_from_text(text, document_id, chunk_index)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Failed to process chunk {chunk_index} after {self.max_retries} attempts: {e}"
                    )
                    raise
                # Exponential backoff with jitter
                backoff = self.base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                logger.warning(
                    f"Chunk {chunk_index} attempt {attempt} failed, retrying in {backoff:.2f}s: {e}"
                )
                await asyncio.sleep(backoff)

    async def _bounded_extract(
        self,
        text: str,
        document_id: str,
        chunk_index: int
    ) -> GraphDocument:
        """
        Extract graph from text with semaphore to control concurrency

        Args:
            text: Text content to process
            document_id: Document identifier
            chunk_index: Index of the chunk

        Returns:
            GraphDocument containing nodes and relationships
        """
        async with self.semaphore:
            return await self._extract_with_retry(text, document_id, chunk_index)

    async def _process_single_chunk(
        self,
        chunk_index: int,
        chunk_text: str,
        document_id: str,
        results: List[Optional[Dict[str, Any]]]
    ) -> None:
        """
        Process a single chunk and store result at the correct index

        Args:
            chunk_index: Index of the chunk in the document
            chunk_text: Text content of the chunk
            document_id: Document identifier
            results: List to store results (maintains order)
        """
        try:
            logger.info(f"Processing chunk {chunk_index + 1} ({len(chunk_text)} chars)")

            graph_doc = await self._bounded_extract(
                text=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index
            )

            # Extract entities
            entities = []
            for node in graph_doc.nodes:
                entities.append({
                    "id": node.id,
                    "type": node.type,
                    "properties": node.properties,
                    "chunk_index": chunk_index
                })

            # Extract relationships
            relationships = []
            for rel in graph_doc.relationships:
                relationships.append({
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type": rel.type,
                    "properties": rel.properties,
                    "chunk_index": chunk_index
                })

            # Store result at the correct index to preserve order
            results[chunk_index] = {
                "graph_doc": graph_doc,
                "entities": entities,
                "relationships": relationships,
                "chunk_index": chunk_index
            }

            logger.info(
                f"Chunk {chunk_index + 1} complete: {len(entities)} entities, "
                f"{len(relationships)} relationships"
            )

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {e}", exc_info=True)
            # Store None to indicate failure but preserve order
            results[chunk_index] = None

    async def process_document(
        self,
        text_content: str,
        document_id: str,
        enable_multipass: bool = None
    ) -> Dict[str, Any]:
        """
        Process entire document text and extract complete knowledge graph with optional multi-pass enrichment

        Multi-pass approach (when enabled):
        - Pass 1: Initial broad extraction - captures main entities and relationships
        - Pass 2: Enrichment pass - finds missing entities mentioned in relationships
        - Pass 3: Relationship enrichment - discovers indirect connections

        Single-pass approach (when disabled):
        - Traditional: chunk text → extract entities/relationships → merge

        Args:
            text_content: Full document text (.txt from Docling or PyMuPDF)
            document_id: Document identifier
            enable_multipass: Override config setting for multi-pass enrichment

        Returns:
            Dict containing extracted graph data and statistics from all passes
        """
        start_time = time.time()

        try:
            # Determine if multi-pass is enabled
            use_multipass = enable_multipass if enable_multipass is not None else settings.graphrag_enable_multipass
            num_passes = settings.graphrag_num_passes if use_multipass else 1

            logger.info(f"Starting GraphRAG processing for document {document_id}")
            logger.info(f"Document length: {len(text_content)} characters")
            logger.info(f"Multi-pass enrichment: {'ENABLED' if use_multipass else 'DISABLED'} ({num_passes} pass(es))")

            # Step 1: Chunk the text into smaller pieces (Microsoft GraphRAG approach)
            chunks = self.text_splitter.split_text(text_content)
            logger.info(f"Split document into {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")

            all_passes_entities = []
            all_passes_relationships = []
            all_passes_graph_docs = []

            # Process each pass
            for pass_num in range(1, num_passes + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"PASS {pass_num}/{num_passes}: {self._get_pass_description(pass_num)}")
                logger.info(f"{'='*80}")

                # For pass 2 and 3, we use existing knowledge to guide extraction
                extraction_context = None
                if pass_num > 1 and all_passes_entities:
                    extraction_context = self._build_extraction_context(
                        all_passes_entities,
                        all_passes_relationships,
                        pass_num
                    )

                # Process chunks for this pass
                pass_result = await self._process_single_pass(
                    chunks=chunks,
                    document_id=document_id,
                    pass_number=pass_num,
                    extraction_context=extraction_context
                )

                # Collect results from this pass
                all_passes_entities.extend(pass_result["entities"])
                all_passes_relationships.extend(pass_result["relationships"])
                all_passes_graph_docs.extend(pass_result["graph_documents"])

                logger.info(
                    f"Pass {pass_num} complete: "
                    f"{len(pass_result['entities'])} new entities, "
                    f"{len(pass_result['relationships'])} new relationships"
                )

            # Final deduplication across all passes
            logger.info("\nPerforming final deduplication across all passes...")
            unique_entities = {}
            for entity in all_passes_entities:
                entity_id = entity["id"]
                if entity_id not in unique_entities:
                    unique_entities[entity_id] = entity
                else:
                    # Merge descriptions if available
                    existing = unique_entities[entity_id]
                    if "description" in entity.get("properties", {}):
                        existing_desc = existing.get("properties", {}).get("description", "")
                        new_desc = entity["properties"]["description"]
                        # Append new description if different
                        if new_desc and new_desc not in existing_desc:
                            combined_desc = f"{existing_desc}; {new_desc}" if existing_desc else new_desc
                            existing["properties"]["description"] = combined_desc

            # Deduplicate relationships (same source-target-type)
            unique_relationships = {}
            for rel in all_passes_relationships:
                rel_key = f"{rel['source']}-{rel['type']}-{rel['target']}"
                if rel_key not in unique_relationships:
                    unique_relationships[rel_key] = rel

            final_entities = list(unique_entities.values())
            final_relationships = list(unique_relationships.values())

            processing_time = int(time.time() - start_time)

            result = {
                "document_id": document_id,
                "entities": final_entities,
                "relationships": final_relationships,
                "entities_count": len(final_entities),
                "relationships_count": len(final_relationships),
                "chunks_processed": len(chunks),
                "num_passes": num_passes,
                "processing_time": processing_time,
                "graph_documents": all_passes_graph_docs  # All graph documents for Neo4j import
            }

            logger.info(
                f"\n{'='*80}"
            )
            logger.info(
                f"GraphRAG processing complete for {document_id}: "
                f"{len(final_entities)} total entities, {len(final_relationships)} total relationships "
                f"from {len(chunks)} chunks across {num_passes} pass(es) in {processing_time}s"
            )
            logger.info(f"{'='*80}\n")

            return result

        except Exception as e:
            logger.error(f"Error processing document {document_id} with GraphRAG: {e}", exc_info=True)
            raise

    def _get_pass_description(self, pass_num: int) -> str:
        """Get description for each pass"""
        descriptions = {
            1: "Initial Extraction - Broad entity and relationship discovery",
            2: "Enrichment Pass - Finding missing entities and connections",
            3: "Relationship Enhancement - Discovering indirect relationships"
        }
        return descriptions.get(pass_num, f"Pass {pass_num}")

    def _build_extraction_context(
        self,
        existing_entities: List[Dict],
        existing_relationships: List[Dict],
        pass_num: int
    ) -> str:
        """
        Build context string to guide subsequent extraction passes

        Args:
            existing_entities: Entities found in previous passes
            existing_relationships: Relationships found in previous passes
            pass_num: Current pass number

        Returns:
            Context string for LLM
        """
        # Get unique entity names (limit to top 50 most frequent)
        entity_names = set()
        for entity in existing_entities[:50]:
            entity_names.add(entity["id"])

        context = f"\nPrevious Pass Information:\n"
        context += f"- Known entities: {', '.join(list(entity_names)[:30])}"

        if pass_num == 2:
            context += "\n- Focus: Find entities that are referenced but not yet extracted"
            context += "\n- Look for: Missing people, organizations, locations, concepts"
        elif pass_num == 3:
            context += "\n- Focus: Discover relationships between existing entities"
            context += "\n- Look for: Indirect connections, inferred relationships, contextual links"

        return context

    async def _process_single_pass(
        self,
        chunks: List[str],
        document_id: str,
        pass_number: int,
        extraction_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process all chunks for a single pass

        Args:
            chunks: List of text chunks
            document_id: Document identifier
            pass_number: Current pass number
            extraction_context: Optional context from previous passes

        Returns:
            Dict with entities, relationships, and graph documents from this pass
        """
        logger.info(f"Starting parallel processing of {len(chunks)} chunks with concurrency={settings.graphrag_concurrency}")

        # Initialize results list to preserve order
        results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

        # Create tasks for all chunks
        tasks = []
        for idx, chunk_text in enumerate(chunks):
            # Add context for enrichment passes
            if extraction_context and pass_number > 1:
                chunk_with_context = f"{extraction_context}\n\n{chunk_text}"
            else:
                chunk_with_context = chunk_text

            task = asyncio.create_task(
                self._process_single_chunk(idx, chunk_with_context, document_id, results)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Collect results from all chunks
        all_entities = []
        all_relationships = []
        all_graph_docs = []

        for result in results:
            if result is not None:
                all_graph_docs.append(result["graph_doc"])
                all_entities.extend(result["entities"])
                all_relationships.extend(result["relationships"])

        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "graph_documents": all_graph_docs,
            "pass_number": pass_number
        }

    async def process_chunks(
        self,
        chunks: List[str],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process multiple text chunks and extract knowledge graph using parallel processing

        Args:
            chunks: List of text chunks
            document_id: Document identifier

        Returns:
            Combined graph data from all chunks
        """
        start_time = time.time()

        try:
            logger.info(f"Processing {len(chunks)} chunks for document {document_id}")
            logger.info(f"Starting parallel processing with concurrency={settings.graphrag_concurrency}")

            # Initialize results list to preserve order
            results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

            # Create tasks for all chunks
            tasks = []
            for idx, chunk_text in enumerate(chunks):
                task = asyncio.create_task(
                    self._process_single_chunk(idx, chunk_text, document_id, results)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks)

            # Collect results from all chunks
            all_entities = []
            all_relationships = []
            all_graph_docs = []

            for result in results:
                if result is not None:
                    all_graph_docs.append(result["graph_doc"])
                    all_entities.extend(result["entities"])
                    all_relationships.extend(result["relationships"])

            processing_time = int(time.time() - start_time)

            final_result = {
                "document_id": document_id,
                "entities": all_entities,
                "relationships": all_relationships,
                "entities_count": len(all_entities),
                "relationships_count": len(all_relationships),
                "processing_time": processing_time,
                "graph_documents": all_graph_docs
            }

            logger.info(
                f"Processed {len(chunks)} chunks for {document_id}: "
                f"{len(all_entities)} entities, {len(all_relationships)} relationships "
                f"in {processing_time}s (parallel processing)"
            )

            return final_result

        except Exception as e:
            logger.error(f"Error processing chunks for document {document_id}: {e}")
            raise
