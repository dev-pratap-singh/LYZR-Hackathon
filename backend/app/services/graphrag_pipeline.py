"""
GraphRAG Pipeline for Entity and Relationship Extraction
Based on Microsoft GraphRAG approach using LangChain
Chunks text into smaller pieces and extracts entities/relationships from each chunk
"""
import logging
from typing import List, Dict, Any, Tuple
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

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100):
        """
        Initialize GraphRAG Pipeline

        Args:
            chunk_size: Size of text chunks for entity extraction (default: 1200)
            chunk_overlap: Overlap between chunks to capture relationships (default: 100)
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

        logger.info(
            f"GraphRAG Pipeline initialized with model: {settings.graphrag_llm_model}, "
            f"chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}"
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

    async def process_document(
        self,
        text_content: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process entire document text and extract complete knowledge graph
        Uses Microsoft GraphRAG approach: chunk text → extract entities/relationships from each chunk → merge

        Args:
            text_content: Full document text (.txt from Docling or PyMuPDF)
            document_id: Document identifier

        Returns:
            Dict containing extracted graph data and statistics
        """
        start_time = time.time()

        try:
            logger.info(f"Starting GraphRAG processing for document {document_id}")
            logger.info(f"Document length: {len(text_content)} characters")

            # Step 1: Chunk the text into smaller pieces (Microsoft GraphRAG approach)
            chunks = self.text_splitter.split_text(text_content)
            logger.info(f"Split document into {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")

            # Step 2: Process each chunk and extract entities/relationships
            all_entities = []
            all_relationships = []
            all_graph_docs = []

            for idx, chunk_text in enumerate(chunks):
                logger.info(f"Processing chunk {idx + 1}/{len(chunks)} ({len(chunk_text)} chars)")

                try:
                    graph_doc = await self.extract_graph_from_text(
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=idx
                    )

                    all_graph_docs.append(graph_doc)

                    # Collect entities from this chunk
                    for node in graph_doc.nodes:
                        all_entities.append({
                            "id": node.id,
                            "type": node.type,
                            "properties": node.properties,
                            "chunk_index": idx
                        })

                    # Collect relationships from this chunk
                    for rel in graph_doc.relationships:
                        all_relationships.append({
                            "source": rel.source.id,
                            "target": rel.target.id,
                            "type": rel.type,
                            "properties": rel.properties,
                            "chunk_index": idx
                        })

                    logger.info(
                        f"Chunk {idx + 1}: {len(graph_doc.nodes)} entities, "
                        f"{len(graph_doc.relationships)} relationships"
                    )

                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {e}", exc_info=True)
                    # Continue processing other chunks even if one fails
                    continue

            # Step 3: Deduplicate entities and relationships
            # Entities with same ID are merged (keeps first occurrence)
            unique_entities = {}
            for entity in all_entities:
                entity_id = entity["id"]
                if entity_id not in unique_entities:
                    unique_entities[entity_id] = entity
                else:
                    # Merge descriptions if available
                    existing = unique_entities[entity_id]
                    if "description" in entity.get("properties", {}):
                        if "description" not in existing.get("properties", {}):
                            existing["properties"]["description"] = entity["properties"]["description"]

            # Deduplicate relationships (same source-target-type)
            unique_relationships = {}
            for rel in all_relationships:
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
                "processing_time": processing_time,
                "graph_documents": all_graph_docs  # All graph documents for Neo4j import
            }

            logger.info(
                f"GraphRAG processing complete for {document_id}: "
                f"{len(final_entities)} entities, {len(final_relationships)} relationships "
                f"from {len(chunks)} chunks in {processing_time}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing document {document_id} with GraphRAG: {e}", exc_info=True)
            raise

    async def process_chunks(
        self,
        chunks: List[str],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Process multiple text chunks and extract knowledge graph

        Args:
            chunks: List of text chunks
            document_id: Document identifier

        Returns:
            Combined graph data from all chunks
        """
        start_time = time.time()

        try:
            logger.info(f"Processing {len(chunks)} chunks for document {document_id}")

            all_entities = []
            all_relationships = []
            all_graph_docs = []

            for idx, chunk_text in enumerate(chunks):
                graph_doc = await self.extract_graph_from_text(
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=idx
                )

                all_graph_docs.append(graph_doc)

                # Collect entities
                for node in graph_doc.nodes:
                    all_entities.append({
                        "id": node.id,
                        "type": node.type,
                        "properties": node.properties,
                        "chunk_index": idx
                    })

                # Collect relationships
                for rel in graph_doc.relationships:
                    all_relationships.append({
                        "source": rel.source.id,
                        "target": rel.target.id,
                        "type": rel.type,
                        "properties": rel.properties,
                        "chunk_index": idx
                    })

            processing_time = int(time.time() - start_time)

            result = {
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
                f"in {processing_time}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing chunks for document {document_id}: {e}")
            raise
