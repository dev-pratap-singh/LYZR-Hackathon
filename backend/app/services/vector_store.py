"""
PGVector Storage and Search Service
Handles storing and searching embeddings in PGVector
"""
import logging
from typing import List, Dict, Optional
from uuid import UUID
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert

from app.models import Embedding
from app.database import get_pgvector_db
from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for PGVector operations"""

    def __init__(self):
        self.top_k = settings.top_k_results

    async def store_embeddings(
        self,
        document_id: UUID,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store chunk embeddings in PGVector

        Args:
            document_id: UUID of the document
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors

        Returns:
            Number of embeddings stored
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

            with get_pgvector_db() as db:
                stored_count = 0

                for chunk, embedding in zip(chunks, embeddings):
                    # Create embedding record
                    embedding_record = Embedding(
                        document_id=document_id,
                        chunk_index=chunk['index'],
                        chunk_text=chunk['text'],
                        chunk_size=chunk['size'],
                        embedding=embedding,
                        chunk_hash=chunk['hash'],
                        chunk_metadata=chunk.get('chunk_metadata', {})
                    )

                    db.add(embedding_record)
                    stored_count += 1

                    # Commit in batches of 100
                    if stored_count % 100 == 0:
                        db.commit()
                        logger.info(f"Stored {stored_count}/{len(chunks)} embeddings")

                # Final commit
                db.commit()

            logger.info(f"Successfully stored {stored_count} embeddings for document {document_id}")
            return stored_count

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        document_id: Optional[UUID] = None
    ) -> List[Dict]:
        """
        Perform vector similarity search

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optional filter by document ID

        Returns:
            List of matching chunks with similarity scores
        """
        try:
            k = top_k or self.top_k

            with get_pgvector_db() as db:
                # Build query
                query = select(
                    Embedding.id,
                    Embedding.document_id,
                    Embedding.chunk_text,
                    Embedding.chunk_index,
                    Embedding.chunk_metadata,
                    Embedding.embedding.cosine_distance(query_embedding).label('distance')
                )

                # Apply document filter if specified
                if document_id:
                    query = query.where(Embedding.document_id == document_id)

                # Order by similarity and limit
                query = query.order_by('distance').limit(k)

                # Execute query
                results = db.execute(query).fetchall()

                # Format results
                search_results = []
                for row in results:
                    search_results.append({
                        'id': str(row.id),
                        'document_id': str(row.document_id),
                        'chunk_text': row.chunk_text,
                        'chunk_index': row.chunk_index,
                        'chunk_metadata': row.chunk_metadata,
                        'similarity_score': 1 - row.distance,  # Convert distance to similarity
                        'distance': row.distance
                    })

                logger.info(f"Vector search returned {len(search_results)} results")
                return search_results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

    async def get_document_chunks(self, document_id: UUID) -> List[Dict]:
        """
        Get all chunks for a document

        Args:
            document_id: Document UUID

        Returns:
            List of chunks
        """
        try:
            with get_pgvector_db() as db:
                query = select(Embedding).where(
                    Embedding.document_id == document_id
                ).order_by(Embedding.chunk_index)

                results = db.execute(query).scalars().all()

                chunks = [{
                    'id': str(chunk.id),
                    'chunk_index': chunk.chunk_index,
                    'chunk_text': chunk.chunk_text,
                    'chunk_size': chunk.chunk_size,
                    'chunk_metadata': chunk.chunk_metadata
                } for chunk in results]

                return chunks

        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise

    async def delete_document_embeddings(self, document_id: UUID) -> int:
        """
        Delete all embeddings for a document

        Args:
            document_id: Document UUID

        Returns:
            Number of embeddings deleted
        """
        try:
            with get_pgvector_db() as db:
                result = db.query(Embedding).filter(
                    Embedding.document_id == document_id
                ).delete()

                db.commit()

                logger.info(f"Deleted {result} embeddings for document {document_id}")
                return result

        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            raise

    async def get_embedding_count(self, document_id: Optional[UUID] = None) -> int:
        """
        Get total number of embeddings

        Args:
            document_id: Optional filter by document

        Returns:
            Count of embeddings
        """
        try:
            with get_pgvector_db() as db:
                query = select(func.count(Embedding.id))

                if document_id:
                    query = query.where(Embedding.document_id == document_id)

                count = db.execute(query).scalar()
                return count or 0

        except Exception as e:
            logger.error(f"Error getting embedding count: {e}")
            raise
