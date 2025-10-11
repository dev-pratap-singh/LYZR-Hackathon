"""
BM25 Keyword Search Service
Provides keyword-based search for hybrid retrieval
"""
import logging
from typing import List, Dict, Optional
from uuid import UUID
import pickle
import os
from pathlib import Path

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.models import Embedding
from app.database import get_pgvector_db
from app.config import settings

logger = logging.getLogger(__name__)


class BM25SearchService:
    """Service for BM25 keyword-based search"""

    def __init__(self):
        self.storage_path = Path(settings.storage_path)
        self.bm25_cache = {}  # Cache BM25 indexes per document
        self.corpus_cache = {}  # Cache corpus per document

    def _get_cache_path(self, document_id: str) -> Path:
        """Get path to BM25 index cache file"""
        return self.storage_path / f"bm25_{document_id}.pkl"

    async def build_index(self, document_id: UUID) -> None:
        """
        Build BM25 index for a document

        Args:
            document_id: Document UUID
        """
        try:
            doc_id_str = str(document_id)

            # Get all chunks for the document
            with get_pgvector_db() as db:
                query = select(
                    Embedding.id,
                    Embedding.chunk_text,
                    Embedding.chunk_index
                ).where(
                    Embedding.document_id == document_id
                ).order_by(Embedding.chunk_index)

                results = db.execute(query).fetchall()

            if not results:
                logger.warning(f"No chunks found for document {doc_id_str}")
                return

            # Tokenize corpus (simple whitespace tokenization)
            corpus = []
            chunk_ids = []

            for row in results:
                # Simple tokenization: lowercase and split
                tokens = row.chunk_text.lower().split()
                corpus.append(tokens)
                chunk_ids.append(str(row.id))

            # Build BM25 index
            bm25 = BM25Okapi(corpus)

            # Cache the index and corpus
            self.bm25_cache[doc_id_str] = bm25
            self.corpus_cache[doc_id_str] = {
                'corpus': corpus,
                'chunk_ids': chunk_ids
            }

            # Persist to disk
            cache_data = {
                'bm25': bm25,
                'corpus': corpus,
                'chunk_ids': chunk_ids
            }

            cache_path = self._get_cache_path(doc_id_str)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"✓ Built BM25 index for document {doc_id_str} ({len(corpus)} chunks)")

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            raise

    async def load_index(self, document_id: str) -> bool:
        """
        Load BM25 index from cache

        Args:
            document_id: Document UUID string

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check memory cache first
            if document_id in self.bm25_cache:
                return True

            # Try loading from disk
            cache_path = self._get_cache_path(document_id)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                self.bm25_cache[document_id] = cache_data['bm25']
                self.corpus_cache[document_id] = {
                    'corpus': cache_data['corpus'],
                    'chunk_ids': cache_data['chunk_ids']
                }

                logger.info(f"✓ Loaded BM25 index for document {document_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False

    async def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform BM25 keyword search

        Args:
            query: Search query
            document_id: Optional document filter
            top_k: Number of results to return

        Returns:
            List of search results with BM25 scores
        """
        try:
            if not document_id:
                logger.warning("BM25 search requires document_id")
                return []

            # Load index if not in cache
            if document_id not in self.bm25_cache:
                loaded = await self.load_index(document_id)
                if not loaded:
                    # Build index if it doesn't exist
                    await self.build_index(UUID(document_id))

            bm25 = self.bm25_cache.get(document_id)
            corpus_data = self.corpus_cache.get(document_id)

            if not bm25 or not corpus_data:
                logger.error(f"BM25 index not available for document {document_id}")
                return []

            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)

            # Get top-k chunk IDs
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            # Fetch full chunk data from database
            chunk_ids = [corpus_data['chunk_ids'][i] for i in top_indices]

            with get_pgvector_db() as db:
                query_stmt = select(Embedding).where(
                    Embedding.id.in_(chunk_ids)
                )
                results = db.execute(query_stmt).scalars().all()

            # Create results dictionary
            results_dict = {str(r.id): r for r in results}

            # Format results with BM25 scores
            search_results = []
            for idx in top_indices:
                chunk_id = corpus_data['chunk_ids'][idx]
                chunk = results_dict.get(chunk_id)

                if chunk:
                    search_results.append({
                        'id': chunk_id,
                        'document_id': str(chunk.document_id),
                        'chunk_text': chunk.chunk_text,
                        'chunk_index': chunk.chunk_index,
                        'chunk_metadata': chunk.chunk_metadata,
                        'bm25_score': float(scores[idx]),
                        'source': 'bm25'
                    })

            logger.info(f"BM25 search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    async def delete_index(self, document_id: str) -> None:
        """
        Delete BM25 index for a document

        Args:
            document_id: Document UUID string
        """
        try:
            # Remove from memory cache
            if document_id in self.bm25_cache:
                del self.bm25_cache[document_id]
            if document_id in self.corpus_cache:
                del self.corpus_cache[document_id]

            # Remove from disk
            cache_path = self._get_cache_path(document_id)
            if cache_path.exists():
                os.remove(cache_path)
                logger.info(f"✓ Deleted BM25 index for document {document_id}")

        except Exception as e:
            logger.error(f"Error deleting BM25 index: {e}")
