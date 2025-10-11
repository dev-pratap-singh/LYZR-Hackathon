"""
Reranker Service
Uses cross-encoder models to rerank search results
"""
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    """Service for reranking search results using cross-encoder models"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model

        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast and accurate)
        """
        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name)
            logger.info(f"✓ Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank search results using cross-encoder

        Args:
            query: User query
            results: List of search results (from vector + BM25)
            top_k: Number of top results to return

        Returns:
            Reranked list of results with rerank_score
        """
        try:
            if not self.model:
                logger.warning("Reranker model not available, returning original results")
                return results[:top_k]

            if not results:
                return []

            # Prepare query-document pairs
            pairs = [(query, result['chunk_text']) for result in results]

            # Get reranking scores
            scores = self.model.predict(pairs)

            # Add rerank scores to results
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

            logger.info(f"✓ Reranked {len(results)} results, returning top {top_k}")

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original results if reranking fails
            return results[:top_k]

    def rerank_with_scores(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> tuple[List[Dict], List[float]]:
        """
        Rerank and return both results and scores separately

        Args:
            query: User query
            results: List of search results
            top_k: Number of top results

        Returns:
            Tuple of (reranked_results, rerank_scores)
        """
        reranked = self.rerank(query, results, top_k)
        scores = [r.get('rerank_score', 0.0) for r in reranked]
        return reranked, scores
