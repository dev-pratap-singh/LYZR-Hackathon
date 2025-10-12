"""
Vector Embedding Service using OpenAI
Generates embeddings for text chunks
"""
import logging
from typing import List
import asyncio

from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating vector embeddings using OpenAI"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.embedding_dimension = 1536  # for text-embedding-3-small

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []

            # Process in batches to avoid rate limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Small delay to avoid rate limiting
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return await self.generate_embedding(query)
