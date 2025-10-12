"""
Elasticsearch Service for Filter Search and Full-Text Search
Handles document indexing, metadata filtering, and hybrid search capabilities
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, ConnectionError as ESConnectionError

from app.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchService:
    """Service for Elasticsearch indexing and search operations"""

    def __init__(self):
        """Initialize Elasticsearch client"""
        self.es_url = settings.elasticsearch_url
        self.index_name = settings.elasticsearch_index_name
        self.client: Optional[Elasticsearch] = None
        self.async_client: Optional[AsyncElasticsearch] = None

        logger.info(f"ElasticsearchService initialized with URL: {self.es_url}")

    def _get_client(self) -> Elasticsearch:
        """Get or create synchronous Elasticsearch client"""
        if self.client is None:
            try:
                self.client = Elasticsearch(
                    [self.es_url],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                # Test connection
                if self.client.ping():
                    logger.info("Successfully connected to Elasticsearch")
                else:
                    logger.error("Failed to ping Elasticsearch")
            except Exception as e:
                logger.error(f"Error connecting to Elasticsearch: {e}")
                raise
        return self.client

    async def _get_async_client(self) -> AsyncElasticsearch:
        """Get or create async Elasticsearch client"""
        if self.async_client is None:
            try:
                self.async_client = AsyncElasticsearch(
                    [self.es_url],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                # Test connection
                if await self.async_client.ping():
                    logger.info("Successfully connected to Elasticsearch (async)")
                else:
                    logger.error("Failed to ping Elasticsearch (async)")
            except Exception as e:
                logger.error(f"Error connecting to Elasticsearch (async): {e}")
                raise
        return self.async_client

    def create_index(self) -> bool:
        """
        Create Elasticsearch index with proper mappings

        Returns:
            bool: True if successful
        """
        try:
            client = self._get_client()

            # Check if index already exists
            if client.indices.exists(index=self.index_name):
                logger.info(f"Index '{self.index_name}' already exists")
                return True

            # Define index mappings
            mappings = {
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "filename": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                            "term_vector": "with_positions_offsets"
                        },
                        "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "document_type": {"type": "keyword"},
                        "categories": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "uploaded_at": {"type": "date"},
                        "processed_at": {"type": "date"},
                        "indexed_at": {"type": "date"},
                        "file_size": {"type": "long"},
                        "chunk_count": {"type": "integer"},
                        "user_id": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": False}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    }
                }
            }

            # Create index
            client.indices.create(index=self.index_name, body=mappings)
            logger.info(f"Created Elasticsearch index: {self.index_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    async def index_document(
        self,
        document_id: str,
        content: str,
        filename: str,
        author: Optional[str] = None,
        document_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        uploaded_at: Optional[datetime] = None,
        processed_at: Optional[datetime] = None,
        file_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Index a document in Elasticsearch

        Args:
            document_id: Unique document identifier
            content: Full text content of the document
            filename: Original filename
            author: Document author (optional)
            document_type: Type of document (pdf, txt, etc.)
            categories: List of categories
            tags: List of tags
            uploaded_at: Upload timestamp
            processed_at: Processing timestamp
            file_size: File size in bytes
            chunk_count: Number of chunks
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            bool: True if successful
        """
        try:
            client = await self._get_async_client()

            # Prepare document
            doc = {
                "document_id": document_id,
                "filename": filename,
                "content": content,
                "indexed_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }

            # Add optional fields
            if author:
                doc["author"] = author
            if document_type:
                doc["document_type"] = document_type
            if categories:
                doc["categories"] = categories
            if tags:
                doc["tags"] = tags
            if uploaded_at:
                doc["uploaded_at"] = uploaded_at.isoformat()
            if processed_at:
                doc["processed_at"] = processed_at.isoformat()
            if file_size:
                doc["file_size"] = file_size
            if chunk_count:
                doc["chunk_count"] = chunk_count
            if metadata:
                doc["metadata"] = metadata

            # Index document
            await client.index(
                index=self.index_name,
                id=document_id,
                document=doc
            )

            # Refresh index to make document immediately searchable
            await client.indices.refresh(index=self.index_name)

            logger.info(f"Indexed document: {document_id} ({filename})")
            return True

        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0
    ) -> Dict[str, Any]:
        """
        Search documents with optional filters

        Args:
            query: Search query (full-text)
            filters: Dictionary of filters (author, categories, tags, date_range, etc.)
            size: Number of results to return
            from_: Offset for pagination

        Returns:
            Dictionary with search results
        """
        try:
            client = await self._get_async_client()

            # Build query
            must_clauses = []
            filter_clauses = []

            # Full-text search on content
            if query:
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "filename", "author"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                })

            # Apply filters
            if filters:
                # Author filter
                if "author" in filters and filters["author"]:
                    filter_clauses.append({
                        "match": {"author": filters["author"]}
                    })

                # Document type filter
                if "document_type" in filters and filters["document_type"]:
                    filter_clauses.append({
                        "term": {"document_type": filters["document_type"]}
                    })

                # Categories filter
                if "categories" in filters and filters["categories"]:
                    filter_clauses.append({
                        "terms": {"categories": filters["categories"]}
                    })

                # Tags filter
                if "tags" in filters and filters["tags"]:
                    filter_clauses.append({
                        "terms": {"tags": filters["tags"]}
                    })

                # Date range filter
                if "date_from" in filters or "date_to" in filters:
                    date_range = {}
                    if "date_from" in filters:
                        date_range["gte"] = filters["date_from"]
                    if "date_to" in filters:
                        date_range["lte"] = filters["date_to"]
                    filter_clauses.append({
                        "range": {"uploaded_at": date_range}
                    })

                # User filter
                if "user_id" in filters and filters["user_id"]:
                    filter_clauses.append({
                        "term": {"user_id": filters["user_id"]}
                    })

            # Build final query
            search_query = {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}],
                    "filter": filter_clauses
                }
            }

            # Execute search
            response = await client.search(
                index=self.index_name,
                query=search_query,
                size=size,
                from_=from_,
                highlight={
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                }
            )

            # Format results
            results = {
                "total": response["hits"]["total"]["value"],
                "documents": []
            }

            for hit in response["hits"]["hits"]:
                doc = {
                    "document_id": hit["_source"]["document_id"],
                    "filename": hit["_source"]["filename"],
                    "score": hit["_score"],
                    "content_preview": hit["_source"]["content"][:500],
                    "author": hit["_source"].get("author"),
                    "document_type": hit["_source"].get("document_type"),
                    "categories": hit["_source"].get("categories", []),
                    "tags": hit["_source"].get("tags", []),
                    "uploaded_at": hit["_source"].get("uploaded_at"),
                    "highlights": hit.get("highlight", {}).get("content", [])
                }
                results["documents"].append(doc)

            logger.info(f"Search returned {len(results['documents'])} results")
            return results

        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}")
            return {"total": 0, "documents": [], "error": str(e)}

    async def get_aggregations(self, field: str) -> Dict[str, Any]:
        """
        Get aggregations (facets) for a field

        Args:
            field: Field to aggregate (e.g., 'author', 'categories', 'tags')

        Returns:
            Dictionary with aggregation results
        """
        try:
            client = await self._get_async_client()

            response = await client.search(
                index=self.index_name,
                size=0,
                aggs={
                    f"{field}_agg": {
                        "terms": {
                            "field": f"{field}.keyword" if field in ["author", "filename"] else field,
                            "size": 50
                        }
                    }
                }
            )

            buckets = response["aggregations"][f"{field}_agg"]["buckets"]
            return {
                "field": field,
                "values": [
                    {"value": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in buckets
                ]
            }

        except Exception as e:
            logger.error(f"Error getting aggregations: {e}")
            return {"field": field, "values": [], "error": str(e)}

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the index

        Args:
            document_id: Document identifier

        Returns:
            bool: True if successful
        """
        try:
            client = await self._get_async_client()
            await client.delete(index=self.index_name, id=document_id)
            logger.info(f"Deleted document: {document_id}")
            return True

        except NotFoundError:
            logger.warning(f"Document not found for deletion: {document_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def close(self):
        """Close Elasticsearch connections"""
        try:
            if self.client:
                self.client.close()
            if self.async_client:
                await self.async_client.close()
            logger.info("Elasticsearch connections closed")
        except Exception as e:
            logger.error(f"Error closing Elasticsearch connections: {e}")


# Global Elasticsearch service instance
elasticsearch_service = ElasticsearchService()
