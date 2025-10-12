"""
Neo4j Service for Graph Storage and Retrieval
Stores GraphRAG extracted entities and relationships in Neo4j
"""
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument

from app.config import settings

logger = logging.getLogger(__name__)

# Suppress Neo4j client notification warnings about unknown property keys
# These are harmless warnings when properties exist but aren't in Neo4j's schema
logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)


class Neo4jService:
    """Service for interacting with Neo4j graph database"""

    def __init__(self):
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password

        self._driver = None
        self._graph = None

        logger.info(f"Neo4j Service configured for: {self.uri}")

    @property
    def driver(self):
        """Lazy initialization of Neo4j driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            logger.info(f"Neo4j driver connected: {self.uri}")
        return self._driver

    @property
    def graph(self):
        """Lazy initialization of LangChain Neo4j Graph wrapper"""
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password
            )
            logger.info(f"Neo4j graph wrapper initialized")
        return self._graph

    def close(self):
        """Close Neo4j driver"""
        if self._driver:
            self._driver.close()

    async def create_constraints(self):
        """Create constraints and indexes for optimal performance"""
        try:
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:__Document__) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    self.graph.query(constraint)
                    logger.info(f"Constraint created: {constraint.split('CONSTRAINT')[1].split('IF')[0].strip()}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

            # Create indexes for better query performance and to suppress warnings
            indexes = [
                "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:__Entity__) ON (e.id)",
                "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:__Document__) ON (d.id)",
                "CREATE INDEX entity_description_index IF NOT EXISTS FOR (e:__Entity__) ON (e.description)",
            ]

            for index in indexes:
                try:
                    self.graph.query(index)
                    logger.info(f"Index created: {index.split('INDEX')[1].split('IF')[0].strip()}")
                except Exception as e:
                    # Indexes may already exist from constraints
                    if "equivalent" in str(e).lower() or "already exists" in str(e).lower():
                        logger.debug(f"Index already exists (possibly from constraint): {e}")
                    else:
                        logger.warning(f"Index creation note: {e}")

            # Create vector index for entity descriptions (optional, requires Neo4j 5.13+)
            try:
                self.graph.query("""
                    CREATE VECTOR INDEX entity_description_embeddings IF NOT EXISTS
                    FOR (e:__Entity__) ON (e.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("✓ Vector index created for entity embeddings")
            except Exception as e:
                # Vector indexes require Neo4j 5.13+, gracefully skip if not supported
                if "Invalid input 'VECTOR'" in str(e) or "SyntaxError" in str(e):
                    logger.info("ℹ️ Vector indexes not supported (requires Neo4j 5.13+), skipping...")
                else:
                    logger.warning(f"Vector index creation skipped: {e}")

        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
            raise

    async def import_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        document_id: str
    ):
        """
        Import graph documents into Neo4j using LangChain

        Args:
            graph_documents: List of GraphDocument objects from LLMGraphTransformer
            document_id: Document identifier
        """
        try:
            logger.info(f"Importing {len(graph_documents)} graph documents for document {document_id}")

            # Create document node first
            self.graph.query(
                """
                MERGE (d:__Document__ {id: $document_id})
                SET d.imported_at = datetime()
                """,
                params={"document_id": document_id}
            )

            # Import graph documents using LangChain's add_graph_documents
            # This handles nodes, relationships, and metadata
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            # Link entities to document
            self.graph.query(
                """
                MATCH (d:__Document__ {id: $document_id})
                MATCH (e:__Entity__)
                WHERE NOT (e)-[:BELONGS_TO]->(:__Document__)
                MERGE (e)-[:BELONGS_TO]->(d)
                """,
                params={"document_id": document_id}
            )

            logger.info(f"Graph documents imported successfully for {document_id}")

        except Exception as e:
            logger.error(f"Error importing graph documents: {e}")
            raise

    async def get_entity_count(self, document_id: str = None) -> int:
        """Get count of entities, optionally filtered by document"""
        try:
            if document_id:
                result = self.graph.query(
                    """
                    MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                    RETURN count(e) as count
                    """,
                    params={"document_id": document_id}
                )
            else:
                result = self.graph.query("MATCH (e:__Entity__) RETURN count(e) as count")

            return result[0]["count"] if result else 0

        except Exception as e:
            logger.error(f"Error getting entity count: {e}")
            return 0

    async def get_relationship_count(self, document_id: str = None) -> int:
        """Get count of relationships, optionally filtered by document"""
        try:
            if document_id:
                result = self.graph.query(
                    """
                    MATCH (e1:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                    MATCH (e1)-[r]-(e2:__Entity__)
                    WHERE e1.id < e2.id
                    RETURN count(r) as count
                    """,
                    params={"document_id": document_id}
                )
            else:
                result = self.graph.query(
                    """
                    MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
                    WHERE NOT type(r) = 'BELONGS_TO'
                    RETURN count(r) as count
                    """
                )

            return result[0]["count"] if result else 0

        except Exception as e:
            logger.error(f"Error getting relationship count: {e}")
            return 0

    async def delete_document_graph(self, document_id: str):
        """Delete all graph data for a specific document"""
        try:
            logger.info(f"Deleting graph data for document {document_id}")

            # Delete entities and their relationships
            self.graph.query(
                """
                MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                DETACH DELETE e
                """,
                params={"document_id": document_id}
            )

            # Delete document node
            self.graph.query(
                """
                MATCH (d:__Document__ {id: $document_id})
                DELETE d
                """,
                params={"document_id": document_id}
            )

            logger.info(f"Graph data deleted for document {document_id}")

        except Exception as e:
            logger.error(f"Error deleting document graph: {e}")
            raise

    async def clear_all_graph_data(self):
        """Clear all graph data from Neo4j - useful for starting fresh"""
        try:
            logger.info("Clearing all graph data from Neo4j")

            # Delete all entities, relationships, and document nodes
            self.graph.query(
                """
                MATCH (n)
                DETACH DELETE n
                """
            )

            logger.info("All graph data cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing all graph data: {e}")
            raise

    async def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """Get statistics for a document's graph"""
        try:
            entities_count = await self.get_entity_count(document_id)
            relationships_count = await self.get_relationship_count(document_id)

            return {
                "document_id": document_id,
                "entities_count": entities_count,
                "relationships_count": relationships_count
            }

        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {
                "document_id": document_id,
                "entities_count": 0,
                "relationships_count": 0
            }

    async def query_graph(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query on the graph"""
        try:
            result = self.graph.query(cypher_query, params=params or {})
            return result

        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise


# Global Neo4j service instance
neo4j_service = Neo4jService()
