"""
Neo4j Service for Graph Storage and Retrieval
Stores GraphRAG extracted entities and relationships in Neo4j
"""
import logging
from typing import List, Dict, Any, Optional
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
        document_id: str,
        pass_number: int = 1
    ):
        """
        Import graph documents into Neo4j using MERGE for deduplication

        Args:
            graph_documents: List of GraphDocument objects from LLMGraphTransformer
            document_id: Document identifier
            pass_number: Which pass this is (1, 2, or 3) for multi-pass enrichment
        """
        try:
            logger.info(f"Importing {len(graph_documents)} graph documents for document {document_id} (Pass {pass_number})")

            # Create document node first
            self.graph.query(
                """
                MERGE (d:__Document__ {id: $document_id})
                SET d.imported_at = datetime(),
                    d.last_updated_pass = $pass_number
                """,
                params={"document_id": document_id, "pass_number": pass_number}
            )

            # Import graph documents using LangChain's add_graph_documents
            # This uses MERGE internally to avoid duplicates
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            # Link entities to document if not already linked
            self.graph.query(
                """
                MATCH (d:__Document__ {id: $document_id})
                MATCH (e:__Entity__)
                WHERE NOT (e)-[:BELONGS_TO]->(:__Document__)
                MERGE (e)-[:BELONGS_TO]->(d)
                SET e.discovery_pass = COALESCE(e.discovery_pass, $pass_number)
                """,
                params={"document_id": document_id, "pass_number": pass_number}
            )

            # Update entity metadata for tracking enrichment passes
            self.graph.query(
                """
                MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                SET e.last_enriched_pass = $pass_number
                """,
                params={"document_id": document_id, "pass_number": pass_number}
            )

            logger.info(f"Graph documents imported successfully for {document_id} (Pass {pass_number})")

        except Exception as e:
            logger.error(f"Error importing graph documents: {e}")
            raise

    async def merge_similar_entities(
        self,
        document_id: str,
        similarity_threshold: float = 0.85
    ):
        """
        Merge entities that are likely duplicates based on name similarity

        This helps consolidate entities that might have been extracted with
        slightly different names across multiple passes

        Args:
            document_id: Document identifier
            similarity_threshold: Minimum similarity score to consider merging (0.0-1.0)
        """
        try:
            logger.info(f"Merging similar entities for document {document_id}")

            # Find potential duplicates using Levenshtein distance or simple matching
            cypher = """
            MATCH (e1:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
            MATCH (e2:__Entity__)-[:BELONGS_TO]->(d)
            WHERE e1.id < e2.id
            AND (
                toLower(e1.id) = toLower(e2.id)
                OR e1.id CONTAINS e2.id
                OR e2.id CONTAINS e1.id
            )
            WITH e1, e2
            // Merge relationships from e2 to e1
            OPTIONAL MATCH (e2)-[r]-(other:__Entity__)
            WHERE other <> e1
            MERGE (e1)-[r2:rel_type(r)]-(other)
            ON CREATE SET r2 = properties(r)
            // Delete e2 and its relationships
            WITH e1, e2
            DETACH DELETE e2
            RETURN count(e2) as merged_count
            """

            result = self.graph.query(cypher, params={
                "document_id": document_id
            })

            merged_count = result[0]["merged_count"] if result else 0
            logger.info(f"Merged {merged_count} duplicate entities for document {document_id}")

            return merged_count

        except Exception as e:
            logger.error(f"Error merging similar entities: {e}")
            return 0

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

    # ===== Graph Update Operations =====

    async def create_node(
        self,
        node_id: str,
        node_type: str = "Entity",
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new node in the graph

        Args:
            node_id: Node identifier/name
            node_type: Type of the node (default: Entity)
            description: Node description
            properties: Additional properties for the node
            document_id: Optional document filter

        Returns:
            Dict with creation status
        """
        try:
            logger.info(f"Creating new node '{node_id}' of type '{node_type}'")

            # Check if node already exists
            check_cypher = """
            MATCH (e:__Entity__ {id: $node_id})
            RETURN e.id as existing_id
            LIMIT 1
            """
            existing = self.graph.query(check_cypher, params={"node_id": node_id})

            if existing:
                return {
                    "success": False,
                    "error": f"Node '{node_id}' already exists"
                }

            # Create the new node
            all_properties = properties or {}
            all_properties["id"] = node_id
            all_properties["description"] = description
            all_properties["type"] = node_type

            cypher = """
            CREATE (e:__Entity__)
            SET e = $properties
            """

            # Link to document if provided
            if document_id:
                cypher += """
                WITH e
                MATCH (d:__Document__ {id: $document_id})
                MERGE (e)-[:BELONGS_TO]->(d)
                """

            cypher += """
            RETURN e.id as node_id, properties(e) as properties
            """

            params = {"properties": all_properties}
            if document_id:
                params["document_id"] = document_id

            result = self.graph.query(cypher, params=params)

            if not result:
                return {
                    "success": False,
                    "error": f"Failed to create node '{node_id}'"
                }

            logger.info(f"Successfully created node '{node_id}'")
            return {
                "success": True,
                "node_id": result[0]["node_id"],
                "properties": result[0]["properties"]
            }

        except Exception as e:
            logger.error(f"Error creating node: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_node_with_relationships(
        self,
        node_id: str,
        node_type: str = "Entity",
        description: str = "",
        relationships: List[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new node and connect it to multiple existing nodes in one operation

        Args:
            node_id: Node identifier/name
            node_type: Type of the node
            description: Node description
            relationships: List of dicts with keys: target_id, relationship_type, properties (optional), direction (optional: 'outgoing' or 'incoming')
            document_id: Optional document filter

        Returns:
            Dict with creation status and relationship counts
        """
        try:
            logger.info(f"Creating node '{node_id}' with {len(relationships or [])} relationships")

            # First create the node
            create_result = await self.create_node(
                node_id=node_id,
                node_type=node_type,
                description=description,
                document_id=document_id
            )

            if not create_result.get("success"):
                return create_result

            # Now create all the relationships
            relationships = relationships or []
            successful_rels = []
            failed_rels = []

            for rel in relationships:
                target_id = rel.get("target_id")
                rel_type = rel.get("relationship_type", "RELATED_TO")
                rel_props = rel.get("properties", {})
                direction = rel.get("direction", "outgoing")  # outgoing: source->target, incoming: target->source

                if direction == "outgoing":
                    result = await self.create_relationship(
                        source_id=node_id,
                        target_id=target_id,
                        relationship_type=rel_type,
                        properties=rel_props,
                        document_id=document_id
                    )
                else:  # incoming
                    result = await self.create_relationship(
                        source_id=target_id,
                        target_id=node_id,
                        relationship_type=rel_type,
                        properties=rel_props,
                        document_id=document_id
                    )

                if result.get("success"):
                    successful_rels.append({
                        "target": target_id,
                        "type": rel_type,
                        "direction": direction
                    })
                else:
                    failed_rels.append({
                        "target": target_id,
                        "error": result.get("error"),
                        "direction": direction
                    })

            logger.info(f"Created node '{node_id}' with {len(successful_rels)} successful relationships, {len(failed_rels)} failed")

            return {
                "success": True,
                "node_id": node_id,
                "node_created": True,
                "relationships_created": len(successful_rels),
                "relationships_failed": len(failed_rels),
                "successful_relationships": successful_rels,
                "failed_relationships": failed_rels if failed_rels else None
            }

        except Exception as e:
            logger.error(f"Error creating node with relationships: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_node_property(
        self,
        node_id: str,
        property_name: str,
        property_value: Any,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update a specific property of a node

        Args:
            node_id: Node identifier (partial match supported)
            property_name: Name of the property to update
            property_value: New value for the property
            document_id: Optional document filter

        Returns:
            Dict with update status and updated node info
        """
        try:
            logger.info(f"Updating property '{property_name}' of node '{node_id}' to '{property_value}'")

            cypher = """
            MATCH (e:__Entity__)
            WHERE toLower(e.id) CONTAINS toLower($node_id)
            """

            if document_id:
                cypher += """
                AND (e)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += f"""
            SET e.{property_name} = $property_value
            RETURN e.id as node_id, e.{property_name} as updated_value, properties(e) as all_properties
            LIMIT 1
            """

            params = {
                "node_id": node_id,
                "property_value": property_value
            }
            if document_id:
                params["document_id"] = document_id

            result = self.graph.query(cypher, params=params)

            if not result:
                return {
                    "success": False,
                    "error": f"Node '{node_id}' not found"
                }

            logger.info(f"Successfully updated property '{property_name}' of node '{result[0]['node_id']}'")
            return {
                "success": True,
                "node_id": result[0]["node_id"],
                "property_name": property_name,
                "new_value": result[0]["updated_value"],
                "all_properties": result[0]["all_properties"]
            }

        except Exception as e:
            logger.error(f"Error updating node property: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_node_description(
        self,
        node_id: str,
        description: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the description of a node

        Args:
            node_id: Node identifier (partial match supported)
            description: New description
            document_id: Optional document filter

        Returns:
            Dict with update status
        """
        return await self.update_node_property(node_id, "description", description, document_id)

    async def delete_node(
        self,
        node_id: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a node and all its relationships

        Args:
            node_id: Node identifier (partial match supported)
            document_id: Optional document filter

        Returns:
            Dict with deletion status
        """
        try:
            logger.info(f"Deleting node '{node_id}'")

            # First get node info before deletion
            cypher_get = """
            MATCH (e:__Entity__)
            WHERE toLower(e.id) CONTAINS toLower($node_id)
            """

            if document_id:
                cypher_get += """
                AND (e)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher_get += """
            RETURN e.id as node_id, e.type as node_type, e.description as description
            LIMIT 1
            """

            params = {"node_id": node_id}
            if document_id:
                params["document_id"] = document_id

            node_info = self.graph.query(cypher_get, params=params)

            if not node_info:
                return {
                    "success": False,
                    "error": f"Node '{node_id}' not found"
                }

            # Delete the node and all its relationships
            cypher_delete = """
            MATCH (e:__Entity__)
            WHERE toLower(e.id) CONTAINS toLower($node_id)
            """

            if document_id:
                cypher_delete += """
                AND (e)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher_delete += """
            DETACH DELETE e
            RETURN count(e) as deleted_count
            """

            result = self.graph.query(cypher_delete, params=params)

            logger.info(f"Successfully deleted node '{node_info[0]['node_id']}'")
            return {
                "success": True,
                "deleted_node": node_info[0],
                "deleted_count": result[0]["deleted_count"] if result else 0
            }

        except Exception as e:
            logger.error(f"Error deleting node: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def merge_nodes(
        self,
        node_id1: str,
        node_id2: str,
        new_node_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge two nodes into one, combining their relationships

        Args:
            node_id1: First node identifier (will be kept)
            node_id2: Second node identifier (will be deleted)
            new_node_id: Optional new ID for merged node (defaults to node_id1)
            document_id: Optional document filter

        Returns:
            Dict with merge status
        """
        try:
            logger.info(f"Merging nodes '{node_id1}' and '{node_id2}'")

            # Get both nodes
            cypher = """
            MATCH (e1:__Entity__)
            WHERE toLower(e1.id) CONTAINS toLower($node_id1)
            """

            if document_id:
                cypher += """
                AND (e1)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """
            MATCH (e2:__Entity__)
            WHERE toLower(e2.id) CONTAINS toLower($node_id2)
            """

            if document_id:
                cypher += """
                AND (e2)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """
            // Merge relationships from e2 to e1
            WITH e1, e2
            OPTIONAL MATCH (e2)-[r]-(other:__Entity__)
            WHERE other <> e1 AND NOT type(r) = 'BELONGS_TO'
            WITH e1, e2, type(r) as rel_type, other, properties(r) as rel_props
            WHERE rel_type IS NOT NULL
            FOREACH (_ IN CASE WHEN rel_type IS NOT NULL THEN [1] ELSE [] END |
                MERGE (e1)-[new_r:MERGED {type: rel_type}]-(other)
                SET new_r = rel_props
            )
            WITH e1, e2
            // Update e1 ID if new_node_id provided
            SET e1.id = CASE WHEN $new_node_id IS NOT NULL THEN $new_node_id ELSE e1.id END,
                e1.description = COALESCE(e1.description, '') + ' | Merged with: ' + COALESCE(e2.description, e2.id)
            // Delete e2
            DETACH DELETE e2
            RETURN e1.id as merged_node_id, properties(e1) as merged_properties
            """

            params = {
                "node_id1": node_id1,
                "node_id2": node_id2,
                "new_node_id": new_node_id
            }
            if document_id:
                params["document_id"] = document_id

            result = self.graph.query(cypher, params=params)

            if not result:
                return {
                    "success": False,
                    "error": f"Could not merge nodes '{node_id1}' and '{node_id2}'"
                }

            logger.info(f"Successfully merged nodes into '{result[0]['merged_node_id']}'")
            return {
                "success": True,
                "merged_node_id": result[0]["merged_node_id"],
                "merged_properties": result[0]["merged_properties"]
            }

        except Exception as e:
            logger.error(f"Error merging nodes: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new relationship between two nodes

        Args:
            source_id: Source node identifier (partial match)
            target_id: Target node identifier (partial match)
            relationship_type: Type of relationship
            properties: Optional properties for the relationship
            document_id: Optional document filter

        Returns:
            Dict with creation status
        """
        try:
            logger.info(f"Creating relationship '{relationship_type}' from '{source_id}' to '{target_id}'")

            cypher = """
            MATCH (source:__Entity__)
            WHERE toLower(source.id) CONTAINS toLower($source_id)
            """

            if document_id:
                cypher += """
                AND (source)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """
            MATCH (target:__Entity__)
            WHERE toLower(target.id) CONTAINS toLower($target_id)
            """

            if document_id:
                cypher += """
                AND (target)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            # Use dynamic relationship type (Neo4j requires special handling)
            cypher += f"""
            MERGE (source)-[r:{relationship_type}]->(target)
            SET r = $properties
            RETURN source.id as source_id, target.id as target_id, type(r) as rel_type, properties(r) as rel_properties
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "properties": properties or {}
            }
            if document_id:
                params["document_id"] = document_id

            result = self.graph.query(cypher, params=params)

            if not result:
                return {
                    "success": False,
                    "error": f"Could not create relationship. Source or target node not found."
                }

            logger.info(f"Successfully created relationship '{relationship_type}'")
            return {
                "success": True,
                "source_id": result[0]["source_id"],
                "target_id": result[0]["target_id"],
                "relationship_type": result[0]["rel_type"],
                "properties": result[0]["rel_properties"]
            }

        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update properties of an existing relationship

        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: New properties for the relationship
            document_id: Optional document filter

        Returns:
            Dict with update status
        """
        try:
            logger.info(f"Updating relationship '{relationship_type}' from '{source_id}' to '{target_id}'")

            cypher = """
            MATCH (source:__Entity__)-[r]-(target:__Entity__)
            WHERE toLower(source.id) CONTAINS toLower($source_id)
            AND toLower(target.id) CONTAINS toLower($target_id)
            AND (type(r) = $relationship_type OR toLower(type(r)) = toLower($relationship_type))
            """

            if document_id:
                cypher += """
                AND (source)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """
            SET r += $properties
            RETURN source.id as source_id, target.id as target_id, type(r) as rel_type, properties(r) as rel_properties
            LIMIT 1
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "properties": properties
            }
            if document_id:
                params["document_id"] = document_id

            result = self.graph.query(cypher, params=params)

            if not result:
                return {
                    "success": False,
                    "error": f"Relationship '{relationship_type}' between '{source_id}' and '{target_id}' not found"
                }

            logger.info(f"Successfully updated relationship '{relationship_type}'")
            return {
                "success": True,
                "source_id": result[0]["source_id"],
                "target_id": result[0]["target_id"],
                "relationship_type": result[0]["rel_type"],
                "updated_properties": result[0]["rel_properties"]
            }

        except Exception as e:
            logger.error(f"Error updating relationship: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a relationship between two nodes

        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Optional specific relationship type (deletes all if not specified)
            document_id: Optional document filter

        Returns:
            Dict with deletion status
        """
        try:
            logger.info(f"Deleting relationship from '{source_id}' to '{target_id}'")

            # First get relationship info
            cypher_get = """
            MATCH (source:__Entity__)-[r]-(target:__Entity__)
            WHERE toLower(source.id) CONTAINS toLower($source_id)
            AND toLower(target.id) CONTAINS toLower($target_id)
            AND NOT type(r) = 'BELONGS_TO'
            """

            if relationship_type:
                cypher_get += """
                AND (type(r) = $relationship_type OR toLower(type(r)) = toLower($relationship_type))
                """

            if document_id:
                cypher_get += """
                AND (source)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher_get += """
            RETURN source.id as source_id, target.id as target_id, type(r) as rel_type, properties(r) as rel_properties
            """

            params = {
                "source_id": source_id,
                "target_id": target_id
            }
            if relationship_type:
                params["relationship_type"] = relationship_type
            if document_id:
                params["document_id"] = document_id

            rel_info = self.graph.query(cypher_get, params=params)

            if not rel_info:
                return {
                    "success": False,
                    "error": f"No relationship found between '{source_id}' and '{target_id}'"
                }

            # Delete the relationship
            cypher_delete = """
            MATCH (source:__Entity__)-[r]-(target:__Entity__)
            WHERE toLower(source.id) CONTAINS toLower($source_id)
            AND toLower(target.id) CONTAINS toLower($target_id)
            AND NOT type(r) = 'BELONGS_TO'
            """

            if relationship_type:
                cypher_delete += """
                AND (type(r) = $relationship_type OR toLower(type(r)) = toLower($relationship_type))
                """

            if document_id:
                cypher_delete += """
                AND (source)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher_delete += """
            DELETE r
            RETURN count(r) as deleted_count
            """

            result = self.graph.query(cypher_delete, params=params)

            logger.info(f"Successfully deleted {result[0]['deleted_count']} relationship(s)")
            return {
                "success": True,
                "deleted_relationships": rel_info,
                "deleted_count": result[0]["deleted_count"] if result else 0
            }

        except Exception as e:
            logger.error(f"Error deleting relationship: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global Neo4j service instance
neo4j_service = Neo4jService()
