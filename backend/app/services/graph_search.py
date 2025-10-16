"""
Graph Search Service using Neo4j for relationship queries
Implements Cypher query generation for knowledge graph retrieval with multi-hop traversal
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.config import settings
from app.services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)


class GraphSearchService:
    """Service for searching relationships and entities in the knowledge graph with intelligent traversal"""

    def __init__(self):
        self.neo4j = neo4j_service
        self._llm = None  # Lazy initialization

        logger.info("GraphSearchService initialized with enhanced traversal capabilities")

    @property
    def llm(self):
        """Lazy initialization of LLM client"""
        if self._llm is None:
            if not settings.openai_api_key:
                error_msg = (
                    "⚠️  OPENAI_API_KEY is not configured. "
                    "AI-powered graph search features require an OpenAI API key. "
                    "Please set the OPENAI_API_KEY environment variable or add your API key via the settings."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Initializing OpenAI client for graph search")
            self._llm = ChatOpenAI(
                model=settings.graphrag_llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
        return self._llm

    async def generate_cypher_query(self, natural_language_query: str) -> str:
        """
        Generate Cypher query from natural language using LLM

        Args:
            natural_language_query: User's question about relationships

        Returns:
            Cypher query string
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Cypher query expert. Convert natural language questions into Cypher queries for Neo4j.

The graph schema:
- Nodes: __Entity__ (properties: id, type, description)
- Relationships: Dynamic types like MENTIONS, EMPLOYED_AT, EDUCATED_AT, LOCATED_IN, etc. (properties may include description)
- Nodes are also connected to __Document__ via BELONGS_TO relationship

Guidelines:
1. Use MATCH clauses to find entities and relationships
2. Use [r] to match ANY relationship type (relationships have dynamic types)
3. Use WHERE clauses for filtering
4. Return relevant entities, relationships, and their descriptions
5. Limit results to top 10 by default
6. For relationship queries, find paths between entities
7. Include entity descriptions and relationship types in results

Example queries:
- "How are X and Y related?" -> MATCH path = (e1:__Entity__)-[r*1..3]-(e2:__Entity__) WHERE e1.id CONTAINS 'X' AND e2.id CONTAINS 'Y' RETURN path LIMIT 10
- "What connects A to B?" -> MATCH path = shortestPath((e1:__Entity__)-[*]-(e2:__Entity__)) WHERE e1.id CONTAINS 'A' AND e2.id CONTAINS 'B' RETURN path
- "Tell me about entity X" -> MATCH (e:__Entity__)-[r]-(n) WHERE e.id CONTAINS 'X' RETURN e, collect({{relation: type(r), entity: n.id}}) as relationships

Return ONLY the Cypher query, no explanations."""),
                ("human", "{query}")
            ])

            response = await self.llm.ainvoke(
                prompt.format_messages(query=natural_language_query)
            )

            cypher_query = response.content.strip()

            # Clean up the query (remove markdown code blocks if present)
            if cypher_query.startswith("```"):
                lines = cypher_query.split("\n")
                cypher_query = "\n".join(lines[1:-1])

            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query

        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            raise

    async def get_entity_with_context(
        self,
        entity_id: str,
        document_id: Optional[str] = None,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about an entity including its connections

        This performs intelligent graph traversal to understand:
        - The entity itself
        - Direct connections (1-hop)
        - Secondary connections (2-hop) for context
        - Relationship types and their descriptions

        Args:
            entity_id: Entity identifier (can be partial match)
            document_id: Optional document filter
            max_hops: Maximum traversal depth (default: 2)

        Returns:
            Dict with entity info, connections, and graph context
        """
        try:
            logger.info(f"Fetching entity context for: {entity_id}")

            # Query to get entity with multi-hop traversal
            # Use case-insensitive matching for better entity discovery
            cypher = """
            // Find the main entity (case-insensitive)
            MATCH (entity:__Entity__)
            WHERE toLower(entity.id) CONTAINS toLower($entity_id)
            """

            if document_id:
                cypher += """
                AND (entity)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """

            // Get 1-hop neighbors (direct connections)
            OPTIONAL MATCH (entity)-[r1]-(neighbor1:__Entity__)
            WHERE NOT type(r1) = 'BELONGS_TO'

            // Get 2-hop neighbors (extended context)
            OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2:__Entity__)
            WHERE NOT type(r2) = 'BELONGS_TO'
            AND neighbor2.id <> entity.id
            AND neighbor2 <> entity

            // Return entity with its network
            RETURN
                entity.id as entity_id,
                entity.type as entity_type,
                entity.description as entity_description,
                collect(DISTINCT {
                    neighbor_id: neighbor1.id,
                    neighbor_type: neighbor1.type,
                    neighbor_description: neighbor1.description,
                    relationship_type: type(r1),
                    relationship_description: r1.description,
                    direction: CASE
                        WHEN startNode(r1) = entity THEN 'outgoing'
                        ELSE 'incoming'
                    END
                }) as direct_connections,
                collect(DISTINCT {
                    second_hop_id: neighbor2.id,
                    second_hop_type: neighbor2.type,
                    path_through: neighbor1.id,
                    relationship_type: type(r2)
                }) as extended_context
            LIMIT 1
            """

            params = {"entity_id": entity_id}
            if document_id:
                params["document_id"] = document_id

            results = await self.neo4j.query_graph(cypher, params)

            if not results:
                return {"error": f"Entity not found: {entity_id}"}

            return results[0]

        except Exception as e:
            logger.error(f"Error getting entity context: {e}")
            return {"error": str(e)}

    async def search_relationships(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> str:
        """
        Enhanced search for relationships in the knowledge graph with intelligent traversal

        Args:
            query: Natural language query about relationships
            document_id: Optional document ID to filter results

        Returns:
            Formatted search results with comprehensive context
        """
        try:
            logger.info(f"Graph search query: {query}")

            # Check if this is a "tell me about X" or "what is X" query
            is_entity_query = any(phrase in query.lower() for phrase in [
                "tell me about", "what is", "who is", "describe", "explain", "info about"
            ])

            # Also treat short queries (< 100 chars) without question marks as potential entity queries
            # This handles cases where the agent passes just the entity name
            is_short_query = len(query) < 100 and '?' not in query

            # Check if query looks like an entity name (no typical question words)
            query_words = query.lower().split()
            question_words = {'how', 'why', 'when', 'where', 'which', 'relate', 'connect',
                             'relationship', 'find', 'search', 'show', 'list', 'get'}
            has_question_words = any(word in question_words for word in query_words)

            if is_entity_query or (is_short_query and not has_question_words):
                # Extract entity name from query or use query as-is for short queries
                entity_name = self._extract_entity_name(query)
                if not entity_name and is_short_query:
                    # For short queries without extraction keywords, use the query itself as entity name
                    entity_name = query.strip()

                if entity_name:
                    logger.info(f"Detected entity query for: {entity_name}")
                    entity_context = await self.get_entity_with_context(
                        entity_name,
                        document_id,
                        max_hops=2
                    )

                    if "error" not in entity_context:
                        return self._format_entity_context(entity_context, query)

            # Generate Cypher query for relationship searches
            cypher_query = await self.generate_cypher_query(query)

            # Add document filter if provided
            if document_id:
                # Inject document filter into the query
                if "WHERE" in cypher_query:
                    cypher_query = cypher_query.replace(
                        "WHERE",
                        f"MATCH (e1)-[:BELONGS_TO]->(d:__Document__ {{id: '{document_id}'}}) WHERE",
                        1
                    )
                else:
                    cypher_query = cypher_query.replace(
                        "RETURN",
                        f"MATCH (e1)-[:BELONGS_TO]->(d:__Document__ {{id: '{document_id}'}}) RETURN"
                    )

            # Execute query
            results = await self.neo4j.query_graph(cypher_query)

            if not results:
                return f"No relationships found for query: {query}"

            # Format results with enhanced context
            formatted_results = self._format_graph_results(results, query)

            logger.info(f"Graph search returned {len(results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return f"Error searching graph: {str(e)}"

    def _extract_entity_name(self, query: str) -> Optional[str]:
        """Extract entity name from natural language query"""
        try:
            # Simple extraction - look for quoted text or text after "about"
            query_lower = query.lower()

            # Try to find quoted entity
            if '"' in query or "'" in query:
                parts = query.replace("'", '"').split('"')
                if len(parts) >= 3:
                    return parts[1].strip()

            # Try to find entity after "about", "is", etc.
            for keyword in ["about ", "is ", "describe ", "explain ", "find ", "search for "]:
                if keyword in query_lower:
                    idx = query_lower.index(keyword) + len(keyword)
                    remaining = query[idx:].strip()
                    # Take everything before comma, question mark, or "use"/"using"
                    entity = remaining.split("?")[0].split(",")[0].split(" use ")[0].split(" using ")[0].strip()
                    # Clean up common trailing words
                    for stop_word in [" only", " please"]:
                        if entity.lower().endswith(stop_word):
                            entity = entity[:-len(stop_word)].strip()
                    return entity

            return None

        except Exception as e:
            logger.error(f"Error extracting entity name: {e}")
            return None

    def _format_entity_context(self, context: Dict[str, Any], query: str) -> str:
        """Format entity context with comprehensive information"""
        try:
            output = []
            entity_id = context.get('entity_id', 'Unknown')
            entity_type = context.get('entity_type', 'N/A')
            entity_desc = context.get('entity_description', 'No description available')

            output.append(f"## 🔍 Knowledge Graph Analysis: {entity_id}\n")

            # Main entity information
            output.append("### 📌 Entity Profile")
            output.append(f"**Name**: {entity_id}")
            output.append(f"**Type**: {entity_type}")
            output.append(f"**Description**: {entity_desc}\n")

            # Direct connections (1-hop)
            direct_connections = [
                c for c in context.get('direct_connections', [])
                if c.get('neighbor_id')  # Filter out empty connections
            ]

            if direct_connections:
                output.append(f"### 🔗 Direct Connections (1-hop Traversal)")
                output.append(f"Found **{len(direct_connections)}** direct relationships:\n")

                # Group by relationship type
                by_rel_type = {}
                for conn in direct_connections:
                    rel_type = conn.get('relationship_type', 'UNKNOWN')
                    if rel_type not in by_rel_type:
                        by_rel_type[rel_type] = []
                    by_rel_type[rel_type].append(conn)

                # Show all relationship types with counts
                output.append("**Relationship Types:**")
                for rel_type, conns in sorted(by_rel_type.items()):
                    output.append(f"- {rel_type}: {len(conns)} connection(s)")
                output.append("")

                # Show detailed connections
                for rel_type, conns in sorted(by_rel_type.items()):
                    output.append(f"**{rel_type}**:")
                    for conn in conns[:8]:  # Show up to 8 per type
                        direction = "→" if conn.get('direction') == 'outgoing' else "←"
                        neighbor_id = conn.get('neighbor_id', 'Unknown')
                        neighbor_type = conn.get('neighbor_type', 'Entity')
                        neighbor_desc = conn.get('neighbor_description', '')

                        rel_desc = conn.get('relationship_description', '')

                        # Format the connection line
                        conn_line = f"  {direction} **{neighbor_id}**"
                        if neighbor_type:
                            conn_line += f" (Type: {neighbor_type})"
                        if rel_desc:
                            conn_line += f"\n      └─ Relationship: {rel_desc}"

                        output.append(conn_line)

                        if neighbor_desc:
                            output.append(f"      └─ Details: {neighbor_desc[:200]}{'...' if len(neighbor_desc) > 200 else ''}")
                        output.append("")  # Add spacing between connections
            else:
                output.append("\n### 🔗 Direct Connections (1-hop Traversal)")
                output.append("⚠️  No direct connections found for this entity in the knowledge graph.")

            # Extended context (2-hop)
            extended_context = [
                c for c in context.get('extended_context', [])
                if c.get('second_hop_id')  # Filter out empty connections
            ]

            if extended_context and len(extended_context) > 0:
                # Remove duplicates and limit
                unique_second_hop = {}
                for conn in extended_context:
                    hop_id = conn.get('second_hop_id')
                    if hop_id and hop_id not in unique_second_hop:
                        unique_second_hop[hop_id] = conn

                if unique_second_hop:
                    output.append(f"\n### 🌐 Extended Network (2-hop Traversal)")
                    output.append(f"Discovered **{len(unique_second_hop)}** entities in the extended network (indirect connections):\n")

                    for hop_id, conn in list(unique_second_hop.items())[:10]:  # Show max 10
                        path_through = conn.get('path_through', 'Unknown')
                        hop_type = conn.get('second_hop_type', 'Entity')
                        rel_type = conn.get('relationship_type', 'CONNECTED')

                        # Format path: Entity -> Intermediary -> Second-hop Entity
                        output.append(f"  • **{hop_id}** (Type: {hop_type})")
                        output.append(f"      └─ Path: {entity_id} → {path_through} → {hop_id}")
                        output.append(f"      └─ Link Type: {rel_type}")
                        output.append("")  # Add spacing

            # Graph statistics
            unique_extended = len(set([c.get('second_hop_id') for c in extended_context if c.get('second_hop_id')]))
            total_reachable = len(direct_connections) + unique_extended

            output.append(f"\n### 📊 Knowledge Graph Statistics")
            output.append(f"- **Entity**: {entity_id}")
            output.append(f"- **Direct connections (1-hop)**: {len(direct_connections)} entities")
            output.append(f"- **Extended network (2-hop)**: {unique_extended} entities")
            output.append(f"- **Total reachable entities**: {total_reachable}")
            output.append(f"- **Relationship types**: {len(set(c.get('relationship_type', '') for c in direct_connections))}")
            output.append(f"- **Network density**: {'High' if len(direct_connections) > 5 else 'Medium' if len(direct_connections) > 2 else 'Low'}")

            # Add interpretation
            output.append(f"\n### 💡 Network Interpretation")
            if len(direct_connections) == 0:
                output.append("This entity appears to be isolated in the knowledge graph with no direct connections.")
            elif len(direct_connections) <= 2:
                output.append("This entity has limited connections, suggesting a peripheral role in the knowledge graph.")
            elif len(direct_connections) <= 5:
                output.append("This entity is moderately connected, playing a supportive role in the knowledge graph.")
            else:
                output.append("This entity is highly connected, indicating a central or important role in the knowledge graph.")

            if unique_extended > 0:
                output.append(f"The extended network reveals {unique_extended} additional entities within 2 hops, showing broader context and indirect relationships.")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Error formatting entity context: {e}")
            return str(context)

    def _format_graph_results(self, results: List[Dict], query: str) -> str:
        """Format graph search results for LLM context with enhanced detail"""
        try:
            formatted = f"## 🕸️  Graph Search Results for: {query}\n\n"
            formatted += f"Found {len(results)} result(s)\n\n"

            for idx, result in enumerate(results[:10], 1):  # Limit to top 10
                formatted += f"### Result {idx}\n"

                # Handle different result structures
                for key, value in result.items():
                    if isinstance(value, dict):
                        # Neo4j node or relationship
                        if 'id' in value:
                            formatted += f"**Entity**: {value.get('id', 'Unknown')}\n"
                            if 'type' in value:
                                formatted += f"  - Type: {value.get('type')}\n"
                            if 'description' in value:
                                formatted += f"  - Description: {value.get('description')}\n"
                    elif isinstance(value, list):
                        # List of relationships or entities
                        if value:  # Only show non-empty lists
                            formatted += f"**{key}**: \n"
                            for item in value[:5]:  # Limit items
                                if isinstance(item, dict):
                                    formatted += f"  - {item}\n"
                                else:
                                    formatted += f"  - {item}\n"
                    else:
                        formatted += f"**{key}**: {value}\n"

                formatted += "\n"

            return formatted

        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return str(results)

    async def find_entity_relationships(
        self,
        entity_name: str,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all relationships for a specific entity

        Args:
            entity_name: Name of the entity
            document_id: Optional document filter

        Returns:
            List of relationships
        """
        try:
            cypher = """
            MATCH (e:__Entity__)-[r]-(other:__Entity__)
            WHERE toLower(e.id) CONTAINS toLower($entity_name)
            """

            if document_id:
                cypher += """
                AND (e)-[:BELONGS_TO]->(:__Document__ {id: $document_id})
                """

            cypher += """
            RETURN e.id as entity,
                   type(r) as relationship_type,
                   r.description as rel_description,
                   other.id as related_entity,
                   other.description as related_description
            LIMIT 20
            """

            params = {"entity_name": entity_name}
            if document_id:
                params["document_id"] = document_id

            results = await self.neo4j.query_graph(cypher, params)
            return results

        except Exception as e:
            logger.error(f"Error finding entity relationships: {e}")
            return []

    async def find_paths_between_entities(
        self,
        entity1: str,
        entity2: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find paths between two entities

        Args:
            entity1: First entity name
            entity2: Second entity name
            max_depth: Maximum path length

        Returns:
            List of paths
        """
        try:
            cypher = f"""
            MATCH path = (e1:__Entity__)-[*1..{max_depth}]-(e2:__Entity__)
            WHERE toLower(e1.id) CONTAINS toLower($entity1) AND toLower(e2.id) CONTAINS toLower($entity2)
            RETURN path,
                   length(path) as path_length,
                   [n in nodes(path) | n.id] as entities,
                   [r in relationships(path) | type(r)] as relationship_types
            ORDER BY path_length
            LIMIT 5
            """

            params = {"entity1": entity1, "entity2": entity2}
            results = await self.neo4j.query_graph(cypher, params)
            return results

        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []


# Global graph search service instance
graph_search_service = GraphSearchService()
