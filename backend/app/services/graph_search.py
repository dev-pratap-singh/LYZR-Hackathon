"""
Graph Search Service using Neo4j for relationship queries
Implements Cypher query generation for knowledge graph retrieval
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.config import settings
from app.services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)


class GraphSearchService:
    """Service for searching relationships and entities in the knowledge graph"""

    def __init__(self):
        self.neo4j = neo4j_service
        self.llm = ChatOpenAI(
            model=settings.graphrag_llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        logger.info("GraphSearchService initialized")

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

    async def search_relationships(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> str:
        """
        Search for relationships in the knowledge graph

        Args:
            query: Natural language query about relationships
            document_id: Optional document ID to filter results

        Returns:
            Formatted search results as string
        """
        try:
            logger.info(f"Graph search query: {query}")

            # Generate Cypher query
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

            # Format results
            formatted_results = self._format_graph_results(results, query)

            logger.info(f"Graph search returned {len(results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return f"Error searching graph: {str(e)}"

    def _format_graph_results(self, results: List[Dict], query: str) -> str:
        """Format graph search results for LLM context"""
        try:
            formatted = f"## Graph Search Results for: {query}\n\n"

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
            WHERE e.id CONTAINS $entity_name
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
            WHERE e1.id CONTAINS $entity1 AND e2.id CONTAINS $entity2
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
