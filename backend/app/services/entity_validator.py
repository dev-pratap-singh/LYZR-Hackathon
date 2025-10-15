"""
Entity Validation Service

This service extracts named entities (especially PERSON entities) from user queries
and validates whether they exist in the knowledge graph before performing RAG retrieval.

This prevents the RAG system from hallucinating information about non-existent entities
by matching semantic similarity on other aspects of the query.
"""

import re
from typing import List, Dict, Optional, Set
from app.services.neo4j_service import Neo4jService
from app.config import settings


class EntityValidator:
    """Service for extracting and validating entities from user queries"""

    def __init__(self):
        self.neo4j_service = Neo4jService()

        # Common question words and patterns to ignore
        self.ignore_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'tell', 'show', 'give', 'find', 'get', 'list', 'search', 'explain',
            'describe', 'summarize', 'provide', 'display', 'where', 'when', 'how', 'why',
            'can', 'could', 'would', 'should', 'may', 'might', 'must', 'will', 'shall'
        }

    def extract_potential_person_names(self, query: str) -> List[str]:
        """
        Extract potential person names from query using simple heuristics.

        This is a lightweight alternative to spaCy that works well for most cases.
        Looks for capitalized words/phrases that could be person names.

        Args:
            query: The user's query string

        Returns:
            List of potential person names found in the query
        """
        # Remove question marks and other punctuation
        cleaned = re.sub(r'[?!.,;:]', '', query)

        # Split into words
        words = cleaned.split()

        potential_names = []
        i = 0

        while i < len(words):
            word = words[i]

            # Check if word starts with capital letter and is not at start of sentence
            if word and word[0].isupper() and word.lower() not in self.ignore_words:
                # Check if it's part of a multi-word name
                name_parts = [word]
                j = i + 1

                # Look ahead for additional capitalized words (multi-word names)
                while j < len(words) and words[j] and words[j][0].isupper():
                    if words[j].lower() not in self.ignore_words:
                        name_parts.append(words[j])
                        j += 1
                    else:
                        break

                full_name = ' '.join(name_parts)
                potential_names.append(full_name)
                i = j
            else:
                i += 1

        return potential_names

    async def get_all_entities_in_graph(self, document_id: Optional[str] = None) -> Set[str]:
        """
        Get all entity names from the knowledge graph.

        Args:
            document_id: Optional document ID to filter entities

        Returns:
            Set of all entity IDs (names) in the graph
        """
        try:
            query = """
            MATCH (e:__Entity__)
            """

            params = {}
            if document_id:
                query += """
                -[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                """
                params['document_id'] = document_id

            query += """
            RETURN DISTINCT e.id as entity_id
            """

            results = await self.neo4j_service.query_graph(query, params)

            # Extract entity IDs and create case-insensitive set
            entity_names = {result['entity_id'].lower() for result in results if result.get('entity_id')}
            return entity_names

        except Exception as e:
            print(f"Error fetching entities from graph: {e}")
            return set()

    async def validate_entities_exist(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Validate whether person names mentioned in the query exist in the knowledge graph.

        Args:
            query: User's query string
            document_id: Optional document ID to filter search

        Returns:
            Dict containing:
                - valid: bool - whether all mentioned entities exist
                - mentioned_entities: List[str] - entities found in query
                - missing_entities: List[str] - entities not found in graph
                - error_message: Optional[str] - user-friendly error message if invalid
        """
        # Extract potential person names from query
        mentioned_entities = self.extract_potential_person_names(query)

        # If no entities mentioned, validation passes
        if not mentioned_entities:
            return {
                'valid': True,
                'mentioned_entities': [],
                'missing_entities': [],
                'error_message': None
            }

        # Get all entities from knowledge graph
        known_entities = await self.get_all_entities_in_graph(document_id)

        # If graph is empty, we can't validate - let query proceed
        # (this handles cases where GraphRAG hasn't been run yet)
        if not known_entities:
            return {
                'valid': True,
                'mentioned_entities': mentioned_entities,
                'missing_entities': [],
                'error_message': None
            }

        # Check which mentioned entities are missing
        # Use partial matching: "Dev" should match "Dev Pratap Singh"
        missing_entities = []
        for entity in mentioned_entities:
            entity_lower = entity.lower()
            # Check for exact match or if the queried name is a substring of any known entity
            is_match = any(
                entity_lower == known_entity or
                entity_lower in known_entity or
                known_entity.startswith(entity_lower + " ")
                for known_entity in known_entities
            )
            if not is_match:
                missing_entities.append(entity)

        # If any entities are missing, validation fails
        if missing_entities:
            if len(missing_entities) == 1:
                error_message = (
                    f"I don't have any information about **{missing_entities[0]}** in the knowledge base. "
                    f"Please verify the name or ask about entities that exist in the uploaded documents."
                )
            else:
                entities_str = ', '.join(f"**{e}**" for e in missing_entities)
                error_message = (
                    f"I don't have any information about {entities_str} in the knowledge base. "
                    f"Please verify the names or ask about entities that exist in the uploaded documents."
                )

            return {
                'valid': False,
                'mentioned_entities': mentioned_entities,
                'missing_entities': missing_entities,
                'error_message': error_message
            }

        # All entities exist
        return {
            'valid': True,
            'mentioned_entities': mentioned_entities,
            'missing_entities': [],
            'error_message': None
        }

    def get_known_entity_names(self, entity_set: Set[str], limit: int = 10) -> str:
        """
        Format a list of known entity names for display to user.

        Args:
            entity_set: Set of entity names
            limit: Maximum number to display

        Returns:
            Formatted string of entity names
        """
        entities = list(entity_set)[:limit]
        if not entities:
            return "No entities found in knowledge base."

        entity_list = '\n'.join(f"- {entity}" for entity in entities)
        more_text = f"\n...and {len(entity_set) - limit} more" if len(entity_set) > limit else ""

        return f"Known entities in the knowledge base:\n{entity_list}{more_text}"


# Singleton instance
_entity_validator = None


def get_entity_validator() -> EntityValidator:
    """Get or create the singleton EntityValidator instance"""
    global _entity_validator
    if _entity_validator is None:
        _entity_validator = EntityValidator()
    return _entity_validator
