"""
Graph Refinement Pipeline
Enhances knowledge graph with entity deduplication, cross-document linking,
and improved entity recognition using BFS traversal
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import asyncio
from datetime import datetime
import logging

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logging.warning("python-Levenshtein not installed. Using basic string similarity.")

from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


# ===== Configuration Models =====
class GraphRefinementConfig(BaseModel):
    """Configuration for graph refinement pipeline"""

    # Feature flags
    enable_deduplication: bool = True
    enable_cross_doc_linking: bool = False  # Disabled by default for single doc processing
    enable_entity_enhancement: bool = True

    # Similarity thresholds
    auto_merge_threshold: float = 0.95  # Auto-merge above this
    suggest_merge_threshold: float = 0.85  # Suggest to user
    min_similarity_threshold: float = 0.75  # Ignore below this

    # Cross-document linking
    cross_doc_similarity_threshold: float = 0.80
    max_cross_doc_links_per_entity: int = 10

    # BFS configuration
    bfs_max_depth: int = 3  # How deep to traverse
    bfs_batch_size: int = 100  # Process in batches

    # Performance
    max_nodes_to_process: int = 10000
    use_embeddings: bool = True
    cache_embeddings: bool = True

    # Neo4j (from settings)
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # OpenAI (from settings)
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"


@dataclass
class EntitySimilarity:
    """Similarity score between two entities"""
    entity_a_id: str
    entity_b_id: str
    string_similarity: float
    semantic_similarity: float
    contextual_similarity: float
    overall_similarity: float
    confidence: str  # "high", "medium", "low"
    reason: str


@dataclass
class MergeOperation:
    """Represents a merge operation between entities"""
    entity_a_id: str
    entity_b_id: str
    merged_entity_id: str
    similarity_score: float
    properties_merged: Dict[str, Any]
    relationships_transferred: int
    method: str  # "auto", "manual", "llm_validated"
    timestamp: datetime


# ===== Main Graph Refinement Pipeline =====
class GraphRefinementPipeline:
    """
    Comprehensive graph refinement pipeline that:
    1. Performs entity deduplication using BFS
    2. Creates cross-document links (optional)
    3. Enhances entity recognition
    4. Preserves all context during merges
    """

    def __init__(self, config: Optional[GraphRefinementConfig] = None):
        if config is None:
            # Create default config from settings
            config = GraphRefinementConfig(
                neo4j_uri=settings.neo4j_uri,
                neo4j_user=settings.neo4j_username,
                neo4j_password=settings.neo4j_password,
                openai_api_key=settings.openai_api_key or "",
                embedding_model=settings.openai_embedding_model,
                llm_model=settings.openai_model,
                auto_merge_threshold=settings.entity_similarity_threshold,
            )

        self.config = config

        # Neo4j connection
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

        # LLM and embeddings
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                openai_api_key=config.openai_api_key,
                temperature=0
            )

            if config.use_embeddings:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.openai_embeddings = OpenAIEmbeddings(
                    model=config.embedding_model,
                    openai_api_key=config.openai_api_key
                )
        else:
            logger.warning("OpenAI API key not provided. LLM features will be disabled.")
            self.llm = None
            self.embeddings_model = None
            self.openai_embeddings = None

        # Caches
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.processed_nodes: Set[str] = set()
        self.merge_log: List[MergeOperation] = []

    async def refine_graph(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point - refines entire graph or specific document

        Args:
            document_id: If provided, only process entities from this document

        Returns:
            Statistics about refinement operations
        """
        stats = {
            'entities_processed': 0,
            'duplicates_found': 0,
            'auto_merged': 0,
            'suggested_merges': 0,
            'cross_doc_links_created': 0,
            'entities_enhanced': 0,
            'total_time_seconds': 0
        }

        start_time = datetime.now()

        try:
            # Step 1: Entity Deduplication
            if self.config.enable_deduplication:
                logger.info("🔍 Step 1: Running entity deduplication...")
                dedup_stats = await self.deduplicate_entities(document_id)
                stats.update(dedup_stats)

            # Step 2: Cross-Document Linking (typically disabled for single doc processing)
            if self.config.enable_cross_doc_linking and not document_id:
                logger.info("🔗 Step 2: Creating cross-document links...")
                link_stats = await self.create_cross_document_links()
                stats['cross_doc_links_created'] = link_stats['links_created']

            # Step 3: Entity Enhancement (optional)
            if self.config.enable_entity_enhancement and self.llm:
                logger.info("✨ Step 3: Enhancing entities...")
                enhance_stats = await self.enhance_entities(document_id)
                stats['entities_enhanced'] = enhance_stats['enhanced_count']

            stats['total_time_seconds'] = (datetime.now() - start_time).total_seconds()

            logger.info(f"✅ Graph refinement complete!")
            logger.info(f"   Entities processed: {stats['entities_processed']}")
            logger.info(f"   Duplicates merged: {stats['auto_merged']}")
            if stats['cross_doc_links_created'] > 0:
                logger.info(f"   Cross-doc links: {stats['cross_doc_links_created']}")
            logger.info(f"   Time: {stats['total_time_seconds']:.2f}s")

            return stats

        except Exception as e:
            import traceback
            logger.error(f"❌ Error during graph refinement: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    # ===== ENTITY DEDUPLICATION =====

    async def deduplicate_entities(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Find and merge duplicate entities using BFS traversal
        """
        logger.info("🔄 Starting BFS deduplication scan...")

        stats = {
            'entities_processed': 0,
            'duplicates_found': 0,
            'auto_merged': 0,
            'suggested_merges': 0
        }

        # Get all entities to process
        entities = self._get_entities_for_processing(document_id)
        logger.info(f"   Found {len(entities)} entities to process")

        if len(entities) == 0:
            logger.info("   No entities to process")
            return stats

        # Track potential duplicates
        potential_duplicates: List[EntitySimilarity] = []

        # BFS traversal for each unprocessed entity
        for entity in entities:
            if entity['id'] in self.processed_nodes:
                continue

            duplicates = await self._bfs_find_duplicates(entity['id'])
            potential_duplicates.extend(duplicates)
            stats['entities_processed'] += 1

            # Process in batches
            if len(potential_duplicates) >= self.config.bfs_batch_size:
                batch_stats = await self._process_duplicate_batch(potential_duplicates)
                stats['duplicates_found'] += len(potential_duplicates)
                stats['auto_merged'] += batch_stats['auto_merged']
                stats['suggested_merges'] += batch_stats['suggested']
                potential_duplicates = []

        # Process remaining duplicates
        if potential_duplicates:
            batch_stats = await self._process_duplicate_batch(potential_duplicates)
            stats['duplicates_found'] += len(potential_duplicates)
            stats['auto_merged'] += batch_stats['auto_merged']
            stats['suggested_merges'] += batch_stats['suggested']

        return stats

    async def _bfs_find_duplicates(
        self,
        start_node_id: str,
        max_depth: Optional[int] = None
    ) -> List[EntitySimilarity]:
        """
        BFS traversal to find similar entities

        Args:
            start_node_id: Starting node for BFS
            max_depth: Maximum depth to traverse (default from config)

        Returns:
            List of potential duplicate entities with similarity scores
        """
        max_depth = max_depth or self.config.bfs_max_depth

        queue = deque([(start_node_id, 0)])  # (node_id, depth)
        visited = {start_node_id}
        potential_duplicates = []

        # Get start node data
        start_node = self._get_node_data(start_node_id)
        if not start_node:
            return []

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(current_id)

            for neighbor in neighbors:
                neighbor_id = neighbor['id']

                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                # Check similarity with start node
                similarity = await self._calculate_entity_similarity(
                    start_node,
                    neighbor
                )

                # Only consider if above minimum threshold
                if similarity.overall_similarity >= self.config.min_similarity_threshold:
                    potential_duplicates.append(similarity)

                # Add to queue for further traversal
                if depth + 1 < max_depth:
                    queue.append((neighbor_id, depth + 1))

            # Mark as processed
            self.processed_nodes.add(current_id)

        return potential_duplicates

    async def _calculate_entity_similarity(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any]
    ) -> EntitySimilarity:
        """
        Calculate similarity between two entities using multiple methods
        """
        # 1. String Similarity (Levenshtein or fallback)
        name_a = entity_a.get('name', '').lower()
        name_b = entity_b.get('name', '').lower()

        if LEVENSHTEIN_AVAILABLE:
            string_sim = 1 - (Levenshtein.distance(name_a, name_b) / max(len(name_a), len(name_b), 1))
        else:
            # Fallback to simple character overlap
            if len(name_a) == 0 or len(name_b) == 0:
                string_sim = 0.0
            else:
                common = len(set(name_a) & set(name_b))
                total = len(set(name_a) | set(name_b))
                string_sim = common / total if total > 0 else 0.0

        # 2. Semantic Similarity (Embeddings)
        semantic_sim = 0.0
        if self.config.use_embeddings and self.embeddings_model:
            # Get descriptions, fallback to name, then to empty string
            desc_a = entity_a.get('description') or name_a or entity_a.get('id', '')
            desc_b = entity_b.get('description') or name_b or entity_b.get('id', '')
            # Only compute if we have valid text for both
            if desc_a and desc_b:
                semantic_sim = await self._compute_semantic_similarity(desc_a, desc_b)

        # 3. Contextual Similarity (Shared relationships)
        contextual_sim = await self._compute_contextual_similarity(
            entity_a['id'],
            entity_b['id']
        )

        # 4. Type similarity
        type_match = 1.0 if entity_a.get('type') == entity_b.get('type') else 0.0

        # Overall similarity (weighted average)
        overall = (
            string_sim * 0.3 +
            semantic_sim * 0.3 +
            contextual_sim * 0.2 +
            type_match * 0.2
        )

        # Determine confidence
        if overall >= self.config.auto_merge_threshold:
            confidence = "high"
            reason = "Very high similarity across all metrics"
        elif overall >= self.config.suggest_merge_threshold:
            confidence = "medium"
            reason = "Moderate similarity - needs review"
        else:
            confidence = "low"
            reason = "Low similarity - likely not duplicates"

        return EntitySimilarity(
            entity_a_id=entity_a['id'],
            entity_b_id=entity_b['id'],
            string_similarity=string_sim,
            semantic_similarity=semantic_sim,
            contextual_similarity=contextual_sim,
            overall_similarity=overall,
            confidence=confidence,
            reason=reason
        )

    async def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity using embeddings"""
        if not self.embeddings_model:
            return 0.0

        # Safety check for None or empty strings
        if not text_a or not text_b:
            return 0.0

        # Check cache
        cache_key_a = hash(text_a)
        cache_key_b = hash(text_b)

        if cache_key_a not in self.embedding_cache:
            emb_a = self.embeddings_model.encode(text_a)
            if self.config.cache_embeddings:
                self.embedding_cache[cache_key_a] = emb_a
        else:
            emb_a = self.embedding_cache[cache_key_a]

        if cache_key_b not in self.embedding_cache:
            emb_b = self.embeddings_model.encode(text_b)
            if self.config.cache_embeddings:
                self.embedding_cache[cache_key_b] = emb_b
        else:
            emb_b = self.embedding_cache[cache_key_b]

        # Cosine similarity
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
        return float(similarity)

    async def _compute_contextual_similarity(
        self,
        entity_a_id: str,
        entity_b_id: str
    ) -> float:
        """
        Compute contextual similarity based on shared relationships
        """
        try:
            with self.driver.session() as session:
                # Use OPTIONAL MATCH to handle cases where entities have no relationships
                result = session.run("""
                    MATCH (a:__Entity__ {id: $entity_a_id})
                    MATCH (b:__Entity__ {id: $entity_b_id})
                    OPTIONAL MATCH (a)-[r1]-(shared)-[r2]-(b)
                    WITH a, b, COUNT(DISTINCT shared) as shared_count
                    OPTIONAL MATCH (a)-[r]-(all_a)
                    WITH a, b, shared_count, COUNT(DISTINCT all_a) as total_a
                    OPTIONAL MATCH (b)-[r2]-(all_b)
                    WITH shared_count, total_a, COUNT(DISTINCT all_b) as total_b
                    RETURN shared_count, total_a, total_b
                """, entity_a_id=entity_a_id, entity_b_id=entity_b_id)

                record = result.single()
                if not record:
                    return 0.0

                # Safely get values with defaults
                shared = record.get('shared_count', 0) or 0
                total_a = record.get('total_a', 0) or 0
                total_b = record.get('total_b', 0) or 0

                if total_a == 0 or total_b == 0:
                    return 0.0

                # Jaccard similarity
                return shared / (total_a + total_b - shared + 1e-10)
        except Exception as e:
            logger.warning(f"Error computing contextual similarity: {e}")
            return 0.0

    async def _process_duplicate_batch(
        self,
        duplicates: List[EntitySimilarity]
    ) -> Dict[str, int]:
        """
        Process a batch of potential duplicates
        """
        stats = {'auto_merged': 0, 'suggested': 0}

        for dup in duplicates:
            if dup.confidence == "high":
                # Auto-merge high confidence duplicates
                try:
                    await self._merge_entities(
                        dup.entity_a_id,
                        dup.entity_b_id,
                        method="auto",
                        similarity_score=dup.overall_similarity
                    )
                    stats['auto_merged'] += 1
                    logger.info(f"   ✅ Auto-merged: {dup.entity_a_id} + {dup.entity_b_id} (score: {dup.overall_similarity:.2f})")
                except Exception as e:
                    logger.error(f"   ❌ Failed to merge {dup.entity_a_id} + {dup.entity_b_id}: {e}")

            elif dup.confidence == "medium":
                # Log for manual review
                stats['suggested'] += 1
                logger.info(f"   ⚠️  Suggest merge: {dup.entity_a_id} + {dup.entity_b_id} (score: {dup.overall_similarity:.2f})")
                await self._log_suggested_merge(dup)

        return stats

    async def _merge_entities(
        self,
        entity_a_id: str,
        entity_b_id: str,
        method: str = "auto",
        similarity_score: float = 0.0
    ) -> str:
        """
        Merge two entities without losing context

        Returns:
            ID of the merged entity
        """
        with self.driver.session() as session:
            # 1. Get both entities
            result = session.run("""
                MATCH (a:__Entity__ {id: $entity_a_id})
                MATCH (b:__Entity__ {id: $entity_b_id})
                RETURN properties(a) as a, properties(b) as b
            """, entity_a_id=entity_a_id, entity_b_id=entity_b_id)

            record = result.single()
            if not record or not record['a'] or not record['b']:
                logger.warning(f"Entities not found for merge: {entity_a_id}, {entity_b_id}")
                return entity_a_id

            entity_a = dict(record['a']) if record['a'] else {}
            entity_b = dict(record['b']) if record['b'] else {}

            # 2. Merge properties (keep most complete)
            merged_props = self._merge_properties(entity_a, entity_b)

            # Use entity_a as the merged entity (update in place)
            merged_entity_id = entity_a_id

            # 3. Update entity A with merged properties
            session.run("""
                MATCH (a:__Entity__ {id: $entity_a_id})
                SET a.description = $description,
                    a.merged_from = coalesce(a.merged_from, []) + [$entity_b_id],
                    a.merge_confidence = $confidence,
                    a.last_merged_at = datetime()
            """,
                entity_a_id=entity_a_id,
                entity_b_id=entity_b_id,
                description=merged_props['description'],
                confidence=similarity_score
            )

            # 4. Transfer all relationships from entity B to entity A
            # Get all relationships from entity B
            result = session.run("""
                MATCH (b:__Entity__ {id: $entity_b_id})-[r]-(other)
                WHERE other.id <> $entity_a_id
                RETURN other.id as other_id, type(r) as rel_type,
                       startNode(r).id = b.id as is_outgoing,
                       properties(r) as props
            """, entity_a_id=entity_a_id, entity_b_id=entity_b_id)

            relationships_transferred = 0
            for record in result:
                other_id = record['other_id']
                rel_type = record['rel_type']
                is_outgoing = record['is_outgoing']
                props = record.get('props', {})

                # Skip if relationship already exists
                try:
                    if is_outgoing:
                        check = session.run("""
                            MATCH (a:__Entity__ {id: $entity_a_id})
                            MATCH (other {id: $other_id})
                            OPTIONAL MATCH (a)-[r:%s]->(other)
                            RETURN count(r) as count
                        """ % rel_type, entity_a_id=entity_a_id, other_id=other_id)
                    else:
                        check = session.run("""
                            MATCH (a:__Entity__ {id: $entity_a_id})
                            MATCH (other {id: $other_id})
                            OPTIONAL MATCH (other)-[r:%s]->(a)
                            RETURN count(r) as count
                        """ % rel_type, entity_a_id=entity_a_id, other_id=other_id)

                    check_result = check.single()
                    if check_result and check_result.get('count', 0) > 0:
                        # Relationship already exists, skip
                        continue
                except Exception as e:
                    logger.warning(f"Error checking relationship: {e}")
                    continue

                # Create new relationship
                try:
                    if is_outgoing:
                        session.run("""
                            MATCH (a:__Entity__ {id: $entity_a_id})
                            MATCH (other {id: $other_id})
                            CREATE (a)-[r:%s]->(other)
                            SET r = $props
                        """ % rel_type, entity_a_id=entity_a_id, other_id=other_id, props=props)
                    else:
                        session.run("""
                            MATCH (a:__Entity__ {id: $entity_a_id})
                            MATCH (other {id: $other_id})
                            CREATE (other)-[r:%s]->(a)
                            SET r = $props
                        """ % rel_type, entity_a_id=entity_a_id, other_id=other_id, props=props)
                    relationships_transferred += 1
                except Exception as e:
                    logger.warning(f"Error creating relationship: {e}")

            # 5. Delete entity B
            session.run("""
                MATCH (b:__Entity__ {id: $entity_b_id})
                DETACH DELETE b
            """, entity_b_id=entity_b_id)

            # 6. Log merge
            merge_op = MergeOperation(
                entity_a_id=entity_a_id,
                entity_b_id=entity_b_id,
                merged_entity_id=merged_entity_id,
                similarity_score=similarity_score,
                properties_merged=merged_props,
                relationships_transferred=relationships_transferred,
                method=method,
                timestamp=datetime.now()
            )
            self.merge_log.append(merge_op)

            return merged_entity_id

    def _merge_properties(
        self,
        entity_a: Dict[str, Any],
        entity_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge properties from two entities, keeping the most complete information
        """
        merged = {}

        # Name: prefer longer/more complete name
        name_a = entity_a.get('id', '')  # Using 'id' as name for Neo4j entities
        name_b = entity_b.get('id', '')
        merged['name'] = name_a if len(name_a) >= len(name_b) else name_b

        # Description: combine both if different
        desc_a = entity_a.get('description', '')
        desc_b = entity_b.get('description', '')
        if desc_a and desc_b and desc_a != desc_b:
            merged['description'] = f"{desc_a}. {desc_b}"
        else:
            merged['description'] = desc_a or desc_b

        return merged

    # ===== CROSS-DOCUMENT LINKING =====

    async def create_cross_document_links(self) -> Dict[str, Any]:
        """
        Create relationships between entities across different documents
        """
        logger.info("🔗 Finding cross-document entity connections...")

        stats = {'links_created': 0}

        # Placeholder for cross-document linking
        # This would require more complex logic to identify entities across documents

        return stats

    # ===== ENTITY ENHANCEMENT =====

    async def enhance_entities(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance entities with better descriptions and metadata
        """
        stats = {'enhanced_count': 0}

        if not self.llm:
            logger.info("Entity enhancement skipped (no LLM available)")
            return stats

        entities = self._get_entities_for_processing(document_id)

        for entity in entities[:10]:  # Limit to first 10 for efficiency
            # Check if entity needs enhancement
            if await self._needs_enhancement(entity):
                enhanced = await self._enhance_entity(entity)
                if enhanced:
                    stats['enhanced_count'] += 1

        return stats

    async def _needs_enhancement(self, entity: Dict) -> bool:
        """Check if entity needs enhancement"""
        # Enhance if missing description or very short description
        desc = entity.get('description', '')
        return not desc or len(desc) < 50

    async def _enhance_entity(self, entity: Dict) -> bool:
        """Use LLM to enhance entity with better description"""
        if not self.llm:
            return False

        prompt = f"""Enhance this knowledge graph entity with a comprehensive description.

Entity Name: {entity.get('name', entity.get('id'))}
Type: {entity.get('type', 'Unknown')}
Current Description: {entity.get('description', 'None')}

Provide a clear, informative description (2-3 sentences) about what this entity represents.
"""

        response = await self.llm.ainvoke(prompt)
        enhanced_description = response.content.strip()

        # Update entity in Neo4j
        with self.driver.session() as session:
            session.run("""
                MATCH (e:__Entity__ {id: $entity_id})
                SET e.description = $description,
                    e.enhanced = true,
                    e.enhanced_at = datetime()
            """, entity_id=entity['id'], description=enhanced_description)

        return True

    # ===== HELPER METHODS =====

    def _get_entities_for_processing(
        self,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get entities to process"""
        try:
            with self.driver.session() as session:
                if document_id:
                    query = """
                        MATCH (e:__Entity__)-[:BELONGS_TO]->(d:__Document__ {id: $document_id})
                        RETURN e.id as id, e.id as name,
                               e.description as description
                        LIMIT $limit
                    """
                    result = session.run(query, document_id=document_id, limit=self.config.max_nodes_to_process)
                else:
                    query = """
                        MATCH (e:__Entity__)
                        RETURN e.id as id, e.id as name,
                               e.description as description
                        LIMIT $limit
                    """
                    result = session.run(query, limit=self.config.max_nodes_to_process)

                entities = []
                for record in result:
                    if record and record.get('id'):
                        entities.append(dict(record))
                return entities
        except Exception as e:
            logger.error(f"Error getting entities for processing: {e}")
            return []

    def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:__Entity__ {id: $node_id})
                    RETURN n.id as id, n.id as name,
                           n.description as description
                """, node_id=node_id)

                record = result.single()
                if record and record['id']:
                    return dict(record)
                return None
        except Exception as e:
            logger.warning(f"Error getting node data for {node_id}: {e}")
            return None

    def _get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Get neighboring nodes"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:__Entity__ {id: $node_id})-[r]-(neighbor:__Entity__)
                    RETURN DISTINCT neighbor.id as id, neighbor.id as name,
                           neighbor.description as description
                    LIMIT 100
                """, node_id=node_id)

                neighbors = []
                for record in result:
                    if record and record.get('id'):
                        neighbors.append(dict(record))
                return neighbors
        except Exception as e:
            logger.warning(f"Error getting neighbors for {node_id}: {e}")
            return []

    async def _log_suggested_merge(self, similarity: EntitySimilarity):
        """Log suggested merges for manual review"""
        with self.driver.session() as session:
            session.run("""
                CREATE (s:SuggestedMerge {
                    entity_a: $entity_a,
                    entity_b: $entity_b,
                    similarity_score: $score,
                    reason: $reason,
                    created_at: datetime(),
                    status: 'pending'
                })
            """,
                entity_a=similarity.entity_a_id,
                entity_b=similarity.entity_b_id,
                score=similarity.overall_similarity,
                reason=similarity.reason
            )

    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get statistics about merge operations"""
        return {
            'total_merges': len(self.merge_log),
            'auto_merges': len([m for m in self.merge_log if m.method == 'auto']),
            'manual_merges': len([m for m in self.merge_log if m.method == 'manual']),
            'avg_similarity': np.mean([m.similarity_score for m in self.merge_log]) if self.merge_log else 0,
            'total_relationships_transferred': sum(m.relationships_transferred for m in self.merge_log)
        }

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
