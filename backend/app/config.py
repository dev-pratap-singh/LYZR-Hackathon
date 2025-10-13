from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = ConfigDict(extra='ignore', env_file='.env', case_sensitive=False)

    # Database - Defaults provided for testing
    postgres_user: str = "test_user"
    postgres_password: str = "test_password"
    postgres_db: str = "rag_database"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # PGVector Database
    pgvector_host: str = "pgvector"
    pgvector_port: int = 5432
    pgvector_db: str = "vector_db"

    # Neo4j - Default auth provided for testing
    neo4j_host: str = "neo4j"
    neo4j_port: int = 7687
    neo4j_auth: str = "neo4j/test_password"  # Format: username/password

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Document Processing
    chunk_size: int = 1200
    chunk_overlap: int = 500  # Increased from 400 to reduce information loss at boundaries
    top_k_results: int = 20  # Increased from 15 to improve Context Recall further

    # Storage
    storage_path: str = "/app/storage"

    # GraphRAG Settings
    graphrag_enabled: bool = True
    graphrag_llm_model: str = "gpt-4o-mini"
    graphrag_embedding_model: str = "text-embedding-3-small"

    # GraphRAG Parallel Processing Settings
    graphrag_concurrency: int = 25  # Number of concurrent chunk processing tasks
    graphrag_max_retries: int = 3  # Maximum retries for failed chunk processing
    graphrag_base_backoff: float = 0.5  # Base backoff time in seconds for retries

    # Multi-Pass Graph Enrichment Settings
    graphrag_enable_multipass: bool = True  # Enable 3-pass graph enrichment
    graphrag_num_passes: int = 3  # Number of passes through document (1-3 recommended)

    # Entity Resolution
    entity_similarity_threshold: float = 0.85
    enable_entity_deduplication: bool = True

    # Elasticsearch Configuration
    elasticsearch_host: str = "elasticsearch"
    elasticsearch_port: int = 9200
    elasticsearch_index_name: str = "rag_documents"

    # Search Configuration
    enable_vector_search: bool = True
    enable_graph_search: bool = True
    enable_filter_search: bool = True
    default_search_tools: str = "auto"
    max_performance: bool = False  # Run all search tools in parallel

    @property
    def elasticsearch_url(self) -> str:
        return f"http://{self.elasticsearch_host}:{self.elasticsearch_port}"

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def pgvector_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"

    @property
    def neo4j_uri(self) -> str:
        return f"bolt://{self.neo4j_host}:{self.neo4j_port}"

    @property
    def neo4j_username(self) -> str:
        return self.neo4j_auth.split('/')[0]

    @property
    def neo4j_password(self) -> str:
        return self.neo4j_auth.split('/')[1]


settings = Settings()
