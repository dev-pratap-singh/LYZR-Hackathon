from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional
import urllib.parse


class Settings(BaseSettings):
    model_config = ConfigDict(extra='ignore', env_file='.env', case_sensitive=False)

    # Deployment Environment
    deployment_environment: str = "local"  # Options: local, azure, production

    # Azure Configuration
    azure_subscription_id: Optional[str] = None
    azure_resource_group: Optional[str] = None
    azure_location: str = "eastus"

    # Database - Defaults provided for testing
    postgres_user: str = "test_user"
    postgres_password: str = "test_password"
    postgres_db: str = "rag_database"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_ssl_mode: str = "prefer"  # Options: disable, prefer, require (Azure requires 'require')
    postgres_ssl_root_cert: Optional[str] = None  # Path to SSL root certificate for Azure

    # PGVector Database
    pgvector_host: str = "pgvector"
    pgvector_port: int = 5432
    pgvector_db: str = "vector_db"
    pgvector_user: Optional[str] = None  # Defaults to postgres_user if not set
    pgvector_password: Optional[str] = None  # Defaults to postgres_password if not set
    pgvector_ssl_mode: str = "prefer"
    pgvector_ssl_root_cert: Optional[str] = None

    # Neo4j - Default auth provided for testing
    neo4j_host: str = "neo4j"
    neo4j_port: int = 7687
    neo4j_auth: str = "neo4j/test_password"  # Format: username/password
    neo4j_protocol: str = "bolt"  # Options: bolt, bolt+s (Azure/AuraDB uses bolt+s)
    neo4j_encrypted: bool = False  # Set to True for Azure/AuraDB

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

    # Memory Management Configuration
    memory_enabled: bool = True
    memory_db_host: str = "postgres"
    memory_db_port: int = 5432
    memory_db_name: str = "rag_database"
    memory_db_user: str = "rag_user"
    memory_db_password: str = "rag_password"
    memory_approach: str = "external_llm"
    memory_model: str = "gpt-4o-mini"
    memory_session_id: str = "default_session"  # Fixed session ID shared across all components

    # Redis Configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None  # Required for Azure Cache for Redis
    redis_ssl: bool = False  # Set to True for Azure Cache for Redis
    redis_ssl_cert_reqs: str = "required"  # Options: none, optional, required

    # Azure Cache for Redis specific
    redis_access_key: Optional[str] = None  # Primary or secondary access key from Azure

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000"  # Comma-separated list

    @property
    def elasticsearch_url(self) -> str:
        return f"http://{self.elasticsearch_host}:{self.elasticsearch_port}"

    @property
    def postgres_url(self) -> str:
        """
        Generate PostgreSQL connection string with SSL support for Azure
        """
        # URL encode password to handle special characters
        encoded_password = urllib.parse.quote_plus(self.postgres_password)

        base_url = f"postgresql://{self.postgres_user}:{encoded_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

        # Add SSL parameters for Azure
        if self.postgres_ssl_mode != "disable":
            ssl_params = f"?sslmode={self.postgres_ssl_mode}"
            if self.postgres_ssl_root_cert:
                ssl_params += f"&sslrootcert={self.postgres_ssl_root_cert}"
            base_url += ssl_params

        return base_url

    @property
    def pgvector_url(self) -> str:
        """
        Generate PGVector PostgreSQL connection string with SSL support for Azure
        """
        # Use separate credentials for pgvector if provided, otherwise use main postgres credentials
        pgvector_user = self.pgvector_user or self.postgres_user
        pgvector_password = self.pgvector_password or self.postgres_password

        # URL encode password to handle special characters
        encoded_password = urllib.parse.quote_plus(pgvector_password)

        base_url = f"postgresql://{pgvector_user}:{encoded_password}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"

        # Add SSL parameters for Azure
        if self.pgvector_ssl_mode != "disable":
            ssl_params = f"?sslmode={self.pgvector_ssl_mode}"
            if self.pgvector_ssl_root_cert:
                ssl_params += f"&sslrootcert={self.pgvector_ssl_root_cert}"
            base_url += ssl_params

        return base_url

    @property
    def neo4j_uri(self) -> str:
        """
        Generate Neo4j connection URI with support for encrypted connections (Azure/AuraDB)
        """
        protocol = self.neo4j_protocol
        if self.neo4j_encrypted and not protocol.endswith("+s"):
            protocol = f"{protocol}+s"
        return f"{protocol}://{self.neo4j_host}:{self.neo4j_port}"

    @property
    def neo4j_username(self) -> str:
        return self.neo4j_auth.split('/')[0]

    @property
    def neo4j_password(self) -> str:
        return self.neo4j_auth.split('/')[1]

    @property
    def memory_db_url(self) -> str:
        """
        Generate memory database connection string (uses same postgres instance)
        """
        # URL encode password to handle special characters
        encoded_password = urllib.parse.quote_plus(self.memory_db_password)

        base_url = f"postgresql://{self.memory_db_user}:{encoded_password}@{self.memory_db_host}:{self.memory_db_port}/{self.memory_db_name}"

        # Add SSL parameters for Azure (use same SSL settings as main postgres)
        if self.postgres_ssl_mode != "disable":
            ssl_params = f"?sslmode={self.postgres_ssl_mode}"
            if self.postgres_ssl_root_cert:
                ssl_params += f"&sslrootcert={self.postgres_ssl_root_cert}"
            base_url += ssl_params

        return base_url

    @property
    def redis_url(self) -> str:
        """
        Generate Redis connection URL with support for Azure Cache for Redis
        """
        # Use redis_access_key if provided (Azure), otherwise use redis_password
        password = self.redis_access_key or self.redis_password

        if password:
            # URL encode password to handle special characters
            encoded_password = urllib.parse.quote_plus(password)
            if self.redis_ssl:
                return f"rediss://:{encoded_password}@{self.redis_host}:{self.redis_port}/0?ssl_cert_reqs={self.redis_ssl_cert_reqs}"
            else:
                return f"redis://:{encoded_password}@{self.redis_host}:{self.redis_port}/0"
        else:
            # No password (local development)
            if self.redis_ssl:
                return f"rediss://{self.redis_host}:{self.redis_port}/0"
            else:
                return f"redis://{self.redis_host}:{self.redis_port}/0"

    def is_azure_deployment(self) -> bool:
        """Check if running in Azure environment"""
        return self.deployment_environment in ["azure", "production"]

    @property
    def cors_allowed_origins(self) -> list[str]:
        """
        Parse CORS origins from comma-separated string
        Returns a list of allowed origins for CORS configuration
        """
        if not self.cors_origins:
            return ["http://localhost:3000"]
        return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]

    def get_database_connection_pool_settings(self) -> dict:
        """
        Get database connection pool settings optimized for deployment environment
        """
        if self.is_azure_deployment():
            return {
                "pool_size": 20,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 3600,  # Recycle connections every hour
                "pool_pre_ping": True,  # Verify connections before using
            }
        else:
            # Local development settings
            return {
                "pool_size": 5,
                "max_overflow": 5,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
            }


settings = Settings()
