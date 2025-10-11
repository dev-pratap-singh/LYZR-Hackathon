from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database - Required from environment
    postgres_user: str
    postgres_password: str
    postgres_db: str = "rag_database"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # PGVector Database
    pgvector_host: str = "pgvector"
    pgvector_port: int = 5432
    pgvector_db: str = "vector_db"

    # Neo4j - Password required from environment
    neo4j_host: str = "neo4j"
    neo4j_port: int = 7687
    neo4j_auth: str  # Format: username/password

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Document Processing
    chunk_size: int = 1200
    chunk_overlap: int = 400
    top_k_results: int = 10

    # Storage
    storage_path: str = "/app/storage"

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def pgvector_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
