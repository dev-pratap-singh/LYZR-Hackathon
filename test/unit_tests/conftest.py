"""
Pytest Configuration and Fixtures
Provides shared fixtures for all tests
"""
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock
import pytest
from uuid import uuid4

# Add backend to Python path so we can import from app
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Mock pgvector and other database-specific modules before importing
sys.modules['pgvector'] = MagicMock()
sys.modules['pgvector.sqlalchemy'] = MagicMock()
sys.modules['psycopg2'] = MagicMock()
sys.modules['neo4j'] = MagicMock()
sys.modules['elasticsearch'] = MagicMock()
sys.modules['elasticsearch.exceptions'] = MagicMock()
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['graphdatascience'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_experimental'] = MagicMock()
sys.modules['langchain_experimental.graph_transformers'] = MagicMock()
sys.modules['langchain_neo4j'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.graphs'] = MagicMock()
sys.modules['langchain_community.graphs.graph_document'] = MagicMock()
sys.modules['elasticsearch_dsl'] = MagicMock()


# ===== Mock Settings =====
@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    from unittest.mock import Mock

    settings = Mock()
    settings.postgres_user = "test_user"
    settings.postgres_password = "test_pass"
    settings.postgres_db = "test_db"
    settings.postgres_host = "localhost"
    settings.postgres_port = 5432
    settings.pgvector_host = "localhost"
    settings.pgvector_port = 5432
    settings.pgvector_db = "vector_test_db"
    settings.neo4j_host = "localhost"
    settings.neo4j_port = 7687
    settings.neo4j_auth = "neo4j/testpassword"
    settings.openai_api_key = "sk-test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.openai_embedding_model = "text-embedding-3-small"
    settings.chunk_size = 1200
    settings.chunk_overlap = 400
    settings.top_k_results = 10
    settings.storage_path = "/tmp/test_storage"
    settings.graphrag_enabled = True
    settings.graphrag_llm_model = "gpt-4o-mini"
    settings.graphrag_embedding_model = "text-embedding-3-small"
    settings.entity_similarity_threshold = 0.85
    settings.enable_entity_deduplication = True
    settings.elasticsearch_host = "localhost"
    settings.elasticsearch_port = 9200
    settings.elasticsearch_index_name = "test_documents"
    settings.enable_vector_search = True
    settings.enable_graph_search = True
    settings.enable_filter_search = True
    settings.default_search_tools = "auto"
    settings.postgres_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
    settings.pgvector_url = "postgresql://test_user:test_pass@localhost:5432/vector_test_db"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_username = "neo4j"
    settings.neo4j_password = "testpassword"
    settings.elasticsearch_url = "http://localhost:9200"

    return settings


# ===== Mock Database Session =====
@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.refresh = Mock()
    session.query = Mock()
    session.execute = Mock()
    session.delete = Mock()
    return session


# ===== Sample Data Fixtures =====
@pytest.fixture
def sample_document_id():
    """Generate a sample document UUID"""
    return str(uuid4())


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing"""
    return [
        {
            'index': 0,
            'text': 'This is the first chunk of text.',
            'size': 33,
            'hash': 'abc123',
            'chunk_metadata': {'chunk_index': 0, 'total_chunks': 3}
        },
        {
            'index': 1,
            'text': 'This is the second chunk of text.',
            'size': 34,
            'hash': 'def456',
            'chunk_metadata': {'chunk_index': 1, 'total_chunks': 3}
        },
        {
            'index': 2,
            'text': 'This is the third chunk of text.',
            'size': 33,
            'hash': 'ghi789',
            'chunk_metadata': {'chunk_index': 2, 'total_chunks': 3}
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Embedding 1
        [0.6, 0.7, 0.8, 0.9, 1.0],  # Embedding 2
        [0.2, 0.3, 0.4, 0.5, 0.6]   # Embedding 3
    ]


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding"""
    return [0.15, 0.25, 0.35, 0.45, 0.55]


@pytest.fixture
def sample_text():
    """Sample text for document processing"""
    return """
    This is a sample document for testing purposes.

    It contains multiple paragraphs and sections.
    The text is long enough to be chunked into multiple pieces.

    This helps us test the document processing pipeline.
    We can verify that text extraction, chunking, and embedding work correctly.
    """


@pytest.fixture
def sample_pdf_content():
    """Sample PDF binary content (mock)"""
    return b"Mock PDF Content Here"


# ===== Mock External Services =====
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = AsyncMock()

    # Mock embeddings response
    embedding_response = Mock()
    embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
    client.embeddings.create = AsyncMock(return_value=embedding_response)

    # Mock chat completion response
    completion_response = Mock()
    completion_response.choices = [Mock(message=Mock(content="Test response"))]
    client.chat.completions.create = AsyncMock(return_value=completion_response)

    return client


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    driver = Mock()
    session = Mock()
    driver.session = Mock(return_value=session)
    session.run = Mock()
    session.close = Mock()
    return driver


@pytest.fixture
def mock_elasticsearch_client():
    """Mock Elasticsearch client"""
    client = Mock()
    client.indices.create = Mock()
    client.indices.exists = Mock(return_value=False)
    client.index = Mock()
    client.search = Mock()
    client.delete = Mock()
    return client


# ===== Async Test Support =====
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
