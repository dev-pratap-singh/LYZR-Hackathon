"""
Unit tests for document_helpers utility functions
Tests the document processing pipeline helper
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime


@pytest.mark.unit
class TestDocumentHelpers:
    """Test document helper functions"""

    @pytest.mark.asyncio
    @patch('app.utils.document_helpers.GraphRAGPipeline')
    @patch('app.utils.document_helpers.BM25SearchService')
    @patch('app.utils.document_helpers.VectorStoreService')
    @patch('app.utils.document_helpers.EmbeddingService')
    @patch('app.utils.document_helpers.DocumentProcessingService')
    @patch('app.utils.document_helpers.settings')
    async def test_process_document_pipeline_success(
        self,
        mock_settings,
        mock_doc_processor_class,
        mock_embedding_class,
        mock_vector_class,
        mock_bm25_class,
        mock_graphrag_class,
        sample_chunks,
        sample_embeddings
    ):
        """Test successful document processing pipeline"""
        from app.utils.document_helpers import process_document_pipeline

        # Mock settings
        mock_settings.graphrag_enabled = False
        mock_settings.enable_filter_search = False

        # Mock document
        mock_document = Mock()
        mock_document.id = uuid4()
        mock_document.filepath = "/tmp/test.pdf"
        mock_document.original_filename = "test.pdf"
        mock_document.text_filepath = None
        mock_document.processing_status = "pending"
        mock_document.is_processed = False

        # Mock DB session
        mock_db = Mock()
        mock_db.commit = Mock()

        # Mock document processor
        mock_doc_processor = AsyncMock()
        mock_process_result = {
            'success': True,
            'text_path': '/tmp/test.txt',
            'chunks': sample_chunks,
            'text_length': 1000
        }
        mock_doc_processor.process_document = AsyncMock(return_value=mock_process_result)
        mock_doc_processor_class.return_value = mock_doc_processor

        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings_batch = AsyncMock(return_value=sample_embeddings)
        mock_embedding_class.return_value = mock_embedding_service

        # Mock vector store
        mock_vector_store = AsyncMock()
        mock_vector_store.store_embeddings = AsyncMock(return_value=3)
        mock_vector_class.return_value = mock_vector_store

        # Mock BM25 service
        mock_bm25_service = AsyncMock()
        mock_bm25_service.build_index = AsyncMock()
        mock_bm25_class.return_value = mock_bm25_service

        # Mock file read
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"PDF content"

            # Test pipeline
            await process_document_pipeline(
                document=mock_document,
                db=mock_db,
                method="pymupdf"
            )

        # Assertions
        assert mock_document.processing_status == "completed"
        assert mock_document.is_processed is True
        assert mock_document.text_filepath == '/tmp/test.txt'
        assert mock_document.total_chunks == 3
        assert mock_db.commit.called

    @pytest.mark.asyncio
    @patch('app.utils.document_helpers.DocumentProcessingService')
    @patch('app.utils.document_helpers.settings')
    async def test_process_document_pipeline_processing_failure(
        self,
        mock_settings,
        mock_doc_processor_class
    ):
        """Test document pipeline with processing failure"""
        from app.utils.document_helpers import process_document_pipeline

        mock_settings.graphrag_enabled = False
        mock_settings.enable_filter_search = False

        # Mock document
        mock_document = Mock()
        mock_document.id = uuid4()
        mock_document.filepath = "/tmp/test.pdf"
        mock_document.original_filename = "test.pdf"
        mock_document.processing_status = "pending"

        # Mock DB session
        mock_db = Mock()
        mock_db.commit = Mock()

        # Mock document processor with failure
        mock_doc_processor = AsyncMock()
        mock_process_result = {
            'success': False,
            'error': 'Processing failed'
        }
        mock_doc_processor.process_document = AsyncMock(return_value=mock_process_result)
        mock_doc_processor_class.return_value = mock_doc_processor

        # Mock file read
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"PDF content"

            # Test pipeline - should raise exception
            with pytest.raises(Exception, match="Processing failed"):
                await process_document_pipeline(
                    document=mock_document,
                    db=mock_db,
                    method="pymupdf"
                )

        # Document should be marked as failed
        assert mock_document.processing_status == "failed"

    @pytest.mark.asyncio
    @patch('app.utils.document_helpers.neo4j_service')
    @patch('app.utils.document_helpers.GraphRAGPipeline')
    @patch('app.utils.document_helpers.BM25SearchService')
    @patch('app.utils.document_helpers.VectorStoreService')
    @patch('app.utils.document_helpers.EmbeddingService')
    @patch('app.utils.document_helpers.DocumentProcessingService')
    @patch('app.utils.document_helpers.settings')
    async def test_process_document_pipeline_with_graphrag(
        self,
        mock_settings,
        mock_doc_processor_class,
        mock_embedding_class,
        mock_vector_class,
        mock_bm25_class,
        mock_graphrag_class,
        mock_neo4j,
        sample_chunks,
        sample_embeddings
    ):
        """Test document pipeline with GraphRAG enabled"""
        from app.utils.document_helpers import process_document_pipeline

        # Mock settings
        mock_settings.graphrag_enabled = True
        mock_settings.enable_filter_search = False

        # Mock document
        mock_document = Mock()
        mock_document.id = uuid4()
        mock_document.filepath = "/tmp/test.pdf"
        mock_document.original_filename = "test.pdf"
        mock_document.text_filepath = "/tmp/test.txt"
        mock_document.processing_status = "pending"
        mock_document.is_processed = False

        # Mock DB session
        mock_db = Mock()
        mock_db.commit = Mock()

        # Mock services (similar to previous test)
        mock_doc_processor = AsyncMock()
        mock_process_result = {
            'success': True,
            'text_path': '/tmp/test.txt',
            'chunks': sample_chunks,
            'text_length': 1000
        }
        mock_doc_processor.process_document = AsyncMock(return_value=mock_process_result)
        mock_doc_processor_class.return_value = mock_doc_processor

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings_batch = AsyncMock(return_value=sample_embeddings)
        mock_embedding_class.return_value = mock_embedding_service

        mock_vector_store = AsyncMock()
        mock_vector_store.store_embeddings = AsyncMock(return_value=3)
        mock_vector_class.return_value = mock_vector_store

        mock_bm25_service = AsyncMock()
        mock_bm25_service.build_index = AsyncMock()
        mock_bm25_class.return_value = mock_bm25_service

        # Mock GraphRAG pipeline
        mock_graphrag = AsyncMock()
        mock_graph_result = {
            'entities_count': 10,
            'relationships_count': 15,
            'graph_documents': [],
            'chunks_processed': 3,
            'processing_time': 1.5
        }
        mock_graphrag.process_document = AsyncMock(return_value=mock_graph_result)
        mock_graphrag_class.return_value = mock_graphrag

        # Mock Neo4j service
        mock_neo4j.create_constraints = AsyncMock()
        mock_neo4j.import_graph_documents = AsyncMock()

        # Mock file read
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.side_effect = [b"PDF content", "Text content"]
            mock_open.return_value = mock_file

            # Test pipeline with GraphRAG
            await process_document_pipeline(
                document=mock_document,
                db=mock_db,
                method="pymupdf"
            )

        # Assertions
        assert mock_document.graph_processed is True
        assert mock_document.graph_entities_count == 10
        assert mock_document.graph_relationships_count == 15
        mock_graphrag.process_document.assert_called_once()
        mock_neo4j.create_constraints.assert_called_once()
        mock_neo4j.import_graph_documents.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.utils.document_helpers.elasticsearch_service')
    @patch('app.utils.document_helpers.BM25SearchService')
    @patch('app.utils.document_helpers.VectorStoreService')
    @patch('app.utils.document_helpers.EmbeddingService')
    @patch('app.utils.document_helpers.DocumentProcessingService')
    @patch('app.utils.document_helpers.settings')
    async def test_process_document_pipeline_with_elasticsearch(
        self,
        mock_settings,
        mock_doc_processor_class,
        mock_embedding_class,
        mock_vector_class,
        mock_bm25_class,
        mock_elasticsearch,
        sample_chunks,
        sample_embeddings
    ):
        """Test document pipeline with Elasticsearch indexing"""
        from app.utils.document_helpers import process_document_pipeline

        # Mock settings
        mock_settings.graphrag_enabled = False
        mock_settings.enable_filter_search = True

        # Mock document
        mock_document = Mock()
        mock_document.id = uuid4()
        mock_document.filepath = "/tmp/test.pdf"
        mock_document.original_filename = "test.pdf"
        mock_document.text_filepath = "/tmp/test.txt"
        mock_document.processing_status = "pending"
        mock_document.is_processed = False
        mock_document.author = "Test Author"
        mock_document.document_type = "pdf"
        mock_document.categories = []
        mock_document.tags = []
        mock_document.uploaded_at = datetime.now()
        mock_document.processed_at = None
        mock_document.file_size = 1024
        mock_document.total_chunks = 0
        mock_document.user_id = "user-123"
        mock_document.doc_metadata = {}

        # Mock DB session
        mock_db = Mock()
        mock_db.commit = Mock()

        # Mock services
        mock_doc_processor = AsyncMock()
        mock_process_result = {
            'success': True,
            'text_path': '/tmp/test.txt',
            'chunks': sample_chunks,
            'text_length': 1000
        }
        mock_doc_processor.process_document = AsyncMock(return_value=mock_process_result)
        mock_doc_processor_class.return_value = mock_doc_processor

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings_batch = AsyncMock(return_value=sample_embeddings)
        mock_embedding_class.return_value = mock_embedding_service

        mock_vector_store = AsyncMock()
        mock_vector_store.store_embeddings = AsyncMock(return_value=3)
        mock_vector_class.return_value = mock_vector_store

        mock_bm25_service = AsyncMock()
        mock_bm25_service.build_index = AsyncMock()
        mock_bm25_class.return_value = mock_bm25_service

        # Mock Elasticsearch
        mock_elasticsearch.create_index = Mock()
        mock_elasticsearch.index_document = AsyncMock(return_value=True)

        # Mock file read
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.__enter__.return_value.read.side_effect = [b"PDF content", "Text content"]
            mock_open.return_value = mock_file

            # Test pipeline with Elasticsearch
            await process_document_pipeline(
                document=mock_document,
                db=mock_db,
                method="pymupdf"
            )

        # Assertions
        assert mock_document.elasticsearch_indexed is True
        assert mock_document.elasticsearch_index_time is not None
        mock_elasticsearch.create_index.assert_called_once()
        mock_elasticsearch.index_document.assert_called_once()
