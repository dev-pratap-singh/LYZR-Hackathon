"""
Unit tests for DocumentProcessingService
Tests PDF processing, text extraction, and chunking
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from pathlib import Path
import hashlib


@pytest.mark.unit
class TestDocumentProcessingService:
    """Test DocumentProcessingService class"""

    @patch('app.services.document_processor.settings')
    def test_init(self, mock_settings):
        """Test DocumentProcessingService initialization"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        service = DocumentProcessingService()

        assert service.storage_path == Path("/tmp/test_storage")
        assert service.text_splitter is not None

    @pytest.mark.asyncio
    @patch('app.services.document_processor.settings')
    async def test_save_uploaded_file(
        self, mock_settings, sample_pdf_content
    ):
        """Test saving uploaded file"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Mock file operations with proper async context manager
        mock_file = AsyncMock()
        mock_file.write = AsyncMock()

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        with patch('app.services.document_processor.aiofiles.open', return_value=mock_context):
            service = DocumentProcessingService()

            # Test file save
            file_path, file_hash = await service.save_uploaded_file(
                sample_pdf_content,
                "test.pdf"
            )

            expected_hash = hashlib.sha256(sample_pdf_content).hexdigest()[:16]
            assert expected_hash in file_path
            assert "test.pdf" in file_path
            assert file_hash == expected_hash
            mock_file.write.assert_called_once_with(sample_pdf_content)

    @pytest.mark.asyncio
    @patch('app.services.document_processor.fitz')
    @patch('app.services.document_processor.settings')
    async def test_process_pdf_with_pymupdf(self, mock_settings, mock_fitz):
        """Test PDF processing with PyMuPDF"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=2)  # 2 pages

        mock_page1 = Mock()
        mock_page1.get_text = Mock(return_value="Page 1 content")

        mock_page2 = Mock()
        mock_page2.get_text = Mock(return_value="Page 2 content")

        mock_doc.__getitem__ = Mock(side_effect=[mock_page1, mock_page2])
        mock_doc.close = Mock()

        mock_fitz.open = Mock(return_value=mock_doc)

        service = DocumentProcessingService()

        # Test processing
        text = await service.process_pdf_with_pymupdf("/tmp/test.pdf")

        assert "Page 1" in text
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentConverter')
    @patch('app.services.document_processor.settings')
    async def test_process_pdf_with_docling(self, mock_settings, mock_converter_class):
        """Test PDF processing with Docling"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Mock Docling converter
        mock_converter = Mock()
        mock_result = Mock()
        mock_document = Mock()
        mock_document.export_to_markdown = Mock(return_value="# Markdown content")
        mock_result.document = mock_document
        mock_converter.convert = Mock(return_value=mock_result)
        mock_converter_class.return_value = mock_converter

        service = DocumentProcessingService()

        # Test processing
        text = await service.process_pdf_with_docling("/tmp/test.pdf")

        assert text == "# Markdown content"
        mock_converter.convert.assert_called_once_with("/tmp/test.pdf")

    @pytest.mark.asyncio
    @patch('app.services.document_processor.settings')
    async def test_save_text_file(self, mock_settings, sample_text):
        """Test saving extracted text to file"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Mock file operations with proper async context manager
        mock_file = AsyncMock()
        mock_file.write = AsyncMock()

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        with patch('app.services.document_processor.aiofiles.open', return_value=mock_context):
            service = DocumentProcessingService()

            # Test text save
            text_path = await service.save_text_file(
                sample_text,
                "test.pdf",
                "abc123"
            )

            assert "abc123" in text_path
            assert "test.txt" in text_path
            mock_file.write.assert_called_once_with(sample_text)

    @patch('app.services.document_processor.settings')
    def test_chunk_text(self, mock_settings, sample_text):
        """Test text chunking"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 100  # Small chunks for testing
        mock_settings.chunk_overlap = 20

        service = DocumentProcessingService()

        # Test chunking
        chunks = service.chunk_text(sample_text)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert 'index' in chunk
            assert 'text' in chunk
            assert 'size' in chunk
            assert 'hash' in chunk
            assert 'chunk_metadata' in chunk
            assert chunk['index'] == i
            assert chunk['chunk_metadata']['chunk_index'] == i
            assert chunk['chunk_metadata']['total_chunks'] == len(chunks)

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessingService.chunk_text')
    @patch('app.services.document_processor.DocumentProcessingService.save_text_file')
    @patch('app.services.document_processor.DocumentProcessingService.process_pdf_with_pymupdf')
    @patch('app.services.document_processor.DocumentProcessingService.save_uploaded_file')
    @patch('app.services.document_processor.settings')
    async def test_process_document_pymupdf(
        self,
        mock_settings,
        mock_save_file,
        mock_process_pdf,
        mock_save_text,
        mock_chunk,
        sample_pdf_content,
        sample_text,
        sample_chunks
    ):
        """Test complete document processing pipeline with PyMuPDF"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Setup mocks
        mock_save_file.return_value = ("/tmp/test.pdf", "abc123")
        mock_process_pdf.return_value = sample_text
        mock_save_text.return_value = "/tmp/test.txt"
        mock_chunk.return_value = sample_chunks

        service = DocumentProcessingService()

        # Test full processing
        result = await service.process_document(
            sample_pdf_content,
            "test.pdf",
            "doc-123",
            method="pymupdf"
        )

        assert result['success'] is True
        assert result['method'] == "pymupdf"
        assert result['pdf_path'] == "/tmp/test.pdf"
        assert result['text_path'] == "/tmp/test.txt"
        assert result['file_hash'] == "abc123"
        assert result['chunks'] == sample_chunks
        assert result['total_chunks'] == len(sample_chunks)

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessingService.chunk_text')
    @patch('app.services.document_processor.DocumentProcessingService.save_text_file')
    @patch('app.services.document_processor.DocumentProcessingService.process_pdf_with_docling')
    @patch('app.services.document_processor.DocumentProcessingService.save_uploaded_file')
    @patch('app.services.document_processor.settings')
    async def test_process_document_docling(
        self,
        mock_settings,
        mock_save_file,
        mock_process_pdf,
        mock_save_text,
        mock_chunk,
        sample_pdf_content,
        sample_text,
        sample_chunks
    ):
        """Test complete document processing pipeline with Docling"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Setup mocks
        mock_save_file.return_value = ("/tmp/test.pdf", "abc123")
        mock_process_pdf.return_value = sample_text
        mock_save_text.return_value = "/tmp/test.txt"
        mock_chunk.return_value = sample_chunks

        service = DocumentProcessingService()

        # Test full processing with docling
        result = await service.process_document(
            sample_pdf_content,
            "test.pdf",
            "doc-123",
            method="docling"
        )

        assert result['success'] is True
        assert result['method'] == "docling"
        mock_process_pdf.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessingService.save_uploaded_file')
    @patch('app.services.document_processor.settings')
    async def test_process_document_invalid_method(
        self,
        mock_settings,
        mock_save_file,
        sample_pdf_content
    ):
        """Test processing with invalid method"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        mock_save_file.return_value = ("/tmp/test.pdf", "abc123")

        service = DocumentProcessingService()

        # Test with invalid method
        result = await service.process_document(
            sample_pdf_content,
            "test.pdf",
            "doc-123",
            method="invalid_method"
        )

        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    @patch('app.services.document_processor.settings')
    async def test_get_document_text(self, mock_settings, sample_text):
        """Test reading document text from file"""
        from app.services.document_processor import DocumentProcessingService

        mock_settings.storage_path = "/tmp/test_storage"
        mock_settings.chunk_size = 1200
        mock_settings.chunk_overlap = 400

        # Mock file operations with proper async context manager
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_text)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_file)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        with patch('app.services.document_processor.aiofiles.open', return_value=mock_context):
            service = DocumentProcessingService()

            # Test reading
            content = await service.get_document_text("/tmp/test.txt")

            assert content == sample_text
            mock_file.read.assert_called_once()
