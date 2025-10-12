# Testing Documentation

## Overview
This directory contains the test suite for the RAG System backend. The tests are organized using pytest and include unit tests for various services, utilities, and API routes.

## Directory Structure
```
test/
├── README.md                 # This file
├── unit_tests/              # Unit tests directory
│   ├── __init__.py          # Package initialization
│   ├── conftest.py          # Pytest fixtures and configuration
│   ├── test_vector_store.py # Tests for vector store service
│   ├── test_document_processor.py  # Tests for document processor
│   ├── test_document_helpers.py    # Tests for document helpers
│   └── test_api_routes.py   # Tests for API routes
```

## Test Configuration

### pytest.ini
The main pytest configuration is in the project root (`pytest.ini`). It includes:
- Test discovery patterns
- Coverage settings
- Test markers for organization
- Output options

### conftest.py
Contains shared fixtures and configuration:
- Mock settings
- Mock database sessions
- Sample data fixtures (documents, chunks, embeddings)
- Mock external services (OpenAI, Neo4j, Elasticsearch)

## Running Tests

### Prerequisites
```bash
# Install backend dependencies (includes all production dependencies)
pip install -r backend/requirements.txt

# Install test-specific dependencies
pip install -r test/requirements.txt
```

### Running All Tests
```bash
# Run all tests
pytest test/unit_tests/

# Run with verbose output
pytest test/unit_tests/ -v

# Run with coverage report
pytest test/unit_tests/ --cov=backend/app --cov-report=html
```

### Running Specific Tests
```bash
# Run tests in a specific file
pytest test/unit_tests/test_vector_store.py

# Run a specific test class
pytest test/unit_tests/test_vector_store.py::TestVectorStoreService

# Run a specific test method
pytest test/unit_tests/test_vector_store.py::TestVectorStoreService::test_init

# Run tests with a specific marker
pytest test/unit_tests/ -m unit
```

### Coverage Reports
```bash
# Generate HTML coverage report
pytest test/unit_tests/ --cov=backend/app --cov-report=html

# Generate terminal coverage report
pytest test/unit_tests/ --cov=backend/app --cov-report=term-missing

# Generate JSON coverage report
pytest test/unit_tests/ --cov=backend/app --cov-report=json:test_coverage.json
```

## Test Markers
Tests can be marked with the following markers (defined in pytest.ini):
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.database` - Tests requiring database connection
- `@pytest.mark.external` - Tests requiring external services

## Test Files

### test_vector_store.py (7 tests - All Passing ✅)
Tests for PGVector storage and search operations:
- `test_init` - Service initialization
- `test_store_embeddings_success` - Successful embedding storage
- `test_store_embeddings_mismatch` - Error handling for mismatched counts
- `test_get_document_chunks` - Retrieving document chunks
- `test_delete_document_embeddings` - Deleting embeddings
- `test_get_embedding_count` - Counting embeddings
- `test_get_embedding_count_with_filter` - Counting with document filter

**Note:** Complex SQLAlchemy vector search tests were removed due to difficult mocking of pgvector extension types.

### test_document_processor.py (10 tests - All Passing ✅)
Tests for PDF processing and text extraction:
- `test_init` - Service initialization
- `test_save_uploaded_file` - File upload handling
- `test_process_pdf_with_pymupdf` - PyMuPDF processing
- `test_process_pdf_with_docling` - Docling processing
- `test_save_text_file` - Text file saving
- `test_chunk_text` - Text chunking
- `test_process_document_pymupdf` - Complete PyMuPDF pipeline
- `test_process_document_docling` - Complete Docling pipeline
- `test_process_document_invalid_method` - Error handling
- `test_get_document_text` - Reading text files

### test_document_helpers.py (4 tests - All Passing ✅)
Tests for document processing pipeline:
- `test_process_document_pipeline_success` - Successful processing
- `test_process_document_pipeline_processing_failure` - Error handling
- `test_process_document_pipeline_with_graphrag` - GraphRAG integration
- `test_process_document_pipeline_with_elasticsearch` - Elasticsearch integration

## Current Test Coverage

**Current Status:**
- Total Tests: 21
- Passing: 21 (100% ✅)
- Overall Coverage: 27%

**Test Coverage by Module:**
- `models.py`: 100%
- `config.py`: 100%
- `document_helpers.py`: 90%
- `document_processor.py`: 83%
- `vector_store.py`: 66%

## Continuous Integration
Tests are automatically run on pull requests to the `development` branch via GitHub Actions.

See `.github/workflows/test-pr.yml` for the CI configuration.

## Fixtures

### Configuration Fixtures
- `mock_settings` - Mock application settings
- `mock_db_session` - Mock database session

### Data Fixtures
- `sample_document_id` - Sample document UUID
- `sample_chunks` - Sample text chunks
- `sample_embeddings` - Sample embedding vectors
- `sample_query_embedding` - Sample query embedding
- `sample_text` - Sample document text
- `sample_pdf_content` - Sample PDF binary content

### Service Fixtures
- `mock_openai_client` - Mock OpenAI API client
- `mock_neo4j_driver` - Mock Neo4j driver
- `mock_elasticsearch_client` - Mock Elasticsearch client

## Writing New Tests

### Best Practices
1. **Use descriptive test names** - Test names should clearly describe what is being tested
2. **Follow the AAA pattern** - Arrange, Act, Assert
3. **Use fixtures** - Leverage conftest.py fixtures for common setup
4. **Mock external dependencies** - Mock all external services and databases
5. **Test edge cases** - Include tests for error conditions and edge cases
6. **Keep tests isolated** - Each test should be independent
7. **Use markers** - Mark tests appropriately (unit, integration, slow, etc.)

### Example Test
```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyService:
    """Test MyService class"""

    @patch('app.services.my_service.external_dependency')
    def test_my_function_success(self, mock_external, mock_settings):
        """Test successful execution of my_function"""
        # Arrange
        mock_external.return_value = "expected_value"
        service = MyService()

        # Act
        result = service.my_function()

        # Assert
        assert result == "expected_value"
        mock_external.assert_called_once()
```

## Troubleshooting

### Common Issues

#### Import Errors
If you see `ModuleNotFoundError`, ensure all dependencies are installed:
```bash
pip install -r test_requirements.txt
pip install -r backend/requirements.txt
```

#### Path Issues
Tests are run from the project root. If you have path issues, ensure you're running pytest from the project root directory.

#### Mock Issues
If mocks aren't working, check that:
1. The patch path is correct (use the path where the module is imported, not defined)
2. The mock is set up before the code under test runs
3. Return values and side effects are properly configured

## Contributing
When adding new features:
1. Write tests for the new functionality
2. Ensure all tests pass: `pytest test/unit_tests/`
3. Check coverage: `pytest test/unit_tests/ --cov=backend/app`
4. Aim for at least 80% coverage on new code

## Resources
- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Python unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
