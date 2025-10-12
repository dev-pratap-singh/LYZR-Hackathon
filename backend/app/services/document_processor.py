"""
Document Processing Service
Handles PDF processing, text extraction, and chunking
Supports both PyMuPDF and Docling for comparison
"""
import os
import logging
import hashlib
from typing import List, Dict, Tuple, Literal
from pathlib import Path
import aiofiles

import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.models import Document
from app.database import get_postgres_db

logger = logging.getLogger(__name__)

ProcessingMethod = Literal["pymupdf", "docling"]


class DocumentProcessingService:
    """Service for processing PDF documents"""

    def __init__(self):
        self.storage_path = Path(settings.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def save_uploaded_file(self, file_content: bytes, filename: str) -> Tuple[str, str]:
        """
        Save uploaded PDF file to storage

        Args:
            file_content: Binary content of the file
            filename: Original filename

        Returns:
            Tuple of (file_path, file_hash)
        """
        try:
            # Generate unique filename
            file_hash = hashlib.sha256(file_content).hexdigest()[:16]
            safe_filename = f"{file_hash}_{filename}"
            file_path = self.storage_path / safe_filename

            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)

            logger.info(f"Saved file: {safe_filename}")
            return str(file_path), file_hash

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise

    async def process_pdf_with_pymupdf(self, pdf_path: str) -> str:
        """
        Process PDF using PyMuPDF (fitz) and extract text

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Processing PDF with PyMuPDF: {pdf_path}")

            doc = fitz.open(pdf_path)
            text = ""
            page_count = len(doc)

            # Extract text from each page
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += f"\n\n=== Page {page_num + 1} ===\n\n{page_text}"

            doc.close()

            logger.info(f"PyMuPDF: Successfully extracted {len(text)} characters from {page_count} pages")
            return text

        except Exception as e:
            logger.error(f"Error processing PDF with PyMuPDF: {e}")
            raise

    async def process_pdf_with_docling(self, pdf_path: str) -> str:
        """
        Process PDF using Docling and convert to markdown

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content in markdown format
        """
        try:
            logger.info(f"Processing PDF with Docling: {pdf_path}")

            # Initialize Docling converter
            converter = DocumentConverter()

            # Convert PDF to document
            result = converter.convert(pdf_path)

            # Export to markdown
            text = result.document.export_to_markdown()

            logger.info(f"Docling: Successfully extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error processing PDF with Docling: {e}")
            raise

    async def save_text_file(self, text_content: str, original_filename: str, file_hash: str) -> str:
        """
        Save extracted text to .txt file

        Args:
            text_content: Extracted text
            original_filename: Original PDF filename
            file_hash: Hash of the original file

        Returns:
            Path to saved text file
        """
        try:
            base_name = Path(original_filename).stem
            text_filename = f"{file_hash}_{base_name}.txt"
            text_path = self.storage_path / text_filename

            async with aiofiles.open(text_path, 'w', encoding='utf-8') as f:
                await f.write(text_content)

            logger.info(f"Saved text file: {text_filename}")
            return str(text_path)

        except Exception as e:
            logger.error(f"Error saving text file: {e}")
            raise

    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Chunk text into smaller segments

        Args:
            text: Full text content

        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Create chunk objects with metadata
            chunk_objects = []
            for idx, chunk in enumerate(chunks):
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                chunk_objects.append({
                    'index': idx,
                    'text': chunk,
                    'size': len(chunk),
                    'hash': chunk_hash,
                    'chunk_metadata': {
                        'chunk_index': idx,
                        'total_chunks': len(chunks)
                    }
                })

            logger.info(f"Created {len(chunk_objects)} chunks")
            return chunk_objects

        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str,
        method: ProcessingMethod = "pymupdf"
    ) -> Dict:
        """
        Complete document processing pipeline

        Args:
            file_content: Binary PDF content
            filename: Original filename
            document_id: UUID of the document record
            method: Processing method to use ("pymupdf" or "docling")

        Returns:
            Dictionary with processing results
        """
        try:
            # Step 1: Save PDF
            pdf_path, file_hash = await self.save_uploaded_file(file_content, filename)

            # Step 2: Extract text based on method
            if method == "pymupdf":
                text_content = await self.process_pdf_with_pymupdf(pdf_path)
            elif method == "docling":
                text_content = await self.process_pdf_with_docling(pdf_path)
            else:
                raise ValueError(f"Unknown processing method: {method}")

            # Step 3: Save text file
            text_path = await self.save_text_file(text_content, filename, file_hash)

            # Step 4: Chunk text
            chunks = self.chunk_text(text_content)

            return {
                'success': True,
                'method': method,
                'pdf_path': pdf_path,
                'text_path': text_path,
                'file_hash': file_hash,
                'text_length': len(text_content),
                'chunks': chunks,
                'total_chunks': len(chunks)
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_document_text(self, text_path: str) -> str:
        """
        Read text content from saved file

        Args:
            text_path: Path to text file

        Returns:
            Text content
        """
        try:
            async with aiofiles.open(text_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise
