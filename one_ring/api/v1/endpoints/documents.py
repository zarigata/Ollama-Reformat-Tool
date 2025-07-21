"""
API endpoints for document ingestion and processing.

This module provides RESTful API endpoints for uploading and processing documents
in various formats (PDF, EPUB, DOCX, TXT, etc.) and extracting text chunks.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from one_ring.data.ingestion import (
    DocumentChunk,
    DocumentMetadata,
    DocumentProcessor,
    document_processor,
)
from one_ring.core.config import settings

# Create router
router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


# Request/Response models
class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    file: UploadFile
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the document"
    )
    chunk_size: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum size of each text chunk in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=2000,
        description="Number of characters to overlap between chunks"
    )


class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    document_id: str
    filename: str
    content_type: str
    size: int
    chunks: List[DocumentChunk]
    metadata: DocumentMetadata


class DocumentChunkResponse(BaseModel):
    """Response model for a single document chunk."""
    text: str
    metadata: DocumentMetadata
    page_number: Optional[int] = None
    chunk_number: int
    chunk_size: int
    chunk_overlap: int


class DocumentListResponse(BaseModel):
    """Response model for listing processed documents."""
    documents: List[dict]
    count: int
    total: int


# Helper functions
async def save_upload_file(upload_file: UploadFile, target_path: Path) -> None:
    """Save an uploaded file to the specified path."""
    try:
        # Create the target directory if it doesn't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file in chunks to handle large files
        with open(target_path, "wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )
    finally:
        await upload_file.seek(0)  # Reset file pointer for potential reprocessing


# API Endpoints
@router.post(
    "/upload",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description="Upload a document file and process it into text chunks.",
    responses={
        201: {"description": "Document processed successfully"},
        400: {"description": "Invalid request or unsupported file type"},
        500: {"description": "Failed to process document"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to process"),
    metadata: Optional[str] = Form("{}", description="JSON string of metadata"),
    chunk_size: int = Form(2000, ge=100, le=10000, description="Chunk size in characters"),
    chunk_overlap: int = Form(200, ge=0, le=2000, description="Overlap between chunks"),
) -> DocumentProcessResponse:
    """Upload and process a document file.
    
    Args:
        file: The document file to upload and process.
        metadata: JSON string containing additional metadata for the document.
        chunk_size: Maximum size of each text chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        
    Returns:
        Information about the processed document and its chunks.
    """
    try:
        import json
        from uuid import uuid4
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a JSON object")
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata JSON: {str(e)}",
            )
        
        # Generate a unique document ID
        document_id = str(uuid4())
        
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Save the uploaded file to the temporary location
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the document
            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            # Process the file
            chunks = processor.process_file(
                file_path=temp_file_path,
                metadata=metadata_dict,
            )
            
            # Create response
            response = DocumentProcessResponse(
                document_id=document_id,
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
                size=len(content),
                chunks=chunks,
                metadata=chunks[0].metadata if chunks else DocumentMetadata(),
            )
            
            # TODO: Store the processed document in a database or storage system
            
            return response
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.post(
    "/process-url",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a document from a URL",
    description="Download and process a document from a URL.",
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid URL or unsupported file type"},
        500: {"description": "Failed to process document"},
    },
)
async def process_document_from_url(
    url: str,
    metadata: Optional[dict] = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> DocumentProcessResponse:
    """Process a document from a URL.
    
    Args:
        url: URL of the document to process.
        metadata: Additional metadata for the document.
        chunk_size: Maximum size of each text chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        
    Returns:
        Information about the processed document and its chunks.
    """
    import tempfile
    import urllib.parse
    from urllib.parse import urlparse
    
    try:
        import requests
        from requests.exceptions import RequestException
        
        # Parse the URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL")
        
        # Extract filename from URL or generate one
        filename = os.path.basename(parsed_url.path) or f"document_{hash(url)}"
        
        # Download the file
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file to save the downloaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Process the document
                processor = DocumentProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                
                # Process the file
                chunks = processor.process_file(
                    file_path=temp_file_path,
                    metadata=metadata or {},
                )
                
                # Create response
                response = DocumentProcessResponse(
                    document_id=str(hash(url)),
                    filename=filename,
                    content_type=response.headers.get("content-type", "application/octet-stream"),
                    size=len(response.content),
                    chunks=chunks,
                    metadata=chunks[0].metadata if chunks else DocumentMetadata(),
                )
                
                # TODO: Store the processed document in a database or storage system
                
                return response
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
        
        except RequestException as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download document from URL: {str(e)}",
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document from URL {url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.get(
    "/supported-formats",
    response_model=List[str],
    summary="Get supported document formats",
    description="Get a list of supported document MIME types.",
    responses={
        200: {"description": "List of supported MIME types"},
    },
)
async def get_supported_formats() -> List[str]:
    """Get a list of supported document MIME types."""
    return document_processor.supported_formats


# Register the router in the FastAPI app
# This is typically done in the main FastAPI application file
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router)

__all__ = ["router"]
