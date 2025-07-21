"""
Document Processing Utilities for the One Ring platform.

This module provides functionality to process various document formats (PDF, EPUB, etc.)
and prepare them for use in training language models.
"""

import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ebooklib
import magic
import PyPDF2
from bs4 import BeautifulSoup
from ebooklib import epub
from loguru import logger
from pypandoc import convert_text
from tqdm import tqdm

from one_ring.core.config import settings
from one_ring.utils.cleanup import create_temp_file, register_temp_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum file size to process (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Supported file types and their MIME types
SUPPORTED_FILE_TYPES = {
    "application/pdf": "pdf",
    "application/epub+zip": "epub",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
}

# Regular expressions for cleaning text
MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
MULTIPLE_SPACES = re.compile(r'\s{2,}')
CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f]')


def detect_file_type(file_path: Union[str, Path]) -> Optional[str]:
    """Detect the file type using libmagic.
    
    Args:
        file_path: Path to the file to detect.
        
    Returns:
        The file type as a string, or None if the type is not supported.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large ({(file_size/1024/1024):.2f}MB > {MAX_FILE_SIZE/1024/1024}MB): {file_path}")
            return None
        
        # Detect MIME type
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(str(file_path))
        
        # Return the corresponding file extension if supported
        return SUPPORTED_FILE_TYPES.get(mime_type)
    
    except Exception as e:
        logger.error(f"Error detecting file type for {file_path}: {e}")
        return None


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        Extracted text as a string.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            
            for page in tqdm(reader.pages, desc="Extracting PDF pages"):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from PDF page: {e}")
            
            return "\n".join(text)
    
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        raise


def extract_text_from_epub(file_path: Union[str, Path]) -> str:
    """Extract text from an EPUB file.
    
    Args:
        file_path: Path to the EPUB file.
        
    Returns:
        Extracted text as a string.
    """
    try:
        book = epub.read_epub(str(file_path))
        text = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse the HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                # Extract text from all paragraphs
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text.append(p.get_text())
        
        return "\n".join(text)
    
    except Exception as e:
        logger.error(f"Error reading EPUB file {file_path}: {e}")
        raise


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """Extract text from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file.
        
    Returns:
        Extracted text as a string.
    """
    try:
        # Convert DOCX to markdown and then to plain text
        md_text = convert_text(
            str(file_path), 'md', 
            format='docx',
            outputfile=None
        )
        return md_text
    
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text to clean.
        
    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    
    # Remove control characters
    text = CONTROL_CHARS.sub(' ', text)
    
    # Normalize whitespace
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    
    # Replace multiple newlines with two newlines
    text = MULTIPLE_NEWLINS.sub('\n\n', text)
    
    # Replace multiple spaces with a single space
    text = MULTIPLE_SPACES.sub(' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def process_document(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """Process a document file and extract its text content.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        A tuple of (success, result) where success is a boolean indicating
        whether processing was successful, and result is either the extracted
        text (if successful) or an error message (if not).
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return False, f"File not found: {file_path}"
        
        # Detect file type
        file_type = detect_file_type(file_path)
        if not file_type:
            return False, f"Unsupported file type: {file_path}"
        
        # Process file based on type
        text = ""
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_type == "epub":
            text = extract_text_from_epub(file_path)
        elif file_type in ["docx", "doc"]:
            text = extract_text_from_docx(file_path)
        elif file_type in ["txt", "md"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            return False, f"Unsupported file type: {file_type}"
        
        # Clean the extracted text
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            return False, f"No text content found in file: {file_path}"
        
        return True, cleaned_text
    
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return False, f"Error processing document: {str(e)}"


def process_documents(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    recursive: bool = False
) -> Dict[str, Dict[str, str]]:
    """Process all supported documents in a directory.
    
    Args:
        input_path: Path to a file or directory containing documents.
        output_dir: Directory to save processed text files. If None, uses a temp directory.
        recursive: Whether to process subdirectories recursively.
        
    Returns:
        A dictionary mapping file paths to processing results.
    """
    input_path = Path(input_path)
    results = {}
    
    # Set up output directory
    if output_dir is None:
        output_dir = settings.DATA_DIR / "processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process a single file
    if input_path.is_file():
        success, result = process_document(input_path)
        
        if success:
            # Save the processed text to a file
            output_file = output_dir / f"{input_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            results[str(input_path)] = {
                "status": "success",
                "output_file": str(output_file),
                "size": len(result)
            }
        else:
            results[str(input_path)] = {
                "status": "error",
                "message": result
            }
    
    # Process a directory
    elif input_path.is_dir():
        # Get all files in the directory (and subdirectories if recursive)
        patterns = [f"*.{ext}" for ext in SUPPORTED_FILE_TYPES.values()]
        files = []
        
        if recursive:
            for pattern in patterns:
                files.extend(input_path.rglob(pattern))
        else:
            for pattern in patterns:
                files.extend(input_path.glob(pattern))
        
        # Process each file
        for file_path in tqdm(files, desc="Processing documents"):
            success, result = process_document(file_path)
            
            if success:
                # Create a relative path for the output file
                rel_path = file_path.relative_to(input_path)
                output_file = output_dir / rel_path.with_suffix('.txt')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the processed text
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                results[str(file_path)] = {
                    "status": "success",
                    "output_file": str(output_file),
                    "size": len(result)
                }
            else:
                results[str(file_path)] = {
                    "status": "error",
                    "message": result
                }
    
    return results


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_or_directory> [output_directory]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = process_documents(input_path, output_dir, recursive=True)
    
    # Print summary
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = len(results) - success_count
    
    print(f"\nProcessing complete:")
    print(f"  - Total files: {len(results)}")
    print(f"  - Success: {success_count}")
    print(f"  - Errors: {error_count}")
    
    if error_count > 0:
        print("\nErrors:")
        for file_path, result in results.items():
            if result["status"] == "error":
                print(f"  - {file_path}: {result['message']}")
