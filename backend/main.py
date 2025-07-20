from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Generator
import subprocess
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import hashlib
import sqlalchemy
from sqlalchemy.orm import Session

from prompt_engineer import PromptEngineer
from .database import engine, get_db
from .models import Book as DBBook, Base

app = FastAPI(title="Ulama LLM Trainer")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class BookMetadata(BaseModel):
    """Metadata for uploaded books/documents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    author: str
    description: Optional[str] = None
    language: str = "en"
    tags: List[str] = []
    source_url: Optional[HttpUrl] = None
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    word_count: int = 0
    page_count: Optional[int] = None
    is_public: bool = False
    custom_metadata: Dict[str, Any] = {}

class BookData(BookMetadata):
    """Book data including content"""
    content: str

class BookUpdate(BaseModel):
    """Model for updating book metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    custom_metadata: Optional[Dict[str, Any]] = None

class TrainingConfig(BaseModel):
    base_model: str
    learning_rate: float = Field(2e-5, ge=1e-8, le=1e-2)
    batch_size: int = Field(4, ge=1, le=64)
    num_epochs: int = Field(3, ge=1, le=100)
    context_length: int = Field(2048, ge=512, le=8192)
    lora_rank: int = Field(8, ge=1, le=128)
    lora_alpha: int = Field(16, ge=1, le=256)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    use_unlock_prompt: bool = False
    custom_prompt_template: Optional[str] = None
    train_test_split: float = Field(0.9, ge=0.5, le=1.0)
    early_stopping_patience: int = Field(3, ge=1, le=10)
    warmup_steps: int = Field(100, ge=0, le=1000)

# Storage directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# API Endpoints
@app.post("/upload/", response_model=BookMetadata, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(None),
    author: str = Form("Unknown"),
    description: str = Form(""),
    tags: str = Form(""),
    is_public: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Handle file uploads with metadata"""
    # Generate a unique filename
    file_ext = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    # Ensure upload directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Generate metadata
    file_hash = calculate_file_hash(file_path)
    
    # Create database record
    db_book = DBBook(
        id=file_id,
        title=title or Path(file.filename).stem,
        author=author,
        content="",  # For large files, we might want to store content separately
        file_path=str(file_path.relative_to(BASE_DIR)),
        file_hash=file_hash,
        file_size=os.path.getsize(file_path),
        mime_type=file.content_type or "application/octet-stream"
    )
    
    try:
        db.add(db_book)
        db.commit()
        db.refresh(db_book)
    except Exception as e:
        db.rollback()
        # Clean up the uploaded file if database operation fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    
    # Convert to response model
    return BookMetadata(
        id=db_book.id,
        title=db_book.title,
        author=db_book.author,
        description=description,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        is_public=is_public,
        file_path=db_book.file_path,
        file_hash=db_book.file_hash,
        created_at=db_book.created_at.isoformat(),
        updated_at=db_book.updated_at.isoformat() if db_book.updated_at else None,
        file_size=db_book.file_size,
        mime_type=db_book.mime_type,
        status="uploaded"
    )

@app.get("/books/", response_model=List[BookMetadata])
async def list_books(
    search: Optional[str] = None,
    tags: List[str] = Query(None),
    author: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all books with optional filtering"""
    query = db.query(DBBook)
    
    # Apply filters
    if search:
        search = f"%{search.lower()}%"
        query = query.filter(sqlalchemy.or_(
            DBBook.title.ilike(search),
            DBBook.author.ilike(search)
        ))
    
    if author:
        query = query.filter(DBBook.author.ilike(f"%{author}%"))
    
    # Note: For tags, we'd need a separate table for many-to-many relationship
    # This is simplified for the example
    
    # Apply pagination
    books = query.offset(offset).limit(limit).all()
    
    # Convert to response model
    return [
        BookMetadata(
            id=book.id,
            title=book.title,
            author=book.author,
            description="",  # Add if you have a description field
            tags=[],  # Add if you implement tags
            is_public=True,  # Default value
            file_path=book.file_path,
            file_hash=book.file_hash,
            created_at=book.created_at.isoformat(),
            updated_at=book.updated_at.isoformat() if book.updated_at else None,
            file_size=book.file_size,
            mime_type=book.mime_type,
            status="processed" if book.content else "uploaded"
        )
        for book in books
    ]

@app.get("/books/{book_id}", response_model=BookMetadata)
async def get_book(book_id: str, db: Session = Depends(get_db)):
    """Get a single book by ID"""
    book = db.query(DBBook).filter(DBBook.id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found"
        )
    
    return BookMetadata(
        id=book.id,
        title=book.title,
        author=book.author,
        description="",  # Add if you have a description field
        tags=[],  # Add if you implement tags
        is_public=True,  # Default value
        file_path=book.file_path,
        file_hash=book.file_hash,
        created_at=book.created_at.isoformat(),
        updated_at=book.updated_at.isoformat() if book.updated_at else None,
        file_size=book.file_size,
        mime_type=book.mime_type,
        status="processed" if book.content else "uploaded"
    )

@app.put("/books/{book_id}", response_model=BookMetadata)
async def update_book(
    book_id: str, 
    update_data: BookUpdate,
    db: Session = Depends(get_db)
):
    """Update book metadata"""
    book = db.query(DBBook).filter(DBBook.id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found"
        )
    
    # Update fields from the request
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        if hasattr(book, field):
            setattr(book, field, value)
    
    try:
        db.commit()
        db.refresh(book)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating book: {str(e)}"
        )
    
    # Return the updated book
    return BookMetadata(
        id=book.id,
        title=book.title,
        author=book.author,
        description="",  # Add if you have a description field
        tags=[],  # Add if you implement tags
        is_public=True,  # Default value
        file_path=book.file_path,
        file_hash=book.file_hash,
        created_at=book.created_at.isoformat(),
        updated_at=book.updated_at.isoformat() if book.updated_at else None,
        file_size=book.file_size,
        mime_type=book.mime_type,
        status="processed" if book.content else "uploaded"
    )

@app.delete("/books/{book_id}")
async def delete_book(book_id: str, db: Session = Depends(get_db)):
    """Delete a book and its associated file"""
    book = db.query(DBBook).filter(DBBook.id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found"
        )
    
    file_path = BASE_DIR / book.file_path
    
    try:
        # Delete the database record
        db.delete(book)
        db.commit()
        
        # Delete the file
        if file_path.exists():
            file_path.unlink()
            
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting book: {str(e)}"
        )
    
    return {
        "status": "success", 
        "message": "Book deleted successfully"
    }

@app.post("/api/train")
async def train_model(
    config: TrainingConfig,
    data_files: List[str],
    custom_prompt: Optional[str] = None,
    use_unlock: bool = False,
    book_ids: Optional[List[str]] = None,
):
    """Start model training with the given configuration"""
    try:
        # Prepare training data
        data_path = UPLOAD_DIR / f"training_data_{int(datetime.now().timestamp())}.jsonl"
        
        # Load content from book IDs if provided
        book_contents = []
        if book_ids:
            db = load_books_db()
            for book in db["books"]:
                if book["id"] in book_ids and os.path.exists(book["file_path"]):
                    with open(book["file_path"], 'r', encoding='utf-8') as f:
                        book_contents.append({
                            "content": f.read(),
                            "metadata": {k: v for k, v in book.items() if k != "content"}
                        })
        
        # Load content from direct file uploads
        file_contents = []
        for file in data_files:
            file_path = UPLOAD_DIR / file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_contents.append(f.read())
        
        # Combine and prepare training examples
        with open(data_path, "w", encoding="utf-8") as f:
            # Add book content as context
            for book in book_contents:
                # Prepare prompt using the prompt engineer
                prompt_data = PromptEngineer.prepare_prompt(
                    user_prompt="",  # Will be filled during training
                    context=book["content"],
                    template_type="unlocked" if use_unlock else "default"
                )
                example = {
                    "prompt": prompt_data["formatted_prompt"],
                    "completion": "",  # Will be filled during training
                    "metadata": book["metadata"]
                }
                f.write(json.dumps(example) + "\n")
            
            # Add direct file content
            for content in file_contents:
                example = {
                    "prompt": content,
                    "completion": "",
                    "metadata": {"source": "direct_upload"}
                }
                f.write(json.dumps(example) + "\n")
        
        # Prepare model name and output path
        timestamp = int(datetime.now().timestamp())
        model_name = f"ulama_{timestamp}"
        output_path = MODELS_DIR / f"{model_name}.bin"
        
        # Prepare training command with advanced parameters
        train_cmd = [
            "ollama", "train",
            "--model", config.base_model,
            "--data", str(data_path),
            "--output", str(output_path),
            "--learning-rate", str(config.learning_rate),
            "--batch-size", str(config.batch_size),
            "--epochs", str(config.num_epochs),
            "--context-length", str(config.context_length),
            "--lora-rank", str(config.lora_rank),
            "--lora-alpha", str(config.lora_alpha),
            "--lora-dropout", str(config.lora_dropout),
            "--warmup-steps", str(config.warmup_steps),
            "--early-stopping-patience", str(config.early_stopping_patience)
        ]
        
        # Add unlock prompt if enabled
        if use_unlock or config.use_unlock_prompt:
            unlock_prompt = custom_prompt or PromptEngineer.UNLOCK_TEMPLATE
            train_cmd.extend(["--prompt", unlock_prompt])
        elif custom_prompt:
            train_cmd.extend(["--prompt", custom_prompt])
            
        # Add train/test split if specified
        if hasattr(config, 'train_test_split') and 0 < config.train_test_split < 1.0:
            train_cmd.extend(["--train-test-split", str(config.train_test_split)])
        
        # Start training
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return {
            "status": "training_started",
            "model_name": model_name,
            "pid": process.pid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all available models"""
    models = []
    if os.path.exists(MODELS_DIR):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.bin')]
    return {"models": models}

@app.get("/api/ollama/models")
async def list_ollama_models():
    """List all available Ollama models"""
    try:
        result = subprocess.run(
            ["ollama", "list", "--json"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return {"models": json.loads(result.stdout)}
        return {"models": []}
    except Exception as e:
        return {"models": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
