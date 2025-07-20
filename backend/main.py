from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import subprocess
import os
import shutil
import uuid
from datetime import datetime
import json
from pathlib import Path
import hashlib

from prompt_engineer import PromptEngineer

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
BOOKS_DB = BASE_DIR / "books_db.json"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Initialize books database if it doesn't exist
if not BOOKS_DB.exists():
    with open(BOOKS_DB, 'w') as f:
        json.dump({"books": [], "next_id": 1}, f)

# Load books database
def load_books_db():
    with open(BOOKS_DB, 'r') as f:
        return json.load(f)

def save_books_db(data):
    with open(BOOKS_DB, 'w') as f:
        json.dump(data, f, default=str)

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# API Endpoints
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(None),
    author: str = Form("Unknown"),
    description: str = Form(""),
    tags: str = Form(""),
    is_public: bool = Form(False)
):
    """Handle file uploads with metadata"""
    try:
        # Save the file
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Get word count (approximate)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            word_count = len(content.split())
        
        # Create book metadata
        book_data = {
            "id": str(uuid.uuid4()),
            "title": title or Path(file.filename).stem,
            "author": author,
            "description": description,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "file_path": str(file_path),
            "file_name": file.filename,
            "file_hash": file_hash,
            "file_size": os.path.getsize(file_path),
            "word_count": word_count,
            "upload_date": datetime.utcnow().isoformat(),
            "is_public": is_public,
            "custom_metadata": {}
        }
        
        # Save to database
        db = load_books_db()
        db["books"].append(book_data)
        save_books_db(db)
        
        return {"status": "success", "book": book_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books")
async def list_books(
    search: Optional[str] = None,
    tags: List[str] = Query(None),
    author: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all books with optional filtering"""
    try:
        db = load_books_db()
        books = db["books"]
        
        # Apply filters
        if search:
            search = search.lower()
            books = [b for b in books if 
                    search in b.get("title", "").lower() or 
                    search in b.get("description", "").lower() or
                    search in b.get("content", "").lower()]
        
        if tags:
            books = [b for b in books if any(tag in b.get("tags", []) for tag in tags)]
            
        if author:
            author = author.lower()
            books = [b for b in books if author in b.get("author", "").lower()]
        
        # Pagination
        total = len(books)
        books = books[offset:offset + limit]
        
        return {
            "total": total,
            "count": len(books),
            "offset": offset,
            "limit": limit,
            "books": books
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books/{book_id}")
async def get_book(book_id: str):
    """Get a single book by ID"""
    try:
        db = load_books_db()
        for book in db["books"]:
            if book["id"] == book_id:
                return book
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/books/{book_id}")
async def update_book(book_id: str, update_data: BookUpdate):
    """Update book metadata"""
    try:
        db = load_books_db()
        for book in db["books"]:
            if book["id"] == book_id:
                # Update fields that are provided
                update_dict = update_data.dict(exclude_unset=True)
                book.update(update_dict)
                save_books_db(db)
                return {"status": "success", "book": book}
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/books/{book_id}")
async def delete_book(book_id: str):
    """Delete a book and its associated file"""
    try:
        db = load_books_db()
        for i, book in enumerate(db["books"]):
            if book["id"] == book_id:
                # Delete the file if it exists
                file_path = book.get("file_path")
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                # Remove from database
                db["books"].pop(i)
                save_books_db(db)
                return {"status": "success", "message": "Book deleted"}
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
