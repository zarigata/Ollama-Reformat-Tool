from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import subprocess
import os
import shutil
from datetime import datetime
import json

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
class BookData(BaseModel):
    title: str
    author: str
    content: str
    tags: List[str] = []

class TrainingConfig(BaseModel):
    base_model: str
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3

# Storage directories
UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# API Endpoints
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads for training data"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.post("/api/train")
async def train_model(
    config: TrainingConfig,
    data_files: List[str],
    custom_prompt: Optional[str] = None
):
    """Start model training with the given configuration"""
    try:
        # Prepare training data
        data_path = os.path.join(UPLOAD_DIR, "training_data.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            for file in data_files:
                file_path = os.path.join(UPLOAD_DIR, file)
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as src:
                        f.write(src.read() + "\n")
        
        # Prepare training command
        model_name = f"ulama_{int(datetime.now().timestamp())}"
        train_cmd = [
            "ollama", "train",
            "--model", config.base_model,
            "--data", data_path,
            "--output", os.path.join(MODELS_DIR, f"{model_name}.bin"),
            "--learning-rate", str(config.learning_rate),
            "--batch-size", str(config.batch_size),
            "--epochs", str(config.num_epochs)
        ]
        
        if custom_prompt:
            train_cmd.extend(["--prompt", custom_prompt])
        
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
