"""
Model Service for the One Ring platform.

This module provides high-level functionality for managing and interacting with
language models, including local models and models hosted on Ollama.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl

from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.integrations.ollama import (
    OllamaClient, 
    OllamaModel, 
    get_ollama_client
)
from one_ring.models.manager import ModelConfig, model_manager
from one_ring.utils.cleanup import register_cleanup_handler


class ModelType(str, Enum):
    """Type of model (local or Ollama)."""
    LOCAL = "local"
    OLLAMA = "ollama"


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str = Field(..., description="Name of the model")
    model_type: ModelType = Field(..., description="Type of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    size: Optional[int] = Field(None, description="Size of the model in bytes")
    parameters: Optional[int] = Field(None, description="Number of parameters")
    quantization: Optional[str] = Field(None, description="Quantization level")
    format: Optional[str] = Field(None, description="Model format")
    modified_at: Optional[str] = Field(None, description="Last modified timestamp")
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            ModelType: lambda v: v.value,
        }


class ModelService:
    """Service for managing and interacting with language models."""
    
    def __init__(self):
        """Initialize the model service."""
        self.ollama_client = get_ollama_client()
        self.loaded_models: Dict[str, Any] = {}
        self.default_model: Optional[str] = None
        
        # Register cleanup handler
        register_cleanup_handler(self.cleanup)
        
        logger.info("Model service initialized")
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """List all available models.
        
        Args:
            model_type: Type of models to list. If None, lists all models.
            
        Returns:
            A list of ModelInfo objects.
        """
        models = []
        
        # List local models
        if model_type is None or model_type == ModelType.LOCAL:
            # TODO: Implement local model discovery
            pass
        
        # List Ollama models
        if model_type is None or model_type == ModelType.OLLAMA:
            try:
                ollama_models = self.ollama_client.list_models()
                for model in ollama_models:
                    models.append(ModelInfo(
                        name=model.name,
                        model_type=ModelType.OLLAMA,
                        description=model.details.get("family", ""),
                        size=model.size,
                        parameters=model.details.get("parameter_size"),
                        quantization=model.details.get("quantization_level"),
                        format=model.details.get("format"),
                        modified_at=model.modified_at,
                    ))
            except Exception as e:
                logger.error(f"Failed to list Ollama models: {e}")
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model to get information about.
            
        Returns:
            A ModelInfo object if found, None otherwise.
        """
        # First check if it's a local model
        # TODO: Implement local model check
        
        # Then check Ollama models
        try:
            model = self.ollama_client.get_model(model_name)
            if model:
                return ModelInfo(
                    name=model.name,
                    model_type=ModelType.OLLAMA,
                    description=model.details.get("family", ""),
                    size=model.size,
                    parameters=model.details.get("parameter_size"),
                    quantization=model.details.get("quantization_level"),
                    format=model.details.get("format"),
                    modified_at=model.modified_at,
                )
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
        
        return None
    
    def load_model(self, model_name: str, model_type: ModelType = ModelType.OLLAMA, **kwargs) -> bool:
        """Load a model for inference.
        
        Args:
            model_name: Name of the model to load.
            model_type: Type of the model (local or Ollama).
            **kwargs: Additional model-specific parameters.
            
        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        try:
            if model_type == ModelType.OLLAMA:
                # For Ollama models, we don't need to load them explicitly
                # Just verify that the model exists
                model_info = self.ollama_client.get_model(model_name)
                if not model_info:
                    logger.error(f"Ollama model not found: {model_name}")
                    return False
                
                self.loaded_models[model_name] = {
                    "type": ModelType.OLLAMA,
                    "info": model_info,
                }
                
                # Set as default model if not set
                if not self.default_model:
                    self.default_model = model_name
                
                logger.info(f"Ollama model loaded: {model_name}")
                return True
                
            elif model_type == ModelType.LOCAL:
                # For local models, use the model manager
                config = ModelConfig(
                    model_name_or_path=model_name,
                    **kwargs
                )
                
                model, tokenizer = model_manager.load_model(config)
                if model and tokenizer:
                    self.loaded_models[model_name] = {
                        "type": ModelType.LOCAL,
                        "model": model,
                        "tokenizer": tokenizer,
                        "config": config,
                    }
                    
                    # Set as default model if not set
                    if not self.default_model:
                        self.default_model = model_name
                    
                    logger.info(f"Local model loaded: {model_name}")
                    return True
                
                return False
            
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model.
        
        Args:
            model_name: Name of the model to unload.
            
        Returns:
            True if the model was unloaded successfully, False otherwise.
        """
        if model_name in self.loaded_models:
            model_info = self.loaded_models.pop(model_name)
            
            # Clear default model if it's the one being unloaded
            if self.default_model == model_name:
                self.default_model = None
                
                # Set another model as default if available
                if self.loaded_models:
                    self.default_model = next(iter(self.loaded_models.keys()))
            
            logger.info(f"Model unloaded: {model_name}")
            return True
        
        logger.warning(f"Model not loaded: {model_name}")
        return False
    
    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            model_name: Name of the model to use. If None, uses the default model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stream: Whether to stream the response.
            **kwargs: Additional generation parameters.
            
        Returns:
            The generated text, or a generator that yields text chunks if streaming.
            
        Raises:
            ValueError: If no model is loaded or the specified model is not found.
        """
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("No model specified and no default model set")
        
        if model_name not in self.loaded_models:
            # Try to load the model if it's not already loaded
            if not self.load_model(model_name):
                raise ValueError(f"Model not found or could not be loaded: {model_name}")
        
        model_info = self.loaded_models[model_name]
        
        if model_info["type"] == ModelType.OLLAMA:
            # Use Ollama API for generation
            response = self.ollama_client.generate(
                prompt=prompt,
                model=model_name,
                stream=stream,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    **kwargs
                }
            )
            
            if stream:
                def response_generator():
                    for chunk in response:
                        if "response" in chunk:
                            yield chunk["response"]
                        elif "error" in chunk:
                            logger.error(f"Error in Ollama response: {chunk['error']}")
                            break
                
                return response_generator()
            else:
                return response.get("response", "")
                
        elif model_info["type"] == ModelType.LOCAL:
            # Use local model for generation
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Encode the input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # Adjust based on model context window
                return_token_type_ids=False,
            ).to(model.device)
            
            # Generate text
            if stream:
                def response_generator():
                    from transformers import TextIteratorStreamer
                    from threading import Thread
                    
                    streamer = TextIteratorStreamer(
                        tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True
                    )
                    
                    generation_kwargs = {
                        **inputs,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "do_sample": True,
                        "streamer": streamer,
                        **kwargs
                    }
                    
                    # Start generation in a separate thread
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # Stream the generated text
                    for new_text in streamer:
                        yield new_text
                    
                    thread.join()
                
                return response_generator()
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    **kwargs
                )
                
                # Decode the generated text
                generated_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                return generated_text
        
        else:
            raise ValueError(f"Unsupported model type: {model_info['type']}")
    
    def create_embedding(
        self,
        text: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Create an embedding for the given text.
        
        Args:
            text: The text to create an embedding for.
            model_name: Name of the model to use. If None, uses the default model.
            **kwargs: Additional parameters for the embedding model.
            
        Returns:
            A list of floats representing the embedding.
            
        Raises:
            ValueError: If no model is loaded or the specified model is not found.
        """
        model_name = model_name or self.default_model
        if not model_name:
            raise ValueError("No model specified and no default model set")
        
        if model_name not in self.loaded_models:
            # Try to load the model if it's not already loaded
            if not self.load_model(model_name):
                raise ValueError(f"Model not found or could not be loaded: {model_name}")
        
        model_info = self.loaded_models[model_name]
        
        if model_info["type"] == ModelType.OLLAMA:
            # Use Ollama API for embeddings
            response = self.ollama_client.create_embedding(
                text=text,
                model=model_name,
                options=kwargs
            )
            
            return response.get("embedding", [])
            
        elif model_info["type"] == ModelType.LOCAL:
            # Use local model for embeddings
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Encode the input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Adjust based on model context window
                return_token_type_ids=False,
            ).to(model.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use the last hidden state as the embedding
                last_hidden_state = outputs.hidden_states[-1]
                
                # Average the token embeddings to get a single vector
                embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
                
                return embedding
        
        else:
            raise ValueError(f"Unsupported model type: {model_info['type']}")
    
    def cleanup(self) -> None:
        """Clean up resources used by the model service."""
        # Unload all loaded models
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        logger.info("Model service cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


# Create a global model service instance
model_service = ModelService()

# Register cleanup handler
import atexit
atexit.register(model_service.cleanup)

__all__ = ["ModelService", "ModelInfo", "ModelType", "model_service"]
