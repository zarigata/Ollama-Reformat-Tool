"""
Ollama Integration for the One Ring platform.

This module provides functionality to interact with Ollama API for model management
and inference, allowing seamless integration with the One Ring platform.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl

from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.utils.cleanup import register_cleanup_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaModelFormat(str, Enum):
    """Supported model formats for Ollama."""
    GGUF = "gguf"
    GGML = "ggml"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"


class OllamaModelFile(BaseModel):
    """Represents a model file in the Ollama format."""
    filename: str
    size: int
    digest: str
    format: OllamaModelFormat
    params: Dict[str, Any] = Field(default_factory=dict)
    system: Optional[str] = None
    template: Optional[str] = None
    license: Optional[str] = None
    modelfile: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OllamaModel(BaseModel):
    """Represents an Ollama model."""
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    files: List[OllamaModelFile] = Field(default_factory=list)
    
    @property
    def parameter_count(self) -> int:
        """Get the number of parameters in the model."""
        return self.details.get("parameter_size", "N/A")
    
    @property
    def quantization_level(self) -> str:
        """Get the quantization level of the model."""
        return self.details.get("quantization_level", "N/A")


class OllamaClientConfig(BaseModel):
    """Configuration for the Ollama client."""
    base_url: HttpUrl = Field(default=settings.OLLAMA_BASE_URL)
    timeout: int = Field(default=settings.OLLAMA_TIMEOUT, ge=10, le=600)
    verify_tls: bool = True
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            HttpUrl: lambda v: str(v),
        }


class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(self, config: Optional[OllamaClientConfig] = None):
        """Initialize the Ollama client.
        
        Args:
            config: Configuration for the client. If None, uses default settings.
        """
        self.config = config or OllamaClientConfig()
        self.session = self._create_session()
        
        # Register cleanup handler
        register_cleanup_handler(self.close)
        
        logger.info(f"Initialized Ollama client with base URL: {self.config.base_url}")
    
    def _create_session(self) -> requests.Session:
        """Create and configure a requests session."""
        session = requests.Session()
        session.timeout = self.config.timeout
        session.verify = self.config.verify_tls
        
        # Add API key if provided
        if self.config.api_key:
            session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}"
            })
        
        return session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make a request to the Ollama API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments to pass to requests.request()
            
        Returns:
            The response from the API.
            
        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def list_models(self) -> List[OllamaModel]:
        """List all available models.
        
        Returns:
            A list of OllamaModel objects.
        """
        response = self._request("GET", "/api/tags")
        data = response.json()
        
        return [
            OllamaModel(**model_data)
            for model_data in data.get("models", [])
        ]
    
    def get_model(self, model_name: str) -> Optional[OllamaModel]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model to get information about.
            
        Returns:
            An OllamaModel object if found, None otherwise.
        """
        try:
            response = self._request("GET", f"/api/show", params={"name": model_name})
            return OllamaModel(**response.json())
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from the Ollama registry.
        
        Args:
            model_name: Name of the model to pull.
            
        Returns:
            Status information about the pull operation.
        """
        response = self._request("POST", "/api/pull", json={"name": model_name}, stream=True)
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def push_model(self, model_name: str) -> Dict[str, Any]:
        """Push a model to the Ollama registry.
        
        Args:
            model_name: Name of the model to push.
            
        Returns:
            Status information about the push operation.
        """
        response = self._request("POST", "/api/push", json={"name": model_name}, stream=True)
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate text using a model.
        
        Args:
            prompt: The prompt to generate text from.
            model: The model to use for generation. If None, uses the default model.
            system: System message to set the behavior of the model.
            template: The full prompt template to use for the model.
            context: The context parameter returned from a previous call to generate.
            stream: If True, returns a generator that yields response chunks.
            raw: If True, no formatting is applied to the prompt.
            format: The format to return the response in (e.g., json).
            options: Additional model parameters.
            
        Returns:
            A dictionary containing the generated text and other information.
        """
        if model is None and self.config.default_model is None:
            raise ValueError("No model specified and no default model set in config")
        
        payload = {
            "model": model or self.config.default_model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }
        
        if system is not None:
            payload["system"] = system
        if template is not None:
            payload["template"] = template
        if context is not None:
            payload["context"] = context
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options
        
        if stream:
            response = self._request("POST", "/api/generate", json=payload, stream=True)
            
            def response_generator():
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            
            return response_generator()
        else:
            response = self._request("POST", "/api/generate", json=payload)
            return response.json()
    
    def create_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an embedding for the given text.
        
        Args:
            text: The text to create an embedding for.
            model: The model to use for creating the embedding.
            options: Additional model parameters.
            
        Returns:
            A dictionary containing the embedding and other information.
        """
        if model is None and self.config.default_model is None:
            raise ValueError("No model specified and no default model set in config")
        
        payload = {
            "model": model or self.config.default_model,
            "prompt": text,
        }
        
        if options is not None:
            payload["options"] = options
        
        response = self._request("POST", "/api/embeddings", json=payload)
        return response.json()
    
    def close(self) -> None:
        """Close the client and release resources."""
        if hasattr(self, 'session'):
            self.session.close()
            logger.info("Ollama client closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Create a global Ollama client instance
_ollama_client = None

def get_ollama_client() -> OllamaClient:
    """Get or create a global Ollama client instance.
    
    Returns:
        An instance of OllamaClient.
    """
    global _ollama_client
    
    if _ollama_client is None:
        _ollama_client = OllamaClient()
        
        # Register cleanup
        import atexit
        atexit.register(_ollama_client.close)
    
    return _ollama_client


__all__ = [
    "OllamaClient",
    "OllamaModel",
    "OllamaModelFile",
    "OllamaModelFormat",
    "OllamaClientConfig",
    "get_ollama_client",
]
