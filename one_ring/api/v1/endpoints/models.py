"""
API endpoints for model management and inference.

This module provides RESTful API endpoints for interacting with language models,
including listing available models, loading/unloading models, and generating text.
"""

from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from one_ring.services.model_service import (
    ModelInfo,
    ModelType,
    model_service,
)
from one_ring.core.config import settings

# Create router
router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


# Request/Response models
class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[ModelInfo]
    count: int


class LoadModelRequest(BaseModel):
    """Request model for loading a model."""
    model_name: str
    model_type: ModelType = ModelType.OLLAMA
    params: dict = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    model_name: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    params: dict = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str
    model: str
    tokens_generated: int
    tokens_input: int
    finish_reason: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    details: Optional[dict] = None


# Helper functions
def get_error_response(
    message: str, 
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[dict] = None
) -> dict:
    """Create a standardized error response."""
    return {
        "error": message,
        "details": details or {}
    }


# API Endpoints
@router.get(
    "/", 
    response_model=ModelListResponse,
    summary="List available models",
    description="List all available models, optionally filtered by type.",
    responses={
        200: {"description": "List of available models"},
        500: {"description": "Internal server error"},
    },
)
async def list_models(
    model_type: Optional[ModelType] = None,
    limit: int = 100,
    offset: int = 0,
) -> ModelListResponse:
    """List all available models.
    
    Args:
        model_type: Filter by model type (local or ollama).
        limit: Maximum number of models to return.
        offset: Number of models to skip.
        
    Returns:
        A list of available models.
    """
    try:
        models = model_service.list_models(model_type=model_type)
        
        # Apply pagination
        paginated_models = models[offset:offset + limit]
        
        return ModelListResponse(
            models=paginated_models,
            count=len(paginated_models),
            total=len(models),
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_error_response("Failed to list models", details={"error": str(e)}),
        )


@router.get(
    "/{model_name}",
    response_model=ModelInfo,
    summary="Get model information",
    description="Get detailed information about a specific model.",
    responses={
        200: {"description": "Model information"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_model(model_name: str) -> ModelInfo:
    """Get information about a specific model.
    
    Args:
        model_name: Name of the model to get information about.
        
    Returns:
        Detailed information about the model.
    """
    try:
        model_info = model_service.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=get_error_response(f"Model not found: {model_name}"),
            )
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_error_response("Failed to get model information", details={"error": str(e)}),
        )


@router.post(
    "/load",
    response_model=ModelInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Load a model",
    description="Load a model for inference.",
    responses={
        201: {"description": "Model loaded successfully"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        500: {"description": "Failed to load model"},
    },
)
async def load_model(request: LoadModelRequest) -> ModelInfo:
    """Load a model for inference.
    
    Args:
        request: Load model request containing model name, type, and parameters.
        
    Returns:
        Information about the loaded model.
    """
    try:
        success = model_service.load_model(
            model_name=request.model_name,
            model_type=request.model_type,
            **request.params,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=get_error_response(f"Failed to load model: {request.model_name}"),
            )
        
        # Get model info
        model_info = model_service.get_model_info(request.model_name)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=get_error_response("Failed to get model information after loading"),
            )
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model {request.model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_error_response("Failed to load model", details={"error": str(e)}),
        )


@router.delete(
    "/{model_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unload a model",
    description="Unload a model to free up resources.",
    responses={
        204: {"description": "Model unloaded successfully"},
        404: {"description": "Model not found or not loaded"},
        500: {"description": "Failed to unload model"},
    },
)
async def unload_model(model_name: str) -> None:
    """Unload a model.
    
    Args:
        model_name: Name of the model to unload.
    """
    try:
        success = model_service.unload_model(model_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=get_error_response(f"Model not found or not loaded: {model_name}"),
            )
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_error_response("Failed to unload model", details={"error": str(e)}),
        )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate text",
    description="Generate text from a prompt using the specified model.",
    responses={
        200: {"description": "Text generated successfully"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        500: {"description": "Failed to generate text"},
    },
)
async def generate_text(
    request: GenerateRequest,
) -> Union[GenerateResponse, StreamingResponse]:
    """Generate text from a prompt.
    
    Args:
        request: Generate request containing prompt and generation parameters.
        
    Returns:
        The generated text or a streaming response.
    """
    try:
        # For streaming responses
        if request.stream:
            async def generate_stream():
                try:
                    for chunk in model_service.generate(
                        prompt=request.prompt,
                        model_name=request.model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        stream=True,
                        **request.params,
                    ):
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                except Exception as e:
                    logger.error(f"Error in stream generation: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        
        # For non-streaming responses
        else:
            generated_text = model_service.generate(
                prompt=request.prompt,
                model_name=request.model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stream=False,
                **request.params,
            )
            
            # For simplicity, we're not counting tokens here
            # In a real implementation, you'd want to count input and output tokens
            return GenerateResponse(
                text=generated_text,
                model=request.model_name or model_service.default_model or "unknown",
                tokens_generated=0,
                tokens_input=0,
                finish_reason="length",
            )
    
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=get_error_response("Failed to generate text", details={"error": str(e)}),
        )


# Register the router in the FastAPI app
# This is typically done in the main FastAPI application file
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router)

__all__ = ["router"]
