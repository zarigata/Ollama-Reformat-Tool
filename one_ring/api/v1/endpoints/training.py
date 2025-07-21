"""
API endpoints for model training.

This module provides RESTful API endpoints for managing model training jobs,
including starting, monitoring, and evaluating fine-tuning processes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from one_ring.services.training_service import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
    training_service,
)
from one_ring.training.trainer import TrainingConfig

# Create router
router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)


# Request/Response models
class TrainingJobCreateRequest(BaseModel):
    """Request model for creating a training job."""
    config: Dict[str, Any] = Field(
        ...,
        description="Training configuration. See TrainingConfig for available options.",
    )
    name: Optional[str] = Field(
        None,
        description="Optional name for the training job.",
    )
    start_immediately: bool = Field(
        True,
        description="Whether to start the training job immediately.",
    )


class TrainingJobResponse(BaseModel):
    """Response model for a training job."""
    id: str
    name: str
    status: TrainingStatus
    config: Dict[str, Any]
    metrics: TrainingMetrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    model_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            TrainingStatus: lambda v: v.value,
        }


class TrainingJobListResponse(BaseModel):
    """Response model for listing training jobs."""
    jobs: List[TrainingJobResponse]
    count: int
    total: int


class TrainingStartResponse(BaseModel):
    """Response model for starting a training job."""
    job_id: str
    status: str
    message: str


# Helper functions
def job_to_response(job: TrainingJob) -> TrainingJobResponse:
    """Convert a TrainingJob to a TrainingJobResponse."""
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        config=job.config,
        metrics=job.metrics,
        start_time=job.start_time,
        end_time=job.end_time,
        error=job.error,
        output_dir=job.output_dir,
        model_name=job.model_name,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# API Endpoints
@router.post(
    "/jobs",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new training job",
    description="Create a new training job with the given configuration.",
    responses={
        201: {"description": "Training job created successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Failed to create training job"},
    },
)
async def create_training_job(
    request: TrainingJobCreateRequest,
    background_tasks: BackgroundTasks,
) -> TrainingJobResponse:
    """Create a new training job.
    
    Args:
        request: Training job creation request.
        background_tasks: FastAPI background tasks.
        
    Returns:
        The created training job.
    """
    try:
        # Create the training job
        job = training_service.create_training_job(
            config=request.config,
            name=request.name,
        )
        
        # Start the training job if requested
        if request.start_immediately:
            background_tasks.add_task(
                training_service.start_training_job,
                job_id=job.id,
                background=True,
            )
        
        return job_to_response(job)
    
    except Exception as e:
        logger.error(f"Failed to create training job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create training job: {str(e)}",
        )


@router.post(
    "/jobs/{job_id}/start",
    response_model=TrainingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a training job",
    description="Start a training job that is in the PENDING state.",
    responses={
        202: {"description": "Training job started"},
        400: {"description": "Invalid request or job cannot be started"},
        404: {"description": "Job not found"},
        500: {"description": "Failed to start training job"},
    },
)
async def start_training_job(
    job_id: str,
    background_tasks: BackgroundTasks,
) -> TrainingStartResponse:
    """Start a training job.
    
    Args:
        job_id: ID of the job to start.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Status of the operation.
    """
    try:
        # Start the training job in the background
        background_tasks.add_task(
            training_service.start_training_job,
            job_id=job_id,
            background=True,
        )
        
        return TrainingStartResponse(
            job_id=job_id,
            status="started",
            message=f"Training job {job_id} is starting in the background",
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to start training job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training job: {str(e)}",
        )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get a training job",
    description="Get information about a specific training job.",
    responses={
        200: {"description": "Training job details"},
        404: {"description": "Job not found"},
    },
)
async def get_training_job(job_id: str) -> TrainingJobResponse:
    """Get a training job by ID.
    
    Args:
        job_id: ID of the job to get.
        
    Returns:
        The training job details.
    """
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )
    
    return job_to_response(job)


@router.get(
    "/jobs",
    response_model=TrainingJobListResponse,
    summary="List training jobs",
    description="List all training jobs, optionally filtered by status.",
    responses={
        200: {"description": "List of training jobs"},
    },
)
async def list_training_jobs(
    status: Optional[List[TrainingStatus]] = Query(
        None,
        description="Filter jobs by status",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of jobs to return",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of jobs to skip",
    ),
) -> TrainingJobListResponse:
    """List training jobs.
    
    Args:
        status: Filter jobs by status.
        limit: Maximum number of jobs to return.
        offset: Number of jobs to skip.
        
    Returns:
        A list of training jobs.
    """
    jobs = training_service.list_jobs(
        status=status,
        limit=limit,
        offset=offset,
    )
    
    # Get total count for pagination
    total = len(training_service.list_jobs())
    
    return TrainingJobListResponse(
        jobs=[job_to_response(job) for job in jobs],
        count=len(jobs),
        total=total,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=TrainingJobResponse,
    summary="Cancel a training job",
    description="Cancel a training job that is in the PENDING or RUNNING state.",
    responses={
        200: {"description": "Training job cancelled"},
        400: {"description": "Job cannot be cancelled"},
        404: {"description": "Job not found"},
    },
)
async def cancel_training_job(job_id: str) -> TrainingJobResponse:
    """Cancel a training job.
    
    Args:
        job_id: ID of the job to cancel.
        
    Returns:
        The updated training job.
    """
    success = training_service.cancel_job(job_id)
    if not success:
        job = training_service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found",
            )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job {job_id} in state {job.status}",
        )
    
    return job_to_response(training_service.get_job(job_id))


@router.get(
    "/jobs/{job_id}/logs",
    summary="Get training job logs",
    description="Stream logs for a training job.",
    responses={
        200: {"description": "Log stream"},
        404: {"description": "Job not found"},
    },
)
async def get_training_job_logs(job_id: str) -> StreamingResponse:
    """Stream logs for a training job.
    
    Args:
        job_id: ID of the job to get logs for.
        
    Returns:
        A streaming response with the logs.
    """
    # TODO: Implement proper log streaming
    # For now, just return a placeholder response
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found",
        )
    
    async def generate_logs():
        yield f"Logs for training job {job_id}\n"
        yield f"Status: {job.status.value}\n"
        yield f"Created at: {job.created_at.isoformat()}\n"
        if job.start_time:
            yield f"Started at: {job.start_time.isoformat()}\n"
        if job.end_time:
            yield f"Ended at: {job.end_time.isoformat()}\n"
        if job.error:
            yield f"\nError: {job.error}\n"
        
        # Simulate some log output
        if job.status == TrainingStatus.RUNNING:
            for i in range(10):
                yield f"[Epoch {i+1}/10] Loss: {10.0 - i:.4f}\n"
                await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_logs(),
        media_type="text/plain",
    )


@router.get(
    "/default-config",
    response_model=Dict[str, Any],
    summary="Get default training configuration",
    description="Get the default training configuration that can be used as a starting point.",
    responses={
        200: {"description": "Default training configuration"},
    },
)
async def get_default_config() -> Dict[str, Any]:
    """Get the default training configuration.
    
    Returns:
        The default training configuration.
    """
    # Create a default config and convert it to a dict
    config = TrainingConfig()
    return asdict(config)


# Register the router in the FastAPI app
# This is typically done in the main FastAPI application file
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router)

__all__ = ["router"]
