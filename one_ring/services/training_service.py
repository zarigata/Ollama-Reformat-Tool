"""
Training Service for the One Ring platform.

This module provides a high-level interface for managing model training jobs,
including starting, monitoring, and evaluating fine-tuning processes.
"""

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from pydantic import BaseModel, Field

from one_ring.core.config import settings
from one_ring.data.ingestion import DocumentChunk, document_processor
from one_ring.training.trainer import TrainingConfig, train_model
from one_ring.utils.cleanup import register_cleanup_handler


class TrainingStatus(str, Enum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingMetrics(BaseModel):
    """Metrics for a training job."""
    epoch: float = 0.0
    step: int = 0
    total_steps: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    eval_loss: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    samples_per_second: float = 0.0
    steps_per_second: float = 0.0
    eta_seconds: float = 0.0


class TrainingJob(BaseModel):
    """Represents a training job."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: TrainingStatus = TrainingStatus.PENDING
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            TrainingStatus: lambda v: v.value,
        }
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update training metrics."""
        self.updated_at = datetime.utcnow()
        
        # Update metrics
        for key, value in metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
            else:
                self.metrics.metrics[key] = value
    
    def start(self) -> None:
        """Mark the job as started."""
        self.status = TrainingStatus.RUNNING
        self.start_time = datetime.utcnow()
        self.updated_at = self.start_time
    
    def complete(self, output_dir: str) -> None:
        """Mark the job as completed."""
        self.status = TrainingStatus.COMPLETED
        self.end_time = datetime.utcnow()
        self.updated_at = self.end_time
        self.output_dir = output_dir
    
    def fail(self, error: str) -> None:
        """Mark the job as failed."""
        self.status = TrainingStatus.FAILED
        self.end_time = datetime.utcnow()
        self.updated_at = self.end_time
        self.error = str(error)
    
    def cancel(self) -> None:
        """Mark the job as cancelled."""
        self.status = TrainingStatus.CANCELLED
        self.end_time = datetime.utcnow()
        self.updated_at = self.end_time


class TrainingService:
    """Service for managing model training jobs."""
    
    def __init__(self):
        """Initialize the training service."""
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_old_jobs,
            daemon=True,
            name="training-service-cleanup"
        )
        self._cleanup_thread.start()
        
        # Register cleanup handler
        register_cleanup_handler(self.cleanup)
        
        logger.info("Training service initialized")
    
    def create_training_job(
        self,
        config: Union[Dict[str, Any], TrainingConfig],
        name: Optional[str] = None,
    ) -> TrainingJob:
        """Create a new training job.
        
        Args:
            config: Training configuration.
            name: Optional name for the training job.
            
        Returns:
            The created training job.
        """
        # Convert TrainingConfig to dict if needed
        if isinstance(config, TrainingConfig):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        # Create job
        job = TrainingJob(
            name=name or f"training-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            config=config_dict,
        )
        
        # Store job
        with self._lock:
            self._jobs[job.id] = job
        
        logger.info(f"Created training job {job.id}")
        return job
    
    def start_training_job(
        self,
        job_id: str,
        background: bool = True,
    ) -> TrainingJob:
        """Start a training job.
        
        Args:
            job_id: ID of the job to start.
            background: Whether to run the job in a background thread.
            
        Returns:
            The updated training job.
            
        Raises:
            ValueError: If the job is not found or cannot be started.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            
            if job.status != TrainingStatus.PENDING:
                raise ValueError(f"Job {job_id} is not in PENDING state")
            
            # Mark job as running
            job.start()
            
            # Start training in a separate thread
            def train_thread():
                try:
                    # Convert config dict to TrainingConfig
                    config_dict = job.config.copy()
                    
                    # Set output directory if not specified
                    if not config_dict.get("output_dir"):
                        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        model_name = config_dict.get("model_name_or_path", "model").split("/")[-1]
                        output_dir = settings.MODEL_SAVE_DIR / f"{model_name}_{timestamp}"
                        config_dict["output_dir"] = str(output_dir)
                    
                    # Create config object
                    config = TrainingConfig(**config_dict)
                    
                    # Save config
                    job.config = asdict(config)
                    
                    # Start training
                    if background:
                        train_model(config)
                        job.complete(config.output_dir)
                    else:
                        # For testing/debugging
                        import time
                        for i in range(10):
                            if job.status == TrainingStatus.CANCELLED:
                                break
                            time.sleep(1)
                            job.update_metrics({
                                "epoch": i / 10.0,
                                "step": i,
                                "total_steps": 10,
                                "loss": 10.0 - i,
                                "learning_rate": 0.0001,
                                "samples_per_second": 10.0,
                                "steps_per_second": 1.0,
                                "eta_seconds": 10 - i,
                            })
                        job.complete("/fake/path/to/model")
                except Exception as e:
                    logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
                    job.fail(str(e))
                finally:
                    logger.info(f"Training job {job_id} completed with status {job.status}")
            
            thread = threading.Thread(
                target=train_thread,
                name=f"training-job-{job_id}",
                daemon=True,
            )
            thread.start()
            
            return job
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID.
        
        Args:
            job_id: ID of the job to get.
            
        Returns:
            The training job, or None if not found.
        """
        with self._lock:
            return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[Union[TrainingStatus, List[TrainingStatus]]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TrainingJob]:
        """List training jobs.
        
        Args:
            status: Filter by status (or list of statuses).
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.
            
        Returns:
            A list of training jobs.
        """
        with self._lock:
            jobs = list(self._jobs.values())
            
            # Filter by status
            if status is not None:
                if isinstance(status, TrainingStatus):
                    status = [status]
                jobs = [job for job in jobs if job.status in status]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply pagination
            return jobs[offset:offset + limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job.
        
        Args:
            job_id: ID of the job to cancel.
            
        Returns:
            True if the job was cancelled, False otherwise.
        """
        with self._lock:
            if job_id not in self._jobs:
                return False
            
            job = self._jobs[job_id]
            
            if job.status not in (TrainingStatus.PENDING, TrainingStatus.RUNNING):
                return False
            
            job.cancel()
            return True
    
    def _cleanup_old_jobs(self) -> None:
        """Clean up old completed/failed jobs."""
        while not self._stop_event.is_set():
            try:
                now = datetime.utcnow()
                max_age = settings.TRAINING_JOB_MAX_AGE_HOURS * 3600  # Convert hours to seconds
                
                with self._lock:
                    # Find old jobs to remove
                    to_remove = []
                    for job_id, job in self._jobs.items():
                        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
                            age = (now - job.updated_at).total_seconds()
                            if age > max_age:
                                to_remove.append(job_id)
                    
                    # Remove old jobs
                    for job_id in to_remove:
                        del self._jobs[job_id]
                    
                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} old training jobs")
                
                # Sleep for a while before checking again
                self._stop_event.wait(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}", exc_info=True)
                time.sleep(60)  # Wait a minute before retrying on error
    
    def cleanup(self) -> None:
        """Clean up resources used by the training service."""
        self._stop_event.set()
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        logger.info("Training service cleaned up")


# Create a global training service instance
training_service = TrainingService()

# Register cleanup handler
import atexit
atexit.register(training_service.cleanup)

__all__ = ["TrainingService", "TrainingJob", "TrainingStatus", "TrainingMetrics", "training_service"]
