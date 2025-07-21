# Training Workflow Documentation

This document provides a comprehensive guide to the training workflow in the One Ring platform, including how to use the training service and API endpoints.

## Table of Contents
- [Overview](#overview)
- [Training Configuration](#training-configuration)
- [Using the Training Service](#using-the-training-service)
  - [Creating a Training Job](#creating-a-training-job)
  - [Starting a Training Job](#starting-a-training-job)
  - [Monitoring Training Progress](#monitoring-training-progress)
  - [Canceling a Training Job](#canceling-a-training-job)
- [Training API Reference](#training-api-reference)
- [Example Workflow](#example-workflow)
- [Troubleshooting](#troubleshooting)

## Overview

The training workflow in One Ring is designed to be flexible and powerful, supporting various fine-tuning techniques like LoRA, QLoRA, and AdaLoRA. The workflow consists of the following key components:

1. **Training Service**: Manages training jobs, tracks progress, and handles job lifecycle.
2. **Training API**: RESTful endpoints for creating, starting, monitoring, and managing training jobs.
3. **Trainer Module**: Implements the actual model training logic with support for various optimization techniques.

## Training Configuration

Training behavior is controlled by the `TrainingConfig` class, which includes parameters such as:

```python
class TrainingConfig:
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    dataset_path: str = ""
    train_file: str = ""
    validation_file: str = ""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-5
    use_peft: bool = True
    peft_method: str = "lora"  # "lora", "qlora", or "adalora"
    # ... and many more
```

## Using the Training Service

### Creating a Training Job

```python
from one_ring.services.training_service import training_service
from one_ring.training.trainer import TrainingConfig

# Create a training configuration
config = TrainingConfig(
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
    train_file="path/to/train.jsonl",
    validation_file="path/to/validation.jsonl",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    use_peft=True,
    peft_method="lora",
)

# Create a training job
job = training_service.create_training_job(
    config=config,
    name="my-first-finetuning"
)
```

### Starting a Training Job

```python
# Start the training job (runs in background)
training_service.start_training_job(job.id, background=True)
```

### Monitoring Training Progress

```python
# Get job status
job = training_service.get_job(job_id)
print(f"Status: {job.status}")
print(f"Progress: {job.metrics.epoch:.1f}/{job.config.num_train_epochs} epochs")
print(f"Current loss: {job.metrics.loss:.4f}")

# List all jobs
jobs = training_service.list_jobs()
for job in jobs:
    print(f"{job.id}: {job.name} - {job.status}")
```

### Canceling a Training Job

```python
# Cancel a running job
success = training_service.cancel_job(job_id)
if success:
    print("Job cancelled successfully")
else:
    print("Failed to cancel job")
```

## Training API Reference

The training API provides RESTful endpoints for managing training jobs:

### Create a Training Job

```http
POST /api/v1/training/jobs
Content-Type: application/json

{
  "config": {
    "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
    "train_file": "path/to/train.jsonl",
    "num_train_epochs": 3,
    "use_peft": true,
    "peft_method": "lora"
  },
  "name": "my-finetuning-job",
  "start_immediately": true
}
```

### Start a Training Job

```http
POST /api/v1/training/jobs/{job_id}/start
```

### Get Training Job Status

```http
GET /api/v1/training/jobs/{job_id}
```

### List Training Jobs

```http
GET /api/v1/training/jobs?status=running,completed&limit=10&offset=0
```

### Cancel a Training Job

```http
POST /api/v1/training/jobs/{job_id}/cancel
```

### Get Training Logs

```http
GET /api/v1/training/jobs/{job_id}/logs
```

## Example Workflow

Here's a complete example of fine-tuning a model using the API:

1. **Prepare your dataset** in JSONL format:
   ```json
   {"text": "This is an example training example."}
   {"text": "Another training example..."}
   ```

2. **Create a training job**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/training/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "config": {
         "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
         "train_file": "data/train.jsonl",
         "validation_file": "data/validation.jsonl",
         "num_train_epochs": 3,
         "per_device_train_batch_size": 4,
         "learning_rate": 2e-5,
         "use_peft": true,
         "peft_method": "lora"
       },
       "name": "my-finetuning-job",
       "start_immediately": true
     }'
   ```

3. **Monitor progress**:
   ```bash
   # Get job status
   curl http://localhost:8000/api/v1/training/jobs/{job_id}
   
   # Stream logs
   curl http://localhost:8000/api/v1/training/jobs/{job_id}/logs
   ```

4. **Use the fine-tuned model**:
   Once training is complete, the model will be saved to the specified output directory and can be loaded using the Transformers library or served through the One Ring model serving API.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Enable gradient accumulation with `gradient_accumulation_steps`
   - Use smaller models or enable 4/8-bit quantization

2. **Training is too slow**
   - Increase `per_device_train_batch_size` if memory allows
   - Use `fp16` or `bf16` mixed precision training
   - Enable gradient checkpointing

3. **Model not converging**
   - Try different learning rates (use learning rate finder)
   - Adjust the learning rate schedule
   - Check your data quality and preprocessing

### Getting Help

For additional help, please:
1. Check the logs using the `/api/v1/training/jobs/{job_id}/logs` endpoint
2. Review the training configuration for potential issues
3. Check the system resource usage (CPU/GPU memory, disk space)
4. Open an issue on our GitHub repository with detailed error messages
