"""
Model Training Module for the One Ring platform.

This module provides functionality for fine-tuning language models
using various techniques like LoRA, QLoRA, and AdaLoRA.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import (
    AdaLoraConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.models.manager import ModelConfig, model_manager
from one_ring.utils.cleanup import register_cleanup_handler


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model configuration
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    "Name or path of the base model"
    
    # Dataset configuration
    dataset_path: str = ""
    "Path to the training dataset (local file or Hugging Face dataset)"
    
    dataset_name: str = ""
    "Name of the dataset (if using Hugging Face datasets)"
    
    train_file: str = ""
    "Path to the training file (if using local files)"
    
    validation_file: str = ""
    "Path to the validation file (if using local files)"
    
    validation_split_percentage: int = 5
    "Percentage of the training set to use for validation"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    "Number of training epochs"
    
    per_device_train_batch_size: int = 4
    "Batch size per device for training"
    
    per_device_eval_batch_size: int = 4
    "Batch size per device for evaluation"
    
    gradient_accumulation_steps: int = 4
    "Number of updates steps to accumulate before performing a backward/update pass"
    
    learning_rate: float = 2e-5
    "Initial learning rate for the AdamW optimizer"
    
    weight_decay: float = 0.01
    "Weight decay for the AdamW optimizer"
    
    warmup_ratio: float = 0.03
    "Ratio of total training steps used for a linear warmup"
    
    max_grad_norm: float = 0.3
    "Maximum gradient norm for gradient clipping"
    
    # Generation parameters
    max_seq_length: int = 2048
    "Maximum sequence length for input sequences"
    
    # PEFT configuration
    use_peft: bool = True
    "Whether to use Parameter-Efficient Fine-Tuning (PEFT)"
    
    peft_method: str = "lora"
    "PEFT method to use (lora, qlora, adalora)"
    
    lora_rank: int = 8
    "Rank of the LoRA update matrices"
    
    lora_alpha: int = 16
    "Alpha parameter for LoRA scaling"
    
    lora_dropout: float = 0.05
    "Dropout probability for LoRA layers"
    
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    "List of target modules for LoRA"
    
    # Quantization configuration
    load_in_4bit: bool = True
    "Whether to load the model in 4-bit precision"
    
    load_in_8bit: bool = False
    "Whether to load the model in 8-bit precision"
    
    bnb_4bit_quant_type: str = "nf4"
    "Quantization type for 4-bit precision"
    
    bnb_4bit_compute_dtype: str = "bfloat16"
    "Compute dtype for 4-bit precision"
    
    # Output configuration
    output_dir: str = ""
    "Output directory for saving checkpoints and final model"
    
    logging_dir: str = ""
    "Directory for storing logs"
    
    save_total_limit: int = 3
    "Maximum number of checkpoints to keep"
    
    save_strategy: str = "epoch"
    "When to save checkpoints (epoch, steps, no)"
    
    evaluation_strategy: str = "epoch"
    "When to evaluate the model (epoch, steps, no)"
    
    logging_steps: int = 10
    "Log every X updates steps"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default output directories if not provided
        if not self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_name_or_path.split("/")[-1]
            self.output_dir = str(settings.MODEL_SAVE_DIR / f"{model_name}_{timestamp}")
        
        if not self.logging_dir:
            self.logging_dir = str(Path(self.output_dir) / "logs")
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Set device
        self.device = hardware_manager.get_default_device()
        
        # Set up compute dtype
        self.bnb_4bit_compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        
        # Validate PEFT method
        if self.use_peft and self.peft_method not in ["lora", "qlora", "adalora"]:
            raise ValueError(
                f"Invalid PEFT method: {self.peft_method}. "
                "Must be one of: lora, qlora, adalora"
            )


def load_and_prepare_dataset(config: TrainingConfig) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and prepare the dataset for training.
    
    Args:
        config: Training configuration.
        
    Returns:
        A tuple of (train_dataset, eval_dataset).
    """
    data_files = {}
    
    # Handle local dataset files
    if config.train_file:
        data_files["train"] = config.train_file
    if config.validation_file:
        data_files["validation"] = config.validation_file
    
    # Load dataset
    if data_files:
        extension = config.train_file.split(".")[-1] if config.train_file else "json"
        
        if extension == "txt":
            extension = "text"
        
        dataset = load_dataset(extension, data_files=data_files)
    elif config.dataset_path:
        # Load from local directory or Hugging Face dataset
        dataset = load_dataset(config.dataset_path, name=config.dataset_name or None)
    else:
        raise ValueError(
            "No dataset provided. Please specify dataset_path or train_file/validation_file."
        )
    
    # Split dataset if needed
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=config.validation_split_percentage / 100.0,
            shuffle=True,
            seed=42,
        )
    
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")
    
    return train_dataset, eval_dataset


def prepare_model_for_training(config: TrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Prepare the model and tokenizer for training.
    
    Args:
        config: Training configuration.
        
    Returns:
        A tuple of (model, tokenizer).
    """
    # Configure quantization
    bnb_config = None
    if config.load_in_4bit or config.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=True,
    )
    
    # Prepare model for k-bit training if needed
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    
    # Apply PEFT if enabled
    if config.use_peft:
        peft_config = None
        
        if config.peft_method == "lora" or config.peft_method == "qlora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
        elif config.peft_method == "adalora":
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                init_r=12,
                target_r=8,
                beta1=0.85,
                beta2=0.85,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                gamma_trainable=False,
                lora_plus=True,
            )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def train_model(config: TrainingConfig) -> None:
    """Train a model with the given configuration.
    
    Args:
        config: Training configuration.
    """
    logger.info(f"Starting training with config: {config}")
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_for_training(config)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
    
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    tokenized_eval_dataset = None
    if eval_dataset is not None:
        tokenized_eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        save_strategy=config.save_strategy,
        evaluation_strategy=config.evaluation_strategy,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        fp16=not config.load_in_4bit and not config.load_in_8bit and torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    
    # Register cleanup handler
    register_cleanup_handler(lambda: trainer.save_model(config.output_dir))
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training configuration
    with open(Path(config.output_dir) / "training_config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig()
    train_model(config)
