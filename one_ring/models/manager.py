"""
Model Manager for the One Ring platform.

This module provides functionality to load, manage, and interact with
various language models in a unified way.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager
from one_ring.utils.cleanup import register_cleanup_handler

# Type aliases
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class ModelConfig:
    """Configuration for loading a model."""
    
    model_name_or_path: str
    "Name or path of the model"
    
    device: str = "auto"
    "Device to load the model on (auto, cuda, cpu, mps, etc.)"
    
    torch_dtype: Optional[torch.dtype] = None
    "Data type to use for model weights"
    
    load_in_8bit: bool = False
    "Whether to load the model in 8-bit precision"
    
    load_in_4bit: bool = False
    "Whether to load the model in 4-bit precision"
    
    use_flash_attention_2: bool = True
    "Whether to use Flash Attention 2 for faster inference"
    
    trust_remote_code: bool = True
    "Whether to trust remote code when loading the model"
    
    # Quantization settings
    bnb_4bit_quant_type: str = "nf4"
    "Quantization type for 4-bit precision"
    
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None
    "Compute dtype for 4-bit precision"
    
    bnb_4bit_use_double_quant: bool = True
    "Whether to use double quantization for 4-bit precision"
    
    # Tokenizer settings
    tokenizer_name_or_path: Optional[str] = None
    "Name or path of the tokenizer (if different from model)"
    
    tokenizer_kwargs: Optional[Dict] = None
    "Additional keyword arguments for the tokenizer"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.device == "auto":
            self.device = hardware_manager.get_default_device()
        
        if self.torch_dtype is None:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.torch_dtype = torch.float32  # MPS doesn't support bf16 yet
            else:
                self.torch_dtype = torch.float32
        
        if self.bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = self.torch_dtype
        
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}


class ModelManager:
    """Manages loading and interacting with language models."""
    
    def __init__(self):
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.config: Optional[ModelConfig] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        
        # Register cleanup handler
        register_cleanup_handler(self.cleanup)
    
    def load_model(self, config: ModelConfig) -> Tuple[PreTrainedModel, Tokenizer]:
        """Load a model and tokenizer with the given configuration.
        
        Args:
            config: Configuration for loading the model.
            
        Returns:
            A tuple of (model, tokenizer).
        """
        logger.info(f"Loading model: {config.model_name_or_path}")
        
        # Clean up any existing model
        self.cleanup()
        
        # Update config with auto-detected values
        config = ModelConfig(**config.__dict__)  # Create a new instance with updated values
        
        # Set up quantization config if needed
        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            logger.info(f"Using {'4-bit' if config.load_in_4bit else '8-bit'} quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            )
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from: {config.tokenizer_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path,
                trust_remote_code=config.trust_remote_code,
                **config.tokenizer_kwargs
            )
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
            # Load model
            logger.info(f"Loading model from: {config.model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                device_map=config.device,
                torch_dtype=config.torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=config.trust_remote_code,
                use_flash_attention_2=config.use_flash_attention_2,
            )
            
            # Update model config
            model.config.pad_token_id = tokenizer.pad_token_id
            
            # Store references
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.device = config.device
            self.is_loaded = True
            
            logger.info(f"Successfully loaded model on device: {self.device}")
            return model, tokenizer
            
        except Exception as e:
            self.cleanup()
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the current model and free up memory."""
        self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources used by the model."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        
        # Run garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            max_length: Maximum length of the generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate.
            do_sample: Whether to use sampling.
            **kwargs: Additional generation parameters.
            
        Returns:
            A list of generated text sequences.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("No model is loaded. Call load_model() first.")
        
        try:
            # Encode the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            ).to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode the generated text
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(
                    output[len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )
                generated_texts.append(text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise


# Create a global model manager instance
model_manager = ModelManager()

__all__ = ["ModelConfig", "ModelManager", "model_manager"]
