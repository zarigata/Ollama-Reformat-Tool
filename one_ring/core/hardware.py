"""
Hardware detection and configuration for the One Ring platform.

This module provides functionality to automatically detect and configure
available hardware resources, including GPUs (CUDA/ROCm) and CPUs.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel

from one_ring.core.logger import get_logger

logger = get_logger(__name__)


class DeviceType(Enum):
    """Enumeration of supported device types."""
    CUDA = auto()
    ROCM = auto()
    MPS = auto()  # Apple Metal Performance Shaders
    CPU = auto()


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    memory_total: int  # in MB
    memory_available: int  # in MB
    compute_capability: Optional[Tuple[int, int]] = None


class HardwareInfo(BaseModel):
    """Comprehensive hardware information."""
    device_type: DeviceType
    device_count: int = 0
    gpus: List[GPUInfo] = []
    cpu_cores: int = 0
    system_ram: int = 0  # in MB
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    os_info: Dict[str, str] = {}


class HardwareManager:
    """Manages hardware detection and configuration."""
    
    def __init__(self):
        self.info = HardwareInfo(device_type=DeviceType.CPU)
        self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect available hardware resources."""
        self._detect_os_info()
        self._detect_cpu_info()
        
        # Check for CUDA
        if torch.cuda.is_available():
            self._detect_cuda()
        # Check for ROCm (AMD GPUs)
        elif self._is_rocm_available():
            self._detect_rocm()
        # Check for MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            self._detect_mps()
        else:
            self._use_cpu()
        
        logger.info(f"Hardware detection complete. Using {self.info.device_type.name} with {self.info.device_count} devices")
    
    def _detect_os_info(self) -> None:
        """Detect operating system information."""
        self.info.os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
    
    def _detect_cpu_info(self) -> None:
        """Detect CPU information."""
        import psutil
        
        self.info.cpu_cores = os.cpu_count() or 1
        self.info.system_ram = psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB
    
    def _detect_cuda(self) -> None:
        """Detect and configure CUDA devices."""
        self.info.device_type = DeviceType.CUDA
        self.info.device_count = torch.cuda.device_count()
        self.info.cuda_version = torch.version.cuda
        
        for i in range(self.info.device_count):
            try:
                name = torch.cuda.get_device_name(i)
                prop = torch.cuda.get_device_properties(i)
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=prop.total_memory // (1024 * 1024),  # Convert to MB
                    memory_available=torch.cuda.memory_reserved(i) // (1024 * 1024),  # Convert to MB
                    compute_capability=(prop.major, prop.minor)
                )
                
                self.info.gpus.append(gpu_info)
                logger.info(f"Detected CUDA device {i}: {name} (Compute Capability: {prop.major}.{prop.minor})")
                
            except Exception as e:
                logger.warning(f"Failed to get info for CUDA device {i}: {e}")
    
    def _is_rocm_available(self) -> bool:
        """Check if ROCm is available."""
        try:
            # Try to import ROCm libraries
            import torch
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except ImportError:
            return False
    
    def _detect_rocm(self) -> None:
        """Detect and configure ROCm devices."""
        self.info.device_type = DeviceType.ROCM
        
        try:
            # ROCm specific detection
            rocm_version = subprocess.check_output(["rocm-smi", "--version"]).decode("utf-8")
            self.info.rocm_version = rocm_version.split("\n")[0] if rocm_version else "Unknown"
            
            # Get GPU info using ROCm SMI
            try:
                rocm_smi = subprocess.check_output(["rocm-smi", "-i", "-m"]).decode("utf-8")
                # Parse ROCm SMI output to get GPU info
                # This is a simplified example - you might need to adjust based on your ROCm version
                gpu_sections = [s.strip() for s in rocm_smi.split("===============================") if s.strip()]
                
                for i, section in enumerate(gpu_sections):
                    if "Card" in section and "Memory" in section:
                        lines = section.split("\n")
                        name = "Unknown"
                        memory_total = 0
                        
                        for line in lines:
                            if "Card" in line and ":" in line:
                                name = line.split(":")[1].strip()
                            elif "GPU Memory" in line and "Total" in line:
                                memory_str = line.split("Total")[1].strip().split(" ")[0]
                                memory_total = int(float(memory_str) * 1024)  # Convert GB to MB
                        
                        gpu_info = GPUInfo(
                            index=i,
                            name=name,
                            memory_total=memory_total,
                            memory_available=memory_total  # Approximate
                        )
                        
                        self.info.gpus.append(gpu_info)
                        logger.info(f"Detected ROCm device {i}: {name} (VRAM: {memory_total}MB)")
                
                self.info.device_count = len(self.info.gpus)
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Failed to get detailed ROCm GPU info. ROCm may not be properly installed.")
                self._use_cpu()
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ROCm not properly installed or not found. Falling back to CPU.")
            self._use_cpu()
    
    def _detect_mps(self) -> None:
        """Detect and configure MPS (Apple Silicon)."""
        self.info.device_type = DeviceType.MPS
        self.info.device_count = 1  # MPS only supports one device
        
        try:
            # Get Apple Silicon GPU info using system_profiler
            if platform.system() == "Darwin":
                gpu_info = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"], 
                    stderr=subprocess.DEVNULL
                ).decode("utf-8")
                
                # Parse GPU info
                gpu_name = "Apple Silicon GPU"
                memory_total = 0
                
                for line in gpu_info.split("\n"):
                    if "Chipset Model" in line:
                        gpu_name = line.split(":")[1].strip()
                    elif "VRAM" in line and "Total" in line:
                        memory_str = line.split("Total")[1].strip().split(" ")[0]
                        memory_total = int(float(memory_str) * 1024)  # Convert GB to MB
                
                gpu_info = GPUInfo(
                    index=0,
                    name=gpu_name,
                    memory_total=memory_total,
                    memory_available=memory_total  # Approximate
                )
                
                self.info.gpus.append(gpu_info)
                logger.info(f"Detected MPS device: {gpu_name} (VRAM: {memory_total}MB)")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to get MPS device info. Falling back to CPU.")
            self._use_cpu()
    
    def _use_cpu(self) -> None:
        """Configure to use CPU."""
        self.info.device_type = DeviceType.CPU
        self.info.device_count = 1
        logger.info("Using CPU for computation")
    
    def get_available_devices(self) -> List[str]:
        """Get a list of available device names."""
        if self.info.device_type == DeviceType.CUDA:
            return [f"cuda:{i}" for i in range(self.info.device_count)]
        elif self.info.device_type == DeviceType.ROCM:
            return [f"rocm:{i}" for i in range(self.info.device_count)]
        elif self.info.device_type == DeviceType.MPS:
            return ["mps"]
        else:
            return ["cpu"]
    
    def get_default_device(self) -> str:
        """Get the default device string for PyTorch."""
        if self.info.device_type == DeviceType.CUDA:
            return "cuda:0"
        elif self.info.device_type == DeviceType.ROCM:
            return "rocm:0"
        elif self.info.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"


# Create a global hardware manager instance
hardware_manager = HardwareManager()

# Export the hardware manager for easy access
__all__ = ["hardware_manager", "HardwareManager", "DeviceType"]
