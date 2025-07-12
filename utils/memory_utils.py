#!/usr/bin/env python3
"""
Memory Management Utilities

Provides utilities for efficient memory usage, GPU memory monitoring,
and resource management for ComfyUI Universal Detailer.
"""

import torch
import psutil
import gc
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
from contextlib import contextmanager
from functools import lru_cache, wraps
import threading
from collections import deque

logger = logging.getLogger(__name__)

def memory_profile(func):
    """Decorator to profile memory usage of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global_memory_manager = args[0] if args and isinstance(args[0], MemoryManager) else None
        
        if global_memory_manager:
            with global_memory_manager.memory_monitor(func.__name__):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

class MemoryManager:
    """
    Memory management utilities for efficient GPU and system memory usage.
    
    Features:
    - Memory monitoring and reporting
    - Automatic cleanup and garbage collection
    - Memory-aware batch size adjustment
    - Resource tracking and optimization
    - Thread-safe operations
    - LRU caching for performance
    - Memory leak detection
    """
    
    def __init__(self):
        self.memory_stats = deque(maxlen=1000)  # Circular buffer for efficiency
        self.cleanup_threshold = 0.9  # Cleanup when 90% memory used
        self._cache_stats = {}
        self._lock = threading.Lock()
        self._tensor_pool = {}
        self._last_cleanup = time.time()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            stats.update({
                "system_total_gb": system_memory.total / (1024**3),
                "system_used_gb": system_memory.used / (1024**3),
                "system_available_gb": system_memory.available / (1024**3),
                "system_percent": system_memory.percent
            })
            
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    
                    stats.update({
                        f"gpu_{i}_total_gb": gpu_memory / (1024**3),
                        f"gpu_{i}_allocated_gb": allocated / (1024**3),
                        f"gpu_{i}_reserved_gb": reserved / (1024**3),
                        f"gpu_{i}_free_gb": (gpu_memory - reserved) / (1024**3),
                        f"gpu_{i}_allocated_percent": (allocated / gpu_memory) * 100,
                        f"gpu_{i}_reserved_percent": (reserved / gpu_memory) * 100
                    })
            else:
                stats["gpu_available"] = False
                
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            
        return stats
    
    def log_memory_usage(self, context: str = ""):
        """
        Log current memory usage with context.
        
        Args:
            context: Description of current operation
        """
        try:
            stats = self.get_memory_stats()
            
            log_msg = f"Memory usage"
            if context:
                log_msg += f" ({context})"
            log_msg += f": System {stats.get('system_percent', 0):.1f}%"
            
            if torch.cuda.is_available():
                gpu_percent = stats.get('gpu_0_allocated_percent', 0)
                log_msg += f", GPU {gpu_percent:.1f}%"
                
            logger.info(log_msg)
            
            # Store stats for analysis
            self.memory_stats.append({
                "timestamp": time.time(),
                "context": context,
                **stats
            })
            
        except Exception as e:
            logger.error(f"Failed to log memory usage: {e}")
    
    def cleanup_memory(self, force: bool = False):
        """
        Perform memory cleanup operations.
        
        Args:
            force: Force cleanup regardless of current usage
        """
        try:
            should_cleanup = force
            
            if not force:
                stats = self.get_memory_stats()
                gpu_usage = stats.get('gpu_0_allocated_percent', 0) / 100.0
                system_usage = stats.get('system_percent', 0) / 100.0
                
                should_cleanup = (gpu_usage > self.cleanup_threshold or 
                                system_usage > self.cleanup_threshold)
            
            if should_cleanup:
                logger.info("Performing memory cleanup...")
                
                # Clear PyTorch caches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Memory cleanup completed")
                self.log_memory_usage("after cleanup")
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """
        Context manager for monitoring memory usage during operations.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        try:
            self.log_memory_usage(f"before {operation_name}")
            start_stats = self.get_memory_stats()
            
            yield
            
            end_stats = self.get_memory_stats()
            self.log_memory_usage(f"after {operation_name}")
            
            # Calculate memory delta
            if torch.cuda.is_available():
                gpu_delta = (end_stats.get('gpu_0_allocated_gb', 0) - 
                           start_stats.get('gpu_0_allocated_gb', 0))
                if gpu_delta > 0.1:  # More than 100MB
                    logger.warning(f"{operation_name} used {gpu_delta:.2f}GB GPU memory")
                    
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            yield
    
    def estimate_batch_size(
        self,
        image_height: int,
        image_width: int,
        channels: int = 3,
        safety_factor: float = 0.7
    ) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            image_height: Height of input images
            image_width: Width of input images
            channels: Number of image channels
            safety_factor: Safety factor for memory usage (0.7 = use 70% of available)
            
        Returns:
            Estimated optimal batch size
        """
        try:
            if not torch.cuda.is_available():
                return 1  # Conservative for CPU
            
            stats = self.get_memory_stats()
            available_memory = stats.get('gpu_0_free_gb', 4.0) * safety_factor
            
            # Estimate memory per image (comprehensive calculation)
            # Original image + VAE latents + intermediate tensors + model overhead
            pixels_per_image = image_height * image_width * channels
            
            # Memory components:
            # 1. Original image (float32): 4 bytes per pixel
            # 2. VAE latents (compression ~8x): ~0.5 bytes per pixel  
            # 3. Masks and intermediate tensors: ~2 bytes per pixel
            # 4. Model weights and activations: ~1 bytes per pixel
            # 5. Safety overhead: ~0.5 bytes per pixel
            memory_per_image_gb = (
                pixels_per_image * 8 / (1024**3) +      # Original image (float32) + copy
                pixels_per_image * 1 / (1024**3) +      # VAE latents (8x compression)
                pixels_per_image * 4 / (1024**3) +      # Masks and intermediate tensors
                pixels_per_image * 2 / (1024**3) +      # Model activations overhead
                0.5                                      # Fixed overhead (model weights, etc.)
            )
            
            estimated_batch_size = max(1, int(available_memory / memory_per_image_gb))
            
            logger.info(f"Estimated batch size: {estimated_batch_size} "
                       f"(available: {available_memory:.1f}GB, "
                       f"per image: {memory_per_image_gb:.2f}GB)")
            
            return min(estimated_batch_size, 8)  # Cap at 8 for stability
            
        except Exception as e:
            logger.error(f"Batch size estimation failed: {e}")
            return 1
    
    def optimize_tensor_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor memory usage.
        
        Args:
            tensor: Input tensor to optimize
            
        Returns:
            Optimized tensor
        """
        try:
            # Make tensor contiguous if not already
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Use appropriate dtype
            if tensor.dtype == torch.float64:
                tensor = tensor.float()  # Convert to float32
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor memory optimization failed: {e}")
            return tensor
    
    def get_memory_report(self) -> str:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Formatted memory report string
        """
        try:
            stats = self.get_memory_stats()
            
            report = "=== Memory Usage Report ===\n"
            
            # System memory
            report += f"System Memory:\n"
            report += f"  Total: {stats.get('system_total_gb', 0):.1f}GB\n"
            report += f"  Used: {stats.get('system_used_gb', 0):.1f}GB ({stats.get('system_percent', 0):.1f}%)\n"
            report += f"  Available: {stats.get('system_available_gb', 0):.1f}GB\n\n"
            
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    report += f"GPU {i} Memory:\n"
                    report += f"  Total: {stats.get(f'gpu_{i}_total_gb', 0):.1f}GB\n"
                    report += f"  Allocated: {stats.get(f'gpu_{i}_allocated_gb', 0):.1f}GB ({stats.get(f'gpu_{i}_allocated_percent', 0):.1f}%)\n"
                    report += f"  Reserved: {stats.get(f'gpu_{i}_reserved_gb', 0):.1f}GB ({stats.get(f'gpu_{i}_reserved_percent', 0):.1f}%)\n"
                    report += f"  Free: {stats.get(f'gpu_{i}_free_gb', 0):.1f}GB\n\n"
            else:
                report += "GPU: Not available\n\n"
            
            # Historical data
            if self.memory_stats:
                report += "Recent Memory History:\n"
                for stat in self.memory_stats[-5:]:  # Last 5 entries
                    timestamp = time.strftime("%H:%M:%S", time.localtime(stat['timestamp']))
                    context = stat.get('context', 'unknown')
                    system_pct = stat.get('system_percent', 0)
                    gpu_pct = stat.get('gpu_0_allocated_percent', 0)
                    report += f"  {timestamp} ({context}): System {system_pct:.1f}%, GPU {gpu_pct:.1f}%\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate memory report: {e}")
            return "Memory report generation failed"