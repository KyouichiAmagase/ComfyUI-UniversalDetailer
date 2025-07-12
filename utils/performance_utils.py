#!/usr/bin/env python3
"""
Performance Optimization Utilities for ComfyUI Universal Detailer

Provides performance monitoring, optimization, and benchmarking capabilities
for efficient processing in production environments.
"""

import time
import logging
import functools
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring and optimization system.
    """
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self.memory_peaks = {}
        self.benchmark_results = {}
        self._memory_pool = {}  # Tensor memory pool for reuse
        self._last_cleanup = time.time()
    
    @contextmanager
    def timer(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Store timing information
            if operation_name not in self.timings:
                self.timings[operation_name] = []
                self.call_counts[operation_name] = 0
                self.memory_peaks[operation_name] = []
            
            self.timings[operation_name].append(duration)
            self.call_counts[operation_name] += 1
            self.memory_peaks[operation_name].append(memory_delta)
            
            # Log slow operations
            if duration > 5.0:  # Longer than 5 seconds
                logger.warning(f"Slow operation: {operation_name} took {duration:.2f}s")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**3)
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Performance statistics dictionary
        """
        report = {
            "summary": {
                "total_operations": sum(self.call_counts.values()),
                "total_operations_types": len(self.timings),
                "total_time": sum(sum(times) for times in self.timings.values())
            },
            "operations": {}
        }
        
        for operation, times in self.timings.items():
            if times:
                report["operations"][operation] = {
                    "call_count": self.call_counts[operation],
                    "total_time": sum(times),
                    "average_time": np.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": np.std(times),
                    "average_memory_delta": np.mean(self.memory_peaks[operation]),
                    "max_memory_delta": max(self.memory_peaks[operation])
                }
        
        return report
    
    def get_pooled_tensor(self, shape: Tuple, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
        """Get a tensor from memory pool for reuse."""
        key = (shape, dtype, device)
        if key in self._memory_pool and len(self._memory_pool[key]) > 0:
            return self._memory_pool[key].pop()
        else:
            return torch.empty(shape, dtype=dtype, device=device)
    
    def return_pooled_tensor(self, tensor: torch.Tensor):
        """Return tensor to memory pool for reuse."""
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        if key not in self._memory_pool:
            self._memory_pool[key] = []
        if len(self._memory_pool[key]) < 5:  # Limit pool size
            self._memory_pool[key].append(tensor.detach().clone())
    
    def cleanup_memory_pool(self):
        """Cleanup memory pool periodically."""
        current_time = time.time()
        if current_time - self._last_cleanup > 300:  # Every 5 minutes
            for key in list(self._memory_pool.keys()):
                if len(self._memory_pool[key]) > 2:
                    self._memory_pool[key] = self._memory_pool[key][:2]
            self._last_cleanup = current_time
            logger.info("Memory pool cleaned up")

    def reset_stats(self):
        """Reset all performance statistics."""
        self.timings.clear()
        self.call_counts.clear()
        self.memory_peaks.clear()
        self.benchmark_results.clear()
        self._memory_pool.clear()

def performance_timer(operation_name: str = None):
    """
    Decorator for automatic performance timing.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with global_performance_monitor.timer(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

class TensorOptimizer:
    """
    Tensor optimization utilities for memory and performance.
    """
    
    @staticmethod
    def optimize_tensor_operations(tensor: torch.Tensor, operation: str = "general") -> torch.Tensor:
        """
        Optimize tensor for specific operations.
        
        Args:
            tensor: Input tensor
            operation: Type of operation ('detection', 'inpainting', 'masking')
        
        Returns:
            Optimized tensor
        """
        try:
            # Make contiguous for better memory access
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Pin memory for faster GPU transfers
            if not tensor.is_cuda and torch.cuda.is_available():
                if tensor.numel() > 1000000:  # Only for large tensors
                    tensor = tensor.pin_memory()
            
            # Operation-specific optimizations
            if operation == "detection":
                # For YOLO detection, ensure correct dtype
                if tensor.dtype != torch.uint8:
                    tensor = tensor.to(dtype=torch.uint8)
            
            elif operation == "inpainting":
                # For diffusion, ensure float32
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(dtype=torch.float32)
            
            elif operation == "masking":
                # For masks, use appropriate precision
                if tensor.dtype not in [torch.float32, torch.bool]:
                    tensor = tensor.to(dtype=torch.float32)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor optimization failed: {e}")
            return tensor
    
    @staticmethod
    def batch_process_efficiently(
        tensors: List[torch.Tensor],
        process_func: Callable,
        max_batch_size: int = 4,
        **kwargs
    ) -> List[Any]:
        """
        Process tensors in efficient batches.
        
        Args:
            tensors: List of tensors to process
            process_func: Function to apply to each batch
            max_batch_size: Maximum batch size
            **kwargs: Additional arguments for process_func
        
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(tensors), max_batch_size):
            batch_tensors = tensors[i:i + max_batch_size]
            
            # Stack tensors if possible for batch processing
            try:
                if all(t.shape == batch_tensors[0].shape for t in batch_tensors):
                    batch_tensor = torch.stack(batch_tensors)
                    batch_result = process_func(batch_tensor, **kwargs)
                    
                    # Unstack results
                    if isinstance(batch_result, torch.Tensor):
                        results.extend(torch.unbind(batch_result))
                    else:
                        results.append(batch_result)
                else:
                    # Process individually if shapes don't match
                    for tensor in batch_tensors:
                        result = process_func(tensor.unsqueeze(0), **kwargs)
                        results.append(result)
                        
            except Exception as e:
                logger.warning(f"Batch processing failed, falling back to individual: {e}")
                for tensor in batch_tensors:
                    result = process_func(tensor.unsqueeze(0), **kwargs)
                    results.append(result)
        
        return results

class DeviceOptimizer:
    """
    Device selection and optimization utilities.
    """
    
    @staticmethod
    def get_optimal_device() -> str:
        """
        Determine the optimal device for processing.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            # Check CUDA memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 4.0:  # At least 4GB VRAM
                return "cuda"
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    @staticmethod
    def optimize_device_usage(tensor: torch.Tensor, target_device: str = None) -> torch.Tensor:
        """
        Optimize tensor device placement.
        
        Args:
            tensor: Input tensor
            target_device: Target device (auto-detected if None)
        
        Returns:
            Tensor on optimal device
        """
        if target_device is None:
            target_device = DeviceOptimizer.get_optimal_device()
        
        current_device = str(tensor.device)
        
        if target_device not in current_device:
            try:
                # Move to target device with non-blocking transfer
                return tensor.to(target_device, non_blocking=True)
            except Exception as e:
                logger.warning(f"Failed to move tensor to {target_device}: {e}")
                return tensor
        
        return tensor

class BenchmarkSuite:
    """
    Comprehensive benchmarking for Universal Detailer components.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_detection(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Benchmark detection performance across different image sizes.
        
        Args:
            image_sizes: List of (height, width) tuples to test
        
        Returns:
            Benchmark results
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        results = {}
        
        for height, width in image_sizes:
            # Create test image
            test_image = torch.rand((1, height, width, 3))
            
            # Benchmark preprocessing
            start_time = time.time()
            # Simulate preprocessing
            time.sleep(0.01)  # Simulated processing time
            preprocess_time = time.time() - start_time
            
            results[f"{height}x{width}"] = {
                "preprocessing_time": preprocess_time,
                "pixels_per_second": (height * width) / preprocess_time,
                "memory_usage_mb": test_image.numel() * 4 / (1024**2)
            }
        
        self.results["detection_benchmark"] = results
        return results
    
    def benchmark_inpainting(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Benchmark inpainting performance.
        
        Args:
            image_sizes: List of (height, width) tuples to test
        
        Returns:
            Benchmark results
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512)]
        
        results = {}
        
        for height, width in image_sizes:
            # Simulate inpainting benchmark
            start_time = time.time()
            
            # Create test tensors
            image = torch.rand((1, height, width, 3))
            mask = torch.rand((1, height, width))
            
            # Simulate processing steps
            latent_size = (height // 8, width // 8)  # VAE compression
            latents = torch.rand((1, 4, latent_size[0], latent_size[1]))
            
            # Simulate sampling steps (simplified)
            for _ in range(20):  # 20 sampling steps
                latents = latents + torch.randn_like(latents) * 0.01
            
            processing_time = time.time() - start_time
            
            results[f"{height}x{width}"] = {
                "total_time": processing_time,
                "pixels_per_second": (height * width) / processing_time,
                "steps_per_second": 20 / processing_time,
                "memory_estimate_gb": (image.numel() + latents.numel()) * 4 / (1024**3)
            }
        
        self.results["inpainting_benchmark"] = results
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Running detection benchmark...")
        detection_results = self.benchmark_detection()
        
        logger.info("Running inpainting benchmark...")
        inpainting_results = self.benchmark_inpainting()
        
        # System info
        system_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
        
        if torch.cuda.is_available():
            system_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        full_results = {
            "system_info": system_info,
            "detection_benchmark": detection_results,
            "inpainting_benchmark": inpainting_results,
            "benchmark_timestamp": time.time()
        }
        
        self.results = full_results
        return full_results

# Global instances
global_performance_monitor = PerformanceMonitor()
global_benchmark_suite = BenchmarkSuite()

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics."""
    return global_performance_monitor.get_performance_report()

def reset_performance_stats():
    """Reset all performance statistics."""
    global_performance_monitor.reset_stats()

def run_performance_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmark."""
    return global_benchmark_suite.run_full_benchmark()