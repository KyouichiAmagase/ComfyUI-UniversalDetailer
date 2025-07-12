#!/usr/bin/env python3
"""
Error Handling Utilities for ComfyUI Universal Detailer

Provides comprehensive error handling, logging, and recovery mechanisms
for robust operation in production environments.
"""

import logging
import traceback
import functools
import sys
import time
from typing import Any, Callable, Dict, Optional, Union, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class UniversalDetailerError(Exception):
    """Base exception class for Universal Detailer errors."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN", details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "error": True,
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }

class ModelLoadError(UniversalDetailerError):
    """Error during model loading."""
    pass

class DetectionError(UniversalDetailerError):
    """Error during object detection."""
    pass

class InpaintingError(UniversalDetailerError):
    """Error during inpainting process."""
    pass

class MaskGenerationError(UniversalDetailerError):
    """Error during mask generation."""
    pass

class MemoryError(UniversalDetailerError):
    """Memory-related errors."""
    pass

class ErrorHandler:
    """
    Comprehensive error handling and recovery system.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.error_counts = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file or 'universal_detailer.log')
            ]
        )
    
    def handle_error(
        self,
        error: Exception,
        context: str = "Unknown",
        fallback_value: Any = None,
        raise_on_critical: bool = True
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Handle errors with context and recovery options.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
            fallback_value: Value to return on recoverable errors
            raise_on_critical: Whether to re-raise critical errors
        
        Returns:
            Tuple of (success, value, error_info)
        """
        error_info = self._create_error_info(error, context)
        
        # Log error
        self._log_error(error, context, error_info)
        
        # Update error counts
        self._update_error_counts(error_info['error_code'])
        
        # Determine if error is critical
        is_critical = self._is_critical_error(error)
        
        if is_critical and raise_on_critical:
            raise error
        
        return False, fallback_value, error_info
    
    def _create_error_info(self, error: Exception, context: str) -> Dict[str, Any]:
        """Create comprehensive error information."""
        error_code = getattr(error, 'error_code', 'UNKNOWN')
        
        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_code": error_code,
            "message": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc(),
            "details": getattr(error, 'details', {})
        }
    
    def _log_error(self, error: Exception, context: str, error_info: Dict[str, Any]):
        """Log error with appropriate level."""
        if self._is_critical_error(error):
            logger.critical(f"Critical error in {context}: {error}")
            logger.critical(f"Traceback: {error_info['traceback']}")
        else:
            logger.error(f"Error in {context}: {error}")
            logger.debug(f"Error details: {error_info}")
    
    def _update_error_counts(self, error_code: str):
        """Update error occurrence counts."""
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
    
    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if an error is critical."""
        critical_types = (
            MemoryError,
            SystemError,
            KeyboardInterrupt,
            SystemExit
        )
        
        # Critical if it's a system-level error
        if isinstance(error, critical_types):
            return True
        
        # Critical if error count exceeds threshold
        error_code = getattr(error, 'error_code', 'UNKNOWN')
        if self.error_counts.get(error_code, 0) > 5:
            return True
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "most_common": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }

def with_error_handling(
    context: str = None,
    fallback_value: Any = None,
    error_handler: ErrorHandler = None,
    raise_on_critical: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        context: Context description for the function
        fallback_value: Value to return on error
        error_handler: ErrorHandler instance to use
        raise_on_critical: Whether to re-raise critical errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            func_context = context or f"{func.__module__}.{func.__name__}"
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                success, value, error_info = handler.handle_error(
                    e, func_context, fallback_value, raise_on_critical
                )
                
                if not success:
                    return value
                
        return wrapper
    return decorator

def safe_execute(
    func: Callable,
    *args,
    context: str = None,
    fallback_value: Any = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> Tuple[bool, Any, Optional[Dict[str, Any]]]:
    """
    Safely execute a function with retries and error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        context: Context description
        fallback_value: Value to return on failure
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (success, result, error_info)
    """
    handler = ErrorHandler()
    func_context = context or f"{func.__module__}.{func.__name__}"
    
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return True, result, None
            
        except Exception as e:
            if attempt == max_retries:
                # Final attempt failed
                success, value, error_info = handler.handle_error(
                    e, func_context, fallback_value, raise_on_critical=False
                )
                return success, value, error_info
            else:
                # Retry after delay
                logger.warning(f"Attempt {attempt + 1} failed for {func_context}, retrying...")
                time.sleep(retry_delay)
    
    return False, fallback_value, {"error": True, "message": "Max retries exceeded"}

def create_safe_fallback_result(
    image_shape: Tuple[int, ...],
    error_message: str = "Processing failed"
) -> Tuple[Any, ...]:
    """
    Create safe fallback result for Universal Detailer processing.
    
    Args:
        image_shape: Shape of the input image
        error_message: Error message to include
    
    Returns:
        Tuple matching Universal Detailer return format
    """
    import torch
    
    # Create empty tensors matching expected shapes
    batch_size = image_shape[0] if len(image_shape) >= 4 else 1
    height = image_shape[1] if len(image_shape) >= 4 else 512
    width = image_shape[2] if len(image_shape) >= 4 else 512
    channels = image_shape[3] if len(image_shape) >= 4 else 3
    
    # Fallback image (black image)
    fallback_image = torch.zeros((batch_size, height, width, channels))
    
    # Empty masks
    empty_mask = torch.zeros((batch_size, height, width))
    
    # Error information
    error_info = {
        "error": True,
        "message": error_message,
        "timestamp": time.time(),
        "fallback_result": True
    }
    
    return (
        fallback_image,      # processed_image
        empty_mask,          # detection_masks
        empty_mask,          # face_masks  
        empty_mask,          # hand_masks
        json.dumps(error_info, indent=2)  # detection_info
    )

# Global error handler instance
global_error_handler = ErrorHandler()

def log_performance_warning(operation: str, duration: float, threshold: float = 30.0):
    """Log performance warnings for slow operations."""
    if duration > threshold:
        logger.warning(f"Performance warning: {operation} took {duration:.2f}s (threshold: {threshold}s)")

def validate_tensor_inputs(*tensors, expected_shapes: Dict[str, Tuple] = None):
    """
    Validate tensor inputs for shape and type consistency.
    
    Args:
        *tensors: Tensors to validate
        expected_shapes: Expected shapes for specific tensors
    
    Raises:
        ValueError: If tensors don't meet requirements
    """
    import torch
    
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Tensor {i} is not a torch.Tensor, got {type(tensor)}")
        
        if expected_shapes and str(i) in expected_shapes:
            expected = expected_shapes[str(i)]
            if tensor.shape != expected:
                raise ValueError(f"Tensor {i} shape mismatch: expected {expected}, got {tensor.shape}")
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any():
            raise ValueError(f"Tensor {i} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValueError(f"Tensor {i} contains infinite values")