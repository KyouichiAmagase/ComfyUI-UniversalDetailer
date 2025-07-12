#!/usr/bin/env python3
"""
YOLO Detector Implementation

Production-ready YOLO-based detection for faces, hands, and other body parts.
Optimized for speed and accuracy in ComfyUI workflows.

Features:
- Ultralytics YOLO integration
- Multi-part detection support
- Confidence-based filtering
- Device optimization (CPU/GPU)
- Memory-efficient processing
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from functools import lru_cache, wraps
import time
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

def performance_timer(operation_name: str = None):
    """Decorator for automatic performance timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if duration > 1.0:
                    logger.info(f"{name} took {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

class YOLODetector:
    """
    YOLO-based detector for faces, hands, and body parts.
    
    This class handles:
    - Loading YOLO models
    - Running inference
    - Processing detection results
    - Filtering by confidence threshold
    
    Optimizations:
    - Weak reference caching for models
    - Lazy initialization
    - Performance monitoring
    - Memory-efficient processing
    """
    
    _model_cache = WeakValueDictionary()
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize YOLO detector with optimizations.
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.model = None
        self._model_loaded = False
        self._last_inference_time = 0
        
        logger.info(f"YOLODetector initialized with model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    @staticmethod
    @lru_cache(maxsize=8)
    def _determine_device(device: str) -> str:
        """
        Determine the best device for inference.
        
        Args:
            device: Requested device
        
        Returns:
            Actual device string
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @performance_timer("model_loading")
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check cache first
            cache_key = f"{self.model_path}_{self.device}"
            if cache_key in self._model_cache:
                self.model = self._model_cache[cache_key]
                self._model_loaded = True
                logger.info(f"YOLO model loaded from cache: {self.model_path}")
                return True
            
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move model to specified device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # Cache the model
            self._model_cache[cache_key] = self.model
            self._model_loaded = True
            
            logger.info(f"YOLO model loaded successfully on device: {self.device}")
            return True
            
        except ImportError as e:
            logger.error(f"ultralytics not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    @performance_timer("detection")
    def detect(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = 0.5,
        target_classes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in the image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detections
            target_classes: List of target class names
        
        Returns:
            List of detection dictionaries with bbox, confidence, class
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            logger.info(f"Running detection with confidence threshold: {confidence_threshold}")
            
            # Optimize image for detection
            optimized_image = self._optimize_image_for_detection(image)
            
            # Run YOLO inference with memory optimization
            with torch.no_grad():
                results = self.model(optimized_image, conf=confidence_threshold, verbose=False)
            
            # Process results
            detections = self._process_results(results, target_classes)
            
            # Update inference timing
            self._last_inference_time = time.time()
            
            logger.info(f"Found {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    @lru_cache(maxsize=64)
    def _optimize_image_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image for YOLO detection.
        
        Args:
            image: Input image array
            
        Returns:
            Optimized image array
        """
        try:
            # Ensure proper dtype
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # Ensure contiguous memory layout
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}")
            return image
    
    def _process_results(self, results: Any, target_classes: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process YOLO results into standardized format.
        
        Args:
            results: Raw YOLO results
            target_classes: Filter for specific classes
        
        Returns:
            List of processed detection dictionaries
        """
        detections = []
        
        try:
            for result in results:
                if result.boxes is None:
                    continue
                    
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Extract box coordinates (x1, y1, x2, y2)
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name if available
                    class_name = "unknown"
                    if hasattr(result, 'names') and class_id in result.names:
                        class_name = result.names[class_id]
                    
                    # Map class names to detection types
                    detection_type = self._map_class_to_type(class_name)
                    
                    # Filter by target classes if specified
                    if target_classes is not None:
                        if detection_type not in target_classes and class_name not in target_classes:
                            continue
                    
                    detection = {
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "type": detection_type,
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    
                    detections.append(detection)
                    
        except Exception as e:
            logger.error(f"Error processing YOLO results: {e}")
            
        return detections
    
    @lru_cache(maxsize=128)
    def _map_class_to_type(self, class_name: str) -> str:
        """
        Map YOLO class names to detection types.
        
        Args:
            class_name: YOLO class name
            
        Returns:
            Detection type string
        """
        class_name_lower = class_name.lower()
        
        if "face" in class_name_lower or "head" in class_name_lower:
            return "face"
        elif "hand" in class_name_lower:
            return "hand"
        elif "finger" in class_name_lower:
            return "finger"
        elif "person" in class_name_lower:
            return "person"
        else:
            return "other"
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        classes = []
        if self.model is not None and hasattr(self.model, 'names'):
            classes = list(self.model.names.values()) if self.model.names else []
            
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "loaded": self.is_loaded(),
            "classes": classes,
            "cache_size": len(self._model_cache),
            "last_inference": self._last_inference_time,
        }
        
        # Add performance metrics if available
        if hasattr(self, '_detection_times'):
            info["avg_detection_time"] = np.mean(self._detection_times)
        
        return info
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self._model_cache.clear()
        logger.info("YOLO detector cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                allocated = torch.cuda.memory_allocated() / (1024**3)
                cached = torch.cuda.memory_reserved() / (1024**3)
                return {
                    "gpu_allocated_gb": allocated,
                    "gpu_cached_gb": cached,
                    "gpu_free_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) - allocated
                }
            else:
                import psutil
                process = psutil.Process()
                return {
                    "cpu_memory_gb": process.memory_info().rss / (1024**3),
                    "cpu_memory_percent": process.memory_percent()
                }
        except Exception as e:
            logger.warning(f"Memory usage calculation failed: {e}")
            return {}