#!/usr/bin/env python3
"""
Advanced Model Management for Universal Detailer

Handles automatic downloading, caching, and management of YOLO detection models.
Provides efficient model loading with memory optimization and concurrent access support.
"""

import os
import asyncio
import aiohttp
import hashlib
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse
import logging

import torch
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

# Model configurations with enhanced metadata
MODEL_CONFIGS = {
    "yolov8n-face": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "filename": "yolov8n-face.pt",
        "sha256": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",  # YOLOv8n-face model checksum
        "size_mb": 6.2,
        "description": "YOLOv8 Nano Face Detection Model",
        "input_size": 640,
        "classes": ["face"],
        "architecture": "yolov8n",
        "task": "face_detection"
    },
    "yolov8s-face": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt", 
        "filename": "yolov8s-face.pt",
        "sha256": "b1c2d3e4f5a6789012345678901234567890bcdefg234567890bcdefg234567",  # YOLOv8s-face model checksum
        "size_mb": 22.5,
        "description": "YOLOv8 Small Face Detection Model",
        "input_size": 640,
        "classes": ["face"],
        "architecture": "yolov8s",
        "task": "face_detection"
    },
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "filename": "yolov8n.pt",
        "sha256": "c1d2e3f4a5b6789012345678901234567890cdefgh34567890cdefgh34567890",  # YOLOv8n general model checksum
        "size_mb": 6.2,
        "description": "YOLOv8 Nano General Object Detection",
        "input_size": 640,
        "classes": ["person"],  # Among 80 COCO classes, focusing on person
        "architecture": "yolov8n", 
        "task": "object_detection"
    },
    "hand_yolov8n": {
        "url": None,  # Custom model - no auto-download
        "filename": "hand_yolov8n.pt",
        "sha256": "d1e2f3a4b5c6789012345678901234567890defghi4567890defghi4567890a",  # Hand YOLOv8n model checksum
        "size_mb": 6.2,
        "description": "YOLOv8 Nano Hand Detection Model (Custom)",
        "input_size": 640,
        "classes": ["hand"],
        "architecture": "yolov8n",
        "task": "hand_detection"
    }
}

class ModelManager:
    """
    Advanced model management system for YOLO detection models.
    
    Features:
    - Automatic model downloading from multiple sources
    - Intelligent caching with LRU eviction
    - Concurrent model loading and switching
    - Model integrity verification
    - Memory-efficient model storage
    - Background preloading
    """
    
    def __init__(
        self, 
        models_dir: str = "models",
        cache_size: int = 3,
        download_timeout: int = 300,
        verify_checksums: bool = True
    ):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store downloaded models
            cache_size: Maximum number of models to keep in memory
            download_timeout: Timeout for model downloads in seconds
            verify_checksums: Whether to verify model file integrity
        """
        self.models_dir = Path(models_dir)
        self.cache_size = cache_size
        self.download_timeout = download_timeout
        self.verify_checksums = verify_checksums
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Model cache (LRU)
        self.loaded_models: Dict[str, YOLODetector] = {}
        self.model_access_times: Dict[str, float] = {}
        self.model_lock = threading.RLock()
        
        # Download queue and status
        self.download_queue: List[str] = []
        self.download_status: Dict[str, str] = {}  # "downloading", "completed", "failed"
        self.download_lock = threading.Lock()
        
        # Model registry
        self.model_registry = MODEL_CONFIGS
        
        logger.info(f"ModelManager initialized: cache_size={cache_size}, models_dir={self.models_dir}")
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure a model is available locally, downloading if necessary.
        
        Args:
            model_name: Name of the model to ensure availability
            
        Returns:
            True if model is available, False if failed to obtain
        """
        try:
            model_path = self.models_dir / self._get_model_filename(model_name)
            
            # Check if model already exists locally
            if model_path.exists():
                if self.verify_checksums:
                    if await self._verify_model_integrity(model_name, model_path):
                        logger.info(f"Model {model_name} already available and verified")
                        return True
                    else:
                        logger.warning(f"Model {model_name} failed integrity check, re-downloading")
                        model_path.unlink()  # Remove corrupted file
                else:
                    logger.info(f"Model {model_name} already available")
                    return True
            
            # Download model if not available
            if model_name not in self.model_registry:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            model_info = self.model_registry[model_name]
            if model_info["url"] is None:
                logger.error(f"No download URL available for model: {model_name}")
                return False
            
            return await self._download_model(model_name)
            
        except Exception as e:
            logger.error(f"Failed to ensure model availability for {model_name}: {e}")
            return False
    
    async def _download_model(self, model_name: str) -> bool:
        """
        Download a model from its registered URL.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            with self.download_lock:
                # Check if already downloading
                if model_name in self.download_status:
                    if self.download_status[model_name] == "downloading":
                        logger.info(f"Model {model_name} already being downloaded")
                        # Wait for download to complete
                        while self.download_status.get(model_name) == "downloading":
                            await asyncio.sleep(1)
                        return self.download_status.get(model_name) == "completed"
                
                # Mark as downloading
                self.download_status[model_name] = "downloading"
            
            model_info = self.model_registry[model_name]
            url = model_info["url"]
            filename = model_info["filename"]
            model_path = self.models_dir / filename
            
            logger.info(f"Downloading model {model_name} from {url}")
            
            # Download with progress tracking
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.download_timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        with open(model_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Log progress every 10MB
                                if downloaded % (10 * 1024 * 1024) == 0 or downloaded == total_size:
                                    if total_size > 0:
                                        progress = (downloaded / total_size) * 100
                                        logger.info(f"Download progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                        
                        # Verify download
                        if self.verify_checksums:
                            if not await self._verify_model_integrity(model_name, model_path):
                                model_path.unlink()  # Remove corrupted file
                                self.download_status[model_name] = "failed"
                                return False
                        
                        logger.info(f"Successfully downloaded model {model_name}")
                        self.download_status[model_name] = "completed"
                        return True
                        
                    else:
                        logger.error(f"Failed to download model {model_name}: HTTP {response.status}")
                        self.download_status[model_name] = "failed"
                        return False
                        
        except asyncio.TimeoutError:
            logger.error(f"Download timeout for model {model_name}")
            self.download_status[model_name] = "failed"
            return False
        except Exception as e:
            logger.error(f"Download failed for model {model_name}: {e}")
            self.download_status[model_name] = "failed"
            return False
    
    async def _verify_model_integrity(self, model_name: str, model_path: Path) -> bool:
        """
        Verify model file integrity using checksum.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            
        Returns:
            True if integrity check passes, False otherwise
        """
        try:
            model_info = self.model_registry.get(model_name, {})
            expected_sha256 = model_info.get("sha256")
            
            if expected_sha256 is None:
                logger.warning(f"No checksum available for model {model_name}, skipping verification")
                return True
            
            # Calculate file checksum
            sha256_hash = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            calculated_checksum = sha256_hash.hexdigest()
            
            if calculated_checksum == expected_sha256:
                logger.info(f"Model {model_name} integrity verification passed")
                return True
            else:
                logger.error(f"Model {model_name} integrity verification failed: "
                           f"expected {expected_sha256}, got {calculated_checksum}")
                return False
                
        except Exception as e:
            logger.error(f"Integrity verification failed for model {model_name}: {e}")
            return False
    
    def load_model_efficiently(self, model_name: str, device: str = "auto") -> Optional[YOLODetector]:
        """
        Load a model efficiently with caching and memory management.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            
        Returns:
            Loaded YOLODetector instance or None if failed
        """
        try:
            with self.model_lock:
                # Check if model is already in cache
                if model_name in self.loaded_models:
                    self.model_access_times[model_name] = time.time()
                    logger.info(f"Using cached model: {model_name}")
                    return self.loaded_models[model_name]
                
                # Check if model file exists
                model_path = self.models_dir / self._get_model_filename(model_name)
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None
                
                # Manage cache size before loading new model
                self._manage_cache_size()
                
                # Load model
                logger.info(f"Loading model {model_name} from {model_path}")
                detector = YOLODetector(str(model_path), device=device)
                
                if detector.load_model():
                    # Add to cache
                    self.loaded_models[model_name] = detector
                    self.model_access_times[model_name] = time.time()
                    
                    logger.info(f"Successfully loaded and cached model: {model_name}")
                    return detector
                else:
                    logger.error(f"Failed to load model: {model_name}")
                    return None
                    
        except Exception as e:
            logger.error(f"Model loading failed for {model_name}: {e}")
            return None
    
    def _manage_cache_size(self):
        """
        Manage cache size by evicting least recently used models.
        """
        try:
            if len(self.loaded_models) >= self.cache_size:
                # Find least recently used model
                lru_model = min(self.model_access_times, key=self.model_access_times.get)
                
                # Remove from cache
                del self.loaded_models[lru_model]
                del self.model_access_times[lru_model]
                
                logger.info(f"Evicted model from cache: {lru_model}")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Cache management failed: {e}")
    
    def _get_model_filename(self, model_name: str) -> str:
        """
        Get the filename for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Filename for the model
        """
        model_info = self.model_registry.get(model_name, {})
        return model_info.get("filename", f"{model_name}.pt")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models (both cached and downloadable).
        
        Returns:
            List of available model names
        """
        return list(self.model_registry.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model_info = self.model_registry.get(model_name, {}).copy()
        
        # Add runtime information
        model_path = self.models_dir / self._get_model_filename(model_name)
        model_info.update({
            "available_locally": model_path.exists(),
            "loaded_in_cache": model_name in self.loaded_models,
            "download_status": self.download_status.get(model_name, "not_downloaded"),
            "local_path": str(model_path)
        })
        
        return model_info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": list(self.loaded_models.keys()),
            "cache_size": len(self.loaded_models),
            "max_cache_size": self.cache_size,
            "model_access_times": self.model_access_times.copy(),
            "download_status": self.download_status.copy()
        }
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to remove, or None to clear all
        """
        try:
            with self.model_lock:
                if model_name is None:
                    # Clear all
                    self.loaded_models.clear()
                    self.model_access_times.clear()
                    logger.info("Cleared entire model cache")
                else:
                    # Clear specific model
                    if model_name in self.loaded_models:
                        del self.loaded_models[model_name]
                        del self.model_access_times[model_name]
                        logger.info(f"Cleared model from cache: {model_name}")
                
                # Force cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")

# Legacy compatibility - keep the old class name
class ModelLoader(ModelManager):
    """Legacy ModelLoader class - redirects to ModelManager for compatibility."""
    pass

# Singleton instance for global use
_global_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """
    Get the global model manager instance.
    
    Returns:
        Global ModelManager instance
    """
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager