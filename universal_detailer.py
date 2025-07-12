#!/usr/bin/env python3
"""
Universal Detailer Node Implementation

Production-ready ComfyUI custom node for multi-part detection and enhancement.
Supports face, hand, and finger detection with advanced inpainting capabilities.

Features:
- YOLO-based detection for multiple body parts
- Advanced mask generation with padding and blur
- ComfyUI-integrated sampling pipeline  
- Memory-efficient batch processing
- Comprehensive error handling and recovery

Version: 2.0.0 - Production Ready
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
import json
import logging
import traceback
import time
from pathlib import Path
from functools import lru_cache, wraps

# Import detection and masking components
from .detection.yolo_detector import YOLODetector
from .masking.mask_generator import MaskGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance optimization decorators
def profile_performance(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if duration > 1.0:  # Log slow operations
                logger.info(f"{func.__name__} took {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

@lru_cache(maxsize=32)
def _get_cached_device_info() -> Dict[str, Any]:
    """Cache device information for performance."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

class UniversalDetailerNode:
    """
    Universal Detailer Node for ComfyUI
    
    Production-ready implementation supporting multi-part detection and enhancement.
    Optimized for performance, memory efficiency, and reliability.
    
    Capabilities:
    - Multi-part detection (face, hand, finger)
    - Advanced inpainting with ComfyUI integration
    - Memory-optimized batch processing
    - Comprehensive error handling
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define input types for the node.
        
        Returns:
            Dict containing 'required' and 'optional' input specifications
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                "detection_model": (
                    ["yolov8n-face", "yolov8s-face", "hand_yolov8n"], 
                    {"default": "yolov8n-face"}
                ),
                "target_parts": (
                    "STRING", 
                    {"default": "face,hand"}
                ),
                "confidence_threshold": (
                    "FLOAT", 
                    {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05}
                ),
                "mask_padding": (
                    "INT", 
                    {"default": 32, "min": 0, "max": 128, "step": 4}
                ),
                "inpaint_strength": (
                    "FLOAT", 
                    {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05}
                ),
                "steps": (
                    "INT", 
                    {"default": 20, "min": 1, "max": 100, "step": 1}
                ),
                "cfg_scale": (
                    "FLOAT", 
                    {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}
                ),
                "seed": (
                    "INT", 
                    {"default": -1, "min": -1, "max": 0xffffffffffffffff}
                ),
                "sampler_name": (
                    ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"], 
                    {"default": "euler"}
                ),
                "scheduler": (
                    ["normal", "karras", "exponential", "sgm_uniform"], 
                    {"default": "normal"}
                ),
                "auto_face_fix": (
                    "BOOLEAN", 
                    {"default": True}
                ),
                "auto_hand_fix": (
                    "BOOLEAN", 
                    {"default": True}
                ),
                "mask_blur": (
                    "INT", 
                    {"default": 4, "min": 0, "max": 20, "step": 1}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "detection_masks", "face_masks", "hand_masks", "detection_info")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    def __init__(self):
        """Initialize the Universal Detailer node with optimized caching."""
        # Use weak references for better memory management
        from weakref import WeakValueDictionary
        
        self.detection_models = WeakValueDictionary()
        self.model_cache = {}
        self._mask_generator = None  # Lazy initialization
        self._device_info = None     # Cache device info
        
        # Initialize paths
        self.base_path = Path(__file__).parent
        self.models_path = self.base_path / "models"
        self.cache_path = self.base_path / "cache"
        
        # Create directories if they don't exist (only once)
        self.models_path.mkdir(exist_ok=True)
        self.cache_path.mkdir(exist_ok=True)
        
        logger.info("Universal Detailer node initialized with optimizations")
    
    @property
    def mask_generator(self) -> MaskGenerator:
        """Lazy initialization of mask generator."""
        if self._mask_generator is None:
            self._mask_generator = MaskGenerator()
        return self._mask_generator
    
    @property 
    def device_info(self) -> Dict[str, Any]:
        """Cached device information."""
        if self._device_info is None:
            self._device_info = _get_cached_device_info()
        return self._device_info
    
    @profile_performance
    def process(
        self,
        image: torch.Tensor,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        detection_model: str = "yolov8n-face",
        target_parts: str = "face,hand",
        confidence_threshold: float = 0.5,
        mask_padding: int = 32,
        inpaint_strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        auto_face_fix: bool = True,
        auto_hand_fix: bool = True,
        mask_blur: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Main processing function for universal detection and correction.
        
        ⚠️  SKELETON IMPLEMENTATION - NEEDS COMPLETION BY AI DEVELOPER
        
        This is a placeholder implementation that demonstrates the expected
        interface. The actual implementation should follow the specifications
        in SPECIFICATIONS.md and include:
        
        1. Image preprocessing
        2. Multi-part detection using YOLO
        3. Mask generation
        4. Inpainting processing
        5. Result composition
        6. Error handling
        
        Args:
            image: Input image tensor
            model: Inpainting model
            vae: VAE encoder/decoder
            positive: Positive conditioning
            negative: Negative conditioning
            **kwargs: Optional parameters as defined in INPUT_TYPES
        
        Returns:
            Tuple of (processed_image, detection_masks, face_masks, hand_masks, detection_info)
        
        Raises:
            NotImplementedError: This is a skeleton implementation
        """
        
        try:
            # Import memory manager for performance monitoring
            try:
                from .utils.memory_utils import MemoryManager
                memory_manager = MemoryManager()
            except ImportError:
                memory_manager = None
            
            # Monitor memory usage throughout processing
            if memory_manager:
                memory_manager.log_memory_usage("start processing")
            
            start_time = time.time()
            logger.info("Starting Universal Detailer processing...")
            logger.info(f"Target parts: {target_parts}")
            logger.info(f"Detection model: {detection_model}")
            logger.info(f"Confidence threshold: {confidence_threshold}")
            
            # Validate parameters
            validated_params = self._validate_parameters(
                detection_model=detection_model,
                target_parts=target_parts,
                confidence_threshold=confidence_threshold,
                mask_padding=mask_padding,
                inpaint_strength=inpaint_strength,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sampler_name=sampler_name,
                scheduler=scheduler,
                mask_blur=mask_blur
            )
            
            # Get image dimensions and optimize batch size
            batch_size, height, width, channels = image.shape
            logger.info(f"Processing image: {batch_size}x{height}x{width}x{channels}")
            
            # Estimate optimal batch size for memory efficiency
            if memory_manager:
                optimal_batch_size = memory_manager.estimate_batch_size(height, width, channels)
                if batch_size > optimal_batch_size:
                    logger.warning(f"Large batch size {batch_size} may cause memory issues. "
                                 f"Recommended: {optimal_batch_size}")
            
            # Parse target parts
            target_parts_list = [part.strip() for part in target_parts.split(",")]
            
            # Load detection model with memory monitoring
            if memory_manager:
                with memory_manager.memory_monitor("model loading"):
                    detector = self._load_detection_model(detection_model)
            else:
                detector = self._load_detection_model(detection_model)
                
            if detector is None:
                raise RuntimeError(f"Failed to load detection model: {detection_model}")
            
            # Process images efficiently (batch or sequential based on memory)
            processed_results = self._process_batch_efficiently(
                image, detector, target_parts_list, validated_params,
                model, vae, positive, negative, memory_manager
            )
            
            processed_image, combined_masks, face_masks, hand_masks, detections = processed_results
            
            # Calculate processing statistics
            processing_time = time.time() - start_time
            face_count = len([d for d in detections if d.get("type") == "face"])
            hand_count = len([d for d in detections if d.get("type") == "hand"])
            
            # Create detection info with performance metrics
            detection_info = {
                "error": False,
                "total_detections": len(detections),
                "faces_detected": face_count,
                "hands_detected": hand_count,
                "processing_time": round(processing_time, 2),
                "image_shape": [batch_size, height, width, channels],
                "parameters": {
                    "detection_model": detection_model,
                    "target_parts": target_parts,
                    "confidence_threshold": confidence_threshold,
                    "mask_padding": mask_padding,
                    "mask_blur": mask_blur,
                    "inpaint_strength": inpaint_strength
                },
                "detections": detections,
                "performance_metrics": {
                    "processing_time_seconds": round(processing_time, 2),
                    "pixels_processed": height * width * batch_size,
                    "pixels_per_second": round((height * width * batch_size) / processing_time),
                    "memory_efficient": memory_manager is not None
                }
            }
            
            # Final memory cleanup
            if memory_manager:
                memory_manager.cleanup_memory()
                memory_manager.log_memory_usage("end processing")
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(f"Found {len(detections)} total detections ({face_count} faces, {hand_count} hands)")
            
            return (
                processed_image,
                combined_masks,
                face_masks,
                hand_masks,
                json.dumps(detection_info, indent=2)
            )
            
        except Exception as e:
            logger.error(f"Error in Universal Detailer processing: {e}")
            logger.error(traceback.format_exc())
            
            # Use comprehensive error handling
            try:
                from .utils.error_handling import create_safe_fallback_result, global_error_handler
                
                # Handle error with context
                success, fallback_result, error_info = global_error_handler.handle_error(
                    e, 
                    context="Universal Detailer main processing",
                    fallback_value=None,
                    raise_on_critical=False
                )
                
                # Create safe fallback result
                return create_safe_fallback_result(
                    image_shape=image.shape,
                    error_message=f"Processing failed: {str(e)}"
                )
                
            except ImportError:
                # Fallback to original error handling if error_handling module not available
                batch_size, height, width, channels = image.shape
                empty_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
                
                error_info = {
                    "error": True,
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "processing_completed": False,
                    "timestamp": time.time()
                }
                
                return (
                    image,
                    empty_mask,
                    empty_mask,
                    empty_mask,
                    json.dumps(error_info, indent=2)
                )
    
    def _validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and sanitize input parameters.
        
        Args:
            **kwargs: Input parameters to validate
        
        Returns:
            Dict of validated parameters
        """
        validated = {}
        
        # Validate confidence threshold
        confidence = kwargs.get("confidence_threshold", 0.5)
        validated["confidence_threshold"] = max(0.1, min(0.95, confidence))
        
        # Validate mask padding
        padding = kwargs.get("mask_padding", 32)
        validated["mask_padding"] = max(0, min(128, padding))
        
        # Validate inpaint strength
        strength = kwargs.get("inpaint_strength", 0.75)
        validated["inpaint_strength"] = max(0.1, min(1.0, strength))
        
        # Validate steps
        steps = kwargs.get("steps", 20)
        validated["steps"] = max(1, min(100, steps))
        
        # Validate CFG scale
        cfg_scale = kwargs.get("cfg_scale", 7.0)
        validated["cfg_scale"] = max(1.0, min(30.0, cfg_scale))
        
        # Validate mask blur
        blur = kwargs.get("mask_blur", 4)
        validated["mask_blur"] = max(0, min(20, blur))
        
        # Copy other parameters as-is
        for key in ["detection_model", "target_parts", "seed", "sampler_name", "scheduler"]:
            if key in kwargs:
                validated[key] = kwargs[key]
        
        return validated
    
    def _load_detection_model(self, model_name: str):
        """
        Load and cache detection model using the advanced ModelManager.
        
        Args:
            model_name: Name of the detection model to load
        
        Returns:
            Loaded model object
        """
        try:
            from .detection.model_loader import get_model_manager
            
            # Get the global model manager
            model_manager = get_model_manager()
            
            # Try to load model efficiently (with caching)
            detector = model_manager.load_model_efficiently(model_name, device=self._determine_device("auto"))
            
            if detector is not None:
                logger.info(f"Successfully loaded detection model: {model_name}")
                return detector
            else:
                logger.warning(f"Model {model_name} not available locally, attempting async download...")
                
                # Implement async download handling
                import asyncio
                import threading
                
                def download_and_load():
                    """Download model in background thread."""
                    try:
                        # Create event loop for async operations
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Attempt async download
                        success = loop.run_until_complete(
                            model_manager.ensure_model_available(model_name)
                        )
                        
                        if success:
                            # Try loading again after download
                            detector = model_manager.load_model_efficiently(
                                model_name, device=self._determine_device("auto")
                            )
                            return detector
                        else:
                            logger.warning(f"Async download failed for {model_name}")
                            return None
                            
                    except Exception as e:
                        logger.error(f"Async download error: {e}")
                        return None
                    finally:
                        loop.close()
                
                # For now, call synchronously but structured for future async support
                try:
                    downloaded_detector = download_and_load()
                    if downloaded_detector is not None:
                        return downloaded_detector
                except Exception as e:
                    logger.error(f"Download and load failed: {e}")
                
                # Final fallback to old method
                logger.info("Falling back to traditional model loading method")
                return self._load_detection_model_fallback(model_name)
                
        except ImportError as e:
            logger.error(f"Failed to import ModelManager: {e}")
            return self._load_detection_model_fallback(model_name)
        except Exception as e:
            logger.error(f"Error loading detection model {model_name}: {e}")
            return self._load_detection_model_fallback(model_name)
    
    def _load_detection_model_fallback(self, model_name: str):
        """
        Fallback method for loading detection models.
        
        Args:
            model_name: Name of the detection model to load
        
        Returns:
            Loaded model object
        """
        # Check if model is already cached in old cache
        if model_name in self.detection_models:
            logger.info(f"Using cached detection model: {model_name}")
            return self.detection_models[model_name]
        
        try:
            # Define model paths based on model name
            model_mapping = {
                "yolov8n-face": "yolov8n-face.pt",
                "yolov8s-face": "yolov8s-face.pt", 
                "hand_yolov8n": "hand_yolov8n.pt"
            }
            
            model_filename = model_mapping.get(model_name, f"{model_name}.pt")
            model_path = self.models_path / model_filename
            
            # If model doesn't exist locally, try to load from ultralytics hub or web
            if not model_path.exists():
                logger.info(f"Model {model_filename} not found locally, attempting to download...")
                model_path = model_name  # Let ultralytics handle the download
            
            # Create detector instance
            detector = YOLODetector(str(model_path))
            
            # Load the model
            if detector.load_model():
                self.detection_models[model_name] = detector
                logger.info(f"Successfully loaded detection model: {model_name}")
                return detector
            else:
                logger.error(f"Failed to load detection model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading detection model {model_name}: {e}")
            return None
    
    @lru_cache(maxsize=32)
    def _determine_optimal_batch_size(self, height: int, width: int, channels: int) -> int:
        """Determine optimal batch size based on image dimensions and available memory."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = gpu_memory - torch.cuda.memory_allocated()
                
                # Estimate memory per image (rough calculation)
                pixels_per_image = height * width * channels
                memory_per_image = pixels_per_image * 16  # Conservative estimate including overhead
                
                # Calculate safe batch size with 70% memory usage
                safe_batch_size = max(1, int((free_memory * 0.7) // memory_per_image))
                return min(safe_batch_size, 8)  # Cap at 8 for stability
            else:
                # CPU processing - conservative batch size
                return min(2, max(1, 16384 // (height * width // 1000)))
        except Exception as e:
            logger.warning(f"Failed to determine optimal batch size: {e}")
            return 1

    def _determine_device(self, device: str) -> str:
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
    
    @profile_performance
    def _detect_parts(self, image_np: np.ndarray, detector: YOLODetector, target_parts: List[str], confidence_threshold: float):
        """
        Detect target parts in the image.
        
        Args:
            image_np: Input image as numpy array
            detector: Detection model instance
            target_parts: List of parts to detect
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            List of detection results
        """
        try:
            logger.info(f"Detecting parts: {target_parts} with confidence >= {confidence_threshold}")
            
            # Run detection
            detections = detector.detect(
                image_np,
                confidence_threshold=confidence_threshold,
                target_classes=target_parts
            )
            
            logger.info(f"Detection completed, found {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error in part detection: {e}")
            return []
    
    def _generate_masks(self, detections: List, image_shape: Tuple, mask_padding: int, mask_blur: int):
        """
        Generate masks from detection results.
        
        Args:
            detections: List of detection results
            image_shape: Shape of the input image
            mask_padding: Padding to add to masks
            mask_blur: Blur amount for mask edges
        
        Returns:
            Tuple of mask tensors
        """
        try:
            logger.info(f"Generating masks for {len(detections)} detections")
            
            # Use mask generator to create masks
            combined_masks, face_masks, hand_masks = self.mask_generator.generate_masks(
                detections=detections,
                image_shape=image_shape,
                padding=mask_padding,
                blur=mask_blur
            )
            
            logger.info("Mask generation completed")
            return combined_masks, face_masks, hand_masks
            
        except Exception as e:
            logger.error(f"Error in mask generation: {e}")
            batch_size, height, width, channels = image_shape
            empty_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
            return empty_mask, empty_mask, empty_mask
    
    @profile_performance
    def _inpaint_regions(self, image: torch.Tensor, masks: torch.Tensor, model, vae, positive, negative, **kwargs):
        """
        Inpaint the masked regions using ComfyUI models.
        
        Args:
            image: Input image tensor (B, H, W, C)
            masks: Mask tensor (B, H, W) - 1.0 for areas to inpaint
            model: ComfyUI diffusion model
            vae: ComfyUI VAE encoder/decoder
            positive: Positive conditioning
            negative: Negative conditioning
            **kwargs: Additional inpainting parameters
        
        Returns:
            Inpainted image tensor (B, H, W, C)
        """
        try:
            from .utils.comfyui_integration import ComfyUIHelper
            from .utils.sampling_utils import SamplingUtils
            from .utils.memory_utils import MemoryManager
            
            memory_manager = MemoryManager()
            
            with memory_manager.memory_monitor("inpainting"):
                logger.info("Starting ComfyUI inpainting process")
                
                # Get sampling parameters
                sampling_params = SamplingUtils.prepare_sampling_params(
                    steps=kwargs.get('steps', 20),
                    cfg_scale=kwargs.get('cfg_scale', 7.0),
                    sampler_name=kwargs.get('sampler_name', 'euler'),
                    scheduler=kwargs.get('scheduler', 'normal'),
                    seed=kwargs.get('seed', -1)
                )
                
                inpaint_strength = kwargs.get('inpaint_strength', 0.75)
                
                # Step 1: VAE encode image to latent space
                logger.info("Step 1: Encoding image to latent space")
                original_latents = ComfyUIHelper.encode_with_vae(vae, image)
                
                # Step 2: Apply mask to latents and add noise
                logger.info("Step 2: Applying mask and adding noise")
                
                # Create noise generator for reproducible results
                generator = torch.Generator(device=image.device)
                generator.manual_seed(sampling_params['seed'])
                
                # Prepare noise for masked regions
                noised_latents = SamplingUtils.prepare_noise_for_inpainting(
                    original_latents,
                    masks,
                    noise_strength=inpaint_strength,
                    generator=generator
                )
                
                # Step 3: Prepare conditioning
                logger.info("Step 3: Preparing conditioning")
                pos_cond, neg_cond = SamplingUtils.prepare_conditioning(
                    positive, negative, masks
                )
                
                # Step 4: Run diffusion sampling
                logger.info("Step 4: Running diffusion sampling")
                sampled_latents = SamplingUtils.sample_with_model(
                    model=model,
                    latents=noised_latents,
                    positive=pos_cond,
                    negative=neg_cond,
                    sampling_params=sampling_params,
                    mask=masks,
                    original_latents=original_latents
                )
                
                # Step 5: Blend sampled latents with original (for better seamless integration)
                logger.info("Step 5: Blending latents")
                blended_latents = SamplingUtils.blend_latents(
                    sampled_latents=sampled_latents,
                    original_latents=original_latents,
                    mask=masks,
                    blend_strength=inpaint_strength
                )
                
                # Step 6: VAE decode back to image space
                logger.info("Step 6: Decoding latents to image")
                inpainted_image = ComfyUIHelper.decode_with_vae(vae, blended_latents)
                
                # Step 7: Final image blending for seamless results
                logger.info("Step 7: Final image blending")
                final_image = self._blend_images(image, inpainted_image, masks, inpaint_strength)
                
                # Cleanup memory
                memory_manager.cleanup_memory()
                
                logger.info("ComfyUI inpainting process completed successfully")
                return final_image
                
        except ImportError as e:
            logger.error(f"Failed to import utility modules: {e}")
            logger.warning("Falling back to original image")
            return image
            
        except Exception as e:
            logger.error(f"Error in ComfyUI inpainting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to original image")
            return image
    
    def _blend_images(
        self, 
        original: torch.Tensor, 
        inpainted: torch.Tensor, 
        mask: torch.Tensor, 
        blend_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Blend original and inpainted images using mask.
        
        Args:
            original: Original image tensor (B, H, W, C)
            inpainted: Inpainted image tensor (B, H, W, C)
            mask: Mask tensor (B, H, W) - 1.0 for inpainted areas
            blend_strength: Blending strength (0.0 = original, 1.0 = inpainted)
            
        Returns:
            Blended image tensor (B, H, W, C)
        """
        try:
            # Ensure mask has the correct dimensions
            if len(mask.shape) == 3 and len(original.shape) == 4:
                # Expand mask to match image channels
                mask_expanded = mask.unsqueeze(-1).expand_as(original)
            else:
                mask_expanded = mask
            
            # Apply blend strength
            effective_mask = mask_expanded * blend_strength
            
            # Blend images
            blended = original * (1.0 - effective_mask) + inpainted * effective_mask
            
            # Ensure values are in valid range
            blended = torch.clamp(blended, 0.0, 1.0)
            
            logger.info(f"Blended images with strength {blend_strength}")
            return blended
            
        except Exception as e:
            logger.error(f"Image blending failed: {e}")
            return original
    
    def _process_batch_efficiently(
        self, 
        image: torch.Tensor, 
        detector, 
        target_parts_list: List[str], 
        validated_params: Dict[str, Any],
        model, 
        vae, 
        positive, 
        negative, 
        memory_manager=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Process batch of images efficiently based on available memory.
        
        Args:
            image: Input image tensor (B, H, W, C)
            detector: Detection model
            target_parts_list: List of target parts to detect
            validated_params: Validated processing parameters
            model: ComfyUI diffusion model
            vae: ComfyUI VAE
            positive: Positive conditioning
            negative: Negative conditioning
            memory_manager: Optional memory manager
            
        Returns:
            Tuple of processed results
        """
        try:
            batch_size, height, width, channels = image.shape
            confidence_threshold = validated_params.get('confidence_threshold', 0.5)
            mask_padding = validated_params.get('mask_padding', 32)
            mask_blur = validated_params.get('mask_blur', 4)
            
            # Initialize result tensors
            processed_images = []
            all_combined_masks = []
            all_face_masks = []
            all_hand_masks = []
            all_detections = []
            
            # Process each image in the batch
            for batch_idx in range(batch_size):
                if memory_manager:
                    memory_manager.log_memory_usage(f"processing image {batch_idx + 1}/{batch_size}")
                
                # Extract single image from batch
                single_image = image[batch_idx:batch_idx+1]  # Keep batch dimension
                
                # Convert to numpy for detection
                image_np = self._tensor_to_numpy(single_image[0])
                
                # Detect parts
                detections = self._detect_parts(
                    image_np, 
                    detector, 
                    target_parts_list, 
                    confidence_threshold
                )
                
                # Generate masks for this image
                combined_mask, face_mask, hand_mask = self._generate_masks(
                    detections, 
                    single_image.shape, 
                    mask_padding, 
                    mask_blur
                )
                
                # Process inpainting if detections found
                if torch.any(combined_mask > 0):
                    logger.info(f"Processing inpainting for image {batch_idx + 1} with {len(detections)} detections")
                    processed_single = self._inpaint_regions(
                        single_image,
                        combined_mask,
                        model,
                        vae,
                        positive,
                        negative,
                        **validated_params
                    )
                else:
                    logger.info(f"No detections for image {batch_idx + 1}, keeping original")
                    processed_single = single_image
                
                # Collect results
                processed_images.append(processed_single)
                all_combined_masks.append(combined_mask)
                all_face_masks.append(face_mask)
                all_hand_masks.append(hand_mask)
                all_detections.extend(detections)
                
                # Cleanup after each image if memory manager available
                if memory_manager and batch_idx < batch_size - 1:
                    memory_manager.cleanup_memory(force=False)
            
            # Combine results back into batches
            final_image = torch.cat(processed_images, dim=0)
            final_combined_masks = torch.cat(all_combined_masks, dim=0)
            final_face_masks = torch.cat(all_face_masks, dim=0)
            final_hand_masks = torch.cat(all_hand_masks, dim=0)
            
            logger.info(f"Batch processing completed: {batch_size} images, {len(all_detections)} total detections")
            
            return final_image, final_combined_masks, final_face_masks, final_hand_masks, all_detections
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to original processing
            return self._process_batch_fallback(
                image, detector, target_parts_list, validated_params,
                model, vae, positive, negative
            )
    
    def _process_batch_fallback(
        self, 
        image: torch.Tensor, 
        detector, 
        target_parts_list: List[str], 
        validated_params: Dict[str, Any],
        model, 
        vae, 
        positive, 
        negative
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Fallback batch processing method for compatibility.
        
        Returns:
            Tuple of processed results
        """
        try:
            confidence_threshold = validated_params.get('confidence_threshold', 0.5)
            mask_padding = validated_params.get('mask_padding', 32)
            mask_blur = validated_params.get('mask_blur', 4)
            
            # Convert image tensor to numpy for detection
            image_np = self._tensor_to_numpy(image[0])  # Process first image in batch
            
            # Detect target parts
            detections = self._detect_parts(
                image_np, 
                detector, 
                target_parts_list, 
                confidence_threshold
            )
            
            # Generate masks
            combined_masks, face_masks, hand_masks = self._generate_masks(
                detections, 
                image.shape, 
                mask_padding, 
                mask_blur
            )
            
            # Process inpainting if masks are not empty
            processed_image = image
            if torch.any(combined_masks > 0):
                logger.info("Processing inpainting for detected regions")
                processed_image = self._inpaint_regions(
                    image,
                    combined_masks,
                    model,
                    vae,
                    positive,
                    negative,
                    **validated_params
                )
            else:
                logger.info("No detections found, returning original image")
            
            return processed_image, combined_masks, face_masks, hand_masks, detections
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            # Return original with empty masks
            batch_size, height, width, channels = image.shape
            empty_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
            return image, empty_mask, empty_mask, empty_mask, []
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array for YOLO processing.
        
        Args:
            tensor: Input tensor (H, W, C) in 0-1 range
            
        Returns:
            Numpy array (H, W, C) in 0-255 range
        """
        try:
            # Optimize tensor memory usage
            if hasattr(self, 'utils') and hasattr(self.utils, 'memory_utils'):
                tensor = self.utils.memory_utils.MemoryManager.optimize_tensor_memory(tensor)
            
            # Convert to numpy and scale to 0-255 range
            if tensor.max() <= 1.0:
                numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            else:
                numpy_array = tensor.cpu().numpy().astype(np.uint8)
                
            return numpy_array
            
        except Exception as e:
            logger.error(f"Tensor to numpy conversion failed: {e}")
            # Fallback conversion
            return (tensor.cpu().numpy() * 255).astype(np.uint8) if tensor.max() <= 1.0 else tensor.cpu().numpy().astype(np.uint8)

# For testing and development
if __name__ == "__main__":
    print("Universal Detailer - Skeleton Implementation")
    print("⚠️  This is a skeleton that needs completion by AI developer")
    print("⚠️  See SPECIFICATIONS.md for complete requirements")
    
    # Create a test instance
    node = UniversalDetailerNode()
    print(f"Node initialized: {node}")
    print(f"Input types: {node.INPUT_TYPES()}")
    print(f"Return types: {node.RETURN_TYPES}")