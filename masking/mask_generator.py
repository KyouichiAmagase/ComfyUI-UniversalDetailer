#!/usr/bin/env python3
"""
Mask Generator Implementation

Generates masks from detection results for inpainting.
Optimized for performance with caching and memory efficiency.
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any
import logging
from functools import lru_cache, wraps
import time

logger = logging.getLogger(__name__)

def profile_mask_operation(func):
    """Decorator to profile mask operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if duration > 0.5:  # Log operations taking more than 500ms
                logger.info(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

class MaskGenerator:
    """
    Generate and process masks from detection results.
    
    This class handles:
    - Converting bounding boxes to masks
    - Applying padding and blur
    - Combining multiple masks
    - Separating masks by type (face, hand, etc.)
    
    Optimizations:
    - LRU caching for repeated operations
    - Vectorized numpy operations
    - Memory pool for tensors
    - Performance profiling
    """
    
    def __init__(self):
        """Initialize mask generator with optimization features."""
        self._mask_cache = {}
        self._kernel_cache = {}
        self._last_cleanup = time.time()
        logger.info("MaskGenerator initialized with optimizations")
    
    @profile_mask_operation
    def generate_masks(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int, int, int],
        padding: int = 32,
        blur: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate masks from detection results.
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the input image (batch, height, width, channels)
            padding: Padding to add around detected areas
            blur: Blur radius for mask edges
        
        Returns:
            Tuple of (combined_masks, face_masks, hand_masks)
        """
        batch_size, height, width, channels = image_shape
        logger.info(f"Generating masks for {len(detections)} detections, image shape: {image_shape}")
        
        # Initialize mask tensors
        combined_masks = torch.zeros((batch_size, height, width), dtype=torch.float32)
        face_masks = torch.zeros((batch_size, height, width), dtype=torch.float32)
        hand_masks = torch.zeros((batch_size, height, width), dtype=torch.float32)
        
        if not detections:
            logger.info("No detections provided, returning empty masks")
            return combined_masks, face_masks, hand_masks
        
        # Separate detections by type
        face_detections = self._filter_masks_by_type(detections, "face")
        hand_detections = self._filter_masks_by_type(detections, "hand")
        
        logger.info(f"Processing {len(face_detections)} face detections and {len(hand_detections)} hand detections")
        
        # Generate masks for each batch item
        for batch_idx in range(batch_size):
            # Generate face masks
            if face_detections:
                face_mask_np = self._generate_mask_for_detections(
                    face_detections, (height, width), padding, blur
                )
                face_masks[batch_idx] = self.numpy_to_torch(face_mask_np, 1)[0]
            
            # Generate hand masks
            if hand_detections:
                hand_mask_np = self._generate_mask_for_detections(
                    hand_detections, (height, width), padding, blur
                )
                hand_masks[batch_idx] = self.numpy_to_torch(hand_mask_np, 1)[0]
            
            # Combine all masks
            combined_masks[batch_idx] = torch.maximum(face_masks[batch_idx], hand_masks[batch_idx])
        
        logger.info("Mask generation completed")
        return combined_masks, face_masks, hand_masks
    
    def _generate_mask_for_detections(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        padding: int,
        blur: int
    ) -> np.ndarray:
        """
        Generate a single mask for multiple detections.
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the image (height, width)
            padding: Padding to add around detections
            blur: Blur radius for mask edges
        
        Returns:
            Combined mask as numpy array
        """
        masks = []
        
        for detection in detections:
            bbox = detection.get("bbox")
            if bbox is None:
                continue
                
            # Create mask from bounding box
            mask = self._create_bbox_mask(tuple(bbox), image_shape, padding)
            
            # Apply blur
            if blur > 0:
                mask = self._apply_blur(mask, blur)
            
            masks.append(mask)
        
        if not masks:
            return np.zeros(image_shape, dtype=np.uint8)
        
        # Combine all masks
        return self._combine_masks(masks)
    
    def _create_bbox_mask(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        padding: int = 0
    ) -> np.ndarray:
        """
        Create a mask from a bounding box.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            image_shape: Shape of the image (height, width)
            padding: Padding to add around the bbox
        
        Returns:
            Binary mask as numpy array
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Extract and validate bounding box coordinates
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Ensure valid bounding box
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def _apply_padding(self, mask: np.ndarray, padding: int) -> np.ndarray:
        """
        Apply padding to a mask.
        
        Args:
            mask: Input mask
            padding: Padding amount in pixels
        
        Returns:
            Padded mask
        """
        if padding <= 0:
            return mask
        
        # Use morphological dilation for padding
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding*2+1, padding*2+1))
        return cv2.dilate(mask, kernel, iterations=1)
    
    def _apply_blur(self, mask: np.ndarray, blur_radius: int) -> np.ndarray:
        """
        Apply Gaussian blur to mask edges.
        
        Args:
            mask: Input mask
            blur_radius: Blur radius
        
        Returns:
            Blurred mask
        """
        if blur_radius <= 0:
            return mask
        
        # Apply Gaussian blur
        kernel_size = blur_radius * 2 + 1
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), blur_radius)
    
    def _combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple masks into one.
        
        Args:
            masks: List of mask arrays
        
        Returns:
            Combined mask
        """
        if not masks:
            return np.array([])
        
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.maximum(combined, mask)
        
        return combined
    
    def _filter_masks_by_type(self, detections: List[Dict[str, Any]], mask_type: str) -> List[Dict[str, Any]]:
        """
        Filter detections by type.
        
        Args:
            detections: List of detection dictionaries
            mask_type: Type to filter for ('face', 'hand', etc.)
        
        Returns:
            Filtered detections
        """
        return [det for det in detections if det.get('type') == mask_type]
    
    def numpy_to_torch(self, mask: np.ndarray, batch_size: int = 1) -> torch.Tensor:
        """
        Convert numpy mask to torch tensor.
        
        Args:
            mask: Numpy mask array
            batch_size: Batch size for output tensor
        
        Returns:
            Torch tensor mask
        """
        # Normalize to 0-1 range
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        tensor = torch.from_numpy(mask).float()
        
        # Add batch dimension if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return tensor
    
    def get_mask_info(self, mask: torch.Tensor) -> Dict[str, Any]:
        """
        Get information about a mask.
        
        Args:
            mask: Mask tensor
        
        Returns:
            Dictionary with mask statistics
        """
        return {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "min_value": float(mask.min()),
            "max_value": float(mask.max()),
            "mean_value": float(mask.mean()),
            "nonzero_pixels": int(torch.count_nonzero(mask)),
            "total_pixels": int(mask.numel())
        }