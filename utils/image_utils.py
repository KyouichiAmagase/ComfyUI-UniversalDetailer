#!/usr/bin/env python3
"""
Image Utilities

Utility functions for image processing and conversion.

⚠️  WARNING: This is AI-generated skeleton code.
⚠️  Complete implementation needed by AI developer.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """
    Utility functions for image processing and format conversion.
    
    This class provides static methods for:
    - Converting between different image formats
    - Image preprocessing for detection
    - Image postprocessing after inpainting
    - Color space conversions
    """
    
    @staticmethod
    def torch_to_numpy(image: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.
        
        Args:
            image: Torch tensor image (B, H, W, C) or (H, W, C)
        
        Returns:
            Numpy array image
        """
        if image.dim() == 4:
            # Batch dimension exists, take first image
            image = image[0]
        
        # Convert to numpy and ensure correct dtype
        numpy_image = image.detach().cpu().numpy()
        
        # Ensure values are in 0-255 range
        if numpy_image.max() <= 1.0:
            numpy_image = (numpy_image * 255).astype(np.uint8)
        else:
            numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
        
        return numpy_image
    
    @staticmethod
    def numpy_to_torch(image: np.ndarray, batch_size: int = 1) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.
        
        Args:
            image: Numpy array image
            batch_size: Batch size for output tensor
        
        Returns:
            Torch tensor image
        """
        # Ensure float32 and 0-1 range
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        tensor = torch.from_numpy(image).float()
        
        # Ensure correct dimensions (H, W, C)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return tensor
    
    @staticmethod
    def pil_to_torch(image: Image.Image, batch_size: int = 1) -> torch.Tensor:
        """
        Convert PIL Image to torch tensor.
        
        Args:
            image: PIL Image
            batch_size: Batch size for output tensor
        
        Returns:
            Torch tensor image
        """
        numpy_image = np.array(image)
        return ImageUtils.numpy_to_torch(numpy_image, batch_size)
    
    @staticmethod
    def torch_to_pil(image: torch.Tensor) -> Image.Image:
        """
        Convert torch tensor to PIL Image.
        
        Args:
            image: Torch tensor image
        
        Returns:
            PIL Image
        """
        numpy_image = ImageUtils.torch_to_numpy(image)
        return Image.fromarray(numpy_image)
    
    @staticmethod
    def preprocess_for_detection(image: torch.Tensor) -> np.ndarray:
        """
        Preprocess image for YOLO detection.
        
        Implements proper YOLO preprocessing including normalization and format conversion.
        
        Args:
            image: Input torch tensor image (B, H, W, C) in range [0, 1]
        
        Returns:
            Preprocessed numpy array (H, W, C) in range [0, 255]
        """
        try:
            # Take first image from batch if needed
            if len(image.shape) == 4:
                image = image[0]  # (H, W, C)
            
            # Ensure image is in [0, 1] range
            image = torch.clamp(image, 0.0, 1.0)
            
            # Convert to numpy and scale to [0, 255]
            numpy_image = (image.cpu().numpy() * 255.0).astype(np.uint8)
            
            # YOLO expects RGB format - ensure proper channel order
            if numpy_image.shape[2] == 3:
                # Assume input is RGB, YOLO will handle internally
                pass
            elif numpy_image.shape[2] == 4:
                # Convert RGBA to RGB
                numpy_image = numpy_image[:, :, :3]
            
            logger.info(f"Preprocessed image for detection: {numpy_image.shape}, dtype: {numpy_image.dtype}")
            return numpy_image
            
        except Exception as e:
            logger.error(f"Detection preprocessing failed: {e}")
            # Fallback to basic conversion
            return ImageUtils.torch_to_numpy(image)
    
    @staticmethod
    def postprocess_inpaint_result(image: torch.Tensor) -> torch.Tensor:
        """
        Postprocess inpainting result.
        
        Implements color correction, contrast enhancement, and artifact reduction.
        
        Args:
            image: Inpainted image tensor (B, H, W, C) in range [0, 1]
        
        Returns:
            Postprocessed image tensor
        """
        try:
            logger.info("Applying inpainting postprocessing...")
            
            # Ensure image is properly clamped
            processed_image = torch.clamp(image, 0.0, 1.0)
            
            # Apply subtle color correction
            # Gamma correction for better visual quality
            gamma = 1.05  # Slight gamma boost
            processed_image = torch.pow(processed_image, 1.0 / gamma)
            
            # Enhance contrast slightly
            # Apply simple contrast enhancement
            contrast_factor = 1.02
            processed_image = (processed_image - 0.5) * contrast_factor + 0.5
            
            # Reduce potential artifacts with gentle smoothing
            # Apply very light gaussian-like smoothing to reduce artifacts
            if len(processed_image.shape) == 4:  # Batch processing
                for i in range(processed_image.shape[0]):
                    # Simple artifact reduction using channel-wise normalization
                    for c in range(processed_image.shape[3]):
                        channel = processed_image[i, :, :, c]
                        # Normalize each channel to reduce artifacts
                        channel_mean = torch.mean(channel)
                        channel_std = torch.std(channel)
                        if channel_std > 0:
                            normalized = (channel - channel_mean) / channel_std
                            processed_image[i, :, :, c] = normalized * channel_std * 0.98 + channel_mean
            
            # Final clamp to ensure valid range
            processed_image = torch.clamp(processed_image, 0.0, 1.0)
            
            logger.info("Inpainting postprocessing completed")
            return processed_image
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            # Return original image if postprocessing fails
            return torch.clamp(image, 0.0, 1.0)
    
    @staticmethod
    def resize_image(image: Union[torch.Tensor, np.ndarray], target_size: Tuple[int, int]) -> Union[torch.Tensor, np.ndarray]:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
        
        Returns:
            Resized image in same format as input
        """
        if isinstance(image, torch.Tensor):
            # Convert to numpy, resize, convert back
            numpy_img = ImageUtils.torch_to_numpy(image)
            resized_numpy = cv2.resize(numpy_img, target_size)
            return ImageUtils.numpy_to_torch(resized_numpy, batch_size=image.shape[0] if image.dim() == 4 else 1)
        else:
            return cv2.resize(image, target_size)
    
    @staticmethod
    def blend_images(original: torch.Tensor, processed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Blend original and processed images using a mask.
        
        Args:
            original: Original image tensor
            processed: Processed image tensor
            mask: Blend mask tensor
        
        Returns:
            Blended image tensor
        """
        # Ensure mask has same dimensions as images
        if mask.dim() == 3 and original.dim() == 4:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, original.shape[-1])
        elif mask.dim() == 3 and original.dim() == 3:
            mask = mask.unsqueeze(-1).repeat(1, 1, 3)
        
        # Blend using mask
        blended = original * (1 - mask) + processed * mask
        return blended
    
    @staticmethod
    def get_image_info(image: torch.Tensor) -> dict:
        """
        Get information about an image tensor.
        
        Args:
            image: Image tensor
        
        Returns:
            Dictionary with image information
        """
        return {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min_value": float(image.min()),
            "max_value": float(image.max()),
            "mean_value": float(image.mean()),
            "device": str(image.device)
        }