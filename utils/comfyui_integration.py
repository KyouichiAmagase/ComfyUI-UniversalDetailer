#!/usr/bin/env python3
"""
ComfyUI Integration Utilities

Helper functions for integrating with ComfyUI's model system and workflows.
Handles VAE encoding/decoding, sampling, and tensor format conversions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ComfyUIHelper:
    """
    Helper class for ComfyUI integration tasks.
    
    Provides utilities for:
    - Tensor format conversions between ComfyUI and PyTorch
    - VAE encoding/decoding operations
    - Device management and memory optimization
    - Sampling parameter preparation
    """
    
    @staticmethod
    def convert_tensor_format(
        tensor: torch.Tensor, 
        source_format: str, 
        target_format: str
    ) -> torch.Tensor:
        """
        Convert tensor between different format conventions.
        
        Args:
            tensor: Input tensor
            source_format: Current format ('BHWC', 'BCHW', etc.)
            target_format: Target format
            
        Returns:
            Converted tensor
        """
        if source_format == target_format:
            return tensor
            
        try:
            if source_format == "BHWC" and target_format == "BCHW":
                # ComfyUI format to PyTorch format
                return tensor.permute(0, 3, 1, 2)
            elif source_format == "BCHW" and target_format == "BHWC":
                # PyTorch format to ComfyUI format
                return tensor.permute(0, 2, 3, 1)
            elif source_format == "HWC" and target_format == "CHW":
                # Single image: HWC to CHW
                return tensor.permute(2, 0, 1)
            elif source_format == "CHW" and target_format == "HWC":
                # Single image: CHW to HWC
                return tensor.permute(1, 2, 0)
            else:
                logger.warning(f"Unsupported conversion: {source_format} -> {target_format}")
                return tensor
                
        except Exception as e:
            logger.error(f"Tensor format conversion failed: {e}")
            return tensor
    
    @staticmethod
    def prepare_image_for_vae(image: torch.Tensor) -> torch.Tensor:
        """
        Prepare image tensor for VAE encoding.
        
        Args:
            image: Image tensor in ComfyUI format (B, H, W, C)
            
        Returns:
            Image tensor ready for VAE (B, C, H, W) in [-1, 1] range
        """
        try:
            # Convert to BCHW format
            if len(image.shape) == 4 and image.shape[3] == 3:
                image = ComfyUIHelper.convert_tensor_format(image, "BHWC", "BCHW")
            
            # Ensure float32
            image = image.float()
            
            # Normalize to [-1, 1] range if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            # Convert [0, 1] to [-1, 1]
            image = image * 2.0 - 1.0
            
            return image
            
        except Exception as e:
            logger.error(f"Image preparation for VAE failed: {e}")
            return image
    
    @staticmethod
    def prepare_image_from_vae(image: torch.Tensor) -> torch.Tensor:
        """
        Convert VAE output back to ComfyUI format.
        
        Args:
            image: Image tensor from VAE (B, C, H, W) in [-1, 1] range
            
        Returns:
            Image tensor in ComfyUI format (B, H, W, C) in [0, 1] range
        """
        try:
            # Convert from [-1, 1] to [0, 1]
            image = (image + 1.0) / 2.0
            
            # Clamp to valid range
            image = torch.clamp(image, 0.0, 1.0)
            
            # Convert to BHWC format
            image = ComfyUIHelper.convert_tensor_format(image, "BCHW", "BHWC")
            
            return image
            
        except Exception as e:
            logger.error(f"Image preparation from VAE failed: {e}")
            return image
    
    @staticmethod
    def encode_with_vae(vae, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using ComfyUI VAE.
        
        Args:
            vae: ComfyUI VAE model
            image: Image tensor (B, H, W, C)
            
        Returns:
            Latent tensor (B, C, H//8, W//8)
        """
        try:
            # Prepare image for VAE
            vae_input = ComfyUIHelper.prepare_image_for_vae(image)
            
            # Encode to latent space
            with torch.no_grad():
                if hasattr(vae, 'encode'):
                    latents = vae.encode(vae_input)
                    if hasattr(latents, 'latent_dist'):
                        latents = latents.latent_dist.sample()
                    elif hasattr(latents, 'sample'):
                        latents = latents.sample()
                else:
                    # Fallback for different VAE interfaces
                    latents = vae(vae_input)
            
            logger.info(f"VAE encoding: {vae_input.shape} -> {latents.shape}")
            return latents
            
        except Exception as e:
            logger.error(f"VAE encoding failed: {e}")
            # Return dummy latents as fallback
            batch_size, height, width, channels = image.shape
            return torch.randn(batch_size, 4, height // 8, width // 8, device=image.device)
    
    @staticmethod
    def decode_with_vae(vae, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image using ComfyUI VAE.
        
        Args:
            vae: ComfyUI VAE model
            latents: Latent tensor (B, C, H, W)
            
        Returns:
            Image tensor (B, H, W, C)
        """
        try:
            # Decode from latent space
            with torch.no_grad():
                if hasattr(vae, 'decode'):
                    decoded = vae.decode(latents)
                else:
                    # Fallback for different VAE interfaces
                    decoded = vae(latents, decode=True)
            
            # Convert back to ComfyUI format
            image = ComfyUIHelper.prepare_image_from_vae(decoded)
            
            logger.info(f"VAE decoding: {latents.shape} -> {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"VAE decoding failed: {e}")
            # Return dummy image as fallback
            batch_size, channels, height, width = latents.shape
            return torch.zeros(batch_size, height * 8, width * 8, 3, device=latents.device)
    
    @staticmethod
    def apply_mask_to_latents(
        latents: torch.Tensor, 
        mask: torch.Tensor, 
        noise_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply mask to latents for inpainting.
        
        Args:
            latents: Latent tensor (B, C, H, W)
            mask: Mask tensor (B, H, W) - 1.0 for areas to inpaint
            noise_strength: Strength of noise to add to masked areas
            
        Returns:
            Masked latents with noise in inpaint areas
        """
        try:
            # Resize mask to match latent dimensions
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            mask_resized = F.interpolate(
                mask.unsqueeze(1).float(), 
                size=(latent_height, latent_width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            # Expand mask to match latent channels
            mask_expanded = mask_resized.unsqueeze(1).expand_as(latents)
            
            # Generate noise for masked areas
            noise = torch.randn_like(latents) * noise_strength
            
            # Apply mask: keep original in non-masked areas, add noise to masked areas
            masked_latents = latents * (1.0 - mask_expanded) + noise * mask_expanded
            
            logger.info(f"Applied mask to latents: {latents.shape}, mask coverage: {mask.sum().item():.1f} pixels")
            return masked_latents
            
        except Exception as e:
            logger.error(f"Mask application to latents failed: {e}")
            return latents
    
    @staticmethod
    def safe_device_transfer(tensor: torch.Tensor, device: str) -> torch.Tensor:
        """
        Safely transfer tensor to target device with fallback.
        
        Args:
            tensor: Tensor to transfer
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
            
        Returns:
            Tensor on target device (or fallback device if transfer fails)
        """
        try:
            # Check if device is available
            if device.startswith('cuda'):
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = 'cpu'
                else:
                    # Check specific GPU availability
                    try:
                        torch.cuda.get_device_properties(device)
                    except:
                        logger.warning(f"Device {device} not available, using default CUDA device")
                        device = 'cuda'
            
            # Transfer tensor
            return tensor.to(device)
            
        except Exception as e:
            logger.error(f"Device transfer failed: {e}, keeping tensor on current device")
            return tensor
    
    @staticmethod
    def get_optimal_device() -> str:
        """
        Get the optimal device for processing.
        
        Returns:
            Device string ('cuda', 'cpu', etc.)
        """
        if torch.cuda.is_available():
            # Check GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 4 * 1024**3:  # 4GB minimum
                    return 'cuda'
                else:
                    logger.warning("GPU has limited memory, considering CPU")
            except:
                pass
        
        return 'cpu'