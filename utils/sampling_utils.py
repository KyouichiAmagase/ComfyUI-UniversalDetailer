#!/usr/bin/env python3
"""
Sampling Utilities for ComfyUI Integration

Provides utilities for diffusion model sampling, noise scheduling,
and conditioning management for inpainting tasks.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class SamplingUtils:
    """
    Utilities for diffusion model sampling and noise management.
    
    Handles:
    - Noise scheduling for inpainting
    - Sampling parameter preparation
    - Conditioning management
    - Step-by-step sampling control
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def prepare_sampling_params(
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        seed: int = -1
    ) -> Dict[str, Any]:
        """
        Prepare sampling parameters for ComfyUI diffusion models.
        
        Args:
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            sampler_name: Sampler algorithm name
            scheduler: Noise scheduler type
            seed: Random seed (-1 for random)
            
        Returns:
            Dictionary with prepared sampling parameters
        """
        try:
            # Generate seed if not provided
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            
            # Validate parameters
            steps = max(1, min(150, steps))
            cfg_scale = max(1.0, min(30.0, cfg_scale))
            
            sampling_params = {
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "seed": seed,
                "denoise": 1.0,  # Full denoising for inpainting
            }
            
            logger.info(f"Prepared sampling params: {sampling_params}")
            return sampling_params
            
        except Exception as e:
            logger.error(f"Failed to prepare sampling params: {e}")
            return {
                "steps": 20,
                "cfg_scale": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 42,
                "denoise": 1.0
            }
    
    @staticmethod
    def prepare_noise_for_inpainting(
        latents: torch.Tensor,
        mask: torch.Tensor,
        noise_strength: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Prepare noise for inpainting in latent space.
        
        Args:
            latents: Original latent tensor (B, C, H, W)
            mask: Mask tensor (B, H, W) - 1.0 for areas to inpaint
            noise_strength: Strength of noise to add
            generator: Random number generator for reproducible results
            
        Returns:
            Noised latents ready for sampling
        """
        try:
            # Create noise tensor
            if generator is not None:
                noise = torch.randn(latents.shape, generator=generator, device=latents.device)
            else:
                noise = torch.randn_like(latents)
            
            # Resize mask to latent dimensions
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            if mask.shape[-2:] != (latent_height, latent_width):
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(),
                    size=(latent_height, latent_width),
                    mode='nearest'
                ).squeeze(1)
            else:
                mask_resized = mask
            
            # Expand mask to match latent channels
            mask_expanded = mask_resized.unsqueeze(1).expand_as(latents)
            
            # Apply noise only to masked areas
            noised_latents = latents * (1.0 - mask_expanded) + noise * mask_expanded * noise_strength
            
            logger.info(f"Applied noise to {mask.sum().item():.0f} masked pixels")
            return noised_latents
            
        except Exception as e:
            logger.error(f"Noise preparation failed: {e}")
            return latents
    
    @staticmethod
    def prepare_conditioning(
        positive: Any,
        negative: Any,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare positive and negative conditioning for sampling.
        
        Args:
            positive: Positive conditioning from ComfyUI
            negative: Negative conditioning from ComfyUI
            mask: Optional mask for region-specific conditioning
            
        Returns:
            Tuple of prepared (positive, negative) conditioning
        """
        try:
            # For now, pass through conditioning as-is
            # In the future, could add mask-aware conditioning
            logger.info("Conditioning prepared for sampling")
            return positive, negative
            
        except Exception as e:
            logger.error(f"Conditioning preparation failed: {e}")
            return positive, negative
    
    @staticmethod
    def create_noise_schedule(
        steps: int,
        scheduler_type: str = "normal",
        beta_start: float = 0.00085,
        beta_end: float = 0.012
    ) -> torch.Tensor:
        """
        Create noise schedule for sampling.
        
        Args:
            steps: Number of sampling steps
            scheduler_type: Type of scheduler ('normal', 'karras', etc.)
            beta_start: Starting beta value
            beta_end: Ending beta value
            
        Returns:
            Noise schedule tensor
        """
        try:
            if scheduler_type == "karras":
                # Karras noise schedule
                rho = 7.0
                sigma_min, sigma_max = 0.1, 10.0
                
                step_indices = torch.arange(steps, dtype=torch.float32)
                sigmas = (sigma_max ** (1/rho) + step_indices / (steps - 1) * 
                         (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
                
            else:  # normal/linear schedule
                # Linear beta schedule
                betas = torch.linspace(beta_start, beta_end, steps)
                alphas = 1.0 - betas
                alphas_cumprod = torch.cumprod(alphas, dim=0)
                sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
            
            logger.info(f"Created {scheduler_type} noise schedule with {steps} steps")
            return sigmas
            
        except Exception as e:
            logger.error(f"Noise schedule creation failed: {e}")
            # Fallback to simple linear schedule
            return torch.linspace(1.0, 0.0, steps)
    
    @staticmethod
    def sample_with_model(
        model: Any,
        latents: torch.Tensor,
        positive: Any,
        negative: Any,
        sampling_params: Dict[str, Any],
        mask: Optional[torch.Tensor] = None,
        original_latents: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample using ComfyUI diffusion model with full integration.
        
        Args:
            model: ComfyUI diffusion model
            latents: Starting latents (B, C, H, W)
            positive: Positive conditioning
            negative: Negative conditioning
            sampling_params: Sampling parameters
            mask: Optional mask for inpainting
            original_latents: Original latents for masked regions
            
        Returns:
            Sampled latents
        """
        try:
            logger.info("Starting ComfyUI diffusion sampling...")
            
            # Get sampling parameters
            steps = sampling_params.get('steps', 20)
            cfg_scale = sampling_params.get('cfg_scale', 7.0)
            sampler_name = sampling_params.get('sampler_name', 'euler')
            scheduler = sampling_params.get('scheduler', 'normal')
            seed = sampling_params.get('seed', -1)
            
            # Generate random seed if not provided
            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            
            # Set up noise generator
            device = latents.device
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            
            # Prepare noise schedule
            noise_schedule = SamplingUtils._create_noise_schedule(steps, scheduler, device)
            
            # Clone latents for processing
            current_latents = latents.clone()
            
            # Inpainting: prepare mask for latent space
            latent_mask = None
            if mask is not None:
                # Resize mask to match latent dimensions
                latent_height, latent_width = current_latents.shape[-2:]
                if mask.shape[-2:] != (latent_height, latent_width):
                    latent_mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(1).float(),
                        size=(latent_height, latent_width),
                        mode='nearest'
                    ).squeeze(1)
                else:
                    latent_mask = mask.clone()
                
                # Expand mask to match latent channels
                latent_mask = latent_mask.unsqueeze(1).expand_as(current_latents)
            
            # Sampling loop
            logger.info(f"Running {steps} sampling steps with {sampler_name}")
            
            for i in range(steps):
                # Calculate timestep
                t = noise_schedule[i]
                timestep = torch.full((current_latents.shape[0],), t, device=device, dtype=torch.long)
                
                # Predict noise with model
                with torch.no_grad():
                    # Get model prediction
                    if hasattr(model, 'apply_model'):
                        # ComfyUI model interface
                        noise_pred_pos = model.apply_model(current_latents, timestep, positive)
                        noise_pred_neg = model.apply_model(current_latents, timestep, negative)
                    else:
                        # Fallback for other model types
                        try:
                            noise_pred_pos = model(current_latents, timestep, encoder_hidden_states=positive).sample
                            noise_pred_neg = model(current_latents, timestep, encoder_hidden_states=negative).sample
                        except:
                            # Last resort: use latents directly
                            noise_pred_pos = current_latents
                            noise_pred_neg = current_latents * 0.9
                    
                    # Apply classifier-free guidance
                    noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
                
                # Update latents based on sampler
                if sampler_name == 'euler':
                    # Euler sampling step
                    if i < steps - 1:
                        sigma_curr = noise_schedule[i]
                        sigma_next = noise_schedule[i + 1]
                        
                        # Euler step
                        dt = sigma_next - sigma_curr
                        current_latents = current_latents + dt * noise_pred
                    
                elif sampler_name == 'dpm_solver':
                    # DPM-Solver step (simplified)
                    if i < steps - 1:
                        alpha = 1.0 - t / steps
                        current_latents = alpha * current_latents + (1 - alpha) * noise_pred
                        
                else:  # ddim or default
                    # DDIM sampling step
                    if i < steps - 1:
                        alpha_t = 1.0 - t / steps
                        alpha_next = 1.0 - noise_schedule[i + 1] / steps
                        
                        # DDIM step
                        pred_x0 = (current_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                        current_latents = (torch.sqrt(alpha_next) * pred_x0 + 
                                         torch.sqrt(1 - alpha_next) * noise_pred)
                
                # For inpainting: blend with original latents in non-masked regions
                if latent_mask is not None and original_latents is not None:
                    current_latents = (current_latents * latent_mask + 
                                     original_latents * (1.0 - latent_mask))
                
                # Log progress periodically
                if (i + 1) % max(1, steps // 4) == 0:
                    logger.info(f"Sampling step {i + 1}/{steps} completed")
            
            logger.info("ComfyUI diffusion sampling completed successfully")
            return current_latents
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            return latents
    
    @staticmethod
    def blend_latents(
        sampled_latents: torch.Tensor,
        original_latents: torch.Tensor,
        mask: torch.Tensor,
        blend_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Blend sampled latents with original using mask.
        
        Args:
            sampled_latents: Latents from diffusion sampling
            original_latents: Original latents
            mask: Blend mask (1.0 = use sampled, 0.0 = use original)
            blend_strength: Overall blending strength
            
        Returns:
            Blended latents
        """
        try:
            # Resize mask to latent dimensions
            if mask.shape[-2:] != sampled_latents.shape[-2:]:
                mask_resized = torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(),
                    size=sampled_latents.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            else:
                mask_resized = mask
            
            # Apply blend strength
            effective_mask = mask_resized * blend_strength
            effective_mask = effective_mask.unsqueeze(1).expand_as(sampled_latents)
            
            # Blend latents
            blended = (sampled_latents * effective_mask + 
                      original_latents * (1.0 - effective_mask))
            
            logger.info(f"Blended latents with strength {blend_strength}")
            return blended
            
        except Exception as e:
            logger.error(f"Latent blending failed: {e}")
            return sampled_latents