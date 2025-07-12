#!/usr/bin/env python3
"""
Unit Tests for MaskGenerator

Tests the mask generation functionality including bbox to mask conversion,
padding, blurring, and tensor operations.
"""

import pytest
import torch
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from masking.mask_generator import MaskGenerator

class TestMaskGenerator:
    """Test suite for MaskGenerator class."""
    
    @pytest.fixture
    def mask_generator(self):
        """Create a MaskGenerator instance for testing."""
        return MaskGenerator()
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detection data for testing."""
        return [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.9,
                "type": "face"
            },
            {
                "bbox": [300, 300, 400, 400],
                "confidence": 0.8,
                "type": "hand"
            }
        ]
    
    @pytest.fixture
    def image_shape(self):
        """Sample image shape for testing."""
        return (1, 512, 512, 3)  # (batch, height, width, channels)
    
    def test_init(self, mask_generator):
        """Test MaskGenerator initialization."""
        assert isinstance(mask_generator, MaskGenerator)
    
    def test_generate_masks_empty_detections(self, mask_generator, image_shape):
        """Test mask generation with empty detections."""
        combined, face, hand = mask_generator.generate_masks([], image_shape)
        
        batch_size, height, width, channels = image_shape
        
        assert combined.shape == (batch_size, height, width)
        assert face.shape == (batch_size, height, width)
        assert hand.shape == (batch_size, height, width)
        assert torch.all(combined == 0)
        assert torch.all(face == 0)
        assert torch.all(hand == 0)
    
    def test_generate_masks_with_detections(self, mask_generator, sample_detections, image_shape):
        """Test mask generation with sample detections."""
        combined, face, hand = mask_generator.generate_masks(
            sample_detections, image_shape, padding=0, blur=0
        )
        
        batch_size, height, width, channels = image_shape
        
        # Check shapes
        assert combined.shape == (batch_size, height, width)
        assert face.shape == (batch_size, height, width)
        assert hand.shape == (batch_size, height, width)
        
        # Check that masks have non-zero values
        assert torch.any(combined > 0)
        assert torch.any(face > 0)
        assert torch.any(hand > 0)
        
        # Check that combined mask is maximum of face and hand masks
        expected_combined = torch.maximum(face, hand)
        assert torch.allclose(combined, expected_combined)
    
    def test_create_bbox_mask(self, mask_generator):
        """Test bbox to mask conversion."""
        bbox = (50, 50, 150, 150)  # (x1, y1, x2, y2)
        image_shape = (256, 256)   # (height, width)
        
        mask = mask_generator._create_bbox_mask(bbox, image_shape, padding=0)
        
        assert mask.shape == image_shape
        assert mask.dtype == np.uint8
        
        # Check that the correct region is filled
        assert np.all(mask[50:150, 50:150] == 255)
        assert np.all(mask[0:50, :] == 0)
        assert np.all(mask[150:, :] == 0)
        assert np.all(mask[:, 0:50] == 0)
        assert np.all(mask[:, 150:] == 0)
    
    def test_create_bbox_mask_with_padding(self, mask_generator):
        """Test bbox to mask conversion with padding."""
        bbox = (100, 100, 200, 200)
        image_shape = (300, 300)
        padding = 10
        
        mask = mask_generator._create_bbox_mask(bbox, image_shape, padding)
        
        # Check that padding expands the mask
        assert np.any(mask[90:210, 90:210] == 255)
        assert np.all(mask[0:89, :] == 0)
        assert np.all(mask[211:, :] == 0)
    
    def test_create_bbox_mask_boundary_clipping(self, mask_generator):
        """Test bbox mask creation with boundary clipping."""
        bbox = (-10, -10, 50, 50)  # Extends beyond image boundaries
        image_shape = (100, 100)
        
        mask = mask_generator._create_bbox_mask(bbox, image_shape, padding=0)
        
        # Should clip to image boundaries
        assert mask.shape == image_shape
        assert np.any(mask[0:50, 0:50] == 255)
    
    def test_apply_padding(self, mask_generator):
        """Test padding application using morphological dilation."""
        # Create a small mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 square
        
        padded = mask_generator._apply_padding(mask, padding=5)
        
        # Check that the mask is larger after padding
        assert padded.shape == mask.shape
        assert np.sum(padded > 0) > np.sum(mask > 0)
    
    def test_apply_padding_zero(self, mask_generator):
        """Test padding with zero padding (should return original)."""
        mask = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = mask_generator._apply_padding(mask, padding=0)
        
        assert np.array_equal(result, mask)
    
    def test_apply_blur(self, mask_generator):
        """Test Gaussian blur application."""
        # Create a sharp mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        blurred = mask_generator._apply_blur(mask, blur_radius=3)
        
        # Check that the mask is blurred (values between 0 and 255)
        assert blurred.shape == mask.shape
        unique_values = np.unique(blurred)
        assert len(unique_values) > 2  # Should have gradual values
    
    def test_apply_blur_zero(self, mask_generator):
        """Test blur with zero radius (should return original)."""
        mask = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = mask_generator._apply_blur(mask, blur_radius=0)
        
        assert np.array_equal(result, mask)
    
    def test_combine_masks(self, mask_generator):
        """Test mask combination."""
        mask1 = np.zeros((50, 50), dtype=np.uint8)
        mask2 = np.zeros((50, 50), dtype=np.uint8)
        
        mask1[10:20, 10:20] = 255
        mask2[15:25, 15:25] = 128
        
        combined = mask_generator._combine_masks([mask1, mask2])
        
        # Should take maximum values
        assert combined.shape == (50, 50)
        assert combined[12, 12] == 255  # From mask1
        assert combined[22, 22] == 128  # From mask2
        assert combined[17, 17] == 255  # Overlap, max value
    
    def test_combine_masks_empty(self, mask_generator):
        """Test combining empty mask list."""
        result = mask_generator._combine_masks([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    
    def test_filter_masks_by_type(self, mask_generator, sample_detections):
        """Test filtering detections by type."""
        face_detections = mask_generator._filter_masks_by_type(sample_detections, "face")
        hand_detections = mask_generator._filter_masks_by_type(sample_detections, "hand")
        
        assert len(face_detections) == 1
        assert len(hand_detections) == 1
        assert face_detections[0]["type"] == "face"
        assert hand_detections[0]["type"] == "hand"
    
    def test_numpy_to_torch(self, mask_generator):
        """Test numpy to torch tensor conversion."""
        mask_np = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        # Test single batch
        tensor = mask_generator.numpy_to_torch(mask_np, batch_size=1)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 64, 64)
        assert tensor.dtype == torch.float32
        assert torch.all(tensor >= 0) and torch.all(tensor <= 1)
        
        # Test multiple batches
        tensor_batch = mask_generator.numpy_to_torch(mask_np, batch_size=3)
        assert tensor_batch.shape == (3, 64, 64)
    
    def test_numpy_to_torch_normalized_input(self, mask_generator):
        """Test numpy to torch with already normalized input."""
        mask_np = np.random.rand(32, 32).astype(np.float32)  # Already 0-1
        
        tensor = mask_generator.numpy_to_torch(mask_np, batch_size=1)
        
        assert torch.all(tensor >= 0) and torch.all(tensor <= 1)
    
    def test_get_mask_info(self, mask_generator):
        """Test mask information extraction."""
        mask = torch.rand((2, 64, 64))
        
        info = mask_generator.get_mask_info(mask)
        
        assert info["shape"] == [2, 64, 64]
        assert "dtype" in info
        assert "min_value" in info
        assert "max_value" in info
        assert "mean_value" in info
        assert "nonzero_pixels" in info
        assert "total_pixels" in info
        assert info["total_pixels"] == 2 * 64 * 64
    
    def test_generate_mask_for_detections(self, mask_generator):
        """Test generating mask for multiple detections."""
        detections = [
            {"bbox": [10, 10, 30, 30], "type": "face"},
            {"bbox": [40, 40, 60, 60], "type": "face"}
        ]
        
        mask = mask_generator._generate_mask_for_detections(
            detections, (100, 100), padding=0, blur=0
        )
        
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        
        # Both regions should be present
        assert np.any(mask[10:30, 10:30] > 0)
        assert np.any(mask[40:60, 40:60] > 0)
    
    @pytest.mark.parametrize("padding,blur", [
        (0, 0),
        (5, 0),
        (0, 3),
        (5, 3)
    ])
    def test_generate_masks_parameters(self, mask_generator, sample_detections, image_shape, padding, blur):
        """Test mask generation with different parameters."""
        combined, face, hand = mask_generator.generate_masks(
            sample_detections, image_shape, padding=padding, blur=blur
        )
        
        # Should not crash and produce valid tensors
        assert isinstance(combined, torch.Tensor)
        assert isinstance(face, torch.Tensor)
        assert isinstance(hand, torch.Tensor)
        assert combined.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__])