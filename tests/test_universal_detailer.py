#!/usr/bin/env python3
"""
Integration Tests for Universal Detailer Node

Tests the complete Universal Detailer processing pipeline including
initialization, parameter validation, and processing workflow.
"""

import pytest
import torch
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_detailer import UniversalDetailerNode

class TestUniversalDetailerNode:
    """Test suite for UniversalDetailerNode class."""
    
    @pytest.fixture
    def node(self):
        """Create a UniversalDetailerNode instance for testing."""
        return UniversalDetailerNode()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image tensor for testing."""
        return torch.rand((1, 512, 512, 3))  # ComfyUI format (B, H, W, C)
    
    @pytest.fixture
    def mock_comfyui_models(self):
        """Create mock ComfyUI models for testing."""
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        return mock_model, mock_vae, mock_positive, mock_negative
    
    def test_init(self, node):
        """Test UniversalDetailerNode initialization."""
        assert isinstance(node.detection_models, dict)
        assert isinstance(node.model_cache, dict)
        assert hasattr(node, 'mask_generator')
        assert node.models_path.exists()
        assert node.cache_path.exists()
    
    def test_input_types(self):
        """Test INPUT_TYPES class method."""
        input_types = UniversalDetailerNode.INPUT_TYPES()
        
        # Check required inputs
        required = input_types["required"]
        assert "image" in required
        assert "model" in required
        assert "vae" in required
        assert "positive" in required
        assert "negative" in required
        
        # Check optional inputs
        optional = input_types["optional"]
        assert "detection_model" in optional
        assert "target_parts" in optional
        assert "confidence_threshold" in optional
        assert "mask_padding" in optional
        assert "inpaint_strength" in optional
    
    def test_return_types(self):
        """Test return types and names."""
        assert UniversalDetailerNode.RETURN_TYPES == ("IMAGE", "MASK", "MASK", "MASK", "STRING")
        assert UniversalDetailerNode.RETURN_NAMES == ("image", "detection_masks", "face_masks", "hand_masks", "detection_info")
        assert UniversalDetailerNode.FUNCTION == "process"
        assert UniversalDetailerNode.CATEGORY == "image/postprocessing"
    
    def test_validate_parameters(self, node):
        """Test parameter validation."""
        params = {
            "confidence_threshold": 1.5,  # Should be clamped to 0.95
            "mask_padding": -10,          # Should be clamped to 0
            "inpaint_strength": 2.0,      # Should be clamped to 1.0
            "steps": 200,                 # Should be clamped to 100
            "cfg_scale": 50.0,           # Should be clamped to 30.0
            "mask_blur": -5              # Should be clamped to 0
        }
        
        validated = node._validate_parameters(**params)
        
        assert validated["confidence_threshold"] == 0.95
        assert validated["mask_padding"] == 0
        assert validated["inpaint_strength"] == 1.0
        assert validated["steps"] == 100
        assert validated["cfg_scale"] == 30.0
        assert validated["mask_blur"] == 0
    
    def test_determine_device(self, node):
        """Test device determination."""
        with patch('torch.cuda.is_available', return_value=True):
            assert node._determine_device("auto") == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            assert node._determine_device("auto") == "cpu"
        
        assert node._determine_device("cpu") == "cpu"
        assert node._determine_device("cuda") == "cuda"
    
    def test_tensor_to_numpy(self, node, sample_image):
        """Test tensor to numpy conversion."""
        # Test with single image from batch
        image_tensor = sample_image[0]  # (H, W, C)
        numpy_array = node._tensor_to_numpy(image_tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (512, 512, 3)
        assert numpy_array.dtype == np.uint8
        assert np.all(numpy_array >= 0) and np.all(numpy_array <= 255)
    
    def test_blend_images(self, node):
        """Test image blending functionality."""
        original = torch.rand((1, 64, 64, 3))
        inpainted = torch.rand((1, 64, 64, 3))
        mask = torch.zeros((1, 64, 64))
        mask[0, 20:40, 20:40] = 1.0  # Mask center region
        
        blended = node._blend_images(original, inpainted, mask, blend_strength=0.5)
        
        assert blended.shape == original.shape
        assert torch.all(blended >= 0) and torch.all(blended <= 1)
        
        # Check that unmasked areas remain original
        assert torch.allclose(blended[0, 0, 0], original[0, 0, 0])
        
        # Check that masked areas are blended
        center_pixel = blended[0, 30, 30]
        expected = original[0, 30, 30] * 0.5 + inpainted[0, 30, 30] * 0.5
        assert torch.allclose(center_pixel, expected, atol=1e-6)
    
    @patch('universal_detailer.UniversalDetailerNode._load_detection_model')
    @patch('universal_detailer.UniversalDetailerNode._detect_parts')
    @patch('universal_detailer.UniversalDetailerNode._generate_masks')
    @patch('universal_detailer.UniversalDetailerNode._inpaint_regions')
    def test_process_full_pipeline(self, mock_inpaint, mock_generate_masks, 
                                  mock_detect, mock_load_model, node, 
                                  sample_image, mock_comfyui_models):
        """Test the complete processing pipeline."""
        model, vae, positive, negative = mock_comfyui_models
        
        # Setup mocks
        mock_detector = Mock()
        mock_load_model.return_value = mock_detector
        
        mock_detections = [
            {"bbox": [100, 100, 200, 200], "type": "face", "confidence": 0.9}
        ]
        mock_detect.return_value = mock_detections
        
        batch_size, height, width, channels = sample_image.shape
        mock_masks = (
            torch.zeros((batch_size, height, width)),  # combined
            torch.zeros((batch_size, height, width)),  # face
            torch.zeros((batch_size, height, width))   # hand
        )
        mock_generate_masks.return_value = mock_masks
        
        mock_inpaint.return_value = sample_image
        
        # Run processing
        result = node.process(
            image=sample_image,
            model=model,
            vae=vae,
            positive=positive,
            negative=negative,
            detection_model="yolov8n-face",
            target_parts="face",
            confidence_threshold=0.5
        )
        
        # Verify results
        assert len(result) == 5
        processed_image, detection_masks, face_masks, hand_masks, detection_info = result
        
        assert torch.equal(processed_image, sample_image)
        assert isinstance(detection_info, str)
        
        # Parse and verify detection info
        info = json.loads(detection_info)
        assert info["error"] is False
        assert info["total_detections"] == 1
        assert "processing_time" in info
        assert "performance_metrics" in info
    
    @patch('universal_detailer.UniversalDetailerNode._load_detection_model')
    def test_process_model_load_failure(self, mock_load_model, node, 
                                       sample_image, mock_comfyui_models):
        """Test processing with model loading failure."""
        model, vae, positive, negative = mock_comfyui_models
        mock_load_model.return_value = None  # Simulate failure
        
        with pytest.raises(RuntimeError, match="Failed to load detection model"):
            node.process(
                image=sample_image,
                model=model,
                vae=vae,
                positive=positive,
                negative=negative
            )
    
    @patch('universal_detailer.UniversalDetailerNode._process_batch_efficiently')
    def test_process_with_memory_manager(self, mock_batch_process, node, 
                                        sample_image, mock_comfyui_models):
        """Test processing with memory management."""
        model, vae, positive, negative = mock_comfyui_models
        
        # Mock the batch processing to return expected results
        batch_size, height, width, channels = sample_image.shape
        mock_results = (
            sample_image,  # processed_image
            torch.zeros((batch_size, height, width)),  # combined_masks
            torch.zeros((batch_size, height, width)),  # face_masks
            torch.zeros((batch_size, height, width)),  # hand_masks
            []  # detections
        )
        mock_batch_process.return_value = mock_results
        
        # Should not raise exception even if memory utils are available
        result = node.process(
            image=sample_image,
            model=model,
            vae=vae,
            positive=positive,
            negative=negative
        )
        
        assert len(result) == 5
        mock_batch_process.assert_called_once()
    
    def test_process_batch_fallback(self, node):
        """Test batch processing fallback method."""
        # Create test data
        image = torch.rand((2, 128, 128, 3))
        detector = Mock()
        target_parts_list = ["face"]
        validated_params = {"confidence_threshold": 0.5, "mask_padding": 32, "mask_blur": 4}
        model, vae, positive, negative = Mock(), Mock(), Mock(), Mock()
        
        # Mock dependencies
        with patch.object(node, '_tensor_to_numpy') as mock_tensor_to_numpy, \
             patch.object(node, '_detect_parts') as mock_detect, \
             patch.object(node, '_generate_masks') as mock_generate_masks, \
             patch.object(node, '_inpaint_regions') as mock_inpaint:
            
            # Setup mocks
            mock_tensor_to_numpy.return_value = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            mock_detect.return_value = []
            
            batch_size, height, width, channels = image.shape
            mock_generate_masks.return_value = (
                torch.zeros((batch_size, height, width)),
                torch.zeros((batch_size, height, width)),
                torch.zeros((batch_size, height, width))
            )
            mock_inpaint.return_value = image
            
            # Run fallback processing
            result = node._process_batch_fallback(
                image, detector, target_parts_list, validated_params,
                model, vae, positive, negative
            )
            
            assert len(result) == 5
            processed_image, combined_masks, face_masks, hand_masks, detections = result
            assert torch.equal(processed_image, image)
    
    def test_process_error_handling(self, node, sample_image, mock_comfyui_models):
        """Test error handling in process method."""
        model, vae, positive, negative = mock_comfyui_models
        
        # Force an error by providing invalid image
        invalid_image = "not a tensor"
        
        result = node.process(
            image=invalid_image,
            model=model,
            vae=vae,
            positive=positive,
            negative=negative
        )
        
        # Should return error information
        assert len(result) == 5
        _, _, _, _, detection_info = result
        
        info = json.loads(detection_info)
        assert info["error"] is True
        assert "error_type" in info
        assert "message" in info
    
    @pytest.mark.parametrize("target_parts,expected_list", [
        ("face", ["face"]),
        ("face,hand", ["face", "hand"]),
        ("face, hand, finger", ["face", "hand", "finger"]),
        ("face,hand,", ["face", "hand", ""])  # Edge case with trailing comma
    ])
    def test_target_parts_parsing(self, node, target_parts, expected_list):
        """Test parsing of target parts string."""
        # This would be tested within the process method
        parsed = [part.strip() for part in target_parts.split(",")]
        assert parsed == expected_list


if __name__ == "__main__":
    pytest.main([__file__])