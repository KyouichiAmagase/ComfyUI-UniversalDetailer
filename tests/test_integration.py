#!/usr/bin/env python3
"""
Integration Tests for Universal Detailer

End-to-end integration tests that verify the complete functionality
of the Universal Detailer system with real-world scenarios.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_detailer import UniversalDetailerNode
from detection.yolo_detector import YOLODetector
from masking.mask_generator import MaskGenerator
from detection.model_loader import ModelManager

class TestIntegration:
    """Integration test suite for Universal Detailer system."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary directory for model files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def node_with_temp_dir(self, temp_models_dir):
        """Create a node instance with temporary model directory."""
        node = UniversalDetailerNode()
        node.models_path = temp_models_dir
        return node
    
    @pytest.fixture
    def sample_batch_images(self):
        """Create a batch of sample images for testing."""
        # Create images with different characteristics
        batch = []
        
        # Image 1: Simple gradient
        img1 = torch.zeros((512, 512, 3))
        for i in range(512):
            img1[i, :, :] = i / 512.0
        batch.append(img1)
        
        # Image 2: Random noise
        img2 = torch.rand((512, 512, 3))
        batch.append(img2)
        
        # Image 3: Checkerboard pattern
        img3 = torch.zeros((512, 512, 3))
        for i in range(0, 512, 64):
            for j in range(0, 512, 64):
                if (i // 64 + j // 64) % 2 == 0:
                    img3[i:i+64, j:j+64, :] = 1.0
        batch.append(img3)
        
        return torch.stack(batch, dim=0)  # (3, 512, 512, 3)
    
    @pytest.fixture
    def mock_ultralytics_model(self):
        """Create a mock ultralytics YOLO model."""
        mock_model = Mock()
        
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_result.boxes.conf = torch.tensor([0.9, 0.8])
        mock_result.boxes.cls = torch.tensor([0, 0])
        mock_result.names = {0: "face"}
        
        mock_model.return_value = [mock_result]
        return mock_model
    
    def test_complete_pipeline_single_image(self, node_with_temp_dir, sample_batch_images):
        """Test complete pipeline with a single image."""
        single_image = sample_batch_images[:1]  # Take first image only
        
        # Mock ComfyUI models
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        # Mock the detection model loading to avoid requiring actual model files
        with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
            mock_detector = Mock()
            mock_detector.detect.return_value = [
                {"bbox": [100, 100, 200, 200], "type": "face", "confidence": 0.9}
            ]
            mock_load.return_value = mock_detector
            
            # Mock inpainting to return modified image
            with patch.object(node_with_temp_dir, '_inpaint_regions') as mock_inpaint:
                mock_inpaint.return_value = single_image * 0.8  # Slightly darker
                
                # Run the complete pipeline
                result = node_with_temp_dir.process(
                    image=single_image,
                    model=mock_model,
                    vae=mock_vae,
                    positive=mock_positive,
                    negative=mock_negative,
                    detection_model="yolov8n-face",
                    target_parts="face",
                    confidence_threshold=0.5,
                    mask_padding=16,
                    inpaint_strength=0.8
                )
                
                # Verify results
                assert len(result) == 5
                processed_image, detection_masks, face_masks, hand_masks, detection_info = result
                
                # Check image processing
                assert processed_image.shape == single_image.shape
                assert not torch.equal(processed_image, single_image)  # Should be modified
                
                # Check masks
                assert detection_masks.shape == (1, 512, 512)
                assert face_masks.shape == (1, 512, 512)
                assert hand_masks.shape == (1, 512, 512)
                
                # Check detection info
                info = json.loads(detection_info)
                assert info["error"] is False
                assert info["total_detections"] == 1
                assert info["faces_detected"] == 1
                assert info["hands_detected"] == 0
    
    def test_complete_pipeline_batch_processing(self, node_with_temp_dir, sample_batch_images):
        """Test complete pipeline with batch processing."""
        # Mock ComfyUI models
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        # Mock detection with different results for each image
        detection_results = [
            [{"bbox": [100, 100, 200, 200], "type": "face", "confidence": 0.9}],  # Image 1
            [],  # Image 2 - no detections
            [{"bbox": [50, 50, 150, 150], "type": "face", "confidence": 0.7},    # Image 3
             {"bbox": [300, 300, 400, 400], "type": "hand", "confidence": 0.8}]
        ]
        
        with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
            mock_detector = Mock()
            # Set up detection to return different results based on call count
            mock_detector.detect.side_effect = detection_results
            mock_load.return_value = mock_detector
            
            with patch.object(node_with_temp_dir, '_inpaint_regions') as mock_inpaint:
                # Return input image unchanged for simplicity
                mock_inpaint.side_effect = lambda img, *args, **kwargs: img
                
                # Run batch processing
                result = node_with_temp_dir.process(
                    image=sample_batch_images,
                    model=mock_model,
                    vae=mock_vae,
                    positive=mock_positive,
                    negative=mock_negative,
                    detection_model="yolov8n-face",
                    target_parts="face,hand",
                    confidence_threshold=0.5
                )
                
                # Verify results
                assert len(result) == 5
                processed_image, detection_masks, face_masks, hand_masks, detection_info = result
                
                # Check batch dimensions
                assert processed_image.shape == sample_batch_images.shape
                assert detection_masks.shape == (3, 512, 512)
                
                # Check detection info
                info = json.loads(detection_info)
                assert info["error"] is False
                assert info["total_detections"] == 3  # 1 + 0 + 2
    
    def test_memory_management_integration(self, node_with_temp_dir):
        """Test memory management during processing."""
        # Create a large image to test memory handling
        large_image = torch.rand((1, 1024, 1024, 3))
        
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
            mock_detector = Mock()
            mock_detector.detect.return_value = []  # No detections to simplify
            mock_load.return_value = mock_detector
            
            # Import memory manager if available
            try:
                from utils.memory_utils import MemoryManager
                memory_available = True
            except ImportError:
                memory_available = False
            
            # Run processing
            result = node_with_temp_dir.process(
                image=large_image,
                model=mock_model,
                vae=mock_vae,
                positive=mock_positive,
                negative=mock_negative
            )
            
            # Should complete without memory errors
            assert len(result) == 5
            
            info = json.loads(result[4])
            if memory_available:
                assert "performance_metrics" in info
                assert "memory_efficient" in info["performance_metrics"]
    
    def test_model_manager_integration(self, temp_models_dir):
        """Test ModelManager integration."""
        model_manager = ModelManager(models_dir=str(temp_models_dir))
        
        # Test model info retrieval
        available_models = model_manager.get_available_models()
        assert len(available_models) > 0
        assert "yolov8n-face" in available_models
        
        # Test model info
        model_info = model_manager.get_model_info("yolov8n-face")
        assert "available_locally" in model_info
        assert "loaded_in_cache" in model_info
        assert "download_status" in model_info
        
        # Test cache statistics
        cache_stats = model_manager.get_cache_stats()
        assert "cached_models" in cache_stats
        assert "cache_size" in cache_stats
        assert "max_cache_size" in cache_stats
    
    def test_error_recovery_and_fallbacks(self, node_with_temp_dir):
        """Test error recovery and fallback mechanisms."""
        # Test with invalid detection model
        invalid_image = torch.rand((1, 256, 256, 3))
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        # Mock model loading failure
        with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
            mock_load.return_value = None  # Simulate failure
            
            # Should raise RuntimeError for model loading failure
            with pytest.raises(RuntimeError):
                node_with_temp_dir.process(
                    image=invalid_image,
                    model=mock_model,
                    vae=mock_vae,
                    positive=mock_positive,
                    negative=mock_negative,
                    detection_model="invalid_model"
                )
    
    def test_performance_benchmarking(self, node_with_temp_dir):
        """Test performance characteristics and benchmarking."""
        # Create test images of different sizes
        test_sizes = [
            (1, 256, 256, 3),
            (1, 512, 512, 3),
            (2, 256, 256, 3)  # Batch of 2
        ]
        
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        performance_results = []
        
        for size in test_sizes:
            test_image = torch.rand(size)
            
            with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
                mock_detector = Mock()
                mock_detector.detect.return_value = []
                mock_load.return_value = mock_detector
                
                # Measure processing time
                import time
                start_time = time.time()
                
                result = node_with_temp_dir.process(
                    image=test_image,
                    model=mock_model,
                    vae=mock_vae,
                    positive=mock_positive,
                    negative=mock_negative
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Extract performance metrics
                info = json.loads(result[4])
                pixels_processed = info["performance_metrics"]["pixels_processed"]
                
                performance_results.append({
                    "size": size,
                    "processing_time": processing_time,
                    "pixels_processed": pixels_processed,
                    "pixels_per_second": pixels_processed / processing_time if processing_time > 0 else 0
                })
        
        # Verify performance scaling
        assert len(performance_results) == len(test_sizes)
        
        # Larger images should process more pixels (obviously)
        assert performance_results[1]["pixels_processed"] > performance_results[0]["pixels_processed"]
    
    def test_different_target_combinations(self, node_with_temp_dir):
        """Test different target part combinations."""
        test_image = torch.rand((1, 256, 256, 3))
        mock_model = Mock()
        mock_vae = Mock()
        mock_positive = Mock()
        mock_negative = Mock()
        
        target_combinations = [
            "face",
            "hand", 
            "face,hand",
            "face,hand,finger"
        ]
        
        for targets in target_combinations:
            with patch.object(node_with_temp_dir, '_load_detection_model') as mock_load:
                mock_detector = Mock()
                mock_detector.detect.return_value = []
                mock_load.return_value = mock_detector
                
                result = node_with_temp_dir.process(
                    image=test_image,
                    model=mock_model,
                    vae=mock_vae,
                    positive=mock_positive,
                    negative=mock_negative,
                    target_parts=targets
                )
                
                # Should complete without errors
                assert len(result) == 5
                
                info = json.loads(result[4])
                assert info["parameters"]["target_parts"] == targets


if __name__ == "__main__":
    pytest.main([__file__])