#!/usr/bin/env python3
"""
Unit Tests for YOLODetector

Tests the YOLO detection functionality including model loading,
detection processing, and result formatting.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.yolo_detector import YOLODetector

class TestYOLODetector:
    """Test suite for YOLODetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a YOLODetector instance for testing."""
        return YOLODetector("dummy_model.pt", device="cpu")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_yolo_result(self):
        """Create a mock YOLO result for testing."""
        mock_result = Mock()
        mock_result.boxes = Mock()
        
        # Mock detection data
        mock_result.boxes.xyxy = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_result.boxes.conf = torch.tensor([0.9, 0.8])
        mock_result.boxes.cls = torch.tensor([0, 0])
        mock_result.names = {0: "face"}
        
        return [mock_result]
    
    def test_init(self, detector):
        """Test YOLODetector initialization."""
        assert detector.model_path == "dummy_model.pt"
        assert detector.device == "cpu"
        assert detector.model is None
        assert not detector.is_loaded()
    
    def test_determine_device(self, detector):
        """Test device determination logic."""
        # Test auto selection
        with patch('torch.cuda.is_available', return_value=True):
            assert detector._determine_device("auto") == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            assert detector._determine_device("auto") == "cpu"
        
        # Test explicit device
        assert detector._determine_device("cpu") == "cpu"
        assert detector._determine_device("cuda") == "cuda"
    
    @patch('detection.yolo_detector.YOLO')
    def test_load_model_success(self, mock_yolo_class, detector):
        """Test successful model loading."""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        result = detector.load_model()
        
        assert result is True
        assert detector.model == mock_model
        assert detector.is_loaded()
        mock_yolo_class.assert_called_once_with("dummy_model.pt")
    
    @patch('detection.yolo_detector.YOLO')
    def test_load_model_failure(self, mock_yolo_class, detector):
        """Test model loading failure."""
        mock_yolo_class.side_effect = Exception("Model not found")
        
        result = detector.load_model()
        
        assert result is False
        assert detector.model is None
        assert not detector.is_loaded()
    
    def test_detect_without_loaded_model(self, detector, sample_image):
        """Test detection without loaded model."""
        result = detector.detect(sample_image)
        assert result == []
    
    @patch('detection.yolo_detector.YOLO')
    def test_detect_success(self, mock_yolo_class, detector, sample_image, mock_yolo_result):
        """Test successful detection."""
        # Setup mock model
        mock_model = Mock()
        mock_model.return_value = mock_yolo_result
        mock_yolo_class.return_value = mock_model
        
        # Load model and run detection
        detector.load_model()
        result = detector.detect(sample_image, confidence_threshold=0.5)
        
        # Verify results
        assert len(result) == 2
        assert all(isinstance(det, dict) for det in result)
        assert all("bbox" in det for det in result)
        assert all("confidence" in det for det in result)
        assert all("type" in det for det in result)
    
    def test_process_results(self, detector, mock_yolo_result):
        """Test result processing."""
        detector.model = Mock()  # Mock loaded model
        
        result = detector._process_results(mock_yolo_result)
        
        assert len(result) == 2
        
        # Check first detection
        det1 = result[0]
        assert det1["bbox"] == [100, 100, 200, 200]
        assert det1["confidence"] == 0.9
        assert det1["class_name"] == "face"
        assert det1["type"] == "face"
        assert det1["area"] == 10000  # (200-100) * (200-100)
    
    def test_map_class_to_type(self, detector):
        """Test class name to type mapping."""
        assert detector._map_class_to_type("face") == "face"
        assert detector._map_class_to_type("hand") == "hand"
        assert detector._map_class_to_type("finger") == "finger"
        assert detector._map_class_to_type("person") == "person"
        assert detector._map_class_to_type("unknown") == "other"
    
    def test_get_model_info_no_model(self, detector):
        """Test model info without loaded model."""
        info = detector.get_model_info()
        
        assert info["model_path"] == "dummy_model.pt"
        assert info["device"] == "cpu"
        assert info["loaded"] is False
        assert info["classes"] == []
    
    @patch('detection.yolo_detector.YOLO')
    def test_get_model_info_with_model(self, mock_yolo_class, detector):
        """Test model info with loaded model."""
        mock_model = Mock()
        mock_model.names = {0: "face", 1: "hand"}
        mock_yolo_class.return_value = mock_model
        
        detector.load_model()
        info = detector.get_model_info()
        
        assert info["loaded"] is True
        assert info["classes"] == ["face", "hand"]
    
    def test_detect_with_target_classes(self, detector, sample_image):
        """Test detection with target class filtering."""
        with patch.object(detector, '_process_results') as mock_process:
            mock_process.return_value = [
                {"type": "face", "confidence": 0.9},
                {"type": "hand", "confidence": 0.8}
            ]
            
            detector.model = Mock()
            detector.model.return_value = []
            
            # Test with face filter
            result = detector.detect(sample_image, target_classes=["face"])
            mock_process.assert_called_once()
            
            # Verify the target_classes parameter was passed
            call_args = mock_process.call_args
            assert call_args[0][1] == ["face"]
    
    def test_detection_error_handling(self, detector, sample_image):
        """Test error handling during detection."""
        detector.model = Mock()
        detector.model.side_effect = Exception("Detection failed")
        
        result = detector.detect(sample_image)
        assert result == []
    
    @pytest.mark.parametrize("confidence,expected_count", [
        (0.1, 2),  # Both detections should pass
        (0.85, 1), # Only first detection should pass
        (0.95, 0)  # No detections should pass
    ])
    def test_confidence_filtering(self, detector, sample_image, mock_yolo_result, confidence, expected_count):
        """Test confidence threshold filtering."""
        with patch('detection.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_model.return_value = mock_yolo_result
            mock_yolo_class.return_value = mock_model
            
            detector.load_model()
            result = detector.detect(sample_image, confidence_threshold=confidence)
            
            # Count detections that should pass the threshold
            expected_results = [det for det in result if det["confidence"] >= confidence]
            assert len(expected_results) == expected_count


if __name__ == "__main__":
    pytest.main([__file__])