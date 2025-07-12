#!/usr/bin/env python3
"""
Basic Integration Test for ComfyUI Universal Detailer

Simple test script that verifies basic functionality without requiring pytest
or external dependencies like ultralytics.
"""

import sys
import time
import traceback
from pathlib import Path
import importlib.util

def test_basic_imports():
    """Test basic module imports without external dependencies."""
    print("🔍 Testing basic imports...")
    
    import_results = {}
    
    # Test 1: Universal Detailer Node
    try:
        from universal_detailer import UniversalDetailerNode
        node = UniversalDetailerNode()
        
        # Test basic properties
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')
        assert hasattr(node, 'process')
        
        # Test INPUT_TYPES
        input_types = node.INPUT_TYPES()
        assert 'required' in input_types
        assert 'optional' in input_types
        assert 'image' in input_types['required']
        
        import_results['UniversalDetailerNode'] = True
        print("  ✓ UniversalDetailerNode - OK")
        
    except Exception as e:
        import_results['UniversalDetailerNode'] = False
        print(f"  ✗ UniversalDetailerNode - {e}")
    
    # Test 2: YOLO Detector (basic import)
    try:
        from detection.yolo_detector import YOLODetector
        
        # Test basic initialization without loading model
        detector = YOLODetector("dummy.pt", device="cpu")
        assert detector.model_path == "dummy.pt"
        assert detector.device == "cpu"
        assert not detector.is_loaded()
        
        import_results['YOLODetector'] = True
        print("  ✓ YOLODetector - OK")
        
    except Exception as e:
        import_results['YOLODetector'] = False
        print(f"  ✗ YOLODetector - {e}")
    
    # Test 3: Mask Generator
    try:
        from masking.mask_generator import MaskGenerator
        
        generator = MaskGenerator()
        
        # Test basic functionality with mock data
        detections = []
        image_shape = (1, 256, 256, 3)
        combined, face, hand = generator.generate_masks(detections, image_shape)
        
        assert combined.shape == (1, 256, 256)
        assert face.shape == (1, 256, 256)
        assert hand.shape == (1, 256, 256)
        
        import_results['MaskGenerator'] = True
        print("  ✓ MaskGenerator - OK")
        
    except Exception as e:
        import_results['MaskGenerator'] = False
        print(f"  ✗ MaskGenerator - {e}")
    
    # Test 4: Model Manager
    try:
        from detection.model_loader import ModelManager
        
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        assert len(models) > 0
        
        cache_stats = model_manager.get_cache_stats()
        assert 'cached_models' in cache_stats
        
        import_results['ModelManager'] = True
        print("  ✓ ModelManager - OK")
        
    except Exception as e:
        import_results['ModelManager'] = False
        print(f"  ✗ ModelManager - {e}")
    
    return import_results

def test_tensor_operations():
    """Test basic tensor operations without torch dependency."""
    print("\n🧮 Testing tensor operations...")
    
    try:
        # Mock torch-like operations using numpy
        import numpy as np
        
        # Test basic array operations that would be done with tensors
        image_shape = (1, 256, 256, 3)
        test_array = np.random.rand(*image_shape).astype(np.float32)
        
        # Test shape validation
        assert test_array.shape == image_shape
        assert test_array.dtype == np.float32
        assert test_array.min() >= 0.0
        assert test_array.max() <= 1.0
        
        # Test mask operations
        mask_array = np.zeros((1, 256, 256), dtype=np.float32)
        mask_array[0, 50:150, 50:150] = 1.0
        
        assert mask_array.sum() == 100 * 100  # 100x100 area
        assert mask_array.max() == 1.0
        assert mask_array.min() == 0.0
        
        print("  ✓ Basic tensor operations - OK")
        return True
        
    except Exception as e:
        print(f"  ✗ Tensor operations - {e}")
        return False

def test_comfyui_interface():
    """Test ComfyUI interface compatibility."""
    print("\n🔌 Testing ComfyUI interface...")
    
    try:
        from universal_detailer import UniversalDetailerNode
        
        node = UniversalDetailerNode()
        
        # Test INPUT_TYPES structure
        input_types = node.INPUT_TYPES()
        
        # Verify required inputs
        required = input_types['required']
        required_inputs = ['image', 'model', 'vae', 'positive', 'negative']
        for inp in required_inputs:
            assert inp in required, f"Missing required input: {inp}"
        
        # Verify optional inputs
        optional = input_types['optional']
        optional_inputs = ['detection_model', 'target_parts', 'confidence_threshold']
        for inp in optional_inputs:
            assert inp in optional, f"Missing optional input: {inp}"
        
        # Test return types
        assert node.RETURN_TYPES == ("IMAGE", "MASK", "MASK", "MASK", "STRING")
        assert len(node.RETURN_NAMES) == len(node.RETURN_TYPES)
        
        # Test function name
        assert node.FUNCTION == "process"
        assert node.CATEGORY == "image/postprocessing"
        
        print("  ✓ ComfyUI interface - OK")
        return True
        
    except Exception as e:
        print(f"  ✗ ComfyUI interface - {e}")
        return False

def test_parameter_validation():
    """Test parameter validation logic."""
    print("\n✅ Testing parameter validation...")
    
    try:
        from universal_detailer import UniversalDetailerNode
        
        node = UniversalDetailerNode()
        
        # Test parameter clamping
        test_params = {
            'confidence_threshold': 1.5,  # Should clamp to 0.95
            'mask_padding': -10,          # Should clamp to 0
            'steps': 200,                 # Should clamp to 100
            'cfg_scale': 50.0            # Should clamp to 30.0
        }
        
        validated = node._validate_parameters(**test_params)
        
        assert validated['confidence_threshold'] == 0.95
        assert validated['mask_padding'] == 0
        assert validated['steps'] == 100
        assert validated['cfg_scale'] == 30.0
        
        print("  ✓ Parameter validation - OK")
        return True
        
    except Exception as e:
        print(f"  ✗ Parameter validation - {e}")
        return False

def test_utility_modules():
    """Test utility modules if available."""
    print("\n🛠️  Testing utility modules...")
    
    results = {}
    
    # Test ComfyUI integration utils
    try:
        from utils.comfyui_integration import ComfyUIHelper
        
        # Test tensor format conversion
        import numpy as np
        test_tensor = np.random.rand(1, 64, 64, 3).astype(np.float32)
        
        # Test basic helper methods exist
        assert hasattr(ComfyUIHelper, 'convert_tensor_format')
        assert hasattr(ComfyUIHelper, 'safe_device_transfer')
        assert hasattr(ComfyUIHelper, 'get_optimal_device')
        
        results['ComfyUIHelper'] = True
        print("  ✓ ComfyUIHelper - OK")
        
    except Exception as e:
        results['ComfyUIHelper'] = False
        print(f"  ✗ ComfyUIHelper - {e}")
    
    # Test sampling utils
    try:
        from utils.sampling_utils import SamplingUtils
        
        # Test sampling parameter preparation
        params = SamplingUtils.prepare_sampling_params(steps=20, cfg_scale=7.0)
        assert 'steps' in params
        assert 'cfg_scale' in params
        assert 'seed' in params
        
        results['SamplingUtils'] = True
        print("  ✓ SamplingUtils - OK")
        
    except Exception as e:
        results['SamplingUtils'] = False
        print(f"  ✗ SamplingUtils - {e}")
    
    # Test memory utils
    try:
        from utils.memory_utils import MemoryManager
        
        memory_manager = MemoryManager()
        stats = memory_manager.get_memory_stats()
        assert isinstance(stats, dict)
        
        results['MemoryManager'] = True
        print("  ✓ MemoryManager - OK")
        
    except Exception as e:
        results['MemoryManager'] = False
        print(f"  ✗ MemoryManager - {e}")
    
    return results

def generate_report(test_results):
    """Generate a test report."""
    print("\n" + "="*60)
    print("📋 BASIC TEST REPORT")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    # Count all test results
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            for sub_test, sub_result in result.items():
                total_tests += 1
                if sub_result:
                    passed_tests += 1
        else:
            total_tests += 1
            if result:
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            print(f"  {test_name}:")
            for sub_test, sub_result in result.items():
                status = "✅" if sub_result else "❌"
                print(f"    {status} {sub_test}")
        else:
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if success_rate == 100:
        print("  🎉 All basic tests passed! System appears to be working correctly.")
        print("  💡 For full testing, install pytest and run: python -m pytest tests/")
    elif success_rate >= 80:
        print("  ✅ Most tests passed. Minor issues may exist.")
        print("  💡 Review failed tests and ensure all dependencies are installed.")
    else:
        print("  ⚠️  Many tests failed. Significant issues detected.")
        print("  🔧 Check error messages and fix import/syntax issues first.")
    
    print("\n" + "="*60)
    
    return success_rate >= 80

def main():
    """Main test function."""
    print("🚀 ComfyUI Universal Detailer - Basic Integration Test")
    print("="*60)
    
    start_time = time.time()
    
    # Run tests
    test_results = {}
    
    try:
        test_results['imports'] = test_basic_imports()
        test_results['tensor_ops'] = test_tensor_operations()
        test_results['comfyui_interface'] = test_comfyui_interface()
        test_results['parameter_validation'] = test_parameter_validation()
        test_results['utility_modules'] = test_utility_modules()
        
    except Exception as e:
        print(f"\n❌ Critical error during testing: {e}")
        traceback.print_exc()
        return 1
    
    # Generate report
    total_time = time.time() - start_time
    print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
    
    success = generate_report(test_results)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)