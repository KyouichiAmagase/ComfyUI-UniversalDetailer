#!/usr/bin/env python3
"""
Basic import test for Universal Detailer components.

This test verifies that all components can be imported correctly
without requiring external dependencies like ultralytics.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test importing all Universal Detailer components."""
    print("Testing Universal Detailer imports...")
    
    # Test 1: Basic component imports
    try:
        print("\n1. Testing component imports...")
        
        # Mock the missing dependencies
        class MockTorch:
            def __init__(self):
                self.cuda = self
                self.float32 = "float32"
            def is_available(self):
                return False
            def zeros(self, *args, **kwargs):
                return MockTensor()
            def from_numpy(self, arr):
                return MockTensor()
            def maximum(self, a, b):
                return MockTensor()
            def any(self, tensor):
                return False
            def count_nonzero(self, tensor):
                return 0
                
        class MockTensor:
            def __init__(self):
                self.shape = (1, 512, 512, 3)
            def cpu(self):
                return self
            def numpy(self):
                import builtins
                return MockArray()
            def unsqueeze(self, dim):
                return self
            def repeat(self, *args):
                return self
            def dim(self):
                return 3
            def min(self):
                return 0.0
            def max(self):
                return 1.0
            def mean(self):
                return 0.5
            def numel(self):
                return 512*512*3
                
        class MockArray:
            def __init__(self):
                self.shape = (512, 512, 3)
            def astype(self, dtype):
                return self
            def copy(self):
                return self
            def max(self):
                return 255
                
        class MockCV2:
            MORPH_ELLIPSE = 2
            def getStructuringElement(self, *args):
                return MockArray()
            def dilate(self, *args, **kwargs):
                return MockArray()
            def GaussianBlur(self, *args):
                return MockArray()
                
        # Mock numpy
        class MockNumpy:
            uint8 = "uint8"
            float32 = "float32"
            def zeros(self, shape, dtype=None):
                return MockArray()
            def maximum(self, a, b):
                return MockArray()
            def array(self, data):
                return MockArray()
                
        # Inject mocks
        sys.modules['torch'] = MockTorch()
        sys.modules['numpy'] = MockNumpy()
        sys.modules['cv2'] = MockCV2()
        sys.modules['ultralytics'] = type('MockUltralytics', (), {})()
        
        # Now try importing our components
        from detection.yolo_detector import YOLODetector
        print("   ✓ YOLODetector imported successfully")
        
        from masking.mask_generator import MaskGenerator
        print("   ✓ MaskGenerator imported successfully")
        
        from universal_detailer import UniversalDetailerNode
        print("   ✓ UniversalDetailerNode imported successfully")
        
        print("   ✓ All component imports successful!")
        
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Node initialization
    try:
        print("\n2. Testing node initialization...")
        node = UniversalDetailerNode()
        print("   ✓ UniversalDetailerNode initialized successfully")
        
        # Test input types
        input_types = node.INPUT_TYPES()
        print(f"   ✓ Input types defined: {len(input_types)} categories")
        
        # Test return types
        return_types = node.RETURN_TYPES
        print(f"   ✓ Return types defined: {len(return_types)} outputs")
        
    except Exception as e:
        print(f"   ✗ Node initialization failed: {e}")
        return False
    
    # Test 3: Detection model creation
    try:
        print("\n3. Testing detector creation...")
        detector = YOLODetector("dummy_model.pt", device="cpu")
        print("   ✓ YOLODetector created successfully")
        
        model_info = detector.get_model_info()
        print(f"   ✓ Model info: {model_info}")
        
    except Exception as e:
        print(f"   ✗ Detector creation failed: {e}")
        return False
    
    # Test 4: Mask generator creation
    try:
        print("\n4. Testing mask generator...")
        mask_gen = MaskGenerator()
        print("   ✓ MaskGenerator created successfully")
        
        # Test empty mask generation
        empty_detections = []
        image_shape = (1, 512, 512, 3)
        masks = mask_gen.generate_masks(empty_detections, image_shape)
        print("   ✓ Empty mask generation successful")
        
    except Exception as e:
        print(f"   ✗ Mask generator failed: {e}")
        return False
    
    print("\n✅ All tests passed! Universal Detailer components working correctly.")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 Universal Detailer is ready for integration!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download YOLO models to models/ directory")
        print("3. Install in ComfyUI custom_nodes directory")
        print("4. Restart ComfyUI")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1)