#!/usr/bin/env python3
"""
Test Runner for ComfyUI Universal Detailer

Comprehensive test runner that executes all tests and provides
detailed reporting on test results and system compatibility.
"""

import sys
import pytest
import time
import traceback
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "torch": "PyTorch",
        "numpy": "NumPy", 
        "cv2": "OpenCV",
        "pytest": "PyTest"
    }
    
    missing_deps = []
    available_deps = []
    
    for dep, name in dependencies.items():
        try:
            if dep == "cv2":
                import cv2
            else:
                importlib.import_module(dep)
            available_deps.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing_deps.append(name)
            print(f"  ✗ {name} (missing)")
    
    return available_deps, missing_deps

def run_syntax_checks():
    """Run syntax checks on all Python files."""
    print("\n📝 Running syntax checks...")
    
    project_root = Path(__file__).parent.parent
    python_files = [
        "universal_detailer.py",
        "detection/yolo_detector.py",
        "masking/mask_generator.py", 
        "detection/model_loader.py",
        "utils/comfyui_integration.py",
        "utils/sampling_utils.py",
        "utils/memory_utils.py"
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                compile(content, str(full_path), 'exec')
                print(f"  ✓ {file_path}")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
                print(f"  ✗ {file_path}: {e}")
        else:
            print(f"  ? {file_path} (not found)")
    
    return syntax_errors

def run_import_tests():
    """Test if all modules can be imported."""
    print("\n📦 Testing module imports...")
    
    import_tests = [
        ("universal_detailer", "UniversalDetailerNode"),
        ("detection.yolo_detector", "YOLODetector"),
        ("masking.mask_generator", "MaskGenerator"),
        ("detection.model_loader", "ModelManager")
    ]
    
    import_errors = []
    
    for module_name, class_name in import_tests:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"  ✓ {module_name}.{class_name}")
            else:
                error_msg = f"{module_name}: {class_name} not found"
                import_errors.append(error_msg)
                print(f"  ✗ {error_msg}")
        except ImportError as e:
            error_msg = f"{module_name}: {e}"
            import_errors.append(error_msg)
            print(f"  ✗ {error_msg}")
    
    return import_errors

def run_unit_tests():
    """Run unit tests using pytest."""
    print("\n🧪 Running unit tests...")
    
    test_files = [
        "tests/test_yolo_detector.py",
        "tests/test_mask_generator.py", 
        "tests/test_universal_detailer.py"
    ]
    
    # Check which test files exist
    existing_tests = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_tests.append(test_file)
        else:
            print(f"  ? {test_file} (not found)")
    
    if not existing_tests:
        print("  ⚠️  No test files found")
        return False
    
    # Run pytest on existing test files
    try:
        print(f"  Running pytest on {len(existing_tests)} test files...")
        
        # Configure pytest arguments
        pytest_args = [
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--color=yes",  # Colored output
        ] + existing_tests
        
        # Run tests
        result = pytest.main(pytest_args)
        
        if result == 0:
            print("  ✅ All unit tests passed!")
            return True
        else:
            print(f"  ❌ Unit tests failed (exit code: {result})")
            return False
            
    except Exception as e:
        print(f"  ❌ Error running unit tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running integration tests...")
    
    integration_test = "tests/test_integration.py"
    
    if not Path(integration_test).exists():
        print(f"  ? {integration_test} (not found)")
        return False
    
    try:
        # Run integration tests with more verbose output
        pytest_args = [
            "-v",
            "--tb=long",
            "--color=yes",
            "-s",  # Don't capture output
            integration_test
        ]
        
        result = pytest.main(pytest_args)
        
        if result == 0:
            print("  ✅ Integration tests passed!")
            return True
        else:
            print(f"  ❌ Integration tests failed (exit code: {result})")
            return False
            
    except Exception as e:
        print(f"  ❌ Error running integration tests: {e}")
        return False

def run_performance_tests():
    """Run basic performance tests."""
    print("\n⚡ Running performance tests...")
    
    try:
        # Mock performance test without dependencies
        print("  📊 Testing basic performance characteristics...")
        
        # Test 1: Import speed
        start_time = time.time()
        import universal_detailer
        import_time = time.time() - start_time
        print(f"    Import time: {import_time:.3f}s")
        
        # Test 2: Node initialization speed
        start_time = time.time()
        node = universal_detailer.UniversalDetailerNode()
        init_time = time.time() - start_time
        print(f"    Node init time: {init_time:.3f}s")
        
        # Test 3: Basic method calls
        start_time = time.time()
        input_types = node.INPUT_TYPES()
        method_time = time.time() - start_time
        print(f"    Method call time: {method_time:.3f}s")
        
        print("  ✅ Basic performance tests completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance tests failed: {e}")
        traceback.print_exc()
        return False

def generate_test_report(results):
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    # Summary
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result.get('passed', False))
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total checks: {total_checks}")
    print(f"  Passed: {passed_checks}")
    print(f"  Failed: {total_checks - passed_checks}")
    print(f"  Success rate: {(passed_checks/total_checks)*100:.1f}%")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for check_name, result in results.items():
        status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        print(f"  {check_name}: {status}")
        
        if 'errors' in result and result['errors']:
            for error in result['errors']:
                print(f"    - {error}")
        
        if 'details' in result:
            print(f"    Details: {result['details']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    failed_critical = any(not result.get('passed', False) 
                         for name, result in results.items() 
                         if 'critical' in name.lower() or 'syntax' in name.lower())
    
    if failed_critical:
        print("  🚨 Critical issues found - fix syntax errors and imports first")
    elif passed_checks == total_checks:
        print("  🎉 All tests passed! System is ready for use")
    else:
        print("  ⚠️  Some tests failed - review error messages above")
        print("  💡 Consider running individual test files for more detail")
    
    print("\n" + "="*60)
    
    return passed_checks == total_checks

def main():
    """Main test runner function."""
    print("🚀 ComfyUI Universal Detailer - Comprehensive Test Suite")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize results tracking
    results = {}
    
    # 1. Check dependencies
    available_deps, missing_deps = check_dependencies()
    results['dependencies'] = {
        'passed': len(missing_deps) == 0,
        'details': f"{len(available_deps)} available, {len(missing_deps)} missing",
        'errors': [f"Missing: {dep}" for dep in missing_deps]
    }
    
    # 2. Syntax checks
    syntax_errors = run_syntax_checks()
    results['syntax_checks'] = {
        'passed': len(syntax_errors) == 0,
        'details': f"{len(syntax_errors)} syntax errors found",
        'errors': syntax_errors
    }
    
    # 3. Import tests
    import_errors = run_import_tests()
    results['import_tests'] = {
        'passed': len(import_errors) == 0,
        'details': f"{len(import_errors)} import errors found",
        'errors': import_errors
    }
    
    # 4. Unit tests (only if imports work)
    if len(import_errors) == 0:
        unit_test_passed = run_unit_tests()
        results['unit_tests'] = {
            'passed': unit_test_passed,
            'details': "Unit tests executed"
        }
    else:
        results['unit_tests'] = {
            'passed': False,
            'details': "Skipped due to import errors"
        }
    
    # 5. Integration tests (only if unit tests pass)
    if results['unit_tests']['passed']:
        integration_test_passed = run_integration_tests()
        results['integration_tests'] = {
            'passed': integration_test_passed,
            'details': "Integration tests executed"
        }
    else:
        results['integration_tests'] = {
            'passed': False,
            'details': "Skipped due to unit test failures"
        }
    
    # 6. Performance tests
    performance_passed = run_performance_tests()
    results['performance_tests'] = {
        'passed': performance_passed,
        'details': "Basic performance characteristics tested"
    }
    
    # Generate final report
    total_time = time.time() - start_time
    print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
    
    overall_success = generate_test_report(results)
    
    # Exit with appropriate code
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)