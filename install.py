#!/usr/bin/env python3
"""
Automatic installation script for ComfyUI Universal Detailer

This script handles:
- Dependency installation
- Model downloading
- Initial setup and configuration

Usage:
    python install.py [--force] [--no-models]
"""

import os
import sys
import subprocess
import argparse
import urllib.request
import shutil
from pathlib import Path
import json

# Script configuration
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements.txt"

# Model download URLs (will be updated with actual URLs)
MODEL_URLS = {
    "yolov8n-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "yolov8s-face.pt": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt",
    # Add more model URLs as needed
}

def print_header():
    """Print installation header with warnings."""
    print("=" * 60)
    print("ComfyUI Universal Detailer - Installation Script")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This is AI-generated code!")
    print("⚠️  Use at your own risk - No support provided")
    print("⚠️  Test in isolated environment first")
    print()
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version check passed: {sys.version.split()[0]}")

def check_comfyui():
    """Check if ComfyUI is available."""
    try:
        # Try to find ComfyUI in the parent directories
        current_dir = Path.cwd()
        for parent in current_dir.parents:
            if (parent / "main.py").exists() and (parent / "nodes.py").exists():
                print(f"✅ ComfyUI detected at: {parent}")
                return True
        
        print("⚠️  Warning: ComfyUI not detected in parent directories")
        print("   Make sure this is installed in ComfyUI/custom_nodes/")
        return False
    except Exception as e:
        print(f"⚠️  Warning: Could not verify ComfyUI installation: {e}")
        return False

def install_dependencies(force=False):
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    if not REQUIREMENTS_FILE.exists():
        print(f"❌ Error: {REQUIREMENTS_FILE} not found")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
        if force:
            cmd.append("--force-reinstall")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        MODELS_DIR,
        SCRIPT_DIR / "cache",
        SCRIPT_DIR / "logs",
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory.name}/")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
            return False
    
    return True

def download_file(url, filepath, description="file"):
    """Download a file with progress indication."""
    try:
        print(f"⬇️  Downloading {description}...")
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r   Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✅ Downloaded: {filepath.name}")
        return True
    
    except Exception as e:
        print(f"\n❌ Error downloading {description}: {e}")
        return False

def download_models(force=False):
    """Download required models."""
    print("🤖 Downloading models...")
    
    success_count = 0
    total_count = len(MODEL_URLS)
    
    for filename, url in MODEL_URLS.items():
        filepath = MODELS_DIR / filename
        
        if filepath.exists() and not force:
            print(f"⏭️  Skipping {filename} (already exists)")
            success_count += 1
            continue
        
        if download_file(url, filepath, filename):
            success_count += 1
    
    print(f"\n📊 Model download summary: {success_count}/{total_count} successful")
    return success_count == total_count

def create_config():
    """Create default configuration file."""
    print("⚙️  Creating configuration...")
    
    config = {
        "version": "1.0.0",
        "installation_date": str(Path().cwd()),
        "models_dir": str(MODELS_DIR),
        "default_settings": {
            "detection_model": "yolov8n-face",
            "confidence_threshold": 0.5,
            "mask_padding": 32,
            "inpaint_strength": 0.75
        }
    }
    
    config_file = SCRIPT_DIR / "config.json"
    
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration created: {config_file.name}")
        return True
    
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def verify_installation():
    """Verify that installation was successful."""
    print("🔍 Verifying installation...")
    
    checks = [
        ("Requirements file", REQUIREMENTS_FILE.exists()),
        ("Models directory", MODELS_DIR.exists()),
        ("Config file", (SCRIPT_DIR / "config.json").exists()),
    ]
    
    # Check for at least one model file
    model_files = list(MODELS_DIR.glob("*.pt"))
    checks.append(("Model files", len(model_files) > 0))
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def print_usage_info():
    """Print post-installation usage information."""
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print()
    print("📝 Next steps:")
    print("   1. Restart ComfyUI")
    print("   2. Look for 'Universal Detailer' in the node menu")
    print("   3. Check the EXAMPLES.md file for usage examples")
    print()
    print("⚠️  Important reminders:")
    print("   - This is experimental AI-generated code")
    print("   - Test thoroughly before production use")
    print("   - No technical support is provided")
    print("   - Use at your own risk")
    print()
    print("📚 Documentation:")
    print("   - README.md: General information")
    print("   - SPECIFICATIONS.md: Technical details")
    print("   - API_REFERENCE.md: Node interface")
    print("   - EXAMPLES.md: Usage examples")
    print()
    print("="*60)

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install ComfyUI Universal Detailer")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstall of dependencies and models")
    parser.add_argument("--no-models", action="store_true",
                       help="Skip model downloads")
    
    args = parser.parse_args()
    
    print_header()
    
    # Pre-installation checks
    check_python_version()
    check_comfyui()
    
    # Installation steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", lambda: install_dependencies(args.force)),
        ("Creating configuration", create_config),
    ]
    
    if not args.no_models:
        steps.append(("Downloading models", lambda: download_models(args.force)))
    
    # Execute installation steps
    for step_name, step_function in steps:
        print(f"\n🔄 {step_name}...")
        if not step_function():
            print(f"\n❌ Installation failed at: {step_name}")
            sys.exit(1)
    
    # Verification
    print("\n🔍 Final verification...")
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print_usage_info()
    else:
        print("\n⚠️  Installation completed with warnings")
        print("   Some components may not work properly")
        print("   Check the error messages above")

if __name__ == "__main__":
    main()