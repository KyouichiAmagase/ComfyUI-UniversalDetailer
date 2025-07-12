#!/usr/bin/env python3
"""
ComfyUI Universal Detailer

Enhanced version of FaceDetailer supporting face, hand, and finger detection/correction.

This custom node extends FaceDetailer functionality to provide comprehensive body part
detection and correction using YOLO models and advanced inpainting techniques.

⚠️  WARNING: This is AI-generated code. Use at your own risk.
⚠️  NO technical support provided. All usage is self-responsibility only.

Author: AI (Claude by Anthropic)
License: Apache 2.0
Version: 1.0.0-dev
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import main node class
try:
    from .universal_detailer import UniversalDetailerNode
    
    # ComfyUI node mapping
    NODE_CLASS_MAPPINGS = {
        "UniversalDetailer": UniversalDetailerNode
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "UniversalDetailer": "Universal Detailer"
    }
    
    # Metadata
    __version__ = "1.0.0-dev"
    __author__ = "AI (Claude by Anthropic)"
    __license__ = "Apache 2.0"
    
    print(f"[Universal Detailer] Loaded successfully (v{__version__})")
    print("[Universal Detailer] ⚠️  AI-generated code - Use at your own risk")
    
except ImportError as e:
    print(f"[Universal Detailer] ERROR: Failed to import main node class: {e}")
    print("[Universal Detailer] Please check installation and dependencies")
    
    # Provide empty mappings to prevent ComfyUI errors
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

except Exception as e:
    print(f"[Universal Detailer] CRITICAL ERROR: {e}")
    print("[Universal Detailer] Node will not be available")
    
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]