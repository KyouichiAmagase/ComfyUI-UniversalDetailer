#!/usr/bin/env python3
"""
Setup script for ComfyUI Universal Detailer

This script provides installation and setup utilities for the Universal Detailer
ComfyUI custom node.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "ComfyUI Universal Detailer - Enhanced face, hand, and finger detection/correction"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0", 
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0"
    ]

setup(
    name="comfyui-universal-detailer",
    version="2.0.0",
    description="ComfyUI custom node for face, hand, and finger detection and enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Claude AI (Anthropic)",
    author_email="claude@anthropic.com",
    url="https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer",
    packages=find_packages(exclude=["tests*", "workflow_examples*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910"
        ],
        "performance": [
            "psutil>=5.8.0",
            "GPUtil>=1.4.0"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords=[
        "comfyui", "stable-diffusion", "face-detection", "hand-detection", 
        "image-enhancement", "inpainting", "yolo", "ai", "machine-learning"
    ],
    project_urls={
        "Documentation": "https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/blob/main/README.md",
        "Source": "https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer",
        "Tracker": "https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/issues",
        "Changelog": "https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/blob/main/CODE_REVIEW_FIXES_REPORT.md"
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
        "workflow_examples": ["*.json", "*.md"],
        "tests": ["*.py"],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "universal-detailer-test=tests.run_tests:main",
            "universal-detailer-benchmark=utils.performance_utils:run_performance_benchmark",
        ],
    },
)