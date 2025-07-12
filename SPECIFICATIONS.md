# ComfyUI Universal Detailer - Development Specifications

> **This document contains the complete development specifications for AI developers.**

## Project Overview

### Node Name
UniversalDetailer (or MultiPartDetailer)

### Purpose & Overview
Extend FaceDetailer functionality to support automatic detection and high-quality correction of faces, hands, fingers, and other body parts. Provide ADetailer-like functionality by combining YOLO detection models with various inpainting models to automatically mask and regenerate detected areas with high quality.

### Target Users
- AI image generation creators
- Users wanting to automate face and hand quality improvement
- Users needing batch processing for large image volumes
- Users requiring body part fixes beyond FaceDetailer's capabilities

## Technical Requirements

### ComfyUI Environment
- **ComfyUI Version**: latest (2024+ versions)
- **Python Requirements**: Python 3.8+
- **Required Libraries**: torch, torchvision, ultralytics, opencv-python, numpy, PIL, segment-anything

### Performance Requirements
- **Processing Time**: Detection + correction within 15 seconds for 1024x1024 images
- **Memory Usage**: System RAM 8GB or less
- **GPU Requirements**: VRAM 6GB+ recommended (8GB+ for comfort)
- **CPU Requirements**: 4+ cores recommended for detection processing

## Node Functional Specifications

### Core Features
1. **Multi-part Detection**: Use YOLO v8/v9 for automatic detection of faces, hands, fingers, and other specified parts
2. **Automatic Mask Generation**: Generate precise masks for detected areas
3. **Selective Correction**: Individual or batch inpainting processing for detected parts
4. **Quality Enhancement**: Regenerate detected areas with high-quality models
5. **Batch Processing**: Efficiently process multiple detected parts

### Input Specifications

#### Required Inputs
| Parameter | Type | Description | Default | Constraints |
|-----------|------|-------------|---------|-------------|
| image | IMAGE | Target image for processing | - | Required |
| model | MODEL | Inpainting model | - | Required |
| vae | VAE | VAE encoder/decoder | - | Required |
| positive | CONDITIONING | Positive prompt | - | Required |
| negative | CONDITIONING | Negative prompt | - | Required |

#### Optional Inputs
| Parameter | Type | Description | Default | Constraints |
|-----------|------|-------------|---------|-------------|
| detection_model | STRING | Detection model type | "yolov8n-face" | yolov8n-face, yolov8s-face, hand_yolov8n, etc. |
| target_parts | STRING | Target detection parts | "face,hand" | face, hand, finger, person, etc. (comma-separated) |
| confidence_threshold | FLOAT | Detection confidence threshold | 0.5 | 0.1-0.95 |
| mask_padding | INT | Mask expansion pixels | 32 | 0-128 |
| inpaint_strength | FLOAT | Inpainting strength | 0.75 | 0.1-1.0 |
| steps | INT | Sampling steps | 20 | 1-100 |
| cfg_scale | FLOAT | CFG scale | 7.0 | 1.0-30.0 |
| seed | INT | Random seed | -1 | -1 or positive |
| sampler_name | STRING | Sampler name | "euler" | ComfyUI standard samplers |
| scheduler | STRING | Scheduler name | "normal" | ComfyUI standard schedulers |
| auto_face_fix | BOOLEAN | Enable automatic face correction | True | True/False |
| auto_hand_fix | BOOLEAN | Enable automatic hand correction | True | True/False |
| mask_blur | INT | Mask blur amount | 4 | 0-20 |

### Output Specifications

| Output | Type | Description |
|--------|------|-----------|
| image | IMAGE | Corrected image |
| detection_masks | MASK | Combined detection masks |
| face_masks | MASK | Face-only masks |
| hand_masks | MASK | Hand-only masks |
| detection_info | STRING | Detection results details (JSON format) |

### Processing Flow
1. **Image Loading**: Acquire input image and perform preprocessing
2. **Part Detection**: Detect target parts using specified detection model
3. **Confidence Filtering**: Exclude detection results below threshold
4. **Mask Generation**: Generate masks for each detected part
5. **Mask Integration**: Combine multiple masks and apply padding and blur
6. **Inpainting Processing**: Regenerate masked areas using model
7. **Result Composition**: Naturally composite original image with corrected parts
8. **Post-processing**: Color adjustment and boundary blending

## UI Requirements

### Node Display
- **Node Title**: "Universal Detailer"
- **Node Category**: "image/postprocessing"
- **Node Color**: Blue-based (easily identifiable as detection/correction node)

### Input Widgets
- **detection_model**: Dropdown menu (display available detection models)
- **target_parts**: Multi-select or text input
- **confidence_threshold**: Slider (0.1-0.95, step 0.05)
- **mask_padding**: Slider (0-128, step 4)
- **inpaint_strength**: Slider (0.1-1.0, step 0.05)
- **steps**: Slider (1-100, step 1)
- **cfg_scale**: Slider (1.0-30.0, step 0.5)
- **auto_face_fix/auto_hand_fix**: Checkboxes

### Preview Features
- Detection result bounding box display
- Generated mask preview display
- Before/after comparison preview

## Implementation Requirements

### File Structure
```
custom_nodes/
└── ComfyUI-UniversalDetailer/
    ├── __init__.py
    ├── universal_detailer.py
    ├── detection/
    │   ├── __init__.py
    │   ├── yolo_detector.py
    │   └── model_loader.py
    ├── masking/
    │   ├── __init__.py
    │   └── mask_generator.py
    ├── models/
    │   └── download_models.py
    ├── utils/
    │   ├── __init__.py
    │   ├── image_utils.py
    │   └── mask_utils.py
    ├── requirements.txt
    ├── README.md
    └── install.py
```

### Class Structure
```python
class UniversalDetailerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                "detection_model": (["yolov8n-face", "yolov8s-face", "hand_yolov8n"], {"default": "yolov8n-face"}),
                "target_parts": ("STRING", {"default": "face,hand"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05}),
                # ... other parameters
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "detection_masks", "face_masks", "hand_masks", "detection_info")
    FUNCTION = "process"
    CATEGORY = "image/postprocessing"
    
    def process(self, image, model, vae, positive, negative, **kwargs):
        # Main processing logic
        pass
```

### Error Handling
- Alternative processing when detection model loading fails
- Automatic batch size adjustment during VRAM shortage
- Continued processing when no detection results found
- Automatic correction of invalid parameter values
- User-friendly error messages (Japanese support)
- Memory cleanup during processing interruption

### Configuration & Caching
- Local caching of detection models (models/ folder)
- User settings storage (JSON format)
- Automatic model download on first run
- Cache size limits and cleanup functionality

## Dependencies

### External Libraries
| Library | Version | Purpose | Required/Optional |
|---------|---------|---------|-------------------|
| ultralytics | >=8.0.0 | YOLO detection models | Required |
| opencv-python | >=4.5.0 | Image processing | Required |
| numpy | >=1.20.0 | Numerical computation | Required |
| PIL | >=8.0.0 | Image I/O | Required |
| torch | >=1.12.0 | Model execution | Required |
| torchvision | >=0.13.0 | Image transformations | Required |
| segment-anything | latest | Segmentation | Optional |
| mediapipe | >=0.9.0 | Hand landmark detection | Optional |

### Models & Data
- YOLOv8 face detection model (auto-download)
- YOLOv8 hand detection model (auto-download)
- MediaPipe hand landmark model (optional)
- Sample image data (for testing)

## Testing Requirements

### Unit Tests
- **Normal cases**: Verify operation with typical images containing faces and hands
- **Boundary values**: Processing with extremely small/large images
- **Error cases**: Operation with images where faces/hands are not detected
- **Performance**: Processing time with large images (2048x2048+)
- **Memory**: Memory leak verification during continuous processing

### Integration Tests
- Actual workflow execution in ComfyUI
- Integration with other preprocessing/postprocessing nodes
- Operation verification with different models (SD1.5, SDXL, etc.)
- Stability testing during batch processing

### Test Data
- Images containing only faces (portraits)
- Images containing only hands (hand photos)
- Images containing both faces and hands (full-body photos)
- Images with multiple people
- Anime/illustration images
- Low-resolution and high-resolution images

## Documentation Requirements

### README.md
- **Installation procedures**: ComfyUI Manager support + manual installation
- **Usage instructions**: Basic workflow examples
- **Parameter explanations**: Detailed descriptions of each setting
- **Sample workflows**: JSON files and screenshots
- **Troubleshooting**: Common problems and solutions
- **Model requirements**: Recommended detection and inpainting models

### In-code Documentation
- Docstrings for all functions and classes (Japanese/English)
- Comments on important algorithm sections
- Detailed parameter descriptions
- Return value format explanations

## Distribution & Installation

### Installation Methods
- **ComfyUI Manager support**: Compatible with search/install functionality
- **Manual installation**: Clone procedures from GitHub
- **Automatic dependency resolution**: Automatic setup via requirements.txt and install.py
- **Automatic model download**: Automatic model acquisition on first run

### License
Apache License 2.0 (Commercial use allowed, open source)

## Additional Requirements & Special Notes

### Performance Optimization
- Detection efficiency through batch processing
- GPU/CPU parallel processing optimization
- Dynamic memory usage adjustment
- Acceleration through caching functionality

### Extensibility
- Easy addition of new detection models
- Support for custom part detection
- Plugin-style feature extensions

### Internationalization
- Multi-language support for error messages
- English version documentation creation

---

## AI Developer Instructions

### Implementation Priority
1. **Highest**: Face detection, mask generation, basic inpainting functionality
2. **High**: Hand detection functionality, parameter adjustment features
3. **Medium**: UI improvements, enhanced error handling
4. **Low**: Preview functionality, batch processing optimization

### Development Guidelines
- **ComfyUI Standards**: Follow existing node coding styles
- **Memory Management**: Proper use of torch.no_grad() and garbage collection
- **Error Handling**: Appropriate error catching with try-except statements
- **Logging**: Use logging module instead of print
- **Type Hints**: Active use of Python type hints

### Reference Implementations
- **FaceDetailer**: Face detection and mask generation logic
- **ADetailer**: Multi-part support and UI design
- **ComfyUI Standard Nodes**: Inpainting processing implementation methods

### Verification Method
1. Create new workflow in ComfyUI
2. Build basic image generation pipeline
3. Add UniversalDetailer node
4. Verify operation with test images
5. Parameter adjustment testing
6. Operation verification in error cases

### Deliverables
- [ ] Complete implementation code (fully functional version)
- [ ] requirements.txt (version pinned)
- [ ] README.md (detailed usage instructions)
- [ ] install.py (automatic setup script)
- [ ] Sample workflows (JSON + explanations)
- [ ] Basic operation verification results (screenshots)
- [ ] List of known issues/limitations

### Development Completion Criteria
- Face and hand detection operates normally
- Mask generation works appropriately
- Inpainting processing outputs expected results
- Completes without errors
- No memory leaks occur
- Normal integration with other nodes