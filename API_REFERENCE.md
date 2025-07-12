# API Reference - Universal Detailer Node

> **Complete API documentation for UniversalDetailer node**

## Node Interface

### Class Definition

```python
class UniversalDetailerNode:
    """
    Universal Detailer Node for ComfyUI
    
    Enhanced version of FaceDetailer supporting multi-part detection
    and correction including faces, hands, and fingers.
    """
```

### Node Metadata

```python
RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "STRING")
RETURN_NAMES = ("image", "detection_masks", "face_masks", "hand_masks", "detection_info")
FUNCTION = "process"
CATEGORY = "image/postprocessing"
```

## Input Types

### Required Inputs

#### `image`
- **Type**: `IMAGE`
- **Description**: Input image to be processed
- **Format**: ComfyUI standard IMAGE tensor
- **Constraints**: Any resolution, RGB format

#### `model`
- **Type**: `MODEL`
- **Description**: Inpainting model for regenerating detected areas
- **Format**: ComfyUI MODEL object
- **Constraints**: Compatible inpainting models (SD, SDXL, etc.)

#### `vae`
- **Type**: `VAE`
- **Description**: VAE encoder/decoder for image processing
- **Format**: ComfyUI VAE object
- **Constraints**: Compatible with the provided model

#### `positive`
- **Type**: `CONDITIONING`
- **Description**: Positive conditioning for inpainting
- **Format**: ComfyUI CONDITIONING object
- **Constraints**: Appropriate for the target generation

#### `negative`
- **Type**: `CONDITIONING`
- **Description**: Negative conditioning for inpainting
- **Format**: ComfyUI CONDITIONING object
- **Constraints**: Appropriate for avoiding unwanted features

### Optional Inputs

#### `detection_model`
- **Type**: `STRING`
- **Description**: Detection model type for part identification
- **Default**: `"yolov8n-face"`
- **Options**: 
  - `"yolov8n-face"`: Lightweight face detection
  - `"yolov8s-face"`: Standard face detection
  - `"hand_yolov8n"`: Hand detection model
  - `"yolov8n-pose"`: Pose detection model

#### `target_parts`
- **Type**: `STRING`
- **Description**: Comma-separated list of parts to detect and correct
- **Default**: `"face,hand"`
- **Format**: `"part1,part2,part3"`
- **Valid Parts**:
  - `face`: Human faces
  - `hand`: Hands and fingers
  - `finger`: Specific finger detection
  - `person`: Full person detection

#### `confidence_threshold`
- **Type**: `FLOAT`
- **Description**: Minimum confidence score for detection acceptance
- **Default**: `0.5`
- **Range**: `0.1` to `0.95`
- **Step**: `0.05`
- **Usage**: Higher values = fewer, more confident detections

#### `mask_padding`
- **Type**: `INT`
- **Description**: Number of pixels to expand mask boundaries
- **Default**: `32`
- **Range**: `0` to `128`
- **Step**: `4`
- **Usage**: Larger values include more surrounding context

#### `inpaint_strength`
- **Type**: `FLOAT`
- **Description**: Strength of inpainting effect
- **Default**: `0.75`
- **Range**: `0.1` to `1.0`
- **Step**: `0.05`
- **Usage**: Higher values = more dramatic changes

#### `steps`
- **Type**: `INT`
- **Description**: Number of sampling steps for inpainting
- **Default**: `20`
- **Range**: `1` to `100`
- **Step**: `1`
- **Usage**: More steps = higher quality, longer processing

#### `cfg_scale`
- **Type**: `FLOAT`
- **Description**: Classifier-free guidance scale
- **Default**: `7.0`
- **Range**: `1.0` to `30.0`
- **Step**: `0.5`
- **Usage**: Higher values = stronger prompt adherence

#### `seed`
- **Type**: `INT`
- **Description**: Random seed for reproducible results
- **Default**: `-1` (random)
- **Range**: `-1` or any positive integer
- **Usage**: Same seed = same results

#### `sampler_name`
- **Type**: `STRING`
- **Description**: Sampling algorithm for inpainting
- **Default**: `"euler"`
- **Options**: All ComfyUI standard samplers
  - `"euler"`, `"euler_ancestral"`, `"heun"`, `"dpm_2"`, etc.

#### `scheduler`
- **Type**: `STRING`
- **Description**: Noise schedule for sampling
- **Default**: `"normal"`
- **Options**: All ComfyUI standard schedulers
  - `"normal"`, `"karras"`, `"exponential"`, etc.

#### `auto_face_fix`
- **Type**: `BOOLEAN`
- **Description**: Automatically fix detected faces
- **Default**: `True`
- **Options**: `True` / `False`

#### `auto_hand_fix`
- **Type**: `BOOLEAN`
- **Description**: Automatically fix detected hands
- **Default**: `True`
- **Options**: `True` / `False`

#### `mask_blur`
- **Type**: `INT`
- **Description**: Blur radius for mask edges
- **Default**: `4`
- **Range**: `0` to `20`
- **Step**: `1`
- **Usage**: Higher values = softer mask transitions

## Output Types

### `image`
- **Type**: `IMAGE`
- **Description**: Final processed image with corrections applied
- **Format**: ComfyUI standard IMAGE tensor
- **Content**: Original image with detected parts corrected

### `detection_masks`
- **Type**: `MASK`
- **Description**: Combined mask of all detected areas
- **Format**: ComfyUI standard MASK tensor
- **Content**: White areas indicate detected/processed regions

### `face_masks`
- **Type**: `MASK`
- **Description**: Mask containing only face detections
- **Format**: ComfyUI standard MASK tensor
- **Content**: Isolated face detection areas

### `hand_masks`
- **Type**: `MASK`
- **Description**: Mask containing only hand detections
- **Format**: ComfyUI standard MASK tensor
- **Content**: Isolated hand detection areas

### `detection_info`
- **Type**: `STRING`
- **Description**: Detailed information about detection results
- **Format**: JSON string
- **Content**: Detection statistics, confidence scores, processing time

#### Detection Info JSON Structure
```json
{
  "total_detections": 3,
  "faces_detected": 1,
  "hands_detected": 2,
  "processing_time": 12.5,
  "detections": [
    {
      "type": "face",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2],
      "processed": true
    },
    {
      "type": "hand",
      "confidence": 0.76,
      "bbox": [x1, y1, x2, y2], 
      "processed": true
    }
  ]
}
```

## Method Signatures

### `INPUT_TYPES()`
```python
@classmethod
def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
    """
    Returns the input type definitions for the node.
    
    Returns:
        Dict containing 'required' and 'optional' input specifications
    """
```

### `process()`
```python
def process(
    self,
    image: torch.Tensor,
    model: Any,
    vae: Any,
    positive: Any,
    negative: Any,
    detection_model: str = "yolov8n-face",
    target_parts: str = "face,hand",
    confidence_threshold: float = 0.5,
    mask_padding: int = 32,
    inpaint_strength: float = 0.75,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    auto_face_fix: bool = True,
    auto_hand_fix: bool = True,
    mask_blur: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Main processing function for universal detection and correction.
    
    Args:
        image: Input image tensor
        model: Inpainting model
        vae: VAE encoder/decoder
        positive: Positive conditioning
        negative: Negative conditioning
        **kwargs: Optional parameters as defined above
    
    Returns:
        Tuple of (processed_image, detection_masks, face_masks, hand_masks, detection_info)
    
    Raises:
        ValueError: Invalid parameter values
        RuntimeError: Processing errors
        MemoryError: Insufficient GPU/system memory
    """
```

## Error Handling

### Exception Types

#### `DetectionModelError`
- **Cause**: Detection model loading/execution failure
- **Recovery**: Falls back to alternative model or skips detection

#### `InpaintingError`
- **Cause**: Inpainting process failure
- **Recovery**: Returns original image with error information

#### `MemoryError`
- **Cause**: Insufficient GPU/system memory
- **Recovery**: Automatic batch size reduction, garbage collection

#### `ValidationError`
- **Cause**: Invalid input parameters
- **Recovery**: Auto-correction to valid ranges

### Error Response Format
```python
{
    "error": True,
    "error_type": "DetectionModelError",
    "message": "Failed to load detection model",
    "recovery_action": "Using fallback model",
    "processing_completed": False
}
```

## Usage Examples

### Basic Usage
```python
# Minimal setup
result = node.process(
    image=input_image,
    model=inpaint_model,
    vae=vae_model,
    positive=positive_prompt,
    negative=negative_prompt
)
```

### Advanced Configuration
```python
# Custom configuration
result = node.process(
    image=input_image,
    model=inpaint_model,
    vae=vae_model,
    positive=positive_prompt,
    negative=negative_prompt,
    detection_model="yolov8s-face",
    target_parts="face,hand,finger",
    confidence_threshold=0.7,
    mask_padding=48,
    inpaint_strength=0.85,
    steps=30,
    cfg_scale=8.5
)
```

### Face-Only Processing
```python
# Face correction only
result = node.process(
    image=input_image,
    model=inpaint_model,
    vae=vae_model,
    positive=positive_prompt,
    negative=negative_prompt,
    target_parts="face",
    auto_hand_fix=False
)
```

## Performance Considerations

### Memory Usage
- **Detection Phase**: ~500MB VRAM for YOLO models
- **Inpainting Phase**: Varies by model size (2-8GB)
- **Peak Usage**: Detection + Inpainting + Original image

### Processing Time
- **Detection**: 1-3 seconds per image
- **Mask Generation**: <1 second
- **Inpainting**: 5-15 seconds (depends on steps, model)
- **Total**: 7-20 seconds for typical workflow

### Optimization Tips
- Lower `steps` for faster processing
- Reduce `confidence_threshold` for more detections
- Use smaller detection models for speed
- Enable specific part processing only when needed