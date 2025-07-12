# Usage Examples - Universal Detailer

> **Practical examples and workflows for Universal Detailer node**

## Basic Workflows

### Example 1: Simple Face and Hand Correction

**Scenario**: Improve AI-generated portrait with face and hand issues

**Workflow**:
```
Load Image → Universal Detailer → Save Image
             ↑
         [Model, VAE, Prompts]
```

**Settings**:
- `target_parts`: "face,hand"
- `confidence_threshold`: 0.5
- `inpaint_strength`: 0.75
- `steps`: 20

**Expected Results**:
- Automatically detects and fixes facial features
- Corrects hand anatomy and finger details
- Maintains overall image composition

### Example 2: Face-Only Enhancement

**Scenario**: Focus only on facial improvement for portrait photography

**Settings**:
- `target_parts`: "face"
- `auto_hand_fix`: False
- `confidence_threshold`: 0.7
- `mask_padding`: 40
- `inpaint_strength`: 0.6

**Use Case**: 
- Professional portrait retouching
- Maintaining natural hand appearance
- Conservative enhancement approach

### Example 3: Hand-Focused Correction

**Scenario**: Fix hand deformities in AI art while preserving face

**Settings**:
- `target_parts`: "hand,finger"
- `auto_face_fix`: False
- `confidence_threshold`: 0.4
- `inpaint_strength`: 0.85
- `steps`: 25

**Benefits**:
- Specialized hand anatomy correction
- Detailed finger structure improvement
- Face remains untouched

## Advanced Workflows

### Example 4: Multi-Stage Processing

**Workflow Chain**:
```
Base Generation → Upscaler → Universal Detailer → Final Touch-ups
                               ↓
                         [High-res correction]
```

**Process**:
1. Generate base image (512x512)
2. Upscale to higher resolution (1024x1024)
3. Apply Universal Detailer for detail correction
4. Optional color/contrast adjustments

**Settings for High-Res**:
- `steps`: 30
- `cfg_scale`: 7.5
- `mask_padding`: 64
- `mask_blur`: 6

### Example 5: Batch Processing Setup

**Scenario**: Process multiple character images consistently

**Recommended Settings**:
- `seed`: Fixed value (e.g., 12345)
- `sampler_name`: "euler"
- `scheduler`: "normal"
- `confidence_threshold`: 0.6

**Benefits**:
- Consistent results across batch
- Predictable quality standards
- Efficient processing pipeline

### Example 6: Style-Specific Adjustments

#### Anime/Manga Style
```json
{
  "detection_model": "yolov8n-face",
  "confidence_threshold": 0.4,
  "inpaint_strength": 0.7,
  "cfg_scale": 6.0,
  "mask_padding": 24
}
```

#### Realistic Photography
```json
{
  "detection_model": "yolov8s-face",
  "confidence_threshold": 0.6,
  "inpaint_strength": 0.65,
  "cfg_scale": 7.5,
  "mask_padding": 32
}
```

#### Artistic/Painted Style
```json
{
  "detection_model": "yolov8n-face",
  "confidence_threshold": 0.5,
  "inpaint_strength": 0.8,
  "cfg_scale": 8.0,
  "mask_padding": 40
}
```

## ComfyUI Workflow Examples

### JSON Workflow 1: Basic Portrait Enhancement

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "portrait.jpg"
    }
  },
  "2": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "realisticVisionV51_v51VAE.safetensors"
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "high quality, detailed face, perfect hands, photorealistic",
      "clip": ["2", 1]
    }
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "blurry, deformed, ugly, bad anatomy, extra fingers",
      "clip": ["2", 1]
    }
  },
  "5": {
    "class_type": "UniversalDetailer",
    "inputs": {
      "image": ["1", 0],
      "model": ["2", 0],
      "vae": ["2", 2],
      "positive": ["3", 0],
      "negative": ["4", 0],
      "target_parts": "face,hand",
      "confidence_threshold": 0.5,
      "inpaint_strength": 0.75,
      "steps": 20
    }
  },
  "6": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["5", 0],
      "filename_prefix": "enhanced_portrait"
    }
  }
}
```

### JSON Workflow 2: Anime Character Correction

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "anime_character.png"
    }
  },
  "2": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "anythingV5_PrtRE.safetensors"
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "anime, high quality, detailed face, perfect hands, clean art",
      "clip": ["2", 1]
    }
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "blurry, deformed, extra fingers, bad hands, mutation",
      "clip": ["2", 1]
    }
  },
  "5": {
    "class_type": "UniversalDetailer",
    "inputs": {
      "image": ["1", 0],
      "model": ["2", 0],
      "vae": ["2", 2],
      "positive": ["3", 0],
      "negative": ["4", 0],
      "detection_model": "yolov8n-face",
      "target_parts": "face,hand",
      "confidence_threshold": 0.4,
      "inpaint_strength": 0.7,
      "cfg_scale": 6.0,
      "steps": 25
    }
  },
  "6": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["5", 0]
    }
  }
}
```

## Troubleshooting Examples

### Problem 1: No Detections Found

**Symptoms**: Detection info shows 0 faces/hands detected

**Solutions**:
1. Lower `confidence_threshold` to 0.3-0.4
2. Try different `detection_model`
3. Check if image contains clear faces/hands
4. Verify image resolution (too low may affect detection)

**Example Fix**:
```json
{
  "confidence_threshold": 0.3,
  "detection_model": "yolov8s-face"
}
```

### Problem 2: Over-Processing/Artifacts

**Symptoms**: Results look artificial or over-enhanced

**Solutions**:
1. Reduce `inpaint_strength` to 0.5-0.6
2. Increase `mask_blur` to 6-8
3. Lower `cfg_scale` to 5-6
4. Reduce `steps` to 15-20

**Example Fix**:
```json
{
  "inpaint_strength": 0.55,
  "mask_blur": 7,
  "cfg_scale": 5.5,
  "steps": 18
}
```

### Problem 3: Memory Errors

**Symptoms**: CUDA out of memory or system crash

**Solutions**:
1. Use smaller detection model: "yolov8n-face"
2. Reduce `steps` to 15
3. Process one part at a time
4. Lower image resolution before processing

**Memory-Efficient Settings**:
```json
{
  "detection_model": "yolov8n-face",
  "steps": 15,
  "target_parts": "face"
}
```

### Problem 4: Inconsistent Results

**Symptoms**: Results vary significantly between runs

**Solutions**:
1. Set fixed `seed` value
2. Use consistent `sampler_name` and `scheduler`
3. Standardize all parameters

**Consistency Settings**:
```json
{
  "seed": 42,
  "sampler_name": "euler",
  "scheduler": "normal"
}
```

## Performance Optimization Examples

### Speed-Optimized Settings
**For fast processing with acceptable quality**:
```json
{
  "detection_model": "yolov8n-face",
  "steps": 12,
  "cfg_scale": 6.0,
  "confidence_threshold": 0.6,
  "sampler_name": "euler"
}
```

### Quality-Optimized Settings
**For best results with longer processing time**:
```json
{
  "detection_model": "yolov8s-face",
  "steps": 35,
  "cfg_scale": 8.0,
  "confidence_threshold": 0.7,
  "mask_padding": 48,
  "mask_blur": 6,
  "sampler_name": "dpm_2m"
}
```

### Balanced Settings
**Good compromise between speed and quality**:
```json
{
  "detection_model": "yolov8n-face",
  "steps": 20,
  "cfg_scale": 7.0,
  "confidence_threshold": 0.5,
  "mask_padding": 32,
  "inpaint_strength": 0.75
}
```

## Integration Examples

### With ControlNet
```
Image → ControlNet Preprocessing → Universal Detailer → Final Output
```

### With Upscaling
```
Low-Res → Real-ESRGAN → Universal Detailer → High-Quality Result
```

### With Style Transfer
```
Base Image → Style Transfer → Universal Detailer → Refined Style
```

## Best Practices

1. **Start Conservative**: Begin with default settings and adjust gradually
2. **Test on Single Images**: Verify settings before batch processing
3. **Monitor Memory**: Watch GPU usage to prevent crashes
4. **Save Successful Configs**: Document working parameter sets
5. **Version Control**: Keep track of model and setting combinations

## Common Use Cases Summary

| Use Case | Target Parts | Strength | Steps | Notes |
|----------|--------------|----------|-------|-------|
| Portrait Touch-up | face | 0.6 | 20 | Conservative enhancement |
| Character Art | face,hand | 0.75 | 25 | Balanced correction |
| Hand Fix Only | hand,finger | 0.85 | 30 | Aggressive hand correction |
| Batch Processing | face,hand | 0.7 | 18 | Speed vs quality balance |
| High-Res Detail | face,hand | 0.65 | 35 | Quality-focused processing |