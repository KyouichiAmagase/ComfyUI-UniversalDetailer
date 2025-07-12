# Universal Detailer Workflow Examples

This directory contains example ComfyUI workflows demonstrating different use cases and configurations for Universal Detailer.

## 📋 Available Workflows

### 1. 🎯 **Basic Face Enhancement** (`basic_face_enhancement.json`)
**Best for**: Portrait photos, headshots, profile pictures  
**Processing time**: 10-15 seconds  
**VRAM requirement**: 6GB

**Features**:
- Simple face detection and enhancement
- Optimized for speed and quality balance
- Beginner-friendly configuration
- Single-part processing (faces only)

**Use cases**:
- Social media profile pictures
- Portrait photography enhancement
- Quick face touch-ups
- Batch processing of similar portraits

---

### 2. 🏃 **Multi-Part Enhancement** (`multi_part_enhancement.json`)
**Best for**: Full-body photos, group photos, complex scenes  
**Processing time**: 20-30 seconds  
**VRAM requirement**: 8GB

**Features**:
- Simultaneous face and hand detection/enhancement
- Advanced masking and blending
- ControlNet integration (optional)
- Comprehensive mask outputs for debugging

**Use cases**:
- Full-body portrait enhancement
- Fashion/modeling photography
- Group photo improvements
- Complex scene processing

---

## 🚀 Quick Setup Guide

### 1. **Load Workflow in ComfyUI**
```bash
# Copy workflow file to ComfyUI workflows directory
cp workflow_examples/basic_face_enhancement.json ComfyUI/workflows/

# Or drag and drop the .json file into ComfyUI interface
```

### 2. **Required Models**
The workflows will automatically download required YOLO models:
- **yolov8n-face.pt** (6MB) - Fast face detection
- **yolov8s-face.pt** (22MB) - High-accuracy face detection  
- **hand_yolov8n.pt** (6MB) - Hand detection (for multi-part workflows)

### 3. **Configure Your Models**
Update these nodes with your models:
- **CheckpointLoaderSimple**: Your diffusion model (.safetensors)
- **CLIPTextEncode**: Adjust prompts for your style
- **VAE**: Use your preferred VAE model

---

## ⚙️ Configuration Guide

### 🎯 **Detection Settings**

| Parameter | Basic Face | Multi-Part | Description |
|-----------|------------|------------|-------------|
| `detection_model` | yolov8n-face | yolov8s-face | Model accuracy vs speed |
| `target_parts` | face | face,hand | Parts to detect |
| `confidence_threshold` | 0.7 | 0.6 | Detection sensitivity |

### 🎨 **Enhancement Settings**

| Parameter | Conservative | Balanced | Aggressive | Description |
|-----------|--------------|----------|------------|-------------|
| `inpaint_strength` | 0.6 | 0.8 | 0.95 | Enhancement intensity |
| `steps` | 15 | 20 | 30 | Quality vs speed |
| `cfg_scale` | 6.0 | 7.5 | 9.0 | Prompt adherence |

### 🔧 **Masking Settings**

| Parameter | Tight | Balanced | Generous | Description |
|-----------|-------|----------|----------|-------------|
| `mask_padding` | 16 | 32 | 48 | Mask expansion |
| `mask_blur` | 2 | 4 | 8 | Edge blending |

---

## 📊 Performance Optimization

### 🚀 **Speed Optimization**
```json
{
  "detection_model": "yolov8n-face",
  "steps": 15,
  "cfg_scale": 6.0,
  "sampler_name": "euler"
}
```

### 🎨 **Quality Optimization**  
```json
{
  "detection_model": "yolov8s-face",
  "steps": 25,
  "cfg_scale": 8.0,
  "sampler_name": "dpm_solver",
  "scheduler": "karras"
}
```

### 💾 **Memory Optimization**
```json
{
  "Enable in universal_detailer.py": "memory_manager optimization",
  "Batch size": "Automatically adjusted based on available VRAM",
  "Model caching": "LRU cache for efficient model management"
}
```

---

## 🔍 **Troubleshooting**

### ❌ **Common Issues**

| Problem | Solution | Settings |
|---------|----------|----------|
| Faces not detected | Lower confidence threshold | `confidence_threshold: 0.5-0.6` |
| Hands not detected | Use multi-part model | `detection_model: "yolov8s-face"` |
| Out of memory | Reduce image size | Resize input to 768x768 |
| Processing too slow | Use faster model | `detection_model: "yolov8n-face"` |
| Artifacts at edges | Increase blending | `mask_blur: 6-8, mask_padding: 48` |

### 🔧 **Advanced Debugging**

1. **Check Detection Info**:
   ```json
   "outputs": {
     "detection_info": ["ShowText", 0]
   }
   ```

2. **Visualize Masks**:
   ```json
   "outputs": {
     "detection_masks": ["SaveImage", 0],
     "face_masks": ["SaveImage", 0]
   }
   ```

3. **Monitor Performance**:
   - Check processing times in detection_info
   - Monitor GPU memory usage
   - Verify detection counts and confidence scores

---

## 🎬 **Example Use Cases**

### 📸 **Portrait Photography**
```json
{
  "workflow": "basic_face_enhancement.json",
  "target_parts": "face",
  "inpaint_strength": 0.7,
  "focus": "Natural skin enhancement, eye clarity"
}
```

### 👥 **Group Photos**
```json
{
  "workflow": "multi_part_enhancement.json", 
  "target_parts": "face",
  "confidence_threshold": 0.6,
  "focus": "Multiple face enhancement, consistent quality"
}
```

### 👗 **Fashion Photography**
```json
{
  "workflow": "multi_part_enhancement.json",
  "target_parts": "face,hand",
  "inpaint_strength": 0.8,
  "focus": "Full model enhancement, hand and face quality"
}
```

### 🎨 **Artistic Enhancement**
```json
{
  "workflow": "multi_part_enhancement.json",
  "cfg_scale": 9.0,
  "steps": 30,
  "focus": "Creative enhancement, artistic interpretation"
}
```

---

## 📚 **Additional Resources**

### 🔗 **Related Documentation**
- [API_REFERENCE.md](../API_REFERENCE.md) - Complete parameter reference
- [EXAMPLES.md](../EXAMPLES.md) - Code examples and integration
- [DEVELOPMENT_GUIDE.md](../DEVELOPMENT_GUIDE.md) - Custom workflow development

### 🤝 **Community Workflows**
Share your custom workflows:
1. Create issue with `workflow-example` label
2. Include .json file and description
3. Specify use case and optimization notes

### 🆘 **Support**
- **Issues**: [GitHub Issues](https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/issues)
- **Questions**: Create issue with `question` label
- **Discussions**: ComfyUI community forums

---

**🎯 Pro Tip**: Start with `basic_face_enhancement.json` to familiarize yourself with the node, then progress to multi-part workflows for advanced use cases!