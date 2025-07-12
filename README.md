# ComfyUI Universal Detailer

**Enhanced version of FaceDetailer supporting face, hand, and finger detection/correction**

![Development Status](https://img.shields.io/badge/status-in%20development-yellow)
![AI Developed](https://img.shields.io/badge/development-AI%20powered-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

## ⚠️ IMPORTANT DISCLAIMERS

### AI Development Notice
**This project is developed entirely by AI (Claude by Anthropic).** 

- **NO TECHNICAL SUPPORT**: We cannot provide any technical support, troubleshooting, or assistance.
- **NO WARRANTIES**: This software is provided "as is" without any warranties of any kind.
- **USE AT YOUR OWN RISK**: Any use of this software is entirely at your own risk.

### Liability and Responsibility

**🚨 CRITICAL WARNING 🚨**

- **SELF-RESPONSIBILITY ONLY**: You are solely responsible for any consequences of using this software.
- **NO LIABILITY**: We accept no responsibility for any damages, data loss, system crashes, or other issues.
- **NO GUARANTEES**: We make no guarantees about functionality, stability, or compatibility.
- **USE IN PRODUCTION**: Absolutely not recommended for production environments.

### Before Using This Software

✅ **You acknowledge that:**
- You understand this is experimental AI-generated code
- You will test thoroughly in isolated environments
- You will not hold the creators liable for any issues
- You have adequate backups of your ComfyUI installation
- You are comfortable with potential system instability

## Overview

Universal Detailer is an advanced ComfyUI custom node that extends FaceDetailer functionality to support detection and correction of:

- 👤 **Faces**: High-quality face detection and enhancement
- ✋ **Hands**: Automatic hand detection and correction
- 👆 **Fingers**: Detailed finger detection and improvement
- 🎯 **Multiple Parts**: Simultaneous processing of different body parts

## Features

### Core Functionality
- **Multi-part Detection**: YOLO-based detection for faces, hands, and fingers
- **Automatic Masking**: Precise mask generation for detected areas
- **Selective Processing**: Individual or batch processing of detected parts
- **Quality Enhancement**: High-quality inpainting for detected areas
- **Flexible Configuration**: Extensive parameter customization

### Technical Features
- **Memory Efficient**: Optimized for various GPU configurations
- **Batch Processing**: Efficient handling of multiple detections
- **Error Handling**: Robust error recovery and reporting
- **Model Agnostic**: Works with various ComfyUI models

## Project Status

🚧 **Currently in Development** 🚧

This project is being actively developed by AI. Check the [SPECIFICATIONS.md](SPECIFICATIONS.md) file for detailed development requirements.

## Installation

**⚠️ NOT YET AVAILABLE ⚠️**

This software is still under development. Installation instructions will be provided once the initial version is completed.

### Planned Installation Methods

1. **ComfyUI Manager** (Recommended)
   ```
   Search for "Universal Detailer" in ComfyUI Manager
   ```

2. **Manual Installation**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer.git
   cd ComfyUI-UniversalDetailer
   pip install -r requirements.txt
   ```

## Requirements

### System Requirements
- **ComfyUI**: Latest version
- **Python**: 3.8+
- **GPU**: 6GB+ VRAM recommended
- **RAM**: 8GB+ system memory

### Dependencies
- `ultralytics>=8.0.0` (YOLO models)
- `opencv-python>=4.5.0`
- `torch>=1.12.0`
- `torchvision>=0.13.0`

## Documentation

- 📋 [**SPECIFICATIONS.md**](SPECIFICATIONS.md) - Detailed development specifications
- 🔧 [**API_REFERENCE.md**](API_REFERENCE.md) - Node interface documentation
- 📚 [**EXAMPLES.md**](EXAMPLES.md) - Usage examples and workflows

## Development

### For AI Developers

This project is specifically designed for AI development. All development tasks should reference the [SPECIFICATIONS.md](SPECIFICATIONS.md) file for complete requirements.

### Development Status

- [ ] Core detection system
- [ ] Face detection integration
- [ ] Hand detection integration
- [ ] Mask generation system
- [ ] Inpainting pipeline
- [ ] UI components
- [ ] Error handling
- [ ] Documentation
- [ ] Testing suite

## Contributing

**Note**: This project is developed by AI. Human contributions are welcome but please understand:

- Review the specifications thoroughly
- Test extensively before submitting PRs
- Document any changes clearly
- Respect the AI-first development approach

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Support

**❌ NO TECHNICAL SUPPORT PROVIDED ❌**

As stated in the disclaimers, we cannot provide technical support. For community discussion:

- 💬 [GitHub Discussions](https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/discussions)
- 🐛 [Issue Tracker](https://github.com/KyouichiAmagase/ComfyUI-UniversalDetailer/issues) (for documentation/specification issues only)

## Acknowledgments

- **FaceDetailer**: Inspiration and base architecture
- **ADetailer**: Multi-part detection concepts
- **ComfyUI Community**: Node development standards
- **Anthropic**: AI development platform (Claude)

---

**Remember: Use at your own risk. No support. No warranties. AI-generated code.**