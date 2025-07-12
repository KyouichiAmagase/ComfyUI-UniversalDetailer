# GitHub Copilot プロンプト集

## 🎯 効果的なCopilotプロンプト集

### 1. アーキテクチャレビュー

#### 全体レビュー
```
@workspace Please review the architecture of this ComfyUI Universal Detailer:

Analyze these aspects:
1. Class design and separation of concerns
2. Interface consistency across components
3. Error handling patterns
4. Memory management strategies
5. ComfyUI integration compliance

Key files:
- universal_detailer.py (main node)
- detection/yolo_detector.py (YOLO integration)  
- masking/mask_generator.py (mask processing)

Suggest improvements for modularity, performance, and maintainability.
```

#### パフォーマンスレビュー
```
@workspace Analyze performance characteristics of this implementation:

Focus on:
1. Memory usage patterns and potential leaks
2. GPU/CPU utilization efficiency
3. Tensor operations optimization
4. Batch processing capabilities

Suggest specific optimizations for:
- Large image processing (4K+)
- Memory-constrained environments
- Multi-GPU scenarios
- Real-time processing requirements
```

### 2. 実装ガイダンス

#### ComfyUIインペインティング統合
```
Please implement ComfyUI-compatible inpainting in the _inpaint_regions method:

Requirements:
1. VAE encode: image tensor → latent space
2. Mask application: apply mask to latent tensors
3. Noise injection: add controlled noise to masked regions
4. Diffusion sampling: use ComfyUI models with conditioning
5. VAE decode: latent → final image
6. Blending: combine with original using mask

Technical constraints:
- Input: image (B,H,W,C), mask (B,H,W), ComfyUI model objects
- Memory efficient for large images
- Batch processing support
- Device management (CPU/GPU auto-switching)
- Error handling for various input sizes

Please provide complete implementation with proper error handling.
```

#### モデル管理システム
```
Please implement automated YOLO model management in detection/model_loader.py:

Features needed:
1. Auto-download from Ultralytics Hub
   - Check local cache first
   - Download if missing
   - Verify integrity (SHA256)
   - Handle network errors

2. Memory-efficient caching
   - LRU cache with configurable size
   - Lazy loading
   - Background preloading
   - Thread-safe operations

3. Model switching
   - Hot-swap without restart
   - Graceful fallback on errors
   - Performance monitoring

Please create ModelManager class with async/await patterns.
```

### 3. 最適化要求

#### メモリ最適化
```
Please optimize this mask generation code for memory efficiency:

Current issues:
- Large tensor allocations
- Potential memory leaks
- Inefficient tensor operations

Target improvements:
- Reduce peak memory usage by 30%
- Implement tensor recycling
- Use in-place operations where safe
- Add memory monitoring

Focus on masking/mask_generator.py generate_masks method.
```

#### GPU加速最適化
```
Please optimize YOLO detection for GPU acceleration:

Current performance:
- Single image: ~2 seconds
- Batch processing: underutilized

Target improvements:
- Batch inference optimization
- Memory-efficient tensor batching
- CUDA stream usage
- Mixed precision support

Analyze detection/yolo_detector.py detect method.
```

### 4. エラーハンドリング

#### 包括的エラー処理
```
Please review and improve error handling across all components:

Current gaps:
- Inconsistent exception types
- Missing resource cleanup
- Poor error messages for users

Improvements needed:
1. Custom exception hierarchy
2. Context managers for resources
3. Detailed logging with context
4. User-friendly error messages
5. Graceful degradation strategies

Add proper error handling to all major methods.
```

#### リソース管理
```
Please implement proper resource management for GPU memory:

Requirements:
1. Automatic cleanup on errors
2. Memory leak detection
3. Resource pooling for repeated operations
4. Graceful handling of OOM conditions

Add context managers and cleanup logic to:
- Model loading/unloading
- Tensor operations
- CUDA memory management
```

### 5. テスト生成

#### ユニットテスト
```
Please generate comprehensive unit tests for YOLODetector:

Test scenarios:
1. Model loading (success/failure cases)
2. Detection with various image sizes
3. Confidence threshold filtering
4. GPU/CPU switching
5. Batch processing
6. Error conditions (corrupted models, invalid inputs)

Use pytest framework with fixtures for model mocking.
Create tests/test_yolo_detector.py
```

#### 統合テスト
```
Please create integration tests for the complete Universal Detailer pipeline:

Test workflow:
1. Load test image
2. Run detection
3. Generate masks
4. Process inpainting
5. Validate output

Test variations:
- Different image sizes (512x512, 1024x1024, 2048x2048)
- Various detection models
- Different mask parameters
- Batch processing (1, 2, 4 images)

Mock ComfyUI models appropriately.
```

### 6. ドキュメント生成

#### API ドキュメント
```
Please generate comprehensive API documentation for UniversalDetailerNode:

Include:
1. Class overview and purpose
2. Method signatures with type hints
3. Parameter descriptions and valid ranges
4. Usage examples
5. Error conditions and handling
6. Performance characteristics

Format as Google-style docstrings.
```

#### ユーザーガイド
```
Please create user-friendly documentation for ComfyUI integration:

Contents:
1. Installation steps
2. Model setup and configuration
3. Node usage in ComfyUI workflows
4. Parameter tuning guide
5. Troubleshooting common issues
6. Performance optimization tips

Target audience: ComfyUI users with basic technical knowledge.
```

### 7. 品質改善

#### コード品質
```
Please review code quality and suggest improvements:

Check for:
1. PEP 8 compliance
2. Type hint completeness
3. Docstring consistency
4. Variable naming conventions
5. Code duplication
6. Unused imports/variables

Apply black formatting and fix linting issues.
```

#### セキュリティレビュー
```
Please review code for security vulnerabilities:

Focus areas:
1. File path validation
2. Model download security
3. Input sanitization
4. Resource exhaustion protection
5. Temporary file handling

Suggest security best practices for:
- Model file validation
- Network operations
- User input processing
```

## 🚀 使用方法

### VS Code での効果的な使用
1. GitHub Copilot 拡張機能を有効化
2. プロジェクトフォルダを開く
3. `Ctrl+Shift+P` → "GitHub Copilot: Open Chat"
4. 上記プロンプトをコピー&ペーストして実行

### 段階的アプローチ
1. **アーキテクチャレビュー** → 全体的な改善提案
2. **実装ガイダンス** → 具体的機能実装
3. **最適化** → パフォーマンス向上
4. **品質保証** → テストとドキュメント

### 継続的改善
- 各実装後にCopilotレビューを実行
- パフォーマンステストで改善効果を確認
- ユーザーフィードバックを収集して改善