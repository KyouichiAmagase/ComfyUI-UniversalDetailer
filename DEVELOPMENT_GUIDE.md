# ComfyUI Universal Detailer - 開発ガイド

## 現在の実装状況

### ✅ 完了済み（優先度1）
- **Face detection**: `detection/yolo_detector.py` - YOLO統合完了
- **Mask generation**: `masking/mask_generator.py` - BBox→マスク変換完了
- **Basic inpainting framework**: `universal_detailer.py` - メイン処理フロー完了

### 🚧 次の実装対象（優先度2）

#### 1. ComfyUIインペインティング統合
**ファイル**: `universal_detailer.py`の`_inpaint_regions()`メソッド
**実装内容**:
```python
def _inpaint_regions(self, image, masks, model, vae, positive, negative, **kwargs):
    # 1. VAEエンコード: image → latent
    # 2. マスク適用: latent space でのノイズ注入
    # 3. 拡散モデル実行: positive/negative conditioning
    # 4. VAEデコード: latent → image
    # 5. 元画像との合成
```

#### 2. モデル管理機能強化
**ファイル**: `detection/model_loader.py`（新規作成）
**実装内容**:
- 自動モデルダウンロード機能
- モデルキャッシュ管理
- 複数モデル同時実行対応

## GitHub Copilot活用戦略

### 🤖 Copilotレビュー観点
1. **コード品質**
   - 型ヒントの一貫性
   - エラーハンドリングの網羅性
   - メモリ効率の最適化

2. **ComfyUI統合**
   - ノードインターフェースの最適化
   - バッチ処理のパフォーマンス
   - メモリリーク対策

3. **YOLO統合**
   - ultralytics最新API準拠
   - GPU/CPU切り替え最適化
   - モデル読み込み効率化

### 📋 レビュー項目リスト

#### A. アーキテクチャレビュー
```
- [ ] クラス設計の単一責任原則遵守
- [ ] インターフェース分離の適切性
- [ ] 依存関係注入の実装
- [ ] 設定管理の集約化
```

#### B. パフォーマンスレビュー
```
- [ ] テンソル操作の最適化
- [ ] メモリ使用量の削減
- [ ] GPU利用効率の改善
- [ ] バッチ処理の並列化
```

#### C. エラーハンドリングレビュー
```
- [ ] 例外階層の適切な設計
- [ ] リソースリークの防止
- [ ] ログ出力の最適化
- [ ] ユーザーフレンドリーなエラーメッセージ
```

## 実装優先順位

### Phase 2A: Core Integration (即座に実装)
1. `_inpaint_regions()`の完全実装
2. ComfyUIサンプラー統合
3. VAE エンコード/デコード処理

### Phase 2B: Advanced Features (並行実装)
1. モデル自動ダウンロード機能
2. 高度なマスク処理（セグメンテーション）
3. バッチ処理最適化

### Phase 2C: Production Ready (最終段階)
1. 包括的テストスイート
2. パフォーマンスベンチマーク
3. ドキュメント完成

## GitHub作業の進め方

### 🔄 開発フロー
1. **ブランチ戦略**: feature/priority-2-implementation
2. **コミット単位**: 機能単位での細かいコミット
3. **PR作成**: Copilotレビュー用の詳細説明
4. **レビュー**: Copilot + 人的レビューの組み合わせ

### 📝 Copilotプロンプト例

#### インペインティング実装時:
```
# GitHub Copilot, please review this ComfyUI inpainting integration:
# Focus on: VAE handling, memory efficiency, batch processing
# Suggest improvements for: tensor operations, error handling, performance
```

#### モデル管理実装時:
```
# GitHub Copilot, please review this YOLO model management:
# Focus on: download safety, cache efficiency, concurrent access
# Suggest improvements for: file handling, network errors, storage optimization
```

## 重要な実装ポイント

### 🎯 ComfyUI特有の考慮事項
1. **テンソル形式**: `(batch, height, width, channels)`
2. **デバイス管理**: CPU/GPU自動切り替え
3. **メモリ管理**: 大画像でのOOM対策
4. **進捗表示**: ComfyUIプログレスバー統合

### 🔧 YOLO統合の最適化
1. **モデル切り替え**: 動的なモデルロード
2. **前処理**: 効率的な画像変換
3. **後処理**: 信頼度フィルタリング
4. **キャッシュ**: 検出結果の再利用

## 次のセッション準備

### 📁 作業ファイル優先順位
1. `universal_detailer.py` - インペインティング統合
2. `detection/model_loader.py` - モデル管理（新規）
3. `utils/comfyui_integration.py` - ComfyUI専用ユーティリティ（新規）
4. `tests/test_integration.py` - 統合テスト（新規）

### 🚀 即座に着手可能なタスク
- VAEエンコード/デコード処理の実装
- ComfyUIサンプラーとの統合
- メモリ効率化の改善
- エラーハンドリングの強化

---
**注記**: このガイドをベースにGitHub Copilotと協力して効率的な開発を進めてください。各実装段階でCopilotのコードレビューを活用し、品質とパフォーマンスの向上を図ります。