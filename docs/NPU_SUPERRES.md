# NPU Super-Resolution Pipeline

## 概要

CPUラスタライズのボトルネックを軽減するため、NPU（24 TOPS INT8）でESPCN-x2超解像アップスケーリングを行う。

```
[通常パイプライン]
  CPU rasterize 1920x1080 (重い) → HDMI出力

[NPU超解像パイプライン]
  CPU rasterize 960x540 (1/4ピクセル) → NPU ESPCN 2x upscale → 1920x1080 HDMI出力
```

ラスタライズのピクセル数が1/4になり、NPU推論コスト(~17ms)を差し引いても高速化が見込める。

## モデル: ESPCN-x2

Efficient Sub-Pixel Convolutional Neural Network。軽量な超解像CNN。

```
入力: [1, 3, 540, 960] uint8
  → Conv(3→64, 5x5) → ReLU
  → Conv(64→32, 3x3) → ReLU
  → Conv(32→12, 3x3) → DepthToSpace(2)   ← Sub-Pixel Shuffle
  → Mul(255) → Clip(0,255) → Cast(uint8)  ← uint8出力（グラフに焼き込み）
出力: [1, 3, 1080, 1920] uint8
```

パラメータ数: 26,796

## 学習

### 学習データ生成

3DGSレンダリング出力を学習データとして使用（外部データセットではなくドメイン固有データ）。

```bash
# 各シーン64フレーム、1080pでダンプ（sudo不要）
build/gs_splat ~/ply/Mars.ply -s 1 --dump data/train/mars -n 64
build/gs_splat ~/ply/InteriorDesign.ply -s 1 --dump data/train/interior -n 64
build/gs_splat ~/ply/Auditorium.ply -s 1 --dump data/train/auditorium -n 64
```

- 合計192フレーム（train 173 / val 19）
- 1920x1080 PPM形式
- HR（正解）= 1080pレンダリング、LR（入力）= 2x2 box downscale → 540p

### 学習コマンド

```bash
# PyTorch CPU (torch-2.10.0+cpu, aarch64)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 学習実行（ARM上で約3時間）
python3 tools/train_espcn.py \
  --data data/train \
  --epochs 100 \
  --output data/models/espcn_weights.npz \
  --max-batches 100
```

### 学習設定

| パラメータ | 値 |
|-----------|-----|
| Loss | L1 (MAE) |
| Optimizer | Adam (lr=0.001) |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Batch size | 16 |
| Batches/epoch | 100 |
| Crop size | 48x48 LR / 96x96 HR |
| Augmentation | Random flip + 90° rotation |
| Downscale | 2x2 box (numpy reshape+mean) |

### 学習結果

```
Epoch   1/100 | PSNR=26.96dB | L1=0.0585
Epoch  10/100 | PSNR=35.76dB | L1=0.0109
Epoch  20/100 | PSNR=38.37dB | L1=0.0092
Epoch  40/100 | PSNR=39.67dB | L1=0.0064
Epoch  50/100 | PSNR=40.10dB | L1=0.0067
Epoch  60/100 | PSNR=41.02dB | L1=0.0048
Epoch  70/100 | PSNR=41.44dB | L1=0.0038
Epoch  80/100 | PSNR=41.90dB | L1=0.0035
Epoch  90/100 | PSNR=42.80dB | L1=0.0035  ← Best
Epoch 100/100 | PSNR=42.72dB | L1=0.0033
```

**Best PSNR: 42.80 dB**（epoch 90）— 肉眼ではほぼ区別不能なレベル。

PSNR参考値:
- 20 dB: 明らかに劣化が見える
- 30 dB: そこそこ良い
- 40 dB: ほぼ見分けがつかない
- 42.80 dB: ほぼ完全一致

### 学習パフォーマンス（ARM Cortex-A55）

- ~109秒/epoch（100 batches × 16 samples）
- 合計約3時間（100 epochs）
- メモリ: ~800MB（画像をuint8でプリロード）
- DataLoaderは使わずmanual batch loop（ARM上でのPyTorch DataLoader shuffle性能問題を回避）

## ONNX エクスポート

学習済み重みをONNXに組み込み、uint8出力ノードを付加:

```bash
python3 tools/gen_espcn_onnx.py --weights data/models/espcn_weights.npz
```

出力: `data/models/espcn_x2.onnx` (105KB)

### DCR→CRD 重み変換

PyTorchのPixelShuffleはDCR (Depth-Column-Row) モード、pulsar2のDepthToSpaceはCRD (Column-Row-Depth) モード。
エクスポート時に最終Conv層の重みをリマッピング:

```
CRD_channel[c * scale² + d] = DCR_channel[d * channels + c]
# scale=2, channels=3 → 12チャンネルの並び替え
```

## pulsar2 コンパイル（x86ホスト）

```bash
# 必要ファイルをx86ホストへ転送
scp device:github/3dgs-pyramid-odyssey/data/models/espcn_x2.onnx .
scp device:github/3dgs-pyramid-odyssey/data/models/espcn_calibration.tar .
scp device:github/3dgs-pyramid-odyssey/data/models/espcn_config.json .

# コンパイル
pulsar2 build \
  --input espcn_x2.onnx \
  --output_dir output \
  --config espcn_config.json \
  --target_hardware AX650

# デプロイ
scp output/espcn_x2.axmodel device:github/3dgs-pyramid-odyssey/data/models/espcn_x2.axmodel
```

### espcn_config.json の内容

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [{
      "tensorName": "input",
      "calibrationDataset": "espcn_calibration.tar",
      "calibrationFormat": "Numpy",
      "calibrationSize": 32
    }],
    "calibration_method": "MinMax"
  },
  "input_processors": [{
    "tensor_name": "DEFAULT",
    "src_dtype": "U8",
    "src_layout": "NCHW",
    "tensor_layout": "NCHW",
    "mean": [0, 0, 0],
    "std": [255, 255, 255]
  }],
  "compiler": {
    "check": 0
  }
}
```

- `input_processors`: uint8入力 → /255正規化をNPUが自動実行
- `quant`: MinMaxキャリブレーション（32サンプル）
- uint8出力変換はONNXグラフ内に焼き込み済み（`output_processors`不要）

## 実行

```bash
# NPU超解像付きで実行（sudo必要）
sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu

# ベンチマーク（32フレーム、タイミング表出力）
sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu --bench 32
```

## ベンチマーク結果

シーン: Mars.ply (62,002 Gaussians, SH degree 3), 32フレーム軌道カメラ平均。

### 構成比較

| 構成 | 解像度 | Proj | Sort | Raster | NPU | Total | FPS |
|------|--------|------|------|--------|-----|-------|-----|
| 1080p ベースライン | 1920x1080 | 30.0ms | 10.7ms | 82.2ms | — | 142.4ms | **7.0** |
| 540p (NPUなし) | 960x540 | 29.9ms | 11.0ms | 27.8ms | — | 77.4ms | **12.9** |
| 540p + NPU | 960x540→1080p | 30.0ms | 10.8ms | 28.0ms | 25.4ms | 100.5ms | **10.0** |
| 540p + NPU + SH0 | 960x540→1080p | 25.0ms | 10.9ms | 27.6ms | 25.4ms | 95.2ms | **10.5** |

### 分析

**1080p → 540p+NPU で 43%高速化（7.0 → 10.0 FPS）、1080p出力を維持**

- ラスタライズ: 82.2ms → 28.0ms（**66%削減**、解像度1/4の効果）
- NPUオーバーヘッド: +25.4ms（in_cvt + 推論 + out_cvt）
- 純粋な高速化: 82.2ms → 28.0ms + 25.4ms = 53.4ms（**35%削減**）
- Projection/Sort: 解像度に依存しない（~30ms/~11ms で一定）

**SH degree 0 で追加5%高速化**
- Projection: 30.0ms → 25.0ms（**-5ms**、SH評価省略）
- 色精度は低下（DC成分のみ、view-dependent color なし）

**NPUなし540p が最速（12.9 FPS）だが出力は540p止まり**
- 1080pディスプレイ出力が必要な場合、NPU超解像が最適解

### ベンチコマンド

```bash
# 各構成のベンチマーク（32フレーム軌道カメラ）
sudo build/gs_splat ~/ply/Mars.ply -s 1 --bench 32              # 1080p baseline
sudo build/gs_splat ~/ply/Mars.ply -s 2 --bench 32              # 540p (no NPU)
sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu --bench 32        # 540p + NPU
sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu --sh-degree 0 --bench 32  # 540p + NPU + SH0
```

### フレームタイム内訳（540p + NPU）

```
Projection  30.0ms  ████████████████████████████░░░░░░░  30%
Sort        10.8ms  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░  11%
Rasterize   28.0ms  ███████████████████████████░░░░░░░░  28%
NPU upscale 25.4ms  ████████████████████████░░░░░░░░░░░  25%
Other        6.3ms  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6%
Total      100.5ms                                        10.0 FPS
```

## NPU推論パイプライン詳細

```
ARGB8888 framebuffer (960x540)
  → argb_to_u8_nchw()     [NEON, ~0.5ms]
  → AX_ENGINE_RunSync()   [NPU, ~17ms]
  → u8_nchw_to_argb()     [NEON, ~3ms]  ← uint8出力で高速化（float32時は14ms）
  → cache invalidate      [~5ms]
  → ARGB8888 framebuffer (1920x1080)
  → HDMI display
  合計: ~25.4ms（ベンチマーク実測値）
```

出力CMM bufferは`AX_SYS_MemAllocCached` + `MinvalidateCache`使用（uncached DRAMでは275msかかる）。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `tools/train_espcn.py` | 学習スクリプト（PyTorch CPU） |
| `tools/gen_espcn_onnx.py` | ONNXエクスポート（--weights で学習済み重み注入） |
| `data/models/espcn_weights.npz` | 学習済み重み (108KB) |
| `data/models/espcn_x2.onnx` | ONNX モデル (105KB, uint8出力) |
| `data/models/espcn_config.json` | pulsar2 設定 |
| `data/models/espcn_calibration.tar` | キャリブレーションデータ (190MB) |
| `data/models/espcn_x2.axmodel` | コンパイル済みモデル (337KB) |
| `data/train/` | 学習データ (192フレーム) |
| `src/gs_npu.cpp` | NPU推論ラッパー（dtype自動検出） |
| `include/gs_npu.h` | NPUコンテキスト定義 |
| `src/gs_renderer.cpp` | レンダラー統合（--npu フラグ） |
