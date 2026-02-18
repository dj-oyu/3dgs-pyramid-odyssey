# Handover — 2026-02-18

## Current Status

NPU super-resolution upscale pipeline is **fully working**.

- `sudo build/gs_splat ~/ply/Mars.ply -s 2` → 960x540 output @ **6.9 FPS**
- `sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu` → 1920x1080 output @ **5.7 FPS** (NPU +33ms)
- Model: ESPCN-x2 with **bilinear weights** (pipeline verification — not trained SR weights yet)
- Image quality: equivalent to hardware bilinear upscale (VO scaler)

## What Was Done

This session implemented Phase 2: NPU Super-Resolution Pipeline Integration (continued from previous session).

- **NPU pipeline** (`include/gs_npu.h`, `src/gs_npu.cpp`):
  - `gs_npu_init()`: Load .axmodel, AX_ENGINE init, parse model I/O metadata (dtype, layout, shape), allocate CMM buffers
  - `gs_npu_upscale()`: ARGB8888 → NCHW → `AX_ENGINE_RunSync()` → NCHW → ARGB8888
  - NEON-optimized format conversions: `argb_to_u8_nchw`, `argb_to_float_nchw`, `u8_nchw_to_argb`, `float_nchw_to_argb`
  - Auto-detects model I/O dtype (UINT8/FLOAT32) and layout (NCHW/NHWC) from compiled model metadata
- **Renderer integration** (`src/gs_renderer.cpp`):
  - `--npu` flag enables NPU upscale path (opt-in, default off)
  - Allocates `upscale_fb` at display resolution; NPU upscales render_fb → upscale_fb → display
  - Falls back to direct display if NPU upscale fails
- **MAU probe fix** (`src/gs_mau.cpp`):
  - Added probe MatMul at end of `gs_mau_init()` — if hardware unavailable (AX650C), cleanly disables MAU without error spam
- **ESPCN model generator** (`tools/gen_espcn_onnx.py`):
  - `--bilinear` mode: analytical bilinear interpolation weights for pipeline verification
  - `--random` mode: random Kaiming weights for benchmark
  - Generates calibration data (32 structured samples) for pulsar2 quantization
- **Cached CMM for NPU output** (`include/gs_memory.h`, `src/gs_memory.cpp`):
  - `gs_cmm_alloc_cached()` — `AX_SYS_MemAllocCached` for NPU output buffer
  - `gs_cmm_invalidate_cache()` — `AX_SYS_MinvalidateCache` after NPU DMA write
  - Fixed output conversion: **275ms → 14ms** (uncached DRAM was the bottleneck)
- **pulsar2 config** (`data/models/espcn_config.json`):
  - `input_processors` with `src_dtype: "U8"`, `mean: [0,0,0]`, `std: [255,255,255]` for uint8 input
  - `output_processors` with `output_dtype: "U8"` was attempted but pulsar2 silently ignored it — output remains float32

## Key Decisions

1. **Cached CMM for NPU output**: CMM default is uncached (DMA-coherent). Reading 24.8MB float32 from uncached = 275ms. `AX_SYS_MemAllocCached` + `MinvalidateCache` after DMA → 14ms. Critical optimization.
2. **Float32 output accepted**: pulsar2 `output_dtype: "U8"` had no effect. Output remains float32 NCHW [0,1]. CPU converts to uint8. 14ms is acceptable.
3. **Bilinear weights for verification**: Analytical bilinear weights make the CNN perform exact bilinear interpolation. Proves pipeline works end-to-end without PyTorch training.
4. **`--npu` opt-in flag**: Prevents accidental activation with wrong/missing model.
5. **Rejected**: NPU MatMul for rasterizer acceleration (INT8-only, needs float precision).

## Open Issues

1. **Output still float32**: pulsar2 `output_processors.output_dtype: "U8"` silently ignored. If fixed, buffer shrinks 4x (24.8MB → 6.2MB), `out_cvt` could drop 14ms → ~3ms.
2. **No trained SR weights**: Bilinear model doesn't improve quality over VO hardware scaler. Need ESPCN training with image pairs (PyTorch on x86).
3. **Rasterizer is main bottleneck**: 95ms at 540p = 55% of frame time.
4. **Uncommitted changes**: All Phase 2 work is uncommitted. Run `git add` + `git commit` before continuing.

## Next Steps (Priority Order)

1. **Commit current work** — large set of uncommitted changes spanning NPU pipeline, MAU probe, memory, renderer
2. **Train ESPCN weights** (quality): PyTorch on x86 → DIV2K dataset → export ONNX → pulsar2 compile → actual super-resolution quality
3. **Fix pulsar2 uint8 output**: Investigate correct config or modify ONNX to include Mul(255)+Clip+Cast at output (saves ~10ms/frame)
4. **Rasterizer optimization**: 95ms is 55% of frame time — tile culling, early termination, FP16 NEON
5. **Benchmark other scenes**: Auditorium.ply, InteriorDesign.ply with `--npu`

## Important Files

### Modified/created this session
- `include/gs_npu.h` — NPU context struct (CmmBuffer I/O, layout/dtype fields)
- `src/gs_npu.cpp` — Full NPU upscale: NEON conversions, AX_ENGINE inference, cached CMM
- `include/gs_memory.h` — Added `gs_cmm_alloc_cached()`, `gs_cmm_invalidate_cache()`
- `src/gs_memory.cpp` — Cached CMM allocation + cache invalidation
- `src/gs_renderer.cpp` — NPU integration: init, upscale_fb, render loop dispatch
- `include/gs_renderer.h` — `use_npu` parameter in `gs_renderer_init()`
- `include/gs_types.h` — `time_upscale_ms` in RenderStats
- `src/main.cpp` — `--npu` flag, NPU timing in stats display
- `src/gs_mau.cpp` — MAU probe test to disable cleanly on AX650C
- `tools/gen_espcn_onnx.py` — Bilinear/random ESPCN-x2 ONNX generator + calibration data
- `data/models/espcn_config.json` — pulsar2 config with `input_processors` for uint8
- `CLAUDE.md` — Updated NPU strategy and current status

### Key reference files
- `data/models/espcn_x2.axmodel` — Compiled model (bilinear weights, uint8 in, float32 out)
- `data/models/espcn_x2.onnx` — Source ONNX model
- `include/ax_sdk/ax_engine_type.h` — AX_ENGINE type definitions (layout/dtype enums)
- `tools/npu_upscale_bench.cpp` — NPU inference latency benchmark

## Context & Notes

- **AX_SYS_MemAllocCached pitfall**: CMM default is uncached. Any buffer read by CPU after NPU DMA MUST use cached alloc + invalidate, or suffer ~90 MB/s (vs ~2 GB/s cached). This was the root cause of 275ms output conversion.
- **pulsar2 config schema**: `quant.input_configs` only has calibration fields (`tensorName`, `calibrationDataset`, `calibrationFormat`, `calibrationSize`, `calibrationMean`, `calibrationStd`). Runtime I/O types are in top-level `input_processors` / `output_processors`. Field names use snake_case.
- **Model I/O format**: Compiled model: UINT8 input NCHW (1,555,200 bytes), FLOAT32 output NCHW (24,883,200 bytes). `eLayout=UNKNOWN(0)` for output but shape `[1,3,1080,1920]` confirms NCHW.
- **NPU timing breakdown**: in_cvt=0.5ms + infer=17ms + out_cvt=14ms = ~32ms total.
- **Mars.ply**: 62,002 gaussians, SH degree 3. ~57K visible at default camera.
- **sudo required**: HDMI rendering and NPU need sudo. Dump mode (`--dump`) works without.
- **Git**: Previous commits on `master`. Current work is uncommitted.
