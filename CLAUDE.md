# 3D Gaussian Splatting on AXera AX650x NPU

## Project Overview
Real-time 3D Gaussian Splatting renderer for M5Stack AI Pyramid-Pro (AXera AX650x).
Renders 3DGS scenes loaded from PLY files, outputting to HDMI via AX_VO API.

## Hardware Target
- **SoC**: Axera AX8850 (AX650C_CHIP) — 24 TOPS @ INT8
- **CPU**: ARM Cortex-A55 x8 (ARMv8.2-A, NEON/ASIMD, fp16, dotprod) @ 1500MHz
- **RAM**: 2GB general + 6GB CMM (contiguous memory manager)
- **NPU**: 24 TOPS INT8, AX_ENGINE v2.12.0s (VIRTUAL_NPU_DISABLE mode)
- **MAU**: Not available on AX650C variant
- **Display**: HDMI up to 4K@60fps via AX_VO API
- **SDK**: V3.6.4
- **Details**: See `docs/HARDWARE_REPORT.md`

## Build
```bash
make          # Build main application (build/gs_splat)
make tools    # Build test tools (build/vo_test, build/ply_info)
make clean    # Clean build artifacts
```

## Run
```bash
# Display test (solid color on HDMI)
./build/vo_test

# PLY file info
./build/ply_info data/point_cloud.ply

# Full renderer
./build/gs_splat data/point_cloud.ply
```

## Controls (during rendering)
- **WASD**: Move camera forward/left/back/right
- **Arrow keys**: Rotate camera
- **Q/E**: Move up/down
- **+/-**: Adjust movement speed
- **O**: Toggle orbit/fly mode
- **R**: Reset camera
- **ESC**: Quit

## Architecture
- CPU+NEON: Projection, culling, SH evaluation, radix sort, tile rasterization
- AX_VO: HDMI frame display (double-buffered ARGB8888)
- CMM: Contiguous memory for scene data and framebuffers

## SDK Libraries (at /soc/lib/)
- `libax_sys.so` - System init, memory allocation
- `libax_vo.so` - Video output (HDMI)
- `libax_ivps.so` - Image/Video processing (CSC, resize)
- `libax_ive.so` - Intelligent video engine
- `libax_engine.so` - NPU inference engine

## Key Files
- `include/gs_types.h` - Core data structures (SoA Gaussians, tiles)
- `src/gs_projector.cpp` - 3D→2D projection, SH evaluation, Jacobian (most complex file)
- `src/gs_rasterizer.cpp` - NEON tile-based rasterizer (8-thread, atomic work-stealing, front-to-back)
- `src/gs_renderer.cpp` - Rendering pipeline orchestrator
- `src/gs_display.cpp` - AX_VO HDMI output (with /dev/fb0 fallback)
- `include/gs_npu.h` / `src/gs_npu.cpp` - Generic AX_ENGINE wrapper (model-agnostic)
- `tools/gen_espcn_onnx.py` - Generate ESPCN-x2 ONNX with trained weights + uint8 output
- `tools/train_espcn.py` - Train ESPCN-x2 on rendered frames (PyTorch CPU)
- `tools/npu_upscale_bench.cpp` - NPU super-resolution latency benchmark
- `docs/RENDERING_DEBUG.md` - **Rendering quality debug status and analysis**

## NPU Strategy: Super-Resolution Upscaling
The NPU MatMul approach for rasterizer acceleration failed (INT8-only NPU, needs float precision).
Current: use NPU for ESPCN-x2 super-resolution upscaling.
- Render at 960x540 (`-s 2`), rasterize 4x fewer pixels
- NPU 2x upscale via ESPCN CNN → 1920x1080 HDMI output
- Enable with `--npu` flag: `sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu`
- Pipeline: ARGB→uint8 NCHW → NPU inference → output→ARGB (total ~25ms)
- ESPCN trained on 192 rendered frames (3 scenes), **42.80 dB PSNR** on validation set
- ONNX model has uint8 output (Mul(255)→Clip→Cast baked into graph) — eliminates float32 conversion
- Compiled .axmodel deployed: `data/models/espcn_x2.axmodel`
- Output CMM buffer uses `AX_SYS_MemAllocCached` + `MinvalidateCache` to avoid uncached DRAM penalty
- Details: See `docs/NPU_SUPERRES.md`

## Renderer Flags
- `--sh-degree N` — Cap SH evaluation (0=DC only, 3=full). Lower = faster projection, less color detail
- `--bench N` — Benchmark mode: render N frames with orbit camera, print timing table, exit
- `--npu` — Enable NPU super-resolution upscaling (requires sudo, `-s 2`)
- `--dump <dir>` — Dump rendered frames to JPEG (no display, no sudo)
- `-s N` — Resolution scale factor (1=1080p, 2=540p)

## Current Status (2026-02-19)
Rendering pipeline produces correct output. Rasterizer optimized (atomic work-stealing, tighter thresholds).
NPU super-resolution pipeline working; ESPCN trained (42.80dB), compiled .axmodel deployed.
MAU acceleration code implemented but inactive (AX650C lacks MAU hardware).
See `docs/HARDWARE_REPORT.md` for full HW capability analysis.
See `docs/RENDERING_DEBUG.md` for rendering quality debug history.

## Test Scenes
PLY files at `~/ply/`: Mars.ply, Auditorium.ply, InteriorDesign.ply (SH degree 3).

Dump mode (render to JPEG without display, no sudo required):
```bash
build/gs_splat ~/ply/Mars.ply -s 2 --dump data/dump/mars -n 8
```
