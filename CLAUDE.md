# 3D Gaussian Splatting on AXera AX650x NPU

## Project Overview
Real-time 3D Gaussian Splatting renderer for M5Stack AI Pyramid-Pro (AXera AX650x).
Renders 3DGS scenes loaded from PLY files, outputting to HDMI via AX_VO API.

## Hardware Target
- **SoC**: AXera AX650x (AI Pyramid-Pro)
- **CPU**: ARM Cortex-A55 x4 (ARMv8.2-A, NEON/ASIMD, fp16, dotprod)
- **RAM**: 2GB general + 6GB CMM (contiguous memory manager)
- **Display**: HDMI up to 4K@60fps via AX_VO API
- **SDK**: V3.6.4

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
- `src/gs_rasterizer.cpp` - NEON tile-based rasterizer (4-thread, front-to-back)
- `src/gs_renderer.cpp` - Rendering pipeline orchestrator
- `src/gs_display.cpp` - AX_VO HDMI output (with /dev/fb0 fallback)
- `docs/RENDERING_DEBUG.md` - **Rendering quality debug status and analysis**

## Current Status (2026-02-17)
Rendering pipeline runs end-to-end but output images are visually broken.
See `docs/RENDERING_DEBUG.md` for detailed analysis and next steps.

## Test Scenes
PLY files at `~/ply/`: Mars.ply, Auditorium.ply, InteriorDesign.ply (SH degree 3).

Dump mode (render to JPEG without display):
```bash
sudo build/gs_splat ~/ply/Mars.ply -s 2 --dump /tmp/dump_mars -n 8
```
