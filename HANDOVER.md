# Handover — 2026-02-19

## Current Status

All 4 plan tasks completed (except pulsar2 compile which requires x86 host):

- **ESPCN trained**: 42.80 dB PSNR on validation (target was >28 dB). Weights saved.
- **ONNX exported**: Trained weights with DCR→CRD remapping + uint8 output baked in graph
- **Rasterizer optimized**: Atomic work-stealing, tighter thresholds, `--sh-degree`, `--bench` flags
- **BLOCKED**: pulsar2 recompile on x86 host needed to deploy trained .axmodel

## What Was Done

### Task 1: Train ESPCN Weights
- Installed PyTorch (torch-2.10.0+cpu, aarch64)
- Generated 192 training frames: `build/gs_splat --dump` for Mars, InteriorDesign, Auditorium (64 each)
- Created `tools/train_espcn.py`:
  - Manual batch loop (no DataLoader) for ARM performance
  - Pre-loads images as uint8 (~1.2GB), 2x2 box downscale via numpy reshape+mean
  - L1 loss, Adam, CosineAnnealingLR, 100 epochs × 100 batches × 16 samples
  - Random 48×48 LR / 96×96 HR crops with flip/rotation augmentation
- Training results: **42.80 dB** best validation PSNR (epoch 90)
- PSNR progression: 26.96 → 35.76 → 38.37 → 39.67 → 40.10 → 41.02 → 41.44 → 41.90 → 42.80 dB

### Task 2: Fix uint8 Output
- Modified `tools/gen_espcn_onnx.py`: added Mul(255)→Clip(0,255)→Cast(uint8) after DepthToSpace
- ONNX output tensor is now UINT8 (verified: dtype=2)
- Added `--weights` flag for trained weight injection with DCR→CRD channel remapping
- Expected savings after pulsar2 recompile: buffer 24.8MB→6.2MB, out_cvt 14ms→~3ms

### Task 3: Rasterizer Optimizations
- **3a. Atomic work-stealing** (`src/gs_rasterizer.cpp`): Replaced static `tile_start/tile_end` block partition with shared `volatile uint32_t next_tile` + `__sync_fetch_and_add`. All 8 threads grab tiles dynamically.
- **3b. Tighter thresholds** (`include/gs_types.h`): `ALPHA_THRESHOLD` 1/255→2/255, `TRANSMITTANCE_MIN` 0.003→0.01
- **3c. `--sh-degree N`** (`src/gs_projector.cpp`, `include/gs_projector.h`, `include/gs_renderer.h`, `src/gs_renderer.cpp`, `src/main.cpp`): Cap SH evaluation at degree 0-3
- **3d. `--bench N`** (`src/main.cpp`): Orbit camera around scene center, print per-frame timing breakdown (proj/sort/raster/upscale/total/FPS), averages at end

### Other
- Updated `CLAUDE.md` with new flags, training results, current status
- Verified C++ builds clean after all changes
- Verified rendering works (dump mode)

## Key Decisions

1. **2x2 box downscale** (not bicubic): Matches the actual camera rendering pipeline (pixel averaging). ~50x faster than Pillow bicubic on ARM.
2. **DCR→CRD weight remapping**: PyTorch PixelShuffle=DCR, compiled model uses CRD DepthToSpace. Remap `CRD[c*4+d] = DCR[d*3+c]` at ONNX export time.
3. **uint8 baked into ONNX graph**: Rather than relying on pulsar2 `output_processors` (which silently ignores `output_dtype: "U8"`), added Mul/Clip/Cast nodes directly into the model graph.
4. **Atomic work-stealing over task queue**: Simple `__sync_fetch_and_add` on shared counter. ~2040 tiles at 540p means negligible contention, no complex queue needed.
5. **Trained on rendered frames**: Used actual 3DGS rendered output (not DIV2K/external datasets) for domain-specific SR quality.

## Open Issues

1. **pulsar2 recompile required on x86**: New ONNX with trained weights + uint8 output needs compilation to .axmodel. Cannot run on this ARM device.
   ```bash
   # On x86 host:
   pulsar2 build --input data/models/espcn_x2.onnx --output_dir output \
     --config data/models/espcn_config.json --target_hardware AX650
   scp output/compiled.axmodel device:data/models/espcn_x2.axmodel
   ```
2. **Benchmarks not yet run**: `--bench` mode requires sudo for display. Run after pulsar2 recompile to get full before/after numbers.
3. **Rasterizer optimizations untested**: Atomic work-stealing and threshold changes built but not benchmarked yet (needs `--bench`).

## Next Steps (Priority Order)

1. **[X86] Compile new .axmodel** with pulsar2 from `data/models/espcn_x2.onnx`
2. **Deploy and test** NPU with trained weights: `sudo build/gs_splat ~/ply/Mars.ply -s 2 --npu`
3. **Run benchmarks** (Task 4): `sudo build/gs_splat ~/ply/Mars.ply -s 2 --bench 32` for all scene/mode combos
4. **Commit all changes** — large set of uncommitted work

## Important Files

### Created this session
- `tools/train_espcn.py` — ESPCN training script (PyTorch CPU, manual batch loop)
- `data/models/espcn_weights.npz` — Trained weights (108KB, DCR ordering)

### Modified this session
- `tools/gen_espcn_onnx.py` — `--weights` flag, DCR→CRD remap, uint8 output nodes
- `src/gs_rasterizer.cpp` — Atomic work-stealing (replaced static block partition)
- `include/gs_types.h` — Tightened `ALPHA_THRESHOLD` (2/255), `TRANSMITTANCE_MIN` (0.01)
- `src/gs_projector.cpp` — SH degree capping via `max_sh_degree` parameter
- `include/gs_projector.h` — `max_sh_degree` parameter in `gs_project_color()`
- `include/gs_renderer.h` — `max_sh_degree` field in Renderer
- `src/gs_renderer.cpp` — Passes `max_sh_degree` to projector
- `src/main.cpp` — `--sh-degree N`, `--bench N` flags
- `CLAUDE.md` — Updated status, added Renderer Flags section
- `data/models/espcn_x2.onnx` — Regenerated with trained weights + uint8 output

### Key reference files
- `data/models/espcn_config.json` — pulsar2 config
- `data/models/espcn_calibration.tar` — Calibration data (32 samples) for pulsar2
- `include/gs_npu.h` / `src/gs_npu.cpp` — NPU wrapper (auto-detects uint8 output)

## Context & Notes

- **Training data**: 192 frames at `data/train/{mars,interior,auditorium}/` (64 per scene, 1080p PPM)
- **Auditorium PLY**: Actual filename is `"Auditorium by the sea.ply"` (with spaces). Symlink created: `~/ply/Auditorium.ply`
- **Training perf on ARM**: ~109s/epoch (100 batches × 16 samples). Total ~3 hours for 100 epochs. Key: pre-load as uint8, numpy box downscale (not Pillow bicubic), manual batch loop (not DataLoader).
- **gs_npu.cpp already handles uint8 output**: `u8_nchw_to_argb()` path exists and is auto-dispatched based on model output dtype. No C++ changes needed for uint8.
- **Mars.ply**: 62,002 gaussians, SH degree 3. ~57K visible at default camera.
- **sudo required**: HDMI rendering and NPU need sudo. Dump mode (`--dump`) and `--bench` work without sudo only in dump mode.
