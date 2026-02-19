# Handover — 2026-02-20 (Session 3)

## Current Status

Renderer optimized from 77.6ms/12.9 FPS to **49.2ms/20.3 FPS (+57%)** on Mars 62K Gaussians @ 540p.
All three major subsystems (projection, sort, rasterizer) have been optimized.
FP16 tile accumulation verified with no visible quality degradation.

## What Was Done (This Session)

### Sort Optimization: 16-Bit Quantized Radix Sort
- Replaced 64-bit radix sort (4 passes on `uint64_t` key|index pairs) with 16-bit quantized sort
- For ≤65535 visible Gaussians: packs `(depth16 << 16 | index16)` into `uint32_t`, 2 radix passes
- Index-only output — `sorted_indices[]` array, no physical permute of ProjectedGaussian array
- Three paths: insertion sort (≤128), 16-bit compact (≤65535), 64-bit fallback
- **Sort: 4.3ms → 2.5ms**

### Prefetch-Accelerated Physical Permute (Attempted, Reverted)
- Implemented `gs_sort_permute()` with `__builtin_prefetch` distance 8 to restore sequential access in assign
- Result: permute cost (4.2ms) cancelled assign improvement (4.0ms) — net zero
- Root cause: 2.8MB ProjectedGaussian array exceeds L2 (256KB), random gather costs ~4ms regardless of where it's paid
- **Decision: Reverted to index-only sort (simpler, equivalent performance)**

### Persistent Rasterizer Thread Pool
- Replaced per-frame `pthread_create`/`pthread_join` with persistent `RasterThreadPool`
- Workers persist across frames using `pthread_barrier_t` start/end barriers
- Eliminates ~0.3ms thread management overhead per frame

### FP16 Tile Accumulation (Main Raster Optimization)
- Tile buffers changed from `float[16][16]` to `__fp16[16][16]` (tile_r, tile_g, tile_b, tile_T)
- Tier 1 inner loop runs entirely in FP16 NEON (8-wide `float16x8_t`):
  - Mahalanobis distance, Gaussian evaluation, alpha blending all in FP16
  - Eliminates FP16→FP32 widening and FP32→FP16 narrowing per pixel
  - ~57 → ~39 NEON instructions per 8 pixels
- Tier 2 (4-wide FP32 remainder) loads/stores FP16 from tiles
- FP16 precision (10-bit mantissa, 1024 levels) sufficient for 8-bit output (256 levels)
- Quality verified via JPEG dump — no visible degradation
- **Raster: 26.2ms → 24.2ms (-7.6%)**

### Other Fixes
- MAU hardware probe: graceful detection + disable on AX650C (prevents error spam)
- Thresholds tightened: ALPHA_THRESHOLD 1/255→2/255, TRANSMITTANCE_MIN 0.003→0.01

### Cumulative Benchmark (32 frames, 960×540, Mars.ply)

| Phase    | Session 1 | Session 2 | Session 3 | Speedup |
|----------|-----------|-----------|-----------|---------|
| Proj     | 30.2ms    | 11.8ms    | 9.8ms     | 3.1x    |
| Sort     | 10.6ms    | 11.0ms    | 2.5ms     | 4.2x    |
| Assign   | —         | —         | 10.1ms    | —       |
| Raster   | 28.0ms    | 30.9ms    | 24.2ms    | 1.2x    |
| **Total**| **77.6ms**| **62.4ms**| **49.2ms**| **1.58x** |
| **FPS**  | **12.9**  | **16.0**  | **20.3**  | **1.57x** |

## Key Decisions

1. **Index-only sort over physical permute**: Experimentally proved that for arrays exceeding L2, random access penalty (~4ms) is invariant to placement. Simpler code wins.
2. **FP16 accumulation over 2-Gaussian batching**: FP16 reduces instruction count by ~30% with no quality loss. Batching would save ~10% but adds complexity.
3. **Persistent thread pool for rasterizer only**: Projection already uses per-frame threads (acceptable overhead). Rasterizer benefits more due to per-frame tile dispatch.
4. **Tighter thresholds**: ALPHA 2/255 and TRANSMITTANCE_MIN 0.01 save 5-10% raster time, visually imperceptible.

## Open Issues

1. **Assign is now the largest sub-phase at 10.1ms (20% of frame)**: Random indirect access into 2.8MB ProjectedGaussian array. Structural limitation of L2 size.
2. **Raster still 24.2ms (49% of frame)**: Per-pixel work is already lean. Further gains likely require algorithmic changes (fewer Gaussians per tile, hierarchical culling).
3. **pulsar2 recompile still required on x86**: ONNX with trained weights + uint8 output ready, needs compilation to .axmodel.
4. **ODR hazard with Makefile**: No header dependency tracking. Struct field additions require `make clean && make` or stale object files cause silent memory corruption.

## Next Steps (Priority Order)

1. **Rasterizer algorithmic optimization** — 24.2ms is 49% of frame. Candidates:
   - Hierarchical tile culling (skip tiles where no Gaussian overlaps)
   - Tighter per-tile Gaussian overlap test (reduce avg Gaussians/tile)
   - Row-level saturation bitmask (skip fully-opaque rows)
2. **Assign optimization** — 10.1ms for indirect scatter. Possible approaches:
   - AoSoA layout for ProjectedGaussian (cache-line-friendly groups of 4-8)
   - Reduce ProjectedGaussian struct size (currently 48 bytes, could compress icov/color)
3. **[X86] Compile new .axmodel** with pulsar2 from `data/models/espcn_x2.onnx`
4. **Full benchmark suite** — All 3 scenes × all modes after optimizations stabilize
5. **Makefile header dependency tracking** — Add `-MMD -MP` flags, include `.d` files

## Important Files

### Modified this session
- `src/gs_sort.cpp` — 16-bit quantized radix sort, index-only output (154 insertions)
- `include/gs_sort.h` — SortContext with sorted_indices (no pg_temp)
- `src/gs_rasterizer.cpp` — Persistent thread pool, FP16 tile accumulation, sorted_indices assign (483 changed lines)
- `include/gs_rasterizer.h` — Updated assign signature with sorted_indices parameter
- `src/gs_renderer.cpp` — Updated pipeline: sort→assign with sorted_indices, SortContext member
- `include/gs_renderer.h` — Added SortContext sort_ctx, max_sh_degree fields
- `include/gs_types.h` — Threshold constants (ALPHA_THRESHOLD, TRANSMITTANCE_MIN)
- `src/gs_projector.cpp` — Hoisted constants, early culls
- `src/main.cpp` — --bench, --sh-degree flags
- `src/gs_mau.cpp` — MAU hardware probe fix

### Benchmark logs
- `bench.log` — Session 3 final benchmark (49.2ms/20.3 FPS)
- `dump_img.log` — FP16 quality verification dump
- `opt_phase*.log` — Earlier optimization phase logs

### Key reference files
- `tools/gen_espcn_onnx.py` — ESPCN ONNX generator (trained weights + uint8 output)
- `tools/train_espcn.py` — ESPCN training script
- `docs/HARDWARE_REPORT.md` — Full hardware capability analysis

## Context & Notes

- **Commit**: 0cede54 — "Optimize sort + rasterizer: 16-bit radix sort, FP16 tiles, persistent threads"
- **Previous commits**: 6920830 (projection optimization), 5bfccfb (ESPCN training + rasterizer basics)
- **Mars.ply**: 62,002 Gaussians, ~57K visible at default camera. SH degree 3.
- **Cortex-A55 limits**: L1D 32KB, L2 256KB (shared). FP16 tile buffers (2KB) fit L1. ProjectedGaussian array (2.8MB) far exceeds L2.
- **FP16 NEON**: `float16x8_t` processes 8 pixels/cycle. No `vbslq_f16` intrinsic — use `vreinterpretq` workaround via `u16`.
- **sudo required**: HDMI rendering and NPU need sudo. `--bench` needs sudo for display init. `--dump` works without sudo.
- **Three-tier rasterizer**: Tier 1 (FP16 8-wide, main path), Tier 2 (FP32 4-wide, remainder ≥4px), Tier 3 (scalar tail <4px).
