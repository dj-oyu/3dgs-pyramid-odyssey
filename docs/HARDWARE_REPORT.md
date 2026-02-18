# AX8850 (AX650C) Hardware Specification & Implementation Status Report

Date: 2026-02-18
Board: M5Stack AI Pyramid-Pro (AX650N_M5stack_8G)
SoC: Axera AX8850 (AX650C_CHIP)

## 1. SoC Identity

| Item | Value | Source |
|------|-------|--------|
| Marketing name | **AX8850** | Product spec |
| Internal chip type | `AX650C_CHIP` | `/proc/ax_proc/chip_type` via hw_probe |
| Board ID | `AX650N_M5stack_8G` | `/proc/ax_proc/board_id` via hw_probe |
| NPU performance | **24 TOPS @ INT8** | Product spec |
| SDK version | V3.6.4 | `/proc/ax_proc/version` |

### Naming Correspondence

| Marketing Name | Internal Chip | Notes |
|----------------|---------------|-------|
| AX8850N | AX650N | Full-featured variant |
| **AX8850** | **AX650C** | **This device** — cost-reduced variant |
| AX650A | AX650A | Alternative variant |

Source: AXERA-TECH/ax-pipeline CMakeLists.txt treats AX650A, AX650N, AX8850, AX8850N all under `AXERA_TARGET_CHIP_AX650` — no code-level differentiation between variants.

## 2. CPU

| Item | Value | Verified By |
|------|-------|-------------|
| Architecture | ARMv8.2-A (AArch64) | `/proc/cpuinfo` |
| Cores | **8 x Cortex-A55** | `/proc/cpuinfo` processor count |
| Max frequency | 1500 MHz | `/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq` |
| NEON/ASIMD | YES | `/proc/cpuinfo` Features |
| FP16 (fphp) | YES | `/proc/cpuinfo` Features |
| DotProd (asimddp) | YES | `/proc/cpuinfo` Features |
| CRC32 | YES | `/proc/cpuinfo` Features |
| LSE Atomics | YES | `/proc/cpuinfo` Features |

**Important**: The renderer currently uses `NUM_RENDER_THREADS = 4` (in `gs_types.h:18`). The device has **8 cores** — increasing to 6-8 threads is a potential optimization.

## 3. Memory

| Item | Value | Source |
|------|-------|--------|
| Total RAM | ~8 GB | Board ID suffix `_8G` |
| System RAM (MemTotal) | ~2 GB | `/proc/meminfo` |
| CMM Total | ~6 GB (6,291,456 KB) | `AX_SYS_MemQueryStatus()` |
| CMM Free | ~5 GB | `AX_SYS_MemQueryStatus()` |

CMM (Contiguous Memory Manager) provides physically contiguous buffers required by all hardware accelerators (VO, IVE, MAU, NPU).

## 4. SDK Subsystem Probe Results

All results from `tools/hw_probe` (programmatic verification):

| Subsystem | API | Status | Error Code |
|-----------|-----|--------|------------|
| System init | `AX_SYS_Init()` | **OK** | — |
| CMM alloc | `AX_SYS_MemAlloc()` | **OK** | — |
| IVE init | `AX_IVE_Init()` | **OK** | — |
| IVE DMA | `AX_IVE_DMA()` | **OK** | — |
| MAU MatMul (FLOAT32) | `AX_IVE_MAU_MatMul()` | **FAIL** | `0x80150103` |
| MAU MatMul (FLOAT16) | `AX_IVE_MAU_MatMul()` | **FAIL** | `0x80150103` |
| MAU MatMul (SINT8) | `AX_IVE_MAU_MatMul()` | **FAIL** | `0x80150103` |
| MAU MatMul (UINT8) | `AX_IVE_MAU_MatMul()` | **FAIL** | `0x80150103` |
| NPU MatMul (via IVE) | `AX_IVE_NPU_CreateMatMulHandle()` | **FAIL** | "get attr failed" |
| NPU Engine | `AX_ENGINE_Init(VIRTUAL_NPU_STD)` | **FAIL** | `0x80060087` |
| NPU Engine | `AX_ENGINE_Init(VIRTUAL_NPU_DISABLE)` | **OK** | v2.12.0s |

Error `0x80150103` = "mau create handle error: get attr error -1" — the MAU hardware attribute query fails, indicating the MAU unit is not present or not exposed on this chip variant.

### AX_ENGINE Init Mode

The initial hw_probe used `AX_ENGINE_VIRTUAL_NPU_STD` which failed (`0x80060087`). After studying AXERA-TECH/ax-pipeline (which uses `VIRTUAL_NPU_DISABLE` for all AX650), the updated probe confirmed:
- `VIRTUAL_NPU_DISABLE` (full NPU, no virtualization): **OK** (v2.12.0s)
- `VIRTUAL_NPU_STD`: FAIL
- `VIRTUAL_NPU_BIG_LITTLE`: Not tested (DISABLE succeeded first)

This means the NPU is available for model inference but does not support virtual NPU partitioning on this chip.

## 5. Kernel Infrastructure

### Kernel Modules (34 modules loaded)

Key modules by function:

| Category | Module | Size | Description |
|----------|--------|------|-------------|
| Core | `ax_sys` | 48 KB | System/memory management |
| Core | `ax_cmm` | 1832 KB | Contiguous memory manager |
| Core | `ax_pool` | 80 KB | Buffer pool management |
| Core | `ax_base` | 24 KB | Base driver |
| NPU | `ax_npu` | 404 KB | NPU compute driver |
| NPU | `ax_proton` | 2676 KB | NPU runtime (largest module) |
| Display | `ax_vo` | 768 KB | Video output (HDMI) |
| Display | `ax_fb` | 16 KB | Framebuffer |
| Video | `ax_ivps` | 7880 KB | Image/Video processing (largest) |
| Video | `ax_venc` | 88 KB | Video encoder |
| Video | `ax_vdec` | 736 KB | Video decoder |
| Video | `ax_jenc` | 72 KB | JPEG encoder |
| Video | `ax_jdec` | 324 KB | JPEG decoder |
| IVE | `ax_ive` | 76 KB | Intelligent Video Engine |
| DSP | `ax_vdsp` | 44 KB | RISC-V DSP coprocessor |
| DSP | `ax_riscv` | 32 KB | RISC-V support |
| GDC | `ax_gdc` | 128 KB | Geometric distortion correction |
| PCIe | `ax_pcie_*` | ~160 KB | PCIe host (4 modules) |

Note: No MAU-specific kernel module is loaded.

### Device Nodes (28 nodes)
- `/dev/npu` — NPU device (confirmed working via AX_ENGINE)
- `/dev/ax_ive` — IVE device (confirmed working)
- `/dev/ax_cmm` — CMM device (confirmed working)
- `/dev/ax_proton` — NPU runtime device
- No `/dev/*mau*` device — confirms MAU hardware is not exposed

## 6. HW Accelerator Availability Matrix

| Accelerator | Spec (AX8850) | This Device | Programmatic Test |
|-------------|---------------|-------------|-------------------|
| CPU 8x A55 NEON | YES | **YES** | `/proc/cpuinfo` |
| CMM (6 GB) | YES | **YES** | `AX_SYS_MemQueryStatus()` |
| AX_VO (HDMI) | YES | **YES** | VO init + display |
| IVE (DMA/filter) | YES | **YES** | `AX_IVE_DMA()` |
| IVPS (CSC/resize) | YES | **YES** | Linked library exists |
| VENC/VDEC | YES | Likely YES | Not tested (unused) |
| MAU (Matrix Unit) | **24 TOPS spec** | **NO** | `AX_IVE_MAU_MatMul()` → 0x80150103 |
| NPU MatMul (IVE) | Unknown | **NO** | `AX_IVE_NPU_CreateMatMulHandle()` → FAIL |
| NPU Inference | 24 TOPS INT8 | **YES** | `AX_ENGINE_Init(DISABLE)` → OK, v2.12.0s |

## 7. MAU Analysis

### What is MAU?

The MAU (Matrix Arithmetic Unit) is a dedicated hardware accelerator exposed via `AX_IVE_MAU_MatMul()` in the AX650 SDK. It supports FLOAT32, FLOAT16, SINT8, UINT8 matrix multiplication. Per the AX8850 spec sheet, 24 TOPS @ INT8 is the aggregate NPU compute budget.

### Why MAU is Unavailable

1. **All data types fail with the same error** (`0x80150103` = "get attr error -1"), indicating the MAU hardware block is not present or not accessible via firmware on this AX650C variant.
2. **No `/dev/*mau*` device node** exists in the kernel.
3. **AXERA-TECH/ax-pipeline does not reference MAU at all** — zero mentions of MAU, MatMul, or matrix multiplication APIs in their entire codebase. This reference project (the official AX650 demo pipeline) uses only `AX_ENGINE` for NPU inference and `AX_IVPS` for image processing.
4. The AX650C may reserve the 24 TOPS compute budget exclusively for model inference (`AX_ENGINE`) rather than exposing it as a general-purpose matrix multiply unit.

### Hypothesis

The MAU is likely a sub-block of the NPU that is only accessible on the full AX650N (AX8850N) variant, or requires specific firmware/SDK enablement not present in SDK V3.6.4. The AX650C (AX8850) cost-reduced variant may physically lack this block, or it may be fused off.

## 8. NPU (AX_ENGINE) Inference Path

The NPU inference engine (`AX_ENGINE`) is the primary way to use the 24 TOPS INT8 compute on AX650 series. It runs pre-compiled neural network models (`.axmodel` format).

### How ax-pipeline Uses AX_ENGINE

```
1. AX_ENGINE_Init(&attr)     // attr.eHardMode = VIRTUAL_NPU_DISABLE
2. AX_ENGINE_CreateHandle()   // Load .axmodel from file
3. AX_ENGINE_CreateContext()   // Create execution context
4. AX_ENGINE_GetIOInfo()       // Query tensor shapes/types
5. AX_ENGINE_RunSync()         // Execute inference
6. AX_ENGINE_DestroyHandle()   // Cleanup
```

### Potential for 3DGS

`AX_ENGINE_Init(VIRTUAL_NPU_DISABLE)` succeeds. This means it is possible to:
1. **Compile a custom MatMul model** — export a simple `[K,6] × [6,256]` matrix multiply as an ONNX model, compile with Axera's `pulsar2` toolchain to `.axmodel`
2. **Run it via AX_ENGINE** — treat the Mahalanobis distance computation as a "model inference" call
3. **Batch operations** — accumulate multiple tiles' G matrices and run a batched inference

Limitations:
- INT8 quantization may reduce precision (OK for rendering, but needs verification)
- Model loading has startup cost
- Fixed input shapes may require multiple models for different tile Gaussian counts
- AX_ENGINE overhead per call is unknown

## 9. Implementation Status

### Completed Code (MAU Path)

All MAU acceleration code has been implemented and compiles cleanly:

| File | Description | Status |
|------|-------------|--------|
| `include/gs_mau.h` | MAUContext struct, API declarations | Complete |
| `src/gs_mau.cpp` | MAU init/deinit, P_T builder, G_T builder, MatMul wrapper | Complete |
| `src/gs_rasterizer.cpp` | `rasterize_tile_mau()` + hybrid dispatch | Complete |
| `include/gs_rasterizer.h` | Updated signature with MAUContext* | Complete |
| `include/gs_renderer.h` | MAUContext in Renderer struct | Complete |
| `src/gs_renderer.cpp` | MAU lifecycle integration | Complete |
| `include/gs_types.h` | mau_tiles/cpu_tiles in RenderStats | Complete |
| `src/main.cpp` | MAU tile stats display | Complete |
| `tools/mau_bench.cpp` | MAU/NPU benchmark tool | Complete |
| `tools/hw_probe.cpp` | Hardware capability probe | Complete |
| `Makefile` | Build rules for new files | Complete |

### Graceful Degradation

The MAU code path is designed for graceful degradation:
- `gs_mau_init()` returns `false` when MAU hardware is unavailable
- Renderer falls back to CPU-only NEON rasterization automatically
- **Zero overhead** when MAU is inactive — no extra function calls or branches in the hot path
- Verified: dump mode produces correct output images with MAU unavailable

### Current Renderer Performance (CPU-only)

Auditorium scene (17K Gaussians, 1920x1080):

| Stage | Time | Share |
|-------|------|-------|
| Projection | 7.3 ms | 3% |
| Sort | 2.5 ms | 1% |
| **Rasterize** | **205 ms** | **89%** |
| Total | 229 ms | **4.4 FPS** |

## 10. Recommended Next Steps

### A. Increase NUM_RENDER_THREADS from 4 to 8 (Quick Win)

The device has 8 cores but the renderer only uses 4 threads. Doubling threads could reduce rasterize time by up to ~40% (limited by memory bandwidth, not pure compute).

Edit `include/gs_types.h:18`:
```cpp
static constexpr uint32_t NUM_RENDER_THREADS = 8;
```

### B. NPU Model-Based Acceleration (AX_ENGINE confirmed working)

`AX_ENGINE_Init(VIRTUAL_NPU_DISABLE)` succeeds — 24 TOPS INT8 NPU is available.

1. Export a simple MatMul ONNX model (`[K,6] × [6,256]`)
2. Compile with Axera `pulsar2` toolchain to `.axmodel`
3. Load via `AX_ENGINE_CreateHandle()` + `AX_ENGINE_RunSync()`
4. Benchmark inference latency at various K sizes
5. If < 50μs/call, integrate as rasterizer acceleration path

### C. CPU-Side Rasterizer Optimization (Guaranteed Path)

Since MAU is unavailable, focus on CPU-side improvements:
- **8-thread rasterization** (see A above)
- Hierarchical tile rejection (coarse 32x32 pre-pass)
- Reduced-precision power computation (FP16 NEON for Mahalanobis)
- Better cache tiling (process 4x4 pixel sub-blocks)
- Explore NEON `fmla` + `frintx` for fast exp approximation

## 11. Test Tools Reference

```bash
# Hardware capability probe (requires sudo for HW access)
sudo ./build/hw_probe

# MAU/NPU benchmark (requires sudo)
sudo ./build/mau_bench

# Renderer benchmark (requires sudo for HDMI, or use --dump for headless)
build/gs_splat ~/ply/Auditorium\ by\ the\ sea.ply -s 2 --dump data/dump/audit -n 8

# PLY file info (no sudo needed)
build/ply_info ~/ply/Mars.ply
```
