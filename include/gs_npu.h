#pragma once

#include "gs_types.h"
#include "gs_memory.h"
#include <cstdint>
#include <pthread.h>

extern "C" {
#include "ax_engine_api.h"
#include "ax_engine_type.h"
}

// Generic NPU (AX_ENGINE) acceleration context.
// Model-agnostic: loads any compiled .axmodel and provides inference.
// For super-resolution upscale, I/O CMM buffers are managed internally.
struct NPUContext {
    bool initialized = false;
    AX_ENGINE_HANDLE model_handle = nullptr;
    AX_ENGINE_IO_INFO_T *io_info = nullptr;
    AX_ENGINE_CONTEXT_T context = nullptr;

    // Mutex for NPU hardware serialization (single NPU unit)
    pthread_mutex_t npu_mutex;

    // Super-resolution I/O buffers (CMM-allocated)
    CmmBuffer input_buf;
    CmmBuffer output_buf;
    uint32_t input_h = 0, input_w = 0;
    uint32_t output_h = 0, output_w = 0;

    // Tensor format (detected from compiled model metadata)
    // Layout: 1=NHWC, 2=NCHW (AX_ENGINE_TENSOR_LAYOUT_T)
    // Dtype: 1=UINT8, 3=FLOAT32, 5=SINT8 (AX_ENGINE_DATA_TYPE_T)
    int input_layout = 0;
    int output_layout = 0;
    int input_dtype = 0;
    int output_dtype = 0;
};

// Initialize NPU context: load .axmodel, AX_ENGINE_Init.
// Returns false if model not found or NPU unavailable (non-fatal).
bool gs_npu_init(NPUContext &ctx, const char *model_path);

// Free NPU resources
void gs_npu_deinit(NPUContext &ctx);

// Run NPU super-resolution: ARGB8888 input → ARGB8888 output.
// Handles ARGB↔RGB planar conversion internally (NEON-optimized).
// Returns false on error (caller should fall back to sending render_fb directly).
bool gs_npu_upscale(NPUContext &ctx, const Framebuffer &input_fb, Framebuffer &output_fb);
