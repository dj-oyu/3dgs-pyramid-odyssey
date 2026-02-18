#include "gs_npu.h"
#include "gs_memory.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <arm_neon.h>

static const char *layout_str(int layout) {
    switch (layout) {
        case 1: return "NHWC";
        case 2: return "NCHW";
        default: return "UNKNOWN";
    }
}

static const char *dtype_str(int dtype) {
    switch (dtype) {
        case 1: return "UINT8";
        case 2: return "UINT16";
        case 3: return "FLOAT32";
        case 4: return "SINT16";
        case 5: return "SINT8";
        case 6: return "SINT32";
        case 7: return "UINT32";
        case 8: return "FLOAT64";
        default: return "UNKNOWN";
    }
}

bool gs_npu_init(NPUContext &ctx, const char *model_path) {
    if (ctx.initialized) return true;

    // 1. Load .axmodel file
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        printf("[gs_npu] Model not found: %s (NPU disabled)\n", model_path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fprintf(stderr, "[gs_npu] Invalid model file: %s\n", model_path);
        fclose(f);
        return false;
    }

    void *model_data = malloc((size_t)file_size);
    if (!model_data) {
        fprintf(stderr, "[gs_npu] Failed to allocate %ld bytes for model\n", file_size);
        fclose(f);
        return false;
    }

    size_t read_bytes = fread(model_data, 1, (size_t)file_size, f);
    fclose(f);

    if ((long)read_bytes != file_size) {
        fprintf(stderr, "[gs_npu] Failed to read model file (got %zu/%ld)\n", read_bytes, file_size);
        free(model_data);
        return false;
    }

    printf("[gs_npu] Loaded model: %s (%ld bytes)\n", model_path, file_size);

    // 2. Initialize AX_ENGINE
    AX_ENGINE_NPU_ATTR_T npu_attr = {};
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;

    AX_S32 ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) {
        fprintf(stderr, "[gs_npu] AX_ENGINE_Init failed: 0x%08x\n", ret);
        free(model_data);
        return false;
    }
    printf("[gs_npu] AX_ENGINE_Init OK (VIRTUAL_NPU_DISABLE)\n");

    // 3. Create model handle
    ret = AX_ENGINE_CreateHandle(&ctx.model_handle, model_data, (AX_U32)file_size);
    free(model_data);

    if (ret != 0) {
        fprintf(stderr, "[gs_npu] AX_ENGINE_CreateHandle failed: 0x%08x\n", ret);
        AX_ENGINE_Deinit();
        return false;
    }
    printf("[gs_npu] Model handle created\n");

    // 4. Get I/O info
    ret = AX_ENGINE_GetIOInfo(ctx.model_handle, &ctx.io_info);
    if (ret != 0) {
        fprintf(stderr, "[gs_npu] AX_ENGINE_GetIOInfo failed: 0x%08x\n", ret);
        AX_ENGINE_DestroyHandle(ctx.model_handle);
        AX_ENGINE_Deinit();
        return false;
    }

    printf("[gs_npu] Model I/O: %u inputs, %u outputs\n",
           ctx.io_info->nInputSize, ctx.io_info->nOutputSize);

    if (ctx.io_info->nInputSize < 1 || ctx.io_info->nOutputSize < 1) {
        fprintf(stderr, "[gs_npu] Model must have at least 1 input and 1 output\n");
        AX_ENGINE_DestroyHandle(ctx.model_handle);
        AX_ENGINE_Deinit();
        return false;
    }

    // Log detailed input/output metadata
    for (AX_U32 i = 0; i < ctx.io_info->nInputSize; i++) {
        auto &meta = ctx.io_info->pInputs[i];
        printf("[gs_npu]   Input[%u] '%s': shape=[", i, meta.pName);
        for (AX_U8 d = 0; d < meta.nShapeSize; d++) {
            printf("%d%s", meta.pShape[d], d + 1 < meta.nShapeSize ? "," : "");
        }
        printf("] dtype=%s(%d) layout=%s(%d) nSize=%u quantVal=%u\n",
               dtype_str(meta.eDataType), meta.eDataType,
               layout_str(meta.eLayout), meta.eLayout,
               meta.nSize, meta.nQuantizationValue);
    }
    for (AX_U32 i = 0; i < ctx.io_info->nOutputSize; i++) {
        auto &meta = ctx.io_info->pOutputs[i];
        printf("[gs_npu]   Output[%u] '%s': shape=[", i, meta.pName);
        for (AX_U8 d = 0; d < meta.nShapeSize; d++) {
            printf("%d%s", meta.pShape[d], d + 1 < meta.nShapeSize ? "," : "");
        }
        printf("] dtype=%s(%d) layout=%s(%d) nSize=%u quantVal=%u\n",
               dtype_str(meta.eDataType), meta.eDataType,
               layout_str(meta.eLayout), meta.eLayout,
               meta.nSize, meta.nQuantizationValue);
    }

    // 5. Create inference context
    ret = AX_ENGINE_CreateContext(ctx.model_handle);
    if (ret != 0) {
        fprintf(stderr, "[gs_npu] AX_ENGINE_CreateContext failed: 0x%08x\n", ret);
        AX_ENGINE_DestroyHandle(ctx.model_handle);
        AX_ENGINE_Deinit();
        return false;
    }
    printf("[gs_npu] Inference context created\n");

    // 6. Parse I/O dimensions, layouts, and dtypes
    {
        auto &in_meta = ctx.io_info->pInputs[0];
        auto &out_meta = ctx.io_info->pOutputs[0];

        ctx.input_layout = (int)in_meta.eLayout;
        ctx.output_layout = (int)out_meta.eLayout;
        ctx.input_dtype = (int)in_meta.eDataType;
        ctx.output_dtype = (int)out_meta.eDataType;

        // Parse dimensions based on layout
        if (in_meta.eLayout == AX_ENGINE_TENSOR_LAYOUT_NHWC && in_meta.nShapeSize >= 4) {
            ctx.input_h = in_meta.pShape[1];
            ctx.input_w = in_meta.pShape[2];
        } else if (in_meta.nShapeSize >= 4) {
            // NCHW or UNKNOWN: [N, C, H, W]
            ctx.input_h = in_meta.pShape[2];
            ctx.input_w = in_meta.pShape[3];
        }

        if (out_meta.eLayout == AX_ENGINE_TENSOR_LAYOUT_NHWC && out_meta.nShapeSize >= 4) {
            ctx.output_h = out_meta.pShape[1];
            ctx.output_w = out_meta.pShape[2];
        } else if (out_meta.nShapeSize >= 4) {
            ctx.output_h = out_meta.pShape[2];
            ctx.output_w = out_meta.pShape[3];
        }

        printf("[gs_npu] Super-res: input %ux%u %s %s (%u bytes) -> output %ux%u %s %s (%u bytes)\n",
               ctx.input_w, ctx.input_h,
               layout_str(ctx.input_layout), dtype_str(ctx.input_dtype), in_meta.nSize,
               ctx.output_w, ctx.output_h,
               layout_str(ctx.output_layout), dtype_str(ctx.output_dtype), out_meta.nSize);

        if (!gs_cmm_alloc(ctx.input_buf, in_meta.nSize, "npu_sr_in")) {
            fprintf(stderr, "[gs_npu] Failed to allocate input CMM buffer (%u bytes)\n", in_meta.nSize);
            AX_ENGINE_DestroyHandle(ctx.model_handle);
            AX_ENGINE_Deinit();
            return false;
        }

        if (!gs_cmm_alloc_cached(ctx.output_buf, out_meta.nSize, "npu_sr_out")) {
            fprintf(stderr, "[gs_npu] Failed to allocate output CMM buffer (%u bytes)\n", out_meta.nSize);
            gs_cmm_free(ctx.input_buf);
            AX_ENGINE_DestroyHandle(ctx.model_handle);
            AX_ENGINE_Deinit();
            return false;
        }
    }

    // 7. Initialize mutex
    pthread_mutex_init(&ctx.npu_mutex, nullptr);

    ctx.initialized = true;
    printf("[gs_npu] NPU context initialized\n");
    return true;
}

void gs_npu_deinit(NPUContext &ctx) {
    if (!ctx.initialized && !ctx.model_handle) return;

    if (ctx.input_buf.virt_addr) gs_cmm_free(ctx.input_buf);
    if (ctx.output_buf.virt_addr) gs_cmm_free(ctx.output_buf);

    if (ctx.model_handle) {
        AX_ENGINE_DestroyHandle(ctx.model_handle);
        ctx.model_handle = nullptr;
    }

    if (ctx.initialized) {
        pthread_mutex_destroy(&ctx.npu_mutex);
        AX_ENGINE_Deinit();
        printf("[gs_npu] NPU context deinitialized\n");
    }

    ctx.io_info = nullptr;
    ctx.context = nullptr;
    ctx.input_h = ctx.input_w = 0;
    ctx.output_h = ctx.output_w = 0;
    ctx.initialized = false;
}

// ──── ARGB8888 → float32 NCHW [0.0, 1.0] ────
// Writes float32 planar: [R plane: H*W floats][G plane][B plane]
static void argb_to_float_nchw(const uint8_t *src, uint32_t stride,
                                float *dst, uint32_t width, uint32_t height) {
    uint32_t plane = width * height;
    float *dst_r = dst;
    float *dst_g = dst + plane;
    float *dst_b = dst + plane * 2;
    const float scale = 1.0f / 255.0f;
    float32x4_t vscale = vdupq_n_f32(scale);

    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *row = src + y * stride;
        uint32_t offset = y * width;
        uint32_t x = 0;

        // NEON: 8 pixels per iteration
        for (; x + 8 <= width; x += 8) {
            // Load 8 BGRA pixels (32 bytes), deinterleaved
            uint8x8x4_t bgra = vld4_u8(row + x * 4);
            // bgra.val[0]=B, val[1]=G, val[2]=R, val[3]=A

            // R channel: uint8 → uint16 → uint32 → float32 × (1/255)
            uint16x8_t r16 = vmovl_u8(bgra.val[2]);
            float32x4_t r_lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))), vscale);
            float32x4_t r_hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16))), vscale);
            vst1q_f32(dst_r + offset + x, r_lo);
            vst1q_f32(dst_r + offset + x + 4, r_hi);

            // G channel
            uint16x8_t g16 = vmovl_u8(bgra.val[1]);
            float32x4_t g_lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))), vscale);
            float32x4_t g_hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16))), vscale);
            vst1q_f32(dst_g + offset + x, g_lo);
            vst1q_f32(dst_g + offset + x + 4, g_hi);

            // B channel
            uint16x8_t b16 = vmovl_u8(bgra.val[0]);
            float32x4_t b_lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))), vscale);
            float32x4_t b_hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16))), vscale);
            vst1q_f32(dst_b + offset + x, b_lo);
            vst1q_f32(dst_b + offset + x + 4, b_hi);
        }

        for (; x < width; x++) {
            uint32_t px = x * 4;
            dst_r[offset + x] = row[px + 2] * scale;
            dst_g[offset + x] = row[px + 1] * scale;
            dst_b[offset + x] = row[px + 0] * scale;
        }
    }
}

// ──── float32 NCHW [0.0, 1.0] → ARGB8888 ────
// Output CMM buffer is allocated with cached mapping. After NPU DMA write,
// gs_cmm_invalidate_cache() is called so CPU reads hit cache, not raw DRAM.
static void float_nchw_to_argb(const float *src, uint8_t *dst, uint32_t stride,
                                uint32_t width, uint32_t height) {
    uint32_t plane = width * height;
    const float *src_r = src;
    const float *src_g = src + plane;
    const float *src_b = src + plane * 2;
    float32x4_t v255 = vdupq_n_f32(255.0f);
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t *row = dst + y * stride;
        uint32_t offset = y * width;
        uint32_t x = 0;

        for (; x + 8 <= width; x += 8) {
            float32x4_t r_lo = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_r + offset + x), v255), vzero), v255);
            float32x4_t r_hi = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_r + offset + x + 4), v255), vzero), v255);
            uint16x4_t r16_lo = vqmovn_u32(vcvtq_u32_f32(r_lo));
            uint16x4_t r16_hi = vqmovn_u32(vcvtq_u32_f32(r_hi));
            uint8x8_t r8 = vqmovn_u16(vcombine_u16(r16_lo, r16_hi));

            float32x4_t g_lo = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_g + offset + x), v255), vzero), v255);
            float32x4_t g_hi = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_g + offset + x + 4), v255), vzero), v255);
            uint16x4_t g16_lo = vqmovn_u32(vcvtq_u32_f32(g_lo));
            uint16x4_t g16_hi = vqmovn_u32(vcvtq_u32_f32(g_hi));
            uint8x8_t g8 = vqmovn_u16(vcombine_u16(g16_lo, g16_hi));

            float32x4_t b_lo = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_b + offset + x), v255), vzero), v255);
            float32x4_t b_hi = vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(src_b + offset + x + 4), v255), vzero), v255);
            uint16x4_t b16_lo = vqmovn_u32(vcvtq_u32_f32(b_lo));
            uint16x4_t b16_hi = vqmovn_u32(vcvtq_u32_f32(b_hi));
            uint8x8_t b8 = vqmovn_u16(vcombine_u16(b16_lo, b16_hi));

            uint8x8_t a8 = vdup_n_u8(0xFF);
            uint8x8x4_t bgra;
            bgra.val[0] = b8;
            bgra.val[1] = g8;
            bgra.val[2] = r8;
            bgra.val[3] = a8;
            vst4_u8(row + x * 4, bgra);
        }

        for (; x < width; x++) {
            uint32_t px = x * 4;
            float r = src_r[offset + x] * 255.0f;
            float g = src_g[offset + x] * 255.0f;
            float b = src_b[offset + x] * 255.0f;
            row[px + 0] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
            row[px + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
            row[px + 2] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
            row[px + 3] = 0xFF;
        }
    }
}

// ──── uint8 NCHW format conversions (planar) ────

static void argb_to_u8_nchw(const uint8_t *src, uint32_t stride,
                             uint8_t *dst, uint32_t width, uint32_t height) {
    uint32_t plane = width * height;
    uint8_t *dst_r = dst;
    uint8_t *dst_g = dst + plane;
    uint8_t *dst_b = dst + plane * 2;

    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *row = src + y * stride;
        uint32_t offset = y * width;
        uint32_t x = 0;

        for (; x + 16 <= width; x += 16) {
            uint8x16x4_t argb = vld4q_u8(row + x * 4);
            vst1q_u8(dst_r + offset + x, argb.val[2]);
            vst1q_u8(dst_g + offset + x, argb.val[1]);
            vst1q_u8(dst_b + offset + x, argb.val[0]);
        }

        for (; x < width; x++) {
            uint32_t px = x * 4;
            dst_b[offset + x] = row[px + 0];
            dst_g[offset + x] = row[px + 1];
            dst_r[offset + x] = row[px + 2];
        }
    }
}

static void u8_nchw_to_argb(const uint8_t *src, uint8_t *dst, uint32_t stride,
                             uint32_t width, uint32_t height) {
    uint32_t plane = width * height;
    const uint8_t *src_r = src;
    const uint8_t *src_g = src + plane;
    const uint8_t *src_b = src + plane * 2;
    uint8x16_t alpha = vdupq_n_u8(0xFF);

    for (uint32_t y = 0; y < height; y++) {
        uint8_t *row = dst + y * stride;
        uint32_t offset = y * width;
        uint32_t x = 0;

        for (; x + 16 <= width; x += 16) {
            uint8x16x4_t argb;
            argb.val[0] = vld1q_u8(src_b + offset + x);
            argb.val[1] = vld1q_u8(src_g + offset + x);
            argb.val[2] = vld1q_u8(src_r + offset + x);
            argb.val[3] = alpha;
            vst4q_u8(row + x * 4, argb);
        }

        for (; x < width; x++) {
            uint32_t px = x * 4;
            row[px + 0] = src_b[offset + x];
            row[px + 1] = src_g[offset + x];
            row[px + 2] = src_r[offset + x];
            row[px + 3] = 0xFF;
        }
    }
}


bool gs_npu_upscale(NPUContext &ctx, const Framebuffer &input_fb, Framebuffer &output_fb) {
    if (!ctx.initialized) return false;

    if (input_fb.width != ctx.input_w || input_fb.height != ctx.input_h) {
        fprintf(stderr, "[gs_npu] Upscale size mismatch: fb=%ux%u, model expects %ux%u\n",
                input_fb.width, input_fb.height, ctx.input_w, ctx.input_h);
        return false;
    }

    uint8_t *in_ptr = static_cast<uint8_t*>(ctx.input_buf.virt_addr);
    uint8_t *out_ptr = static_cast<uint8_t*>(ctx.output_buf.virt_addr);
    bool is_float_in = (ctx.input_dtype == 3);   // FLOAT32
    bool is_float_out = (ctx.output_dtype == 3);  // FLOAT32

    // 1. Convert input ARGB8888 → NPU input buffer (NCHW)
    if (is_float_in) {
        argb_to_float_nchw(input_fb.data, input_fb.stride,
                           reinterpret_cast<float*>(in_ptr),
                           ctx.input_w, ctx.input_h);
    } else {
        argb_to_u8_nchw(input_fb.data, input_fb.stride,
                        in_ptr, ctx.input_w, ctx.input_h);
    }

    // 2. Run NPU inference
    AX_ENGINE_IO_BUFFER_T input_io = {};
    input_io.phyAddr = ctx.input_buf.phys_addr;
    input_io.pVirAddr = ctx.input_buf.virt_addr;
    input_io.nSize = ctx.input_buf.size;

    AX_ENGINE_IO_BUFFER_T output_io = {};
    output_io.phyAddr = ctx.output_buf.phys_addr;
    output_io.pVirAddr = ctx.output_buf.virt_addr;
    output_io.nSize = ctx.output_buf.size;

    AX_ENGINE_IO_T io = {};
    io.pInputs = &input_io;
    io.nInputSize = 1;
    io.pOutputs = &output_io;
    io.nOutputSize = 1;

    pthread_mutex_lock(&ctx.npu_mutex);
    AX_S32 ret = AX_ENGINE_RunSync(ctx.model_handle, &io);
    pthread_mutex_unlock(&ctx.npu_mutex);

    if (ret != 0) {
        fprintf(stderr, "[gs_npu] AX_ENGINE_RunSync failed: 0x%08x\n", ret);
        return false;
    }

    // Invalidate CPU cache for output buffer (NPU wrote via DMA, cache is stale)
    gs_cmm_invalidate_cache(ctx.output_buf);

    // 3. Convert NPU output → ARGB8888
    if (is_float_out) {
        float_nchw_to_argb(reinterpret_cast<float*>(out_ptr),
                           output_fb.data, output_fb.stride,
                           ctx.output_w, ctx.output_h);
    } else {
        u8_nchw_to_argb(out_ptr, output_fb.data, output_fb.stride,
                        ctx.output_w, ctx.output_h);
    }

    return true;
}
