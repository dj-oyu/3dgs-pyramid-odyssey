#include "gs_mau.h"
#include "gs_memory.h"
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
#include "ax_ive_api.h"
}

// Build the fixed pixel feature matrix P_T[6, 256].
// For a 16x16 tile, column c = ly*16 + lx encodes local pixel (lx, ly):
//   Row 0: lx + 0.5
//   Row 1: ly + 0.5
//   Row 2: (lx + 0.5)^2
//   Row 3: (ly + 0.5)^2
//   Row 4: (lx + 0.5) * (ly + 0.5)
//   Row 5: 1.0
static void build_pixel_features(float *P_T) {
    for (uint32_t ly = 0; ly < TILE_SIZE; ly++) {
        for (uint32_t lx = 0; lx < TILE_SIZE; lx++) {
            uint32_t col = ly * TILE_SIZE + lx;
            float fx = (float)lx + 0.5f;
            float fy = (float)ly + 0.5f;
            P_T[0 * 256 + col] = fx;            // lx + 0.5
            P_T[1 * 256 + col] = fy;            // ly + 0.5
            P_T[2 * 256 + col] = fx * fx;        // (lx + 0.5)^2
            P_T[3 * 256 + col] = fy * fy;        // (ly + 0.5)^2
            P_T[4 * 256 + col] = fx * fy;        // (lx + 0.5)(ly + 0.5)
            P_T[5 * 256 + col] = 1.0f;           // constant
        }
    }
}

bool gs_mau_init(MAUContext &ctx) {
    if (ctx.initialized) return true;

    // Initialize IVE subsystem (for MAU access)
    AX_S32 ret = AX_IVE_Init();
    if (ret != AX_SUCCESS) {
        fprintf(stderr, "[gs_mau] AX_IVE_Init failed: 0x%08x (MAU unavailable)\n", ret);
        return false;
    }
    printf("[gs_mau] AX_IVE_Init OK\n");

    // Allocate P_T[6, 256] in CMM
    ctx.pixel_features_virt = gs_cmm_alloc_floats(6 * 256, "mau_P_T", &ctx.pixel_features_phys);
    if (!ctx.pixel_features_virt) {
        fprintf(stderr, "[gs_mau] Failed to allocate P_T matrix\n");
        AX_IVE_Exit();
        return false;
    }
    ctx.pixel_features_shape[0] = 6;
    ctx.pixel_features_shape[1] = 256;
    build_pixel_features(ctx.pixel_features_virt);
    printf("[gs_mau] P_T[6,256] built (%.1f KB)\n", 6 * 256 * 4 / 1024.0f);

    // Allocate per-thread scratch buffers
    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        auto &tb = ctx.thread_bufs[t];
        char tag[32];

        snprintf(tag, sizeof(tag), "mau_G_%u", t);
        tb.g_matrix_virt = gs_cmm_alloc_floats(ctx.max_k * 6, tag, &tb.g_matrix_phys);

        snprintf(tag, sizeof(tag), "mau_R_%u", t);
        tb.result_virt = gs_cmm_alloc_floats(ctx.max_k * 256, tag, &tb.result_phys);

        if (!tb.g_matrix_virt || !tb.result_virt) {
            fprintf(stderr, "[gs_mau] Failed to allocate thread %u buffers\n", t);
            gs_mau_deinit(ctx);
            return false;
        }

        tb.g_shape[0] = 0; tb.g_shape[1] = 6;
        tb.r_shape[0] = 0; tb.r_shape[1] = 256;
    }

    float total_kb = (6 * 256 * 4 +
                      NUM_RENDER_THREADS * ctx.max_k * 6 * 4 +
                      NUM_RENDER_THREADS * ctx.max_k * 256 * 4) / 1024.0f;
    printf("[gs_mau] Per-thread buffers: G_T[%u,6] + Result[%u,256] x %u threads\n",
           ctx.max_k, ctx.max_k, NUM_RENDER_THREADS);
    printf("[gs_mau] Total CMM: %.1f KB\n", total_kb);

    // Initialize mutex for MAU hardware serialization
    pthread_mutex_init(&ctx.mau_mutex, nullptr);

    ctx.initialized = true;

    // Probe test: run a small MatMul to verify MAU hardware is functional.
    // AX650C lacks MAU — IVE init succeeds but actual MatMul fails.
    {
        auto &tb = ctx.thread_bufs[0];
        // Use first 8 rows of the existing buffers for a quick 8x6 * 6x256 test
        tb.g_shape[0] = 8; tb.g_shape[1] = 6;
        tb.r_shape[0] = 8; tb.r_shape[1] = 256;
        memset(tb.g_matrix_virt, 0, 8 * 6 * sizeof(float));

        AX_IVE_MAU_MATMUL_INPUT_T probe_in = {};
        probe_in.stMatQ.u64PhyAddr  = tb.g_matrix_phys;
        probe_in.stMatQ.pVirAddr    = tb.g_matrix_virt;
        probe_in.stMatQ.pShape      = tb.g_shape;
        probe_in.stMatQ.u8ShapeSize = 2;
        probe_in.stMatQ.enDataType  = AX_IVE_MAU_DT_FLOAT32;
        probe_in.stMatB.u64PhyAddr  = ctx.pixel_features_phys;
        probe_in.stMatB.pVirAddr    = ctx.pixel_features_virt;
        probe_in.stMatB.pShape      = ctx.pixel_features_shape;
        probe_in.stMatB.u8ShapeSize = 2;
        probe_in.stMatB.enDataType  = AX_IVE_MAU_DT_FLOAT32;

        AX_IVE_MAU_MATMUL_OUTPUT_T probe_out = {};
        probe_out.stMulRes.u64PhyAddr  = tb.result_phys;
        probe_out.stMulRes.pVirAddr    = tb.result_virt;
        probe_out.stMulRes.pShape      = tb.r_shape;
        probe_out.stMulRes.u8ShapeSize = 2;
        probe_out.stMulRes.enDataType  = AX_IVE_MAU_DT_FLOAT32;

        AX_IVE_MAU_MATMUL_CTRL_T probe_ctrl = {};
        probe_ctrl.enMauId = AX_IVE_MAU_ID_0;
        probe_ctrl.s32DdrReadBandwidthLimit = -1;
        probe_ctrl.bEnableMulRes  = AX_TRUE;
        probe_ctrl.bEnableTopNRes = AX_FALSE;

        AX_IVE_HANDLE h;
        AX_S32 probe_ret = AX_IVE_MAU_MatMul(&h, &probe_in, &probe_out, &probe_ctrl,
                                               AX_IVE_ENGINE_MAU, AX_TRUE);
        if (probe_ret != AX_SUCCESS) {
            printf("[gs_mau] MAU probe MatMul failed: 0x%08x — hardware not available, disabling\n", probe_ret);
            gs_mau_deinit(ctx);
            return false;
        }
    }

    printf("[gs_mau] MAU context initialized (tile_threshold=%u)\n", ctx.tile_threshold);
    return true;
}

void gs_mau_deinit(MAUContext &ctx) {
    if (!ctx.initialized && !ctx.pixel_features_virt) return;

    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        auto &tb = ctx.thread_bufs[t];
        if (tb.g_matrix_virt) {
            gs_cmm_free_floats(tb.g_matrix_virt, ctx.max_k * 6);
            tb.g_matrix_virt = nullptr;
        }
        if (tb.result_virt) {
            gs_cmm_free_floats(tb.result_virt, ctx.max_k * 256);
            tb.result_virt = nullptr;
        }
    }

    if (ctx.pixel_features_virt) {
        gs_cmm_free_floats(ctx.pixel_features_virt, 6 * 256);
        ctx.pixel_features_virt = nullptr;
    }

    if (ctx.initialized) {
        pthread_mutex_destroy(&ctx.mau_mutex);
        AX_IVE_Exit();
        printf("[gs_mau] MAU context deinitialized\n");
    }

    ctx.initialized = false;
}

void gs_mau_build_g_matrix(float *G_T, const ProjectedGaussian *gaussians,
                           const uint32_t *indices, uint32_t K,
                           float tile_ox, float tile_oy) {
    // For each Gaussian k, compute 6 coefficients that encode the quadratic form
    // power = G_T[k,:] . P_T[:,pixel] where:
    //
    // The Mahalanobis distance at pixel (px, py) is:
    //   power = 0.5 * (inv_a*dx^2 + 2*inv_b*dx*dy + inv_c*dy^2)
    //   where dx = px - screen_x, dy = py - screen_y
    //
    // Let lx = px - tile_ox, ly = py - tile_oy (local tile coords)
    // Then dx = (lx + 0.5) + (tile_ox - screen_x) = (lx+0.5) - Sx
    //   where Sx = screen_x - tile_ox, Sy = screen_y - tile_oy
    //
    // Expanding:
    //   power = 0.5*inv_a*(lx+0.5)^2 - inv_a*Sx*(lx+0.5) + 0.5*inv_a*Sx^2
    //         + inv_b*(lx+0.5)*(ly+0.5) - inv_b*Sy*(lx+0.5) - inv_b*Sx*(ly+0.5) + inv_b*Sx*Sy
    //         + 0.5*inv_c*(ly+0.5)^2 - inv_c*Sy*(ly+0.5) + 0.5*inv_c*Sy^2
    //
    // Grouping by P_T features:
    //   coeff[0] * (lx+0.5)  = -(inv_a*Sx + inv_b*Sy)
    //   coeff[1] * (ly+0.5)  = -(inv_b*Sx + inv_c*Sy)
    //   coeff[2] * (lx+0.5)^2 = 0.5*inv_a
    //   coeff[3] * (ly+0.5)^2 = 0.5*inv_c
    //   coeff[4] * (lx+0.5)(ly+0.5) = inv_b
    //   coeff[5] * 1.0        = 0.5*(inv_a*Sx^2 + 2*inv_b*Sx*Sy + inv_c*Sy^2)

    for (uint32_t k = 0; k < K; k++) {
        const ProjectedGaussian &pg = gaussians[indices[k]];

        // cov2d_a/b/c are precomputed: ha=0.5*inv_a, hb=inv_b, hc=0.5*inv_c
        float inv_a = 2.0f * pg.cov2d_a;   // ha * 2
        float inv_b = pg.cov2d_b;           // hb = inv_b
        float inv_c = 2.0f * pg.cov2d_c;   // hc * 2

        // Gaussian center offset from tile origin
        float Sx = pg.screen_x - tile_ox;
        float Sy = pg.screen_y - tile_oy;

        float *row = &G_T[k * 6];
        row[0] = -(inv_a * Sx + inv_b * Sy);                        // coeff for (lx+0.5)
        row[1] = -(inv_b * Sx + inv_c * Sy);                        // coeff for (ly+0.5)
        row[2] = 0.5f * inv_a;                                       // coeff for (lx+0.5)^2 = ha
        row[3] = 0.5f * inv_c;                                       // coeff for (ly+0.5)^2 = hc
        row[4] = inv_b;                                               // coeff for (lx+0.5)(ly+0.5)
        row[5] = 0.5f * (inv_a * Sx * Sx + 2.0f * inv_b * Sx * Sy + inv_c * Sy * Sy);  // constant
    }
}

bool gs_mau_matmul(MAUContext &ctx, MAUContext::ThreadBuf &tbuf, uint32_t K) {
    // Set up shapes for this call
    tbuf.g_shape[0] = (int32_t)K;
    tbuf.g_shape[1] = 6;
    tbuf.r_shape[0] = (int32_t)K;
    tbuf.r_shape[1] = 256;

    AX_IVE_MAU_MATMUL_INPUT_T input = {};
    input.stMatQ.u64PhyAddr  = tbuf.g_matrix_phys;
    input.stMatQ.pVirAddr    = tbuf.g_matrix_virt;
    input.stMatQ.pShape      = tbuf.g_shape;
    input.stMatQ.u8ShapeSize = 2;
    input.stMatQ.enDataType  = AX_IVE_MAU_DT_FLOAT32;

    input.stMatB.u64PhyAddr  = ctx.pixel_features_phys;
    input.stMatB.pVirAddr    = ctx.pixel_features_virt;
    input.stMatB.pShape      = ctx.pixel_features_shape;
    input.stMatB.u8ShapeSize = 2;
    input.stMatB.enDataType  = AX_IVE_MAU_DT_FLOAT32;

    AX_IVE_MAU_MATMUL_OUTPUT_T output = {};
    output.stMulRes.u64PhyAddr  = tbuf.result_phys;
    output.stMulRes.pVirAddr    = tbuf.result_virt;
    output.stMulRes.pShape      = tbuf.r_shape;
    output.stMulRes.u8ShapeSize = 2;
    output.stMulRes.enDataType  = AX_IVE_MAU_DT_FLOAT32;

    AX_IVE_MAU_MATMUL_CTRL_T ctrl = {};
    ctrl.enMauId = AX_IVE_MAU_ID_0;
    ctrl.s32DdrReadBandwidthLimit = -1;
    ctrl.bEnableMulRes  = AX_TRUE;
    ctrl.bEnableTopNRes = AX_FALSE;

    // Serialize MAU hardware access across threads
    pthread_mutex_lock(&ctx.mau_mutex);

    AX_IVE_HANDLE handle;
    AX_S32 ret = AX_IVE_MAU_MatMul(&handle, &input, &output, &ctrl,
                                     AX_IVE_ENGINE_MAU, AX_TRUE);

    pthread_mutex_unlock(&ctx.mau_mutex);

    if (ret != AX_SUCCESS) {
        fprintf(stderr, "[gs_mau] MAU_MatMul failed: 0x%08x (K=%u)\n", ret, K);
        return false;
    }
    return true;
}
