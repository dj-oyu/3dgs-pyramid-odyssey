#include "gs_rasterizer.h"
#include "gs_mau.h"
#include "gs_math.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <pthread.h>
#include <algorithm>
#include <arm_neon.h>

bool gs_raster_alloc(RasterContext &ctx, uint32_t width, uint32_t height) {
    ctx.render_width = width;
    ctx.render_height = height;

    uint32_t tiles_x = (width  + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    ctx.num_tiles = tiles_x * tiles_y;

    ctx.tiles = static_cast<TileGaussians*>(
        calloc(ctx.num_tiles, sizeof(TileGaussians)));
    if (!ctx.tiles) return false;

    // Pre-allocate initial capacity for each tile
    for (uint32_t i = 0; i < ctx.num_tiles; i++) {
        ctx.tiles[i].capacity = 256;
        ctx.tiles[i].indices = static_cast<uint32_t*>(
            malloc(256 * sizeof(uint32_t)));
        ctx.tiles[i].count = 0;
    }

    ctx.allocated = true;
    return true;
}

void gs_raster_free(RasterContext &ctx) {
    if (ctx.tiles) {
        for (uint32_t i = 0; i < ctx.num_tiles; i++) {
            free(ctx.tiles[i].indices);
        }
        free(ctx.tiles);
        ctx.tiles = nullptr;
    }
    ctx.num_tiles = 0;
    ctx.allocated = false;
}

void gs_raster_assign_tiles(RasterContext &ctx, const ProjectedGaussian *gaussians, uint32_t count) {
    uint32_t tiles_x = (ctx.render_width  + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tiles_y = (ctx.render_height + TILE_SIZE - 1) / TILE_SIZE;

    // Reset tile counts
    for (uint32_t i = 0; i < ctx.num_tiles; i++) {
        ctx.tiles[i].count = 0;
    }

    for (uint32_t gi = 0; gi < count; gi++) {
        const ProjectedGaussian &pg = gaussians[gi];

        // Early reject tiny/transparent Gaussians
        if (pg.radius < 0.5f) continue;
        if (pg.opacity < ALPHA_THRESHOLD) continue;

        // Compute tile range from bounding circle
        int min_tx = std::max(0, (int)floorf((pg.screen_x - pg.radius) / TILE_SIZE));
        int max_tx = std::min((int)tiles_x - 1, (int)floorf((pg.screen_x + pg.radius) / TILE_SIZE));
        int min_ty = std::max(0, (int)floorf((pg.screen_y - pg.radius) / TILE_SIZE));
        int max_ty = std::min((int)tiles_y - 1, (int)floorf((pg.screen_y + pg.radius) / TILE_SIZE));

        for (int ty = min_ty; ty <= max_ty; ty++) {
            for (int tx = min_tx; tx <= max_tx; tx++) {
                uint32_t tile_idx = ty * tiles_x + tx;
                TileGaussians &tile = ctx.tiles[tile_idx];

                // Grow if needed
                if (tile.count >= tile.capacity) {
                    if (tile.capacity >= MAX_GAUSSIANS_PER_TILE) continue;
                    uint32_t new_cap = std::min(tile.capacity * 2, MAX_GAUSSIANS_PER_TILE);
                    auto *new_indices = static_cast<uint32_t*>(
                        realloc(tile.indices, new_cap * sizeof(uint32_t)));
                    if (!new_indices) continue;
                    tile.indices = new_indices;
                    tile.capacity = new_cap;
                }

                tile.indices[tile.count++] = gi;
            }
        }
    }
}

// Forward declaration for CPU fallback in MAU path
static void rasterize_tile(const TileGaussians &tile,
                           const ProjectedGaussian *gaussians,
                           uint8_t *fb_data, uint32_t fb_stride,
                           uint32_t tile_x, uint32_t tile_y,
                           uint32_t fb_width, uint32_t fb_height,
                           uint32_t bg_color);

// MAU-assisted tile rasterizer: uses precomputed power values from MAU MatMul
static void rasterize_tile_mau(const TileGaussians &tile,
                               const ProjectedGaussian *gaussians,
                               uint8_t *fb_data, uint32_t fb_stride,
                               uint32_t tile_x, uint32_t tile_y,
                               uint32_t fb_width, uint32_t fb_height,
                               uint32_t bg_color,
                               MAUContext &mau_ctx,
                               MAUContext::ThreadBuf &tbuf) {
    uint32_t px_start = tile_x * TILE_SIZE;
    uint32_t py_start = tile_y * TILE_SIZE;
    uint32_t px_end = std::min(px_start + TILE_SIZE, fb_width);
    uint32_t py_end = std::min(py_start + TILE_SIZE, fb_height);

    uint32_t K = std::min(tile.count, mau_ctx.max_k);

    // Phase A: Build G_T[K, 6] on CPU
    float tile_ox = (float)px_start;
    float tile_oy = (float)py_start;
    gs_mau_build_g_matrix(tbuf.g_matrix_virt, gaussians, tile.indices, K,
                          tile_ox, tile_oy);

    // Phase B: MAU MatMul → Power[K, 256]
    if (!gs_mau_matmul(mau_ctx, tbuf, K)) {
        // MAU failed — fall back to CPU rasterizer for this tile
        rasterize_tile(tile, gaussians, fb_data, fb_stride,
                       tile_x, tile_y, fb_width, fb_height, bg_color);
        return;
    }

    // Phase C: NEON sequential alpha compositing using precomputed power values
    float tile_r[TILE_SIZE][TILE_SIZE] = {};
    float tile_g[TILE_SIZE][TILE_SIZE] = {};
    float tile_b[TILE_SIZE][TILE_SIZE] = {};
    float tile_T[TILE_SIZE][TILE_SIZE];
    for (uint32_t y = 0; y < TILE_SIZE; y++)
        for (uint32_t x = 0; x < TILE_SIZE; x++)
            tile_T[y][x] = 1.0f;

    float32x4_t v_trans_min = vdupq_n_f32(TRANSMITTANCE_MIN);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_min_alpha = vdupq_n_f32(ALPHA_THRESHOLD);
    float32x4_t v_power_max = vdupq_n_f32(4.0f);

    for (uint32_t ki = 0; ki < K; ki++) {
        const ProjectedGaussian &pg = gaussians[tile.indices[ki]];

        // Skip degenerate Gaussians (same as CPU path)
        float det = pg.cov2d_a * pg.cov2d_c - pg.cov2d_b * pg.cov2d_b;
        if (det <= 0.0f) continue;

        float alpha = pg.opacity;
        float cr = pg.color_r * alpha;
        float cg = pg.color_g * alpha;
        float cb = pg.color_b * alpha;

        float32x4_t v_cr = vdupq_n_f32(cr);
        float32x4_t v_cg = vdupq_n_f32(cg);
        float32x4_t v_cb = vdupq_n_f32(cb);
        float32x4_t v_alpha = vdupq_n_f32(alpha);

        // Power values for this Gaussian: contiguous row in result matrix
        float *power_row = &tbuf.result_virt[ki * 256];

        uint32_t ly_end = py_end - py_start;
        uint32_t lx_end = px_end - px_start;

        for (uint32_t ly = 0; ly < ly_end; ly++) {
            // Process 4 pixels at a time with NEON
            uint32_t lx = 0;
            for (; lx + 4 <= lx_end; lx += 4) {
                // Load transmittances
                float32x4_t v_T = vld1q_f32(&tile_T[ly][lx]);

                // Check if all transmittances are below threshold
                uint32x4_t mask_alive = vcgtq_f32(v_T, v_trans_min);
                if (vmaxvq_u32(mask_alive) == 0) continue;

                // Load precomputed power values (contiguous in result matrix)
                float32x4_t v_power = vld1q_f32(&power_row[ly * TILE_SIZE + lx]);

                // Power cutoff
                uint32x4_t mask_power = vcltq_f32(v_power, v_power_max);
                // Also skip negative power (shouldn't happen but safety check)
                mask_power = vandq_u32(mask_power, vcgeq_f32(v_power, vdupq_n_f32(0.0f)));
                if (vmaxvq_u32(mask_power) == 0) continue;

                // Gaussian weight: exp(-power) * alpha
                float32x4_t v_gauss = neon_fast_exp_neg(v_power);
                float32x4_t v_w = vmulq_f32(v_gauss, v_alpha);

                // Visibility mask
                uint32x4_t mask_visible = vandq_u32(
                    vandq_u32(vcgtq_f32(v_w, v_min_alpha), mask_power), mask_alive);

                // Apply: color += T * gauss * (color * opacity)
                float32x4_t v_contrib = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v_T, v_gauss)), mask_visible));

                float32x4_t v_r = vld1q_f32(&tile_r[ly][lx]);
                float32x4_t v_g = vld1q_f32(&tile_g[ly][lx]);
                float32x4_t v_b = vld1q_f32(&tile_b[ly][lx]);

                vst1q_f32(&tile_r[ly][lx], vmlaq_f32(v_r, v_contrib, v_cr));
                vst1q_f32(&tile_g[ly][lx], vmlaq_f32(v_g, v_contrib, v_cg));
                vst1q_f32(&tile_b[ly][lx], vmlaq_f32(v_b, v_contrib, v_cb));

                // Update transmittance
                float32x4_t v_one_minus_w = vsubq_f32(v_one, v_w);
                v_one_minus_w = vbslq_f32(mask_visible, v_one_minus_w, v_one);
                vst1q_f32(&tile_T[ly][lx], vmulq_f32(v_T, v_one_minus_w));
            }

            // Scalar remainder
            for (; lx < lx_end; lx++) {
                float T = tile_T[ly][lx];
                if (T < TRANSMITTANCE_MIN) continue;

                float power = power_row[ly * TILE_SIZE + lx];
                if (power < 0.0f || power > 4.0f) continue;

                float gauss = expf(-power);
                float w = gauss * alpha;
                if (w < ALPHA_THRESHOLD) continue;

                float contrib = T * gauss;
                tile_r[ly][lx] += contrib * cr;
                tile_g[ly][lx] += contrib * cg;
                tile_b[ly][lx] += contrib * cb;
                tile_T[ly][lx] *= (1.0f - w);
            }
        }

        // Tile-level transmittance saturation check (every 4 Gaussians)
        if ((ki & 3u) == 3u) {
            bool tile_saturated = true;
            for (uint32_t y = 0; y < TILE_SIZE && tile_saturated; y++)
                for (uint32_t x = 0; x < TILE_SIZE; x += 4) {
                    float32x4_t v_T = vld1q_f32(&tile_T[y][x]);
                    if (vmaxvq_u32(vcgeq_f32(v_T, vdupq_n_f32(TRANSMITTANCE_MIN))) != 0) {
                        tile_saturated = false;
                        break;
                    }
                }
            if (tile_saturated) break;
        }
    }

    // Process overflow Gaussians (beyond max_k) with CPU path inline
    // (unlikely for typical scenes, but handles edge cases)
    if (tile.count > K) {
        for (uint32_t gi = K; gi < tile.count; gi++) {
            const ProjectedGaussian &pg = gaussians[tile.indices[gi]];

            float det = pg.cov2d_a * pg.cov2d_c - pg.cov2d_b * pg.cov2d_b;
            if (det <= 0.0f) continue;
            float inv_det = 1.0f / det;
            float ha = 0.5f *  pg.cov2d_c * inv_det;
            float hb =        -pg.cov2d_b * inv_det;
            float hc = 0.5f *  pg.cov2d_a * inv_det;

            float alpha = pg.opacity;
            float cr = pg.color_r * alpha;
            float cg = pg.color_g * alpha;
            float cb = pg.color_b * alpha;

            uint32_t ly_end = py_end - py_start;
            uint32_t lx_end = px_end - px_start;

            for (uint32_t ly = 0; ly < ly_end; ly++) {
                float py = (float)(py_start + ly) + 0.5f;
                float dy = py - pg.screen_y;
                float dy_sq_term = hc * dy * dy;
                float dy_cross_term = hb * dy;

                for (uint32_t lx = 0; lx < lx_end; lx++) {
                    float T = tile_T[ly][lx];
                    if (T < TRANSMITTANCE_MIN) continue;

                    float px = (float)(px_start + lx) + 0.5f;
                    float dx = px - pg.screen_x;
                    float power = ha*dx*dx + dy_cross_term*dx + dy_sq_term;
                    if (power > 4.0f) continue;

                    float gauss = expf(-power);
                    float w = gauss * alpha;
                    if (w < ALPHA_THRESHOLD) continue;

                    float contrib = T * gauss;
                    tile_r[ly][lx] += contrib * cr;
                    tile_g[ly][lx] += contrib * cg;
                    tile_b[ly][lx] += contrib * cb;
                    tile_T[ly][lx] *= (1.0f - w);
                }
            }
        }
    }

    // Write tile to framebuffer (ARGB8888) — NEON 4-pixel vectorized
    {
        float32x4_t v_bg_r = vdupq_n_f32((float)((bg_color >> 16) & 0xFF) / 255.0f);
        float32x4_t v_bg_g = vdupq_n_f32((float)((bg_color >> 8)  & 0xFF) / 255.0f);
        float32x4_t v_bg_b = vdupq_n_f32((float)( bg_color        & 0xFF) / 255.0f);
        float32x4_t v_255  = vdupq_n_f32(255.0f);
        float32x4_t v_zero = vdupq_n_f32(0.0f);
        uint32x4_t  v_alpha_byte = vdupq_n_u32(0xFF000000u);

        for (uint32_t py = py_start; py < py_end; py++) {
            uint32_t ly = py - py_start;
            uint32_t *row = reinterpret_cast<uint32_t*>(fb_data + py * fb_stride);

            uint32_t px = px_start;
            for (; px + 4 <= px_end; px += 4) {
                uint32_t lx = px - px_start;

                float32x4_t v_T = vld1q_f32(&tile_T[ly][lx]);
                float32x4_t v_r = vld1q_f32(&tile_r[ly][lx]);
                float32x4_t v_g = vld1q_f32(&tile_g[ly][lx]);
                float32x4_t v_b = vld1q_f32(&tile_b[ly][lx]);

                v_r = vmlaq_f32(v_r, v_T, v_bg_r);
                v_g = vmlaq_f32(v_g, v_T, v_bg_g);
                v_b = vmlaq_f32(v_b, v_T, v_bg_b);

                v_r = vmaxq_f32(vminq_f32(vmulq_f32(v_r, v_255), v_255), v_zero);
                v_g = vmaxq_f32(vminq_f32(vmulq_f32(v_g, v_255), v_255), v_zero);
                v_b = vmaxq_f32(vminq_f32(vmulq_f32(v_b, v_255), v_255), v_zero);

                uint32x4_t u_r = vcvtq_u32_f32(v_r);
                uint32x4_t u_g = vcvtq_u32_f32(v_g);
                uint32x4_t u_b = vcvtq_u32_f32(v_b);

                uint32x4_t argb = vorrq_u32(v_alpha_byte,
                                  vorrq_u32(vshlq_n_u32(u_r, 16),
                                  vorrq_u32(vshlq_n_u32(u_g, 8), u_b)));

                vst1q_u32(&row[px], argb);
            }

            for (; px < px_end; px++) {
                uint32_t lx = px - px_start;
                float T = tile_T[ly][lx];

                float r = tile_r[ly][lx] + T * ((float)((bg_color >> 16) & 0xFF) / 255.0f);
                float g = tile_g[ly][lx] + T * ((float)((bg_color >> 8)  & 0xFF) / 255.0f);
                float b = tile_b[ly][lx] + T * ((float)( bg_color        & 0xFF) / 255.0f);

                uint8_t ri = (uint8_t)std::min(255.0f, std::max(0.0f, r * 255.0f));
                uint8_t gi_val = (uint8_t)std::min(255.0f, std::max(0.0f, g * 255.0f));
                uint8_t bi = (uint8_t)std::min(255.0f, std::max(0.0f, b * 255.0f));

                row[px] = (0xFFu << 24) | ((uint32_t)ri << 16) | ((uint32_t)gi_val << 8) | bi;
            }
        }
    }
}

// Per-thread rasterization work
struct RasterThreadArg {
    const RasterContext *ctx;
    const ProjectedGaussian *gaussians;
    Framebuffer *fb;
    uint32_t tile_start;
    uint32_t tile_end;
    uint32_t bg_color;
    MAUContext *mau_ctx;     // nullptr if MAU unavailable
    uint32_t thread_id;
    uint32_t mau_tile_count; // output: tiles processed by MAU path
    uint32_t cpu_tile_count; // output: tiles processed by CPU path
};

static void rasterize_tile(const TileGaussians &tile,
                           const ProjectedGaussian *gaussians,
                           uint8_t *fb_data, uint32_t fb_stride,
                           uint32_t tile_x, uint32_t tile_y,
                           uint32_t fb_width, uint32_t fb_height,
                           uint32_t bg_color) {
    uint32_t px_start = tile_x * TILE_SIZE;
    uint32_t py_start = tile_y * TILE_SIZE;
    uint32_t px_end = std::min(px_start + TILE_SIZE, fb_width);
    uint32_t py_end = std::min(py_start + TILE_SIZE, fb_height);

    // Initialize tile pixels with background + full transmittance
    float tile_r[TILE_SIZE][TILE_SIZE] = {};
    float tile_g[TILE_SIZE][TILE_SIZE] = {};
    float tile_b[TILE_SIZE][TILE_SIZE] = {};
    float tile_T[TILE_SIZE][TILE_SIZE];
    for (uint32_t y = 0; y < TILE_SIZE; y++)
        for (uint32_t x = 0; x < TILE_SIZE; x++)
            tile_T[y][x] = 1.0f;

    // Tile center for Opt-R2 pre-test
    float tcx = (float)px_start + TILE_SIZE * 0.5f;
    float tcy = (float)py_start + TILE_SIZE * 0.5f;
    // Half-diagonal of tile (conservative margin for tile extent)
    static constexpr float TILE_MARGIN = TILE_SIZE * 0.7071f;  // sqrt(2)/2 * TILE_SIZE

    // Loop-invariant NEON constants (hoisted out of per-Gaussian loop)
    float32x4_t v_trans_min = vdupq_n_f32(TRANSMITTANCE_MIN);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_min_alpha = vdupq_n_f32(ALPHA_THRESHOLD);
    float32x4_t v_power_max = vdupq_n_f32(4.0f);
    float32x4_t v_step4 = {0.0f, 1.0f, 2.0f, 3.0f};
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    float16x8_t v_power_max_h = vdupq_n_f16((__fp16)4.0f);
    float16x8_t v_step8_h = {(__fp16)0.0f, (__fp16)1.0f, (__fp16)2.0f, (__fp16)3.0f,
                              (__fp16)4.0f, (__fp16)5.0f, (__fp16)6.0f, (__fp16)7.0f};
#endif

    // Process Gaussians front-to-back (already sorted)
    for (uint32_t gi = 0; gi < tile.count; gi++) {
        // Prefetch next Gaussian data to hide memory latency
        if (gi + 1 < tile.count)
            __builtin_prefetch(&gaussians[tile.indices[gi + 1]], 0, 1);

        const ProjectedGaussian &pg = gaussians[tile.indices[gi]];

        // Precompute inverse covariance with 0.5 pre-multiplied into a and c
        float det = pg.cov2d_a * pg.cov2d_c - pg.cov2d_b * pg.cov2d_b;
        if (det <= 0.0f) continue;
        float inv_det = 1.0f / det;
        float ha = 0.5f *  pg.cov2d_c * inv_det;  // 0.5 * inv_a
        float hb =        -pg.cov2d_b * inv_det;   // inv_b (cross term already has implicit 2x)
        float hc = 0.5f *  pg.cov2d_a * inv_det;  // 0.5 * inv_c

        // Opt-R2: Tile-level Gaussian pre-test
        // Compute Mahalanobis distance at tile center, subtract conservative margin.
        // If the closest possible pixel still has power > 4.0, skip entire Gaussian.
        float tdx = tcx - pg.screen_x;
        float tdy = tcy - pg.screen_y;
        float tile_power = ha*tdx*tdx + hb*tdx*tdy + hc*tdy*tdy;
        float max_hinv = fmaxf(ha, hc);
        if (tile_power - max_hinv * TILE_MARGIN * TILE_MARGIN > 4.0f) continue;

        // Clamp pixel range to Gaussian bounding box within tile
        float rad = pg.radius;
        uint32_t g_px0 = (uint32_t)std::max((int)px_start, (int)floorf(pg.screen_x - rad));
        uint32_t g_px1 = std::min(px_end, (uint32_t)ceilf(pg.screen_x + rad));
        uint32_t g_py0 = (uint32_t)std::max((int)py_start, (int)floorf(pg.screen_y - rad));
        uint32_t g_py1 = std::min(py_end, (uint32_t)ceilf(pg.screen_y + rad));
        if (g_px0 >= g_px1 || g_py0 >= g_py1) continue;

        // Align g_px0 down to 8-pixel boundary (relative to px_start) for FP16 NEON
        uint32_t lx0_aligned = ((g_px0 - px_start) & ~7u);
        g_px0 = px_start + lx0_aligned;

        float alpha = pg.opacity;
        float cr = pg.color_r * alpha;
        float cg = pg.color_g * alpha;
        float cb = pg.color_b * alpha;

        // Per-Gaussian NEON vectors
        float32x4_t v_cr = vdupq_n_f32(cr);
        float32x4_t v_cg = vdupq_n_f32(cg);
        float32x4_t v_cb = vdupq_n_f32(cb);
        float32x4_t v_alpha = vdupq_n_f32(alpha);
        float32x4_t v_ha = vdupq_n_f32(ha);

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        float16x8_t v_ha_h = vdupq_n_f16((__fp16)ha);
#endif

        for (uint32_t py = g_py0; py < g_py1; py++) {
            float dy = (float)py + 0.5f - pg.screen_y;
            float dy_sq_term = hc * dy * dy;
            float dy_cross_term = hb * dy;
            uint32_t ly = py - py_start;

            uint32_t px = g_px0;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            // Tier 1: 8-wide FP16 Mahalanobis + exp, FP32 color accumulation
            float16x8_t v_dy_sq_h = vdupq_n_f16((__fp16)dy_sq_term);
            float16x8_t v_dy_cross_h = vdupq_n_f16((__fp16)dy_cross_term);

            for (; px + 8 <= g_px1; px += 8) {
                uint32_t lx = px - px_start;

                // Load transmittances (2x FP32 loads for 8 pixels)
                float32x4_t v_T_lo = vld1q_f32(&tile_T[ly][lx]);
                float32x4_t v_T_hi = vld1q_f32(&tile_T[ly][lx + 4]);

                // Check if all 8 transmittances are below threshold
                uint32x4_t mask_alive_lo = vcgtq_f32(v_T_lo, v_trans_min);
                uint32x4_t mask_alive_hi = vcgtq_f32(v_T_hi, v_trans_min);
                if (vmaxvq_u32(mask_alive_lo) == 0 && vmaxvq_u32(mask_alive_hi) == 0) continue;

                // Compute dx for 8 pixels in FP16
                float16x8_t v_dx_h = vaddq_f16(vdupq_n_f16((__fp16)((float)px + 0.5f - pg.screen_x)),
                                                v_step8_h);

                // Mahalanobis distance in FP16: power = ha*dx^2 + hb*dx*dy + hc*dy^2
                float16x8_t v_power_h = vmulq_f16(v_ha_h, vmulq_f16(v_dx_h, v_dx_h));
                v_power_h = vfmaq_f16(v_power_h, v_dy_cross_h, v_dx_h);
                v_power_h = vaddq_f16(v_power_h, v_dy_sq_h);

                // Power cutoff in FP16
                uint16x8_t mask_power_h = vcltq_f16(v_power_h, v_power_max_h);
                if (vmaxvq_u16(mask_power_h) == 0) continue;

                // FP16 exp(-power)
                float16x8_t v_gauss_h = neon_fast_exp_neg_f16(v_power_h);

                // Convert to 2x FP32 for color accumulation
                float32x4_t v_gauss_lo = vcvt_f32_f16(vget_low_f16(v_gauss_h));
                float32x4_t v_gauss_hi = vcvt_f32_f16(vget_high_f16(v_gauss_h));

                // Widen power mask to full 32-bit masks (0xFFFF -> 0xFFFFFFFF)
                uint32x4_t mask_power_lo = vcgtq_u32(vmovl_u16(vget_low_u16(mask_power_h)),
                                                      vdupq_n_u32(0));
                uint32x4_t mask_power_hi = vcgtq_u32(vmovl_u16(vget_high_u16(mask_power_h)),
                                                      vdupq_n_u32(0));

                // -- Process low 4 pixels --
                float32x4_t v_w_lo = vmulq_f32(v_gauss_lo, v_alpha);
                uint32x4_t mask_vis_lo = vandq_u32(
                    vandq_u32(vcgtq_f32(v_w_lo, v_min_alpha), mask_power_lo), mask_alive_lo);

                float32x4_t v_contrib_lo = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v_T_lo, v_gauss_lo)), mask_vis_lo));

                float32x4_t v_r_lo = vld1q_f32(&tile_r[ly][lx]);
                float32x4_t v_g_lo = vld1q_f32(&tile_g[ly][lx]);
                float32x4_t v_b_lo = vld1q_f32(&tile_b[ly][lx]);

                vst1q_f32(&tile_r[ly][lx], vmlaq_f32(v_r_lo, v_contrib_lo, v_cr));
                vst1q_f32(&tile_g[ly][lx], vmlaq_f32(v_g_lo, v_contrib_lo, v_cg));
                vst1q_f32(&tile_b[ly][lx], vmlaq_f32(v_b_lo, v_contrib_lo, v_cb));

                float32x4_t v_omw_lo = vsubq_f32(v_one, v_w_lo);
                v_omw_lo = vbslq_f32(mask_vis_lo, v_omw_lo, v_one);
                vst1q_f32(&tile_T[ly][lx], vmulq_f32(v_T_lo, v_omw_lo));

                // -- Process high 4 pixels --
                float32x4_t v_w_hi = vmulq_f32(v_gauss_hi, v_alpha);
                uint32x4_t mask_vis_hi = vandq_u32(
                    vandq_u32(vcgtq_f32(v_w_hi, v_min_alpha), mask_power_hi), mask_alive_hi);

                float32x4_t v_contrib_hi = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v_T_hi, v_gauss_hi)), mask_vis_hi));

                float32x4_t v_r_hi = vld1q_f32(&tile_r[ly][lx + 4]);
                float32x4_t v_g_hi = vld1q_f32(&tile_g[ly][lx + 4]);
                float32x4_t v_b_hi = vld1q_f32(&tile_b[ly][lx + 4]);

                vst1q_f32(&tile_r[ly][lx + 4], vmlaq_f32(v_r_hi, v_contrib_hi, v_cr));
                vst1q_f32(&tile_g[ly][lx + 4], vmlaq_f32(v_g_hi, v_contrib_hi, v_cg));
                vst1q_f32(&tile_b[ly][lx + 4], vmlaq_f32(v_b_hi, v_contrib_hi, v_cb));

                float32x4_t v_omw_hi = vsubq_f32(v_one, v_w_hi);
                v_omw_hi = vbslq_f32(mask_vis_hi, v_omw_hi, v_one);
                vst1q_f32(&tile_T[ly][lx + 4], vmulq_f32(v_T_hi, v_omw_hi));
            }
#endif

            // Tier 2: 4-wide FP32 NEON for remaining 4-7 pixels
            float32x4_t v_dy_sq = vdupq_n_f32(dy_sq_term);
            float32x4_t v_dy_cross = vdupq_n_f32(dy_cross_term);

            for (; px + 4 <= g_px1; px += 4) {
                uint32_t lx = px - px_start;

                float32x4_t v_T = vld1q_f32(&tile_T[ly][lx]);
                uint32x4_t mask_alive = vcgtq_f32(v_T, v_trans_min);
                if (vmaxvq_u32(mask_alive) == 0) continue;

                float base_dx = (float)px + 0.5f - pg.screen_x;
                float32x4_t v_dx = vaddq_f32(vdupq_n_f32(base_dx), v_step4);

                float32x4_t v_power = vmulq_f32(v_ha, vmulq_f32(v_dx, v_dx));
                v_power = vmlaq_f32(v_power, v_dy_cross, v_dx);
                v_power = vaddq_f32(v_power, v_dy_sq);

                uint32x4_t mask_power = vcltq_f32(v_power, v_power_max);
                if (vmaxvq_u32(mask_power) == 0) continue;

                float32x4_t v_gauss = neon_fast_exp_neg(v_power);
                float32x4_t v_w = vmulq_f32(v_gauss, v_alpha);

                uint32x4_t mask_visible = vandq_u32(
                    vandq_u32(vcgtq_f32(v_w, v_min_alpha), mask_power), mask_alive);

                float32x4_t v_contrib = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v_T, v_gauss)), mask_visible));

                float32x4_t v_r = vld1q_f32(&tile_r[ly][lx]);
                float32x4_t v_g = vld1q_f32(&tile_g[ly][lx]);
                float32x4_t v_b = vld1q_f32(&tile_b[ly][lx]);

                vst1q_f32(&tile_r[ly][lx], vmlaq_f32(v_r, v_contrib, v_cr));
                vst1q_f32(&tile_g[ly][lx], vmlaq_f32(v_g, v_contrib, v_cg));
                vst1q_f32(&tile_b[ly][lx], vmlaq_f32(v_b, v_contrib, v_cb));

                float32x4_t v_one_minus_w = vsubq_f32(v_one, v_w);
                v_one_minus_w = vbslq_f32(mask_visible, v_one_minus_w, v_one);
                vst1q_f32(&tile_T[ly][lx], vmulq_f32(v_T, v_one_minus_w));
            }

            // Tier 3: Scalar tail for remaining 0-3 pixels
            for (; px < g_px1; px++) {
                uint32_t lx = px - px_start;
                float T = tile_T[ly][lx];
                if (T < TRANSMITTANCE_MIN) continue;

                float dx = (float)px + 0.5f - pg.screen_x;
                float power = ha*dx*dx + hb*dx*dy + hc*dy*dy;
                if (power > 4.0f) continue;

                float gauss = expf(-power);
                float w = gauss * alpha;
                if (w < ALPHA_THRESHOLD) continue;

                float contrib = T * gauss;
                tile_r[ly][lx] += contrib * cr;
                tile_g[ly][lx] += contrib * cg;
                tile_b[ly][lx] += contrib * cb;
                tile_T[ly][lx] *= (1.0f - w);
            }
        }

        // Opt-R4: Tile-level transmittance saturation check (every 4 Gaussians)
        if ((gi & 3u) == 3u) {
            bool tile_saturated = true;
            for (uint32_t y = 0; y < TILE_SIZE && tile_saturated; y++)
                for (uint32_t x = 0; x < TILE_SIZE; x += 4) {
                    float32x4_t v_T = vld1q_f32(&tile_T[y][x]);
                    if (vmaxvq_u32(vcgeq_f32(v_T, vdupq_n_f32(TRANSMITTANCE_MIN))) != 0) {
                        tile_saturated = false;
                        break;
                    }
                }
            if (tile_saturated) break;
        }
    }

    // Write tile to framebuffer (ARGB8888) — NEON 4-pixel vectorized
    float32x4_t v_bg_r = vdupq_n_f32((float)((bg_color >> 16) & 0xFF) / 255.0f);
    float32x4_t v_bg_g = vdupq_n_f32((float)((bg_color >> 8)  & 0xFF) / 255.0f);
    float32x4_t v_bg_b = vdupq_n_f32((float)( bg_color        & 0xFF) / 255.0f);
    float32x4_t v_255  = vdupq_n_f32(255.0f);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    uint32x4_t  v_alpha_byte = vdupq_n_u32(0xFF000000u);

    for (uint32_t py = py_start; py < py_end; py++) {
        uint32_t ly = py - py_start;
        uint32_t *row = reinterpret_cast<uint32_t*>(fb_data + py * fb_stride);

        // NEON path: 4 pixels at a time
        uint32_t px = px_start;
        for (; px + 4 <= px_end; px += 4) {
            uint32_t lx = px - px_start;

            // Load tile data
            float32x4_t v_T = vld1q_f32(&tile_T[ly][lx]);
            float32x4_t v_r = vld1q_f32(&tile_r[ly][lx]);
            float32x4_t v_g = vld1q_f32(&tile_g[ly][lx]);
            float32x4_t v_b = vld1q_f32(&tile_b[ly][lx]);

            // Add background: color + T * bg_color
            v_r = vmlaq_f32(v_r, v_T, v_bg_r);
            v_g = vmlaq_f32(v_g, v_T, v_bg_g);
            v_b = vmlaq_f32(v_b, v_T, v_bg_b);

            // Convert to [0,255] and clamp
            v_r = vmaxq_f32(vminq_f32(vmulq_f32(v_r, v_255), v_255), v_zero);
            v_g = vmaxq_f32(vminq_f32(vmulq_f32(v_g, v_255), v_255), v_zero);
            v_b = vmaxq_f32(vminq_f32(vmulq_f32(v_b, v_255), v_255), v_zero);

            // Convert to uint32 and pack ARGB8888
            uint32x4_t u_r = vcvtq_u32_f32(v_r);
            uint32x4_t u_g = vcvtq_u32_f32(v_g);
            uint32x4_t u_b = vcvtq_u32_f32(v_b);

            uint32x4_t argb = vorrq_u32(v_alpha_byte,
                              vorrq_u32(vshlq_n_u32(u_r, 16),
                              vorrq_u32(vshlq_n_u32(u_g, 8), u_b)));

            vst1q_u32(&row[px], argb);
        }

        // Scalar tail for edge tiles
        for (; px < px_end; px++) {
            uint32_t lx = px - px_start;
            float T = tile_T[ly][lx];

            float r = tile_r[ly][lx] + T * ((float)((bg_color >> 16) & 0xFF) / 255.0f);
            float g = tile_g[ly][lx] + T * ((float)((bg_color >> 8)  & 0xFF) / 255.0f);
            float b = tile_b[ly][lx] + T * ((float)( bg_color        & 0xFF) / 255.0f);

            uint8_t ri = (uint8_t)std::min(255.0f, std::max(0.0f, r * 255.0f));
            uint8_t gi_val = (uint8_t)std::min(255.0f, std::max(0.0f, g * 255.0f));
            uint8_t bi = (uint8_t)std::min(255.0f, std::max(0.0f, b * 255.0f));

            row[px] = (0xFFu << 24) | ((uint32_t)ri << 16) | ((uint32_t)gi_val << 8) | bi;
        }
    }
}

static void *raster_thread_func(void *arg) {
    auto *a = static_cast<RasterThreadArg*>(arg);
    uint32_t tiles_x = (a->ctx->render_width + TILE_SIZE - 1) / TILE_SIZE;
    a->mau_tile_count = 0;
    a->cpu_tile_count = 0;

    for (uint32_t t = a->tile_start; t < a->tile_end; t++) {
        if (a->ctx->tiles[t].count == 0) {
            // Fill empty tile with background color
            uint32_t tx = t % tiles_x;
            uint32_t ty = t / tiles_x;
            uint32_t px_start = tx * TILE_SIZE;
            uint32_t py_start = ty * TILE_SIZE;
            uint32_t px_end = std::min(px_start + TILE_SIZE, a->fb->width);
            uint32_t py_end = std::min(py_start + TILE_SIZE, a->fb->height);

            uint32x4_t bg_vec = vdupq_n_u32(a->bg_color);
            for (uint32_t py = py_start; py < py_end; py++) {
                uint32_t *row = reinterpret_cast<uint32_t*>(a->fb->data + py * a->fb->stride);
                uint32_t px = px_start;
                for (; px + 4 <= px_end; px += 4) vst1q_u32(&row[px], bg_vec);
                for (; px < px_end; px++) row[px] = a->bg_color;
            }
            continue;
        }

        uint32_t tx = t % tiles_x;
        uint32_t ty = t / tiles_x;
        uint32_t tile_count = a->ctx->tiles[t].count;

        // Dispatch priority: MAU > CPU
        if (a->mau_ctx && a->mau_ctx->initialized &&
            tile_count >= a->mau_ctx->tile_threshold) {
            rasterize_tile_mau(a->ctx->tiles[t], a->gaussians,
                              a->fb->data, a->fb->stride,
                              tx, ty, a->fb->width, a->fb->height,
                              a->bg_color,
                              *a->mau_ctx,
                              a->mau_ctx->thread_bufs[a->thread_id]);
            a->mau_tile_count++;
        } else {
            rasterize_tile(a->ctx->tiles[t], a->gaussians,
                          a->fb->data, a->fb->stride,
                          tx, ty, a->fb->width, a->fb->height,
                          a->bg_color);
            a->cpu_tile_count++;
        }
    }
    return nullptr;
}

void gs_rasterize(const RasterContext &ctx, const ProjectedGaussian *gaussians,
                  Framebuffer &fb, uint32_t bg_color,
                  MAUContext *mau_ctx,
                  uint32_t *out_mau_tiles,
                  uint32_t *out_cpu_tiles) {
    if (!ctx.allocated || ctx.num_tiles == 0) return;

    pthread_t threads[NUM_RENDER_THREADS];
    RasterThreadArg args[NUM_RENDER_THREADS];

    uint32_t tiles_per_thread = (ctx.num_tiles + NUM_RENDER_THREADS - 1) / NUM_RENDER_THREADS;

    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        args[t].ctx = &ctx;
        args[t].gaussians = gaussians;
        args[t].fb = &fb;
        args[t].tile_start = t * tiles_per_thread;
        args[t].tile_end = std::min((t + 1) * tiles_per_thread, ctx.num_tiles);
        args[t].bg_color = bg_color;
        args[t].mau_ctx = mau_ctx;
        args[t].thread_id = t;
        args[t].mau_tile_count = 0;
        args[t].cpu_tile_count = 0;

        if (args[t].tile_start < args[t].tile_end) {
            pthread_create(&threads[t], nullptr, raster_thread_func, &args[t]);
        }
    }

    uint32_t total_mau = 0, total_cpu = 0;
    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        if (args[t].tile_start < args[t].tile_end) {
            pthread_join(threads[t], nullptr);
            total_mau += args[t].mau_tile_count;
            total_cpu += args[t].cpu_tile_count;
        }
    }

    if (out_mau_tiles) *out_mau_tiles = total_mau;
    if (out_cpu_tiles) *out_cpu_tiles = total_cpu;
}
