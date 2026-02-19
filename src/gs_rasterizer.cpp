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

// Forward declarations
static void pool_deinit();
static void rasterize_tile(const TileGaussians &tile,
                           const ProjectedGaussian *gaussians,
                           uint8_t *fb_data, uint32_t fb_stride,
                           uint32_t tile_x, uint32_t tile_y,
                           uint32_t fb_width, uint32_t fb_height,
                           uint32_t bg_color);

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
    pool_deinit();
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

void gs_raster_assign_tiles(RasterContext &ctx, const ProjectedGaussian *gaussians,
                            const uint32_t *sorted_indices, uint32_t count) {
    uint32_t tiles_x = (ctx.render_width  + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tiles_y = (ctx.render_height + TILE_SIZE - 1) / TILE_SIZE;

    // Reset tile counts
    for (uint32_t i = 0; i < ctx.num_tiles; i++) {
        ctx.tiles[i].count = 0;
    }

    float inv_tile = 1.0f / (float)TILE_SIZE;
    int itiles_x = (int)tiles_x;
    int itiles_y = (int)tiles_y;

    for (uint32_t si = 0; si < count; si++) {
        // Prefetch Gaussian data 4 elements ahead (hide DRAM latency for indirect access)
        if (si + 4 < count)
            __builtin_prefetch(&gaussians[sorted_indices[si + 4]], 0, 0);

        uint32_t gi = sorted_indices[si];
        const ProjectedGaussian &pg = gaussians[gi];

        // Early reject tiny/transparent Gaussians
        if (pg.radius < 0.5f) continue;
        if (pg.opacity < ALPHA_THRESHOLD) continue;

        // Compute tile range (integer math where possible)
        float rad = pg.radius;
        int min_tx = std::max(0, (int)((pg.screen_x - rad) * inv_tile));
        int max_tx = std::min(itiles_x - 1, (int)((pg.screen_x + rad) * inv_tile));
        int min_ty = std::max(0, (int)((pg.screen_y - rad) * inv_tile));
        int max_ty = std::min(itiles_y - 1, (int)((pg.screen_y + rad) * inv_tile));

        for (int ty = min_ty; ty <= max_ty; ty++) {
            uint32_t row_base = ty * tiles_x;
            for (int tx = min_tx; tx <= max_tx; tx++) {
                TileGaussians &tile = ctx.tiles[row_base + tx];

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

                // Store original Gaussian index (not sorted position)
                tile.indices[tile.count++] = gi;
            }
        }
    }
}

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

        // ha/hb/hc and color*opacity precomputed in assembly; det>0 guaranteed
        float alpha = pg.opacity;
        float cr = pg.color_r;  // already pre-multiplied
        float cg = pg.color_g;
        float cb = pg.color_b;

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

            // ha/hb/hc and color*opacity precomputed in assembly
            float ha = pg.cov2d_a;
            float hb = pg.cov2d_b;
            float hc = pg.cov2d_c;

            float alpha = pg.opacity;
            float cr = pg.color_r;  // already pre-multiplied
            float cg = pg.color_g;
            float cb = pg.color_b;

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

// Persistent thread pool — eliminates pthread_create/join overhead per frame
struct RasterThreadPool {
    pthread_t threads[NUM_RENDER_THREADS];
    pthread_barrier_t start_barrier;  // count = workers + main
    pthread_barrier_t end_barrier;    // count = workers + main
    volatile bool shutdown;

    // Per-frame shared state (set by main before start_barrier)
    const RasterContext *ctx;
    const ProjectedGaussian *gaussians;
    Framebuffer *fb;
    uint32_t bg_color;
    MAUContext *mau_ctx;
    volatile uint32_t next_tile;
    uint32_t total_tiles;

    // Per-thread output
    uint32_t mau_tile_counts[NUM_RENDER_THREADS];
    uint32_t cpu_tile_counts[NUM_RENDER_THREADS];

    bool initialized;
};

static RasterThreadPool g_pool = {};

static void *pool_worker_func(void *arg) {
    uint32_t tid = (uint32_t)(uintptr_t)arg;

    while (true) {
        pthread_barrier_wait(&g_pool.start_barrier);
        if (g_pool.shutdown) return nullptr;

        uint32_t tiles_x = (g_pool.ctx->render_width + TILE_SIZE - 1) / TILE_SIZE;
        g_pool.mau_tile_counts[tid] = 0;
        g_pool.cpu_tile_counts[tid] = 0;

        uint32_t t;
        while ((t = __sync_fetch_and_add(&g_pool.next_tile, 1)) < g_pool.total_tiles) {
            if (g_pool.ctx->tiles[t].count == 0) {
                // Fill empty tile with background color
                uint32_t tx = t % tiles_x;
                uint32_t ty = t / tiles_x;
                uint32_t px_start = tx * TILE_SIZE;
                uint32_t py_start = ty * TILE_SIZE;
                uint32_t px_end = std::min(px_start + TILE_SIZE, g_pool.fb->width);
                uint32_t py_end = std::min(py_start + TILE_SIZE, g_pool.fb->height);

                uint32x4_t bg_vec = vdupq_n_u32(g_pool.bg_color);
                for (uint32_t py = py_start; py < py_end; py++) {
                    uint32_t *row = reinterpret_cast<uint32_t*>(g_pool.fb->data + py * g_pool.fb->stride);
                    uint32_t px = px_start;
                    for (; px + 4 <= px_end; px += 4) vst1q_u32(&row[px], bg_vec);
                    for (; px < px_end; px++) row[px] = g_pool.bg_color;
                }
                continue;
            }

            uint32_t tx = t % tiles_x;
            uint32_t ty = t / tiles_x;
            uint32_t tile_count = g_pool.ctx->tiles[t].count;

            if (g_pool.mau_ctx && g_pool.mau_ctx->initialized &&
                tile_count >= g_pool.mau_ctx->tile_threshold) {
                rasterize_tile_mau(g_pool.ctx->tiles[t], g_pool.gaussians,
                                  g_pool.fb->data, g_pool.fb->stride,
                                  tx, ty, g_pool.fb->width, g_pool.fb->height,
                                  g_pool.bg_color,
                                  *g_pool.mau_ctx,
                                  g_pool.mau_ctx->thread_bufs[tid]);
                g_pool.mau_tile_counts[tid]++;
            } else {
                rasterize_tile(g_pool.ctx->tiles[t], g_pool.gaussians,
                              g_pool.fb->data, g_pool.fb->stride,
                              tx, ty, g_pool.fb->width, g_pool.fb->height,
                              g_pool.bg_color);
                g_pool.cpu_tile_counts[tid]++;
            }
        }

        pthread_barrier_wait(&g_pool.end_barrier);
    }
    return nullptr;
}

static void pool_init() {
    pthread_barrier_init(&g_pool.start_barrier, nullptr, NUM_RENDER_THREADS + 1);
    pthread_barrier_init(&g_pool.end_barrier, nullptr, NUM_RENDER_THREADS + 1);
    g_pool.shutdown = false;

    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        pthread_create(&g_pool.threads[t], nullptr, pool_worker_func, (void*)(uintptr_t)t);
    }
    g_pool.initialized = true;
}

static void pool_deinit() {
    if (!g_pool.initialized) return;
    g_pool.shutdown = true;
    pthread_barrier_wait(&g_pool.start_barrier);  // release workers to see shutdown
    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        pthread_join(g_pool.threads[t], nullptr);
    }
    pthread_barrier_destroy(&g_pool.start_barrier);
    pthread_barrier_destroy(&g_pool.end_barrier);
    g_pool.initialized = false;
}

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

    // FP16 tile accumulators (half memory, better cache utilization)
    __fp16 tile_r[TILE_SIZE][TILE_SIZE] = {};
    __fp16 tile_g[TILE_SIZE][TILE_SIZE] = {};
    __fp16 tile_b[TILE_SIZE][TILE_SIZE] = {};
    __fp16 tile_T[TILE_SIZE][TILE_SIZE];

    // Tile center for Opt-R2 pre-test
    float tcx = (float)px_start + TILE_SIZE * 0.5f;
    float tcy = (float)py_start + TILE_SIZE * 0.5f;
    static constexpr float TILE_MARGIN = TILE_SIZE * 0.7071f;  // sqrt(2)/2 * TILE_SIZE

    // FP32 NEON constants (Tier 2 fallback + framebuffer write)
    float32x4_t v_trans_min = vdupq_n_f32(TRANSMITTANCE_MIN);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_min_alpha = vdupq_n_f32(ALPHA_THRESHOLD);
    float32x4_t v_power_max = vdupq_n_f32(4.0f);
    float32x4_t v_step4 = {0.0f, 1.0f, 2.0f, 3.0f};
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // FP16 NEON constants (Tier 1 main loop)
    float16x8_t v_trans_min_h = vdupq_n_f16((__fp16)TRANSMITTANCE_MIN);
    float16x8_t v_one_h = vdupq_n_f16((__fp16)1.0f);
    float16x8_t v_min_alpha_h = vdupq_n_f16((__fp16)ALPHA_THRESHOLD);
    float16x8_t v_power_max_h = vdupq_n_f16((__fp16)4.0f);
    float16x8_t v_step8_h = {(__fp16)0.0f, (__fp16)1.0f, (__fp16)2.0f, (__fp16)3.0f,
                              (__fp16)4.0f, (__fp16)5.0f, (__fp16)6.0f, (__fp16)7.0f};
    // NEON-init tile_T to 1.0
    {
        float16x8_t v_init = vdupq_n_f16((__fp16)1.0f);
        for (uint32_t y = 0; y < TILE_SIZE; y++)
            for (uint32_t x = 0; x < TILE_SIZE; x += 8)
                vst1q_f16(&tile_T[y][x], v_init);
    }
#else
    for (uint32_t y = 0; y < TILE_SIZE; y++)
        for (uint32_t x = 0; x < TILE_SIZE; x++)
            tile_T[y][x] = (__fp16)1.0f;
#endif

    // Process Gaussians front-to-back (already sorted)
    for (uint32_t gi = 0; gi < tile.count; gi++) {
        // Prefetch next Gaussian data to hide memory latency
        if (gi + 1 < tile.count)
            __builtin_prefetch(&gaussians[tile.indices[gi + 1]], 0, 1);

        const ProjectedGaussian &pg = gaussians[tile.indices[gi]];

        // ha/hb/hc and color*opacity are precomputed in assembly phase
        float ha = pg.cov2d_a;
        float hb = pg.cov2d_b;
        float hc = pg.cov2d_c;

        // Opt-R2: Tile-level Gaussian pre-test
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
        float cr = pg.color_r;  // already pre-multiplied by opacity
        float cg = pg.color_g;
        float cb = pg.color_b;

        // FP32 per-Gaussian vectors (Tier 2 fallback)
        float32x4_t v_cr = vdupq_n_f32(cr);
        float32x4_t v_cg = vdupq_n_f32(cg);
        float32x4_t v_cb = vdupq_n_f32(cb);
        float32x4_t v_alpha = vdupq_n_f32(alpha);
        float32x4_t v_ha = vdupq_n_f32(ha);

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        // FP16 per-Gaussian vectors (Tier 1)
        float16x8_t v_ha_h = vdupq_n_f16((__fp16)ha);
        float16x8_t v_cr_h = vdupq_n_f16((__fp16)cr);
        float16x8_t v_cg_h = vdupq_n_f16((__fp16)cg);
        float16x8_t v_cb_h = vdupq_n_f16((__fp16)cb);
        float16x8_t v_alpha_h = vdupq_n_f16((__fp16)alpha);
#endif

        for (uint32_t py = g_py0; py < g_py1; py++) {
            float dy = (float)py + 0.5f - pg.screen_y;
            float dy_sq_term = hc * dy * dy;
            float dy_cross_term = hb * dy;
            uint32_t ly = py - py_start;

            uint32_t px = g_px0;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            // Tier 1: Full FP16 — Mahalanobis, exp, AND color accumulation
            float16x8_t v_dy_sq_h = vdupq_n_f16((__fp16)dy_sq_term);
            float16x8_t v_dy_cross_h = vdupq_n_f16((__fp16)dy_cross_term);

            for (; px + 8 <= g_px1; px += 8) {
                uint32_t lx = px - px_start;

                // Load transmittance (single FP16 load for 8 pixels)
                float16x8_t v_T_h = vld1q_f16(&tile_T[ly][lx]);
                uint16x8_t mask_alive_h = vcgtq_f16(v_T_h, v_trans_min_h);
                if (vmaxvq_u16(mask_alive_h) == 0) continue;

                // Mahalanobis in FP16
                float16x8_t v_dx_h = vaddq_f16(
                    vdupq_n_f16((__fp16)((float)px + 0.5f - pg.screen_x)), v_step8_h);
                float16x8_t v_power_h = vmulq_f16(v_ha_h, vmulq_f16(v_dx_h, v_dx_h));
                v_power_h = vfmaq_f16(v_power_h, v_dy_cross_h, v_dx_h);
                v_power_h = vaddq_f16(v_power_h, v_dy_sq_h);

                uint16x8_t mask_power_h = vcltq_f16(v_power_h, v_power_max_h);
                if (vmaxvq_u16(mask_power_h) == 0) continue;

                // Exp in FP16
                float16x8_t v_gauss_h = neon_fast_exp_neg_f16(v_power_h);

                // w = gauss * alpha (FP16, no conversion)
                float16x8_t v_w_h = vmulq_f16(v_gauss_h, v_alpha_h);

                // Visibility mask (FP16)
                uint16x8_t mask_vis_h = vandq_u16(
                    vandq_u16(vcgtq_f16(v_w_h, v_min_alpha_h), mask_power_h), mask_alive_h);

                // Contrib = T * gauss (masked, FP16)
                float16x8_t v_contrib_h = vreinterpretq_f16_u16(
                    vandq_u16(vreinterpretq_u16_f16(vmulq_f16(v_T_h, v_gauss_h)), mask_vis_h));

                // Color accumulation (FP16, 8 pixels per channel, single load+fma+store)
                vst1q_f16(&tile_r[ly][lx], vfmaq_f16(vld1q_f16(&tile_r[ly][lx]), v_contrib_h, v_cr_h));
                vst1q_f16(&tile_g[ly][lx], vfmaq_f16(vld1q_f16(&tile_g[ly][lx]), v_contrib_h, v_cg_h));
                vst1q_f16(&tile_b[ly][lx], vfmaq_f16(vld1q_f16(&tile_b[ly][lx]), v_contrib_h, v_cb_h));

                // T update (FP16)
                float16x8_t v_omw_h = vsubq_f16(v_one_h, v_w_h);
                v_omw_h = vreinterpretq_f16_u16(vbslq_u16(mask_vis_h,
                    vreinterpretq_u16_f16(v_omw_h), vreinterpretq_u16_f16(v_one_h)));
                vst1q_f16(&tile_T[ly][lx], vmulq_f16(v_T_h, v_omw_h));
            }
#endif

            // Tier 2: FP32 computation with FP16 tile load/store (remainder 4-7 pixels)
            float32x4_t v_dy_sq = vdupq_n_f32(dy_sq_term);
            float32x4_t v_dy_cross = vdupq_n_f32(dy_cross_term);

            for (; px + 4 <= g_px1; px += 4) {
                uint32_t lx = px - px_start;

                float32x4_t v_T = vcvt_f32_f16(vld1_f16(&tile_T[ly][lx]));
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

                // Load FP16 → FP32, accumulate, FP32 → FP16 store
                float32x4_t v_r = vcvt_f32_f16(vld1_f16(&tile_r[ly][lx]));
                float32x4_t v_g = vcvt_f32_f16(vld1_f16(&tile_g[ly][lx]));
                float32x4_t v_b = vcvt_f32_f16(vld1_f16(&tile_b[ly][lx]));

                vst1_f16(&tile_r[ly][lx], vcvt_f16_f32(vmlaq_f32(v_r, v_contrib, v_cr)));
                vst1_f16(&tile_g[ly][lx], vcvt_f16_f32(vmlaq_f32(v_g, v_contrib, v_cg)));
                vst1_f16(&tile_b[ly][lx], vcvt_f16_f32(vmlaq_f32(v_b, v_contrib, v_cb)));

                float32x4_t v_one_minus_w = vsubq_f32(v_one, v_w);
                v_one_minus_w = vbslq_f32(mask_visible, v_one_minus_w, v_one);
                vst1_f16(&tile_T[ly][lx], vcvt_f16_f32(vmulq_f32(v_T, v_one_minus_w)));
            }

            // Tier 3: Scalar tail (0-3 pixels, FP16 tile access)
            for (; px < g_px1; px++) {
                uint32_t lx = px - px_start;
                float T = (float)tile_T[ly][lx];
                if (T < TRANSMITTANCE_MIN) continue;

                float dx = (float)px + 0.5f - pg.screen_x;
                float power = ha*dx*dx + hb*dx*dy + hc*dy*dy;
                if (power > 4.0f) continue;

                float gauss = expf(-power);
                float w = gauss * alpha;
                if (w < ALPHA_THRESHOLD) continue;

                float contrib = T * gauss;
                tile_r[ly][lx] = (__fp16)((float)tile_r[ly][lx] + contrib * cr);
                tile_g[ly][lx] = (__fp16)((float)tile_g[ly][lx] + contrib * cg);
                tile_b[ly][lx] = (__fp16)((float)tile_b[ly][lx] + contrib * cb);
                tile_T[ly][lx] = (__fp16)(T * (1.0f - w));
            }
        }

        // Opt-R4: Tile-level transmittance saturation check (every 4 Gaussians)
        if ((gi & 3u) == 3u) {
            bool tile_saturated = true;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            for (uint32_t y = 0; y < TILE_SIZE && tile_saturated; y++)
                for (uint32_t x = 0; x < TILE_SIZE; x += 8) {
                    float16x8_t v_T_h = vld1q_f16(&tile_T[y][x]);
                    if (vmaxvq_u16(vcgeq_f16(v_T_h, v_trans_min_h)) != 0) {
                        tile_saturated = false;
                        break;
                    }
                }
#else
            for (uint32_t y = 0; y < TILE_SIZE && tile_saturated; y++)
                for (uint32_t x = 0; x < TILE_SIZE; x += 4) {
                    float32x4_t v_T = vcvt_f32_f16(vld1_f16(&tile_T[y][x]));
                    if (vmaxvq_u32(vcgeq_f32(v_T, vdupq_n_f32(TRANSMITTANCE_MIN))) != 0) {
                        tile_saturated = false;
                        break;
                    }
                }
#endif
            if (tile_saturated) break;
        }
    }

    // Write tile to framebuffer (FP16 → FP32 → ARGB8888)
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

            // FP16 → FP32 conversion
            float32x4_t v_T = vcvt_f32_f16(vld1_f16(&tile_T[ly][lx]));
            float32x4_t v_r = vcvt_f32_f16(vld1_f16(&tile_r[ly][lx]));
            float32x4_t v_g = vcvt_f32_f16(vld1_f16(&tile_g[ly][lx]));
            float32x4_t v_b = vcvt_f32_f16(vld1_f16(&tile_b[ly][lx]));

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
            float T = (float)tile_T[ly][lx];

            float r = (float)tile_r[ly][lx] + T * ((float)((bg_color >> 16) & 0xFF) / 255.0f);
            float g = (float)tile_g[ly][lx] + T * ((float)((bg_color >> 8)  & 0xFF) / 255.0f);
            float b = (float)tile_b[ly][lx] + T * ((float)( bg_color        & 0xFF) / 255.0f);

            uint8_t ri = (uint8_t)std::min(255.0f, std::max(0.0f, r * 255.0f));
            uint8_t gi_val = (uint8_t)std::min(255.0f, std::max(0.0f, g * 255.0f));
            uint8_t bi = (uint8_t)std::min(255.0f, std::max(0.0f, b * 255.0f));

            row[px] = (0xFFu << 24) | ((uint32_t)ri << 16) | ((uint32_t)gi_val << 8) | bi;
        }
    }
}

void gs_rasterize(const RasterContext &ctx, const ProjectedGaussian *gaussians,
                  Framebuffer &fb, uint32_t bg_color,
                  MAUContext *mau_ctx,
                  uint32_t *out_mau_tiles,
                  uint32_t *out_cpu_tiles) {
    if (!ctx.allocated || ctx.num_tiles == 0) return;

    // Lazy-init persistent thread pool
    if (!g_pool.initialized) pool_init();

    // Set per-frame shared state
    g_pool.ctx = &ctx;
    g_pool.gaussians = gaussians;
    g_pool.fb = &fb;
    g_pool.bg_color = bg_color;
    g_pool.mau_ctx = mau_ctx;
    g_pool.next_tile = 0;
    g_pool.total_tiles = ctx.num_tiles;

    // Signal workers to start and wait for completion
    pthread_barrier_wait(&g_pool.start_barrier);
    pthread_barrier_wait(&g_pool.end_barrier);

    // Collect results
    uint32_t total_mau = 0, total_cpu = 0;
    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        total_mau += g_pool.mau_tile_counts[t];
        total_cpu += g_pool.cpu_tile_counts[t];
    }

    if (out_mau_tiles) *out_mau_tiles = total_mau;
    if (out_cpu_tiles) *out_cpu_tiles = total_cpu;
}
