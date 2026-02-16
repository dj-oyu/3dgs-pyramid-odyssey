#include "gs_rasterizer.h"
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

// Per-thread rasterization work
struct RasterThreadArg {
    const RasterContext *ctx;
    const ProjectedGaussian *gaussians;
    Framebuffer *fb;
    uint32_t tile_start;
    uint32_t tile_end;
    uint32_t bg_color;
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

    // Process Gaussians front-to-back (already sorted)
    for (uint32_t gi = 0; gi < tile.count; gi++) {
        const ProjectedGaussian &pg = gaussians[tile.indices[gi]];

        // Precompute inverse covariance
        float det = pg.cov2d_a * pg.cov2d_c - pg.cov2d_b * pg.cov2d_b;
        if (det <= 0.0f) continue;
        float inv_det = 1.0f / det;
        float inv_a =  pg.cov2d_c * inv_det;
        float inv_b = -pg.cov2d_b * inv_det;
        float inv_c =  pg.cov2d_a * inv_det;

        // Clamp pixel range to Gaussian bounding box within tile
        float rad = pg.radius;
        uint32_t g_px0 = (uint32_t)std::max((int)px_start, (int)floorf(pg.screen_x - rad));
        uint32_t g_px1 = std::min(px_end, (uint32_t)ceilf(pg.screen_x + rad));
        uint32_t g_py0 = (uint32_t)std::max((int)py_start, (int)floorf(pg.screen_y - rad));
        uint32_t g_py1 = std::min(py_end, (uint32_t)ceilf(pg.screen_y + rad));
        if (g_px0 >= g_px1 || g_py0 >= g_py1) continue;

        // Align g_px0 down to 4-pixel boundary (relative to px_start) for NEON
        uint32_t lx0_aligned = ((g_px0 - px_start) & ~3u);
        g_px0 = px_start + lx0_aligned;

        float alpha = pg.opacity;
        float cr = pg.color_r * alpha;
        float cg = pg.color_g * alpha;
        float cb = pg.color_b * alpha;

        // NEON vectors
        float32x4_t v_cr = vdupq_n_f32(cr);
        float32x4_t v_cg = vdupq_n_f32(cg);
        float32x4_t v_cb = vdupq_n_f32(cb);
        float32x4_t v_alpha = vdupq_n_f32(alpha);
        float32x4_t v_half = vdupq_n_f32(0.5f);
        float32x4_t v_trans_min = vdupq_n_f32(TRANSMITTANCE_MIN);
        float32x4_t v_one = vdupq_n_f32(1.0f);
        float32x4_t v_min_alpha = vdupq_n_f32(ALPHA_THRESHOLD);
        float32x4_t v_inv_a = vdupq_n_f32(inv_a);
        float32x4_t v_step = {0.0f, 1.0f, 2.0f, 3.0f};

        for (uint32_t py = g_py0; py < g_py1; py++) {
            float dy = (float)py + 0.5f - pg.screen_y;
            float dy_sq_term = inv_c * dy * dy;
            float dy_cross_term = 2.0f * inv_b * dy;
            float32x4_t v_dy_sq = vdupq_n_f32(dy_sq_term);
            float32x4_t v_dy_cross = vdupq_n_f32(dy_cross_term);
            uint32_t ly = py - py_start;

            // NEON path: process 4 pixels at a time
            uint32_t px = g_px0;
            for (; px + 4 <= g_px1; px += 4) {
                uint32_t lx = px - px_start;

                // Load transmittances
                float32x4_t v_T = vld1q_f32(&tile_T[ly][lx]);

                // Check if all transmittances are below threshold
                uint32x4_t mask_alive = vcgtq_f32(v_T, v_trans_min);
                if (vmaxvq_u32(mask_alive) == 0) continue;

                // Compute dx for 4 pixels
                float base_dx = (float)px + 0.5f - pg.screen_x;
                float32x4_t v_dx = vaddq_f32(vdupq_n_f32(base_dx), v_step);

                // Mahalanobis distance: 0.5 * (inv_a*dx^2 + 2*inv_b*dx*dy + inv_c*dy^2)
                float32x4_t v_power = vmulq_f32(v_inv_a, vmulq_f32(v_dx, v_dx));
                v_power = vmlaq_f32(v_power, v_dy_cross, v_dx);
                v_power = vaddq_f32(v_power, v_dy_sq);
                v_power = vmulq_f32(v_half, v_power);

                // Gaussian weight: exp(-power) * alpha
                float32x4_t v_gauss = neon_fast_exp_neg(v_power);
                float32x4_t v_w = vmulq_f32(v_gauss, v_alpha);

                // Visibility mask
                uint32x4_t mask_visible = vandq_u32(vcgtq_f32(v_w, v_min_alpha), mask_alive);

                // Apply: color += T * gauss * (color * opacity)
                float32x4_t v_contrib = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(vmulq_f32(v_T, v_gauss)), mask_visible));

                float32x4_t v_r = vld1q_f32(&tile_r[ly][lx]);
                float32x4_t v_g = vld1q_f32(&tile_g[ly][lx]);
                float32x4_t v_b = vld1q_f32(&tile_b[ly][lx]);

                vst1q_f32(&tile_r[ly][lx], vmlaq_f32(v_r, v_contrib, v_cr));
                vst1q_f32(&tile_g[ly][lx], vmlaq_f32(v_g, v_contrib, v_cg));
                vst1q_f32(&tile_b[ly][lx], vmlaq_f32(v_b, v_contrib, v_cb));

                // Update transmittance: T *= (1 - w) where visible, T unchanged otherwise
                float32x4_t v_one_minus_w = vsubq_f32(v_one, v_w);
                v_one_minus_w = vbslq_f32(mask_visible, v_one_minus_w, v_one);
                vst1q_f32(&tile_T[ly][lx], vmulq_f32(v_T, v_one_minus_w));
            }

            // Handle remaining pixels scalar
            for (; px < g_px1; px++) {
                uint32_t lx = px - px_start;
                float T = tile_T[ly][lx];
                if (T < TRANSMITTANCE_MIN) continue;

                float dx = (float)px + 0.5f - pg.screen_x;
                float power = 0.5f * (inv_a*dx*dx + 2.0f*inv_b*dx*dy + inv_c*dy*dy);
                if (power > 7.0f) continue;

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

    // Write tile to framebuffer (ARGB8888)
    uint8_t bg_r = (bg_color >> 16) & 0xFF;
    uint8_t bg_g = (bg_color >> 8)  & 0xFF;
    uint8_t bg_b =  bg_color        & 0xFF;

    for (uint32_t py = py_start; py < py_end; py++) {
        uint32_t ly = py - py_start;
        uint32_t *row = reinterpret_cast<uint32_t*>(fb_data + py * fb_stride);

        for (uint32_t px = px_start; px < px_end; px++) {
            uint32_t lx = px - px_start;
            float T = tile_T[ly][lx];

            // Add background contribution
            float r = tile_r[ly][lx] + T * (bg_r / 255.0f);
            float g = tile_g[ly][lx] + T * (bg_g / 255.0f);
            float b = tile_b[ly][lx] + T * (bg_b / 255.0f);

            // Convert to 8-bit
            uint8_t ri = (uint8_t)std::min(255.0f, std::max(0.0f, r * 255.0f));
            uint8_t gi = (uint8_t)std::min(255.0f, std::max(0.0f, g * 255.0f));
            uint8_t bi = (uint8_t)std::min(255.0f, std::max(0.0f, b * 255.0f));

            // ARGB8888: Alpha=0xFF
            row[px] = (0xFFu << 24) | ((uint32_t)ri << 16) | ((uint32_t)gi << 8) | bi;
        }
    }
}

static void *raster_thread_func(void *arg) {
    auto *a = static_cast<RasterThreadArg*>(arg);
    uint32_t tiles_x = (a->ctx->render_width + TILE_SIZE - 1) / TILE_SIZE;

    for (uint32_t t = a->tile_start; t < a->tile_end; t++) {
        if (a->ctx->tiles[t].count == 0) {
            // Fill empty tile with background color
            uint32_t tx = t % tiles_x;
            uint32_t ty = t / tiles_x;
            uint32_t px_start = tx * TILE_SIZE;
            uint32_t py_start = ty * TILE_SIZE;
            uint32_t px_end = std::min(px_start + TILE_SIZE, a->fb->width);
            uint32_t py_end = std::min(py_start + TILE_SIZE, a->fb->height);

            for (uint32_t py = py_start; py < py_end; py++) {
                uint32_t *row = reinterpret_cast<uint32_t*>(a->fb->data + py * a->fb->stride);
                for (uint32_t px = px_start; px < px_end; px++) {
                    row[px] = a->bg_color;
                }
            }
            continue;
        }

        uint32_t tx = t % tiles_x;
        uint32_t ty = t / tiles_x;
        rasterize_tile(a->ctx->tiles[t], a->gaussians,
                       a->fb->data, a->fb->stride,
                       tx, ty, a->fb->width, a->fb->height,
                       a->bg_color);
    }
    return nullptr;
}

void gs_rasterize(const RasterContext &ctx, const ProjectedGaussian *gaussians,
                  Framebuffer &fb, uint32_t bg_color) {
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

        if (args[t].tile_start < args[t].tile_end) {
            pthread_create(&threads[t], nullptr, raster_thread_func, &args[t]);
        }
    }

    for (uint32_t t = 0; t < NUM_RENDER_THREADS; t++) {
        if (args[t].tile_start < args[t].tile_end) {
            pthread_join(threads[t], nullptr);
        }
    }
}
