#include "gs_renderer.h"
#include "gs_sort.h"
#include <cstdio>
#include <cstring>
#include <ctime>

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

bool gs_renderer_init(Renderer &r, uint32_t width, uint32_t height, bool headless) {
    if (!headless) {
        // Initialize display (uses its own default resolution from DisplayContext)
        if (!gs_display_init(r.display)) {
            fprintf(stderr, "[gs_renderer] Display init failed\n");
            return false;
        }
    }

    r.render_width = width;
    r.render_height = height;

    // Allocate double framebuffers at render resolution
    // Use cached memory (need_phys=false) for headless or FB backend
    bool need_phys = !headless && (r.display.backend == DisplayBackend::VO);
    for (int i = 0; i < 2; i++) {
        if (!gs_framebuffer_alloc(r.framebuffers[i], width, height, need_phys)) {
            fprintf(stderr, "[gs_renderer] Framebuffer %d alloc failed\n", i);
            if (!headless) gs_display_deinit(r.display);
            return false;
        }
    }
    r.current_fb = 0;

    if (!headless && width != r.display.width && height != r.display.height) {
        r.needs_upscale = true;
        printf("[gs_renderer] Upscale: %ux%u -> %ux%u (in display layer)\n",
               width, height, r.display.width, r.display.height);
    }

    // Allocate raster context
    if (!gs_raster_alloc(r.raster, width, height)) {
        fprintf(stderr, "[gs_renderer] Raster alloc failed\n");
        gs_framebuffer_free(r.framebuffers[0]);
        gs_framebuffer_free(r.framebuffers[1]);
        gs_display_deinit(r.display);
        return false;
    }

    // Initialize MAU acceleration (non-fatal if unavailable)
    if (!gs_mau_init(r.mau)) {
        printf("[gs_renderer] MAU unavailable, using CPU-only rasterizer\n");
    }

    r.initialized = true;
    printf("[gs_renderer] Renderer initialized: %ux%u\n", width, height);
    return true;
}

void gs_renderer_deinit(Renderer &r) {
    if (!r.initialized) return;

    gs_mau_deinit(r.mau);
    gs_projection_free(r.projection);
    gs_projection_batch_free(r.proj_batch);
    gs_raster_free(r.raster);
    gs_framebuffer_free(r.framebuffers[0]);
    gs_framebuffer_free(r.framebuffers[1]);
    gs_display_deinit(r.display);

    r.initialized = false;
    printf("[gs_renderer] Renderer deinitialized\n");
}


void gs_renderer_render_frame(Renderer &r, const GaussianScene &scene, const CameraParams &cam) {
    if (!r.initialized) return;

    double t_start = get_time_ms();

    // Lazy-allocate projection buffer and batch
    if (!r.projection.gaussians) {
        if (!gs_projection_alloc(r.projection, scene.num_gaussians)) {
            fprintf(stderr, "[gs_renderer] Projection alloc failed (%u gaussians)\n", scene.num_gaussians);
            return;
        }
    }
    if (!r.proj_batch.backing_ptr) {
        if (!gs_projection_batch_alloc(r.proj_batch, scene.num_gaussians)) {
            fprintf(stderr, "[gs_renderer] ProjectionBatch alloc failed (%u gaussians)\n", scene.num_gaussians);
            return;
        }
    }

    // 1. Project (phased pipeline with per-phase timing)
    double t_proj_start = get_time_ms();

    double t_cull_start = get_time_ms();
    gs_project_cull(scene, cam, r.proj_batch);
    double t_cull_end = get_time_ms();

    double t_cov_start = get_time_ms();
    if (r.proj_batch.cull_count > 0)
        gs_project_cov(scene, cam, r.proj_batch);
    double t_cov_end = get_time_ms();

    double t_color_start = get_time_ms();
    if (r.proj_batch.cov_count > 0)
        gs_project_color(scene, cam, r.proj_batch);
    double t_color_end = get_time_ms();

    double t_asm_start = get_time_ms();
    gs_project_assemble(scene, r.proj_batch, r.projection);
    double t_asm_end = get_time_ms();

    double t_proj_end = get_time_ms();

    // 2. Sort by depth (front-to-back)
    double t_sort_start = get_time_ms();
    gs_radix_sort_by_depth(r.projection.gaussians, r.projection.count);
    double t_sort_end = get_time_ms();

    // 3. Assign to tiles
    gs_raster_assign_tiles(r.raster, r.projection.gaussians, r.projection.count);

    // 4. Rasterize into current framebuffer
    double t_raster_start = get_time_ms();
    Framebuffer &fb = r.framebuffers[r.current_fb];
    uint32_t mau_tiles = 0, cpu_tiles = 0;
    gs_rasterize(r.raster, r.projection.gaussians, fb, 0xFF000000,
                 r.mau.initialized ? &r.mau : nullptr,
                 &mau_tiles, &cpu_tiles);
    double t_raster_end = get_time_ms();

    // 5. Send to display (upscale handled by display layer if needed)
    gs_display_send_frame(r.display, fb);

    // Swap framebuffers
    r.current_fb = 1 - r.current_fb;

    double t_end = get_time_ms();

    // Update stats
    r.stats.num_visible = r.projection.count;
    r.stats.time_project_ms = (float)(t_proj_end - t_proj_start);
    r.stats.time_sort_ms = (float)(t_sort_end - t_sort_start);
    r.stats.time_raster_ms = (float)(t_raster_end - t_raster_start);
    r.stats.time_total_ms = (float)(t_end - t_start);
    r.stats.fps = (r.stats.time_total_ms > 0) ? 1000.0f / r.stats.time_total_ms : 0.0f;
    r.stats.time_cull_ms = (float)(t_cull_end - t_cull_start);
    r.stats.time_cov_ms = (float)(t_cov_end - t_cov_start);
    r.stats.time_color_ms = (float)(t_color_end - t_color_start);
    r.stats.time_assemble_ms = (float)(t_asm_end - t_asm_start);
    r.stats.mau_tiles = mau_tiles;
    r.stats.cpu_tiles = cpu_tiles;

    // Count active tiles
    uint32_t active = 0;
    for (uint32_t i = 0; i < r.raster.num_tiles; i++) {
        if (r.raster.tiles[i].count > 0) active++;
    }
    r.stats.num_tiles_active = active;
}

const RenderStats &gs_renderer_get_stats(const Renderer &r) {
    return r.stats;
}
