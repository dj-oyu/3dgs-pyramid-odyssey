#pragma once

#include "gs_types.h"
#include "gs_projector.h"
#include "gs_rasterizer.h"
#include "gs_npu.h"
#include "gs_mau.h"
#include "gs_display.h"
#include "gs_camera.h"

struct Renderer {
    ProjectionResult projection;
    ProjectionBatch proj_batch;
    RasterContext raster;
    NPUContext npu;
    MAUContext mau;
    Framebuffer framebuffers[2];
    Framebuffer upscale_fb;
    int current_fb = 0;
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    bool needs_upscale = false;
    DisplayContext display;
    RenderStats stats = {};
    bool initialized = false;
    int max_sh_degree = MAX_SH_DEGREE;  // Cap SH evaluation (0=DC only, 3=full)
    uint32_t max_gaussians = 0;         // Cap visible Gaussians after sort (0=unlimited)
};

// Initialize renderer (display, framebuffers, raster context)
// headless=true skips display init (for dump mode without AX_SYS)
// use_npu=true enables NPU super-resolution upscale (requires trained .axmodel)
bool gs_renderer_init(Renderer &r, uint32_t width, uint32_t height, bool headless = false, bool use_npu = false);
void gs_renderer_deinit(Renderer &r);

// Render one frame: project → sort → rasterize → display
void gs_renderer_render_frame(Renderer &r, const GaussianScene &scene, const CameraParams &cam);

// Get last frame stats
const RenderStats &gs_renderer_get_stats(const Renderer &r);
