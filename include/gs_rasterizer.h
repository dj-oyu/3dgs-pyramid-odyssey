#pragma once

#include "gs_types.h"
#include "gs_projector.h"

struct MAUContext;  // Forward declaration

struct RasterContext {
    TileGaussians *tiles = nullptr;
    uint32_t num_tiles = 0;
    uint32_t render_width = DISPLAY_WIDTH;
    uint32_t render_height = DISPLAY_HEIGHT;
    bool allocated = false;
};

// Allocate tile structures
bool gs_raster_alloc(RasterContext &ctx, uint32_t width, uint32_t height);
void gs_raster_free(RasterContext &ctx);

// Assign projected Gaussians to tiles
// sorted_indices: indices into gaussians[] in depth order (from gs_sort_by_depth)
void gs_raster_assign_tiles(RasterContext &ctx, const ProjectedGaussian *gaussians,
                            const uint32_t *sorted_indices, uint32_t count);

// Rasterize tiles into framebuffer (multi-threaded, NEON alpha compositing)
// mau_ctx: optional MAU acceleration context (nullptr to skip)
// Dispatch priority: MAU > CPU
void gs_rasterize(const RasterContext &ctx, const ProjectedGaussian *gaussians,
                  Framebuffer &fb, uint32_t bg_color = 0xFF000000,
                  MAUContext *mau_ctx = nullptr,
                  uint32_t *out_mau_tiles = nullptr,
                  uint32_t *out_cpu_tiles = nullptr);
