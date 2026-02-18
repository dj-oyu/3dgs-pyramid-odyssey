#pragma once

#include <cstdint>
#include <cstddef>

// Display constants
static constexpr uint32_t DISPLAY_WIDTH  = 1920;
static constexpr uint32_t DISPLAY_HEIGHT = 1080;
static constexpr uint32_t TILE_SIZE      = 16;
static constexpr uint32_t TILES_X        = (DISPLAY_WIDTH  + TILE_SIZE - 1) / TILE_SIZE;
static constexpr uint32_t TILES_Y        = (DISPLAY_HEIGHT + TILE_SIZE - 1) / TILE_SIZE;
static constexpr uint32_t NUM_TILES      = TILES_X * TILES_Y;

// Rasterizer constants
static constexpr float ALPHA_THRESHOLD   = 1.0f / 255.0f;
static constexpr float TRANSMITTANCE_MIN = 0.003f;
static constexpr uint32_t MAX_GAUSSIANS_PER_TILE = 4096;
static constexpr uint32_t NUM_RENDER_THREADS = 8;

// SH constants
static constexpr int MAX_SH_DEGREE = 3;
static constexpr int SH_COEFFS_PER_DEGREE[] = {1, 4, 9, 16};

// SH coefficient count for degrees 1-3 (excluding DC)
inline int sh_rest_count(int degree) {
    if (degree <= 0) return 0;
    return (SH_COEFFS_PER_DEGREE[degree] - 1) * 3;  // 3 color channels
}

// Structure-of-Arrays for Gaussian scene data (CMM-allocated)
struct GaussianScene {
    uint32_t num_gaussians = 0;
    int sh_degree = 0;

    // Positions
    float *pos_x = nullptr;
    float *pos_y = nullptr;
    float *pos_z = nullptr;

    // SH DC color (base color)
    float *sh_r = nullptr;
    float *sh_g = nullptr;
    float *sh_b = nullptr;

    // SH higher-order coefficients (NULL if sh_degree == 0)
    // Layout: [N * sh_rest_count(sh_degree)]
    float *sh_rest = nullptr;

    // Rotation quaternion (w, x, y, z)
    float *rot_w = nullptr;
    float *rot_x = nullptr;
    float *rot_y = nullptr;
    float *rot_z = nullptr;

    // Log-scale (pre-exp)
    float *scale_x = nullptr;
    float *scale_y = nullptr;
    float *scale_z = nullptr;

    // Opacity (post-sigmoid)
    float *opacity = nullptr;

    // Bounding box
    float bbox_min[3] = {0};
    float bbox_max[3] = {0};
};

// Per-frame projected Gaussian data
struct ProjectedGaussian {
    float screen_x, screen_y;     // 2D screen position
    float depth;                  // Depth for sorting
    float cov2d_a, cov2d_b, cov2d_c;  // 2D covariance (symmetric: [[a,b],[b,c]])
    float color_r, color_g, color_b;   // Evaluated color
    float opacity;                // Opacity
    float radius;                 // Bounding radius in pixels
    uint32_t orig_idx;            // Index into original scene arrays
};

// Tile descriptor for rasterization
struct TileGaussians {
    uint32_t *indices;        // Indices into projected array
    uint32_t count;           // Number of Gaussians in this tile
    uint32_t capacity;
};

// Framebuffer (CMM or heap allocated, ARGB8888)
struct Framebuffer {
    uint8_t *data = nullptr;  // ARGB8888 pixel data
    uint64_t phys_addr = 0;   // Physical address for VO (0 if heap-allocated)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;      // Bytes per row
    uint32_t size = 0;        // Total size in bytes
};

// Camera parameters
struct CameraParams {
    float view_matrix[16];    // 4x4 column-major
    float proj_matrix[16];    // 4x4 column-major
    float position[3];        // Camera world position
    float fov_y;              // Vertical FOV in radians
    float aspect;             // Width / Height
    float near_plane;
    float far_plane;
    uint32_t width;
    uint32_t height;
};

// Render statistics
struct RenderStats {
    uint32_t num_visible;     // Gaussians after culling
    uint32_t num_tiles_active;
    float time_project_ms;
    float time_sort_ms;
    float time_raster_ms;
    float time_total_ms;
    float fps;
    // Per-phase projection timing
    float time_cull_ms;
    float time_cov_ms;
    float time_color_ms;
    float time_assemble_ms;
    // MAU rasterizer stats
    uint32_t mau_tiles;       // Tiles processed via MAU path
    uint32_t cpu_tiles;       // Tiles processed via CPU-only path
};
