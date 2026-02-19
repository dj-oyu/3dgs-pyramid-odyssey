#pragma once

#include "gs_types.h"

struct ProjectionResult {
    ProjectedGaussian *gaussians = nullptr;  // Array of projected Gaussians
    uint32_t count = 0;                      // Number of visible Gaussians
    uint32_t capacity = 0;
};

// Allocate projection result buffers
bool gs_projection_alloc(ProjectionResult &result, uint32_t max_gaussians);
void gs_projection_free(ProjectionResult &result);

// --- Phased projection pipeline ---

enum class ProjMemType { HEAP, CMM };

// SoA intermediate buffers for phased projection pipeline.
// Single contiguous allocation for future CMM/NPU zero-copy.
struct ProjectionBatch {
    uint32_t capacity = 0;
    ProjMemType mem_type = ProjMemType::HEAP;
    void    *backing_ptr  = nullptr;
    uint64_t backing_phys = 0;       // for MAU/NPU zero-copy
    uint32_t backing_size = 0;

    // Phase 1: cull outputs
    uint32_t  cull_count = 0;
    uint32_t *visible_indices = nullptr;  // [M] original scene indices
    float *cam_x = nullptr, *cam_y = nullptr, *cam_z = nullptr;  // [M] view-space coords

    // Phase 2: covariance + screen outputs
    uint32_t cov_count = 0;
    float *screen_x = nullptr, *screen_y = nullptr;  // [M2]
    float *depth = nullptr;                            // [M2]
    float *cov2d_a = nullptr, *cov2d_b = nullptr, *cov2d_c = nullptr;  // [M2]
    float *radius = nullptr;                            // [M2]

    // Phase 3: color outputs
    float *color_r = nullptr, *color_g = nullptr, *color_b = nullptr;  // [M2]
};

// Allocate/free ProjectionBatch (single contiguous allocation, 14 floats + 1 uint32 per Gaussian)
bool gs_projection_batch_alloc(ProjectionBatch &batch, uint32_t max_gaussians);
void gs_projection_batch_free(ProjectionBatch &batch);

// Phase 1: View transform + frustum cull (NEON batch-of-4)
// Outputs: visible_indices, cam_x/y/z, cull_count
uint32_t gs_project_cull(const GaussianScene &scene, const CameraParams &cam,
                          ProjectionBatch &batch);

// Phase 2: Screen projection + covariance + eigenvalue cap + radius + screen bounds cull
// Outputs: screen_x/y, depth, cov2d_a/b/c, radius, cov_count
uint32_t gs_project_cov(const GaussianScene &scene, const CameraParams &cam,
                         ProjectionBatch &batch);

// Phase 3: SH evaluation
// Outputs: color_r/g/b
// max_sh_degree: cap SH evaluation at this degree (-1 = use scene's degree)
void gs_project_color(const GaussianScene &scene, const CameraParams &cam,
                       ProjectionBatch &batch, int max_sh_degree = -1);

// Phase 4: Pack SoA → AoS for rasterizer
void gs_project_assemble(const GaussianScene &scene, const ProjectionBatch &batch,
                          ProjectionResult &result);

// Full phased pipeline (orchestrator)
void gs_project(const GaussianScene &scene, const CameraParams &cam,
                ProjectionBatch &batch, ProjectionResult &result);

// Backward-compatible 3-arg wrapper (allocates internal batch on stack)
void gs_project(const GaussianScene &scene, const CameraParams &cam, ProjectionResult &result);
