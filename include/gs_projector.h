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

// Project all Gaussians: view transform, frustum cull, 2D covariance, SH evaluation
void gs_project(const GaussianScene &scene, const CameraParams &cam, ProjectionResult &result);
