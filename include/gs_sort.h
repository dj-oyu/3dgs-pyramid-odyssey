#pragma once

#include "gs_types.h"

// Radix sort projected Gaussians by depth (front-to-back)
// Uses float-to-uint conversion for correct ordering
void gs_radix_sort_by_depth(ProjectedGaussian *gaussians, uint32_t count);
