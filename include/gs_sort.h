#pragma once

#include "gs_types.h"

// Pre-allocated sort buffers (avoid per-frame malloc/free)
struct SortContext {
    uint32_t capacity = 0;
    // Radix sort working space (reused as 32-bit or 64-bit pairs)
    uint64_t *pairs = nullptr;
    uint64_t *pairs_temp = nullptr;
    // Output: sorted indices into the original gaussians array
    uint32_t *sorted_indices = nullptr;
};

bool gs_sort_alloc(SortContext &ctx, uint32_t max_count);
void gs_sort_free(SortContext &ctx);

// Radix sort by depth — produces sorted_indices[] (no physical reorder)
// After return, ctx.sorted_indices[0..count-1] are indices into gaussians[]
// in front-to-back depth order.
void gs_sort_by_depth(SortContext &ctx, const ProjectedGaussian *gaussians, uint32_t count);
