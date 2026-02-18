#pragma once

#include "gs_types.h"
#include <cstdint>
#include <pthread.h>

// Forward declaration
struct ProjectedGaussian;

// MAU (Matrix Arithmetic Unit) acceleration context for tile rasterization.
// Offloads Mahalanobis distance computation: Power[K,256] = G_T[K,6] x P_T[6,256]
struct MAUContext {
    bool initialized = false;

    // P_T[6, 256]: pixel feature matrix (fixed, precomputed once)
    // Row 0: lx+0.5, Row 1: ly+0.5, Row 2: (lx+0.5)^2,
    // Row 3: (ly+0.5)^2, Row 4: (lx+0.5)(ly+0.5), Row 5: 1.0
    float *pixel_features_virt = nullptr;
    uint64_t pixel_features_phys = 0;
    int32_t pixel_features_shape[2];  // {6, 256}

    // Per-thread scratch buffers for G_T and result matrices
    struct ThreadBuf {
        float *g_matrix_virt = nullptr;    // G_T[max_k, 6] in CMM
        uint64_t g_matrix_phys = 0;
        float *result_virt = nullptr;      // Result[max_k, 256] in CMM
        uint64_t result_phys = 0;
        int32_t g_shape[2];                // rewritten per tile: {K, 6}
        int32_t r_shape[2];                // rewritten per tile: {K, 256}
    } thread_bufs[NUM_RENDER_THREADS];

    uint32_t max_k = 256;                  // max Gaussians per MAU call
    uint32_t tile_threshold = 16;          // min Gaussians for MAU path (tuned from benchmark)

    // Mutex for MAU hardware serialization (single HW unit)
    pthread_mutex_t mau_mutex;
};

// Initialize MAU context: AX_IVE_Init + CMM allocation + build P_T matrix
// Returns false if MAU is unavailable (non-fatal for renderer)
bool gs_mau_init(MAUContext &ctx);

// Free MAU resources
void gs_mau_deinit(MAUContext &ctx);

// Build G_T coefficient matrix for a tile's Gaussians.
// G_T[K, 6] where each row k has coefficients derived from the inverse covariance
// and the Gaussian's offset from the tile origin.
// tile_ox/tile_oy: pixel coordinates of the tile's top-left corner
void gs_mau_build_g_matrix(float *G_T, const ProjectedGaussian *gaussians,
                           const uint32_t *indices, uint32_t K,
                           float tile_ox, float tile_oy);

// Execute MAU MatMul: G_T[K,6] x P_T[6,256] -> Result[K,256]
// Thread-safe (acquires mau_mutex internally).
// Returns false on MAU error.
bool gs_mau_matmul(MAUContext &ctx, MAUContext::ThreadBuf &tbuf, uint32_t K);
