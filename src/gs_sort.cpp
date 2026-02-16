#include "gs_sort.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Convert float to uint32 that sorts correctly with integer comparison
static inline uint32_t float_to_sortable(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // If positive: flip sign bit to make it larger than negatives
    // If negative: flip all bits to reverse order
    uint32_t mask = (-(bits >> 31)) | 0x80000000u;
    return bits ^ mask;
}

void gs_radix_sort_by_depth(ProjectedGaussian *gaussians, uint32_t count) {
    if (count <= 1) return;

    // For small arrays, use insertion sort
    if (count <= 64) {
        for (uint32_t i = 1; i < count; i++) {
            ProjectedGaussian key = gaussians[i];
            int j = (int)i - 1;
            while (j >= 0 && gaussians[j].depth > key.depth) {
                gaussians[j+1] = gaussians[j];
                j--;
            }
            gaussians[j+1] = key;
        }
        return;
    }

    // Radix sort: 8-bit passes, 4 passes for 32-bit keys
    auto *temp = static_cast<ProjectedGaussian*>(
        malloc(count * sizeof(ProjectedGaussian)));
    if (!temp) return;

    // Pre-compute sortable keys and all 4 histograms in one pass
    auto *keys = static_cast<uint32_t*>(malloc(count * sizeof(uint32_t)));
    auto *keys_temp = static_cast<uint32_t*>(malloc(count * sizeof(uint32_t)));
    if (!keys || !keys_temp) {
        free(temp); free(keys); free(keys_temp);
        return;
    }

    uint32_t hist[4][256] = {};
    for (uint32_t i = 0; i < count; i++) {
        uint32_t k = float_to_sortable(gaussians[i].depth);
        keys[i] = k;
        hist[0][ k        & 0xFF]++;
        hist[1][(k >>  8) & 0xFF]++;
        hist[2][(k >> 16) & 0xFF]++;
        hist[3][(k >> 24) & 0xFF]++;
    }

    ProjectedGaussian *src = gaussians;
    ProjectedGaussian *dst = temp;
    uint32_t *ksrc = keys;
    uint32_t *kdst = keys_temp;

    for (int pass = 0; pass < 4; pass++) {
        // Skip pass if all keys have the same byte value
        bool skip = false;
        for (int b = 0; b < 256; b++) {
            if (hist[pass][b] == count) { skip = true; break; }
        }
        if (skip) continue;

        // Prefix sum
        uint32_t *h = hist[pass];
        uint32_t total = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t c = h[b];
            h[b] = total;
            total += c;
        }

        // Scatter
        int shift = pass * 8;
        for (uint32_t i = 0; i < count; i++) {
            uint8_t byte = (ksrc[i] >> shift) & 0xFF;
            uint32_t idx = h[byte]++;
            dst[idx] = src[i];
            kdst[idx] = ksrc[i];
        }

        // Swap src/dst
        ProjectedGaussian *tmp_p = src; src = dst; dst = tmp_p;
        uint32_t *tmp_k = ksrc; ksrc = kdst; kdst = tmp_k;
    }

    // If result is in temp buffer, copy back
    if (src != gaussians) {
        memcpy(gaussians, src, count * sizeof(ProjectedGaussian));
    }

    free(temp);
    free(keys);
    free(keys_temp);
}
