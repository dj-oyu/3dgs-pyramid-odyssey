#include "gs_sort.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>

bool gs_sort_alloc(SortContext &ctx, uint32_t max_count) {
    ctx.pairs = static_cast<uint64_t*>(malloc(max_count * sizeof(uint64_t)));
    ctx.pairs_temp = static_cast<uint64_t*>(malloc(max_count * sizeof(uint64_t)));
    ctx.sorted_indices = static_cast<uint32_t*>(malloc(max_count * sizeof(uint32_t)));
    if (!ctx.pairs || !ctx.pairs_temp || !ctx.sorted_indices) {
        gs_sort_free(ctx);
        return false;
    }
    ctx.capacity = max_count;
    return true;
}

void gs_sort_free(SortContext &ctx) {
    free(ctx.pairs);
    free(ctx.pairs_temp);
    free(ctx.sorted_indices);
    ctx.pairs = nullptr;
    ctx.pairs_temp = nullptr;
    ctx.sorted_indices = nullptr;
    ctx.capacity = 0;
}

void gs_sort_by_depth(SortContext &ctx, const ProjectedGaussian *gaussians, uint32_t count) {
    if (count <= 1) {
        if (count == 1) ctx.sorted_indices[0] = 0;
        return;
    }

    // Insertion sort for small arrays (produces indices, not physical reorder)
    if (count <= 128) {
        for (uint32_t i = 0; i < count; i++) ctx.sorted_indices[i] = i;
        for (uint32_t i = 1; i < count; i++) {
            uint32_t key_idx = ctx.sorted_indices[i];
            float key_depth = gaussians[key_idx].depth;
            int j = (int)i - 1;
            while (j >= 0 && gaussians[ctx.sorted_indices[j]].depth > key_depth) {
                ctx.sorted_indices[j+1] = ctx.sorted_indices[j];
                j--;
            }
            ctx.sorted_indices[j+1] = key_idx;
        }
        return;
    }

    // Compact 16-bit sort: 32-bit pairs (key16 << 16 | index16), 2 radix passes
    if (count <= 65535) {
        // Find depth range
        float min_d = gaussians[0].depth, max_d = gaussians[0].depth;
        for (uint32_t i = 1; i < count; i++) {
            float d = gaussians[i].depth;
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
        }

        float range = max_d - min_d;
        if (range < 1e-8f) {
            for (uint32_t i = 0; i < count; i++) ctx.sorted_indices[i] = i;
            return;
        }
        float scale = 65534.0f / range;

        uint32_t *pairs32 = reinterpret_cast<uint32_t*>(ctx.pairs);
        uint32_t *temp32 = reinterpret_cast<uint32_t*>(ctx.pairs_temp);

        uint32_t hist[2][256] = {};
        for (uint32_t i = 0; i < count; i++) {
            uint32_t k = (uint32_t)((gaussians[i].depth - min_d) * scale);
            if (k > 65534) k = 65534;
            pairs32[i] = (k << 16) | (uint16_t)i;
            hist[0][ k       & 0xFF]++;
            hist[1][(k >> 8) & 0xFF]++;
        }

        uint32_t *src32 = pairs32, *dst32 = temp32;

        for (int pass = 0; pass < 2; pass++) {
            bool skip = false;
            for (int b = 0; b < 256; b++) {
                if (hist[pass][b] == count) { skip = true; break; }
            }
            if (skip) continue;

            uint32_t *h = hist[pass];
            uint32_t total = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t c = h[b];
                h[b] = total;
                total += c;
            }

            int shift = 16 + pass * 8;
            for (uint32_t i = 0; i < count; i++) {
                uint8_t byte = (src32[i] >> shift) & 0xFF;
                dst32[h[byte]++] = src32[i];
            }

            uint32_t *t = src32; src32 = dst32; dst32 = t;
        }

        // Extract sorted indices (no permute of gaussians)
        for (uint32_t i = 0; i < count; i++) {
            ctx.sorted_indices[i] = src32[i] & 0xFFFF;
        }
        return;
    }

    // Fallback: 64-bit pairs with 4 radix passes (for count > 65535)
    uint32_t hist[4][256] = {};
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bits;
        float d = gaussians[i].depth;
        memcpy(&bits, &d, sizeof(bits));
        uint32_t mask = (-(bits >> 31)) | 0x80000000u;
        uint32_t k = bits ^ mask;
        ctx.pairs[i] = ((uint64_t)k << 32) | i;
        hist[0][ k        & 0xFF]++;
        hist[1][(k >>  8) & 0xFF]++;
        hist[2][(k >> 16) & 0xFF]++;
        hist[3][(k >> 24) & 0xFF]++;
    }

    uint64_t *src = ctx.pairs;
    uint64_t *dst = ctx.pairs_temp;

    for (int pass = 0; pass < 4; pass++) {
        bool skip = false;
        for (int b = 0; b < 256; b++) {
            if (hist[pass][b] == count) { skip = true; break; }
        }
        if (skip) continue;

        uint32_t *h = hist[pass];
        uint32_t total = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t c = h[b];
            h[b] = total;
            total += c;
        }

        int shift = 32 + pass * 8;
        for (uint32_t i = 0; i < count; i++) {
            uint8_t byte = (src[i] >> shift) & 0xFF;
            dst[h[byte]++] = src[i];
        }

        uint64_t *tmp = src; src = dst; dst = tmp;
    }

    // Extract sorted indices (no permute)
    for (uint32_t i = 0; i < count; i++) {
        ctx.sorted_indices[i] = (uint32_t)(src[i] & 0xFFFFFFFF);
    }
}
