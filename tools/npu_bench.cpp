// NPU Inference Benchmark Tool
// Tests AX_ENGINE_RunSync latency for the MatMul .axmodel
// Usage: sudo ./build/npu_bench <model.axmodel> [iterations]
#include "gs_npu.h"
#include "gs_mau.h"
#include "gs_memory.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <arm_neon.h>

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// CPU reference matmul: C[M,N] = A[M,K] * B[K,N]
static void cpu_matmul(const float *A, const float *B, float *C,
                       int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Build P_T pixel feature matrix (matches gs_mau.cpp)
static void build_pixel_features(float *P_T) {
    for (uint32_t ly = 0; ly < TILE_SIZE; ly++) {
        for (uint32_t lx = 0; lx < TILE_SIZE; lx++) {
            uint32_t col = ly * TILE_SIZE + lx;
            float fx = (float)lx + 0.5f;
            float fy = (float)ly + 0.5f;
            P_T[0 * 256 + col] = fx;
            P_T[1 * 256 + col] = fy;
            P_T[2 * 256 + col] = fx * fx;
            P_T[3 * 256 + col] = fy * fy;
            P_T[4 * 256 + col] = fx * fy;
            P_T[5 * 256 + col] = 1.0f;
        }
    }
}

// Generate realistic G_T matrix for K Gaussians
static void fill_g_matrix(float *G_T, uint32_t K) {
    srand(42);
    for (uint32_t k = 0; k < K; k++) {
        G_T[k * 6 + 0] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // linear x
        G_T[k * 6 + 1] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // linear y
        G_T[k * 6 + 2] = ((float)rand() / RAND_MAX) * 5.0f;            // quadratic x
        G_T[k * 6 + 3] = ((float)rand() / RAND_MAX) * 5.0f;            // quadratic y
        G_T[k * 6 + 4] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;   // cross term
        G_T[k * 6 + 5] = ((float)rand() / RAND_MAX) * 20.0f;           // constant
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.axmodel> [iterations]\n", argv[0]);
        printf("  Tests AX_ENGINE inference latency for MatMul model\n");
        return 1;
    }

    const char *model_path = argv[1];
    int iters = (argc >= 3) ? atoi(argv[2]) : 200;
    if (iters < 1) iters = 200;

    printf("=== NPU Inference Benchmark ===\n");
    printf("Model: %s\n", model_path);
    printf("Iterations: %d\n\n", iters);

    // Initialize system
    if (!gs_sys_init()) {
        fprintf(stderr, "AX_SYS_Init failed\n");
        return 1;
    }
    gs_cmm_print_status();

    // Initialize NPU
    NPUContext npu = {};
    if (!gs_npu_init(npu, model_path)) {
        fprintf(stderr, "NPU init failed\n");
        gs_sys_deinit();
        return 1;
    }

    // Build P_T for CPU reference
    float *P_T = (float *)calloc(6 * 256, sizeof(float));
    build_pixel_features(P_T);

    // Test K sizes
    uint32_t test_K[] = {16, 32, 64, 128, 256};
    int num_tests = sizeof(test_K) / sizeof(test_K[0]);

    printf("\n%-6s | %-12s | %-12s | %-12s | %-10s | %-10s | %s\n",
           "K", "NPU (us)", "Min (us)", "Max (us)", "GFLOP/s", "Speedup", "MaxErr");
    printf("-------+-------------+-------------+-------------+------------+------------+--------\n");

    NPUContext::ThreadBuf &tbuf = npu.thread_bufs[0];

    for (int ti = 0; ti < num_tests; ti++) {
        uint32_t K = test_K[ti];

        // Fill input with test data
        fill_g_matrix(tbuf.input_virt, K);
        // Zero-pad remainder
        if (K < npu.max_k) {
            memset(&tbuf.input_virt[K * 6], 0, (npu.max_k - K) * 6 * sizeof(float));
        }

        // CPU reference: G_T[K,6] x P_T[6,256] -> [K,256]
        float *cpu_result = (float *)calloc(npu.max_k * 256, sizeof(float));
        cpu_matmul(tbuf.input_virt, P_T, cpu_result, npu.max_k, 6, 256);

        // NPU warmup
        static constexpr int WARMUP = 10;
        for (int i = 0; i < WARMUP; i++) {
            gs_npu_infer(npu, tbuf, K);
        }

        // NPU benchmark
        double min_us = 1e9, max_us = 0, total_us = 0;
        bool all_ok = true;
        for (int i = 0; i < iters; i++) {
            // Refill input each iteration (simulate real workload)
            fill_g_matrix(tbuf.input_virt, K);
            if (K < npu.max_k) {
                memset(&tbuf.input_virt[K * 6], 0, (npu.max_k - K) * 6 * sizeof(float));
            }

            double t0 = get_time_us();
            bool ok = gs_npu_infer(npu, tbuf, K);
            double t1 = get_time_us();

            if (!ok) { all_ok = false; break; }

            double dt = t1 - t0;
            total_us += dt;
            if (dt < min_us) min_us = dt;
            if (dt > max_us) max_us = dt;
        }

        if (!all_ok) {
            printf("%-6u | %10s  | %10s  | %10s  | %9s  | %9s  | FAIL\n",
                   K, "FAILED", "-", "-", "-", "-");
            free(cpu_result);
            continue;
        }

        double avg_us = total_us / iters;

        // Verify output vs CPU reference (using last iteration's data)
        cpu_matmul(tbuf.input_virt, P_T, cpu_result, npu.max_k, 6, 256);
        float max_err = 0.0f;
        for (uint32_t i = 0; i < K * 256; i++) {
            float err = fabsf(tbuf.output_virt[i] - cpu_result[i]);
            if (err > max_err) max_err = err;
        }

        // CPU reference timing for speedup calculation
        double cpu_t0 = get_time_us();
        for (int i = 0; i < iters; i++) {
            cpu_matmul(tbuf.input_virt, P_T, cpu_result, npu.max_k, 6, 256);
        }
        double cpu_t1 = get_time_us();
        double cpu_avg_us = (cpu_t1 - cpu_t0) / iters;

        double flops = 2.0 * npu.max_k * 6 * 256;
        double gflops = flops / avg_us / 1e3;

        printf("%-6u | %10.1f  | %10.1f  | %10.1f  | %9.3f  | %9.2fx | %.4f\n",
               K, avg_us, min_us, max_us, gflops, cpu_avg_us / avg_us, max_err);

        free(cpu_result);
    }

    // Summary
    printf("\n--- Break-Even Analysis ---\n");
    printf("If NPU avg < CPU avg for a given K, NPU acceleration is beneficial.\n");
    printf("Set tile_threshold to the smallest K where NPU wins.\n");
    printf("Current tile_threshold = %u\n", npu.tile_threshold);

    // Cleanup
    free(P_T);
    gs_npu_deinit(npu);
    gs_sys_deinit();

    printf("\nDone.\n");
    return 0;
}
