// MAU/NPU Matrix Multiply Benchmark Tool
// Tests AX_IVE_MAU_MatMul and AX_IVE_NPU_MatMul for HW acceleration of
// Mahalanobis distance computation.
// Usage: sudo ./build/mau_bench
#include "gs_memory.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <arm_neon.h>

extern "C" {
#include "ax_ive_api.h"
}

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

// NEON matmul: C[M,N] = A[M,K] * B[K,N], N must be multiple of 4
static void neon_matmul(const float *A, const float *B, float *C,
                        int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; k++) {
                float32x4_t a = vdupq_n_f32(A[i * K + k]);
                float32x4_t b = vld1q_f32(&B[k * N + j]);
                sum = vmlaq_f32(sum, a, b);
            }
            vst1q_f32(&C[i * N + j], sum);
        }
    }
}

// Verify result against CPU reference
static bool verify_result(const float *hw_result, const float *cpu_result,
                          int M, int N, const char *label, float tolerance = 1e-3f) {
    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(hw_result[i] - cpu_result[i]);
        float rel = (fabsf(cpu_result[i]) > 1e-6f) ? err / fabsf(cpu_result[i]) : err;
        if (rel > tolerance) {
            if (err_count < 3) {
                printf("  %s MISMATCH [%d]: HW=%.6f CPU=%.6f err=%.6f\n",
                       label, i, hw_result[i], cpu_result[i], err);
            }
            err_count++;
        }
        if (err > max_err) max_err = err;
    }
    if (err_count > 0) {
        printf("  %s: %d mismatches out of %d (max_err=%.6f)\n", label, err_count, M * N, max_err);
        return false;
    }
    return true;
}

struct BenchBuffers {
    float *q_virt;     uint64_t q_phys;   // [M, K]
    float *b_virt;     uint64_t b_phys;   // [K, N]
    float *r_virt;     uint64_t r_phys;   // [M, N]
    float *cpu_result;                     // heap, for verification
    AX_S32 q_shape[2];
    AX_S32 b_shape[2];
    AX_S32 r_shape[2];
    int M, K, N;
};

static bool alloc_bench_buffers(BenchBuffers &buf, int M, int K, int N) {
    buf.M = M; buf.K = K; buf.N = N;

    buf.q_virt = gs_cmm_alloc_floats(M * K, "bench_Q", &buf.q_phys);
    buf.b_virt = gs_cmm_alloc_floats(K * N, "bench_B", &buf.b_phys);
    buf.r_virt = gs_cmm_alloc_floats(M * N, "bench_R", &buf.r_phys);
    buf.cpu_result = (float *)calloc(M * N, sizeof(float));

    if (!buf.q_virt || !buf.b_virt || !buf.r_virt || !buf.cpu_result) {
        fprintf(stderr, "Failed to allocate buffers for M=%d K=%d N=%d\n", M, K, N);
        return false;
    }

    buf.q_shape[0] = M; buf.q_shape[1] = K;
    buf.b_shape[0] = K; buf.b_shape[1] = N;
    buf.r_shape[0] = M; buf.r_shape[1] = N;

    // Fill with random data
    srand(42);
    for (int i = 0; i < M * K; i++) buf.q_virt[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) buf.b_virt[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    return true;
}

static void free_bench_buffers(BenchBuffers &buf) {
    if (buf.q_virt) gs_cmm_free_floats(buf.q_virt, buf.M * buf.K);
    if (buf.b_virt) gs_cmm_free_floats(buf.b_virt, buf.K * buf.N);
    if (buf.r_virt) gs_cmm_free_floats(buf.r_virt, buf.M * buf.N);
    free(buf.cpu_result);
    memset(&buf, 0, sizeof(buf));
}

// --- MAU Engine ---
static bool run_mau_matmul(BenchBuffers &buf) {
    AX_IVE_MAU_MATMUL_INPUT_T input = {};
    input.stMatQ.u64PhyAddr = buf.q_phys;
    input.stMatQ.pVirAddr   = buf.q_virt;
    input.stMatQ.pShape     = buf.q_shape;
    input.stMatQ.u8ShapeSize = 2;
    input.stMatQ.enDataType = AX_IVE_MAU_DT_FLOAT32;

    input.stMatB.u64PhyAddr = buf.b_phys;
    input.stMatB.pVirAddr   = buf.b_virt;
    input.stMatB.pShape     = buf.b_shape;
    input.stMatB.u8ShapeSize = 2;
    input.stMatB.enDataType = AX_IVE_MAU_DT_FLOAT32;

    AX_IVE_MAU_MATMUL_OUTPUT_T output = {};
    output.stMulRes.u64PhyAddr = buf.r_phys;
    output.stMulRes.pVirAddr   = buf.r_virt;
    output.stMulRes.pShape     = buf.r_shape;
    output.stMulRes.u8ShapeSize = 2;
    output.stMulRes.enDataType = AX_IVE_MAU_DT_FLOAT32;

    AX_IVE_MAU_MATMUL_CTRL_T ctrl = {};
    ctrl.enMauId = AX_IVE_MAU_ID_0;
    ctrl.s32DdrReadBandwidthLimit = -1;
    ctrl.bEnableMulRes  = AX_TRUE;
    ctrl.bEnableTopNRes = AX_FALSE;

    AX_IVE_HANDLE handle;
    AX_S32 ret = AX_IVE_MAU_MatMul(&handle, &input, &output, &ctrl,
                                     AX_IVE_ENGINE_MAU, AX_TRUE);
    if (ret != AX_SUCCESS) {
        return false;
    }
    return true;
}

// --- NPU Engine ---
static bool run_npu_matmul(AX_IVE_MATMUL_HANDLE npu_handle, BenchBuffers &buf) {
    AX_IVE_MAU_MATMUL_INPUT_T input = {};
    input.stMatQ.u64PhyAddr = buf.q_phys;
    input.stMatQ.pVirAddr   = buf.q_virt;
    input.stMatQ.pShape     = buf.q_shape;
    input.stMatQ.u8ShapeSize = 2;
    input.stMatQ.enDataType = AX_IVE_MAU_DT_FLOAT32;

    input.stMatB.u64PhyAddr = buf.b_phys;
    input.stMatB.pVirAddr   = buf.b_virt;
    input.stMatB.pShape     = buf.b_shape;
    input.stMatB.u8ShapeSize = 2;
    input.stMatB.enDataType = AX_IVE_MAU_DT_FLOAT32;

    AX_IVE_MAU_MATMUL_OUTPUT_T output = {};
    output.stMulRes.u64PhyAddr = buf.r_phys;
    output.stMulRes.pVirAddr   = buf.r_virt;
    output.stMulRes.pShape     = buf.r_shape;
    output.stMulRes.u8ShapeSize = 2;
    output.stMulRes.enDataType = AX_IVE_MAU_DT_FLOAT32;

    AX_S32 ret = AX_IVE_NPU_MatMul(npu_handle, &input, &output,
                                     AX_IVE_ENGINE_NPU, AX_TRUE);
    if (ret != AX_SUCCESS) {
        return false;
    }
    return true;
}

static void print_table_header(const char *engine_name) {
    printf("\n=== %s MatMul: Q[M,6] x B[6,256] -> R[M,256] (FLOAT32) ===\n", engine_name);
    printf("%-6s | %-12s | %-12s | %-12s | %-10s | %-10s | %s\n",
           "M", "HW (us)", "NEON (us)", "CPU (us)", "HW GFLOP/s", "Speedup", "Correct");
    printf("-------+-------------+-------------+-------------+------------+------------+--------\n");
}

int main() {
    printf("=== Matrix Multiply Benchmark Tool ===\n");
    printf("Tests MAU and NPU engines for HW-accelerated MatMul\n\n");

    // Initialize system
    if (!gs_sys_init()) {
        fprintf(stderr, "AX_SYS_Init failed\n");
        return 1;
    }
    gs_cmm_print_status();

    // Detect chip type
    FILE *f = fopen("/proc/ax_proc/chip_type", "r");
    if (f) {
        char chip[64] = {};
        if (fgets(chip, sizeof(chip), f)) {
            // Strip newline
            char *nl = strchr(chip, '\n');
            if (nl) *nl = '\0';
            printf("[bench] Chip: %s\n", chip);
        }
        fclose(f);
    }

    // Initialize IVE
    AX_S32 ret = AX_IVE_Init();
    if (ret != AX_SUCCESS) {
        fprintf(stderr, "AX_IVE_Init failed: 0x%08x\n", ret);
        gs_sys_deinit();
        return 1;
    }
    printf("[bench] AX_IVE_Init OK\n");

    // Fixed inner dim K=6, N=256 (tile pixel count) - matches our use case
    static constexpr int K = 6;
    static constexpr int N = 256;  // TILE_SIZE * TILE_SIZE = 16*16
    int test_M[] = {1, 4, 8, 16, 32, 64, 128, 256};
    int num_tests = sizeof(test_M) / sizeof(test_M[0]);

    static constexpr int WARMUP = 10;
    static constexpr int ITERS  = 200;

    // ========== Test 1: MAU Engine ==========
    {
        printf("\n--- Probing MAU engine ---\n");
        BenchBuffers probe = {};
        bool mau_available = false;
        if (alloc_bench_buffers(probe, 1, K, N)) {
            mau_available = run_mau_matmul(probe);
            free_bench_buffers(probe);
        }

        if (mau_available) {
            printf("MAU engine: AVAILABLE\n");
            print_table_header("MAU");

            for (int ti = 0; ti < num_tests; ti++) {
                int M = test_M[ti];
                BenchBuffers buf = {};
                if (!alloc_bench_buffers(buf, M, K, N)) continue;

                cpu_matmul(buf.q_virt, buf.b_virt, buf.cpu_result, M, K, N);

                // MAU warmup + bench
                for (int i = 0; i < WARMUP; i++) run_mau_matmul(buf);
                double t0 = get_time_us();
                for (int i = 0; i < ITERS; i++) run_mau_matmul(buf);
                double t1 = get_time_us();
                double hw_us = (t1 - t0) / ITERS;

                bool correct = verify_result(buf.r_virt, buf.cpu_result, M, N, "MAU");

                // NEON bench
                float *neon_res = (float *)calloc(M * N, sizeof(float));
                for (int i = 0; i < WARMUP; i++) neon_matmul(buf.q_virt, buf.b_virt, neon_res, M, K, N);
                t0 = get_time_us();
                for (int i = 0; i < ITERS; i++) neon_matmul(buf.q_virt, buf.b_virt, neon_res, M, K, N);
                t1 = get_time_us();
                double neon_us = (t1 - t0) / ITERS;

                // CPU bench
                t0 = get_time_us();
                for (int i = 0; i < ITERS; i++) cpu_matmul(buf.q_virt, buf.b_virt, buf.cpu_result, M, K, N);
                t1 = get_time_us();
                double cpu_us = (t1 - t0) / ITERS;

                double flops = 2.0 * M * K * N;
                printf("%-6d | %10.1f  | %10.1f  | %10.1f  | %9.3f  | %9.2fx | %s\n",
                       M, hw_us, neon_us, cpu_us, flops / hw_us / 1e3, neon_us / hw_us,
                       correct ? "OK" : "FAIL");

                free(neon_res);
                free_bench_buffers(buf);
            }
        } else {
            printf("MAU engine: NOT AVAILABLE (AX650C lacks MAU hardware)\n");
        }
    }

    // ========== Test 2: NPU Engine ==========
    {
        printf("\n--- Probing NPU engine ---\n");

        // Try creating NPU MatMul handle
        // Model cache dir (NPU may need this to store compiled kernels)
        char model_dir[] = "/tmp/npu_matmul_cache";
        // Create the cache directory
        {
            char cmd[128];
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", model_dir);
            (void)system(cmd);
        }

        AX_IVE_NPU_MATMUL_CTRL_T npu_ctrl = {};
        npu_ctrl.pchModelDir = model_dir;
        npu_ctrl.enDataType = AX_IVE_MAU_DT_FLOAT32;
        npu_ctrl.s32KSize = K;  // Inner dimension

        AX_IVE_MATMUL_HANDLE npu_handle = nullptr;
        ret = AX_IVE_NPU_CreateMatMulHandle(&npu_handle, &npu_ctrl);
        if (ret != AX_SUCCESS) {
            printf("NPU engine: CreateHandle failed (0x%08x)\n", ret);
            printf("  Trying with different K sizes...\n");

            // Try different K sizes in case the API interprets it differently
            int try_k[] = {6, 256, 1536};  // K=6, or N=256, or M*K
            bool npu_ok = false;
            for (int tk : try_k) {
                npu_ctrl.s32KSize = tk;
                ret = AX_IVE_NPU_CreateMatMulHandle(&npu_handle, &npu_ctrl);
                if (ret == AX_SUCCESS) {
                    printf("  Success with s32KSize=%d\n", tk);
                    npu_ok = true;
                    break;
                }
                printf("  s32KSize=%d -> 0x%08x\n", tk, ret);
            }
            if (!npu_ok) {
                printf("NPU engine: NOT AVAILABLE\n");
                goto cleanup;
            }
        } else {
            printf("NPU engine: Handle created OK (K=%d)\n", K);
        }

        // Test NPU MatMul
        {
            BenchBuffers probe = {};
            bool npu_works = false;
            if (alloc_bench_buffers(probe, 4, K, N)) {
                cpu_matmul(probe.q_virt, probe.b_virt, probe.cpu_result, 4, K, N);
                npu_works = run_npu_matmul(npu_handle, probe);
                if (npu_works) {
                    bool correct = verify_result(probe.r_virt, probe.cpu_result, 4, N, "NPU");
                    printf("NPU probe: %s (correct=%s)\n",
                           npu_works ? "OK" : "FAIL", correct ? "yes" : "no");
                } else {
                    printf("NPU probe: MatMul call FAILED\n");
                }
                free_bench_buffers(probe);
            }

            if (npu_works) {
                print_table_header("NPU");

                for (int ti = 0; ti < num_tests; ti++) {
                    int M = test_M[ti];
                    BenchBuffers buf = {};
                    if (!alloc_bench_buffers(buf, M, K, N)) continue;

                    cpu_matmul(buf.q_virt, buf.b_virt, buf.cpu_result, M, K, N);

                    // NPU warmup + bench
                    bool npu_ok = true;
                    for (int i = 0; i < WARMUP; i++) {
                        if (!run_npu_matmul(npu_handle, buf)) { npu_ok = false; break; }
                    }

                    double hw_us = 0;
                    if (npu_ok) {
                        double t0 = get_time_us();
                        for (int i = 0; i < ITERS; i++) {
                            if (!run_npu_matmul(npu_handle, buf)) { npu_ok = false; break; }
                        }
                        double t1 = get_time_us();
                        hw_us = (t1 - t0) / ITERS;
                    }

                    bool correct = npu_ok && verify_result(buf.r_virt, buf.cpu_result, M, N, "NPU");

                    // NEON bench
                    float *neon_res = (float *)calloc(M * N, sizeof(float));
                    for (int i = 0; i < WARMUP; i++)
                        neon_matmul(buf.q_virt, buf.b_virt, neon_res, M, K, N);
                    double t0 = get_time_us();
                    for (int i = 0; i < ITERS; i++)
                        neon_matmul(buf.q_virt, buf.b_virt, neon_res, M, K, N);
                    double t1 = get_time_us();
                    double neon_us = (t1 - t0) / ITERS;

                    // CPU bench
                    t0 = get_time_us();
                    for (int i = 0; i < ITERS; i++)
                        cpu_matmul(buf.q_virt, buf.b_virt, buf.cpu_result, M, K, N);
                    t1 = get_time_us();
                    double cpu_us = (t1 - t0) / ITERS;

                    double flops = 2.0 * M * K * N;
                    if (npu_ok) {
                        printf("%-6d | %10.1f  | %10.1f  | %10.1f  | %9.3f  | %9.2fx | %s\n",
                               M, hw_us, neon_us, cpu_us, flops / hw_us / 1e3,
                               neon_us / hw_us, correct ? "OK" : "FAIL");
                    } else {
                        printf("%-6d | %10s  | %10.1f  | %10.1f  | %9s  | %9s  | %s\n",
                               M, "FAILED", neon_us, cpu_us, "N/A", "N/A", "N/A");
                    }

                    free(neon_res);
                    free_bench_buffers(buf);
                }
            }
        }

        if (npu_handle) {
            AX_IVE_NPU_DestroyMatMulHandle(&npu_handle);
        }
    }

cleanup:
    printf("\n--- Decision Gate ---\n");
    printf("If HW latency for M=32..64 (typical dense tile) is:\n");
    printf("  < 20 us  : Per-tile HW accel for all non-empty tiles\n");
    printf("  20-100 us: HW only for dense tiles (count >= threshold)\n");
    printf("  > 100 us : HW overhead too high, use CPU-only rasterizer\n");

    printf("\nDone.\n");
    AX_IVE_Exit();
    gs_sys_deinit();
    return 0;
}
