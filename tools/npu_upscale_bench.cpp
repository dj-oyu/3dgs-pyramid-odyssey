// NPU Super-Resolution Upscale Benchmark
// Tests AX_ENGINE inference latency for ESPCN-x2 .axmodel
// Go/no-go: avg <20ms -> proceed, >50ms -> abandon NPU
//
// Usage: sudo ./build/npu_upscale_bench <espcn_x2.axmodel> [iterations]

#include "gs_npu.h"
#include "gs_memory.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>

static double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <espcn_x2.axmodel> [iterations]\n", argv[0]);
        printf("  Benchmarks NPU super-resolution inference latency\n");
        printf("  Go/no-go: avg <20ms -> proceed, >50ms -> abandon\n");
        return 1;
    }

    const char *model_path = argv[1];
    int iters = (argc >= 3) ? atoi(argv[2]) : 100;
    if (iters < 1) iters = 100;

    printf("=== NPU Super-Resolution Benchmark ===\n");
    printf("Model: %s\n", model_path);
    printf("Iterations: %d\n\n", iters);

    // Initialize system
    if (!gs_sys_init()) {
        fprintf(stderr, "AX_SYS_Init failed\n");
        return 1;
    }
    gs_cmm_print_status();

    // Initialize NPU with model
    NPUContext npu = {};
    if (!gs_npu_init(npu, model_path)) {
        fprintf(stderr, "NPU init failed\n");
        gs_sys_deinit();
        return 1;
    }

    // Print model I/O shapes
    printf("\nModel I/O info:\n");
    for (AX_U32 i = 0; i < npu.io_info->nInputSize; i++) {
        auto &meta = npu.io_info->pInputs[i];
        printf("  Input[%u] '%s': shape=[", i, meta.pName);
        for (AX_U8 d = 0; d < meta.nShapeSize; d++) {
            printf("%d%s", meta.pShape[d], d + 1 < meta.nShapeSize ? "," : "");
        }
        printf("] dtype=%d nSize=%u\n", meta.eDataType, meta.nSize);
    }
    for (AX_U32 i = 0; i < npu.io_info->nOutputSize; i++) {
        auto &meta = npu.io_info->pOutputs[i];
        printf("  Output[%u] '%s': shape=[", i, meta.pName);
        for (AX_U8 d = 0; d < meta.nShapeSize; d++) {
            printf("%d%s", meta.pShape[d], d + 1 < meta.nShapeSize ? "," : "");
        }
        printf("] dtype=%d nSize=%u\n", meta.eDataType, meta.nSize);
    }

    // Determine buffer sizes from model I/O metadata
    uint32_t input_size = npu.io_info->pInputs[0].nSize;
    uint32_t output_size = npu.io_info->pOutputs[0].nSize;

    if (input_size == 0 || output_size == 0) {
        fprintf(stderr, "Model reports zero-size I/O buffers\n");
        gs_npu_deinit(npu);
        gs_sys_deinit();
        return 1;
    }

    printf("\nAllocating CMM buffers: input=%u bytes (%.1f MB), output=%u bytes (%.1f MB)\n",
           input_size, input_size / (1024.0f * 1024.0f),
           output_size, output_size / (1024.0f * 1024.0f));

    // Allocate CMM buffers
    CmmBuffer in_buf = {}, out_buf = {};
    if (!gs_cmm_alloc(in_buf, input_size, "upscale_in")) {
        fprintf(stderr, "Failed to allocate input CMM buffer (%u bytes)\n", input_size);
        gs_npu_deinit(npu);
        gs_sys_deinit();
        return 1;
    }
    if (!gs_cmm_alloc(out_buf, output_size, "upscale_out")) {
        fprintf(stderr, "Failed to allocate output CMM buffer (%u bytes)\n", output_size);
        gs_cmm_free(in_buf);
        gs_npu_deinit(npu);
        gs_sys_deinit();
        return 1;
    }

    // Fill input with synthetic image data (gradient pattern)
    uint8_t *in_bytes = static_cast<uint8_t*>(in_buf.virt_addr);
    for (uint32_t i = 0; i < input_size; i++) {
        in_bytes[i] = (uint8_t)(i % 256);
    }

    // Build AX_ENGINE_IO_T
    AX_ENGINE_IO_BUFFER_T input_io = {};
    input_io.phyAddr = in_buf.phys_addr;
    input_io.pVirAddr = in_buf.virt_addr;
    input_io.nSize = input_size;

    AX_ENGINE_IO_BUFFER_T output_io = {};
    output_io.phyAddr = out_buf.phys_addr;
    output_io.pVirAddr = out_buf.virt_addr;
    output_io.nSize = output_size;

    AX_ENGINE_IO_T io = {};
    io.pInputs = &input_io;
    io.nInputSize = 1;
    io.pOutputs = &output_io;
    io.nOutputSize = 1;

    // Warmup
    printf("\nWarming up (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        AX_S32 ret = AX_ENGINE_RunSync(npu.model_handle, &io);
        if (ret != 0) {
            fprintf(stderr, "Warmup failed at iteration %d: 0x%08x\n", i, ret);
            gs_cmm_free(in_buf);
            gs_cmm_free(out_buf);
            gs_npu_deinit(npu);
            gs_sys_deinit();
            return 1;
        }
    }
    printf("Warmup OK\n");

    // Benchmark
    printf("\nRunning %d iterations...\n", iters);
    double min_us = 1e15, max_us = 0, total_us = 0;
    int successful = 0;

    for (int i = 0; i < iters; i++) {
        double t0 = get_time_us();
        AX_S32 ret = AX_ENGINE_RunSync(npu.model_handle, &io);
        double t1 = get_time_us();

        if (ret != 0) {
            fprintf(stderr, "Inference failed at iteration %d: 0x%08x\n", i, ret);
            break;
        }

        successful++;
        double dt = t1 - t0;
        total_us += dt;
        if (dt < min_us) min_us = dt;
        if (dt > max_us) max_us = dt;
    }

    if (successful == 0) {
        fprintf(stderr, "All iterations failed!\n");
        gs_cmm_free(in_buf);
        gs_cmm_free(out_buf);
        gs_npu_deinit(npu);
        gs_sys_deinit();
        return 1;
    }

    double avg_us = total_us / successful;
    double avg_ms = avg_us / 1000.0;
    double min_ms = min_us / 1000.0;
    double max_ms = max_us / 1000.0;

    printf("\n=== Results (%d/%d successful) ===\n", successful, iters);
    printf("  Min:    %8.2f ms\n", min_ms);
    printf("  Avg:    %8.2f ms\n", avg_ms);
    printf("  Max:    %8.2f ms\n", max_ms);
    printf("  Jitter: %8.2f ms (max-min)\n", max_ms - min_ms);

    // Go/no-go decision
    printf("\n=== Go/No-Go Decision ===\n");
    if (avg_ms < 20.0) {
        printf("  RESULT: GO (%.1f ms < 20ms threshold)\n", avg_ms);
        printf("  Expected: render@540p (~50ms) + upscale (%.1fms) = %.1fms -> %.1f FPS\n",
               avg_ms, 50.0 + avg_ms, 1000.0 / (50.0 + avg_ms));
    } else if (avg_ms < 50.0) {
        printf("  RESULT: MARGINAL (%.1f ms, between 20-50ms)\n", avg_ms);
        printf("  Still better than native 1080p (~130ms -> ~8 FPS)\n");
        printf("  Expected: render@540p (~50ms) + upscale (%.1fms) = %.1fms -> %.1f FPS\n",
               avg_ms, 50.0 + avg_ms, 1000.0 / (50.0 + avg_ms));
    } else {
        printf("  RESULT: NO-GO (%.1f ms > 50ms threshold)\n", avg_ms);
        printf("  NPU upscale too slow. Abandon NPU approach.\n");
    }

    // Cleanup
    gs_cmm_free(in_buf);
    gs_cmm_free(out_buf);
    gs_npu_deinit(npu);
    gs_sys_deinit();

    printf("\nDone.\n");
    return 0;
}
