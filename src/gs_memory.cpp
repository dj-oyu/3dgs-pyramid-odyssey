#include "gs_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "ax_sys_api.h"
}

bool gs_sys_init() {
    AX_S32 ret = AX_SYS_Init();
    if (ret != 0) {
        fprintf(stderr, "[gs_memory] AX_SYS_Init failed: 0x%08x\n", ret);
        return false;
    }
    printf("[gs_memory] AX_SYS_Init OK\n");
    return true;
}

void gs_sys_deinit() {
    AX_SYS_Deinit();
    printf("[gs_memory] AX_SYS_Deinit OK\n");
}

bool gs_cmm_alloc(CmmBuffer &buf, uint32_t size, const char *tag) {
    AX_U64 phys = 0;
    AX_VOID *virt = nullptr;
    AX_S32 ret = AX_SYS_MemAlloc(&phys, &virt, size, 4096,
                                   reinterpret_cast<const AX_S8*>(tag));
    if (ret != 0) {
        fprintf(stderr, "[gs_memory] AX_SYS_MemAlloc(%u, %s) failed: 0x%08x\n", size, tag, ret);
        return false;
    }
    buf.virt_addr = virt;
    buf.phys_addr = phys;
    buf.size = size;
    return true;
}

void gs_cmm_free(CmmBuffer &buf) {
    if (buf.virt_addr && buf.phys_addr) {
        AX_SYS_MemFree(buf.phys_addr, buf.virt_addr);
    }
    buf.virt_addr = nullptr;
    buf.phys_addr = 0;
    buf.size = 0;
}

float *gs_cmm_alloc_floats(uint32_t count, const char *tag, uint64_t *phys_out) {
    uint32_t size = count * sizeof(float);
    // Round up to 16-byte alignment for NEON
    size = (size + 15) & ~15u;

    AX_U64 phys = 0;
    AX_VOID *virt = nullptr;
    AX_S32 ret = AX_SYS_MemAlloc(&phys, &virt, size, 16,
                                   reinterpret_cast<const AX_S8*>(tag));
    if (ret != 0) {
        fprintf(stderr, "[gs_memory] CMM float alloc(%u, %s) failed: 0x%08x\n", count, tag, ret);
        return nullptr;
    }
    if (phys_out) *phys_out = phys;
    memset(virt, 0, size);
    return reinterpret_cast<float*>(virt);
}

void gs_cmm_free_floats(float *ptr, uint32_t count) {
    if (!ptr) return;
    AX_U64 phys = 0;
    AX_S32 memType = 0;
    AX_S32 ret = AX_SYS_MemGetBlockInfoByVirt(ptr, &phys, &memType);
    if (ret == 0) {
        AX_SYS_MemFree(phys, ptr);
    }
    (void)count;
}

float *gs_heap_alloc_floats(uint32_t count) {
    uint32_t size = count * sizeof(float);
    size = (size + 63) & ~63u;  // 64-byte align for NEON cache lines
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) return nullptr;
    memset(ptr, 0, size);
    return reinterpret_cast<float*>(ptr);
}

void gs_heap_free_floats(float *ptr) {
    free(ptr);
}

void gs_cmm_print_status() {
    AX_CMM_STATUS_T status;
    memset(&status, 0, sizeof(status));
    AX_S32 ret = AX_SYS_MemQueryStatus(&status);
    if (ret != 0) {
        fprintf(stderr, "[gs_memory] MemQueryStatus failed: 0x%08x\n", ret);
        return;
    }
    printf("[gs_memory] CMM Status: total=%u KB, remain=%u KB, blocks=%u\n",
           status.TotalSize, status.RemainSize, status.BlockCnt);
}
