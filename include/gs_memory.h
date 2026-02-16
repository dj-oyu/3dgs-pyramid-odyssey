#pragma once

#include <cstdint>
#include <cstddef>

// CMM memory allocation wrapper
struct CmmBuffer {
    void     *virt_addr = nullptr;
    uint64_t  phys_addr = 0;
    uint32_t  size      = 0;
};

// Initialize the AX system (must call before any CMM allocation)
bool gs_sys_init();
void gs_sys_deinit();

// Allocate contiguous memory via CMM
bool gs_cmm_alloc(CmmBuffer &buf, uint32_t size, const char *tag);
void gs_cmm_free(CmmBuffer &buf);

// Allocate aligned float arrays from CMM (16-byte aligned for NEON)
float *gs_cmm_alloc_floats(uint32_t count, const char *tag, uint64_t *phys_out = nullptr);
void gs_cmm_free_floats(float *ptr, uint32_t count);

// Allocate aligned float arrays from heap (cached, 64-byte aligned for NEON)
float *gs_heap_alloc_floats(uint32_t count);
void gs_heap_free_floats(float *ptr);

// Query CMM status
void gs_cmm_print_status();
