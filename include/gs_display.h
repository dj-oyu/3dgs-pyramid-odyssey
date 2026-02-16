#pragma once

#include "gs_types.h"
#include <cstdint>

enum class DisplayBackend {
    VO,           // AX_VO API (preferred)
    Framebuffer,  // /dev/fbN (fallback)
};

struct DisplayContext {
    DisplayBackend backend = DisplayBackend::Framebuffer;
    // VO backend
    uint32_t vo_dev    = 1;
    uint32_t vo_layer  = 0;
    uint32_t vo_chn    = 0;
    // FB backend
    int      fb_fd     = -1;
    uint8_t *fb_mem    = nullptr;
    uint32_t fb_mem_size = 0;
    uint32_t fb_stride = 0;
    uint32_t fb_bpp    = 32;
    bool     fb_swap_rb = false;  // true if fb expects ABGR instead of ARGB
    bool     fb_vsync   = false; // wait for vsync before returning
    // Common
    uint32_t width     = DISPLAY_WIDTH;
    uint32_t height    = DISPLAY_HEIGHT;
    bool     initialized = false;
};

// Initialize display (tries VO first, falls back to /dev/fb)
bool gs_display_init(DisplayContext &ctx);
void gs_display_deinit(DisplayContext &ctx);

// Send a framebuffer to the display
bool gs_display_send_frame(DisplayContext &ctx, const Framebuffer &fb);

// Allocate a framebuffer suitable for display (ARGB8888)
// If need_phys=true, uses CMM (required for VO backend). Otherwise uses cached malloc.
bool gs_framebuffer_alloc(Framebuffer &fb, uint32_t width, uint32_t height, bool need_phys = true);
void gs_framebuffer_free(Framebuffer &fb);
