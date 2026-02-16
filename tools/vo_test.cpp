// VO display test: Displays test patterns on HDMI
#include "gs_memory.h"
#include "gs_display.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <csignal>
#include <unistd.h>

static volatile bool g_running = true;

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
}

static void fill_solid(Framebuffer &fb, uint32_t argb) {
    uint32_t *pixels = reinterpret_cast<uint32_t*>(fb.data);
    for (uint32_t y = 0; y < fb.height; y++) {
        for (uint32_t x = 0; x < fb.width; x++) {
            pixels[y * (fb.stride / 4) + x] = argb;
        }
    }
}

static void fill_color_bars(Framebuffer &fb) {
    // 8 vertical color bars: white, yellow, cyan, green, magenta, red, blue, black
    uint32_t colors[] = {
        0xFFFFFFFF, 0xFFFFFF00, 0xFF00FFFF, 0xFF00FF00,
        0xFFFF00FF, 0xFFFF0000, 0xFF0000FF, 0xFF000000,
    };
    uint32_t *pixels = reinterpret_cast<uint32_t*>(fb.data);
    uint32_t bar_w = fb.width / 8;

    for (uint32_t y = 0; y < fb.height; y++) {
        for (uint32_t x = 0; x < fb.width; x++) {
            uint32_t bar = x / bar_w;
            if (bar > 7) bar = 7;
            pixels[y * (fb.stride / 4) + x] = colors[bar];
        }
    }
}

static void fill_gradient(Framebuffer &fb, float phase) {
    uint32_t *pixels = reinterpret_cast<uint32_t*>(fb.data);
    for (uint32_t y = 0; y < fb.height; y++) {
        for (uint32_t x = 0; x < fb.width; x++) {
            float fx = (float)x / fb.width;
            float fy = (float)y / fb.height;

            uint8_t r = (uint8_t)(255.0f * (0.5f + 0.5f * sinf(fx * 6.28f + phase)));
            uint8_t g = (uint8_t)(255.0f * (0.5f + 0.5f * sinf(fy * 6.28f + phase * 1.3f)));
            uint8_t b = (uint8_t)(255.0f * (0.5f + 0.5f * sinf((fx + fy) * 3.14f + phase * 0.7f)));

            pixels[y * (fb.stride / 4) + x] = (0xFFu << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
        }
    }
}

int main() {
    printf("=== VO Display Test ===\n");

    if (!gs_sys_init()) {
        fprintf(stderr, "Failed to init AX system\n");
        return 1;
    }
    gs_cmm_print_status();

    DisplayContext display;
    if (!gs_display_init(display)) {
        fprintf(stderr, "Failed to init display\n");
        gs_sys_deinit();
        return 1;
    }

    Framebuffer fb;
    if (!gs_framebuffer_alloc(fb, display.width, display.height)) {
        fprintf(stderr, "Failed to alloc framebuffer\n");
        gs_display_deinit(display);
        gs_sys_deinit();
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Phase 1: Solid colors (2 sec each)
    struct { uint32_t color; const char *name; } solids[] = {
        {0xFFFF0000, "RED"},
        {0xFF00FF00, "GREEN"},
        {0xFF0000FF, "BLUE"},
        {0xFFFFFFFF, "WHITE"},
    };

    for (auto &s : solids) {
        if (!g_running) break;
        printf("Solid: %s\n", s.name);
        fill_solid(fb, s.color);
        gs_display_send_frame(display, fb);
        sleep(2);
    }

    // Phase 2: Color bars (3 sec)
    if (g_running) {
        printf("Color bars\n");
        fill_color_bars(fb);
        gs_display_send_frame(display, fb);
        sleep(3);
    }

    // Phase 3: Animated gradient
    if (g_running) {
        printf("Animated gradient. Press Ctrl+C to stop.\n");
    }
    int frame = 0;
    while (g_running) {
        fill_gradient(fb, frame * 0.02f);
        gs_display_send_frame(display, fb);
        frame++;
        if (frame % 60 == 0) printf("Frame %d\n", frame);
        usleep(16666);
    }

    printf("\nCleaning up...\n");
    gs_framebuffer_free(fb);
    gs_display_deinit(display);
    gs_sys_deinit();
    printf("Done.\n");
    return 0;
}
