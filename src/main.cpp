#include "gs_types.h"
#include "gs_memory.h"
#include "gs_display.h"
#include "gs_ply_loader.h"
#include "gs_scene.h"
#include "gs_camera.h"
#include "gs_renderer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <csignal>

static volatile bool g_running = true;

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
}

// Save framebuffer as PPM (simple, no library needed)
static bool save_ppm(const Framebuffer &fb, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", fb.width, fb.height);
    for (uint32_t y = 0; y < fb.height; y++) {
        const uint32_t *row = reinterpret_cast<const uint32_t*>(fb.data + y * fb.stride);
        for (uint32_t x = 0; x < fb.width; x++) {
            uint32_t pixel = row[x];  // ARGB8888
            uint8_t rgb[3] = {
                (uint8_t)((pixel >> 16) & 0xFF),  // R
                (uint8_t)((pixel >>  8) & 0xFF),  // G
                (uint8_t)( pixel        & 0xFF),  // B
            };
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
    return true;
}

static void print_usage(const char *prog) {
    printf("Usage: %s <ply_file> [options]\n", prog);
    printf("Options:\n");
    printf("  -w <width>    Render width  (default: 1920)\n");
    printf("  -h <height>   Render height (default: 1080)\n");
    printf("  -s <scale>    Render scale 1=full, 2=half, 4=quarter (default: 1)\n");
    printf("  -f <fov>      FOV in degrees (default: 60)\n");
    printf("  --info        Print PLY info and exit\n");
    printf("  --vsync       Enable vsync\n");
    printf("  --dump <dir>  Render from random viewpoints, save as JPEG, then exit\n");
    printf("  -n <count>    Number of dump frames (default: 8)\n");
    printf("\nControls:\n");
    printf("  WASD       Move camera\n");
    printf("  Q/E        Up/Down\n");
    printf("  Arrows     Rotate\n");
    printf("  +/-        Speed\n");
    printf("  O          Toggle orbit/fly\n");
    printf("  R          Reset camera\n");
    printf("  P          Save current frame as PPM\n");
    printf("  ESC        Quit\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *ply_path = nullptr;
    const char *dump_dir = nullptr;
    uint32_t width = DISPLAY_WIDTH;
    uint32_t height = DISPLAY_HEIGHT;
    int render_scale = 1;
    float fov = 60.0f;
    bool info_only = false;
    bool vsync = false;
    int dump_count = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 && i+1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i+1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) {
            render_scale = atoi(argv[++i]);
            if (render_scale < 1) render_scale = 1;
            if (render_scale > 4) render_scale = 4;
        } else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            fov = atof(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            dump_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = true;
        } else if (strcmp(argv[i], "--vsync") == 0) {
            vsync = true;
        } else if (strcmp(argv[i], "--dump") == 0 && i+1 < argc) {
            dump_dir = argv[++i];
        } else if (argv[i][0] != '-') {
            ply_path = argv[i];
        }
    }

    // Apply render scale
    uint32_t render_w = width / render_scale;
    uint32_t render_h = height / render_scale;

    if (!ply_path) {
        fprintf(stderr, "Error: No PLY file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    // Initialize system
    if (!gs_sys_init()) {
        fprintf(stderr, "Failed to initialize AX system\n");
        return 1;
    }

    gs_cmm_print_status();

    // Load PLY
    printf("Loading %s...\n", ply_path);
    GaussianScene scene;
    if (!gs_ply_load(ply_path, scene)) {
        fprintf(stderr, "Failed to load PLY: %s\n", ply_path);
        gs_sys_deinit();
        return 1;
    }

    gs_scene_print_info(scene);

    if (info_only) {
        gs_scene_free(scene);
        gs_sys_deinit();
        return 0;
    }

    // Initialize renderer (at render resolution)
    Renderer renderer;
    renderer.display.fb_vsync = vsync;
    if (!gs_renderer_init(renderer, render_w, render_h)) {
        fprintf(stderr, "Failed to initialize renderer\n");
        gs_scene_free(scene);
        gs_sys_deinit();
        return 1;
    }

    if (render_scale > 1) {
        printf("[main] Render scale: 1/%d (%ux%u -> %ux%u display)\n",
               render_scale, render_w, render_h, width, height);
    }

    // Setup camera
    Camera camera;
    camera.fov_y = fov;
    gs_camera_reset(camera, scene);

    // === Dump mode: render from random viewpoints, save images, exit ===
    if (dump_dir) {
        // Ensure dump directory exists
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", dump_dir);
        system(cmd);

        float cx = (scene.bbox_min[0] + scene.bbox_max[0]) * 0.5f;
        float cy = (scene.bbox_min[1] + scene.bbox_max[1]) * 0.5f;
        float cz = (scene.bbox_min[2] + scene.bbox_max[2]) * 0.5f;
        float dx = scene.bbox_max[0] - scene.bbox_min[0];
        float dy = scene.bbox_max[1] - scene.bbox_min[1];
        float dz = scene.bbox_max[2] - scene.bbox_min[2];
        float extent = sqrtf(dx*dx + dy*dy + dz*dz);

        // Print scene diagnostics
        printf("[dump] Scene center: (%.2f, %.2f, %.2f), extent: %.2f\n", cx, cy, cz, extent);
        printf("[dump] BBox: (%.2f..%.2f, %.2f..%.2f, %.2f..%.2f)\n",
               scene.bbox_min[0], scene.bbox_max[0],
               scene.bbox_min[1], scene.bbox_max[1],
               scene.bbox_min[2], scene.bbox_max[2]);

        // Sample some Gaussians for diagnostics
        for (int si = 0; si < 5 && si < (int)scene.num_gaussians; si++) {
            uint32_t idx = si * (scene.num_gaussians / 5);
            printf("[dump] Gaussian %u: pos=(%.3f,%.3f,%.3f) scale=(%.4f,%.4f,%.4f) opacity=%.3f\n",
                   idx, scene.pos_x[idx], scene.pos_y[idx], scene.pos_z[idx],
                   scene.scale_x[idx], scene.scale_y[idx], scene.scale_z[idx],
                   scene.opacity[idx]);
        }

        // Viewpoints: mix of default view, close orbits, and medium orbits
        for (int vi = 0; vi < dump_count; vi++) {
            float dist, angle, elev;

            if (vi == 0) {
                // Default view: same as gs_camera_reset (along +Z)
                dist = extent * 1.2f;
                angle = M_PI * 0.5f;  // +Z direction
                elev = 0.0f;
            } else if (vi <= dump_count / 2) {
                // Close orbit (0.3-0.5 * extent)
                angle = (float)(vi - 1) / (float)(dump_count / 2) * 2.0f * M_PI;
                elev = -0.15f + 0.3f * ((vi % 3) / 2.0f);
                dist = extent * (0.3f + 0.2f * ((vi % 3) / 2.0f));
            } else {
                // Medium orbit (0.6-0.9 * extent)
                angle = (float)(vi - dump_count/2) / (float)(dump_count - dump_count/2) * 2.0f * M_PI;
                elev = -0.25f + 0.5f * ((vi % 3) / 2.0f);
                dist = extent * (0.6f + 0.3f * ((vi % 3) / 2.0f));
            }

            camera.pos[0] = cx + dist * cosf(elev) * cosf(angle);
            camera.pos[1] = cy + dist * sinf(elev);
            camera.pos[2] = cz + dist * cosf(elev) * sinf(angle);

            // Look at center
            float look_dx = cx - camera.pos[0];
            float look_dy = cy - camera.pos[1];
            float look_dz = cz - camera.pos[2];
            float look_len = sqrtf(look_dx*look_dx + look_dz*look_dz);
            camera.yaw = atan2f(look_dz, look_dx) * (180.0f / M_PI);
            camera.pitch = atan2f(look_dy, look_len) * (180.0f / M_PI);
            camera.target[0] = cx;
            camera.target[1] = cy;
            camera.target[2] = cz;

            CameraParams cam_params;
            gs_camera_get_params(camera, render_w, render_h, cam_params);
            gs_renderer_render_frame(renderer, scene, cam_params);

            const RenderStats &stats = gs_renderer_get_stats(renderer);

            // Save PPM
            Framebuffer &fb = renderer.framebuffers[1 - renderer.current_fb];  // last rendered
            char ppm_path[512], jpg_path[512];
            snprintf(ppm_path, sizeof(ppm_path), "%s/frame_%02d.ppm", dump_dir, vi);
            snprintf(jpg_path, sizeof(jpg_path), "%s/frame_%02d.jpg", dump_dir, vi);

            save_ppm(fb, ppm_path);

            // Convert to JPEG via Python/Pillow
            char convert_cmd[1024];
            snprintf(convert_cmd, sizeof(convert_cmd),
                     "python3 -c \"from PIL import Image; Image.open('%s').save('%s', quality=90)\" 2>/dev/null",
                     ppm_path, jpg_path);
            system(convert_cmd);

            // Remove PPM if JPEG was created
            snprintf(convert_cmd, sizeof(convert_cmd), "[ -f '%s' ] && rm '%s'", jpg_path, ppm_path);
            system(convert_cmd);

            printf("[dump] Frame %d/%d: pos=(%.1f, %.1f, %.1f) yaw=%.0f pitch=%.0f | "
                   "Visible: %u/%u | %.1fms -> %s\n",
                   vi + 1, dump_count,
                   camera.pos[0], camera.pos[1], camera.pos[2],
                   camera.yaw, camera.pitch,
                   stats.num_visible, scene.num_gaussians,
                   stats.time_total_ms, jpg_path);
        }

        printf("[dump] Done. Saved %d frames to %s/\n", dump_count, dump_dir);

        gs_renderer_deinit(renderer);
        gs_scene_free(scene);
        gs_sys_deinit();
        return 0;
    }

    // === Interactive mode ===
    gs_camera_init_input();

    // Signal handler for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\n=== Rendering ===\n");
    printf("Press ESC to quit, WASD to move, arrows to rotate, P to save frame\n\n");

    uint32_t frame_count = 0;
    uint32_t save_count = 0;
    float fps_accum = 0.0f;

    while (g_running) {
        // Process input
        if (!gs_camera_update(camera)) {
            break;  // ESC pressed
        }

        // Get camera parameters (at render resolution)
        CameraParams cam_params;
        gs_camera_get_params(camera, render_w, render_h, cam_params);

        // Render frame
        gs_renderer_render_frame(renderer, scene, cam_params);

        // Print stats periodically
        const RenderStats &stats = gs_renderer_get_stats(renderer);
        fps_accum += stats.fps;
        frame_count++;

        if (frame_count % 30 == 0) {
            float avg_fps = fps_accum / 30.0f;
            printf("\rFPS: %.1f | Visible: %u/%u | Proj: %.1fms Sort: %.1fms Rast: %.1fms Total: %.1fms | Tiles: %u  ",
                   avg_fps, stats.num_visible, scene.num_gaussians,
                   stats.time_project_ms, stats.time_sort_ms,
                   stats.time_raster_ms, stats.time_total_ms,
                   stats.num_tiles_active);
            fflush(stdout);
            fps_accum = 0.0f;
        }
    }

    printf("\n\nShutting down...\n");

    gs_camera_restore_input();
    gs_renderer_deinit(renderer);
    gs_scene_free(scene);
    gs_sys_deinit();

    printf("Done.\n");
    return 0;
}
