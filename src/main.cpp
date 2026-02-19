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
    printf("  --npu         Enable NPU super-resolution upscale (requires trained .axmodel)\n");
    printf("  --vsync       Enable vsync\n");
    printf("  --dump <dir>  Render from random viewpoints, save as JPEG, then exit\n");
    printf("  -n <count>    Number of dump frames (default: 8)\n");
    printf("  --sh-degree <N>  Cap SH evaluation degree (0-3, default: 3)\n");
    printf("  --bench <N>   Benchmark mode: render N frames with orbit camera, print timing, exit\n");
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
    bool use_npu = false;
    int dump_count = 8;
    int max_sh_degree = MAX_SH_DEGREE;
    int bench_frames = 0;

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
        } else if (strcmp(argv[i], "--npu") == 0) {
            use_npu = true;
        } else if (strcmp(argv[i], "--dump") == 0 && i+1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--sh-degree") == 0 && i+1 < argc) {
            max_sh_degree = atoi(argv[++i]);
            if (max_sh_degree < 0) max_sh_degree = 0;
            if (max_sh_degree > 3) max_sh_degree = 3;
        } else if (strcmp(argv[i], "--bench") == 0 && i+1 < argc) {
            bench_frames = atoi(argv[++i]);
            if (bench_frames < 1) bench_frames = 1;
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

    // Initialize system (skip for dump mode - no AX hardware needed)
    bool headless = (dump_dir != nullptr);
    if (!headless) {
        if (!gs_sys_init()) {
            fprintf(stderr, "Failed to initialize AX system\n");
            return 1;
        }
        gs_cmm_print_status();
    }

    // Load PLY
    printf("Loading %s...\n", ply_path);
    GaussianScene scene;
    if (!gs_ply_load(ply_path, scene)) {
        fprintf(stderr, "Failed to load PLY: %s\n", ply_path);
        if (!headless) gs_sys_deinit();
        return 1;
    }

    gs_scene_print_info(scene);

    if (info_only) {
        gs_scene_free(scene);
        if (!headless) gs_sys_deinit();
        return 0;
    }

    // Initialize renderer (at render resolution)
    Renderer renderer;
    renderer.display.fb_vsync = vsync;
    if (!gs_renderer_init(renderer, render_w, render_h, headless, use_npu)) {
        fprintf(stderr, "Failed to initialize renderer\n");
        gs_scene_free(scene);
        if (!headless) gs_sys_deinit();
        return 1;
    }

    renderer.max_sh_degree = max_sh_degree;
    if (max_sh_degree < MAX_SH_DEGREE) {
        printf("[main] SH degree capped: %d (scene has %d)\n", max_sh_degree, MAX_SH_DEGREE);
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

        // Compute robust scene center using mean of Gaussian positions
        // (much better than bbox center which is skewed by outliers)
        float cx = 0, cy = 0, cz = 0;
        for (uint32_t gi = 0; gi < scene.num_gaussians; gi++) {
            cx += scene.pos_x[gi];
            cy += scene.pos_y[gi];
            cz += scene.pos_z[gi];
        }
        cx /= scene.num_gaussians;
        cy /= scene.num_gaussians;
        cz /= scene.num_gaussians;

        // Compute typical distance of Gaussians from center (use RMS for robustness)
        float rms_dist = 0;
        for (uint32_t gi = 0; gi < scene.num_gaussians; gi++) {
            float ddx = scene.pos_x[gi] - cx;
            float ddy = scene.pos_y[gi] - cy;
            float ddz = scene.pos_z[gi] - cz;
            rms_dist += ddx*ddx + ddy*ddy + ddz*ddz;
        }
        rms_dist = sqrtf(rms_dist / scene.num_gaussians);

        // Use RMS distance as a better "extent" for camera placement
        // Camera should be at roughly 1-2x RMS distance from center
        float scene_radius = rms_dist;

        // Print scene diagnostics
        printf("[dump] Mean center: (%.2f, %.2f, %.2f), RMS radius: %.2f\n", cx, cy, cz, scene_radius);
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

        // Viewpoints: orbit around mean center at various distances
        // Distance based on RMS radius (much closer than bbox diagonal)
        for (int vi = 0; vi < dump_count; vi++) {
            float dist, angle, elev;

            if (vi == 0) {
                // Front view: close
                dist = scene_radius * 1.5f;
                angle = M_PI * 0.5f;
                elev = 0.0f;
            } else if (vi <= dump_count / 2) {
                // Close orbit (0.5-1.0 * radius)
                angle = (float)(vi - 1) / (float)(dump_count / 2) * 2.0f * M_PI;
                elev = -0.15f + 0.3f * ((vi % 3) / 2.0f);
                dist = scene_radius * (0.5f + 0.5f * ((vi % 3) / 2.0f));
            } else {
                // Medium orbit (1.0-2.0 * radius)
                angle = (float)(vi - dump_count/2) / (float)(dump_count - dump_count/2) * 2.0f * M_PI;
                elev = -0.25f + 0.5f * ((vi % 3) / 2.0f);
                dist = scene_radius * (1.0f + 1.0f * ((vi % 3) / 2.0f));
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

            // Print projection diagnostics for first frame
            if (vi == 0) {
                const ProjectedGaussian *pgs = renderer.projection.gaussians;
                uint32_t pc = renderer.projection.count;
                float max_radius = 0, avg_radius = 0;
                uint32_t spike_count = 0;  // eigenvalue ratio > 100
                float max_eigen_ratio = 0;
                for (uint32_t pi = 0; pi < pc; pi++) {
                    float a = pgs[pi].cov2d_a, b = pgs[pi].cov2d_b, c = pgs[pi].cov2d_c;
                    float det = a * c - b * b;
                    float mid = 0.5f * (a + c);
                    float disc = mid * mid - det;
                    float lmax = mid + sqrtf(fmaxf(0.0f, disc));
                    float lmin = mid - sqrtf(fmaxf(0.0f, disc));
                    if (lmin < 0.001f) lmin = 0.001f;
                    float ratio = lmax / lmin;
                    if (ratio > max_eigen_ratio) max_eigen_ratio = ratio;
                    if (ratio > 100.0f) spike_count++;
                    avg_radius += pgs[pi].radius;
                    if (pgs[pi].radius > max_radius) max_radius = pgs[pi].radius;
                }
                avg_radius /= (pc > 0 ? pc : 1);
                printf("[diag] Projected: %u gaussians\n", pc);
                printf("[diag] Radius: avg=%.1f max=%.1f\n", avg_radius, max_radius);
                printf("[diag] Eigenvalue ratio: max=%.1f, spikes(>100x)=%u (%.1f%%)\n",
                       max_eigen_ratio, spike_count, 100.0f * spike_count / (pc > 0 ? pc : 1));
                // Sample a few projected Gaussians
                for (uint32_t si = 0; si < 5 && si < pc; si++) {
                    uint32_t pi = si * (pc / 5);
                    printf("[diag] PG[%u]: screen=(%.1f,%.1f) depth=%.2f radius=%.1f "
                           "cov=(%.2f,%.2f,%.2f) color=(%.2f,%.2f,%.2f) opacity=%.2f\n",
                           pi, pgs[pi].screen_x, pgs[pi].screen_y, pgs[pi].depth,
                           pgs[pi].radius, pgs[pi].cov2d_a, pgs[pi].cov2d_b, pgs[pi].cov2d_c,
                           pgs[pi].color_r, pgs[pi].color_g, pgs[pi].color_b, pgs[pi].opacity);
                }
            }

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
        if (!headless) gs_sys_deinit();
        return 0;
    }

    // === Benchmark mode: render N frames with orbit camera, print timing ===
    if (bench_frames > 0) {
        // Compute scene center and radius (same as dump mode)
        float cx = 0, cy = 0, cz = 0;
        for (uint32_t gi = 0; gi < scene.num_gaussians; gi++) {
            cx += scene.pos_x[gi]; cy += scene.pos_y[gi]; cz += scene.pos_z[gi];
        }
        cx /= scene.num_gaussians; cy /= scene.num_gaussians; cz /= scene.num_gaussians;
        float rms_dist = 0;
        for (uint32_t gi = 0; gi < scene.num_gaussians; gi++) {
            float ddx = scene.pos_x[gi] - cx, ddy = scene.pos_y[gi] - cy, ddz = scene.pos_z[gi] - cz;
            rms_dist += ddx*ddx + ddy*ddy + ddz*ddz;
        }
        rms_dist = sqrtf(rms_dist / scene.num_gaussians);
        float dist = rms_dist * 1.5f;

        printf("\n=== Benchmark: %d frames, %ux%u, SH degree %d ===\n",
               bench_frames, render_w, render_h, max_sh_degree);
        printf("%5s %8s %8s %8s %8s %8s %8s\n",
               "Frame", "Proj", "Sort", "Raster", "Upscale", "Total", "FPS");

        float sum_proj = 0, sum_sort = 0, sum_raster = 0, sum_upscale = 0, sum_total = 0;

        for (int fi = 0; fi < bench_frames; fi++) {
            float angle = (float)fi / (float)bench_frames * 2.0f * M_PI;
            float elev = 0.1f * sinf(angle * 3.0f);  // gentle vertical oscillation

            camera.pos[0] = cx + dist * cosf(elev) * cosf(angle);
            camera.pos[1] = cy + dist * sinf(elev);
            camera.pos[2] = cz + dist * cosf(elev) * sinf(angle);

            float look_dx = cx - camera.pos[0], look_dz = cz - camera.pos[2];
            float look_dy = cy - camera.pos[1];
            float look_len = sqrtf(look_dx*look_dx + look_dz*look_dz);
            camera.yaw = atan2f(look_dz, look_dx) * (180.0f / M_PI);
            camera.pitch = atan2f(look_dy, look_len) * (180.0f / M_PI);
            camera.target[0] = cx; camera.target[1] = cy; camera.target[2] = cz;

            CameraParams cam_params;
            gs_camera_get_params(camera, render_w, render_h, cam_params);
            gs_renderer_render_frame(renderer, scene, cam_params);

            const RenderStats &s = gs_renderer_get_stats(renderer);
            printf("%5d %7.1fms %7.1fms %7.1fms %7.1fms %7.1fms %7.1f\n",
                   fi, s.time_project_ms, s.time_sort_ms, s.time_raster_ms,
                   s.time_upscale_ms, s.time_total_ms, s.fps);

            sum_proj += s.time_project_ms;
            sum_sort += s.time_sort_ms;
            sum_raster += s.time_raster_ms;
            sum_upscale += s.time_upscale_ms;
            sum_total += s.time_total_ms;
        }

        float n = (float)bench_frames;
        printf("──────────────────────────────────────────────────────\n");
        printf("  AVG %7.1fms %7.1fms %7.1fms %7.1fms %7.1fms %7.1f\n",
               sum_proj/n, sum_sort/n, sum_raster/n, sum_upscale/n, sum_total/n,
               1000.0f / (sum_total/n));
        printf("\nVisible (last frame): %u/%u | Tiles: %u\n",
               renderer.stats.num_visible, scene.num_gaussians,
               renderer.stats.num_tiles_active);

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
            if (stats.time_upscale_ms > 0) {
                printf("\rFPS: %.1f | Visible: %u/%u | Proj: %.1fms Sort: %.1fms Rast: %.1fms NPU: %.1fms Total: %.1fms | Tiles: %u  ",
                       avg_fps, stats.num_visible, scene.num_gaussians,
                       stats.time_project_ms, stats.time_sort_ms,
                       stats.time_raster_ms, stats.time_upscale_ms,
                       stats.time_total_ms, stats.num_tiles_active);
            } else if (stats.mau_tiles > 0) {
                printf("\rFPS: %.1f | Visible: %u/%u | Proj: %.1fms Sort: %.1fms Rast: %.1fms Total: %.1fms | Tiles: %u (MAU:%u CPU:%u)  ",
                       avg_fps, stats.num_visible, scene.num_gaussians,
                       stats.time_project_ms, stats.time_sort_ms,
                       stats.time_raster_ms, stats.time_total_ms,
                       stats.num_tiles_active, stats.mau_tiles, stats.cpu_tiles);
            } else {
                printf("\rFPS: %.1f | Visible: %u/%u | Proj: %.1fms Sort: %.1fms Rast: %.1fms Total: %.1fms | Tiles: %u  ",
                       avg_fps, stats.num_visible, scene.num_gaussians,
                       stats.time_project_ms, stats.time_sort_ms,
                       stats.time_raster_ms, stats.time_total_ms,
                       stats.num_tiles_active);
            }
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
