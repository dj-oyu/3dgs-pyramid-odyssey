#include "gs_camera.h"
#include "gs_math.h"
#include <cstdio>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>

static struct termios orig_termios;
static bool input_initialized = false;

void gs_camera_init_input() {
    if (input_initialized) return;
    struct termios raw;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);

    // Set stdin to non-blocking
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    input_initialized = true;
}

void gs_camera_restore_input() {
    if (!input_initialized) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
    input_initialized = false;
}

static int read_key() {
    unsigned char c;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == 27) {
            // Escape sequence
            unsigned char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) != 1) return 27;  // ESC
            if (read(STDIN_FILENO, &seq[1], 1) != 1) return 27;
            if (seq[0] == '[') {
                switch (seq[1]) {
                    case 'A': return 256;  // UP
                    case 'B': return 257;  // DOWN
                    case 'C': return 258;  // RIGHT
                    case 'D': return 259;  // LEFT
                }
            }
            return 27;
        }
        return c;
    }
    return -1;
}

bool gs_camera_update(Camera &cam) {
    int key;
    while ((key = read_key()) >= 0) {
        float rad_yaw = cam.yaw * (M_PI / 180.0f);
        float rad_pitch = cam.pitch * (M_PI / 180.0f);

        // Forward direction
        float fwd[3] = {
            cosf(rad_pitch) * cosf(rad_yaw),
            sinf(rad_pitch),
            cosf(rad_pitch) * sinf(rad_yaw)
        };

        // Right direction
        float right[3] = {
            -sinf(rad_yaw),
            0.0f,
            cosf(rad_yaw)
        };

        switch (key) {
            case 27:  // ESC
                return false;

            case 'w': case 'W':
                cam.pos[0] += fwd[0] * cam.move_speed;
                cam.pos[1] += fwd[1] * cam.move_speed;
                cam.pos[2] += fwd[2] * cam.move_speed;
                break;
            case 's': case 'S':
                cam.pos[0] -= fwd[0] * cam.move_speed;
                cam.pos[1] -= fwd[1] * cam.move_speed;
                cam.pos[2] -= fwd[2] * cam.move_speed;
                break;
            case 'a': case 'A':
                cam.pos[0] -= right[0] * cam.move_speed;
                cam.pos[2] -= right[2] * cam.move_speed;
                break;
            case 'd': case 'D':
                cam.pos[0] += right[0] * cam.move_speed;
                cam.pos[2] += right[2] * cam.move_speed;
                break;
            case 'q': case 'Q':
                cam.pos[1] += cam.move_speed;
                break;
            case 'e': case 'E':
                cam.pos[1] -= cam.move_speed;
                break;

            case 256:  // UP arrow
                cam.pitch += cam.rotate_speed;
                if (cam.pitch > 89.0f) cam.pitch = 89.0f;
                break;
            case 257:  // DOWN arrow
                cam.pitch -= cam.rotate_speed;
                if (cam.pitch < -89.0f) cam.pitch = -89.0f;
                break;
            case 258:  // RIGHT arrow
                cam.yaw += cam.rotate_speed;
                break;
            case 259:  // LEFT arrow
                cam.yaw -= cam.rotate_speed;
                break;

            case '+': case '=':
                cam.move_speed *= 1.5f;
                printf("Speed: %.3f\n", cam.move_speed);
                break;
            case '-': case '_':
                cam.move_speed /= 1.5f;
                printf("Speed: %.3f\n", cam.move_speed);
                break;

            case 'o': case 'O':
                cam.mode = (cam.mode == CameraMode::Fly) ? CameraMode::Orbit : CameraMode::Fly;
                printf("Camera mode: %s\n", cam.mode == CameraMode::Fly ? "Fly" : "Orbit");
                break;

            case 'r': case 'R':
                cam.pos[0] = cam.reset_pos[0];
                cam.pos[1] = cam.reset_pos[1];
                cam.pos[2] = cam.reset_pos[2];
                cam.yaw = -90; cam.pitch = 0;
                cam.move_speed = cam.reset_speed;
                printf("Camera reset\n");
                break;
        }
    }

    // Update target based on yaw/pitch for fly mode
    if (cam.mode == CameraMode::Fly) {
        float rad_yaw = cam.yaw * (M_PI / 180.0f);
        float rad_pitch = cam.pitch * (M_PI / 180.0f);
        cam.target[0] = cam.pos[0] + cosf(rad_pitch) * cosf(rad_yaw);
        cam.target[1] = cam.pos[1] + sinf(rad_pitch);
        cam.target[2] = cam.pos[2] + cosf(rad_pitch) * sinf(rad_yaw);
    }

    return true;
}

void gs_camera_get_params(const Camera &cam, uint32_t width, uint32_t height, CameraParams &params) {
    params.position[0] = cam.pos[0];
    params.position[1] = cam.pos[1];
    params.position[2] = cam.pos[2];
    params.fov_y = cam.fov_y * (M_PI / 180.0f);
    params.aspect = static_cast<float>(width) / static_cast<float>(height);
    params.near_plane = cam.near_plane;
    params.far_plane = cam.far_plane;
    params.width = width;
    params.height = height;

    float up[3] = {cam.up[0], cam.up[1], cam.up[2]};
    mat4_look_at(cam.pos, cam.target, up, params.view_matrix);
    mat4_perspective(params.fov_y, params.aspect, params.near_plane, params.far_plane, params.proj_matrix);
}

void gs_camera_reset(Camera &cam, const GaussianScene &scene) {
    // Use mean position for center (robust against outliers)
    float cx = 0, cy = 0, cz = 0;
    for (uint32_t i = 0; i < scene.num_gaussians; i++) {
        cx += scene.pos_x[i];
        cy += scene.pos_y[i];
        cz += scene.pos_z[i];
    }
    cx /= scene.num_gaussians;
    cy /= scene.num_gaussians;
    cz /= scene.num_gaussians;

    // RMS distance from center as scene radius
    float rms_dist = 0;
    for (uint32_t i = 0; i < scene.num_gaussians; i++) {
        float dx = scene.pos_x[i] - cx;
        float dy = scene.pos_y[i] - cy;
        float dz = scene.pos_z[i] - cz;
        rms_dist += dx*dx + dy*dy + dz*dz;
    }
    float extent = sqrtf(rms_dist / scene.num_gaussians);

    // Position camera at 1.5x RMS distance
    float dist = extent * 1.5f;
    cam.pos[0] = cx;
    cam.pos[1] = cy;
    cam.pos[2] = cz + dist;
    cam.target[0] = cx;
    cam.target[1] = cy;
    cam.target[2] = cz;
    cam.yaw = -90.0f;
    cam.pitch = 0.0f;
    cam.orbit_radius = dist;
    cam.move_speed = extent * 0.02f;
    if (cam.move_speed < 0.01f) cam.move_speed = 0.01f;

    // Adjust near/far planes to scene size
    cam.near_plane = 0.01f;
    cam.far_plane = extent * 50.0f;
    if (cam.far_plane < 100.0f) cam.far_plane = 100.0f;

    // Store for 'R' key reset
    cam.reset_pos[0] = cam.pos[0];
    cam.reset_pos[1] = cam.pos[1];
    cam.reset_pos[2] = cam.pos[2];
    cam.reset_speed = cam.move_speed;

    printf("[gs_camera] Reset: pos=(%.2f, %.2f, %.2f), target=(%.2f, %.2f, %.2f), speed=%.3f\n",
           cam.pos[0], cam.pos[1], cam.pos[2],
           cam.target[0], cam.target[1], cam.target[2], cam.move_speed);
}
