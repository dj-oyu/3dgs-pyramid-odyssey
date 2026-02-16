#pragma once

#include "gs_types.h"

enum class CameraMode {
    Fly,
    Orbit
};

struct Camera {
    float pos[3]     = {0.0f, 0.0f, 3.0f};
    float target[3]  = {0.0f, 0.0f, 0.0f};
    float up[3]      = {0.0f, 1.0f, 0.0f};
    float yaw         = -90.0f;  // degrees
    float pitch        = 0.0f;   // degrees
    float fov_y        = 60.0f;  // degrees
    float near_plane   = 0.1f;
    float far_plane    = 100.0f;
    float move_speed   = 0.1f;
    float rotate_speed = 2.0f;
    float orbit_radius = 3.0f;
    CameraMode mode    = CameraMode::Fly;
    // Stored from scene reset for 'R' key
    float reset_pos[3] = {0.0f, 0.0f, 3.0f};
    float reset_speed  = 0.1f;
};

// Initialize non-blocking keyboard input
void gs_camera_init_input();
void gs_camera_restore_input();

// Process keyboard input and update camera state
// Returns false if user pressed ESC (quit)
bool gs_camera_update(Camera &cam);

// Compute view and projection matrices from camera state
void gs_camera_get_params(const Camera &cam, uint32_t width, uint32_t height, CameraParams &params);

// Reset camera to default position looking at scene center
void gs_camera_reset(Camera &cam, const GaussianScene &scene);
