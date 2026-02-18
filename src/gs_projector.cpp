#include "gs_projector.h"
#include "gs_math.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <arm_neon.h>

// Max eigenvalue ratio for 2D covariance (prevents spike artifacts)
static constexpr float MAX_EIGEN_RATIO = 200.0f;

// --- Allocation ---

bool gs_projection_alloc(ProjectionResult &result, uint32_t max_gaussians) {
    size_t size = max_gaussians * sizeof(ProjectedGaussian);
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0 || !ptr) return false;
    result.gaussians = static_cast<ProjectedGaussian*>(ptr);
    result.capacity = max_gaussians;
    result.count = 0;
    return true;
}

void gs_projection_free(ProjectionResult &result) {
    free(result.gaussians);
    result.gaussians = nullptr;
    result.capacity = 0;
    result.count = 0;
}

bool gs_projection_batch_alloc(ProjectionBatch &batch, uint32_t max_gaussians) {
    // Single contiguous allocation: 14 floats + 1 uint32 per Gaussian = 60 bytes each
    size_t per_gauss = 14 * sizeof(float) + 1 * sizeof(uint32_t);
    size_t total = (size_t)max_gaussians * per_gauss;
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, total) != 0 || !ptr) return false;
    memset(ptr, 0, total);

    batch.backing_ptr = ptr;
    batch.backing_size = (uint32_t)total;
    batch.capacity = max_gaussians;
    batch.mem_type = ProjMemType::HEAP;

    // Layout sub-arrays within the contiguous block
    uint8_t *p = static_cast<uint8_t*>(ptr);
    uint32_t N = max_gaussians;

    batch.visible_indices = reinterpret_cast<uint32_t*>(p);  p += N * sizeof(uint32_t);
    batch.cam_x   = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.cam_y   = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.cam_z   = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.screen_x = reinterpret_cast<float*>(p); p += N * sizeof(float);
    batch.screen_y = reinterpret_cast<float*>(p); p += N * sizeof(float);
    batch.depth    = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.cov2d_a  = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.cov2d_b  = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.cov2d_c  = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.radius   = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.color_r  = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.color_g  = reinterpret_cast<float*>(p);  p += N * sizeof(float);
    batch.color_b  = reinterpret_cast<float*>(p);  p += N * sizeof(float);

    batch.cull_count = 0;
    batch.cov_count = 0;

    return true;
}

void gs_projection_batch_free(ProjectionBatch &batch) {
    free(batch.backing_ptr);
    batch.backing_ptr = nullptr;
    batch.visible_indices = nullptr;
    batch.cam_x = batch.cam_y = batch.cam_z = nullptr;
    batch.screen_x = batch.screen_y = batch.depth = nullptr;
    batch.cov2d_a = batch.cov2d_b = batch.cov2d_c = nullptr;
    batch.radius = nullptr;
    batch.color_r = batch.color_g = batch.color_b = nullptr;
    batch.capacity = 0;
    batch.backing_size = 0;
    batch.cull_count = 0;
    batch.cov_count = 0;
}

// --- Static helpers (unchanged) ---

// Compute 3D covariance matrix from quaternion and scale
static void compute_cov3d(float qw, float qx, float qy, float qz,
                           float sx, float sy, float sz,
                           float cov[6]) {
    float r00 = 1.0f - 2.0f*(qy*qy + qz*qz);
    float r01 = 2.0f*(qx*qy - qz*qw);
    float r02 = 2.0f*(qx*qz + qy*qw);
    float r10 = 2.0f*(qx*qy + qz*qw);
    float r11 = 1.0f - 2.0f*(qx*qx + qz*qz);
    float r12 = 2.0f*(qy*qz - qx*qw);
    float r20 = 2.0f*(qx*qz - qy*qw);
    float r21 = 2.0f*(qy*qz + qx*qw);
    float r22 = 1.0f - 2.0f*(qx*qx + qy*qy);

    float m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
    float m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
    float m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;

    cov[0] = m00*m00 + m01*m01 + m02*m02;
    cov[1] = m00*m10 + m01*m11 + m02*m12;
    cov[2] = m00*m20 + m01*m21 + m02*m22;
    cov[3] = m10*m10 + m11*m11 + m12*m12;
    cov[4] = m10*m20 + m11*m21 + m12*m22;
    cov[5] = m20*m20 + m21*m21 + m22*m22;
}

// Project 3D covariance to 2D given Jacobian of perspective projection
static void project_cov2d(const float cov3d[6],
                           const float *view_matrix,
                           float tx, float ty, float tz,
                           float focal_x, float focal_y,
                           float tan_fovx, float tan_fovy,
                           float &cov2d_a, float &cov2d_b, float &cov2d_c) {
    float inv_z = 1.0f / tz;
    float inv_z2 = inv_z * inv_z;

    float z = -tz;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    tx = fminf(limx, fmaxf(-limx, tx / z)) * z;
    ty = fminf(limy, fmaxf(-limy, ty / z)) * z;

    float j00 = -focal_x * inv_z;
    float j02 = focal_x * tx * inv_z2;
    float j11 = -focal_y * inv_z;
    float j12 = focal_y * ty * inv_z2;

    float w00 = view_matrix[0], w01 = view_matrix[4], w02 = view_matrix[8];
    float w10 = view_matrix[1], w11 = view_matrix[5], w12 = view_matrix[9];
    float w20 = view_matrix[2], w21 = view_matrix[6], w22 = view_matrix[10];

    float t00 = j00*w00 + j02*w20;
    float t01 = j00*w01 + j02*w21;
    float t02 = j00*w02 + j02*w22;
    float t10 = j11*w10 + j12*w20;
    float t11 = j11*w11 + j12*w21;
    float t12 = j11*w12 + j12*w22;

    float m00_v = t00*cov3d[0] + t01*cov3d[1] + t02*cov3d[2];
    float m01_v = t00*cov3d[1] + t01*cov3d[3] + t02*cov3d[4];
    float m02_v = t00*cov3d[2] + t01*cov3d[4] + t02*cov3d[5];
    float m10_v = t10*cov3d[0] + t11*cov3d[1] + t12*cov3d[2];
    float m11_v = t10*cov3d[1] + t11*cov3d[3] + t12*cov3d[4];
    float m12_v = t10*cov3d[2] + t11*cov3d[4] + t12*cov3d[5];

    cov2d_a = m00_v*t00 + m01_v*t01 + m02_v*t02;
    cov2d_b = m00_v*t10 + m01_v*t11 + m02_v*t12;
    cov2d_c = m10_v*t10 + m11_v*t11 + m12_v*t12;

    cov2d_a += 0.3f;
    cov2d_c += 0.3f;
}

// Evaluate SH color (degrees 0-3)
static void eval_sh(int degree,
                     float dc_r, float dc_g, float dc_b,
                     const float *sh_rest, int basis_per_channel,
                     float dir_x, float dir_y, float dir_z,
                     float &r, float &g, float &b) {
    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_C1 = 0.4886025119029199f;
    constexpr float SH_C2[] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    };
    constexpr float SH_C3[] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.4453057213202769f,
        -0.5900435899266435f
    };

    float x = dir_x, y = dir_y, z = dir_z;

    r = SH_C0 * dc_r;
    g = SH_C0 * dc_g;
    b = SH_C0 * dc_b;

    if (degree >= 1 && sh_rest) {
        const float *sr = sh_rest;
        const float *sg = sh_rest + basis_per_channel;
        const float *sb = sh_rest + 2 * basis_per_channel;

        r += SH_C1 * (-y * sr[0] + z * sr[1] - x * sr[2]);
        g += SH_C1 * (-y * sg[0] + z * sg[1] - x * sg[2]);
        b += SH_C1 * (-y * sb[0] + z * sb[1] - x * sb[2]);

        if (degree >= 2) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            r += SH_C2[0] * xy * sr[3] + SH_C2[1] * yz * sr[4] +
                 SH_C2[2] * (2.0f * zz - xx - yy) * sr[5] +
                 SH_C2[3] * xz * sr[6] + SH_C2[4] * (xx - yy) * sr[7];
            g += SH_C2[0] * xy * sg[3] + SH_C2[1] * yz * sg[4] +
                 SH_C2[2] * (2.0f * zz - xx - yy) * sg[5] +
                 SH_C2[3] * xz * sg[6] + SH_C2[4] * (xx - yy) * sg[7];
            b += SH_C2[0] * xy * sb[3] + SH_C2[1] * yz * sb[4] +
                 SH_C2[2] * (2.0f * zz - xx - yy) * sb[5] +
                 SH_C2[3] * xz * sb[6] + SH_C2[4] * (xx - yy) * sb[7];

            if (degree >= 3) {
                r += SH_C3[0] * y * (3.0f * xx - yy) * sr[8] +
                     SH_C3[1] * xy * z * sr[9] +
                     SH_C3[2] * y * (4.0f * zz - xx - yy) * sr[10] +
                     SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sr[11] +
                     SH_C3[4] * x * (4.0f * zz - xx - yy) * sr[12] +
                     SH_C3[5] * z * (xx - yy) * sr[13] +
                     SH_C3[6] * x * (xx - 3.0f * yy) * sr[14];
                g += SH_C3[0] * y * (3.0f * xx - yy) * sg[8] +
                     SH_C3[1] * xy * z * sg[9] +
                     SH_C3[2] * y * (4.0f * zz - xx - yy) * sg[10] +
                     SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sg[11] +
                     SH_C3[4] * x * (4.0f * zz - xx - yy) * sg[12] +
                     SH_C3[5] * z * (xx - yy) * sg[13] +
                     SH_C3[6] * x * (xx - 3.0f * yy) * sg[14];
                b += SH_C3[0] * y * (3.0f * xx - yy) * sb[8] +
                     SH_C3[1] * xy * z * sb[9] +
                     SH_C3[2] * y * (4.0f * zz - xx - yy) * sb[10] +
                     SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sb[11] +
                     SH_C3[4] * x * (4.0f * zz - xx - yy) * sb[12] +
                     SH_C3[5] * z * (xx - yy) * sb[13] +
                     SH_C3[6] * x * (xx - 3.0f * yy) * sb[14];
            }
        }
    }

    r = fmaxf(0.0f, fminf(1.0f, r + 0.5f));
    g = fmaxf(0.0f, fminf(1.0f, g + 0.5f));
    b = fmaxf(0.0f, fminf(1.0f, b + 0.5f));
}

// --- Phase 1: View transform + frustum cull ---

uint32_t gs_project_cull(const GaussianScene &scene, const CameraParams &cam,
                          ProjectionBatch &batch) {
    batch.cull_count = 0;
    if (scene.num_gaussians == 0) return 0;

    const float *V = cam.view_matrix;
    uint32_t n = scene.num_gaussians;
    uint32_t out = 0;

    // NEON broadcast view matrix rows
    float32x4_t v_r0x = vdupq_n_f32(V[0]),  v_r0y = vdupq_n_f32(V[4]);
    float32x4_t v_r0z = vdupq_n_f32(V[8]),  v_r0w = vdupq_n_f32(V[12]);
    float32x4_t v_r1x = vdupq_n_f32(V[1]),  v_r1y = vdupq_n_f32(V[5]);
    float32x4_t v_r1z = vdupq_n_f32(V[9]),  v_r1w = vdupq_n_f32(V[13]);
    float32x4_t v_r2x = vdupq_n_f32(V[2]),  v_r2y = vdupq_n_f32(V[6]);
    float32x4_t v_r2z = vdupq_n_f32(V[10]), v_r2w = vdupq_n_f32(V[14]);

    float32x4_t v_neg_near = vdupq_n_f32(-cam.near_plane);
    float32x4_t v_far = vdupq_n_f32(cam.far_plane);

    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        if (i + 8 <= n) {
            __builtin_prefetch(&scene.pos_x[i + 8], 0, 1);
            __builtin_prefetch(&scene.pos_y[i + 8], 0, 1);
            __builtin_prefetch(&scene.pos_z[i + 8], 0, 1);
        }

        float32x4_t px = vld1q_f32(&scene.pos_x[i]);
        float32x4_t py = vld1q_f32(&scene.pos_y[i]);
        float32x4_t pz = vld1q_f32(&scene.pos_z[i]);

        float32x4_t vx = vmlaq_f32(v_r0w, v_r0x, px);
        vx = vmlaq_f32(vx, v_r0y, py);
        vx = vmlaq_f32(vx, v_r0z, pz);

        float32x4_t vy = vmlaq_f32(v_r1w, v_r1x, px);
        vy = vmlaq_f32(vy, v_r1y, py);
        vy = vmlaq_f32(vy, v_r1z, pz);

        float32x4_t vz = vmlaq_f32(v_r2w, v_r2x, px);
        vz = vmlaq_f32(vz, v_r2y, py);
        vz = vmlaq_f32(vz, v_r2z, pz);

        float32x4_t neg_vz = vnegq_f32(vz);
        uint32x4_t mask = vandq_u32(vcltq_f32(vz, v_neg_near),
                                     vcleq_f32(neg_vz, v_far));
        if (vmaxvq_u32(mask) == 0) continue;

        // Store results and stream-compact survivors
        float vx_a[4], vy_a[4], vz_a[4];
        vst1q_f32(vx_a, vx); vst1q_f32(vy_a, vy); vst1q_f32(vz_a, vz);
        uint32_t bits[4];
        vst1q_u32(bits, mask);

        for (int k = 0; k < 4; k++) {
            if (bits[k] == 0) continue;
            if (out >= batch.capacity) break;
            batch.visible_indices[out] = i + k;
            batch.cam_x[out] = vx_a[k];
            batch.cam_y[out] = vy_a[k];
            batch.cam_z[out] = vz_a[k];
            out++;
        }
    }

    // Scalar tail
    for (; i < n; i++) {
        if (out >= batch.capacity) break;
        float px = scene.pos_x[i];
        float py = scene.pos_y[i];
        float pz = scene.pos_z[i];

        float vx = V[0]*px + V[4]*py + V[8]*pz  + V[12];
        float vy = V[1]*px + V[5]*py + V[9]*pz  + V[13];
        float vz = V[2]*px + V[6]*py + V[10]*pz + V[14];

        if (vz >= -cam.near_plane) continue;
        float z = -vz;
        if (z > cam.far_plane) continue;

        batch.visible_indices[out] = i;
        batch.cam_x[out] = vx;
        batch.cam_y[out] = vy;
        batch.cam_z[out] = vz;
        out++;
    }

    batch.cull_count = out;
    return out;
}

// --- Phase 2: Screen projection + covariance + radius ---

uint32_t gs_project_cov(const GaussianScene &scene, const CameraParams &cam,
                         ProjectionBatch &batch) {
    batch.cov_count = 0;
    if (batch.cull_count == 0) return 0;

    float tan_fovy = tanf(cam.fov_y * 0.5f);
    float tan_fovx = tan_fovy * cam.aspect;
    float focal_y = cam.height / (2.0f * tan_fovy);
    float focal_x = focal_y;
    float half_w = cam.width  * 0.5f;
    float half_h = cam.height * 0.5f;
    float screen_w = (float)cam.width;
    float screen_h = (float)cam.height;
    const float *V = cam.view_matrix;

    uint32_t out = 0;
    for (uint32_t j = 0; j < batch.cull_count; j++) {
        uint32_t idx = batch.visible_indices[j];
        float vx = batch.cam_x[j];
        float vy = batch.cam_y[j];
        float vz = batch.cam_z[j];
        float z = -vz;
        float inv_z = 1.0f / z;

        float sx = focal_x * vx * inv_z + half_w;
        float sy = focal_y * vy * inv_z + half_h;

        // Compute 3D covariance from quaternion + scale
        float cov3d[6];
        compute_cov3d(scene.rot_w[idx], scene.rot_x[idx],
                      scene.rot_y[idx], scene.rot_z[idx],
                      scene.scale_x[idx], scene.scale_y[idx],
                      scene.scale_z[idx], cov3d);

        // Project to 2D
        float ca, cb, cc;
        project_cov2d(cov3d, V, vx, vy, vz, focal_x, focal_y, tan_fovx, tan_fovy, ca, cb, cc);

        float det = ca * cc - cb * cb;
        if (det <= 0.0f) continue;

        float mid = 0.5f * (ca + cc);
        float disc = mid * mid - det;
        float sqrt_disc = sqrtf(fmaxf(0.0f, disc));
        float lambda_max = mid + sqrt_disc;
        float lambda_min = fmaxf(0.3f, mid - sqrt_disc);

        // Cap eigenvalue ratio to prevent spike artifacts
        if (lambda_max > MAX_EIGEN_RATIO * lambda_min) {
            float target_min = lambda_max / MAX_EIGEN_RATIO;
            float inflate = target_min - lambda_min;
            ca += inflate;
            cc += inflate;
            det = ca * cc - cb * cb;
            mid = 0.5f * (ca + cc);
            disc = mid * mid - det;
            lambda_max = mid + sqrtf(fmaxf(0.0f, disc));
        }

        float rad = 2.0f * sqrtf(lambda_max);

        if (rad > screen_w) continue;
        if (sx + rad < 0 || sx - rad >= screen_w) continue;
        if (sy + rad < 0 || sy - rad >= screen_h) continue;
        if (rad < 0.3f) continue;

        // Compact into output slot
        batch.visible_indices[out] = idx;  // overwrite in-place (out <= j always)
        batch.screen_x[out] = sx;
        batch.screen_y[out] = sy;
        batch.depth[out] = z;
        batch.cov2d_a[out] = ca;
        batch.cov2d_b[out] = cb;
        batch.cov2d_c[out] = cc;
        batch.radius[out] = rad;
        out++;
    }

    batch.cov_count = out;
    return out;
}

// --- Phase 3: SH evaluation ---

void gs_project_color(const GaussianScene &scene, const CameraParams &cam,
                       ProjectionBatch &batch) {
    if (batch.cov_count == 0) return;

    int sh_degree = scene.sh_degree;
    int rest_per_vertex = sh_rest_count(sh_degree);
    int basis_per_channel = (sh_degree > 0) ? ((sh_degree + 1) * (sh_degree + 1) - 1) : 0;
    const float *cam_pos = cam.position;

    for (uint32_t j = 0; j < batch.cov_count; j++) {
        uint32_t idx = batch.visible_indices[j];

        // Prefetch SH data for next iteration
        if (j + 1 < batch.cov_count) {
            uint32_t next_idx = batch.visible_indices[j + 1];
            __builtin_prefetch(&scene.sh_r[next_idx], 0, 1);
            if (scene.sh_rest && rest_per_vertex > 0)
                __builtin_prefetch(&scene.sh_rest[next_idx * rest_per_vertex], 0, 1);
        }

        // View direction (camera → Gaussian, normalized)
        float dx = scene.pos_x[idx] - cam_pos[0];
        float dy = scene.pos_y[idx] - cam_pos[1];
        float dz = scene.pos_z[idx] - cam_pos[2];
        float dlen = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dlen > 1e-8f) { float inv = 1.0f / dlen; dx *= inv; dy *= inv; dz *= inv; }

        float cr, cg, cb_val;
        const float *sh_rest_ptr = (rest_per_vertex > 0 && scene.sh_rest) ?
                                    &scene.sh_rest[idx * rest_per_vertex] : nullptr;
        eval_sh(sh_degree, scene.sh_r[idx], scene.sh_g[idx], scene.sh_b[idx],
                sh_rest_ptr, basis_per_channel, dx, dy, dz, cr, cg, cb_val);

        batch.color_r[j] = cr;
        batch.color_g[j] = cg;
        batch.color_b[j] = cb_val;
    }
}

// --- Phase 4: Pack SoA → AoS ---

void gs_project_assemble(const GaussianScene &scene, const ProjectionBatch &batch,
                          ProjectionResult &result) {
    result.count = 0;
    if (batch.cov_count == 0) return;

    uint32_t count = batch.cov_count;
    if (count > result.capacity) count = result.capacity;

    for (uint32_t j = 0; j < count; j++) {
        ProjectedGaussian &pg = result.gaussians[j];
        pg.screen_x = batch.screen_x[j];
        pg.screen_y = batch.screen_y[j];
        pg.depth    = batch.depth[j];
        pg.cov2d_a  = batch.cov2d_a[j];
        pg.cov2d_b  = batch.cov2d_b[j];
        pg.cov2d_c  = batch.cov2d_c[j];
        pg.color_r  = batch.color_r[j];
        pg.color_g  = batch.color_g[j];
        pg.color_b  = batch.color_b[j];
        pg.opacity  = scene.opacity[batch.visible_indices[j]];
        pg.radius   = batch.radius[j];
        pg.orig_idx = batch.visible_indices[j];
    }

    result.count = count;
}

// --- Orchestrator (4-arg) ---

void gs_project(const GaussianScene &scene, const CameraParams &cam,
                ProjectionBatch &batch, ProjectionResult &result) {
    gs_project_cull(scene, cam, batch);
    if (batch.cull_count == 0) { result.count = 0; return; }
    gs_project_cov(scene, cam, batch);
    if (batch.cov_count == 0) { result.count = 0; return; }
    gs_project_color(scene, cam, batch);
    gs_project_assemble(scene, batch, result);
}

// --- Backward-compatible 3-arg wrapper ---

void gs_project(const GaussianScene &scene, const CameraParams &cam, ProjectionResult &result) {
    ProjectionBatch batch;
    if (!gs_projection_batch_alloc(batch, scene.num_gaussians)) {
        result.count = 0;
        return;
    }
    gs_project(scene, cam, batch, result);
    gs_projection_batch_free(batch);
}
