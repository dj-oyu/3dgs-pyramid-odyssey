#include "gs_projector.h"
#include "gs_math.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <arm_neon.h>

// Max eigenvalue ratio for 2D covariance (prevents spike artifacts)
static constexpr float MAX_EIGEN_RATIO = 200.0f;

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

// Compute 3D covariance matrix from quaternion and scale
static void compute_cov3d(float qw, float qx, float qy, float qz,
                           float sx, float sy, float sz,
                           float cov[6]) {
    // Rotation matrix from quaternion
    float r00 = 1.0f - 2.0f*(qy*qy + qz*qz);
    float r01 = 2.0f*(qx*qy - qz*qw);
    float r02 = 2.0f*(qx*qz + qy*qw);
    float r10 = 2.0f*(qx*qy + qz*qw);
    float r11 = 1.0f - 2.0f*(qx*qx + qz*qz);
    float r12 = 2.0f*(qy*qz - qx*qw);
    float r20 = 2.0f*(qx*qz - qy*qw);
    float r21 = 2.0f*(qy*qz + qx*qw);
    float r22 = 1.0f - 2.0f*(qx*qx + qy*qy);

    // S = diag(sx, sy, sz)
    // M = R * S
    float m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
    float m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
    float m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;

    // Cov = M * M^T (symmetric, store upper triangle)
    cov[0] = m00*m00 + m01*m01 + m02*m02;  // [0,0]
    cov[1] = m00*m10 + m01*m11 + m02*m12;  // [0,1]
    cov[2] = m00*m20 + m01*m21 + m02*m22;  // [0,2]
    cov[3] = m10*m10 + m11*m11 + m12*m12;  // [1,1]
    cov[4] = m10*m20 + m11*m21 + m12*m22;  // [1,2]
    cov[5] = m20*m20 + m21*m21 + m22*m22;  // [2,2]
}

// Project 3D covariance to 2D given Jacobian of perspective projection
static void project_cov2d(const float cov3d[6],
                           const float *view_matrix,
                           float tx, float ty, float tz,
                           float focal_x, float focal_y,
                           float tan_fovx, float tan_fovy,
                           float &cov2d_a, float &cov2d_b, float &cov2d_c) {
    // Jacobian of projection: J = d(screen) / d(camera)
    // Our projection: sx = fx*vx/z + hw, sy = fy*vy/z + hh  (z = -vz > 0)
    // Equivalent to: sx = -fx*vx/vz, sy = -fy*vy/vz
    // dsx/dvx = -fx/vz, dsx/dvz = fx*vx/vz²
    // dsy/dvy = -fy/vz, dsy/dvz = fy*vy/vz²
    float inv_z = 1.0f / tz;
    float inv_z2 = inv_z * inv_z;

    // Clamp tangent angles to prevent extreme Jacobian values (reference 3DGS)
    // tx=vx, ty=vy, tz=vz (vz < 0). z = -vz > 0.
    // Tangent of angle from optical axis: vx/z, vy/z
    float z = -tz;  // positive depth
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    tx = fminf(limx, fmaxf(-limx, tx / z)) * z;  // clamped vx
    ty = fminf(limy, fmaxf(-limy, ty / z)) * z;  // clamped vy

    float j00 = -focal_x * inv_z;
    float j02 = focal_x * tx * inv_z2;
    float j11 = -focal_y * inv_z;
    float j12 = focal_y * ty * inv_z2;

    // W = upper-left 3x3 of view matrix (column-major)
    float w00 = view_matrix[0], w01 = view_matrix[4], w02 = view_matrix[8];
    float w10 = view_matrix[1], w11 = view_matrix[5], w12 = view_matrix[9];
    float w20 = view_matrix[2], w21 = view_matrix[6], w22 = view_matrix[10];

    // T = J * W (2x3 matrix)
    float t00 = j00*w00 + j02*w20;
    float t01 = j00*w01 + j02*w21;
    float t02 = j00*w02 + j02*w22;
    float t10 = j11*w10 + j12*w20;
    float t11 = j11*w11 + j12*w21;
    float t12 = j11*w12 + j12*w22;

    // 2D cov = T * Cov3D * T^T, where T is 2x3 and Cov3D is 3x3
    // First: M = T * Cov3D (2x3)
    float m00_v = t00*cov3d[0] + t01*cov3d[1] + t02*cov3d[2];
    float m01_v = t00*cov3d[1] + t01*cov3d[3] + t02*cov3d[4];
    float m02_v = t00*cov3d[2] + t01*cov3d[4] + t02*cov3d[5];
    float m10_v = t10*cov3d[0] + t11*cov3d[1] + t12*cov3d[2];
    float m11_v = t10*cov3d[1] + t11*cov3d[3] + t12*cov3d[4];
    float m12_v = t10*cov3d[2] + t11*cov3d[4] + t12*cov3d[5];

    // Result = M * T^T (2x2)
    cov2d_a = m00_v*t00 + m01_v*t01 + m02_v*t02;
    cov2d_b = m00_v*t10 + m01_v*t11 + m02_v*t12;
    cov2d_c = m10_v*t10 + m11_v*t11 + m12_v*t12;

    // Add small regularization for numerical stability
    cov2d_a += 0.3f;
    cov2d_c += 0.3f;
}

// Evaluate SH color (degrees 0-3)
// dir = normalized direction from camera to Gaussian (pos - campos)
// sh_rest layout per Gaussian: [R_basis0..R_basisK, G_basis0..G_basisK, B_basis0..B_basisK]
// where K = (degree+1)^2 - 2 (index of last basis)
static void eval_sh(int degree,
                     float dc_r, float dc_g, float dc_b,
                     const float *sh_rest, int basis_per_channel,
                     float dir_x, float dir_y, float dir_z,
                     float &r, float &g, float &b) {
    // SH constants matching the reference 3DGS implementation
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

    // Degree 0 (DC)
    r = SH_C0 * dc_r;
    g = SH_C0 * dc_g;
    b = SH_C0 * dc_b;

    if (degree >= 1 && sh_rest) {
        // Degree 1: 3 basis functions per channel
        // Offsets into sh_rest for each color channel
        const float *sr = sh_rest;                       // R channel
        const float *sg = sh_rest + basis_per_channel;   // G channel
        const float *sb = sh_rest + 2 * basis_per_channel; // B channel

        r += SH_C1 * (-y * sr[0] + z * sr[1] - x * sr[2]);
        g += SH_C1 * (-y * sg[0] + z * sg[1] - x * sg[2]);
        b += SH_C1 * (-y * sb[0] + z * sb[1] - x * sb[2]);

        if (degree >= 2) {
            // Degree 2: 5 basis functions per channel (indices 3-7)
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
                // Degree 3: 7 basis functions per channel (indices 8-14)
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

    // Bias + clamp
    r = fmaxf(0.0f, fminf(1.0f, r + 0.5f));
    g = fmaxf(0.0f, fminf(1.0f, g + 0.5f));
    b = fmaxf(0.0f, fminf(1.0f, b + 0.5f));
}

// Process a single Gaussian (used by both scalar and batch paths)
static inline bool project_one(
    uint32_t idx, const GaussianScene &scene,
    const float *V, const float *cam_pos,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    float half_w, float half_h, float near_plane, float far_plane,
    float screen_w, float screen_h,
    int sh_degree, int rest_per_vertex, int basis_per_channel,
    ProjectedGaussian &pg)
{
    float px = scene.pos_x[idx];
    float py = scene.pos_y[idx];
    float pz = scene.pos_z[idx];

    float vx = V[0]*px + V[4]*py + V[8]*pz  + V[12];
    float vy = V[1]*px + V[5]*py + V[9]*pz  + V[13];
    float vz = V[2]*px + V[6]*py + V[10]*pz + V[14];

    if (vz >= -near_plane) return false;
    float z = -vz;
    if (z > far_plane) return false;

    float inv_z = 1.0f / z;
    float sx = focal_x * vx * inv_z + half_w;
    float sy = focal_y * vy * inv_z + half_h;

    float cov3d[6];
    compute_cov3d(scene.rot_w[idx], scene.rot_x[idx], scene.rot_y[idx], scene.rot_z[idx],
                  scene.scale_x[idx], scene.scale_y[idx], scene.scale_z[idx], cov3d);

    float cov2d_a, cov2d_b, cov2d_c;
    project_cov2d(cov3d, V, vx, vy, vz, focal_x, focal_y, tan_fovx, tan_fovy, cov2d_a, cov2d_b, cov2d_c);

    float det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
    if (det <= 0.0f) return false;

    float mid = 0.5f * (cov2d_a + cov2d_c);
    float disc = mid * mid - det;
    float sqrt_disc = sqrtf(fmaxf(0.0f, disc));
    float lambda_max = mid + sqrt_disc;
    float lambda_min = fmaxf(0.3f, mid - sqrt_disc);

    // Cap eigenvalue ratio to prevent extreme spike artifacts
    // When a Gaussian is very close to the camera, perspective projection can
    // create extreme aspect ratios (>1000:1), causing needle-like artifacts.
    constexpr float MAX_EIGEN_RATIO = 200.0f;
    if (lambda_max > MAX_EIGEN_RATIO * lambda_min) {
        // Inflate the covariance to reduce aspect ratio
        // Add isotropic component: inflate both eigenvalues equally
        float target_min = lambda_max / MAX_EIGEN_RATIO;
        float inflate = target_min - lambda_min;
        cov2d_a += inflate;
        cov2d_c += inflate;
        // Recompute for radius
        det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
        mid = 0.5f * (cov2d_a + cov2d_c);
        disc = mid * mid - det;
        lambda_max = mid + sqrtf(fmaxf(0.0f, disc));
    }

    float radius = 3.0f * sqrtf(lambda_max);

    if (radius > screen_w) return false;  // cap enormous Gaussians
    if (sx + radius < 0 || sx - radius >= screen_w) return false;
    if (sy + radius < 0 || sy - radius >= screen_h) return false;
    if (radius < 0.3f) return false;

    // Compute view direction (camera → Gaussian, normalized) for SH
    // Reference 3DGS uses dir = pos - campos
    float dx = px - cam_pos[0];
    float dy = py - cam_pos[1];
    float dz = pz - cam_pos[2];
    float dlen = sqrtf(dx*dx + dy*dy + dz*dz);
    if (dlen > 1e-8f) { float inv = 1.0f / dlen; dx *= inv; dy *= inv; dz *= inv; }

    float cr, cg, cb;
    const float *sh_rest_ptr = (rest_per_vertex > 0 && scene.sh_rest) ?
                                &scene.sh_rest[idx * rest_per_vertex] : nullptr;
    eval_sh(sh_degree, scene.sh_r[idx], scene.sh_g[idx], scene.sh_b[idx],
            sh_rest_ptr, basis_per_channel, dx, dy, dz, cr, cg, cb);

    pg.screen_x = sx;
    pg.screen_y = sy;
    pg.depth = z;
    pg.cov2d_a = cov2d_a;
    pg.cov2d_b = cov2d_b;
    pg.cov2d_c = cov2d_c;
    pg.color_r = cr;
    pg.color_g = cg;
    pg.color_b = cb;
    pg.opacity = scene.opacity[idx];
    pg.radius = radius;
    pg.orig_idx = idx;
    return true;
}

void gs_project(const GaussianScene &scene, const CameraParams &cam, ProjectionResult &result) {
    result.count = 0;
    if (scene.num_gaussians == 0) return;

    float tan_fovy = tanf(cam.fov_y * 0.5f);
    float tan_fovx = tan_fovy * cam.aspect;
    float focal_y = cam.height / (2.0f * tan_fovy);
    float focal_x = focal_y;

    float half_w = cam.width  * 0.5f;
    float half_h = cam.height * 0.5f;
    float screen_w = (float)cam.width;
    float screen_h = (float)cam.height;

    const float *V = cam.view_matrix;
    const float *cam_pos = cam.position;
    int sh_degree = scene.sh_degree;
    int rest_per_vertex = sh_rest_count(sh_degree);
    int basis_per_channel = (sh_degree > 0) ? ((sh_degree + 1) * (sh_degree + 1) - 1) : 0;
    uint32_t n = scene.num_gaussians;

    // All 3 view matrix rows as NEON broadcast vectors
    float32x4_t v_r0x = vdupq_n_f32(V[0]),  v_r0y = vdupq_n_f32(V[4]);
    float32x4_t v_r0z = vdupq_n_f32(V[8]),  v_r0w = vdupq_n_f32(V[12]);
    float32x4_t v_r1x = vdupq_n_f32(V[1]),  v_r1y = vdupq_n_f32(V[5]);
    float32x4_t v_r1z = vdupq_n_f32(V[9]),  v_r1w = vdupq_n_f32(V[13]);
    float32x4_t v_r2x = vdupq_n_f32(V[2]),  v_r2y = vdupq_n_f32(V[6]);
    float32x4_t v_r2z = vdupq_n_f32(V[10]), v_r2w = vdupq_n_f32(V[14]);

    float32x4_t v_neg_near = vdupq_n_f32(-cam.near_plane);
    float32x4_t v_far = vdupq_n_f32(cam.far_plane);
    float32x4_t v_one = vdupq_n_f32(1.0f);

    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Prefetch next batch (positions + covariance data)
        if (i + 8 <= n) {
            __builtin_prefetch(&scene.pos_x[i + 8], 0, 1);
            __builtin_prefetch(&scene.pos_y[i + 8], 0, 1);
            __builtin_prefetch(&scene.pos_z[i + 8], 0, 1);
            __builtin_prefetch(&scene.rot_w[i + 4], 0, 1);
            __builtin_prefetch(&scene.rot_x[i + 4], 0, 1);
            __builtin_prefetch(&scene.rot_y[i + 4], 0, 1);
            __builtin_prefetch(&scene.rot_z[i + 4], 0, 1);
            __builtin_prefetch(&scene.scale_x[i + 4], 0, 1);
            __builtin_prefetch(&scene.scale_y[i + 4], 0, 1);
            __builtin_prefetch(&scene.scale_z[i + 4], 0, 1);
        }

        // Load 4 positions (SoA - contiguous loads)
        float32x4_t px = vld1q_f32(&scene.pos_x[i]);
        float32x4_t py = vld1q_f32(&scene.pos_y[i]);
        float32x4_t pz = vld1q_f32(&scene.pos_z[i]);

        // Full view transform for all 3 components (vx, vy, vz) × 4 Gaussians
        float32x4_t vx = vmlaq_f32(v_r0w, v_r0x, px);
        vx = vmlaq_f32(vx, v_r0y, py);
        vx = vmlaq_f32(vx, v_r0z, pz);

        float32x4_t vy = vmlaq_f32(v_r1w, v_r1x, px);
        vy = vmlaq_f32(vy, v_r1y, py);
        vy = vmlaq_f32(vy, v_r1z, pz);

        float32x4_t vz = vmlaq_f32(v_r2w, v_r2x, px);
        vz = vmlaq_f32(vz, v_r2y, py);
        vz = vmlaq_f32(vz, v_r2z, pz);

        // Frustum cull: vz < -near_plane AND -vz <= far_plane
        float32x4_t neg_vz = vnegq_f32(vz);
        uint32x4_t mask = vandq_u32(vcltq_f32(vz, v_neg_near),
                                     vcleq_f32(neg_vz, v_far));
        if (vmaxvq_u32(mask) == 0) continue;

        // Screen projection for 4 Gaussians: sx = fx*vx/z + hw, sy = fy*(-vy)/z + hh
        float32x4_t z = neg_vz;
        float32x4_t inv_z = vdivq_f32(v_one, z);

        float32x4_t sx = vmlaq_f32(vdupq_n_f32(half_w),
                                     vdupq_n_f32(focal_x), vmulq_f32(vx, inv_z));
        float32x4_t sy = vmlaq_f32(vdupq_n_f32(half_h),
                                     vdupq_n_f32(focal_y), vmulq_f32(vy, inv_z));

        // Store batch results to process survivors
        float vx_a[4], vy_a[4], vz_a[4], z_a[4], sx_a[4], sy_a[4];
        vst1q_f32(vx_a, vx);  vst1q_f32(vy_a, vy);  vst1q_f32(vz_a, vz);
        vst1q_f32(z_a, z);    vst1q_f32(sx_a, sx);   vst1q_f32(sy_a, sy);

        uint32_t bits[4];
        vst1q_u32(bits, mask);

        for (int k = 0; k < 4; k++) {
            if (bits[k] == 0) continue;
            if (result.count >= result.capacity) break;

            uint32_t idx = i + k;

            // Covariance (scalar - complex dependent operations)
            float cov3d[6];
            compute_cov3d(scene.rot_w[idx], scene.rot_x[idx],
                          scene.rot_y[idx], scene.rot_z[idx],
                          scene.scale_x[idx], scene.scale_y[idx],
                          scene.scale_z[idx], cov3d);

            float cov2d_a, cov2d_b, cov2d_c;
            project_cov2d(cov3d, V, vx_a[k], vy_a[k], vz_a[k],
                          focal_x, focal_y, tan_fovx, tan_fovy,
                          cov2d_a, cov2d_b, cov2d_c);

            float det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
            if (det <= 0.0f) continue;

            float mid = 0.5f * (cov2d_a + cov2d_c);
            float disc = mid * mid - det;
            float sqrt_disc = sqrtf(fmaxf(0.0f, disc));
            float lambda_max = mid + sqrt_disc;
            float lambda_min = fmaxf(0.3f, mid - sqrt_disc);

            // Cap eigenvalue ratio to prevent spike artifacts
            if (lambda_max > MAX_EIGEN_RATIO * lambda_min) {
                float target_min = lambda_max / MAX_EIGEN_RATIO;
                float inflate = target_min - lambda_min;
                cov2d_a += inflate;
                cov2d_c += inflate;
                det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
                mid = 0.5f * (cov2d_a + cov2d_c);
                disc = mid * mid - det;
                lambda_max = mid + sqrtf(fmaxf(0.0f, disc));
            }

            float radius = 3.0f * sqrtf(lambda_max);

            if (radius > screen_w) continue;  // cap enormous Gaussians
            float scx = sx_a[k], scy = sy_a[k];
            if (scx + radius < 0 || scx - radius >= screen_w) continue;
            if (scy + radius < 0 || scy - radius >= screen_h) continue;
            if (radius < 0.3f) continue;

            // Compute view direction (camera → Gaussian) for SH evaluation
            // Reference 3DGS uses dir = pos - campos
            float gpx = scene.pos_x[idx], gpy = scene.pos_y[idx], gpz = scene.pos_z[idx];
            float ddx = gpx - cam_pos[0], ddy = gpy - cam_pos[1], ddz = gpz - cam_pos[2];
            float dlen = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
            if (dlen > 1e-8f) { float inv = 1.0f / dlen; ddx *= inv; ddy *= inv; ddz *= inv; }

            float cr, cg, cb;
            const float *sh_rest_ptr = (rest_per_vertex > 0 && scene.sh_rest) ?
                                        &scene.sh_rest[idx * rest_per_vertex] : nullptr;
            eval_sh(sh_degree, scene.sh_r[idx], scene.sh_g[idx], scene.sh_b[idx],
                    sh_rest_ptr, basis_per_channel, ddx, ddy, ddz, cr, cg, cb);

            ProjectedGaussian &pg = result.gaussians[result.count];
            pg.screen_x = scx;
            pg.screen_y = scy;
            pg.depth    = z_a[k];
            pg.cov2d_a  = cov2d_a;
            pg.cov2d_b  = cov2d_b;
            pg.cov2d_c  = cov2d_c;
            pg.color_r  = cr;
            pg.color_g  = cg;
            pg.color_b  = cb;
            pg.opacity  = scene.opacity[idx];
            pg.radius   = radius;
            pg.orig_idx = idx;
            result.count++;
        }
    }

    // Handle remaining Gaussians
    for (; i < n; i++) {
        if (result.count >= result.capacity) break;
        ProjectedGaussian &pg = result.gaussians[result.count];
        if (project_one(i, scene, V, cam_pos, focal_x, focal_y,
                       tan_fovx, tan_fovy,
                       half_w, half_h, cam.near_plane, cam.far_plane,
                       screen_w, screen_h,
                       sh_degree, rest_per_vertex, basis_per_channel, pg)) {
            result.count++;
        }
    }
}
