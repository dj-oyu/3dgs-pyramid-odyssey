#include "gs_projector.h"
#include "gs_math.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <arm_neon.h>

// Max eigenvalue ratio for 2D covariance (prevents spike artifacts)
static constexpr float MAX_EIGEN_RATIO = 200.0f;

// --- Fast approximate math (NEON) ---
// vrsqrte + 1 Newton-Raphson: ~6 cycles vs FSQRT ~14 cycles on A55

static inline float fast_rsqrtf(float x) {
    float32x2_t v = vdup_n_f32(x);
    float32x2_t est = vrsqrte_f32(v);
    est = vmul_f32(est, vrsqrts_f32(vmul_f32(v, est), est));
    return vget_lane_f32(est, 0);
}

static inline float fast_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    return x * fast_rsqrtf(x);
}

static inline float fast_recipf(float x) {
    float32x2_t v = vdup_n_f32(x);
    float32x2_t est = vrecpe_f32(v);
    est = vmul_f32(est, vrecps_f32(v, est));
    return vget_lane_f32(est, 0);
}

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
                           float inv_z,
                           float focal_x, float focal_y,
                           float tan_fovx, float tan_fovy,
                           float &cov2d_a, float &cov2d_b, float &cov2d_c) {
    float inv_z2 = inv_z * inv_z;

    float z = -tz;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    tx = fminf(limx, fmaxf(-limx, tx * inv_z)) * z;
    ty = fminf(limy, fmaxf(-limy, ty * inv_z)) * z;

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

// --- NEON 4-wide helpers ---

static inline float32x4_t gather4(const float *arr, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    float tmp[4] = {arr[i0], arr[i1], arr[i2], arr[i3]};
    return vld1q_f32(tmp);
}

static inline float32x4_t safe_sqrtq(float32x4_t x) {
    float32x4_t tiny = vdupq_n_f32(1e-30f);
    float32x4_t xc = vmaxq_f32(x, tiny);
    float32x4_t est = vrsqrteq_f32(xc);
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(xc, est), est));
    float32x4_t result = vmulq_f32(x, est);
    // Zero out where x <= 0
    uint32x4_t pos = vcgtq_f32(x, vdupq_n_f32(0.0f));
    return vbslq_f32(pos, result, vdupq_n_f32(0.0f));
}

static inline void compute_cov3d_x4(
    float32x4_t qw, float32x4_t qx, float32x4_t qy, float32x4_t qz,
    float32x4_t sx, float32x4_t sy, float32x4_t sz,
    float32x4_t cov[6])
{
    float32x4_t two = vdupq_n_f32(2.0f);
    float32x4_t one = vdupq_n_f32(1.0f);

    float32x4_t qyqy = vmulq_f32(qy, qy), qzqz = vmulq_f32(qz, qz);
    float32x4_t qxqx = vmulq_f32(qx, qx);
    float32x4_t qxqy = vmulq_f32(qx, qy), qxqz = vmulq_f32(qx, qz);
    float32x4_t qyqz = vmulq_f32(qy, qz);
    float32x4_t qzqw = vmulq_f32(qz, qw), qyqw = vmulq_f32(qy, qw);
    float32x4_t qxqw = vmulq_f32(qx, qw);

    float32x4_t r00 = vsubq_f32(one, vmulq_f32(two, vaddq_f32(qyqy, qzqz)));
    float32x4_t r01 = vmulq_f32(two, vsubq_f32(qxqy, qzqw));
    float32x4_t r02 = vmulq_f32(two, vaddq_f32(qxqz, qyqw));
    float32x4_t r10 = vmulq_f32(two, vaddq_f32(qxqy, qzqw));
    float32x4_t r11 = vsubq_f32(one, vmulq_f32(two, vaddq_f32(qxqx, qzqz)));
    float32x4_t r12 = vmulq_f32(two, vsubq_f32(qyqz, qxqw));
    float32x4_t r20 = vmulq_f32(two, vsubq_f32(qxqz, qyqw));
    float32x4_t r21 = vmulq_f32(two, vaddq_f32(qyqz, qxqw));
    float32x4_t r22 = vsubq_f32(one, vmulq_f32(two, vaddq_f32(qxqx, qyqy)));

    // M = R * diag(S)
    float32x4_t m00 = vmulq_f32(r00, sx), m01 = vmulq_f32(r01, sy), m02 = vmulq_f32(r02, sz);
    float32x4_t m10 = vmulq_f32(r10, sx), m11 = vmulq_f32(r11, sy), m12 = vmulq_f32(r12, sz);
    float32x4_t m20 = vmulq_f32(r20, sx), m21 = vmulq_f32(r21, sy), m22 = vmulq_f32(r22, sz);

    // Cov3D = M * M^T (symmetric)
    cov[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(m00,m00), m01,m01), m02,m02);
    cov[1] = vmlaq_f32(vmlaq_f32(vmulq_f32(m00,m10), m01,m11), m02,m12);
    cov[2] = vmlaq_f32(vmlaq_f32(vmulq_f32(m00,m20), m01,m21), m02,m22);
    cov[3] = vmlaq_f32(vmlaq_f32(vmulq_f32(m10,m10), m11,m11), m12,m12);
    cov[4] = vmlaq_f32(vmlaq_f32(vmulq_f32(m10,m20), m11,m21), m12,m22);
    cov[5] = vmlaq_f32(vmlaq_f32(vmulq_f32(m20,m20), m21,m21), m22,m22);
}

static inline void project_cov2d_x4(
    const float32x4_t cov3d[6], const float *V,
    float32x4_t tx, float32x4_t ty, float32x4_t tz, float32x4_t inv_z,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy,
    float32x4_t &ca, float32x4_t &cb, float32x4_t &cc)
{
    float32x4_t inv_z2 = vmulq_f32(inv_z, inv_z);
    float32x4_t z = vnegq_f32(tz);

    float32x4_t limx = vdupq_n_f32(1.3f * tan_fovx);
    float32x4_t limy = vdupq_n_f32(1.3f * tan_fovy);
    tx = vmulq_f32(vminq_f32(limx, vmaxq_f32(vnegq_f32(limx), vmulq_f32(tx, inv_z))), z);
    ty = vmulq_f32(vminq_f32(limy, vmaxq_f32(vnegq_f32(limy), vmulq_f32(ty, inv_z))), z);

    float32x4_t v_nfx = vdupq_n_f32(-focal_x);
    float32x4_t v_nfy = vdupq_n_f32(-focal_y);
    float32x4_t v_pfx = vdupq_n_f32(focal_x);
    float32x4_t v_pfy = vdupq_n_f32(focal_y);

    float32x4_t j00 = vmulq_f32(v_nfx, inv_z);
    float32x4_t j02 = vmulq_f32(vmulq_f32(v_pfx, tx), inv_z2);
    float32x4_t j11 = vmulq_f32(v_nfy, inv_z);
    float32x4_t j12 = vmulq_f32(vmulq_f32(v_pfy, ty), inv_z2);

    // T = J * W (view matrix broadcast)
    float32x4_t w00 = vdupq_n_f32(V[0]), w01 = vdupq_n_f32(V[4]), w02 = vdupq_n_f32(V[8]);
    float32x4_t w10 = vdupq_n_f32(V[1]), w11 = vdupq_n_f32(V[5]), w12 = vdupq_n_f32(V[9]);
    float32x4_t w20 = vdupq_n_f32(V[2]), w21 = vdupq_n_f32(V[6]), w22 = vdupq_n_f32(V[10]);

    float32x4_t t00 = vmlaq_f32(vmulq_f32(j00, w00), j02, w20);
    float32x4_t t01 = vmlaq_f32(vmulq_f32(j00, w01), j02, w21);
    float32x4_t t02 = vmlaq_f32(vmulq_f32(j00, w02), j02, w22);
    float32x4_t t10 = vmlaq_f32(vmulq_f32(j11, w10), j12, w20);
    float32x4_t t11 = vmlaq_f32(vmulq_f32(j11, w11), j12, w21);
    float32x4_t t12 = vmlaq_f32(vmulq_f32(j11, w12), j12, w22);

    // T * Cov3D
    float32x4_t m00v = vmlaq_f32(vmlaq_f32(vmulq_f32(t00,cov3d[0]), t01,cov3d[1]), t02,cov3d[2]);
    float32x4_t m01v = vmlaq_f32(vmlaq_f32(vmulq_f32(t00,cov3d[1]), t01,cov3d[3]), t02,cov3d[4]);
    float32x4_t m02v = vmlaq_f32(vmlaq_f32(vmulq_f32(t00,cov3d[2]), t01,cov3d[4]), t02,cov3d[5]);
    float32x4_t m10v = vmlaq_f32(vmlaq_f32(vmulq_f32(t10,cov3d[0]), t11,cov3d[1]), t12,cov3d[2]);
    float32x4_t m11v = vmlaq_f32(vmlaq_f32(vmulq_f32(t10,cov3d[1]), t11,cov3d[3]), t12,cov3d[4]);
    float32x4_t m12v = vmlaq_f32(vmlaq_f32(vmulq_f32(t10,cov3d[2]), t11,cov3d[4]), t12,cov3d[5]);

    // Cov2D = (T * Cov3D) * T^T
    ca = vmlaq_f32(vmlaq_f32(vmulq_f32(m00v,t00), m01v,t01), m02v,t02);
    cb = vmlaq_f32(vmlaq_f32(vmulq_f32(m00v,t10), m01v,t11), m02v,t12);
    cc = vmlaq_f32(vmlaq_f32(vmulq_f32(m10v,t10), m11v,t11), m12v,t12);

    float32x4_t reg = vdupq_n_f32(0.3f);
    ca = vaddq_f32(ca, reg);
    cc = vaddq_f32(cc, reg);
}

// --- Multi-threaded projection helpers ---

static constexpr uint32_t NUM_PROJ_THREADS = NUM_RENDER_THREADS;
static constexpr uint32_t MT_THRESHOLD = 512;  // Don't thread if fewer than this

struct CovThreadArg {
    const GaussianScene *scene;
    const float *view_matrix;
    ProjectionBatch *batch;
    uint32_t j_start, j_end;
    uint32_t out_count;
    float tan_fovx, tan_fovy, focal_x, focal_y;
    float half_w, half_h, screen_w, screen_h;
};

static void *cov_thread_func(void *arg) {
    CovThreadArg &a = *(CovThreadArg*)arg;
    ProjectionBatch &batch = *a.batch;
    const GaussianScene &scene = *a.scene;
    const float *V = a.view_matrix;

    float32x4_t v_focal_x = vdupq_n_f32(a.focal_x);
    float32x4_t v_focal_y = vdupq_n_f32(a.focal_y);
    float32x4_t v_half_w = vdupq_n_f32(a.half_w);
    float32x4_t v_half_h = vdupq_n_f32(a.half_h);
    float32x4_t v_alpha = vdupq_n_f32(ALPHA_THRESHOLD);

    uint32_t out = a.j_start;
    uint32_t j = a.j_start;

    // --- NEON 4-wide path ---
    for (; j + 4 <= a.j_end; j += 4) {
        uint32_t i0 = batch.visible_indices[j],   i1 = batch.visible_indices[j+1];
        uint32_t i2 = batch.visible_indices[j+2], i3 = batch.visible_indices[j+3];

        // Contiguous load from batch
        float32x4_t vx4 = vld1q_f32(&batch.cam_x[j]);
        float32x4_t vy4 = vld1q_f32(&batch.cam_y[j]);
        float32x4_t vz4 = vld1q_f32(&batch.cam_z[j]);

        // Screen projection (4-wide)
        float32x4_t z4 = vnegq_f32(vz4);
        float32x4_t inv_z4 = vrecpeq_f32(z4);
        inv_z4 = vmulq_f32(inv_z4, vrecpsq_f32(z4, inv_z4));
        float32x4_t sx4 = vaddq_f32(vmulq_f32(vmulq_f32(v_focal_x, vx4), inv_z4), v_half_w);
        float32x4_t sy4 = vaddq_f32(vmulq_f32(vmulq_f32(v_focal_y, vy4), inv_z4), v_half_h);

        // Opacity early cull (4-wide)
        float32x4_t op4 = gather4(scene.opacity, i0, i1, i2, i3);
        uint32x4_t alive = vcgeq_f32(op4, v_alpha);
        if (vmaxvq_u32(alive) == 0) continue;

        // Gather rot/scale (7 arrays)
        float32x4_t qw4 = gather4(scene.rot_w, i0, i1, i2, i3);
        float32x4_t qx4 = gather4(scene.rot_x, i0, i1, i2, i3);
        float32x4_t qy4 = gather4(scene.rot_y, i0, i1, i2, i3);
        float32x4_t qz4 = gather4(scene.rot_z, i0, i1, i2, i3);
        float32x4_t scx4 = gather4(scene.scale_x, i0, i1, i2, i3);
        float32x4_t scy4 = gather4(scene.scale_y, i0, i1, i2, i3);
        float32x4_t scz4 = gather4(scene.scale_z, i0, i1, i2, i3);

        // compute_cov3d (NEON 4-wide)
        float32x4_t cov3d[6];
        compute_cov3d_x4(qw4, qx4, qy4, qz4, scx4, scy4, scz4, cov3d);

        // project_cov2d (NEON 4-wide)
        float32x4_t ca4, cb4, cc4;
        project_cov2d_x4(cov3d, V, vx4, vy4, vz4, inv_z4,
                         a.focal_x, a.focal_y, a.tan_fovx, a.tan_fovy,
                         ca4, cb4, cc4);

        // Extract to scalar for eigenvalue + culling + compact
        float sx_a[4], sy_a[4], z_a[4];
        float ca_a[4], cb_a[4], cc_a[4];
        vst1q_f32(sx_a, sx4); vst1q_f32(sy_a, sy4); vst1q_f32(z_a, z4);
        vst1q_f32(ca_a, ca4); vst1q_f32(cb_a, cb4); vst1q_f32(cc_a, cc4);
        uint32_t alive_bits[4]; vst1q_u32(alive_bits, alive);
        uint32_t idx4[4] = {i0, i1, i2, i3};

        for (int k = 0; k < 4; k++) {
            if (!alive_bits[k]) continue;

            float ca = ca_a[k], cb = cb_a[k], cc = cc_a[k];
            float det = ca * cc - cb * cb;
            if (det <= 0.0f) continue;

            float mid = 0.5f * (ca + cc);
            float disc = mid * mid - det;
            float sqrt_disc = fast_sqrtf(fmaxf(0.0f, disc));
            float lambda_max = mid + sqrt_disc;
            float lambda_min = fmaxf(0.3f, mid - sqrt_disc);

            if (lambda_max > MAX_EIGEN_RATIO * lambda_min) {
                float target_min = lambda_max * fast_recipf(MAX_EIGEN_RATIO);
                float inflate = target_min - lambda_min;
                ca += inflate; cc += inflate;
                det = ca * cc - cb * cb;
                mid = 0.5f * (ca + cc);
                disc = mid * mid - det;
                lambda_max = mid + fast_sqrtf(fmaxf(0.0f, disc));
            }

            float rad = 2.0f * fast_sqrtf(lambda_max);
            if (rad > a.screen_w) continue;
            float sx = sx_a[k], sy = sy_a[k];
            if (sx + rad < 0 || sx - rad >= a.screen_w) continue;
            if (sy + rad < 0 || sy - rad >= a.screen_h) continue;
            if (rad < 0.3f) continue;

            batch.visible_indices[out] = idx4[k];
            batch.screen_x[out] = sx;
            batch.screen_y[out] = sy;
            batch.depth[out] = z_a[k];
            batch.cov2d_a[out] = ca;
            batch.cov2d_b[out] = cb;
            batch.cov2d_c[out] = cc;
            batch.radius[out] = rad;
            out++;
        }
    }

    // --- Scalar tail ---
    for (; j < a.j_end; j++) {
        uint32_t idx = batch.visible_indices[j];
        float vx = batch.cam_x[j], vy = batch.cam_y[j], vz = batch.cam_z[j];
        float z = -vz;
        float inv_z = fast_recipf(z);
        float sx = a.focal_x * vx * inv_z + a.half_w;
        float sy = a.focal_y * vy * inv_z + a.half_h;
        if (scene.opacity[idx] < ALPHA_THRESHOLD) continue;

        float cov3d[6];
        compute_cov3d(scene.rot_w[idx], scene.rot_x[idx], scene.rot_y[idx], scene.rot_z[idx],
                      scene.scale_x[idx], scene.scale_y[idx], scene.scale_z[idx], cov3d);
        float ca, cb, cc;
        project_cov2d(cov3d, V, vx, vy, vz, inv_z,
                      a.focal_x, a.focal_y, a.tan_fovx, a.tan_fovy, ca, cb, cc);

        float det = ca * cc - cb * cb;
        if (det <= 0.0f) continue;
        float mid = 0.5f * (ca + cc);
        float disc = mid * mid - det;
        float sqrt_disc = fast_sqrtf(fmaxf(0.0f, disc));
        float lambda_max = mid + sqrt_disc;
        float lambda_min = fmaxf(0.3f, mid - sqrt_disc);
        if (lambda_max > MAX_EIGEN_RATIO * lambda_min) {
            float target_min = lambda_max * fast_recipf(MAX_EIGEN_RATIO);
            float inflate = target_min - lambda_min;
            ca += inflate; cc += inflate;
            det = ca * cc - cb * cb; mid = 0.5f * (ca + cc);
            disc = mid * mid - det;
            lambda_max = mid + fast_sqrtf(fmaxf(0.0f, disc));
        }
        float rad = 2.0f * fast_sqrtf(lambda_max);
        if (rad > a.screen_w) continue;
        if (sx + rad < 0 || sx - rad >= a.screen_w) continue;
        if (sy + rad < 0 || sy - rad >= a.screen_h) continue;
        if (rad < 0.3f) continue;

        batch.visible_indices[out] = idx;
        batch.screen_x[out] = sx; batch.screen_y[out] = sy;
        batch.depth[out] = z;
        batch.cov2d_a[out] = ca; batch.cov2d_b[out] = cb; batch.cov2d_c[out] = cc;
        batch.radius[out] = rad;
        out++;
    }

    a.out_count = out - a.j_start;
    return nullptr;
}

struct ColorThreadArg {
    const GaussianScene *scene;
    const float *cam_pos;
    ProjectionBatch *batch;
    uint32_t j_start, j_end;
    int sh_degree;
    int rest_per_vertex;
    int basis_per_channel;
};

static void *color_thread_func(void *arg) {
    ColorThreadArg &a = *(ColorThreadArg*)arg;
    ProjectionBatch &batch = *a.batch;
    const GaussianScene &scene = *a.scene;

    for (uint32_t j = a.j_start; j < a.j_end; j++) {
        uint32_t idx = batch.visible_indices[j];

        if (j + 1 < a.j_end) {
            uint32_t next_idx = batch.visible_indices[j + 1];
            __builtin_prefetch(&scene.sh_r[next_idx], 0, 1);
            if (scene.sh_rest && a.rest_per_vertex > 0)
                __builtin_prefetch(&scene.sh_rest[next_idx * a.rest_per_vertex], 0, 1);
        }

        float dx = scene.pos_x[idx] - a.cam_pos[0];
        float dy = scene.pos_y[idx] - a.cam_pos[1];
        float dz = scene.pos_z[idx] - a.cam_pos[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq > 1e-16f) {
            float inv_len = fast_rsqrtf(dist_sq);
            dx *= inv_len; dy *= inv_len; dz *= inv_len;
        }

        float cr, cg, cb_val;
        const float *sh_rest_ptr = (a.rest_per_vertex > 0 && scene.sh_rest) ?
                                    &scene.sh_rest[idx * a.rest_per_vertex] : nullptr;
        eval_sh(a.sh_degree, scene.sh_r[idx], scene.sh_g[idx], scene.sh_b[idx],
                sh_rest_ptr, a.basis_per_channel, dx, dy, dz, cr, cg, cb_val);

        batch.color_r[j] = cr;
        batch.color_g[j] = cg;
        batch.color_b[j] = cb_val;
    }
    return nullptr;
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

// --- Phase 2: Screen projection + covariance + radius (multi-threaded) ---

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

    uint32_t n = batch.cull_count;
    uint32_t num_threads = NUM_PROJ_THREADS;
    if (n < MT_THRESHOLD) num_threads = 1;

    pthread_t threads[NUM_PROJ_THREADS];
    CovThreadArg args[NUM_PROJ_THREADS];
    uint32_t chunk = n / num_threads;

    for (uint32_t t = 0; t < num_threads; t++) {
        args[t].scene = &scene;
        args[t].view_matrix = cam.view_matrix;
        args[t].batch = &batch;
        args[t].j_start = t * chunk;
        args[t].j_end = (t == num_threads - 1) ? n : (t + 1) * chunk;
        args[t].out_count = 0;
        args[t].tan_fovx = tan_fovx;  args[t].tan_fovy = tan_fovy;
        args[t].focal_x = focal_x;    args[t].focal_y = focal_y;
        args[t].half_w = half_w;       args[t].half_h = half_h;
        args[t].screen_w = screen_w;   args[t].screen_h = screen_h;
    }

    // Launch worker threads (main thread handles chunk 0)
    for (uint32_t t = 1; t < num_threads; t++)
        pthread_create(&threads[t], nullptr, cov_thread_func, &args[t]);
    cov_thread_func(&args[0]);
    for (uint32_t t = 1; t < num_threads; t++)
        pthread_join(threads[t], nullptr);

    // Compact: shift each thread's output to be contiguous
    uint32_t total_out = args[0].out_count;
    for (uint32_t t = 1; t < num_threads; t++) {
        if (args[t].out_count == 0) continue;
        uint32_t src = args[t].j_start;
        uint32_t dst = total_out;
        uint32_t cnt = args[t].out_count;
        if (src != dst) {
            memmove(&batch.visible_indices[dst], &batch.visible_indices[src], cnt * sizeof(uint32_t));
            memmove(&batch.screen_x[dst], &batch.screen_x[src], cnt * sizeof(float));
            memmove(&batch.screen_y[dst], &batch.screen_y[src], cnt * sizeof(float));
            memmove(&batch.depth[dst],    &batch.depth[src],    cnt * sizeof(float));
            memmove(&batch.cov2d_a[dst],  &batch.cov2d_a[src],  cnt * sizeof(float));
            memmove(&batch.cov2d_b[dst],  &batch.cov2d_b[src],  cnt * sizeof(float));
            memmove(&batch.cov2d_c[dst],  &batch.cov2d_c[src],  cnt * sizeof(float));
            memmove(&batch.radius[dst],   &batch.radius[src],   cnt * sizeof(float));
        }
        total_out += cnt;
    }

    batch.cov_count = total_out;
    return total_out;
}

// --- Phase 3: SH evaluation (multi-threaded) ---

void gs_project_color(const GaussianScene &scene, const CameraParams &cam,
                       ProjectionBatch &batch, int max_sh_degree) {
    if (batch.cov_count == 0) return;

    int sh_degree = scene.sh_degree;
    if (max_sh_degree >= 0 && max_sh_degree < sh_degree)
        sh_degree = max_sh_degree;
    int rest_per_vertex = sh_rest_count(sh_degree);
    int basis_per_channel = (sh_degree > 0) ? ((sh_degree + 1) * (sh_degree + 1) - 1) : 0;

    uint32_t n = batch.cov_count;
    uint32_t num_threads = NUM_PROJ_THREADS;
    if (n < MT_THRESHOLD) num_threads = 1;

    pthread_t threads[NUM_PROJ_THREADS];
    ColorThreadArg args[NUM_PROJ_THREADS];
    uint32_t chunk = n / num_threads;

    for (uint32_t t = 0; t < num_threads; t++) {
        args[t].scene = &scene;
        args[t].cam_pos = cam.position;
        args[t].batch = &batch;
        args[t].j_start = t * chunk;
        args[t].j_end = (t == num_threads - 1) ? n : (t + 1) * chunk;
        args[t].sh_degree = sh_degree;
        args[t].rest_per_vertex = rest_per_vertex;
        args[t].basis_per_channel = basis_per_channel;
    }

    // Launch worker threads (main thread handles chunk 0)
    for (uint32_t t = 1; t < num_threads; t++)
        pthread_create(&threads[t], nullptr, color_thread_func, &args[t]);
    color_thread_func(&args[0]);
    for (uint32_t t = 1; t < num_threads; t++)
        pthread_join(threads[t], nullptr);
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
