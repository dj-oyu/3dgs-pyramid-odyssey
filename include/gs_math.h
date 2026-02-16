#pragma once

#include <cstdint>
#include <cmath>
#include <arm_neon.h>

// 4x4 matrix multiply (column-major)
void mat4_multiply(const float *a, const float *b, float *out);

// 4x4 matrix * vec4
void mat4_vec4(const float *m, const float *v, float *out);

// Build look-at view matrix (column-major)
void mat4_look_at(const float *eye, const float *center, const float *up, float *out);

// Build perspective projection matrix (column-major)
void mat4_perspective(float fov_y_rad, float aspect, float near, float far, float *out);

// Fast sigmoid approximation
inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fast exp approximation (Schraudolph's method for alpha blending)
inline float fast_exp(float x) {
    // Clamp to avoid overflow in integer conversion
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) return expf(80.0f);
    return expf(x);
}

// NEON: fast exp approximation for 4 floats
inline float32x4_t neon_fast_exp_neg(float32x4_t x) {
    // Compute exp(-x) using polynomial approximation
    // For alpha blending: exp(-0.5 * d^2), input x >= 0
    // Using 4th order Taylor: 1 - x + x^2/2 - x^3/6 + x^4/24
    // But clip to [0, ~7] range for reasonable accuracy
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t max_val = vdupq_n_f32(7.0f);
    x = vmaxq_f32(x, zero);
    x = vminq_f32(x, max_val);

    // Schraudolph-style: exp(-x) ≈ reinterpret(int(a - b*x))
    // a = 127*2^23 = 1065353216, b = 2^23/ln(2) ≈ 12102203
    float32x4_t b = vdupq_n_f32(12102203.0f);
    float32x4_t a = vdupq_n_f32(1065353216.0f);
    float32x4_t val = vsubq_f32(a, vmulq_f32(b, x));
    int32x4_t ival = vcvtq_s32_f32(val);
    return vreinterpretq_f32_s32(ival);
}

// Vector normalize (3-component)
inline void vec3_normalize(float *v) {
    float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-8f) {
        float inv = 1.0f / len;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
}

// Vector cross product
inline void vec3_cross(const float *a, const float *b, float *out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// Vector dot product
inline float vec3_dot(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
