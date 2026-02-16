#include "gs_math.h"
#include <cstring>

void mat4_multiply(const float *a, const float *b, float *out) {
    float tmp[16];
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a[k*4 + row] * b[col*4 + k];
            }
            tmp[col*4 + row] = sum;
        }
    }
    memcpy(out, tmp, 16 * sizeof(float));
}

void mat4_vec4(const float *m, const float *v, float *out) {
    float tmp[4];
    for (int row = 0; row < 4; row++) {
        tmp[row] = m[0*4+row]*v[0] + m[1*4+row]*v[1] + m[2*4+row]*v[2] + m[3*4+row]*v[3];
    }
    memcpy(out, tmp, 4 * sizeof(float));
}

void mat4_look_at(const float *eye, const float *center, const float *up, float *out) {
    float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
    vec3_normalize(f);

    float s[3];
    vec3_cross(f, up, s);
    vec3_normalize(s);

    float u[3];
    vec3_cross(s, f, u);

    memset(out, 0, 16 * sizeof(float));
    // Column-major
    out[0*4+0] = s[0];
    out[1*4+0] = s[1];
    out[2*4+0] = s[2];
    out[3*4+0] = -vec3_dot(s, eye);

    out[0*4+1] = u[0];
    out[1*4+1] = u[1];
    out[2*4+1] = u[2];
    out[3*4+1] = -vec3_dot(u, eye);

    out[0*4+2] = -f[0];
    out[1*4+2] = -f[1];
    out[2*4+2] = -f[2];
    out[3*4+2] = vec3_dot(f, eye);

    out[0*4+3] = 0.0f;
    out[1*4+3] = 0.0f;
    out[2*4+3] = 0.0f;
    out[3*4+3] = 1.0f;
}

void mat4_perspective(float fov_y_rad, float aspect, float near, float far, float *out) {
    float tan_half = tanf(fov_y_rad * 0.5f);

    memset(out, 0, 16 * sizeof(float));
    out[0*4+0] = 1.0f / (aspect * tan_half);
    out[1*4+1] = 1.0f / tan_half;
    out[2*4+2] = -(far + near) / (far - near);
    out[2*4+3] = -1.0f;
    out[3*4+2] = -(2.0f * far * near) / (far - near);
}
