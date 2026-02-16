// Generate a synthetic 3DGS PLY file for testing
// Creates a colorful sphere of Gaussians
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>

struct Gaussian {
    float x, y, z;           // position
    float nx, ny, nz;        // normal (unused, set to 0)
    float f_dc_0, f_dc_1, f_dc_2;  // SH DC (RGB)
    float opacity;           // logit(opacity)
    float scale_0, scale_1, scale_2;  // log(scale)
    float rot_0, rot_1, rot_2, rot_3; // quaternion (w,x,y,z)
};

// Inverse sigmoid: logit(p) = log(p / (1-p))
static float logit(float p) {
    if (p <= 0.001f) p = 0.001f;
    if (p >= 0.999f) p = 0.999f;
    return logf(p / (1.0f - p));
}

int main(int argc, char *argv[]) {
    int num_gaussians = 5000;
    const char *output = "data/test_sphere.ply";
    float radius = 2.0f;

    if (argc > 1) num_gaussians = atoi(argv[1]);
    if (argc > 2) output = argv[2];

    printf("Generating %d Gaussians -> %s\n", num_gaussians, output);

    FILE *fp = fopen(output, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s for writing\n", output);
        return 1;
    }

    // Write PLY header
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", num_gaussians);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property float nx\n");
    fprintf(fp, "property float ny\n");
    fprintf(fp, "property float nz\n");
    fprintf(fp, "property float f_dc_0\n");
    fprintf(fp, "property float f_dc_1\n");
    fprintf(fp, "property float f_dc_2\n");
    fprintf(fp, "property float opacity\n");
    fprintf(fp, "property float scale_0\n");
    fprintf(fp, "property float scale_1\n");
    fprintf(fp, "property float scale_2\n");
    fprintf(fp, "property float rot_0\n");
    fprintf(fp, "property float rot_1\n");
    fprintf(fp, "property float rot_2\n");
    fprintf(fp, "property float rot_3\n");
    fprintf(fp, "end_header\n");

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    for (int i = 0; i < num_gaussians; i++) {
        Gaussian g;
        memset(&g, 0, sizeof(g));

        // Random point on/near sphere using Fibonacci spiral + noise
        float t = (float)i / (float)num_gaussians;
        float golden = (1.0f + sqrtf(5.0f)) / 2.0f;
        float theta = 2.0f * M_PI * i / golden;
        float phi = acosf(1.0f - 2.0f * t);

        float r = radius * (0.8f + 0.4f * uniform01(rng));
        g.x = r * sinf(phi) * cosf(theta);
        g.y = r * sinf(phi) * sinf(theta);
        g.z = r * cosf(phi);

        // Add some noise
        g.x += normal(rng) * 0.05f;
        g.y += normal(rng) * 0.05f;
        g.z += normal(rng) * 0.05f;

        g.nx = g.ny = g.nz = 0.0f;

        // SH DC color: map position to color (rainbow sphere)
        // SH DC coefficients are in range ~ [-1, 1], will be converted to color via
        // C = 0.5 + 0.2820948 * sh_dc (the C0 SH constant is 0.2820948)
        // So sh_dc ≈ (color - 0.5) / 0.2820948
        float norm_y = (g.y + radius) / (2.0f * radius);  // 0..1
        float norm_x = (g.x + radius) / (2.0f * radius);
        float norm_z = (g.z + radius) / (2.0f * radius);

        // Map to vibrant colors
        float color_r = 0.3f + 0.7f * norm_y;
        float color_g = 0.3f + 0.7f * norm_x;
        float color_b = 0.3f + 0.7f * norm_z;

        // Convert to SH DC: color = 0.5 + C0 * sh_dc => sh_dc = (color - 0.5) / C0
        const float C0 = 0.2820948f;
        g.f_dc_0 = (color_r - 0.5f) / C0;
        g.f_dc_1 = (color_g - 0.5f) / C0;
        g.f_dc_2 = (color_b - 0.5f) / C0;

        // Opacity: high (0.8-0.95)
        float opacity_val = 0.8f + 0.15f * uniform01(rng);
        g.opacity = logit(opacity_val);

        // Scale: small Gaussians (log space)
        float base_scale = 0.02f + 0.01f * uniform01(rng);
        g.scale_0 = logf(base_scale);
        g.scale_1 = logf(base_scale);
        g.scale_2 = logf(base_scale);

        // Rotation: identity quaternion with slight perturbation
        float qx = normal(rng) * 0.1f;
        float qy = normal(rng) * 0.1f;
        float qz = normal(rng) * 0.1f;
        float qw = 1.0f;
        float qnorm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        g.rot_0 = qw / qnorm;
        g.rot_1 = qx / qnorm;
        g.rot_2 = qy / qnorm;
        g.rot_3 = qz / qnorm;

        fwrite(&g, sizeof(g), 1, fp);
    }

    fclose(fp);
    printf("Written %d Gaussians (%zu bytes each, total %zu bytes payload)\n",
           num_gaussians, sizeof(Gaussian), (size_t)num_gaussians * sizeof(Gaussian));
    return 0;
}
