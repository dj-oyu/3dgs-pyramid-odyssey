#include "gs_ply_loader.h"
#include "gs_memory.h"
#include "gs_math.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

struct PlyProperty {
    std::string name;
    std::string type;
    int byte_size;
};

static int type_size(const std::string &t) {
    if (t == "float" || t == "float32")  return 4;
    if (t == "double" || t == "float64") return 8;
    if (t == "uchar" || t == "uint8")    return 1;
    if (t == "char" || t == "int8")      return 1;
    if (t == "ushort" || t == "uint16")  return 2;
    if (t == "short" || t == "int16")    return 2;
    if (t == "uint" || t == "uint32")    return 4;
    if (t == "int" || t == "int32")      return 4;
    return 0;
}

static float read_float_prop(const uint8_t *data, const PlyProperty &prop) {
    if (prop.type == "float" || prop.type == "float32") {
        float v;
        memcpy(&v, data, 4);
        return v;
    }
    if (prop.type == "double" || prop.type == "float64") {
        double v;
        memcpy(&v, data, 8);
        return static_cast<float>(v);
    }
    if (prop.type == "uchar" || prop.type == "uint8") {
        return static_cast<float>(data[0]) / 255.0f;
    }
    return 0.0f;
}

bool gs_ply_load(const std::string &path, GaussianScene &scene) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[gs_ply] Cannot open: %s\n", path.c_str());
        return false;
    }

    // Parse header
    std::vector<PlyProperty> properties;
    uint32_t num_vertices = 0;
    bool is_binary = false;
    bool header_done = false;
    char line[512];

    while (fgets(line, sizeof(line), f)) {
        // Strip newline
        char *nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        char *cr = strchr(line, '\r');
        if (cr) *cr = '\0';

        if (strncmp(line, "element vertex ", 15) == 0) {
            num_vertices = atoi(line + 15);
        } else if (strncmp(line, "property ", 9) == 0) {
            // Parse "property <type> <name>"
            char type_str[64] = {}, name_str[64] = {};
            if (sscanf(line + 9, "%63s %63s", type_str, name_str) == 2) {
                PlyProperty p;
                p.type = type_str;
                p.name = name_str;
                p.byte_size = type_size(type_str);
                properties.push_back(p);
            }
        } else if (strncmp(line, "format binary_little_endian", 27) == 0) {
            is_binary = true;
        } else if (strcmp(line, "end_header") == 0) {
            header_done = true;
            break;
        }
    }

    if (!header_done || num_vertices == 0) {
        fprintf(stderr, "[gs_ply] Invalid PLY header in %s\n", path.c_str());
        fclose(f);
        return false;
    }

    printf("[gs_ply] Loading %u vertices, %zu properties, binary=%d\n",
           num_vertices, properties.size(), is_binary);

    // Find property indices
    auto find_prop = [&](const char *name) -> int {
        for (size_t i = 0; i < properties.size(); i++) {
            if (properties[i].name == name) return (int)i;
        }
        return -1;
    };

    int idx_x = find_prop("x"), idx_y = find_prop("y"), idx_z = find_prop("z");
    int idx_opacity = find_prop("opacity");
    int idx_scale_0 = find_prop("scale_0");
    int idx_scale_1 = find_prop("scale_1");
    int idx_scale_2 = find_prop("scale_2");
    int idx_rot_0 = find_prop("rot_0");
    int idx_rot_1 = find_prop("rot_1");
    int idx_rot_2 = find_prop("rot_2");
    int idx_rot_3 = find_prop("rot_3");

    // SH: f_dc_0/1/2 and f_rest_0..f_rest_N
    int idx_dc_0 = find_prop("f_dc_0");
    int idx_dc_1 = find_prop("f_dc_1");
    int idx_dc_2 = find_prop("f_dc_2");

    // Count f_rest properties
    int num_rest = 0;
    while (find_prop(("f_rest_" + std::to_string(num_rest)).c_str()) >= 0) {
        num_rest++;
    }

    // Determine SH degree from rest count
    // rest_count = (SH_COEFFS - 1) * 3
    // degree 0: 0 rest, degree 1: 9, degree 2: 24, degree 3: 45
    int sh_degree = 0;
    if (num_rest >= 45) sh_degree = 3;
    else if (num_rest >= 24) sh_degree = 2;
    else if (num_rest >= 9) sh_degree = 1;

    if (idx_x < 0 || idx_y < 0 || idx_z < 0) {
        fprintf(stderr, "[gs_ply] Missing position properties\n");
        fclose(f);
        return false;
    }

    // Compute byte offset for each property
    std::vector<int> offsets(properties.size());
    int vertex_size = 0;
    for (size_t i = 0; i < properties.size(); i++) {
        offsets[i] = vertex_size;
        vertex_size += properties[i].byte_size;
    }

    printf("[gs_ply] Vertex size: %d bytes, SH degree: %d, SH rest: %d\n",
           vertex_size, sh_degree, num_rest);

    // Allocate scene arrays in cached heap memory (much faster for CPU reads)
    scene.num_gaussians = num_vertices;
    scene.sh_degree = sh_degree;

    scene.pos_x   = gs_heap_alloc_floats(num_vertices);
    scene.pos_y   = gs_heap_alloc_floats(num_vertices);
    scene.pos_z   = gs_heap_alloc_floats(num_vertices);
    scene.sh_r    = gs_heap_alloc_floats(num_vertices);
    scene.sh_g    = gs_heap_alloc_floats(num_vertices);
    scene.sh_b    = gs_heap_alloc_floats(num_vertices);
    scene.rot_w   = gs_heap_alloc_floats(num_vertices);
    scene.rot_x   = gs_heap_alloc_floats(num_vertices);
    scene.rot_y   = gs_heap_alloc_floats(num_vertices);
    scene.rot_z   = gs_heap_alloc_floats(num_vertices);
    scene.scale_x = gs_heap_alloc_floats(num_vertices);
    scene.scale_y = gs_heap_alloc_floats(num_vertices);
    scene.scale_z = gs_heap_alloc_floats(num_vertices);
    scene.opacity = gs_heap_alloc_floats(num_vertices);

    int rest_per_vertex = sh_rest_count(sh_degree);
    if (rest_per_vertex > 0) {
        scene.sh_rest = gs_heap_alloc_floats(num_vertices * rest_per_vertex);
    }

    // Verify allocations
    if (!scene.pos_x || !scene.pos_y || !scene.pos_z ||
        !scene.sh_r || !scene.sh_g || !scene.sh_b ||
        !scene.rot_w || !scene.opacity) {
        fprintf(stderr, "[gs_ply] Failed to allocate memory for scene\n");
        fclose(f);
        return false;
    }

    // Read vertex data
    std::vector<uint8_t> vertex_buf(vertex_size);

    // Initialize bbox
    scene.bbox_min[0] = scene.bbox_min[1] = scene.bbox_min[2] = 1e30f;
    scene.bbox_max[0] = scene.bbox_max[1] = scene.bbox_max[2] = -1e30f;

    for (uint32_t i = 0; i < num_vertices; i++) {
        if (is_binary) {
            if (fread(vertex_buf.data(), 1, vertex_size, f) != (size_t)vertex_size) {
                fprintf(stderr, "[gs_ply] Unexpected EOF at vertex %u\n", i);
                fclose(f);
                return false;
            }
        } else {
            // ASCII: read line and parse
            if (!fgets(line, sizeof(line), f)) {
                fprintf(stderr, "[gs_ply] Unexpected EOF at vertex %u\n", i);
                fclose(f);
                return false;
            }
            // Parse space-separated floats into vertex_buf as float32
            const char *ptr = line;
            for (size_t p = 0; p < properties.size(); p++) {
                float val = strtof(ptr, const_cast<char**>(&ptr));
                memcpy(&vertex_buf[offsets[p]], &val, 4);
            }
        }

        // Extract properties
        float px = read_float_prop(&vertex_buf[offsets[idx_x]], properties[idx_x]);
        float py = read_float_prop(&vertex_buf[offsets[idx_y]], properties[idx_y]);
        float pz = read_float_prop(&vertex_buf[offsets[idx_z]], properties[idx_z]);

        scene.pos_x[i] = px;
        scene.pos_y[i] = py;
        scene.pos_z[i] = pz;

        // Update bbox
        scene.bbox_min[0] = std::min(scene.bbox_min[0], px);
        scene.bbox_min[1] = std::min(scene.bbox_min[1], py);
        scene.bbox_min[2] = std::min(scene.bbox_min[2], pz);
        scene.bbox_max[0] = std::max(scene.bbox_max[0], px);
        scene.bbox_max[1] = std::max(scene.bbox_max[1], py);
        scene.bbox_max[2] = std::max(scene.bbox_max[2], pz);

        // SH DC (f_dc_0/1/2) - stored as SH coefficient, convert to linear color
        if (idx_dc_0 >= 0) {
            float c0 = read_float_prop(&vertex_buf[offsets[idx_dc_0]], properties[idx_dc_0]);
            float c1 = read_float_prop(&vertex_buf[offsets[idx_dc_1]], properties[idx_dc_1]);
            float c2 = read_float_prop(&vertex_buf[offsets[idx_dc_2]], properties[idx_dc_2]);
            // SH DC coefficient: C0 = 0.28209479177387814
            constexpr float SH_C0 = 0.28209479177387814f;
            scene.sh_r[i] = c0;
            scene.sh_g[i] = c1;
            scene.sh_b[i] = c2;
            (void)SH_C0;
        } else {
            // Fallback: try red/green/blue or set default
            int idx_r = find_prop("red");
            int idx_g = find_prop("green");
            int idx_b = find_prop("blue");
            if (idx_r >= 0) {
                scene.sh_r[i] = read_float_prop(&vertex_buf[offsets[idx_r]], properties[idx_r]);
                scene.sh_g[i] = read_float_prop(&vertex_buf[offsets[idx_g]], properties[idx_g]);
                scene.sh_b[i] = read_float_prop(&vertex_buf[offsets[idx_b]], properties[idx_b]);
            } else {
                scene.sh_r[i] = scene.sh_g[i] = scene.sh_b[i] = 0.5f;
            }
        }

        // SH rest coefficients
        if (rest_per_vertex > 0 && scene.sh_rest) {
            int actual_rest = std::min(num_rest, rest_per_vertex);
            for (int r = 0; r < actual_rest; r++) {
                int ri = find_prop(("f_rest_" + std::to_string(r)).c_str());
                if (ri >= 0) {
                    scene.sh_rest[i * rest_per_vertex + r] =
                        read_float_prop(&vertex_buf[offsets[ri]], properties[ri]);
                }
            }
        }

        // Rotation quaternion
        if (idx_rot_0 >= 0) {
            scene.rot_w[i] = read_float_prop(&vertex_buf[offsets[idx_rot_0]], properties[idx_rot_0]);
            scene.rot_x[i] = read_float_prop(&vertex_buf[offsets[idx_rot_1]], properties[idx_rot_1]);
            scene.rot_y[i] = read_float_prop(&vertex_buf[offsets[idx_rot_2]], properties[idx_rot_2]);
            scene.rot_z[i] = read_float_prop(&vertex_buf[offsets[idx_rot_3]], properties[idx_rot_3]);
            // Normalize quaternion
            float qlen = sqrtf(scene.rot_w[i]*scene.rot_w[i] + scene.rot_x[i]*scene.rot_x[i] +
                               scene.rot_y[i]*scene.rot_y[i] + scene.rot_z[i]*scene.rot_z[i]);
            if (qlen > 1e-8f) {
                float inv = 1.0f / qlen;
                scene.rot_w[i] *= inv;
                scene.rot_x[i] *= inv;
                scene.rot_y[i] *= inv;
                scene.rot_z[i] *= inv;
            }
        } else {
            scene.rot_w[i] = 1.0f;
            scene.rot_x[i] = scene.rot_y[i] = scene.rot_z[i] = 0.0f;
        }

        // Scale (stored as log-scale, apply exp)
        if (idx_scale_0 >= 0) {
            scene.scale_x[i] = expf(read_float_prop(&vertex_buf[offsets[idx_scale_0]], properties[idx_scale_0]));
            scene.scale_y[i] = expf(read_float_prop(&vertex_buf[offsets[idx_scale_1]], properties[idx_scale_1]));
            scene.scale_z[i] = expf(read_float_prop(&vertex_buf[offsets[idx_scale_2]], properties[idx_scale_2]));
        } else {
            scene.scale_x[i] = scene.scale_y[i] = scene.scale_z[i] = 0.01f;
        }

        // Opacity (stored as logit, apply sigmoid)
        if (idx_opacity >= 0) {
            float raw = read_float_prop(&vertex_buf[offsets[idx_opacity]], properties[idx_opacity]);
            scene.opacity[i] = fast_sigmoid(raw);
        } else {
            scene.opacity[i] = 1.0f;
        }
    }

    fclose(f);

    printf("[gs_ply] Loaded %u Gaussians from %s\n", num_vertices, path.c_str());
    printf("[gs_ply] BBox: (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n",
           scene.bbox_min[0], scene.bbox_min[1], scene.bbox_min[2],
           scene.bbox_max[0], scene.bbox_max[1], scene.bbox_max[2]);

    return true;
}

void gs_scene_print_info(const GaussianScene &scene) {
    printf("=== Gaussian Scene Info ===\n");
    printf("  Gaussians: %u\n", scene.num_gaussians);
    printf("  SH Degree: %d\n", scene.sh_degree);
    printf("  BBox Min:  (%.4f, %.4f, %.4f)\n",
           scene.bbox_min[0], scene.bbox_min[1], scene.bbox_min[2]);
    printf("  BBox Max:  (%.4f, %.4f, %.4f)\n",
           scene.bbox_max[0], scene.bbox_max[1], scene.bbox_max[2]);

    float cx = (scene.bbox_min[0] + scene.bbox_max[0]) * 0.5f;
    float cy = (scene.bbox_min[1] + scene.bbox_max[1]) * 0.5f;
    float cz = (scene.bbox_min[2] + scene.bbox_max[2]) * 0.5f;
    printf("  Center:    (%.4f, %.4f, %.4f)\n", cx, cy, cz);

    float dx = scene.bbox_max[0] - scene.bbox_min[0];
    float dy = scene.bbox_max[1] - scene.bbox_min[1];
    float dz = scene.bbox_max[2] - scene.bbox_min[2];
    printf("  Extent:    (%.4f, %.4f, %.4f)\n", dx, dy, dz);

    // Memory estimate
    uint32_t n = scene.num_gaussians;
    size_t mem = n * 14 * sizeof(float);  // 14 float arrays
    if (scene.sh_rest) {
        mem += n * sh_rest_count(scene.sh_degree) * sizeof(float);
    }
    printf("  Est. Memory: %.1f MB\n", mem / (1024.0 * 1024.0));

    // Sample some opacity/scale stats
    if (n > 0) {
        float min_opacity = 1.0f, max_opacity = 0.0f;
        float avg_opacity = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            float o = scene.opacity[i];
            if (o < min_opacity) min_opacity = o;
            if (o > max_opacity) max_opacity = o;
            avg_opacity += o;
        }
        avg_opacity /= n;
        printf("  Opacity: min=%.4f, max=%.4f, avg=%.4f\n", min_opacity, max_opacity, avg_opacity);
    }
    printf("===========================\n");
}
