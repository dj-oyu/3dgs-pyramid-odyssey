#pragma once

#include "gs_types.h"
#include <string>

// Load a 3DGS PLY file into a GaussianScene (SoA layout, CMM-allocated)
// Supports binary_little_endian and ASCII formats.
// Post-processes: sigmoid(opacity), exp(scale)
bool gs_ply_load(const std::string &path, GaussianScene &scene);

// Print scene statistics
void gs_scene_print_info(const GaussianScene &scene);
