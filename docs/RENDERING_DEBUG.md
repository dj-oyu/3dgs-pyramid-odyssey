# Rendering Debug Status

## Status: RESOLVED (2026-02-17)

The rendering pipeline produces correct output. The primary issue was **camera placement**,
not rendering math bugs. Secondary improvement: eigenvalue ratio capping for spike artifact
reduction.

## Root Cause Analysis

### Primary: Camera placed too far from scene

The initial `gs_camera_reset` used bounding-box diagonal to compute scene extent, placing
the camera at ~87 units from scenes where Gaussian scales are 0.01-0.1 units. This caused
all Gaussians to project to 1-3 pixels, producing a tiny blob in the center of a black frame.

**Fix**: Changed to mean-position + RMS-radius based camera placement. This places the
camera at ~5-10 units from the scene center, matching the scale at which the Gaussians
were trained.

```cpp
// Before: bbox diagonal (too far)
float extent = vec3_length(bbox_max - bbox_min);
float dist = extent * 1.2f;  // ~87 units

// After: RMS of Gaussian positions (correct scale)
float rms_dist = sqrt(mean(|pos - center|^2));
float dist = rms_dist * 1.5f;  // ~5-10 units
```

### Secondary: Spike artifacts from extreme eigenvalue ratios

Close-up Gaussians can have 2D covariance eigenvalue ratios >10,000:1, causing needle-like
line artifacts. Added eigenvalue ratio capping at 200:1 via isotropic covariance inflation.

**Fix**: In `gs_projector.cpp`, after computing 2D covariance eigenvalues, if
`lambda_max > 200 * lambda_min`, inflate the diagonal of the 2D covariance matrix
to bring the ratio down to 200:1.

## Fixes Applied (chronological)

### Sessions 1-2: Initial pipeline build
Full pipeline working: PLY load -> project -> sort -> rasterize -> display.

### Session 3: Rendering quality fixes
1. **Jacobian sign fix (j00, j02)** - Corrected signs for perspective projection derivative
2. **SH degree 3 implementation** - Was only using DC (degree 0), implemented full SH 0-3
3. **Y-axis negation removed** - Removed `-vy` in screen projection

### Session 4: Additional fixes
4. **SH view direction reversed** - Changed from `campos - pos` to `pos - campos`
5. **Jacobian clamping** - Clamps `vx/z` and `vy/z` to `1.3 * tan(fov/2)`
6. **Max radius cap** - Rejects Gaussians with radius > screen_width

### Session 5: Root cause identified and fixed
7. **Camera placement fix (ROOT CAUSE)** - Changed from bbox-diagonal to mean+RMS placement
8. **Headless mode** - Added headless rendering (skips AX_SYS_Init) for dump mode without sudo
9. **Eigenvalue ratio capping** - Cap eigenvalue ratio at 200:1 via isotropic inflation

## Rendering Quality Assessment

### InteriorDesign (136K Gaussians)
- frame_01 (RMS distance): Nearly reference quality. Room walls, furniture, carpet, curtains,
  flower vase, ceiling beams all clearly visible with correct colors.
- frame_00 (1.5x RMS): Some spike artifacts on edges from close Gaussians, center looks good.

### Mars (136K Gaussians)
- frame_01 (RMS distance): Dome structure, roads, vehicles, surrounding terrain clearly visible.
  Good match with reference image.
- frame_00 (1.5x RMS): Dome visible, surrounding area has expected edge artifacts.

### Auditorium (17K Gaussians)
- frame_01 (RMS distance): White church with cross, blue sky, architectural details clearly
  visible. Recognizable scene.
- Edge artifacts at scene boundaries are present in BOTH our renders and the reference viewer.

### Comparison with Reference
Reference images (from another 3DGS viewer) also show spike artifacts at scene boundaries.
This is inherent to 3DGS, not a bug in our renderer.

## Verified Correct (no bugs found)

The following components were thoroughly audited and found to be mathematically correct:

- **Projection formula**: `sx = fx * vx / z + W/2`, consistent with OpenGL camera convention
- **Jacobian**: Signs match projection, clamping prevents edge divergence
- **2D covariance**: `Σ_2D = (J * W) * Σ_3D * (J * W)^T` matches reference
- **3D covariance**: `Σ = R * S * S^T * R^T` with exp(scale) and sigmoid(opacity) activations
- **SH evaluation**: All 16 basis functions (degree 0-3), correct channel layout (R/G/B interleaved)
- **PLY loading**: exp(scale), sigmoid(opacity) activations correctly applied
- **Radix sort**: Float-to-sortable conversion handles sign bit correctly
- **Alpha compositing**: Front-to-back with transmittance tracking, correct blending formula

## File Reference

| File | Role | Key Functions |
|------|------|--------------|
| `src/gs_projector.cpp` | 3D->2D projection | `gs_project`, `project_cov2d`, `eval_sh`, `compute_cov3d` |
| `src/gs_rasterizer.cpp` | Tile-based alpha compositing | `rasterize_tile`, `gs_rasterize` |
| `src/gs_ply_loader.cpp` | PLY parser, SH/scale/opacity | `gs_ply_load` |
| `src/gs_sort.cpp` | Radix sort by depth | `gs_radix_sort_by_depth` |
| `src/gs_renderer.cpp` | Pipeline orchestrator | `gs_renderer_render_frame` |
| `src/gs_camera.cpp` | Camera + keyboard input | `gs_camera_get_params`, `gs_camera_reset` |
| `src/gs_math.cpp` | Matrix ops | `mat4_look_at`, `mat4_perspective` |
| `src/gs_display.cpp` | VO/FB display | `gs_display_send_frame` |

## Test Scenes

| Scene | Gaussians | SH Degree | Notes |
|-------|-----------|-----------|-------|
| Mars | 136,801 | 3 | Outdoor, Mars base with dome |
| Auditorium | 17,200 | 3 | Outdoor, white church by the sea |
| InteriorDesign | 136,801 | 3 | Indoor living room |

All stored at `~/ply/` as standard 3DGS PLY files (binary_little_endian).

## Dump Commands

```bash
# Headless dump (no sudo required)
build/gs_splat ~/ply/InteriorDesign.ply -s 2 --dump data/dump/interior -n 8
build/gs_splat ~/ply/Mars.ply -s 2 --dump data/dump/mars -n 8
build/gs_splat "~/ply/Auditorium by the sea.ply" -s 2 --dump data/dump/auditorium -n 8
```
