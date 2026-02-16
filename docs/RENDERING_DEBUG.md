# Rendering Debug Status

## Symptom

All three test scenes (Mars, Auditorium, InteriorDesign) produce visually broken dump images.
The Auditorium scene was partially recognizable (church/building visible in close-up) but
Mars and InteriorDesign show scattered/spiky artifacts with incorrect colors.

Test command:
```bash
sudo build/gs_splat ~/ply/Mars.ply -s 2 --dump /tmp/dump_mars_v2 -n 8
```

## Fixes Applied (chronological)

### Session 1-2: Initial pipeline build (Steps 1-8)
Full pipeline working: PLY load -> project -> sort -> rasterize -> display.

### Session 3: Rendering quality fixes

1. **Jacobian sign fix (j00, j02)** - Corrected signs for perspective projection derivative.
2. **SH degree 3 implementation** - Was only using DC (degree 0). Implemented full SH 0-3.
3. **Y-axis negation removed** - Removed `-vy` in screen projection. Rationale was that
   Auditorium appeared upside-down with the negation.

### Session 4 (current): Additional fixes

4. **SH view direction reversed** - Was `campos - pos`, changed to `pos - campos` to match
   reference 3DGS (`dir = pos - campos` in `computeColorFromSH`).
5. **Jacobian clamping added** - Clamps `vx/z` and `vy/z` to `1.3 * tan(fov/2)` to prevent
   extreme covariance for edge Gaussians (causes spike artifacts).
6. **Max radius cap** - Rejects Gaussians with radius > screen_width.

**Status**: Fixes 4-6 have NOT been verified with dump images yet.

---

## Known Issues & Suspected Root Causes

### Issue A: Y-axis convention mismatch (HIGH PRIORITY)

**The problem**: OpenGL camera (Y-up, Z-backward) vs COLMAP/3DGS (Y-down, Z-forward).

Our `mat4_look_at` produces an OpenGL view matrix:
- Row 0 = right vector (X)
- Row 1 = up vector (Y-up)
- Row 2 = -forward vector (Z-backward, vz < 0 for visible)

The reference 3DGS uses COLMAP camera convention:
- `t.z > 0` for visible objects
- `t.y` points down

**Screen projection comparison**:

| | Reference 3DGS | Our code |
|---|---|---|
| Screen X | `fx * t.x / t.z + W/2` | `fx * vx / (-vz) + W/2` |
| Screen Y | `fy * t.y / t.z + H/2` (t.y is Y-down) | `fy * vy / (-vz) + H/2` (vy is Y-up) |
| Jacobian j00 | `fx / t.z` | `-fx / vz` (equivalent) |
| Jacobian j11 | `fy / t.z` | `-fy / vz` |

**The critical question**: Without `-vy` in the Y projection, a point above the camera
(vy > 0) maps to `sy > H/2` (bottom of screen). This is Y-flipped vs the reference.

In Session 3, `-vy` was removed because the Auditorium appeared upside-down. However,
this may have been a cascading effect of other bugs at the time (Jacobian signs were
also wrong). **This needs re-evaluation with all other fixes in place.**

**Mathematical proof**:
```
Reference:  sy = fy * t.y / t.z + H/2
Converted:  sy = fy * (-vy) / (-vz) + H/2 = fy * vy / vz + H/2
Our code:   sy = fy * vy / (-vz) + H/2 = -fy * vy / vz + H/2  <-- DIFFERENT!
```

To match the reference, we need `sy = fy * (-vy) / (-vz) + H/2`, i.e., negate vy.
If `-vy` is restored, the Jacobian j11 and j12 must also change:
```
With -vy:    j11 = fy / vz (= fy * inv_z),  j12 = -fy * vy / vz²
Without -vy: j11 = -fy / vz,                 j12 = fy * vy / vz²
```

Both are internally consistent (Jacobian matches projection), so the **2D covariance
shape is correct either way**. The difference is only a vertical flip of the rendered
image. But this vertical flip also affects SH evaluation indirectly through the dump
camera positioning logic.

**Recommendation**: Try restoring `-vy` with the correct j11/j12 signs, test with
all three scenes.

### Issue B: Projection formula (vx/z vs -fx*vx/vz)

The projection in `project_one` uses:
```cpp
float inv_z = 1.0f / z;  // z = -vz > 0, so inv_z > 0
float sx = focal_x * vx * inv_z + half_w;
```

This gives `sx = fx * vx / z`. For a point to the right of the camera (vx > 0),
sx > half_w, correctly mapping to the right side of the screen.

The Jacobian in `project_cov2d` uses:
```cpp
float inv_z = 1.0f / tz;  // tz = vz < 0, so inv_z < 0
float j00 = -focal_x * inv_z;  // = -fx / vz = fx / z > 0
```

These are consistent. **No issue here.**

### Issue C: SH data layout in PLY

PLY stores `f_rest_0..f_rest_44` (for degree 3) with layout:
```
f_rest_0..14  = R channel, basis functions 0-14
f_rest_15..29 = G channel, basis functions 0-14
f_rest_30..44 = B channel, basis functions 0-14
```

This comes from `features_rest.transpose(1,2).flatten()` in the reference training code,
where `features_rest` has shape `[N, 15, 3]` (15 bases, 3 colors).

Our loader stores them linearly in `sh_rest[i * 45 + r]`, and `eval_sh` accesses them as:
```cpp
const float *sr = sh_rest;                       // R: indices 0-14
const float *sg = sh_rest + basis_per_channel;   // G: indices 15-29
const float *sb = sh_rest + 2 * basis_per_channel; // B: indices 30-44
```

**This is correct.** Verified against reference.

### Issue D: SH basis function convention

Our `eval_sh` for degree 1:
```cpp
r += SH_C1 * (-y * sr[0] + z * sr[1] - x * sr[2]);
```

Reference (`computeColorFromSH` in CUDA):
```cpp
result = result - C1 * y * sh[1] + C1 * z * sh[2] - C1 * x * sh[3];
```

These match (sh[1]=sr[0], sh[2]=sr[1], sh[3]=sr[2] since sh[0] is DC). **Correct.**

But note: the `x, y, z` in eval_sh are the view direction components. Since we
just fixed the direction from `campos - pos` to `pos - campos` (fix #4), the SH
should now be correct. **Needs verification.**

### Issue E: View matrix construction

`mat4_look_at` builds a standard OpenGL view matrix. The reference 3DGS uses COLMAP's
`getWorld2View2(R, t)` which produces a different matrix. However, both transform
world coordinates to their respective camera spaces correctly. The key is that our
Jacobian and projection are consistent with our camera space convention.

**No issue expected**, but could verify by comparing a known point's screen projection.

### Issue F: Covariance matrix convention

Our `compute_cov3d` computes `Σ = R * S * S^T * R^T = M * M^T` where `M = R * S`.
This is the WORLD-space 3D covariance. The reference does the same.

Then `project_cov2d` computes `Σ_2D = T * Σ_3D * T^T` where `T = J * W` (2x3).
J is the projection Jacobian (2x3), W is the 3x3 rotation part of the view matrix.

**This matches the reference.** The matrix multiplication order is correct.

---

## Debugging Strategy for Next Session

### Priority 1: Verify fixes 4-6 with dump images
Run dumps and visually inspect. If still broken, continue with Priority 2.

### Priority 2: Ground-truth comparison tool
Create a minimal Python script that:
1. Loads the same PLY file
2. Uses the EXACT same camera parameters (position, fov, view matrix)
3. Projects a few Gaussians to screen coordinates
4. Compares with our C++ output

This would definitively identify which stage (projection, covariance, SH, rasterization)
is producing wrong results.

```python
# Pseudocode for comparison tool
import numpy as np

# Load PLY, pick first 10 Gaussians
# Compute view matrix (same as our mat4_look_at)
# Project to screen: sx = fx*vx/z + W/2
# Compute 2D covariance: J * W * Sigma3D * W^T * J^T
# Print results
# Compare with our --dump diagnostic output
```

### Priority 3: Single-Gaussian test
Create a synthetic PLY with exactly 1 Gaussian at the origin.
Render from a known camera position. The result should be a single colored ellipse.
This isolates the rendering pipeline from scene complexity.

### Priority 4: Reference renderer comparison
Use the standard `diff-gaussian-rasterization` Python package to render the same
scene from the same viewpoint. Compare pixel-by-pixel.

```bash
pip install diff-gaussian-rasterization  # May need CUDA, could use CPU fallback
```

### Priority 5: Y-axis investigation
Test with `-vy` restored (and matching Jacobian j11/j12 signs). The covariance
shape won't change, but the vertical orientation of the image will. Check if
scenes appear more recognizable.

---

## File Reference

| File | Role | Key Functions |
|------|------|--------------|
| `src/gs_projector.cpp` | 3D->2D projection | `gs_project`, `project_cov2d`, `eval_sh`, `compute_cov3d` |
| `src/gs_rasterizer.cpp` | Tile-based alpha compositing | `rasterize_tile`, `gs_rasterize` |
| `src/gs_ply_loader.cpp` | PLY parser, SH/scale/opacity activation | `gs_ply_load` |
| `src/gs_sort.cpp` | Radix sort by depth | `gs_radix_sort_by_depth` |
| `src/gs_renderer.cpp` | Pipeline orchestrator | `gs_renderer_render_frame` |
| `src/gs_camera.cpp` | Camera + keyboard input | `gs_camera_get_params`, `gs_camera_reset` |
| `src/gs_math.cpp` | Matrix ops | `mat4_look_at`, `mat4_perspective` |
| `src/gs_display.cpp` | VO/FB display | `gs_display_send_frame` |

## Test Scenes

| Scene | Gaussians | SH Degree | Notes |
|-------|-----------|-----------|-------|
| Mars | ~200K? | 3 | Outdoor, planet surface |
| Auditorium | ~300K? | 3 | Indoor/outdoor, church building |
| InteriorDesign | ~200K? | 3 | Indoor room |

All stored at `~/ply/` as standard 3DGS PLY files (binary_little_endian).
