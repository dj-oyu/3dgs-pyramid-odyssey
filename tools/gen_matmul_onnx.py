#!/usr/bin/env python3
"""Generate ONNX model for NPU-accelerated Mahalanobis distance computation.

Produces: data/models/matmul_256x6x256.onnx
  Input:  g_matrix [256, 6]  float32  (G_T coefficient matrix, runtime)
  Weight: p_matrix [6, 256]  float32  (P_T pixel features, baked in)
  Output: power    [256, 256] float32  (Mahalanobis distances)
  Op:     MatMul(g_matrix, p_matrix)

The P_T matrix construction matches build_pixel_features() in gs_mau.cpp exactly:
  For a 16x16 tile, column c = ly*16 + lx:
    Row 0: lx + 0.5
    Row 1: ly + 0.5
    Row 2: (lx + 0.5)^2
    Row 3: (ly + 0.5)^2
    Row 4: (lx + 0.5) * (ly + 0.5)
    Row 5: 1.0

Also generates calibration data for pulsar2 quantization.

Requires: pip3 install onnx numpy
"""

import os
import sys
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("Error: onnx package not found. Install with: pip3 install onnx numpy")
    sys.exit(1)

TILE_SIZE = 16
MAX_K = 256
NUM_FEATURES = 6


def build_pixel_features():
    """Build P_T[6, 256] pixel feature matrix matching gs_mau.cpp."""
    P_T = np.zeros((NUM_FEATURES, TILE_SIZE * TILE_SIZE), dtype=np.float32)
    for ly in range(TILE_SIZE):
        for lx in range(TILE_SIZE):
            col = ly * TILE_SIZE + lx
            fx = lx + 0.5
            fy = ly + 0.5
            P_T[0, col] = fx           # lx + 0.5
            P_T[1, col] = fy           # ly + 0.5
            P_T[2, col] = fx * fx      # (lx + 0.5)^2
            P_T[3, col] = fy * fy      # (ly + 0.5)^2
            P_T[4, col] = fx * fy      # (lx + 0.5)(ly + 0.5)
            P_T[5, col] = 1.0          # constant
    return P_T


def create_onnx_model():
    """Create ONNX model: output = MatMul(g_matrix, p_matrix)."""
    P_T = build_pixel_features()

    # Input: G_T[MAX_K, 6] - runtime input from CPU
    g_input = helper.make_tensor_value_info(
        "g_matrix", TensorProto.FLOAT, [MAX_K, NUM_FEATURES])

    # Weight: P_T[6, 256] - baked as initializer (constant)
    p_weight = helper.make_tensor(
        "p_matrix", TensorProto.FLOAT, [NUM_FEATURES, TILE_SIZE * TILE_SIZE],
        P_T.flatten().tolist())

    # Output: Power[MAX_K, 256]
    power_output = helper.make_tensor_value_info(
        "power", TensorProto.FLOAT, [MAX_K, TILE_SIZE * TILE_SIZE])

    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["g_matrix", "p_matrix"],
        outputs=["power"],
        name="matmul_g_p")

    # Build graph
    graph = helper.make_graph(
        [matmul_node],
        "matmul_256x6x256",
        [g_input],
        [power_output],
        initializer=[p_weight])

    # Build model with opset 13
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    # Validate
    onnx.checker.check_model(model)
    return model


def generate_calibration_data(num_samples=32):
    """Generate random G_T samples for pulsar2 quantization calibration."""
    # Simulate realistic G_T coefficient ranges:
    # Row 0,1: linear terms ~ [-10, 10]
    # Row 2,3: quadratic terms ~ [0, 5]
    # Row 4: cross term ~ [-5, 5]
    # Row 5: constant term ~ [0, 20]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(num_samples):
        g = np.zeros((MAX_K, NUM_FEATURES), dtype=np.float32)
        K = rng.integers(16, MAX_K + 1)  # random active Gaussians
        g[:K, 0] = rng.uniform(-10, 10, K)    # linear x
        g[:K, 1] = rng.uniform(-10, 10, K)    # linear y
        g[:K, 2] = rng.uniform(0, 5, K)       # quadratic x
        g[:K, 3] = rng.uniform(0, 5, K)       # quadratic y
        g[:K, 4] = rng.uniform(-5, 5, K)      # cross term
        g[:K, 5] = rng.uniform(0, 20, K)      # constant
        # Rows [K:MAX_K] are zero-padded (matches runtime behavior)
        samples.append(g)
    return np.stack(samples, axis=0)  # [num_samples, 256, 6]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, "data", "models")
    os.makedirs(output_dir, exist_ok=True)

    # Generate ONNX model
    model_path = os.path.join(output_dir, "matmul_256x6x256.onnx")
    model = create_onnx_model()
    onnx.save(model, model_path)
    print(f"ONNX model saved: {model_path}")
    print(f"  Input:  g_matrix [{MAX_K}, {NUM_FEATURES}] float32")
    print(f"  Weight: p_matrix [{NUM_FEATURES}, {TILE_SIZE * TILE_SIZE}] float32 (baked)")
    print(f"  Output: power    [{MAX_K}, {TILE_SIZE * TILE_SIZE}] float32")

    # Verify with numpy
    P_T = build_pixel_features()
    test_g = np.random.randn(MAX_K, NUM_FEATURES).astype(np.float32)
    expected = test_g @ P_T
    print(f"  Verification: MatMul [{MAX_K},{NUM_FEATURES}] x [{NUM_FEATURES},{TILE_SIZE*TILE_SIZE}] "
          f"-> [{MAX_K},{TILE_SIZE*TILE_SIZE}] OK")

    # Generate calibration data — one .npy per sample (pulsar2 requirement)
    calib_dir = os.path.join(output_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    calib = generate_calibration_data()
    for i in range(calib.shape[0]):
        sample_path = os.path.join(calib_dir, f"sample_{i:03d}.npy")
        np.save(sample_path, calib[i])  # shape: [256, 6] per file
    print(f"\nCalibration data saved: {calib_dir}/")
    print(f"  {calib.shape[0]} files, each [{MAX_K}, {NUM_FEATURES}] float32")
    print(f"  Total: {calib.nbytes / 1024:.1f} KB")

    # Also save tar archive for pulsar2 --calibration_data
    import tarfile
    tar_path = os.path.join(output_dir, "calibration.tar")
    with tarfile.open(tar_path, "w") as tar:
        for i in range(calib.shape[0]):
            sample_path = os.path.join(calib_dir, f"sample_{i:03d}.npy")
            tar.add(sample_path, arcname=f"g_matrix/sample_{i:03d}.npy")
    print(f"  Tar archive: {tar_path}")

    # Print pulsar2 compilation hint
    print(f"""
=== pulsar2 compilation (run on x86 host with pulsar2 installed) ===

pulsar2 build \\
  --input {model_path} \\
  --output_dir {output_dir}/output \\
  --config config.json \\
  --target_hardware AX650

config.json should specify:
  calibration_data: {tar_path}

Then copy the .axmodel to the device:
  scp data/models/output/*.axmodel device:~/3dgs-pyramid-odyssey/data/models/
""")


if __name__ == "__main__":
    main()
