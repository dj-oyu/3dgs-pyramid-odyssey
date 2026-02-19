#!/usr/bin/env python3
"""Generate ESPCN-x2 super-resolution ONNX model.

Produces:
  data/models/espcn_x2.onnx         - ESPCN-x2 model
  data/models/espcn_calibration.tar  - Calibration data for pulsar2

Architecture (ESPCN-x2):
  Conv(3->64, 5x5, pad=2) -> ReLU
  Conv(64->32, 3x3, pad=1) -> ReLU
  Conv(32->12, 3x3, pad=1) -> PixelShuffle(2)

Input:  [1, 3, 540, 960]  float32 (NCHW, half-res)
Output: [1, 3, 1080, 1920] uint8 (NCHW, full-res, 0-255)

Modes:
  --random    Random weights (benchmark only, produces noise)
  --bilinear  Analytical bilinear upscale weights (pipeline verification)
  --weights <path>  Load trained weights from .npz file (DCR->CRD remapped)

Requires: pip3 install onnx numpy
"""

import os
import sys
import io
import tarfile
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("Error: onnx package not found. Install with: pip3 install onnx numpy")
    sys.exit(1)

# Model dimensions
SCALE = 2
IN_H, IN_W = 540, 960
OUT_H, OUT_W = IN_H * SCALE, IN_W * SCALE
IN_C = 3
OUT_C = 3


def kaiming_init(shape, fan_in):
    """Kaiming He initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.default_rng(42).normal(0, std, shape).astype(np.float32)


def bilinear_weights():
    """Create weights that make ESPCN perform bilinear interpolation.

    Strategy:
      Conv1 (3->64, 5x5): Identity pass-through for channels 0-2
      Conv2 (64->32, 3x3): Identity pass-through for channels 0-2
      Conv3 (32->12, 3x3) + PixelShuffle(2): Bilinear sub-pixel upscale

    PixelShuffle CRD mode maps 12 channels -> 3 colors x 2x2 subpixels:
      ch[c*4+0] -> subpixel (0,0): top-left     -> copy center pixel
      ch[c*4+1] -> subpixel (0,1): top-right     -> avg(center, right)
      ch[c*4+2] -> subpixel (1,0): bottom-left   -> avg(center, below)
      ch[c*4+3] -> subpixel (1,1): bottom-right   -> avg(center, right, below, below-right)
    """
    # Conv1: 3->64, 5x5. Identity for first 3 channels.
    w1 = np.zeros([64, IN_C, 5, 5], dtype=np.float32)
    for c in range(IN_C):
        w1[c, c, 2, 2] = 1.0  # center of 5x5
    b1 = np.zeros(64, dtype=np.float32)

    # Conv2: 64->32, 3x3. Identity for first 3 channels.
    w2 = np.zeros([32, 64, 3, 3], dtype=np.float32)
    for c in range(IN_C):
        w2[c, c, 1, 1] = 1.0  # center of 3x3
    b2 = np.zeros(32, dtype=np.float32)

    # Conv3: 32->12, 3x3. Bilinear sub-pixel weights.
    sub_pixel_c = OUT_C * SCALE * SCALE  # 12
    w3 = np.zeros([sub_pixel_c, 32, 3, 3], dtype=np.float32)
    b3 = np.zeros(sub_pixel_c, dtype=np.float32)

    for c in range(OUT_C):
        # Subpixel (0,0) top-left: copy center pixel
        w3[c * 4 + 0, c, 1, 1] = 1.0

        # Subpixel (0,1) top-right: avg(center, right)
        w3[c * 4 + 1, c, 1, 1] = 0.5
        w3[c * 4 + 1, c, 1, 2] = 0.5

        # Subpixel (1,0) bottom-left: avg(center, below)
        w3[c * 4 + 2, c, 1, 1] = 0.5
        w3[c * 4 + 2, c, 2, 1] = 0.5

        # Subpixel (1,1) bottom-right: avg(4 neighbors)
        w3[c * 4 + 3, c, 1, 1] = 0.25
        w3[c * 4 + 3, c, 1, 2] = 0.25
        w3[c * 4 + 3, c, 2, 1] = 0.25
        w3[c * 4 + 3, c, 2, 2] = 0.25

    return w1, b1, w2, b2, w3, b3


def load_trained_weights(npz_path):
    """Load trained weights from .npz and remap Conv3 from DCR to CRD ordering.

    PyTorch PixelShuffle uses DCR mode: channels are ordered as
      [r2*C, ...] where subpixel index varies fastest.
      For scale=2, C=3: [sub00_c0, sub01_c0, sub10_c0, sub11_c0, sub00_c1, ...]
      which is the same as CRD for our case (C outer, r*r inner).

    Actually for scale=2, C=3, total 12 channels:
      DCR: depth(r^2) varies fastest -> ch[d*C + c] = subpixel d, color c
           ch0=(0,0)R, ch1=(0,0)G, ch2=(0,0)B, ch3=(0,1)R, ...
      CRD: color varies slowest      -> ch[c*r^2 + d] = color c, subpixel d
           ch0=(0,0)R, ch1=(0,1)R, ch2=(1,0)R, ch3=(1,1)R, ch4=(0,0)G, ...

    Remapping: CRD_ch[c*4+d] = DCR_ch[d*3+c]
    """
    data = np.load(npz_path)
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']
    w3_dcr = data['w3']
    b3_dcr = data['b3']

    # Remap Conv3 weights from DCR to CRD channel ordering
    r2 = SCALE * SCALE  # 4
    w3_crd = np.zeros_like(w3_dcr)
    b3_crd = np.zeros_like(b3_dcr)
    for c in range(OUT_C):
        for d in range(r2):
            crd_idx = c * r2 + d
            dcr_idx = d * OUT_C + c
            w3_crd[crd_idx] = w3_dcr[dcr_idx]
            b3_crd[crd_idx] = b3_dcr[dcr_idx]

    print(f"  Loaded trained weights from {npz_path}")
    print(f"  Conv3 remapped DCR -> CRD ({w3_dcr.shape[0]} channels)")

    return w1, b1, w2, b2, w3_crd, b3_crd


def create_espcn_model(mode="bilinear", weights_path=None):
    """Create ESPCN-x2 ONNX model with uint8 output."""

    sub_pixel_c = OUT_C * SCALE * SCALE  # 12

    if weights_path:
        w1, b1, w2, b2, w3, b3 = load_trained_weights(weights_path)
    elif mode == "bilinear":
        w1, b1, w2, b2, w3, b3 = bilinear_weights()
    else:
        # Random weights (benchmark only)
        w1 = kaiming_init([64, IN_C, 5, 5], IN_C * 5 * 5)
        b1 = np.zeros(64, dtype=np.float32)
        w2 = kaiming_init([32, 64, 3, 3], 64 * 3 * 3)
        b2 = np.zeros(32, dtype=np.float32)
        w3 = kaiming_init([sub_pixel_c, 32, 3, 3], 32 * 3 * 3)
        b3 = np.zeros(sub_pixel_c, dtype=np.float32)

    w1_shape = list(w1.shape)
    w2_shape = list(w2.shape)
    w3_shape = list(w3.shape)

    # Define graph nodes
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["conv1"],
                         name="conv1", kernel_shape=[5, 5], pads=[2, 2, 2, 2]),
        helper.make_node("Relu", ["conv1"], ["relu1"], name="relu1"),
        helper.make_node("Conv", ["relu1", "w2", "b2"], ["conv2"],
                         name="conv2", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv2"], ["relu2"], name="relu2"),
        helper.make_node("Conv", ["relu2", "w3", "b3"], ["conv3"],
                         name="conv3", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("DepthToSpace", ["conv3"], ["shuffle_out"],
                         name="pixel_shuffle", blocksize=SCALE, mode="CRD"),
        # uint8 output: Mul(255) -> Clip(0,255) -> Cast(uint8)
        helper.make_node("Mul", ["shuffle_out", "scale_255"], ["scaled"],
                         name="mul_255"),
        helper.make_node("Clip", ["scaled", "clip_min", "clip_max"], ["clipped"],
                         name="clip_0_255"),
        helper.make_node("Cast", ["clipped"], ["output"],
                         name="cast_u8", to=TensorProto.UINT8),
    ]

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, IN_C, IN_H, IN_W])
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.UINT8, [1, OUT_C, OUT_H, OUT_W])

    initializers = [
        helper.make_tensor("w1", TensorProto.FLOAT, w1_shape, w1.flatten().tolist()),
        helper.make_tensor("b1", TensorProto.FLOAT, [64], b1.tolist()),
        helper.make_tensor("w2", TensorProto.FLOAT, w2_shape, w2.flatten().tolist()),
        helper.make_tensor("b2", TensorProto.FLOAT, [32], b2.tolist()),
        helper.make_tensor("w3", TensorProto.FLOAT, w3_shape, w3.flatten().tolist()),
        helper.make_tensor("b3", TensorProto.FLOAT, [sub_pixel_c], b3.tolist()),
        # Constants for uint8 conversion
        helper.make_tensor("scale_255", TensorProto.FLOAT, [], [255.0]),
        helper.make_tensor("clip_min", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("clip_max", TensorProto.FLOAT, [], [255.0]),
    ]

    graph = helper.make_graph(
        nodes, "espcn_x2",
        [input_info], [output_info],
        initializer=initializers)

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    return model


def generate_calibration_data(num_samples=32):
    """Generate synthetic image calibration data for pulsar2 quantization.

    Uses structured patterns (gradients, blocks) instead of pure noise
    for better quantization coverage of typical image content.
    """
    rng = np.random.default_rng(123)
    samples = []
    for i in range(num_samples):
        img = np.zeros((1, IN_C, IN_H, IN_W), dtype=np.float32)
        kind = i % 4
        if kind == 0:
            # Horizontal gradient
            grad = np.linspace(0, 1, IN_W, dtype=np.float32)
            for c in range(IN_C):
                offset = rng.uniform(0, 0.3)
                img[0, c, :, :] = np.clip(grad[None, :] + offset, 0, 1)
        elif kind == 1:
            # Vertical gradient
            grad = np.linspace(0, 1, IN_H, dtype=np.float32)
            for c in range(IN_C):
                offset = rng.uniform(0, 0.3)
                img[0, c, :, :] = np.clip(grad[:, None] + offset, 0, 1)
        elif kind == 2:
            # Checkerboard blocks
            block = rng.integers(8, 64)
            yy = np.arange(IN_H) // block
            xx = np.arange(IN_W) // block
            checker = ((yy[:, None] + xx[None, :]) % 2).astype(np.float32)
            for c in range(IN_C):
                lo, hi = rng.uniform(0, 0.4), rng.uniform(0.6, 1.0)
                img[0, c, :, :] = checker * (hi - lo) + lo
        else:
            # Smooth random patches (image-like)
            img[0] = rng.uniform(0, 1, (IN_C, IN_H, IN_W)).astype(np.float32)
        samples.append(img)
    return samples


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, "data", "models")
    os.makedirs(output_dir, exist_ok=True)

    # Parse mode
    mode = "bilinear"
    weights_path = None
    if "--random" in sys.argv:
        mode = "random"
    elif "--bilinear" in sys.argv:
        mode = "bilinear"

    # Check for --weights flag
    for i, arg in enumerate(sys.argv):
        if arg == "--weights" and i + 1 < len(sys.argv):
            weights_path = sys.argv[i + 1]
            mode = "trained"
            break

    # Generate ONNX model
    print(f"Generating ESPCN-x2 model (mode={mode})...")
    model = create_espcn_model(mode, weights_path)
    model_path = os.path.join(output_dir, "espcn_x2.onnx")
    onnx.save(model, model_path)
    model_size = os.path.getsize(model_path)
    print(f"  Saved: {model_path} ({model_size / 1024:.1f} KB)")
    print(f"  Input:  [1, {IN_C}, {IN_H}, {IN_W}] float32")
    print(f"  Output: [1, {OUT_C}, {OUT_H}, {OUT_W}] uint8")
    if weights_path:
        print(f"  Weights: trained ({weights_path})")
    elif mode == "bilinear":
        print(f"  Weights: analytical bilinear upscale (pipeline verification)")
    else:
        print(f"  Weights: random Kaiming init (benchmark only)")

    # Verify output dtype
    out_tensor = model.graph.output[0]
    out_dtype = out_tensor.type.tensor_type.elem_type
    assert out_dtype == TensorProto.UINT8, f"Output dtype is {out_dtype}, expected UINT8"
    print(f"  Output dtype verified: UINT8")

    # Generate calibration data
    print("\nGenerating calibration data...")
    samples = generate_calibration_data()

    tar_path = os.path.join(output_dir, "espcn_calibration.tar")
    with tarfile.open(tar_path, "w") as tar:
        for i, sample in enumerate(samples):
            buf = io.BytesIO()
            np.save(buf, sample)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"input/sample_{i:03d}.npy")
            info.size = buf.getbuffer().nbytes
            tar.addfile(info, buf)

    tar_size = os.path.getsize(tar_path)
    print(f"  Saved: {tar_path} ({tar_size / 1024 / 1024:.1f} MB)")
    print(f"  {len(samples)} samples, each [1, {IN_C}, {IN_H}, {IN_W}] float32")

    print(f"""
=== Next: compile with pulsar2 (on x86 host) ===

pulsar2 build \\
  --input {model_path} \\
  --output_dir {output_dir}/output \\
  --config {os.path.join(output_dir, 'espcn_config.json')} \\
  --target_hardware AX650

Then copy to device:
  scp output/compiled.axmodel device:data/models/espcn_x2.axmodel

Test:
  sudo ./build/gs_splat ~/ply/Mars.ply -s 2 --npu
""")


if __name__ == "__main__":
    main()
