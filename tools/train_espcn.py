#!/usr/bin/env python3
"""Train ESPCN-x2 super-resolution model on 3DGS rendered frames.

Architecture matches gen_espcn_onnx.py exactly:
  Conv(3->64, 5x5, pad=2) -> ReLU
  Conv(64->32, 3x3, pad=1) -> ReLU
  Conv(32->12, 3x3, pad=1) -> PixelShuffle(2)

Training data: 1080p frames rendered via dump mode, box-downsampled to 540p.

Usage:
  # 1. Generate training data (no sudo needed):
  build/gs_splat ~/ply/Mars.ply -s 1 --dump data/train/mars -n 64
  build/gs_splat ~/ply/InteriorDesign.ply -s 1 --dump data/train/interior -n 64
  build/gs_splat ~/ply/Auditorium.ply -s 1 --dump data/train/auditorium -n 64

  # 2. Train:
  python3 tools/train_espcn.py --data data/train --epochs 100 --output data/models/espcn_weights.npz

  # 3. Export ONNX with trained weights:
  python3 tools/gen_espcn_onnx.py --weights data/models/espcn_weights.npz

Requires: pip3 install torch (CPU-only: --index-url https://download.pytorch.org/whl/cpu)
"""

import os
import sys
import argparse
import glob
import math
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("Error: PyTorch not found. Install with:")
    print("  pip3 install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not found. Install with: pip3 install Pillow")
    sys.exit(1)

SCALE = 2


class ESPCN(nn.Module):
    """ESPCN-x2 matching the ONNX architecture exactly."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * SCALE * SCALE, 3, padding=1)
        self.shuffle = nn.PixelShuffle(SCALE)  # DCR mode in PyTorch

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.shuffle(x)
        return x


def random_crop_batch(images, batch_size, crop_lr, augment=True):
    """Extract random crop pairs directly as a batch tensor. No DataLoader overhead."""
    crop_hr = crop_lr * SCALE
    bs = batch_size
    lr_batch = np.empty((bs, 3, crop_lr, crop_lr), dtype=np.float32)
    hr_batch = np.empty((bs, 3, crop_hr, crop_hr), dtype=np.float32)

    for i in range(bs):
        img = images[np.random.randint(len(images))]
        h, w = img.shape[:2]

        y = np.random.randint(0, h - crop_hr + 1)
        x = np.random.randint(0, w - crop_hr + 1)
        patch = img[y:y+crop_hr, x:x+crop_hr]

        # Augmentation
        if augment:
            if np.random.random() < 0.5:
                patch = patch[::-1, :, :]
            if np.random.random() < 0.5:
                patch = patch[:, ::-1, :]
            k = np.random.randint(0, 4)
            if k > 0:
                patch = np.rot90(patch, k)

        patch = np.ascontiguousarray(patch)
        hr = patch.astype(np.float32) / 255.0

        # 2x2 box downscale for LR
        lr = hr.reshape(crop_lr, SCALE, crop_lr, SCALE, 3).mean(axis=(1, 3))

        lr_batch[i] = lr.transpose(2, 0, 1)
        hr_batch[i] = hr.transpose(2, 0, 1)

    return torch.from_numpy(lr_batch), torch.from_numpy(hr_batch)


def calc_psnr(pred, target):
    """Calculate PSNR between two tensors (0-1 range)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def find_images(data_dir):
    """Find all training images in data directory (recursive)."""
    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.ppm']
    images = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    images.sort()
    return images


def train(args):
    device = torch.device('cpu')

    # Find training images
    all_paths = find_images(args.data)
    if len(all_paths) == 0:
        print(f"Error: No images found in {args.data}")
        print("Generate training data first:")
        print("  build/gs_splat ~/ply/Mars.ply -s 1 --dump data/train/mars -n 64")
        sys.exit(1)
    print(f"Found {len(all_paths)} images in {args.data}")

    # Split into train/val (90%/10%)
    np.random.seed(42)
    np.random.shuffle(all_paths)
    split = max(1, len(all_paths) // 10)
    val_paths = all_paths[:split]
    train_paths = all_paths[split:]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Load all images as uint8 (~6MB each, ~1.2GB for 192)
    print("Loading images...")
    train_imgs = []
    for i, p in enumerate(train_paths):
        train_imgs.append(np.array(Image.open(p).convert('RGB'), dtype=np.uint8))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(train_paths)} train...")
    print(f"  {len(train_imgs)} train images loaded")

    val_imgs = []
    for p in val_paths:
        val_imgs.append(np.array(Image.open(p).convert('RGB'), dtype=np.uint8))
    print(f"  {len(val_imgs)} val images loaded")

    # Model
    model = ESPCN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # Loss, optimizer, scheduler
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0.0
    best_weights = None

    for epoch in range(args.epochs):
        t0 = time.time()

        # Training: manual batch loop (no DataLoader overhead)
        model.train()
        train_loss = 0.0
        for bi in range(args.max_batches):
            lr_batch, hr_batch = random_crop_batch(
                train_imgs, args.batch_size, args.crop_size, augment=True)

            pred = model(lr_batch)
            loss = criterion(pred, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / args.max_batches
        elapsed = time.time() - t0

        # Validation (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            val_psnr = 0.0
            val_n = 0
            with torch.no_grad():
                for _ in range(min(50, len(val_imgs) * 4)):
                    lr_b, hr_b = random_crop_batch(val_imgs, args.batch_size, args.crop_size, augment=False)
                    pred = torch.clamp(model(lr_b), 0, 1)
                    val_psnr += calc_psnr(pred, hr_b)
                    val_n += 1

            avg_psnr = val_psnr / max(1, val_n)
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | L1={avg_loss:.5f} | "
                  f"PSNR={avg_psnr:.2f}dB | LR={lr:.6f} | {elapsed:.0f}s")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs} | L1={avg_loss:.5f} | LR={lr:.6f} | {elapsed:.0f}s")

    print(f"\nBest validation PSNR: {best_psnr:.2f} dB")

    # Save weights as numpy (for gen_espcn_onnx.py --weights)
    if best_weights is None:
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.savez(args.output,
             w1=model.conv1.weight.detach().numpy(),
             b1=model.conv1.bias.detach().numpy(),
             w2=model.conv2.weight.detach().numpy(),
             b2=model.conv2.bias.detach().numpy(),
             w3=model.conv3.weight.detach().numpy(),  # DCR ordering (PyTorch native)
             b3=model.conv3.bias.detach().numpy())
    print(f"Saved weights: {args.output}")
    print(f"\nNext: python3 tools/gen_espcn_onnx.py --weights {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Train ESPCN-x2 super-resolution')
    parser.add_argument('--data', type=str, default='data/train',
                        help='Training data directory (default: data/train)')
    parser.add_argument('--output', type=str, default='data/models/espcn_weights.npz',
                        help='Output weights path (default: data/models/espcn_weights.npz)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--crop-size', type=int, default=48,
                        help='LR crop size (default: 48)')
    parser.add_argument('--max-batches', type=int, default=100,
                        help='Batches per epoch (default: 100)')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
