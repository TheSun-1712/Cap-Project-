"""Phase 7: Procedural Synthetic Defect Generator for Pencils & Manufactured Parts.

Generates realistic industrial defects on normal surface images:
1. Scratches: Random thin bezier / polylines with varying brightness and width (1-3 px).
2. Dents / Depressions: Elliptical shadows with gradient lighting.
3. Chips / Voids: Irregular polygonal boundary erasures.
4. Stains / Discoloration: Perlin / Gaussian noise color blobs with alpha blending.
5. Cracks: Branching random-walk fractures.

Outputs paired synthetic defective images and binary ground-truth defect masks,
formatted for Anomalib's Folder / Custom Datamodule.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def add_scratch(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a thin, directional surface scratch."""
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    defective = image.copy()

    # Random starting point
    x1, y1 = random.randint(int(w * 0.1), int(w * 0.9)), random.randint(int(h * 0.1), int(h * 0.9))
    length = random.randint(30, 150)
    angle = random.uniform(0, 2 * np.pi)
    x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
    y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))

    thickness = random.randint(1, 3)
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    # Scratch color: brighter or darker than surrounding
    color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    cv2.line(defective, (x1, y1), (x2, y2), color, thickness)
    return defective, mask


def add_dent(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a localized dent or indentation with specular shadow."""
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    defective = image.copy().astype(np.float32)

    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    axes = (random.randint(8, 25), random.randint(8, 25))
    angle = random.randint(0, 180)

    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    # Darken indented region
    darken_factor = random.uniform(0.4, 0.7)
    dent_indices = mask > 0
    defective[dent_indices] *= darken_factor

    return np.clip(defective, 0, 255).astype(np.uint8), mask


def add_chip(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a chipped edge or flake."""
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    defective = image.copy()

    cx, cy = random.randint(int(w * 0.2), int(w * 0.8)), random.randint(int(h * 0.2), int(h * 0.8))
    num_pts = random.randint(4, 7)
    radius = random.randint(10, 30)

    pts = []
    for i in range(num_pts):
        th = i * (2 * np.pi / num_pts) + random.uniform(-0.3, 0.3)
        r = radius * random.uniform(0.6, 1.4)
        pts.append([int(cx + r * np.cos(th)), int(cy + r * np.sin(th))])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    chip_color = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))
    cv2.fillPoly(defective, pts, chip_color)
    return defective, mask


def add_stain(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate oil stain, ink blot, or discoloration."""
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(int(w * 0.2), int(w * 0.8)), random.randint(int(h * 0.2), int(h * 0.8))
    axes = (random.randint(15, 45), random.randint(15, 45))
    angle = random.randint(0, 180)

    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    # Gaussian blur mask to create smooth gradient
    mask = cv2.GaussianBlur(mask, (31, 31), 10)

    alpha = (mask / 255.0)[:, :, np.newaxis] * random.uniform(0.4, 0.7)
    stain_color = np.array([random.randint(20, 70), random.randint(20, 60), random.randint(10, 40)], dtype=np.float32)

    defective = (image * (1.0 - alpha) + stain_color * alpha).astype(np.uint8)
    binary_mask = (mask > 30).astype(np.uint8) * 255
    return defective, binary_mask


def generate_synthetic_dataset(
    input_normal_dir: str,
    output_dir: str,
    num_synthetic_per_image: int = 2,
):
    """Generate synthetic anomalies from a collection of normal product images."""
    in_dir = Path(input_normal_dir)
    out_dir = Path(output_dir)

    out_img_dir = out_dir / "test" / "NG"
    out_mask_dir = out_dir / "ground_truth" / "NG"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    normal_images = list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg"))
    if not normal_images:
        print(f"[!] No images found in {in_dir}")
        return

    print(f"[*] Processing {len(normal_images)} normal images into synthetic dataset...")
    generators = [
        ("scratch", add_scratch),
        ("dent", add_dent),
        ("chip", add_chip),
        ("stain", add_stain),
    ]

    count = 0
    for img_p in normal_images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        for i in range(num_synthetic_per_image):
            defect_type, gen_func = random.choice(generators)
            defective_img, mask = gen_func(img)

            stem = img_p.stem
            out_img_name = f"{stem}_synth_{defect_type}_{i}.png"
            out_mask_name = f"{stem}_synth_{defect_type}_{i}_mask.png"

            cv2.imwrite(str(out_img_dir / out_img_name), defective_img)
            cv2.imwrite(str(out_mask_dir / out_mask_name), mask)
            count += 1

    print(f"[+] Successfully generated {count} paired synthetic defect images and masks in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic defects for pencils and parts.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to normal reference images")
    parser.add_argument("--output-dir", type=str, default="./datasets/synthetic_pencils", help="Output directory")
    parser.add_argument("--num-per-image", type=int, default=3, help="Number of synthetic samples per normal image")
    args = parser.parse_args()

    generate_synthetic_dataset(args.input_dir, args.output_dir, args.num_per_image)


if __name__ == "__main__":
    main()
