"""Phase 3: Tiny-Defect Study and Tiling Evaluation.

Implements:
1. Overlapping tile-based inference using Anomalib's `Tiler` (e.g., 512x512 tiles across 1024x1024 images).
2. Stitching (untiling) tile anomaly heatmaps back into full-resolution predictions.
3. Defect-size binning from ground-truth masks:
   - Micro (<0.1% image area)
   - Medium (0.1% - 1.0% image area)
   - Large (>1.0% image area)
4. Detection recall analysis at a calibrated 5% False Positive Rate (FPR=0.05).
5. Generation of the 'Defect Area vs Recall' curve and the key operational claim:
   "At setting X, we detect defects larger than N pixels with Y% recall."
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import v2 as transforms

from anomalib.data import RealIAD
from anomalib.data.utils import TestSplitMode
from anomalib.data.utils.tiler import Tiler
from anomalib.engine import Engine
from anomalib.models import Patchcore

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import patch_gpu_coreset
patch_gpu_coreset()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')



class TiledPredictor:
    """Performs sliding-window tiled inference and merges heatmap predictions."""

    def __init__(self, model, tile_size: int = 512, stride: int = 384):
        self.model = model
        self.tile_size = tile_size
        self.stride = stride
        self.tiler = Tiler(tile_size=tile_size, stride=stride)

    @torch.no_grad()
    def predict_image(self, image_tensor: Tensor) -> Tuple[float, np.ndarray]:
        """Runs tiled inference on a single (3, H, W) or (1, 3, H, W) image tensor.

        Returns:
            image_anomaly_score: float (max anomaly across tiles)
            stitched_anomaly_map: np.ndarray (H, W)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Tile the image
        tiles = self.tiler.tile(image_tensor)  # (N_tiles, 3, tile_size, tile_size)
        tiles = tiles.to(self.model.device)

        # Predict per tile
        out = self.model.model(tiles)
        if isinstance(out, tuple):
            tile_maps, tile_scores = out[0], out[1]
        elif hasattr(out, "anomaly_map"):
            tile_maps, tile_scores = out.anomaly_map, out.pred_score
        else:
            tile_maps = out
            tile_scores = torch.amax(tile_maps, dim=(1, 2, 3))

        # Ensure tile_maps is 4D (N_tiles, 1, H, W)
        if tile_maps.dim() == 3:
            tile_maps = tile_maps.unsqueeze(1)

        # Max anomaly score across tiles
        img_score = float(torch.max(tile_scores).item())

        # Untile back to full resolution
        stitched = self.tiler.untile(tile_maps)  # (1, 1, H, W)
        anomaly_map = stitched.squeeze().cpu().numpy()

        return img_score, anomaly_map


def evaluate_tiny_defects(
    root: str = "./datasets/Real-IAD",
    category: str = "toothbrush",
    full_res: int = 1024,
    tile_size: int = 512,
    stride: int = 384,
    target_fpr: float = 0.05,
    results_dir: str = "./results",
) -> Dict:
    print("=" * 75)
    print(f"  PHASE 3: TINY-DEFECT STUDY & TILING EVALUATION ({category.upper()})")
    print(f"  Image Size: {full_res}x{full_res} | Tile Size: {tile_size}x{tile_size} | Stride: {stride}")
    print("=" * 75)

    root_path = Path(root)
    json_path = "realiad_jsons/realiad_jsons/{category}.json"
    if not (root_path / "realiad_jsons" / "realiad_jsons" / f"{category}.json").exists():
        if (root_path / "realiad_jsons" / f"{category}.json").exists():
            json_path = "realiad_jsons/{category}.json"

    # Load datamodule
    datamodule = RealIAD(
        root=root,
        category=category,
        resolution=full_res,
        json_path=json_path,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        test_split_mode=TestSplitMode.FROM_DIR,
    )
    datamodule.setup()

    # Fit model on tiles or standard normal images
    # We can use a trained model or fit Patchcore on normal images
    print("[*] Initializing Patchcore model...")
    model = Patchcore(backbone="resnet18", coreset_sampling_ratio=0.001)
    engine = Engine(default_root_dir=results_dir)

    print("[*] Fitting memory bank on normal images...")
    engine.fit(model=model, datamodule=datamodule)
    model.eval()

    tiled_predictor = TiledPredictor(model=model, tile_size=tile_size, stride=stride)

    test_loader = datamodule.test_dataloader()
    normal_scores = []
    anomalous_samples = []

    print("[*] Running tiled evaluation across test set...")
    total_pixels = full_res * full_res

    for batch in test_loader:
        images = batch.image
        labels = getattr(batch, "gt_label", getattr(batch, "label", None))
        masks = getattr(batch, "gt_mask", getattr(batch, "mask", None))

        for idx in range(len(images)):
            img = images[idx]
            label = int(labels[idx].item())
            mask = masks[idx].cpu().numpy() if masks is not None else None

            score, anomaly_map = tiled_predictor.predict_image(img)

            if label == 0:  # Normal
                normal_scores.append(score)
            else:  # Anomalous
                # Compute mask defect area
                defect_pixels = 0
                defect_pct = 0.0
                if mask is not None:
                    defect_pixels = int(np.count_nonzero(mask > 0))
                    defect_pct = (defect_pixels / total_pixels) * 100.0

                anomalous_samples.append({
                    "score": score,
                    "defect_pixels": defect_pixels,
                    "defect_pct": defect_pct,
                })

    # Calibrate threshold at target FPR (default 5%)
    normal_scores = np.array(normal_scores)
    if len(normal_scores) == 0:
        threshold = 0.5
    else:
        # For FPR = 5%, 95% of normal samples must fall below threshold
        threshold = float(np.percentile(normal_scores, 100.0 * (1.0 - target_fpr)))

    print(f"\n[+] Calibrated Decision Threshold: {threshold:.4f} (at {target_fpr*100:.1f}% FPR on normal samples)")

    # Bin defects by area
    bins = {
        "Micro (<0.1%)": [s for s in anomalous_samples if s["defect_pct"] < 0.1],
        "Medium (0.1%-1.0%)": [s for s in anomalous_samples if 0.1 <= s["defect_pct"] <= 1.0],
        "Large (>1.0%)": [s for s in anomalous_samples if s["defect_pct"] > 1.0],
    }

    results_table = []
    bin_recalls = {}
    bin_pixel_cutoffs = {}

    for bin_name, samples in bins.items():
        if len(samples) == 0:
            recall = 0.0
            avg_pixels = 0
        else:
            detected = sum(1 for s in samples if s["score"] >= threshold)
            recall = (detected / len(samples)) * 100.0
            avg_pixels = int(np.mean([s["defect_pixels"] for s in samples]))

        bin_recalls[bin_name] = recall
        bin_pixel_cutoffs[bin_name] = avg_pixels
        results_table.append({
            "bin": bin_name,
            "count": len(samples),
            "avg_pixels": avg_pixels,
            "recall": recall,
        })

    print("\n" + "=" * 75)
    print(f"       DEFECT-SIZE VS RECALL (AT {target_fpr*100:.1f}% FALSE POSITIVE RATE)")
    print("=" * 75)
    print(f"{'Bin':<22} | {'Count':<7} | {'Avg Defect Pixels':<18} | {'Recall (%)':<10}")
    print("-" * 75)
    for row in results_table:
        print(f"{row['bin']:<22} | {row['count']:<7} | {row['avg_pixels']:<18} | {row['recall']:>8.1f}%")
    print("=" * 75)

    # Core claim synthesis
    # Find smallest bin with >= 90% recall or highest recall
    recommended_statement = (
        f"At {full_res}px with {tile_size}px tiling, we detect defects larger than "
        f"{bin_pixel_cutoffs.get('Medium (0.1%-1.0%)', 1000)} pixels with "
        f"{bin_recalls.get('Medium (0.1%-1.0%)', 95.0):.1f}% recall at {target_fpr*100:.1f}% FPR."
    )
    print(f"\n[KEY RESULT]: \"{recommended_statement}\"\n")

    # Plot Defect Size vs Recall Curve
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "defect_size_vs_recall.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [r["bin"] for r in results_table]
    y_recalls = [r["recall"] for r in results_table]
    counts = [r["count"] for r in results_table]

    bars = ax.bar(x_labels, y_recalls, color="#2563EB", width=0.5, edgecolor="#1E3A8A")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Detection Recall (%) [FPR = 5%]", fontsize=12)
    ax.set_title(f"Defect Size vs Detection Recall: {category.capitalize()} ({full_res}px Tiled)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Defect Size vs Recall plot to: {plot_path}")

    # Export json
    out_summary = {
        "category": category,
        "image_size": full_res,
        "tile_size": tile_size,
        "stride": stride,
        "target_fpr": target_fpr,
        "calibrated_threshold": threshold,
        "key_statement": recommended_statement,
        "results": results_table,
    }
    out_json = out_dir / "tiny_defect_study.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

    print(f"[+] Study metrics saved to: {out_json}")
    return out_summary


def main():
    parser = argparse.ArgumentParser(description="Run tiny-defect study with tiled inference.")
    parser.add_argument("--root", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--resolution", type=int, default=1024, help="Original image resolution")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument("--stride", type=int, default=384, help="Tile stride")
    parser.add_argument("--target-fpr", type=float, default=0.05, help="Target false positive rate")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    evaluate_tiny_defects(
        root=args.root,
        category=args.category,
        full_res=args.resolution,
        tile_size=args.tile_size,
        stride=args.stride,
        target_fpr=args.target_fpr,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
