"""Phase 2 & Phase 3 Resolution Ablation: PatchCore Baseline.

Trains PatchCore on normal images and evaluates on test anomalies.
Records:
- Image AUROC
- Pixel AUROC
- Image F1 / Pixel F1
- Execution Runtime (seconds)
- Peak GPU VRAM allocated (GB)
- Saves metrics to `results/baseline_{category}_{res}.json`.

Supports resolution ablation (256, 512, 1024) and lightweight backbones (e.g. resnet18).
"""

import argparse
import json
import time
from pathlib import Path
import torch

from anomalib.data import RealIAD
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import patch_gpu_coreset
patch_gpu_coreset()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')



def run_baseline(
    root: str = "./datasets/Real-IAD",
    category: str = "toothbrush",
    resolution: int = 256,
    backbone: str = "resnet18",
    coreset_ratio: float = 0.001,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 0,  # 0 is recommended on Windows to avoid spawn deadlocks
    results_dir: str = "./results",
) -> dict:
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("=" * 70)
    print(f"  PATCHCORE BASELINE: {category.upper()} @ {resolution}x{resolution}")
    print(f"  Backbone: {backbone} | Coreset ratio: {coreset_ratio}")
    print("=" * 70)

    start_time = time.time()

    # Configure DataModule
    # Check if nested json folder exists
    root_path = Path(root)
    json_path = "realiad_jsons/realiad_jsons/{category}.json"
    if not (root_path / "realiad_jsons" / "realiad_jsons" / f"{category}.json").exists():
        if (root_path / "realiad_jsons" / f"{category}.json").exists():
            json_path = "realiad_jsons/{category}.json"

    print(f"[*] Initializing RealIAD DataModule (resolution={resolution}, json={json_path})...")
    datamodule = RealIAD(
        root=root,
        category=category,
        resolution=resolution,
        json_path=json_path,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        test_split_mode=TestSplitMode.FROM_DIR,
    )

    # Configure Model
    print(f"[*] Initializing Patchcore with backbone={backbone}...")
    model = Patchcore(
        backbone=backbone,
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=coreset_ratio,
    )

    # Engine
    out_dir = Path(results_dir) / f"{category}_{resolution}_{backbone}"
    engine = Engine(default_root_dir=str(out_dir))

    # Train (fitting memory bank)
    print("[*] Fitting Patchcore memory bank on normal samples...")
    train_start = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_duration = time.time() - train_start
    print(f"[+] Memory bank fitting complete in {train_duration:.2f}s")

    # Test
    print("[*] Evaluating on test set...")
    test_start = time.time()
    test_results = engine.test(model=model, datamodule=datamodule)
    test_duration = time.time() - test_start
    total_duration = time.time() - start_time

    # Peak VRAM
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n" + "=" * 70)
    print("                 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Category:                 {category}")
    print(f"Resolution:               {resolution}x{resolution}")
    print(f"Backbone:                 {backbone}")
    print(f"Coreset Sampling Ratio:   {coreset_ratio}")
    print(f"Peak VRAM used:           {peak_vram_gb:.3f} GB")
    print(f"Fit duration:             {train_duration:.2f}s")
    print(f"Test duration:            {test_duration:.2f}s")
    print(f"Total runtime:            {total_duration:.2f}s")
    print(f"Test Metrics:             {test_results}")
    print("=" * 70)

    # Export metrics
    summary = {
        "category": category,
        "resolution": resolution,
        "backbone": backbone,
        "coreset_ratio": coreset_ratio,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "train_duration_sec": train_duration,
        "test_duration_sec": test_duration,
        "total_duration_sec": total_duration,
        "peak_vram_gb": peak_vram_gb,
        "metrics": test_results,
    }

    metrics_file = Path(results_dir) / f"baseline_{category}_{resolution}_{backbone}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Metrics exported to: {metrics_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run PatchCore baseline on Real-IAD.")
    parser.add_argument("--root", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024], help="Image resolution")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Feature extractor backbone")
    parser.add_argument("--coreset-ratio", type=float, default=0.001, help="Patchcore coreset sampling ratio")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (0 for Windows stability)")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    run_baseline(
        root=args.root,
        category=args.category,
        resolution=args.resolution,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
