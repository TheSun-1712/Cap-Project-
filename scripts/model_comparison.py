"""Phase 5: Model Comparison Framework.

Benchmarking candidate anomaly detection models on the same dataset:
1. PatchCore (WideResNet-50 / ResNet-18)
2. PaDiM (ResNet-18 Gaussian modeling)
3. EfficientAD (Student-Teacher distillation)

Measures and compiles into a single structured comparative table:
- Model Name & Backbone
- Resolution
- Image AUROC & Pixel AUROC
- Sample / Part Level AUROC
- Inference Latency (ms/image) & Throughput (FPS)
- Peak GPU VRAM allocated (GB)
- Exports Markdown table and JSON summary to `results/`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anomalib.data import RealIAD
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import patch_gpu_coreset
patch_gpu_coreset()
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')



def build_model(model_name: str, backbone: str = "resnet18"):
    """Instantiate the requested model architecture."""
    name = model_name.lower()
    if name == "patchcore":
        return Patchcore(backbone=backbone, coreset_sampling_ratio=0.001)
    elif name == "padim":
        padim_backbone = "resnet18" if "wide" in backbone else backbone
        return Padim(backbone=padim_backbone)
    elif name in ["efficientad", "efficient_ad"]:
        return EfficientAd(model_size="small")
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: patchcore, padim, efficientad")


def benchmark_model(
    model_name: str,
    root: str,
    category: str,
    resolution: int,
    backbone: str,
    results_dir: str,
) -> Dict:
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 75)
    print(f"  BENCHMARKING MODEL: {model_name.upper()} @ {resolution}x{resolution}")
    print("=" * 75)

    root_path = Path(root)
    json_path = "realiad_jsons/realiad_jsons/{category}.json"
    if not (root_path / "realiad_jsons" / "realiad_jsons" / f"{category}.json").exists():
        if (root_path / "realiad_jsons" / f"{category}.json").exists():
            json_path = "realiad_jsons/{category}.json"

    datamodule = RealIAD(
        root=root,
        category=category,
        resolution=resolution,
        json_path=json_path,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        test_split_mode=TestSplitMode.FROM_DIR,
    )
    datamodule.setup()

    model = build_model(model_name, backbone=backbone)
    engine_dir = Path(results_dir) / f"{model_name}_{resolution}"
    engine = Engine(default_root_dir=str(engine_dir))

    # Fit / Train
    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    fit_time = time.time() - t0

    # Test & Standard Metrics
    test_results = engine.test(model=model, datamodule=datamodule)

    # Extract image and pixel AUROC from test results
    img_auroc = 0.0
    px_auroc = 0.0
    for res_dict in test_results:
        for k, v in res_dict.items():
            if "image_AUROC" in k or (k == "image_auroc"):
                img_auroc = float(v)
            elif "pixel_AUROC" in k or (k == "pixel_auroc"):
                px_auroc = float(v)

    # Part-Level AUROC & Inference Speed Benchmark
    from collections import defaultdict
    part_groups = defaultdict(list)

    t_start = time.perf_counter()
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    total_infer_time = time.perf_counter() - t_start

    total_images_evaluated = 0
    for batch in predictions:
        scores = batch.pred_score.cpu().numpy()
        labels = getattr(batch, "gt_label", getattr(batch, "label", None)).cpu().numpy()
        image_paths = batch.image_path
        total_images_evaluated += len(image_paths)

        for img_p, lbl, sc in zip(image_paths, labels, scores):
            sample_id = Path(img_p).parent.name
            part_groups[sample_id].append({"ground_truth": int(lbl), "score": float(sc)})

    avg_latency_ms = float((total_infer_time / max(1, total_images_evaluated)) * 1000.0)
    throughput_fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    part_y_true = [1 if any(v["ground_truth"] == 1 for v in g) else 0 for g in part_groups.values()]
    part_scores = [max(v["score"] for v in g) for g in part_groups.values()]
    part_auroc = float(roc_auc_score(part_y_true, part_scores)) if len(np.unique(part_y_true)) > 1 else 0.0

    avg_latency_ms = float(np.mean(latencies) * 1000.0) if latencies else 0.0
    throughput_fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = float(torch.cuda.max_memory_allocated() / 1e9)

    summary = {
        "model": model_name,
        "backbone": backbone if model_name == "patchcore" else "default",
        "resolution": resolution,
        "image_auroc": img_auroc,
        "pixel_auroc": px_auroc,
        "part_auroc": part_auroc,
        "latency_ms": avg_latency_ms,
        "throughput_fps": throughput_fps,
        "peak_vram_gb": peak_vram,
        "fit_time_sec": fit_time,
    }

    print(f"[+] {model_name.upper()} Results:")
    print(f"    Image AUROC: {img_auroc:.4f} | Pixel AUROC: {px_auroc:.4f} | Part AUROC: {part_auroc:.4f}")
    print(f"    Latency:     {avg_latency_ms:.2f} ms/image ({throughput_fps:.1f} FPS)")
    print(f"    Peak VRAM:   {peak_vram:.3f} GB")
    return summary


def run_comparison(
    root: str = "./datasets/Real-IAD",
    category: str = "toothbrush",
    resolution: int = 512,
    models: List[str] = None,
    results_dir: str = "./results",
):
    if models is None:
        models = ["patchcore", "padim", "efficientad"]

    records = []
    for m in models:
        try:
            res = benchmark_model(
                model_name=m,
                root=root,
                category=category,
                resolution=resolution,
                backbone="wide_resnet50_2" if m == "patchcore" else "resnet18",
                results_dir=results_dir,
            )
            records.append(res)
        except Exception as e:
            print(f"[!] Error running {m}: {e}")

    # Generate Markdown Table
    md_lines = [
        f"# Model Comparison Benchmark: {category.capitalize()} ({resolution}x{resolution})",
        "",
        "| Model | Resolution | Image AUROC | Pixel AUROC | Part AUROC (5-View) | Latency (ms) | Throughput (FPS) | Peak VRAM (GB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in records:
        md_lines.append(
            f"| **{r['model'].upper()}** | {r['resolution']} | "
            f"{r['image_auroc']:.4f} | {r['pixel_auroc']:.4f} | {r['part_auroc']:.4f} | "
            f"{r['latency_ms']:.2f} ms | {r['throughput_fps']:.1f} | {r['peak_vram_gb']:.2f} GB |"
        )

    md_content = "\n".join(md_lines)
    print("\n" + md_content + "\n")

    out_p = Path(results_dir) / "model_comparison_table.md"
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_p = Path(results_dir) / "model_comparison.json"
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[+] Saved comparison table to: {out_p}")
    print(f"[+] Saved comparison json to:  {json_p}")


def main():
    parser = argparse.ArgumentParser(description="Compare PatchCore, PaDiM, and EfficientAD.")
    parser.add_argument("--root", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024], help="Resolution")
    parser.add_argument("--models", nargs="+", default=["patchcore", "padim", "efficientad"], help="List of models")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    run_comparison(
        root=args.root,
        category=args.category,
        resolution=args.resolution,
        models=args.models,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
