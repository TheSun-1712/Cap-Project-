"""Phase 4: Multi-View Decision per Part.

Solves the single-camera blind spot problem:
1. Groups test predictions by physical sample ID (part).
2. Aggregates the 5 camera viewpoints (C1, C2, C3, C4, C5):
   Part Score = max(view_1, view_2, view_3, view_4, view_5).
   Decision: Part FAILS if any view exceeds the threshold.
3. Compares Single-View AUROC (C1 through C5) vs Multi-View Aggregated AUROC.
4. Reports False Reject Rate (FRR) and False Accept Rate (FAR).
5. Generates the visual proof that a 5-view inspection system catches defects hidden to single cameras.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
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



def extract_view_and_sample_id(image_path: str) -> Tuple[str, str]:
    """Parse sample ID and camera view from image file path.

    Real-IAD convention: .../{sample_id}/{view_name}.(png|jpg)
    """
    p = Path(image_path)
    sample_id = p.parent.name
    filename = p.name.upper()

    view_name = "C1"
    for v in ["C1", "C2", "C3", "C4", "C5"]:
        if v in filename or v in str(p).upper():
            view_name = v
            break

    return sample_id, view_name


def run_multiview_evaluation(
    root: str = "./datasets/Real-IAD",
    category: str = "toothbrush",
    resolution: int = 256,
    backbone: str = "resnet18",
    coreset_ratio: float = 0.001,
    results_dir: str = "./results",
) -> Dict:
    print("=" * 75)
    print(f"  PHASE 4: MULTI-VIEW PER-PART EVALUATION ({category.upper()})")
    print(f"  Aggregating 5 viewpoints (C1-C5) into unified part decisions")
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

    print("[*] Training Patchcore model on normal images...")
    model = Patchcore(backbone=backbone, coreset_sampling_ratio=coreset_ratio)
    engine = Engine(default_root_dir=results_dir)
    engine.fit(model=model, datamodule=datamodule)
    model.eval()

    # Collect predictions per image
    print("[*] Running inference on all test images...")
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    records = []
    for batch in predictions:
        scores = batch.pred_score.cpu().numpy()
        labels = getattr(batch, "gt_label", getattr(batch, "label", None)).cpu().numpy()
        image_paths = batch.image_path

        for img_p, lbl, sc in zip(image_paths, labels, scores):
            sample_id, view_name = extract_view_and_sample_id(img_p)
            records.append({
                "image_path": str(img_p),
                "sample_id": sample_id,
                "view": view_name,
                "ground_truth": int(lbl),
                "score": float(sc),
            })

    print(f"[+] Processed {len(records)} test images across viewpoints.")

    # 1. Single View Evaluation (C1 to C5)
    views = ["C1", "C2", "C3", "C4", "C5"]
    single_view_metrics = {}

    for v in views:
        v_records = [r for r in records if r["view"] == v]
        if len(v_records) == 0:
            continue

        y_true = np.array([r["ground_truth"] for r in v_records])
        y_scores = np.array([r["score"] for r in v_records])

        # If both classes present
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_scores))
        else:
            auc = 0.0

        single_view_metrics[v] = {
            "image_count": len(v_records),
            "normal_count": int(np.sum(y_true == 0)),
            "defect_count": int(np.sum(y_true == 1)),
            "auroc": auc,
        }

    # 2. Multi-View Per-Part Aggregation
    part_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        part_groups[r["sample_id"]].append(r)

    part_y_true = []
    part_max_scores = []
    part_mean_scores = []

    defects_missed_by_view = {v: 0 for v in views}
    total_defective_parts = 0

    for sample_id, sample_views in part_groups.items():
        # A part is defective if any view has ground-truth defect
        part_label = 1 if any(v["ground_truth"] == 1 for v in sample_views) else 0
        part_y_true.append(part_label)

        v_scores = [v["score"] for v in sample_views]
        max_score = max(v_scores)
        mean_score = float(np.mean(v_scores))

        part_max_scores.append(max_score)
        part_mean_scores.append(mean_score)

        if part_label == 1:
            total_defective_parts += 1
            # Check which views had ground truth normal while another view had defect
            for v in sample_views:
                if v["ground_truth"] == 0:
                    defects_missed_by_view[v["view"]] += 1

    part_y_true = np.array(part_y_true)
    part_max_scores = np.array(part_max_scores)
    part_mean_scores = np.array(part_mean_scores)

    part_max_auroc = float(roc_auc_score(part_y_true, part_max_scores)) if len(np.unique(part_y_true)) > 1 else 0.0
    part_mean_auroc = float(roc_auc_score(part_y_true, part_mean_scores)) if len(np.unique(part_y_true)) > 1 else 0.0

    # Optimal threshold for multi-view at target 5% FPR (FRR = 5%)
    normal_part_scores = part_max_scores[part_y_true == 0]
    if len(normal_part_scores) > 0:
        threshold = float(np.percentile(normal_part_scores, 95.0))
        part_frr = float(np.mean(normal_part_scores >= threshold)) * 100.0  # False reject rate (~5%)
        defective_part_scores = part_max_scores[part_y_true == 1]
        part_far = float(np.mean(defective_part_scores < threshold)) * 100.0 if len(defective_part_scores) > 0 else 0.0
        part_recall = 100.0 - part_far
    else:
        threshold = 0.5
        part_frr = 0.0
        part_far = 0.0
        part_recall = 0.0

    print("\n" + "=" * 75)
    print("           SINGLE-VIEW VS MULTI-VIEW AGGREGATION")
    print("=" * 75)
    print(f"{'Inspection Strategy':<25} | {'AUROC':<10} | {'Status'}")
    print("-" * 75)
    for v in views:
        if v in single_view_metrics:
            print(f"Camera View {v:<13} | {single_view_metrics[v]['auroc']:<10.4f} | Single-view (Blind spots exist)")
    print("-" * 75)
    print(f"{'5-View Mean Aggregation':<25} | {part_mean_auroc:<10.4f} | Average across all views")
    print(f"{'5-View MAX (Recommended)':<25} | {part_max_auroc:<10.4f} | FAIL if ANY view > threshold")
    print("=" * 75)

    print("\n[+] Part-Level Decision Quality (Calibrated at 5% False Reject Rate):")
    print(f"    - Operating Threshold:       {threshold:.4f}")
    print(f"    - False Reject Rate (FRR):   {part_frr:.2f}% (Normal parts mistakenly rejected)")
    print(f"    - False Accept Rate (FAR):   {part_far:.2f}% (Defective parts mistakenly passed)")
    print(f"    - Part Defect Catch Rate:    {part_recall:.2f}%")

    # Plot comparison bar chart
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "multiview_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_names = [f"View {v}" for v in views if v in single_view_metrics] + ["5-View Mean", "5-View MAX (Part)"]
    bar_values = [single_view_metrics[v]["auroc"] * 100.0 for v in views if v in single_view_metrics] + [
        part_mean_auroc * 100.0,
        part_max_auroc * 100.0,
    ]
    colors = ["#94A3B8"] * (len(bar_names) - 2) + ["#3B82F6", "#10B981"]

    bars = ax.bar(bar_names, bar_values, color=colors, width=0.55, edgecolor="#1F2937")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Inspection AUROC (%)", fontsize=12)
    ax.set_title(f"Multi-View Advantage: 1 Camera vs 5-Camera Part Aggregation ({category.capitalize()})", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
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
    print(f"[+] Comparison figure saved to: {plot_path}")

    # Export metrics JSON
    summary = {
        "category": category,
        "total_test_images": len(records),
        "total_unique_parts": len(part_groups),
        "single_view_metrics": single_view_metrics,
        "multi_view_metrics": {
            "mean_auroc": part_mean_auroc,
            "max_auroc": part_max_auroc,
            "calibrated_threshold": threshold,
            "false_reject_rate_pct": part_frr,
            "false_accept_rate_pct": part_far,
            "part_defect_recall_pct": part_recall,
        },
    }

    out_json = out_dir / "multiview_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Multi-view metrics saved to: {out_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-view per-part aggregation.")
    parser.add_argument("--root", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024], help="Image resolution")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone")
    parser.add_argument("--coreset-ratio", type=float, default=0.001, help="Coreset ratio")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    args = parser.parse_args()

    run_multiview_evaluation(
        root=args.root,
        category=args.category,
        resolution=args.resolution,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
