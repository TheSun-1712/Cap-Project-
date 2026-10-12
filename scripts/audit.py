"""Phase 1.4 Data Audit script for Real-IAD toothbrush (and other categories).

Answers:
1. Exact count of OK and NG samples in train and in test splits.
2. Number of camera viewpoints per sample (e.g., C1-C5).
3. Defect types breakdown and image frequency for toothbrush.
4. Defect area quantification (in pixels and percentage of image area) calculated
   from ground-truth segmentation masks, binned into:
   - Micro defects (< 0.1% of image area)
   - Medium defects (0.1% to 1.0% of image area)
   - Large defects (> 1.0% of image area)
5. Saves audit metrics to `results/audit_summary.json`.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


def find_json_path(dataset_root: Path, category: str) -> Optional[Path]:
    """Find the metadata json across possible unpacking structures."""
    candidate_paths = [
        dataset_root / "realiad_jsons" / "realiad_jsons" / f"{category}.json",
        dataset_root / "realiad_jsons" / f"{category}.json",
        dataset_root / f"{category}.json",
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return None


def run_audit(
    dataset_root: str = "./datasets/Real-IAD",
    category: str = "toothbrush",
    output_json: str = "./results/audit_summary.json",
    compute_mask_areas: bool = True,
) -> Dict[str, Any]:
    dataset_root_path = Path(dataset_root)
    json_path = find_json_path(dataset_root_path, category)

    if not json_path or not json_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON for '{category}' not found under '{dataset_root}'. "
            f"Please run `python scripts/download_data.py --extract` or verify extraction."
        )

    print(f"[*] Loading metadata from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_items: List[Dict[str, Any]] = meta.get("train", [])
    test_items: List[Dict[str, Any]] = meta.get("test", [])

    print("\n" + "=" * 70)
    print(f"       REAL-IAD DATA AUDIT: {category.upper()}")
    print("=" * 70)

    # 1. OK vs NG count in train and test
    def count_classes(items: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter()
        for item in items:
            c = item.get("anomaly_class", "UNKNOWN")
            counts["OK" if c == "OK" else "NG"] += 1
        return dict(counts)

    train_class_counts = count_classes(train_items)
    test_class_counts = count_classes(test_items)

    print("\n[1] Sample & Image Distribution:")
    print(f"    Train: Total = {len(train_items)} (OK: {train_class_counts.get('OK', 0)}, NG: {train_class_counts.get('NG', 0)})")
    print(f"    Test:  Total = {len(test_items)} (OK: {test_class_counts.get('OK', 0)}, NG: {test_class_counts.get('NG', 0)})")

    # 2. Viewpoint & Sample ID Analysis
    # Real-IAD images typically follow paths like: .../sample_id/view.jpg or have explicit keys
    sample_to_views: Dict[str, set] = defaultdict(set)
    view_counter = Counter()

    for item in train_items + test_items:
        img_p = item.get("image_path", "")
        # Real-IAD path pattern: {category}/{split_or_status}/{sample_name}/C{1..5}.jpg
        parts = Path(img_p).parts
        sample_name = parts[-2] if len(parts) >= 2 else "unknown"
        filename = parts[-1] if len(parts) >= 1 else "unknown"

        # View name detection
        view_name = "unknown"
        for v in ["C1", "C2", "C3", "C4", "C5"]:
            if v in filename.upper() or v in img_p.upper():
                view_name = v
                break

        sample_to_views[sample_name].add(view_name)
        view_counter[view_name] += 1

    views_per_sample_distribution = Counter([len(views) for views in sample_to_views.values()])
    print("\n[2] Camera Viewpoints Analysis:")
    print(f"    Unique physical parts / samples: {len(sample_to_views)}")
    print(f"    Views captured per sample distribution: {dict(views_per_sample_distribution)}")
    print(f"    Image counts per viewpoint: {dict(sorted(view_counter.items()))}")

    # 3. Defect Types Breakdown
    defect_types = Counter()
    for item in test_items:
        c = item.get("anomaly_class", "UNKNOWN")
        if c != "OK":
            defect_types[c] += 1

    print("\n[3] Defect Types in Test Set:")
    for defect, count in defect_types.most_common():
        print(f"    - {defect:<20}: {count:>4} images ({count / len(test_items) * 100:.1f}%)")

    # 4. Defect Area Quantification from Segmentation Masks
    area_stats = {
        "micro (<0.1%)": 0,
        "medium (0.1%-1.0%)": 0,
        "large (>1.0%)": 0,
        "total_anomalies_evaluated": 0,
        "pixel_areas": [],
        "percentage_areas": [],
    }

    if compute_mask_areas:
        print("\n[4] Defect Size & Area Quantification (from masks):")
        mask_errors = 0

        # Find where images/masks reside
        for item in test_items:
            mask_rel = item.get("mask_path")
            if not mask_rel or item.get("anomaly_class") == "OK":
                continue

            # Look across possible resolution folders (1024 first)
            mask_full_path = None
            for res_dir in ["realiad_1024", "realiad_512", "realiad_256"]:
                candidate = dataset_root_path / res_dir / category / mask_rel
                if candidate.exists():
                    mask_full_path = candidate
                    break
                candidate2 = dataset_root_path / res_dir / mask_rel
                if candidate2.exists():
                    mask_full_path = candidate2
                    break

            if not mask_full_path or not mask_full_path.exists():
                mask_errors += 1
                continue

            # Read mask
            mask = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask_errors += 1
                continue

            total_pixels = mask.shape[0] * mask.shape[1]
            defect_pixels = int(np.count_nonzero(mask > 0))
            defect_percent = (defect_pixels / total_pixels) * 100.0

            area_stats["total_anomalies_evaluated"] += 1
            area_stats["pixel_areas"].append(defect_pixels)
            area_stats["percentage_areas"].append(defect_percent)

            if defect_percent < 0.1:
                area_stats["micro (<0.1%)"] += 1
            elif defect_percent <= 1.0:
                area_stats["medium (0.1%-1.0%)"] += 1
            else:
                area_stats["large (>1.0%)"] += 1

        if area_stats["total_anomalies_evaluated"] > 0:
            pcts = area_stats["percentage_areas"]
            pixs = area_stats["pixel_areas"]
            print(f"    Evaluated {area_stats['total_anomalies_evaluated']} defective masks:")
            print(f"    - Micro defects (<0.1% area):   {area_stats['micro (<0.1%)']:>4} ({area_stats['micro (<0.1%)'] / len(pcts) * 100:.1f}%)")
            print(f"    - Medium defects (0.1%-1.0%):   {area_stats['medium (0.1%-1.0%)']:>4} ({area_stats['medium (0.1%-1.0%)'] / len(pcts) * 100:.1f}%)")
            print(f"    - Large defects (>1.0% area):   {area_stats['large (>1.0%)']:>4} ({area_stats['large (>1.0%)'] / len(pcts) * 100:.1f}%)")
            print(f"    - Median defect area: {np.median(pcts):.3f}% ({int(np.median(pixs))} px)")
            print(f"    - Min defect area:    {np.min(pcts):.4f}% ({int(np.min(pixs))} px)")
            print(f"    - Max defect area:    {np.max(pcts):.2f}% ({int(np.max(pixs))} px)")
        else:
            if mask_errors > 0:
                print(f"    [!] Masks not found locally on disk ({mask_errors} missing). Ensure images are extracted.")
            else:
                print("    [!] No mask paths found in metadata.")

    # Save summary to results
    summary = {
        "category": category,
        "json_path": str(json_path),
        "train_samples": len(train_items),
        "train_class_counts": train_class_counts,
        "test_samples": len(test_items),
        "test_class_counts": test_class_counts,
        "unique_parts": len(sample_to_views),
        "views_per_sample": {str(k): v for k, v in views_per_sample_distribution.items()},
        "viewpoint_counts": dict(view_counter),
        "defect_types": dict(defect_types),
        "area_distribution": {
            "micro (<0.1%)": area_stats["micro (<0.1%)"],
            "medium (0.1%-1.0%)": area_stats["medium (0.1%-1.0%)"],
            "large (>1.0%)": area_stats["large (>1.0%)"],
            "total_masks_analyzed": area_stats["total_anomalies_evaluated"],
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Audit report saved to: {out_p}")
    print("=" * 70)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit Real-IAD category metadata.")
    parser.add_argument("--root", type=str, default="./datasets/Real-IAD", help="Dataset root directory")
    parser.add_argument("--category", type=str, default="toothbrush", help="Category name")
    parser.add_argument("--output", type=str, default="./results/audit_summary.json", help="Summary output JSON")
    args = parser.parse_args()

    run_audit(dataset_root=args.root, category=args.category, output_json=args.output)


if __name__ == "__main__":
    main()
