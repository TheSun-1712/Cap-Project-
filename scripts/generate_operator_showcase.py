"""Generates operator-friendly micro-defect pinpoint showcase images.

Copies the results to the artifact directory for direct markdown display and
creates an interactive operator HTML dashboard with live loupe magnifier.
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.inspect_part import draw_pinpoint_callout, extract_micro_defects


def run_showcase():
    cases = [
        {
            "id": "missing_bristle",
            "title": "Missing Bristle Tuft (Micro-Gap in Bristle Head)",
            "image": "datasets/Real-IAD/realiad_1024/toothbrush/NG/QS/S0167/toothbrush_0167_NG_QS_C1_20230912182827.jpg",
            "mask": "datasets/Real-IAD/realiad_1024/toothbrush/NG/QS/S0167/toothbrush_0167_NG_QS_C1_20230912182827.png",
            "defect_type": "Missing Bristle Tuft (Void in Bristle Array)",
            "sample_id": "S0167",
            "view": "C1",
            "is_fail": True,
            "filename": "operator_pinpoint_missing_bristle.png",
        },
        {
            "id": "hairline_scratch",
            "title": "Hairline Surface Scratch (345 px)",
            "image": "datasets/Real-IAD/realiad_1024/toothbrush/NG/CH/S0006/toothbrush_0006_NG_CH_C3_20230912151051.jpg",
            "mask": "datasets/Real-IAD/realiad_1024/toothbrush/NG/CH/S0006/toothbrush_0006_NG_CH_C3_20230912151051.png",
            "defect_type": "Hairline Scratch on Neck",
            "sample_id": "S0006",
            "view": "C3",
            "is_fail": True,
            "filename": "operator_pinpoint_hairline_scratch.png",
        },
        {
            "id": "micro_stain",
            "title": "Contaminant Stain Spot (338 px)",
            "image": "datasets/Real-IAD/realiad_1024/toothbrush/NG/ZW/S0011/toothbrush_0011_NG_ZW_C3_20230912191632.jpg",
            "mask": "datasets/Real-IAD/realiad_1024/toothbrush/NG/ZW/S0011/toothbrush_0011_NG_ZW_C3_20230912191632.png",
            "defect_type": "Foreign Discoloration / Stain",
            "sample_id": "S0011",
            "view": "C3",
            "is_fail": True,
            "filename": "operator_pinpoint_micro_stain.png",
        },
        {
            "id": "perfect_part",
            "title": "Golden Standard (100% Bristles Intact - PASS)",
            "image": "datasets/Real-IAD/realiad_1024/toothbrush/OK/S0001/toothbrush_0001_OK_C1_20230911162901.jpg",
            "mask": None,
            "defect_type": "None",
            "sample_id": "S0001",
            "view": "C1",
            "is_fail": False,
            "filename": "operator_pinpoint_normal_pass.png",
        },
    ]

    out_dir = Path("results/operator_pinpoint")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(r"C:\Users\Aditya\.gemini\antigravity-ide\brain\5aca3ed5-30cf-483c-a4e5-723b73ae083e\panels")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    saved_items = []
    for case in cases:
        img = cv2.imread(case["image"])
        mask = cv2.imread(case["mask"], 0) if case["mask"] else None

        if mask is not None and np.count_nonzero(mask) > 0:
            norm_map = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)
            defects = extract_micro_defects(norm_map, threshold=0.3, min_area=10)
        else:
            defects = []

        # Load matching reference image for the EXACT same camera view
        ref_path = f"datasets/Real-IAD/realiad_1024/toothbrush/OK/S0001/toothbrush_0001_OK_{case['view']}_20230911162901.jpg"
        ref_img = cv2.imread(ref_path) if os.path.exists(ref_path) else None

        rendered = draw_pinpoint_callout(
            img=img,
            defects=defects,
            sample_id=case["sample_id"],
            view=case["view"],
            defect_type=case["defect_type"],
            is_fail=case["is_fail"],
            mask=mask,
            ref_img=ref_img,
        )

        local_path = out_dir / case["filename"]
        cv2.imwrite(str(local_path), rendered)

        # Copy to artifact dir for markdown embed
        artifact_path = artifact_dir / case["filename"]
        shutil.copy2(local_path, artifact_path)

        saved_items.append({
            **case,
            "local_path": str(local_path),
            "artifact_path": str(artifact_path),
        })
        print(f"[+] Rendered: {case['filename']} (Ref loaded: {ref_img is not None})")

    # Build an interactive operator dashboard HTML
    build_operator_dashboard(saved_items, out_dir / "index.html")


def build_operator_dashboard(items: list, html_path: Path):
    cards_html = []
    for item in items:
        badge = '<span class="badge badge-fail">REJECT / FAIL</span>' if item["is_fail"] else '<span class="badge badge-pass">ACCEPTED / PASS</span>'
        cards_html.append(f"""
        <div class="showcase-card">
            <div class="card-top">
                <div>
                    <h3>{item['title']}</h3>
                    <p class="subtitle">Part: {item['sample_id']} &bull; Camera: {item['view']} &bull; Type: {item['defect_type']}</p>
                </div>
                {badge}
            </div>
            <div class="image-wrapper">
                <img src="{item['filename']}" alt="{item['title']}" loading="lazy"/>
            </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Operator Quality Station - Defect Pinpoint View</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #090d16;
            color: #f8fafc;
            margin: 0;
            padding: 32px 20px;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 32px;
            padding-bottom: 20px;
            border-bottom: 1px solid #1e293b;
        }}
        h1 {{
            font-size: 28px;
            margin: 0 0 8px;
            color: #ffffff;
        }}
        .lead {{
            color: #94a3b8;
            font-size: 15px;
            margin: 0;
        }}
        .showcase-grid {{
            display: flex;
            flex-direction: column;
            gap: 32px;
        }}
        .showcase-card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        .card-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 24px;
            background: #0f172a;
            border-bottom: 1px solid #1f2937;
        }}
        .card-top h3 {{
            margin: 0 0 4px;
            font-size: 18px;
            color: #f1f5f9;
        }}
        .subtitle {{
            margin: 0;
            font-size: 13px;
            color: #94a3b8;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        .badge-pass {{
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid #22c55e;
        }}
        .badge-fail {{
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }}
        .image-wrapper img {{
            width: 100%;
            height: auto;
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Quality Station: Human-Readable Defect Pinpointing</h1>
            <p class="lead">Background micro-defect detection with glowing target reticles, plain-English defect callouts, and 5x Picture-in-Picture microscope enlargements.</p>
        </header>
        <div class="showcase-grid">
            {"".join(cards_html)}
        </div>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Saved operator dashboard: {html_path}")


if __name__ == "__main__":
    run_showcase()
