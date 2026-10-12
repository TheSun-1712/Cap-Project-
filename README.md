# Automated Visual Inspection: Real-IAD Toothbrush to Pencils Pipeline

An end-to-end multi-view industrial defect inspection framework built on Intel's **Anomalib 2.7.0** and **PyTorch 2.11.0 (CUDA 12.8)**, tailored for NVIDIA Blackwell GPUs (RTX 5060, `sm_120`) and high-resolution manufacturing inspection.

---

## 1. System Architecture & Problem Statement

### Problem
Manufacturing inspection requires autonomous, high-speed **PASS/FAIL** decisions, accurate defect localization, multi-view surface coverage (eliminating camera blind spots), and high sensitivity to micro-defects (scratches, dents, chips).

### Solution Architecture
```
                  ┌────────────────────────────────────────┐
                  │ 5 Camera Viewpoints (C1..C5) per Part  │
                  └──────────────────┬─────────────────────┘
                                     │
                             Pre-Processing
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
   [Anomaly Branch]                                   [Supervised Branch]
  One-Class PatchCore / PaDiM / EfficientAD            Defect Classifier / Detector
  Learns normal variation only                        Names known defect types
  Catches 100% of unseen defects
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                          [Multi-View Decision Engine]
                     Part Anomaly Score = max(C1, ..., C5)
                    FAIL if any view > Calibrated Threshold
                                     │
                                     ▼
                Decision Overlay + Defect Peak Coordinates + Heatmap
```

---

## 2. Hardware & Environment Verification

This repository is optimized for your machine specs:
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell Architecture, Compute Capability `sm_120`, ~8.5 GB VRAM)
- **PyTorch**: `2.11.0+cu128` with CUDA 12.8 and native `sm_120` support
- **CPU**: AMD Ryzen 9 8940HX (16 Cores / 32 Threads)
- **RAM**: 32 GB RAM

### Environment Activation
```powershell
.\.venv\Scripts\activate
```

Verify GPU & Architecture:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
# Output includes: 'sm_120' and 'NVIDIA GeForce RTX 5060 Laptop GPU'
```

---

## 3. Implementation Workflow & Scripts

### Step 1: Download & Extract Dataset (Phase 1.2 & 1.3)
Download the gated Real-IAD toothbrush files (1.46 GB) from Hugging Face Hub, verify SHA256 checksum, extract, and auto-generate 512px and 256px downscaled versions:
```powershell
python scripts/download_data.py --download --extract --generate-res --token <YOUR_HF_TOKEN>
```
*(If you already downloaded the zip files manually, place them in `datasets/Real-IAD/` and run `python scripts/download_data.py --extract --generate-res`)*

### Step 2: Run Data Audit (Phase 1.4)
Quantify sample counts, camera viewpoints, defect types, and mask defect areas:
```powershell
python scripts/audit.py --category toothbrush
```
Output:
- Counts of OK and NG samples in Train and Test splits.
- 5 viewpoints per sample validation.
- Defect area breakdown: Micro (<0.1%), Medium (0.1%–1.0%), Large (>1.0%).
- Saved summary to `results/audit_summary.json`.

### Step 3: Run Baseline PatchCore (Phase 2 & Phase 3 Resolution Ablation)
Train PatchCore memory bank and evaluate metrics (AUROC, F1, VRAM, and runtime):
```powershell
# Fast baseline at 256x256
python scripts/baseline.py --resolution 256 --backbone wide_resnet50_2

# Baseline at 512x512
python scripts/baseline.py --resolution 512 --backbone wide_resnet50_2

# High-resolution at 1024x1024 (uses ResNet-18 or Coreset 0.05 to preserve VRAM)
python scripts/baseline.py --resolution 1024 --backbone resnet18 --coreset-ratio 0.05
```

### Step 4: Tiny-Defect Study with Tiling (Phase 3)
Run 512x512 sliding-window tiled inference across 1024x1024 images, calibrate decision threshold at 5% False Positive Rate, and plot Defect Size vs. Recall curve:
```powershell
python scripts/tiling_eval.py --resolution 1024 --tile-size 512 --stride 384 --target-fpr 0.05
```
Output:
- Generates `results/defect_size_vs_recall.png`
- Exports statement: *"At setting X, we detect defects larger than N pixels with Y% recall."*

### Step 5: Multi-View Decision per Part (Phase 4)
Evaluate the 5 camera viewpoints (C1 to C5) and demonstrate why multi-view eliminates single-camera blind spots:
```powershell
python scripts/multiview_eval.py --resolution 512
```
Output:
- Single-view AUROC (C1 through C5) vs. Part AUROC (`max(C1..C5)`).
- False Reject Rate (FRR) and False Accept Rate (FAR) trade-offs.
- Comparison plot saved to `results/multiview_comparison.png`.

### Step 6: Model Comparison Benchmark (Phase 5)
Benchmark PatchCore vs. PaDiM vs. EfficientAD on the exact same category and resolution:
```powershell
python scripts/model_comparison.py --resolution 512 --models patchcore padim efficientad
```
Output:
- Saves comparison table to `results/model_comparison_table.md` and `results/model_comparison.json`.

### Step 7: 4-Panel Visualization Suite (Phase 6)
Generate the 4-panel inspection graphics (Raw Image, Ground Truth Mask, Heatmap, and Decision Overlay with Bounding Box & Peak Location):
```powershell
python scripts/viz.py --resolution 512 --samples 15
```
Saved to: `results/heatmaps/*.png`.

### Step 8: Move to Pencils & Synthetic Defect Generation (Phase 7)
Generate synthetic scratches, dents, chips, and stains on normal pencil reference images:
```powershell
python scripts/synthetic_defect_generator.py --input-dir datasets/pencils/normal --output-dir datasets/pencils/synthetic
```

---

## 4. Directory Structure
```
Cap Project/
├── .venv/                      # Python 3.11 with PyTorch 2.11 cu128 (Blackwell sm_120)
├── configs/
│   └── pipeline_config.yaml    # Hyperparameter & pipeline configuration
├── datasets/
│   └── Real-IAD/               # Dataset root
├── results/
│   ├── heatmaps/               # 4-panel inspection decision graphics
│   ├── audit_summary.json      # Phase 1.4 data audit metrics
│   ├── defect_size_vs_recall.png # Phase 3 tiny-defect study curve
│   ├── multiview_comparison.png # Phase 4 single-view vs multi-view curve
│   └── model_comparison_table.md # Phase 5 model comparison benchmark
├── scripts/
│   ├── download_data.py        # Gated download, SHA256 check, unpack & downscaler
│   ├── audit.py                # Data audit script
│   ├── baseline.py             # PatchCore baseline & resolution ablation
│   ├── tiling_eval.py          # Tiled inference & defect-size vs recall
│   ├── multiview_eval.py       # Multi-view aggregation & part PASS/FAIL
│   ├── model_comparison.py     # PatchCore vs PaDiM vs EfficientAD comparison
│   ├── viz.py                  # 4-panel visualizer & decision engine
│   └── synthetic_defect_generator.py # Procedural defect generator for pencils
└── README.md
```
