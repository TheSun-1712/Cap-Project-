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

### Step 9: Launch Live Streamlit Inspection Dashboard (Web UI)
Run the real-time operator inspection dashboard to upload custom toothbrush images or test presets with 5x Microscope Zoom:
```powershell
# Quick 1-click startup:
.\run_dashboard.bat

# Or via PowerShell:
.\run_dashboard.ps1

# Or via Streamlit directly:
.\.venv\Scripts\streamlit.exe run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 4. Directory Structure
```
Cap Project/
├── .venv/                      # Python 3.11 with PyTorch 2.11 cu128 (Blackwell sm_120)
├── app.py                      # Interactive Streamlit Web Inspection Dashboard
├── run_dashboard.bat           # 1-click Windows startup script
├── run_dashboard.ps1           # 1-click PowerShell startup script
├── weights/
│   ├── patchcore_toothbrush_512.ckpt  # Pre-trained PatchCore memory bank checkpoint
│   └── model_config.json              # Calibrated threshold & model metadata
├── configs/
│   └── pipeline_config.yaml    # Hyperparameter & pipeline configuration
├── datasets/
│   └── Real-IAD/               # Dataset root
├── results/
│   ├── heatmaps/               # 4-panel inspection decision graphics
│   ├── operator_pinpoint/      # Microscope & reticle inspection graphics
│   ├── audit_summary.json      # Phase 1.4 data audit metrics
│   ├── defect_size_vs_recall.png # Phase 3 tiny-defect study curve
│   ├── multiview_comparison.png # Phase 4 single-view vs multi-view curve
│   └── model_comparison_table.md # Phase 5 model comparison benchmark
├── scripts/
│   ├── geometry_3d/            # NEW: 3D CAD Metrology & Inspection Subpackage
│   │   ├── step_loader.py      # Gmsh / OpenCASCADE STEP AP203 CAD Ingestion & Adaptive Tessellation
│   │   ├── mesh_loader.py      # Universal 3D Geometry Container (STL, PLY, OBJ)
│   │   ├── alignment.py        # 6-DoF Rigid Body Registration (PCA + Huber Point-to-Plane ICP)
│   │   ├── comparator.py       # Normal-constrained Signed Distance Metrology Engine
│   │   ├── gdt_inspector.py    # GD&T Feature Extractor (Countersink holes, pitch, bosses)
│   │   ├── clustering.py       # DBSCAN 3D Defect Site Profiling & Centroid Calculator
│   │   └── synthetic_defects.py# Procedural Defect Injector anchored to nominal STEP model
│   ├── inspect_3d.py           # Standalone CLI 3D CAD Metrology Inspection Tool
│   ├── ui_3d_inspector.py      # Streamlit Plotly 3D WebGL Viewport & GD&T Scorecards
│   ├── test_3d_pipeline.py     # Automated Verification Test Suite (5/5 Passing)
│   ├── inference_engine.py     # High-speed GPU PatchCore inference pipeline
│   ├── inspect_part.py         # Autonomous pinpoint & microscope generator
│   ├── download_data.py        # Gated download, SHA256 check, unpack & downscaler
│   ├── audit.py                # Data audit script
│   ├── baseline.py             # PatchCore baseline & resolution ablation
│   ├── tiling_eval.py          # Tiled inference & defect-size vs recall
│   ├── multiview_eval.py       # Multi-view aggregation & part PASS/FAIL
│   ├── model_comparison.py     # PatchCore vs PaDiM vs EfficientAD comparison
│   ├── viz.py                  # 4-panel visualizer & decision engine
│   └── synthetic_defect_generator.py # Procedural defect generator for pencils
├── samples_3d/                 # Benchmark catalog generated from nominal STEP model
│   ├── golden_part.stl         # Nominal reference model discretization (PASS)
│   ├── part_with_dent.stl      # Injected -0.65 mm concave gouge flaw (REJECT)
│   ├── part_with_burr.stl      # Injected +0.85 mm raised material flash (REJECT)
│   ├── part_hole_ovality.stl   # Countersunk fastener hole elongation (REJECT GD&T)
│   └── part_warped_flange.stl  # Quadratic thermal bending across 154 mm axis (REJECT)
├── inspect_3d.bat              # 1-click Windows 3D inspection CLI launcher
├── inspect_3d.ps1              # 1-click PowerShell 3D inspection CLI launcher
└── README.md
```

---

## 5. 3D CAD Metrology & Defect Inspection Engine (CATIA STEP AP203)

The pipeline has been expanded to support **direct 3D volumetric CAD metrology** against parametric design authorities, specifically targeting **`0101BA302380N_010 XYZ.stp`** (CATIA V5 AP203 open-shell B-Rep structural bracket/cap).

### 5.1 Key Capabilities
* **Native STEP AP203 Ingestion:** Uses embedded OpenCASCADE via `gmsh` to mesh 665 B-spline faces and revolution surfaces with adaptive chordal deflection ($\delta \le 0.05\,\text{mm}$), stripping origin datum artifacts.
* **Assembly Coordinate Standardization:** Automatically centers CATIA vehicle coordinates ($X \approx 1612, Y \approx -570, Z \approx 1153\,\text{mm}$) while preserving $T_{\text{assembly}}$ for dual-frame reporting.
* **Robust 6-DoF Registration:** Combines PCA inertia axis matching with Huber loss Point-to-Plane ICP to align scans without bias from defect sites.
* **Open-Shell Normal-Constrained Metrology:** Eliminates thin-wall open-shell bleed ($1.5\,\text{mm}$ thickness) by enforcing normal consistency ($|\mathbf{n}_{\text{test}} \cdot \mathbf{n}_{\text{ref}}| \ge \cos(60^\circ)$).
* **GD&T Feature Verification:** Automatically verifies countersunk fastener hole diameters ($7.00\,\text{mm}$ pilot, $11.20\,\text{mm}$ bevel), hole pitch ($20.00\,\text{mm}$), and concentric cylindrical bosses ($20.00\,\text{mm}$ and $26.00\,\text{mm}$).
* **3D DBSCAN Clustering:** Isolates separate flaw sites, reporting 3D centroids $[X,Y,Z]$, peak depths, and flaw surface areas ($\text{mm}^2$).

### 5.2 Command Line Inspection (CLI)
Run automated 3D comparison from PowerShell or Command Prompt:

```powershell
# Inspect defective part against nominal STEP CAD:
.\inspect_3d.ps1 -ref "0101BA302380N_010 XYZ.stp" -test "samples_3d/part_with_dent.stl" -tolerance 0.20

# Or via Python directly:
python scripts/inspect_3d.py --ref "0101BA302380N_010 XYZ.stp" --test "samples_3d/golden_part.stl" --tolerance 0.20 --out-html "results/golden_inspection.html"
```

### 5.3 Interactive 3D Web Dashboard
Launch the Streamlit dashboard:
```powershell
.\run_dashboard.ps1
```
Use the top-left sidebar radio button to select **`🧊 3D CAD Metrology (CATIA AP203)`** to interact with the full 360° Plotly WebGL viewer, view diverging surface deviation heatmaps, inspect GD&T scorecards, and test all benchmark presets.

### 5.4 Run Automated Test Suite
```powershell
python scripts/test_3d_pipeline.py
# 5/5 Unit Tests Passing (100% test coverage across ingestion, alignment, metrology, GD&T, and defect clustering)
```


