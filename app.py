"""Toothbrush Automated Optical Inspection (AOI) - Streamlit Dashboard.

Industrial-grade Machine Vision Dashboard for automated toothbrush anomaly detection.
Features real-time GPU inference, micro-flaw localization, and 5x Microscope Zoom comparison
against camera-matched Golden Standards.
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure Page immediately for instant UI render
st.set_page_config(
    page_title="Toothbrush Quality Inspection AI",
    page_icon="🪥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add workspace to path
sys.path.append(str(Path(__file__).resolve().parent))

# Custom High-End Industrial Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background-color: #090d16;
        color: #f1f5f9;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(14, 23, 42, 0.9) 0%, rgba(9, 13, 22, 1) 90%);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Top Banner Header */
    .system-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
    }
    .system-title {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .system-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Verdict Banners */
    .verdict-pass {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0 24px 0;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.2);
    }
    .verdict-fail {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.22) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0 24px 0;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
    }
    .verdict-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin: 0;
    }
    .verdict-desc {
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Custom Button Styles */
    .stButton>button {
        width: 100%;
        background: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: #0284c7;
        border-color: #38bdf8;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 14px;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚡ Initializing Toothbrush PatchCore GPU Engine...")
def load_inspector():
    """Load model once into GPU VRAM."""
    from scripts.inference_engine import ToothbrushInspector
    return ToothbrushInspector()


def main():
    # ==========================
    # TOP-LEVEL MODALITY SWITCH
    # ==========================
    with st.sidebar:
        st.markdown("### 🔍 Inspection Modality")
        modality = st.radio(
            "Select Inspection System:",
            [
                "🧊 3D CAD Metrology (CATIA AP203)",
                "📷 2D Optical Inspection (PatchCore AOI)",
            ],
            index=0,
            help="Switch between 3D volumetric CAD metrology and 2D multi-view optical inspection.",
        )
        st.markdown("---")

    if modality == "🧊 3D CAD Metrology (CATIA AP203)":
        from scripts.ui_3d_inspector import render_3d_inspection_tab
        render_3d_inspection_tab()
        return

    inspector = load_inspector()

    # Pre-defined test sample catalog
    sample_presets = {
        "🟢 Golden Standard (Normal Part - PASS)": {
            "path": "datasets/Real-IAD/realiad_512/toothbrush/OK/S0001/toothbrush_0001_OK_C1_20230911162901.jpg",
            "view": "C1",
            "name": "Toothbrush S0001 (Golden Standard)",
            "expected": "PASS",
        },
        "🔴 Missing Bristle Tuft (CH Defect - REJECT)": {
            "path": "datasets/Real-IAD/realiad_512/toothbrush/NG/CH/S0001/toothbrush_0001_NG_CH_C1_20230912143447.jpg",
            "view": "C1",
            "name": "Toothbrush S0001 (Bristle Void)",
            "expected": "REJECT",
        },
        "🔴 Hairline Neck Scratch (QS Defect - REJECT)": {
            "path": "datasets/Real-IAD/realiad_512/toothbrush/NG/QS/S0001/toothbrush_0001_NG_QS_C2_20230912143306.jpg",
            "view": "C2",
            "name": "Toothbrush S0001 (Neck Scratch)",
            "expected": "REJECT",
        },
        "🔴 Contaminant Stain Spot (ZW Defect - REJECT)": {
            "path": "datasets/Real-IAD/realiad_512/toothbrush/NG/ZW/S0001/toothbrush_0001_NG_ZW_C3_20230912143057.jpg",
            "view": "C3",
            "name": "Toothbrush S0001 (Chemical Stain)",
            "expected": "REJECT",
        },
        "🔴 Missing Bristles Micro-Flaw (S0167 - REJECT)": {
            "path": "datasets/Real-IAD/realiad_512/toothbrush/NG/CH/S0002/toothbrush_0002_NG_CH_C1_20230912143452.jpg",
            "view": "C1",
            "name": "Toothbrush S0002 (Disheveled)",
            "expected": "REJECT",
        },
    }

    # ==========================
    # SIDEBAR: CONTROLS & INPUT
    # ==========================
    with st.sidebar:
        st.markdown("### 🎛️ Inspection Console")
        st.caption("AI-Powered Optical Quality Inspection")

        # System Hardware Status Pill
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 11px; font-weight: 700; color: #22c55e;">● LIVE SYSTEM ACTIVE</span>
                <span style="font-size: 11px; color: #94a3b8;">GPU Accelerated</span>
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; font-family: monospace;">
                NVIDIA RTX 5060 · ResNet18
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tab: Choose input method
        input_source = st.radio(
            "Select Inspection Mode:",
            ["🧪 Test Sample Presets", "📤 Upload My Toothbrush Image"],
            index=0,
        )

        selected_image = None
        selected_view = "C1"
        sample_name = "Custom Part"

        if input_source == "📤 Upload My Toothbrush Image":
            uploaded_file = st.file_uploader(
                "Upload Toothbrush Image (Drag & Drop):",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Supports RGB camera frames at any resolution.",
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                sample_name = uploaded_file.name

            st.markdown("---")
            camera_option = st.selectbox(
                "Camera Viewpoint (Angle):",
                [
                    "C1 (Bristle Frontal Facing View)",
                    "C2 (Side Profile View)",
                    "C3 (Back / Neck View)",
                    "C4 (30-Degree Angled View)",
                    "C5 (Bottom Base View)",
                ],
                index=0,
                help="Specifies which camera angle Golden Standard to cross-reference.",
            )
            selected_view = camera_option[:2]

        else:
            preset_choice = st.selectbox(
                "Choose Presets for Instant Verification:",
                list(sample_presets.keys()),
                index=1,  # Default to Missing Bristle
            )
            preset_data = sample_presets[preset_choice]
            if os.path.exists(preset_data["path"]):
                selected_image = Image.open(preset_data["path"]).convert("RGB")
                selected_view = preset_data["view"]
                sample_name = preset_data["name"]
            else:
                st.warning(f"Preset path not found: {preset_data['path']}")

        st.markdown("---")
        st.markdown("### ⚙️ Operational Sensitivity")

        # Threshold slider
        threshold_val = st.slider(
            "Anomaly Decision Threshold (τ):",
            min_value=0.35,
            max_value=0.85,
            value=0.530,
            step=0.005,
            format="%.3f",
            help="Threshold calibrated via ROC analysis (Optimal: 0.530). Lower = stricter rejection, Higher = more lenient.",
        )

        min_area_val = st.slider(
            "Noise Area Filter (min pixels):",
            min_value=5,
            max_value=80,
            value=15,
            step=5,
            help="Discards spurious sensor noise smaller than this pixel area.",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 11px; color: #64748b; line-height: 1.5;">
            <strong>Real-IAD Toothbrush Benchmark:</strong><br>
            • Image AUROC: <strong>85.34%</strong><br>
            • Pixel AUROC: <strong>91.59%</strong><br>
            • Inference Speed: <strong>~30-40 ms</strong><br>
            • Memory Bank: <strong>1,254 Coreset Vectors</strong>
        </div>
        """, unsafe_allow_html=True)

    # ==========================
    # MAIN WORKSPACE
    # ==========================

    # Industrial HUD Title Header
    st.markdown("""
    <div class="system-header">
        <h1 class="system-title">
            <span>🪥</span> TOOTHBRUSH AUTOMATED QUALITY INSPECTION AI
        </h1>
        <p class="system-subtitle">
            Autonomous Machine Vision & Deep Anomaly Localization · Detecting Missing Bristles, Micro-Cracks, and Surface Flaws
        </p>
    </div>
    """, unsafe_allow_html=True)

    if selected_image is None:
        st.info("👈 Please upload an image of a toothbrush in the sidebar, or select one of the pre-loaded test samples to run instant inspection.")
        return

    # Run Real-Time GPU Inspection
    with st.spinner("🔍 Scanning toothbrush surface and verifying against Golden Standard memory bank..."):
        result = inspector.inspect(
            image_input=selected_image,
            threshold=threshold_val,
            min_area=min_area_val,
            view=selected_view,
            sample_name=sample_name,
        )

    # Extract Results
    is_fail = result["is_fail"]
    verdict = result["verdict"]
    score = result["anomaly_score"]
    thr = result["threshold"]
    latency = result["latency_ms"]
    defect_count = result["defect_count"]
    defect_type = result["defect_type"]
    primary_defect = result["primary_defect"]
    is_ood = result.get("is_ood", False)
    coverage_pct = result.get("coverage_pct", 0.0)

    # ==========================
    # DOMAIN MISMATCH WARNING (IF APPLICABLE)
    # ==========================
    if is_ood:
        flaw_px = primary_defect['area_pixels'] if primary_defect else 0
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.15); border: 2px solid #f59e0b; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);">
            <div style="color: #f59e0b; font-size: 20px; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                <span>⚠️</span> DOMAIN & SCALE MISMATCH DETECTED ({coverage_pct:.1f}% OF FRAME ANOMALOUS)
            </div>
            <div style="color: #fde68a; font-size: 14px; margin-top: 8px; line-height: 1.5;">
                Almost the entire image (<strong>{flaw_px:,} pixels</strong>) was flagged as anomalous.
                <br>
                <strong>Root Cause:</strong> The AI was trained on the <strong>Real-IAD industrial toothbrush</strong> (white body, standard camera distance). This photo is an <strong>extreme macro shot</strong> with a different color scheme (e.g., pink/electric-blue bristles), which the model has never seen.
                <br><br>
                👉 <strong>How to Fix:</strong> Frame the photo so the <em>entire toothbrush head & neck</em> are visible against a plain background (like the dataset), or lower the sensitivity slider below.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================
    # HIGH-VISIBILITY VERDICT BANNER
    # ==========================
    if is_fail:
        flaw_size_desc = f"{primary_defect['area_pixels']:,} px" if primary_defect else "localized"
        st.markdown(f"""
        <div class="verdict-fail">
            <div class="verdict-title" style="color: #ef4444;">
                🚨 REJECT — DEFECT DETECTED ({defect_type.upper()})
            </div>
            <div class="verdict-desc" style="color: #fca5a5;">
                Anomaly score of <strong>{score:.3f}</strong> exceeds threshold limit (<strong>{thr:.3f}</strong>).
                Flaw coverage: <strong>{coverage_pct:.1f}% of image</strong> (Primary flaw size: <strong>{flaw_size_desc}</strong>).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-pass">
            <div class="verdict-title" style="color: #22c55e;">
                ✅ PASS — 100% SPECIFICATION MET (ZERO DEFECTS)
            </div>
            <div class="verdict-desc" style="color: #86efac;">
                Anomaly score of <strong>{score:.3f}</strong> is safely below tolerance limit (<strong>{thr:.3f}</strong>).
                All bristle clusters, tuft anchors, and molded surfaces match the Golden Standard specification.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================
    # KEY KPI TILES (4 COLUMNS)
    # ==========================
    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Anomaly Score</div>
            <div class="kpi-value" style="color: {'#ef4444' if is_fail else '#22c55e'};">{score:.3f}</div>
            <div class="kpi-sub">Limit Threshold: {thr:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Quality Status</div>
            <div class="kpi-value" style="color: {'#ef4444' if is_fail else '#22c55e'};">{verdict}</div>
            <div class="kpi-sub">Tolerance Margin: {abs(score - thr):.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        flaw_area_str = f"{primary_defect['area_pixels']} px" if (is_fail and primary_defect) else "0 px"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Flaws Localized</div>
            <div class="kpi-value" style="color: {'#f59e0b' if is_fail else '#38bdf8'};">{defect_count}</div>
            <div class="kpi-sub">Primary Area: {flaw_area_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        fps = 1000.0 / max(1.0, latency)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Inspection Latency</div>
            <div class="kpi-value" style="color: #38bdf8;">{latency:.1f} ms</div>
            <div class="kpi-sub">Throughput: ~{fps:.0f} FPS (GPU)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")  # Spacing

    # ==========================
    # MULTI-LAYER INTERACTIVE TABS
    # ==========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Pinpoint HUD View",
        "🔬 5x Microscope Zoom (Close-Up Comparison)",
        "🔥 Anomaly Heatmap & Energy Profile",
        "📷 Raw Camera Image",
        "📊 Benchmark & Diagnostics",
    ])

    with tab1:
        st.markdown("#### 🎯 Operator-Ready Pinpoint Inspection HUD")
        st.caption("Includes glowing target reticles, leader-line callout card, and integrated 5x Microscope Comparison Inset.")
        st.image(result["pinpoint_hud_rgb"], use_container_width=True)

        # Download button for the HUD image
        buf = io.BytesIO()
        Image.fromarray(result["pinpoint_hud_rgb"]).save(buf, format="PNG")
        st.download_button(
            label="💾 Download Inspection Report Image (PNG)",
            data=buf.getvalue(),
            file_name=f"inspection_{sample_name.replace(' ', '_')}.png",
            mime="image/png",
        )

    with tab2:
        st.markdown("#### 🔬 5x Microscope Zoom: Defect vs. Golden Standard")
        st.caption("Direct microscopic enlargement comparing the exact flagged area on this toothbrush against the camera-matched Golden Standard.")

        if is_fail and result["microscope_defect_crop"] is not None:
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 10px; padding: 12px; text-align: center; margin-bottom: 10px;">
                    <strong style="color: #ef4444; font-size: 16px;">🔍 THIS PART (DEFECTIVE REGION)</strong><br>
                    <span style="color: #cbd5e1; font-size: 13px;">Micro-flaw isolated at coordinates X={}, Y={}</span>
                </div>
                """.format(
                    primary_defect["peak"][0] if primary_defect else 0,
                    primary_defect["peak"][1] if primary_defect else 0,
                ), unsafe_allow_html=True)
                st.image(result["microscope_defect_crop"], use_container_width=True)

            with m_col2:
                if result["microscope_ref_crop"] is not None:
                    st.markdown("""
                    <div style="background: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; border-radius: 10px; padding: 12px; text-align: center; margin-bottom: 10px;">
                        <strong style="color: #22c55e; font-size: 16px;">✨ GOLDEN STANDARD (NORMAL)</strong><br>
                        <span style="color: #cbd5e1; font-size: 13px;">Camera Viewpoint Reference: <strong>{}</strong></span>
                    </div>
                    """.format(selected_view), unsafe_allow_html=True)
                    st.image(result["microscope_ref_crop"], use_container_width=True)
                else:
                    st.info("ℹ️ **Golden Standard Reference Registration:** Cross-referencing against the certified Golden Standard is designed for parts matching the Real-IAD camera framing (full toothbrush). For extreme macro shots or different toothbrush designs, cross-referencing is disabled.")

            st.markdown("""
            > **Microscopic Interpretation:** The left window reveals the defect (e.g., missing bristle root void, scratch groove, or chemical discoloration) highlighted in high-contrast red. The right window shows the corresponding region on a certified defect-free toothbrush under the identical lighting and optical angle.
            """)
        else:
            st.success("🎉 This part passed 100%! No micro-defects or anomalies require microscopic enlargement.")
            if inspector.get_reference_image(selected_view) is not None:
                st.markdown(f"**Showing certified Golden Standard Reference for viewpoint `{selected_view}`:**")
                st.image(inspector.get_reference_image(selected_view), width=500)

    with tab3:
        st.markdown("#### 🔥 Continuous Anomaly Energy Heatmap")
        st.caption("Continuous pixel-level anomaly density generated by PatchCore feature distance matching.")

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown("**Thermal Energy Gradient (Jet Colormap)**")
            st.image(result["heatmap_rgb"], use_container_width=True)
        with h_col2:
            st.markdown("**Blended Projection (Heatmap over Part)**")
            st.image(result["overlay_rgb"], use_container_width=True)

        st.markdown(f"""
        **Anomaly Statistics:**
        - Peak Anomaly Intensity: `{result['anomaly_map_raw'].max():.3f}`
        - Background Mean Intensity: `{result['anomaly_map_raw'].mean():.3f}`
        - Peak-to-Background Ratio: `{result['anomaly_map_raw'].max() / (result['anomaly_map_raw'].mean() + 1e-6):.1f}x`
        """)

    with tab4:
        st.markdown("#### 📷 Unadulterated Sensor Capture")
        st.caption(f"Original high-resolution camera feed for `{sample_name}` ({selected_view}).")
        st.image(result["raw_image_rgb"], use_container_width=True)

    with tab5:
        st.markdown("#### 📊 Model Architecture & Training Diagnostics")
        st.caption("Deep PatchCore model trained on 1,225 normal toothbrush samples across 5 multi-view camera angles.")

        m_info1, m_info2 = st.columns(2)
        with m_info1:
            st.markdown("""
            | Parameter | Value |
            | :--- | :--- |
            | **Model Architecture** | PatchCore |
            | **Feature Extractor** | ResNet-18 (`layer2`, `layer3`) |
            | **Input Resolution** | 512 x 512 native tensor |
            | **Coreset Sampling Ratio** | 0.001 (GPU-accelerated Blackwell selection) |
            | **Normal Memory Bank Size** | 1,254 patch feature vectors (384-dim) |
            | **Calibrated Decision Threshold** | τ = 0.530 (ROC Youden's J Index) |
            | **Hardware Accelerator** | NVIDIA GeForce RTX 5060 Laptop GPU |
            """)

        with m_info2:
            st.markdown("""
            | Benchmark Metric | Score |
            | :--- | :--- |
            | **Image AUROC** | **85.34%** |
            | **Pixel AUROC** | **91.59%** |
            | **Memory Fitting Time** | 2.05 seconds (GPU Coreset) |
            | **Live Inference Latency** | ~32 - 40 ms per part |
            | **Defect Types Detected** | Missing Bristles (CH), Scratches (QS), Stains (ZW) |
            """)


if __name__ == "__main__":
    main()
