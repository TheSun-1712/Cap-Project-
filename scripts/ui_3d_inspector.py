"""
3D CAD Metrology & Free Model Explorer Streamlit Component.
Provides:
1. 🔭 Free 3D CAD Model Explorer: Full 360° zoom, orbit, pan, cross-section slicing, and CAD wireframes.
2. 📐 3D Defect Metrology & Heatmap: Solid surface deviation maps, GD&T feature tables, and defect profiling.
"""

from pathlib import Path
import tempfile
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh
from scipy.spatial import cKDTree

from scripts.geometry_3d import (
    load_3d_model,
    align_point_clouds,
    compute_signed_deviation,
    inspect_gdt_features,
    cluster_defect_sites,
)


@st.cache_resource(show_spinner="⏳ Loading Nominal CAD Reference Model...")
def load_cached_ref_mesh():
    """Loads the nominal CAD triangulated mesh for solid 3D rendering."""
    golden_stl = Path("samples_3d/golden_part.stl")
    if not golden_stl.exists():
        from scripts.geometry_3d.step_loader import tessellate_step_to_mesh
        tessellate_step_to_mesh("0101BA302380N_010 XYZ.stp", output_stl_path=golden_stl)
    
    mesh = trimesh.load(str(golden_stl))
    return mesh


@st.cache_resource(show_spinner="⏳ Loading Nominal CAD Point Cloud...")
def load_cached_ref_points(num_samples: int = 35000):
    """Loads sampled surface points for metrology."""
    golden_stl = Path("samples_3d/golden_part.stl")
    geo = load_3d_model(golden_stl, num_samples=num_samples)
    if np.linalg.norm(geo.assembly_offset) < 100.0:
        geo.assembly_offset = np.array([1612.54, -569.57, 1152.75])
    return geo


def render_3d_inspection_tab():
    """Renders the comprehensive 3D CAD Metrology & Free Model Explorer workspace."""

    # Presets catalog - Default is the pure Nominal CAD model (PASS)
    presets_3d = {
        "🟢 Golden Standard (Nominal Model - PASS)": {
            "path": "samples_3d/golden_part.stl",
            "desc": "Certified 100% defect-free CAD authority (Your original model)",
            "expected": "PASS",
        },
        "🔴 Surface Impact Dent (-0.65 mm Flaw - REJECT)": {
            "path": "samples_3d/part_with_dent.stl",
            "desc": "Test specimen with synthetic concave gouge on curved face",
            "expected": "REJECT",
        },
        "🔴 Material Flash / Burr (+0.85 mm Flaw - REJECT)": {
            "path": "samples_3d/part_with_burr.stl",
            "desc": "Test specimen with raised manufacturing flash on flange edge",
            "expected": "REJECT",
        },
        "🔴 Countersunk Hole Ovality (REJECT GD&T)": {
            "path": "samples_3d/part_hole_ovality.stl",
            "desc": "Test specimen with countersunk fastener hole drill elongation",
            "expected": "REJECT",
        },
        "🔴 Flange Thermal Warping (154 mm Sag - REJECT)": {
            "path": "samples_3d/part_warped_flange.stl",
            "desc": "Test specimen with thermal curvature bow along long axis",
            "expected": "REJECT",
        },
    }

    # ==========================
    # SIDEBAR CONTROLS
    # ==========================
    with st.sidebar:
        st.markdown("### 🧊 3D Workspace Navigation")

        active_tab = st.radio(
            "Select 3D Mode:",
            [
                "🔭 Free 3D CAD Model Explorer",
                "📐 3D Defect Metrology & Heatmap",
            ],
            index=0,  # Default to Explorer
            help="Switch between free 3D CAD exploration and automated quality comparison.",
        )

        st.markdown("---")

        if active_tab == "🔭 Free 3D CAD Model Explorer":
            st.markdown("### 🎨 CAD Shading & Material")
            cad_color = st.selectbox(
                "Surface Material Finish:",
                [
                    "Titanium Silver (#cbd5e1)",
                    "Aero Metallic Blue (#38bdf8)",
                    "Industrial Slate (#64748b)",
                    "Gold Brass (#f59e0b)",
                    "Deep Carbon (#334155)",
                ],
                index=0,
            )
            color_map = {
                "Titanium Silver (#cbd5e1)": "#cbd5e1",
                "Aero Metallic Blue (#38bdf8)": "#38bdf8",
                "Industrial Slate (#64748b)": "#64748b",
                "Gold Brass (#f59e0b)": "#f59e0b",
                "Deep Carbon (#334155)": "#334155",
            }
            chosen_color = color_map[cad_color]

            shading_mode = st.radio(
                "Render Style:",
                ["Solid Shaded Surface", "CAD Wireframe (Facets)", "Both (Surface + Wireframe)"],
                index=0,
            )

            show_annotations = st.checkbox("Show 3D Feature Pin Callouts", value=True)

            st.markdown("---")
            st.markdown("### ✂️ Cross-Section Slicer")
            enable_slice = st.checkbox("Enable Interior Cross-Section Slicing", value=False)
            slice_axis = "Z"
            slice_val = 0.5
            if enable_slice:
                slice_axis = st.selectbox("Cut Plane Axis:", ["Z (Length 154mm)", "Y (Width 52mm)", "X (Thickness 18mm)"])
                slice_val = st.slider("Cut Position (% of part):", 0, 100, 50, step=2)

        else:
            # Metrology Settings
            st.markdown("### ⚙️ Inspection Configuration")
            mode_choice = st.radio(
                "Select Inspected Part:",
                ["🧪 Benchmark Test Presets", "📤 Upload Custom 3D Part"],
                index=0,
            )

            test_file_path = "samples_3d/golden_part.stl"
            part_display_name = "Golden Standard Part"

            if mode_choice == "🧪 Benchmark Test Presets":
                selected_preset_key = st.selectbox(
                    "Choose Test Model:",
                    list(presets_3d.keys()),
                    index=0,  # Default to GOLDEN STANDARD (PASS)
                )
                preset_info = presets_3d[selected_preset_key]
                test_file_path = preset_info["path"]
                part_display_name = selected_preset_key.split("(")[0].strip()
                st.info(f"ℹ️ **Inspecting:** {preset_info['desc']}")
            else:
                uploaded_3d = st.file_uploader(
                    "Upload 3D Part (STEP, STL, PLY, OBJ):",
                    type=["stp", "step", "stl", "ply", "obj"],
                    help="Upload a manufactured part scan to compare against your STEP authority.",
                )
                if uploaded_3d is not None:
                    suffix = Path(uploaded_3d.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                        tmp_f.write(uploaded_3d.read())
                        test_file_path = tmp_f.name
                    part_display_name = uploaded_3d.name

            st.markdown("---")
            st.markdown("### 📏 Tolerance Limits")
            tol_val = st.slider(
                "Dimensional Tolerance (±τ in mm):",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.01,
                help="Bilateral tolerance band. Points with deviation inside ±τ are green.",
            )

            render_style_metro = st.radio(
                "Heatmap Presentation:",
                ["Solid 3D Surface Mesh (Recommended)", "Sampled Point Cloud"],
                index=0,
            )

    # =========================================================================
    # TAB 1: FREE 3D CAD MODEL EXPLORER
    # =========================================================================
    if active_tab == "🔭 Free 3D CAD Model Explorer":
        st.markdown("""
        <div class="system-header">
            <h1 class="system-title">
                <span>🔭</span> 3D CAD MODEL EXPLORER — 0101BA302380N_010 XYZ
            </h1>
            <p class="system-subtitle">
                Interactive 360° Inspection Viewport · CATIA V5 AP203 B-Rep Geometry · True Dimensions: 18.32 x 51.89 x 153.93 mm
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Mouse Navigation Legend Card
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: space-between;">
            <div style="font-size: 13px; color: #cbd5e1;"><strong>🖱️ Left-Click + Drag:</strong> Full 360° 3D Orbit / Rotate</div>
            <div style="font-size: 13px; color: #cbd5e1;"><strong>🖱️ Right-Click + Drag:</strong> Pan / Translate View</div>
            <div style="font-size: 13px; color: #cbd5e1;"><strong>🖱️ Scroll Wheel:</strong> Smooth Zoom In / Zoom Out</div>
            <div style="font-size: 13px; color: #cbd5e1;"><strong>🖱️ Double-Click:</strong> Reset Camera Frame</div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Camera View Buttons
        st.markdown("**Quick Camera Angle Shortcuts:**")
        v_col1, v_col2, v_col3, v_col4, v_col5, v_col6, v_col7 = st.columns(7)
        cam_eye = dict(x=1.8, y=1.8, z=1.4)

        if v_col1.button("🔄 Isometric", use_container_width=True):
            cam_eye = dict(x=1.8, y=1.8, z=1.4)
        if v_col2.button("⬆️ Top", use_container_width=True):
            cam_eye = dict(x=0.0, y=0.0, z=3.0)
        if v_col3.button("⬇️ Bottom", use_container_width=True):
            cam_eye = dict(x=0.0, y=0.0, z=-3.0)
        if v_col4.button("⬅️ Left", use_container_width=True):
            cam_eye = dict(x=0.0, y=-3.0, z=0.0)
        if v_col5.button("➡️ Right", use_container_width=True):
            cam_eye = dict(x=0.0, y=3.0, z=0.0)
        if v_col6.button("🎯 Front", use_container_width=True):
            cam_eye = dict(x=3.0, y=0.0, z=0.0)
        if v_col7.button("🔙 Back", use_container_width=True):
            cam_eye = dict(x=-3.0, y=0.0, z=0.0)

        # Load Mesh
        mesh = load_cached_ref_mesh()
        v = mesh.vertices.copy()
        f = mesh.faces.copy()

        # Zero-center for intuitive CAD orbit around origin
        centroid = (v.min(axis=0) + v.max(axis=0)) / 2.0
        v_local = v - centroid

        # Handle Cross-Section Slice
        if enable_slice:
            axis_idx = 2 if "Z" in slice_axis else (1 if "Y" in slice_axis else 0)
            span_min = v_local[:, axis_idx].min()
            span_max = v_local[:, axis_idx].max()
            cut_pos = span_min + (span_max - span_min) * (slice_val / 100.0)
            
            tri_centers = v_local[f].mean(axis=1)
            valid_tri = tri_centers[:, axis_idx] <= cut_pos
            f = f[valid_tri]
            st.caption(f"✂️ Slicing {slice_axis[:1]} at position `{cut_pos:+.2f} mm` — Exposing internal cavities and wall thickness.")

        plot_elements = []

        # Solid Shaded Surface Mesh
        if shading_mode in ["Solid Shaded Surface", "Both (Surface + Wireframe)"]:
            solid_trace = go.Mesh3d(
                x=v_local[:, 0],
                y=v_local[:, 1],
                z=v_local[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=chosen_color,
                flatshading=False,
                lighting=dict(
                    ambient=0.45,
                    diffuse=0.85,
                    specular=0.55,
                    roughness=0.45,
                    fresnel=0.35,
                ),
                lightposition=dict(x=100, y=200, z=150),
                opacity=1.0,
                name="Solid CAD Body",
                hovertemplate="<b>X:</b> %{x:.2f} mm<br><b>Y:</b> %{y:.2f} mm<br><b>Z:</b> %{z:.2f} mm<extra></extra>",
            )
            plot_elements.append(solid_trace)

        # Wireframe / Facets
        if shading_mode in ["CAD Wireframe (Facets)", "Both (Surface + Wireframe)"]:
            wire_trace = go.Mesh3d(
                x=v_local[:, 0],
                y=v_local[:, 1],
                z=v_local[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color="#38bdf8",
                wireframe=dict(show=True, color="#0284c7", width=1),
                opacity=0.3 if shading_mode == "Both (Surface + Wireframe)" else 1.0,
                name="Wireframe Facets",
            )
            plot_elements.append(wire_trace)

        # 3D Feature Callouts / Annotations
        if show_annotations:
            # Shift known world feature coordinates to local coordinates
            # Hole 1: Z ~ 1092.06, X ~ 1608.05, Y ~ -571.11
            # Hole 2: Z ~ 1112.06, X ~ 1608.28, Y ~ -570.60
            # Boss: Z ~ 1165.51, X ~ 1616.96, Y ~ -583.28
            features_to_mark = [
                ("Countersunk Hole #1 (Ø 7.0 mm)", np.array([1608.05, -571.11, 1092.06]) - centroid, "#22c55e"),
                ("Countersunk Hole #2 (Ø 7.0 mm)", np.array([1608.28, -570.60, 1112.06]) - centroid, "#22c55e"),
                ("Cylindrical Boss (Ø 20 & 26 mm)", np.array([1616.96, -583.28, 1165.51]) - centroid, "#38bdf8"),
                ("154 mm Flange Rim", np.array([1603.4, -543.7, 1229.4]) - centroid, "#f59e0b"),
            ]
            for label, pt, pcolor in features_to_mark:
                plot_elements.append(go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode="markers+text",
                    marker=dict(size=8, color=pcolor, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
                    text=[label],
                    textposition="top right",
                    textfont=dict(color="#f8fafc", size=11),
                    name=label,
                ))

        fig_cad = go.Figure(data=plot_elements)
        fig_cad.update_layout(
            template="plotly_dark",
            paper_bgcolor="#090d16",
            plot_bgcolor="#090d16",
            margin=dict(l=0, r=0, b=0, t=10),
            scene=dict(
                xaxis=dict(title="X Thickness (mm)", backgroundcolor="#090d16", gridcolor="#1e293b", showbackground=True),
                yaxis=dict(title="Y Width (mm)", backgroundcolor="#090d16", gridcolor="#1e293b", showbackground=True),
                zaxis=dict(title="Z Length (154mm)", backgroundcolor="#090d16", gridcolor="#1e293b", showbackground=True),
                aspectmode="data",
                camera=dict(eye=cam_eye),
            ),
            height=680,
        )

        st.plotly_chart(fig_cad, use_container_width=True)

        # Specifications & Geometric Properties
        st.markdown("### 📋 Part Dimensional Summary")
        dim_c1, dim_c2, dim_c3, dim_c4 = st.columns(4)
        dim_c1.metric("Thickness (X-span)", f"{mesh.extents[0]:.2f} mm")
        dim_c2.metric("Width (Y-span)", f"{mesh.extents[1]:.2f} mm")
        dim_c3.metric("Length (Z-span)", f"{mesh.extents[2]:.2f} mm")
        dim_c4.metric("Surface Area", f"{mesh.area:,.1f} mm²")

        return

    # =========================================================================
    # TAB 2: 3D DEFECT METROLOGY & HEATMAP
    # =========================================================================
    st.markdown("""
    <div class="system-header">
        <h1 class="system-title">
            <span>📐</span> 3D METROLOGY & SURFACE DEVIATION HEATMAP
        </h1>
        <p class="system-subtitle">
            Autonomous CAD Comparison · Normal-Constrained Signed Deviation · GD&T Tolerancing
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Explanatory Callout Banner
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
        <strong style="color: #38bdf8; font-size: 15px;">ℹ️ How 3D Quality Decisions (PASS / REJECT) Work:</strong>
        <p style="color: #cbd5e1; font-size: 13px; margin-top: 6px; margin-bottom: 4px; line-height: 1.5;">
            In industrial metrology, quality inspection requires comparing a <strong>Nominal CAD Authority</strong> (your master <code>0101BA302380N_010 XYZ.stp</code>) against a <strong>Manufactured Physical Part</strong> (e.g. an optical 3D scan).
            <br>
            • <strong>Your original STEP file is the 100% perfect reference standard.</strong> It was <em>NOT</em> rejected!
            <br>
            • To inspect your own CAD model, ensure <strong>🟢 Golden Standard (Nominal Model)</strong> is selected in the sidebar — it will score <strong>100% PASS</strong>.
            <br>
            • The other test options (e.g. <em>Surface Impact Dent</em>) are pre-loaded defective benchmark samples to demonstrate how real-world factory flaws (dents, burrs, warping) are automatically caught and flagged.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load Nominal Reference Points & Inspected Test Part
    ref_step_path = "0101BA302380N_010 XYZ.stp"
    with st.spinner("⚡ Loading nominal CAD reference and test models..."):
        ref_geo = load_cached_ref_points(num_samples=35000)
        test_geo = load_3d_model(test_file_path, num_samples=35000)

    # 6-DoF Alignment
    with st.spinner("🔄 Aligning 6-DoF coordinates (PCA + Huber Point-to-Plane ICP)..."):
        align_res = align_point_clouds(
            source_pts=test_geo.points,
            target_pts=ref_geo.points,
            source_normals=test_geo.point_normals,
            target_normals=ref_geo.point_normals,
        )

    # Signed Surface Deviation Metrology
    with st.spinner("📐 Computing signed surface deviation field..."):
        metro_res = compute_signed_deviation(
            test_points=align_res.aligned_points,
            ref_points=ref_geo.points,
            ref_normals=ref_geo.point_normals,
            test_normals=align_res.aligned_normals,
            tolerance=tol_val,
        )

    # GD&T Features
    with st.spinner("🔩 Inspecting GD&T feature conformance..."):
        gdt_res = inspect_gdt_features(
            test_points=align_res.aligned_points,
            test_normals=align_res.aligned_normals,
            assembly_offset=ref_geo.assembly_offset,
        )

    # Defect Clusters
    with st.spinner("🏷️ Segmenting 3D defect sites..."):
        clusters = cluster_defect_sites(
            test_points=align_res.aligned_points,
            signed_deviations=metro_res.signed_deviations,
            tolerance=tol_val,
            assembly_offset=ref_geo.assembly_offset,
        )

    # Verdict
    is_fail = (metro_res.verdict == "FAIL") or (gdt_res.overall_verdict == "FAIL")

    # High-Visibility Verdict Banner
    if is_fail:
        fail_reasons_html = ""
        if metro_res.verdict == "FAIL":
            fail_reasons_html += f"• Peak Surface Deviation: <strong>{metro_res.max_deviation:.3f} mm</strong> exceeds tolerance limit (±{tol_val:.2f} mm).<br>"
        if gdt_res.overall_verdict == "FAIL":
            failed_gdt = [f.name for f in gdt_res.features if f.verdict == "FAIL"]
            fail_reasons_html += f"• GD&T Feature Non-Conformance: <strong>{', '.join(failed_gdt)}</strong>.<br>"

        st.markdown(f"""
        <div class="verdict-fail">
            <div class="verdict-title" style="color: #ef4444;">
                🚨 REJECT — DEFECT DETECTED ON {part_display_name.upper()}
            </div>
            <div class="verdict-desc" style="color: #fca5a5;">
                {fail_reasons_html}
                Total out-of-tolerance surface area: <strong>{metro_res.defect_ratio * 100:.2f}%</strong> ({len(clusters)} isolated defect site(s)).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-pass">
            <div class="verdict-title" style="color: #22c55e;">
                ✅ PASS — 100% SPECIFICATION CONFORMANCE ({part_display_name.upper()})
            </div>
            <div class="verdict-desc" style="color: #86efac;">
                Surface conforms strictly to the CAD design authority within <strong>±{tol_val:.2f} mm</strong> (RMSE: <strong>{metro_res.rmse:.4f} mm</strong>).
                Countersunk hole diameters (7.0 mm), fastener pitch (20.0 mm), and cylindrical bosses are 100% in specification.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Peak Deviation (d_max)</div>
            <div class="kpi-value" style="color: {'#ef4444' if metro_res.max_deviation > tol_val * 2.0 else '#22c55e'};">
                {metro_res.max_deviation:.3f} mm
            </div>
            <div class="kpi-sub">Tolerance Envelope: ±{tol_val:.2f} mm</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Surface RMSE</div>
            <div class="kpi-value" style="color: {'#ef4444' if metro_res.rmse > tol_val else '#38bdf8'};">
                {metro_res.rmse:.4f} mm
            </div>
            <div class="kpi-sub">Overall Geometric Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Defective Area Ratio</div>
            <div class="kpi-value" style="color: {'#ef4444' if metro_res.defect_ratio > 0.015 else '#22c55e'};">
                {metro_res.defect_ratio * 100:.2f}%
            </div>
            <div class="kpi-sub">Max Allowable: 1.50%</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">GD&T Conformance</div>
            <div class="kpi-value" style="color: {'#22c55e' if gdt_res.overall_verdict == 'PASS' else '#ef4444'};">
                {gdt_res.overall_verdict}
            </div>
            <div class="kpi-sub">{sum(1 for f in gdt_res.features if f.verdict == 'PASS')} / {len(gdt_res.features)} Checks Passed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================
    # SOLID SURFACE HEATMAP VIEW
    # ==========================
    st.markdown("### 🗺️ Continuous Surface Deviation Heatmap")

    # Heatmap Color Legend Guide
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-radius: 8px; padding: 12px 18px; margin-bottom: 14px;">
        <strong style="color: #f8fafc; font-size: 14px;">🎨 How to Read the Heatmap Colors:</strong>
        <div style="display: flex; flex-wrap: wrap; gap: 24px; margin-top: 8px; font-size: 13px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background-color: #22c55e;"></span>
                <span style="color: #cbd5e1;"><strong>Green (±0.05 mm):</strong> Nominal Surface (100% In-Spec)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background-color: #ef4444;"></span>
                <span style="color: #cbd5e1;"><strong>Red (< -0.20 mm):</strong> Material Deficit (Dent / Hole / Cavity)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; border-radius: 3px; background-color: #3b82f6;"></span>
                <span style="color: #cbd5e1;"><strong>Blue (> +0.20 mm):</strong> Material Excess (Burr / Flash / Protrusion)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render Heatmap as Solid 3D Surface Mesh
    if "Solid 3D Surface" in render_style_metro and Path(test_file_path).exists():
        test_mesh = trimesh.load(test_file_path)
        v_test = test_mesh.vertices.copy()
        f_test = test_mesh.faces.copy()
        
        # Center mesh
        c_test = (v_test.min(axis=0) + v_test.max(axis=0)) / 2.0
        v_test_local = v_test - c_test

        # Calculate deviation directly on mesh vertices
        ref_tree = cKDTree(ref_geo.points)
        _, v_idxs = ref_tree.query(v_test_local)
        v_diffs = v_test_local - ref_geo.points[v_idxs]
        v_signed_devs = np.sum(v_diffs * ref_geo.point_normals[v_idxs], axis=1)

        heatmap_trace = go.Mesh3d(
            x=v_test_local[:, 0],
            y=v_test_local[:, 1],
            z=v_test_local[:, 2],
            i=f_test[:, 0],
            j=f_test[:, 1],
            k=f_test[:, 2],
            intensity=v_signed_devs,
            colorscale=[
                [0.0, "#ef4444"],    # Deep Red (Dent / Hole)
                [0.35, "#f59e0b"],   # Orange (Moderate Deficit)
                [0.45, "#22c55e"],   # Green (Nominal)
                [0.55, "#22c55e"],   # Green (Nominal)
                [0.65, "#38bdf8"],   # Cyan (Moderate Excess)
                [1.0, "#2563eb"],    # Deep Blue (Burr / Protrusion)
            ],
            cmin=-tol_val * 2.0,
            cmax=tol_val * 2.0,
            colorbar=dict(
                title=dict(text="Deviation (mm)", font=dict(color="#f8fafc", size=13)),
                tickfont=dict(color="#cbd5e1", size=11),
                thickness=18,
                len=0.75,
            ),
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.5),
            name="Metrology Surface Mesh",
            hovertemplate="<b>X:</b> %{x:.2f} mm<br><b>Y:</b> %{y:.2f} mm<br><b>Z:</b> %{z:.2f} mm<br><b>Deviation:</b> %{intensity:+.3f} mm<extra></extra>",
        )
        metro_plot = [heatmap_trace]
    else:
        # Fallback to Point Cloud
        pts = align_res.aligned_points
        devs = metro_res.signed_deviations
        sub_n = min(25000, len(pts))
        s_idx = np.random.choice(len(pts), sub_n, replace=False)
        pts_sub = pts[s_idx]
        devs_sub = devs[s_idx]

        scatter_trace = go.Scatter3d(
            x=pts_sub[:, 0], y=pts_sub[:, 1], z=pts_sub[:, 2],
            mode="markers",
            marker=dict(
                size=3.0,
                color=devs_sub,
                colorscale="RdYlGn_r",
                cmin=-tol_val * 2.0,
                cmax=tol_val * 2.0,
                colorbar=dict(title="Deviation (mm)", thickness=18),
                opacity=0.9,
            ),
            name="Point Cloud",
        )
        metro_plot = [scatter_trace]

    # Overlay Defect Cluster Markers
    for c in clusters:
        metro_plot.append(go.Scatter3d(
            x=[c.centroid_local[0]],
            y=[c.centroid_local[1]],
            z=[c.centroid_local[2]],
            mode="markers+text",
            marker=dict(size=10, color="#ef4444", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            text=[f"Flaw #{c.cluster_id}: {c.defect_type} ({c.peak_deviation:.2f}mm)"],
            textposition="top center",
            textfont=dict(color="#fca5a5", size=11),
            name=f"Flaw #{c.cluster_id}",
        ))

    fig_metro = go.Figure(data=metro_plot)
    fig_metro.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=0, r=0, b=0, t=10),
        scene=dict(
            xaxis=dict(title="X Thickness (mm)", backgroundcolor="#090d16", gridcolor="#1e293b"),
            yaxis=dict(title="Y Width (mm)", backgroundcolor="#090d16", gridcolor="#1e293b"),
            zaxis=dict(title="Z Length (154mm)", backgroundcolor="#090d16", gridcolor="#1e293b"),
            aspectmode="data",
        ),
        height=620,
    )
    st.plotly_chart(fig_metro, use_container_width=True)

    # Deviation Histogram Distribution
    st.markdown("#### 📊 Millimeter Deviation Distribution")
    hist_vals = metro_res.signed_deviations
    hist_fig = go.Figure(data=[go.Histogram(
        x=hist_vals,
        nbinsx=60,
        marker_color="#38bdf8",
        opacity=0.8,
    )])
    hist_fig.add_vline(x=-tol_val, line_dash="dash", line_color="#ef4444", annotation_text=f"-{tol_val}mm (Tol Limit)")
    hist_fig.add_vline(x=+tol_val, line_dash="dash", line_color="#ef4444", annotation_text=f"+{tol_val}mm (Tol Limit)")
    hist_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        xaxis_title="Signed Deviation (mm)",
        yaxis_title="Point Count",
        height=240,
        margin=dict(l=40, r=40, b=30, t=30),
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Sub-tabs for Details
    d_tab1, d_tab2 = st.tabs(["📐 GD&T Feature Conformance Table", "🏷️ 3D Defect Site Details"])

    with d_tab1:
        gdt_rows = []
        for f in gdt_res.features:
            status_badge = "🟢 PASS" if f.verdict == "PASS" else "🔴 FAIL"
            gdt_rows.append({
                "Feature Name": f.name,
                "Nominal": f"{f.nominal:.2f} {f.unit}",
                "Measured": f"{f.measured:.2f} {f.unit}",
                "Deviation": f"{f.deviation:+.2f} {f.unit}",
                "Tolerance": f"±{f.tolerance:.2f} {f.unit}",
                "Status": status_badge,
            })
        st.dataframe(gdt_rows, use_container_width=True, hide_index=True)

    with d_tab2:
        if len(clusters) == 0:
            st.success("🎉 No isolated defect clusters detected on this part.")
        else:
            for c in clusters:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                    <strong style="color: #ef4444; font-size: 15px;">Site #{c.cluster_id}: {c.defect_type}</strong><br>
                    <span style="font-size: 13px; color: #cbd5e1;">Peak Flaw Magnitude: <strong>{c.peak_deviation:.3f} mm</strong></span> |
                    <span style="font-size: 13px; color: #cbd5e1;">Flaw Area: <strong>{c.estimated_area_mm2:.1f} mm²</strong> ({c.point_count} pts)</span><br>
                    <span style="font-size: 12px; color: #94a3b8;">World Coordinates: X={c.centroid_world[0]:.1f}, Y={c.centroid_world[1]:.1f}, Z={c.centroid_world[2]:.1f} mm</span>
                </div>
                """, unsafe_allow_html=True)
