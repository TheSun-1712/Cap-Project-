"""
Standalone CLI for 3D CAD Metrology and Defect Inspection.
Performs 6-DoF alignment, signed deviation calculation, GD&T feature verification,
and 3D defect site clustering on CATIA STEP models and manufactured 3D meshes.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from scripts.geometry_3d import (
    load_3d_model,
    align_point_clouds,
    compute_signed_deviation,
    inspect_gdt_features,
    cluster_defect_sites,
)


def run_3d_inspection(
    ref_path: str,
    test_path: str,
    tolerance: float = 0.20,
    num_samples: int = 50000,
    output_json: Optional[str] = None,
    output_html: Optional[str] = None,
) -> dict:
    print("\n" + "=" * 76)
    print("      3D CAD METROLOGY & DEFECT INSPECTION ENGINE (ISO 10303 AP203)")
    print("=" * 76)
    print(f"[*] Reference CAD Model : {ref_path}")
    print(f"[*] Manufactured Part   : {test_path}")
    print(f"[*] Dimensional Tol     : +/- {tolerance:.2f} mm")
    print(f"[*] Point Sampling      : {num_samples:,} surface points")
    print("-" * 76)

    # 1. Ingestion
    print("[1/5] Ingesting and discretizing geometries...")
    ref_geo = load_3d_model(ref_path, num_samples=num_samples)
    test_geo = load_3d_model(test_path, num_samples=num_samples)
    print(f"      Ref : {ref_geo.name} ({ref_geo.format.upper()}) | Envelope: {ref_geo.dimensions[0]:.1f} x {ref_geo.dimensions[1]:.1f} x {ref_geo.dimensions[2]:.1f} mm")
    print(f"      Test: {test_geo.name} ({test_geo.format.upper()}) | Envelope: {test_geo.dimensions[0]:.1f} x {test_geo.dimensions[1]:.1f} x {test_geo.dimensions[2]:.1f} mm")

    # 2. Alignment
    print("[2/5] Performing 6-DoF rigid body registration (PCA + Huber ICP)...")
    align_res = align_point_clouds(
        source_pts=test_geo.points,
        target_pts=ref_geo.points,
        source_normals=test_geo.point_normals,
        target_normals=ref_geo.point_normals,
    )
    print(f"      Alignment Residual RMSE: {align_res.rmse:.4f} mm ({align_res.iterations} iterations)")

    # 3. Metrology
    print("[3/5] Computing normal-constrained signed surface deviation...")
    metro_res = compute_signed_deviation(
        test_points=align_res.aligned_points,
        ref_points=ref_geo.points,
        ref_normals=ref_geo.point_normals,
        test_normals=align_res.aligned_normals,
        tolerance=tolerance,
    )

    # 4. GD&T Features
    print("[4/5] Inspecting GD&T critical features (Countersink holes & Bosses)...")
    gdt_res = inspect_gdt_features(
        test_points=align_res.aligned_points,
        test_normals=align_res.aligned_normals,
        assembly_offset=ref_geo.assembly_offset,
    )

    # 5. Defect Clustering
    print("[5/5] Segmenting 3D defect sites (DBSCAN)...")
    clusters = cluster_defect_sites(
        test_points=align_res.aligned_points,
        signed_deviations=metro_res.signed_deviations,
        tolerance=tolerance,
        assembly_offset=ref_geo.assembly_offset,
    )

    # Overall Verdict
    final_verdict = "PASS" if (metro_res.verdict == "PASS" and gdt_res.overall_verdict == "PASS") else "FAIL"

    # Display Results
    print("\n" + "=" * 76)
    print(f"                        OPERATIONAL VERDICT: {final_verdict}")
    print("=" * 76)

    print("\n--- [QUANTITATIVE METROLOGY METRICS] ---")
    print(f"  Peak Flaw Deviation (d_max) : {metro_res.max_deviation:.3f} mm")
    print(f"  Max Material Excess (Burr)  : {metro_res.max_positive:+.3f} mm")
    print(f"  Max Material Deficit (Dent) : {metro_res.max_negative:+.3f} mm")
    print(f"  Root Mean Square Error (RMSE): {metro_res.rmse:.4f} mm")
    print(f"  Mean Absolute Error (MAE)   : {metro_res.mae:.4f} mm")
    print(f"  Out-of-Tolerance Area Ratio : {metro_res.defect_ratio * 100:.2f}%")

    print("\n--- [GD&T FEATURE CONFORMANCE] ---")
    for f in gdt_res.features:
        status_symbol = "[PASS]" if f.verdict == "PASS" else "[FAIL]"
        print(f"  {status_symbol} {f.name:<34} : {f.measured:.2f} {f.unit} (Nominal: {f.nominal:.2f}, Dev: {f.deviation:+.2f} {f.unit}, Tol: +/-{f.tolerance:.2f})")

    print("\n--- [ISOLATED DEFECT SITES] ---")
    if len(clusters) == 0:
        print("  No isolated defect clusters detected exceeding tolerance.")
    else:
        for c in clusters:
            print(f"  Site #{c.cluster_id}: {c.defect_type}")
            print(f"    - Peak Severity : {c.peak_deviation:.3f} mm (Mean: {c.mean_deviation:+.3f} mm)")
            print(f"    - Flaw Area     : {c.estimated_area_mm2:.1f} mm^2 ({c.point_count} points)")
            print(f"    - World Centroid: [{c.centroid_world[0]:.1f}, {c.centroid_world[1]:.1f}, {c.centroid_world[2]:.1f}] mm")

    print("=" * 76 + "\n")

    # Build structured report dict
    report = {
        "reference_file": str(ref_path),
        "test_file": str(test_path),
        "tolerance_mm": tolerance,
        "operational_verdict": final_verdict,
        "metrology": {
            "max_deviation_mm": metro_res.max_deviation,
            "max_positive_mm": metro_res.max_positive,
            "max_negative_mm": metro_res.max_negative,
            "rmse_mm": metro_res.rmse,
            "mae_mm": metro_res.mae,
            "defect_ratio_pct": metro_res.defect_ratio * 100.0,
            "surface_verdict": metro_res.verdict,
            "failure_reasons": metro_res.verdict_reasons,
        },
        "gdt_features": [
            {
                "name": f.name,
                "nominal": f.nominal,
                "measured": f.measured,
                "deviation": f.deviation,
                "tolerance": f.tolerance,
                "verdict": f.verdict,
            }
            for f in gdt_res.features
        ],
        "defect_clusters": [
            {
                "cluster_id": c.cluster_id,
                "type": c.defect_type,
                "peak_mm": c.peak_deviation,
                "mean_mm": c.mean_deviation,
                "area_mm2": c.estimated_area_mm2,
                "world_centroid": c.centroid_world.tolist(),
            }
            for c in clusters
        ],
    }

    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[+] Saved structured inspection report: {out_p}")

    if output_html is not None:
        _export_interactive_html(
            output_html=output_html,
            ref_geo=ref_geo,
            aligned_pts=align_res.aligned_points,
            signed_devs=metro_res.signed_deviations,
            tolerance=tolerance,
            clusters=clusters,
        )

    return report


def _export_interactive_html(
    output_html: str,
    ref_geo,
    aligned_pts: np.ndarray,
    signed_devs: np.ndarray,
    tolerance: float,
    clusters: list,
):
    """Exports interactive 3D WebGL HTML visualizer using Plotly."""
    import plotly.graph_objects as go

    out_p = Path(output_html)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Subsample for smooth 60fps WebGL rendering
    sub_n = min(30000, len(aligned_pts))
    idx = np.random.choice(len(aligned_pts), sub_n, replace=False)
    sub_pts = aligned_pts[idx]
    sub_devs = signed_devs[idx]

    # Create Scatter3d point cloud with diverging colorscale
    scatter = go.Scatter3d(
        x=sub_pts[:, 0],
        y=sub_pts[:, 1],
        z=sub_pts[:, 2],
        mode="markers",
        marker=dict(
            size=2.5,
            color=sub_devs,
            colorscale="RdYlGn_r",
            cmin=-tolerance * 2.0,
            cmax=tolerance * 2.0,
            colorbar=dict(title="Deviation (mm)", thickness=15),
            opacity=0.85,
        ),
        name="Inspected Surface",
    )

    data = [scatter]

    # Add cluster centroid markers
    for c in clusters:
        data.append(go.Scatter3d(
            x=[c.centroid_local[0]],
            y=[c.centroid_local[1]],
            z=[c.centroid_local[2]],
            mode="markers+text",
            marker=dict(size=8, color="red", symbol="diamond"),
            text=[f"Flaw #{c.cluster_id}: {c.defect_type} ({c.peak_deviation:.2f}mm)"],
            textposition="top center",
            name=f"Site #{c.cluster_id}",
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"3D Model Inspection Deviation Map (Tolerance: +/-{tolerance} mm)",
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(out_p))
    print(f"[+] Saved interactive 3D WebGL report: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="3D CAD Metrology & Defect Inspection CLI")
    parser.add_argument("--ref", type=str, default="0101BA302380N_010 XYZ.stp", help="Reference CAD file (STEP, STL)")
    parser.add_argument("--test", type=str, required=True, help="Test manufactured 3D part (STL, PLY, OBJ, STEP)")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Tolerance band (+/- mm)")
    parser.add_argument("--samples", type=int, default=40000, help="Surface points to sample")
    parser.add_argument("--out-json", type=str, default="results/3d_inspection_report.json", help="Path for JSON output")
    parser.add_argument("--out-html", type=str, default=None, help="Path for interactive 3D HTML output")

    args = parser.parse_args()
    res = run_3d_inspection(
        ref_path=args.ref,
        test_path=args.test,
        tolerance=args.tolerance,
        num_samples=args.samples,
        output_json=args.out_json,
        output_html=args.out_html,
    )

    sys.exit(0 if res["operational_verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
