"""
Procedural 3D Defect Synthesizer anchored to 0101BA302380N_010 XYZ.stp.
Generates realistic industrial defect test cases:
1. golden_part.stl: Pure CAD nominal reference
2. part_with_dent.stl: Concave indentation / impact dent (-0.6 mm)
3. part_with_burr.stl: Raised flash / material protrusion (+0.8 mm)
4. part_hole_ovality.stl: Countersunk hole elongation (dia 7.45 mm)
5. part_warped_flange.stl: Thermal bending / curvature sag along the 154 mm axis
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import trimesh

from .step_loader import tessellate_step_to_mesh


def generate_benchmark_suite(
    step_path: Union[str, Path] = "0101BA302380N_010 XYZ.stp",
    output_dir: Union[str, Path] = "samples_3d",
) -> dict:
    """
    Generates all reference and defective 3D models for automated benchmarking.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    golden_path = out_dir / "golden_part.stl"
    print(f"[*] Tessellating nominal STEP model to {golden_path}...")
    base_mesh = tessellate_step_to_mesh(step_path, output_stl_path=golden_path)

    v_orig = base_mesh.vertices.copy()
    faces = base_mesh.faces.copy()
    bounds_min = v_orig.min(axis=0)
    bounds_max = v_orig.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0

    generated_files = {"golden": str(golden_path.resolve())}

    # -------------------------------------------------------------
    # 1. Defect: Concave Dent (-0.65 mm) on the curved B-Spline face
    # -------------------------------------------------------------
    v_dent = v_orig.copy()
    dent_center = np.array([center[0], center[1], center[2] - 15.0])
    r_dent = 7.0  # mm radius of dent
    d_dists = np.linalg.norm(v_dent - dent_center, axis=1)
    dent_mask = d_dists < r_dent

    if np.any(dent_mask):
        # Cosine bell displacement along surface normal (inward)
        dent_disp = -0.65 * (0.5 * (1.0 + np.cos(np.pi * d_dists[dent_mask] / r_dent)))
        v_dent[dent_mask] += base_mesh.vertex_normals[dent_mask] * dent_disp[:, np.newaxis]

    dent_mesh = trimesh.Trimesh(vertices=v_dent, faces=faces, process=True)
    dent_path = out_dir / "part_with_dent.stl"
    dent_mesh.export(str(dent_path))
    generated_files["dent"] = str(dent_path.resolve())
    print(f"[+] Wrote {dent_path.name}")

    # -------------------------------------------------------------
    # 2. Defect: Burr / Material Flash (+0.85 mm) along flange edge
    # -------------------------------------------------------------
    v_burr = v_orig.copy()
    burr_center = np.array([center[0] + 6.0, center[1] + 8.0, center[2] + 20.0])
    r_burr = 8.5
    b_dists = np.linalg.norm(v_burr - burr_center, axis=1)
    burr_mask = b_dists < r_burr

    if np.any(burr_mask):
        burr_disp = +0.85 * (0.5 * (1.0 + np.cos(np.pi * b_dists[burr_mask] / r_burr)))
        v_burr[burr_mask] += base_mesh.vertex_normals[burr_mask] * burr_disp[:, np.newaxis]

    burr_mesh = trimesh.Trimesh(vertices=v_burr, faces=faces, process=True)
    burr_path = out_dir / "part_with_burr.stl"
    burr_mesh.export(str(burr_path))
    generated_files["burr"] = str(burr_path.resolve())
    print(f"[+] Wrote {burr_path.name}")

    # -------------------------------------------------------------
    # 3. Defect: Countersunk Hole Ovality / Elongation
    # -------------------------------------------------------------
    v_hole = v_orig.copy()
    # Hole 1 is located around Z ~ 1092.06 mm in world coordinates
    # Let's find hole 1 vertices
    h1_center = np.array([1608.05, -571.11, 1092.06])
    h_dists = np.linalg.norm(v_hole - h1_center, axis=1)
    h_mask = h_dists < 5.5  # near hole boundary

    if np.any(h_mask):
        # Stretch in Y direction by +0.45 mm (making diameter 7.45 mm)
        v_hole[h_mask, 1] += 0.45 * np.exp(-((v_hole[h_mask, 2] - 1092.06)**2) / 4.0)

    hole_mesh = trimesh.Trimesh(vertices=v_hole, faces=faces, process=True)
    hole_path = out_dir / "part_hole_ovality.stl"
    hole_mesh.export(str(hole_path))
    generated_files["hole_ovality"] = str(hole_path.resolve())
    print(f"[+] Wrote {hole_path.name}")

    # -------------------------------------------------------------
    # 4. Defect: Thermal Warping / Bending along 154 mm Z-span
    # -------------------------------------------------------------
    v_warp = v_orig.copy()
    z_span = bounds_max[2] - bounds_min[2]
    z_norm = (v_warp[:, 2] - center[2]) / (z_span / 2.0)  # -1 to +1

    # Quadratic bow sag in X direction (1.2 mm peak sag at ends)
    warp_disp_x = 1.20 * (z_norm ** 2)
    v_warp[:, 0] += warp_disp_x

    warp_mesh = trimesh.Trimesh(vertices=v_warp, faces=faces, process=True)
    warp_path = out_dir / "part_warped_flange.stl"
    warp_mesh.export(str(warp_path))
    generated_files["warped"] = str(warp_path.resolve())
    print(f"[+] Wrote {warp_path.name}")

    return generated_files


if __name__ == "__main__":
    generate_benchmark_suite()
