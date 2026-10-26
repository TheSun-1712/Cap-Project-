"""
Automated Verification and Regression Test Suite for 3D CAD Metrology Pipeline.
Tests end-to-end ingestion, 6-DoF alignment, signed deviation calculation,
GD&T feature checks, and DBSCAN defect clustering on 0101BA302380N_010 XYZ.stp.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.geometry_3d import (
    load_3d_model,
    align_point_clouds,
    compute_signed_deviation,
    inspect_gdt_features,
    cluster_defect_sites,
)


class Test3DInspectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_path = "0101BA302380N_010 XYZ.stp"
        print(f"\n[*] Loading master reference CAD model ({cls.step_path})...")
        cls.ref = load_3d_model(cls.step_path, num_samples=30000)

    def test_01_step_ingestion_dimensions(self):
        """Verify STEP AP203 B-Rep parsing, coordinate zero-centering, and bounding envelope."""
        self.assertEqual(self.ref.format, "step")
        self.assertGreater(len(self.ref.points), 20000)
        self.assertGreater(len(self.ref.vertices), 5000)

        # Dimensions should be roughly ~18 mm x 52 mm x 154 mm
        dx, dy, dz = self.ref.dimensions
        self.assertAlmostEqual(dx, 18.2, delta=1.5)
        self.assertAlmostEqual(dy, 51.8, delta=2.0)
        self.assertAlmostEqual(dz, 153.5, delta=3.0)

        # Assembly offset should reflect CATIA vehicle coordinate position
        self.assertAlmostEqual(self.ref.assembly_offset[0], 1612.5, delta=5.0)
        self.assertAlmostEqual(self.ref.assembly_offset[1], -569.6, delta=5.0)
        self.assertAlmostEqual(self.ref.assembly_offset[2], 1152.7, delta=5.0)
        print("  [+] test_01_step_ingestion_dimensions passed")

    def test_02_golden_part_pass_verdict(self):
        """Verify that comparing nominal model against itself produces PASS verdict."""
        test = load_3d_model("samples_3d/golden_part.stl", num_samples=30000)
        align_res = align_point_clouds(test.points, self.ref.points, target_normals=self.ref.point_normals)
        metro = compute_signed_deviation(align_res.aligned_points, self.ref.points, self.ref.point_normals, align_res.aligned_normals, tolerance=0.20)
        gdt = inspect_gdt_features(align_res.aligned_points, assembly_offset=self.ref.assembly_offset)

        self.assertLess(align_res.rmse, 0.08)
        self.assertLess(metro.rmse, 0.08)
        self.assertLess(metro.defect_ratio, 0.015)
        self.assertEqual(gdt.overall_verdict, "PASS")
        print("  [+] test_02_golden_part_pass_verdict passed (RMSE < 0.08 mm, GD&T PASS)")

    def test_03_dent_detection(self):
        """Verify that part with dent is flagged as REJECT and cluster is isolated."""
        test = load_3d_model("samples_3d/part_with_dent.stl", num_samples=30000)
        align_res = align_point_clouds(test.points, self.ref.points, target_normals=self.ref.point_normals)
        metro = compute_signed_deviation(align_res.aligned_points, self.ref.points, self.ref.point_normals, align_res.aligned_normals, tolerance=0.20)
        clusters = cluster_defect_sites(align_res.aligned_points, metro.signed_deviations, tolerance=0.20, assembly_offset=self.ref.assembly_offset)

        self.assertGreater(metro.max_deviation, 0.50)
        self.assertGreater(len(clusters), 0)
        # Primary defect should be dent / underfill
        self.assertIn("DENT", clusters[0].defect_type)
        print(f"  [+] test_03_dent_detection passed (Flaw: {clusters[0].defect_type}, Peak: {clusters[0].peak_deviation:.3f} mm)")

    def test_04_burr_detection(self):
        """Verify that part with raised flash is flagged with positive signed deviation."""
        test = load_3d_model("samples_3d/part_with_burr.stl", num_samples=30000)
        align_res = align_point_clouds(test.points, self.ref.points, target_normals=self.ref.point_normals)
        metro = compute_signed_deviation(align_res.aligned_points, self.ref.points, self.ref.point_normals, align_res.aligned_normals, tolerance=0.20)
        clusters = cluster_defect_sites(align_res.aligned_points, metro.signed_deviations, tolerance=0.20, assembly_offset=self.ref.assembly_offset)

        self.assertGreaterEqual(metro.max_positive, 0.45)
        self.assertGreater(len(clusters), 0)
        print(f"  [+] test_04_burr_detection passed (Max Positive Excess: {metro.max_positive:+.3f} mm)")

    def test_05_warping_detection(self):
        """Verify that warped flange triggers high defect ratio and warping cluster."""
        test = load_3d_model("samples_3d/part_warped_flange.stl", num_samples=30000)
        align_res = align_point_clouds(test.points, self.ref.points, target_normals=self.ref.point_normals)
        metro = compute_signed_deviation(align_res.aligned_points, self.ref.points, self.ref.point_normals, align_res.aligned_normals, tolerance=0.20)
        clusters = cluster_defect_sites(align_res.aligned_points, metro.signed_deviations, tolerance=0.20, assembly_offset=self.ref.assembly_offset)

        self.assertEqual(metro.verdict, "FAIL")
        self.assertGreater(metro.defect_ratio, 0.20)  # > 20% of surface out of tolerance
        print(f"  [+] test_05_warping_detection passed (Defect Surface Area: {metro.defect_ratio*100:.1f}%)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
