"""
Signed Surface Metrology and Deviation Engine for Open-Shell CAD and Manufactured Parts.
Enforces point-to-plane normal projection to measure true surface deviations (dents, burrs, warp)
independent of tangential sampling discretizations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class MetrologyResult:
    """Quantitative metrology comparison results."""
    signed_deviations: np.ndarray    # (N,) signed deviations in mm (negative = dent/hole, positive = burr/excess)
    absolute_deviations: np.ndarray  # (N,) |s_i| in mm
    max_deviation: float             # Peak flaw magnitude in mm
    max_positive: float              # Peak excess material (burr/flash) in mm
    max_negative: float              # Peak missing material (dent/cavity) in mm
    rmse: float                      # Global Root Mean Square Error in mm
    mae: float                       # Mean Absolute Error in mm
    defect_ratio: float              # Ratio of surface points exceeding tolerance
    excess_ratio: float              # Ratio of points with s_i > +tolerance
    deficit_ratio: float             # Ratio of points with s_i < -tolerance
    tolerance: float                 # Applied tolerance band (+/- tau in mm)
    verdict: str                     # "PASS" or "FAIL"
    verdict_reasons: list            # Specific reasons if FAIL


def compute_signed_deviation(
    test_points: np.ndarray,
    ref_points: np.ndarray,
    ref_normals: np.ndarray,
    test_normals: Optional[np.ndarray] = None,
    tolerance: float = 0.20,
    max_peak_tolerance: Optional[float] = None,
    max_defect_ratio_tolerance: float = 0.015,
    normal_consistency_angle_deg: float = 60.0,
) -> MetrologyResult:
    """
    Computes signed surface deviations using point-to-plane normal projection:
        s_i = (p_i - q_nn) . n_nn
    This isolates true surface normal displacement while eliminating in-plane sampling spacing noise.
    """
    if max_peak_tolerance is None:
        max_peak_tolerance = tolerance * 2.5

    ref_tree = cKDTree(ref_points)
    num_pts = len(test_points)

    cos_thresh = np.cos(np.radians(normal_consistency_angle_deg))

    # Fast vectorized query for k-nearest neighbors
    k_query = 5
    dists, idxs = ref_tree.query(test_points, k=k_query)

    if test_normals is not None:
        # Extract candidate normals: (N, k, 3)
        cand_normals = ref_normals[idxs]  # (N, k, 3)
        t_norms_expanded = test_normals[:, np.newaxis, :]  # (N, 1, 3)
        
        # Dot product: (N, k)
        dots = np.abs(np.sum(cand_normals * t_norms_expanded, axis=2))
        
        # Find first index >= cos_thresh, or fallback to k=0
        valid_mask = dots >= cos_thresh
        best_k = np.zeros(num_pts, dtype=np.int32)
        for ki in range(k_query):
            unassigned = best_k == 0
            has_valid = valid_mask[:, ki] & unassigned
            best_k[has_valid] = ki

        chosen_ref_idx = idxs[np.arange(num_pts), best_k]
    else:
        chosen_ref_idx = idxs[:, 0]

    chosen_ref_pts = ref_points[chosen_ref_idx]
    chosen_ref_norms = ref_normals[chosen_ref_idx]

    # True point-to-plane signed deviation
    diffs = test_points - chosen_ref_pts
    signed_devs = np.sum(diffs * chosen_ref_norms, axis=1)

    abs_devs = np.abs(signed_devs)
    raw_max_dev = float(np.max(abs_devs))
    # Robust peak flaw filter (99.8th percentile) to reject isolated single-point knife-edge artifacts
    robust_peak_dev = float(np.percentile(abs_devs, 99.8))
    max_dev = robust_peak_dev if robust_peak_dev > 0.10 else raw_max_dev
    max_pos = float(np.max(signed_devs))
    max_neg = float(np.min(signed_devs))
    rmse = float(np.sqrt(np.mean(signed_devs**2)))
    mae = float(np.mean(abs_devs))

    defective_mask = abs_devs > tolerance
    excess_mask = signed_devs > tolerance
    deficit_mask = signed_devs < -tolerance

    defect_ratio = float(np.mean(defective_mask))
    excess_ratio = float(np.mean(excess_mask))
    deficit_ratio = float(np.mean(deficit_mask))

    # Tolerance decision logic
    reasons = []
    if robust_peak_dev >= max_peak_tolerance:
        reasons.append(f"Peak deviation ({robust_peak_dev:.3f} mm) exceeds allowable peak limit ({max_peak_tolerance:.3f} mm)")
    if defect_ratio >= max_defect_ratio_tolerance:
        reasons.append(f"Defective surface area ({defect_ratio*100:.2f}%) exceeds allowable ({max_defect_ratio_tolerance*100:.2f}%)")

    verdict = "FAIL" if len(reasons) > 0 else "PASS"

    return MetrologyResult(
        signed_deviations=signed_devs,
        absolute_deviations=abs_devs,
        max_deviation=max_dev,
        max_positive=max_pos,
        max_negative=max_neg,
        rmse=rmse,
        mae=mae,
        defect_ratio=defect_ratio,
        excess_ratio=excess_ratio,
        deficit_ratio=deficit_ratio,
        tolerance=tolerance,
        verdict=verdict,
        verdict_reasons=reasons,
    )
