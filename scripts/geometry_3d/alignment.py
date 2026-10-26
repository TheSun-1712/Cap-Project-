"""
6-DoF Rigid Body Spatial Alignment Engine.
Combines Principal Component Analysis (PCA) coarse alignment with candidate disambiguation
and robust Huber Point-to-Plane Iterative Closest Point (ICP) fine registration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class AlignmentResult:
    """Stores the outcome of 6-DoF spatial registration."""
    transformation_matrix: np.ndarray  # 4x4 SE(3) matrix [R, t; 0, 1]
    aligned_points: np.ndarray         # (N, 3) aligned test points
    aligned_normals: np.ndarray        # (N, 3) aligned test normals
    rmse: float                        # Root mean square residual (mm)
    iterations: int                    # ICP iterations taken
    converged: bool                    # Convergence flag


def align_point_clouds(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: Optional[np.ndarray] = None,
    source_normals: Optional[np.ndarray] = None,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
    huber_delta: float = 0.4,
) -> AlignmentResult:
    """
    Aligns source point cloud (Test Part) to target point cloud (Golden CAD Part).
    Returns the optimal 4x4 transformation matrix and aligned points.
    """
    src = source_pts.copy()
    tgt = target_pts.copy()

    tgt_tree = cKDTree(tgt)

    if target_normals is None:
        target_normals = _estimate_normals(tgt)
    if source_normals is None:
        source_normals = _estimate_normals(src)

    src_norm = source_normals.copy()
    tgt_norm = target_normals.copy()

    # Step 1: Coarse Registration
    # Check if parts are already in same approximate coordinate frame (Identity pose test)
    dists_init, _ = tgt_tree.query(src[:2000], k=1)
    mean_dist_init = np.mean(dists_init)

    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    # Candidate coarse transforms
    candidates = []

    # Candidate 0: Centroid translation only (preserving current orientation)
    T_ident = np.eye(4, dtype=np.float64)
    T_ident[:3, 3] = tgt_center - src_center
    candidates.append(T_ident)

    # PCA axes candidates
    src_centered = src - src_center
    tgt_centered = tgt - tgt_center
    cov_src = np.cov(src_centered, rowvar=False)
    cov_tgt = np.cov(tgt_centered, rowvar=False)

    _, evec_src = np.linalg.eigh(cov_src)
    _, evec_tgt = np.linalg.eigh(cov_tgt)
    evec_src = evec_src[:, ::-1]
    evec_tgt = evec_tgt[:, ::-1]

    sign_flips = [
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]

    for S in sign_flips:
        R_cand = (evec_tgt @ S) @ evec_src.T
        if np.linalg.det(R_cand) < 0:
            continue
        T_cand = np.eye(4, dtype=np.float64)
        T_cand[:3, :3] = R_cand
        T_cand[:3, 3] = tgt_center - R_cand @ src_center
        candidates.append(T_cand)

    # Pick the best coarse candidate
    sample_size = min(1500, len(src))
    sub_idx = np.random.choice(len(src), sample_size, replace=False)
    sub_src = src[sub_idx]

    best_T = candidates[0]
    best_dist = float("inf")

    for T_cand in candidates:
        transformed = (sub_src @ T_cand[:3, :3].T) + T_cand[:3, 3]
        dists, _ = tgt_tree.query(transformed, k=1)
        mean_d = float(np.mean(dists))
        if mean_d < best_dist:
            best_dist = mean_d
            best_T = T_cand

    curr_pts = (src @ best_T[:3, :3].T) + best_T[:3, 3]
    curr_normals = src_norm @ best_T[:3, :3].T
    T_cumulative = best_T.copy()

    # Step 2: Fine Robust Point-to-Plane ICP
    prev_rmse = float("inf")
    converged = False
    iterations_run = 0

    # Subsample for ICP speed if points > 25000
    if len(curr_pts) > 25000:
        icp_idx = np.random.choice(len(curr_pts), 25000, replace=False)
    else:
        icp_idx = np.arange(len(curr_pts))

    for it in range(max_iterations):
        iterations_run = it + 1
        pts_sub = curr_pts[icp_idx]
        norms_sub = curr_normals[icp_idx]

        dists, idxs = tgt_tree.query(pts_sub, k=1)
        matched_tgt = tgt[idxs]
        matched_norms = tgt_norm[idxs]

        # Residual: point-to-plane distance
        diff = pts_sub - matched_tgt
        residuals = np.sum(diff * matched_norms, axis=1)

        # Huber loss weighting
        abs_r = np.abs(residuals)
        weights = np.ones_like(residuals)
        outlier_mask = abs_r > huber_delta
        weights[outlier_mask] = huber_delta / (abs_r[outlier_mask] + 1e-9)

        rmse = float(np.sqrt(np.mean(residuals**2)))
        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            break
        prev_rmse = rmse

        # Build Point-to-Plane Linear System: J^T W J xi = -J^T W r
        cross_prod = np.cross(pts_sub, matched_norms)  # (M, 3)
        J = np.hstack([cross_prod, matched_norms])       # (M, 6)

        W = weights[:, np.newaxis]
        JtW = (J * W).T                                  # (6, M)
        A = JtW @ J                                      # (6, 6)
        b = -JtW @ residuals                             # (6,)

        # Regularized solve
        A += 1e-5 * np.eye(6)
        try:
            xi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Lie algebra exp map for small twist xi = [w, v]
        w = xi[:3]
        v = xi[3:]
        theta = np.linalg.norm(w)
        if theta > 1e-9:
            axis = w / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_step = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        else:
            R_step = np.eye(3)

        t_step = v

        # Update all points and cumulative transform
        curr_pts = (curr_pts @ R_step.T) + t_step
        curr_normals = curr_normals @ R_step.T

        T_step = np.eye(4)
        T_step[:3, :3] = R_step
        T_step[:3, 3] = t_step
        T_cumulative = T_step @ T_cumulative

        if np.linalg.norm(xi) < tolerance:
            converged = True
            break

    return AlignmentResult(
        transformation_matrix=T_cumulative,
        aligned_points=curr_pts,
        aligned_normals=curr_normals,
        rmse=float(rmse),
        iterations=iterations_run,
        converged=converged,
    )


def align_meshes(source_geo, target_geo) -> AlignmentResult:
    """Convenience helper to align two GeometryData objects."""
    return align_point_clouds(
        source_pts=source_geo.points,
        target_pts=target_geo.points,
        source_normals=source_geo.point_normals,
        target_normals=target_geo.point_normals,
    )


def _estimate_normals(pts: np.ndarray, k: int = 15) -> np.ndarray:
    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=min(k, len(pts)))
    normals = np.zeros_like(pts)
    for i, idx in enumerate(idxs):
        cov = np.cov(pts[idx] - pts[idx].mean(axis=0), rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        normals[i] = evecs[:, 0]
    return normals
