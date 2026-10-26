"""
3D Spatial Defect Clustering and Site Profiling Engine.
Uses DBSCAN density clustering to isolate separate defect regions (Dents, Burrs, Cracks, Warping),
computing 3D centroids, bounding boxes, peak depths, and surface areas.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class DefectCluster:
    cluster_id: int
    defect_type: str                  # "DENT", "BURR / FLASH", "WARPING", "HOLE DEFECT"
    centroid_local: np.ndarray        # [x, y, z] mm in local coordinates
    centroid_world: np.ndarray        # [x, y, z] mm in CATIA assembly coordinates
    peak_deviation: float             # Max |s_i| in mm
    mean_deviation: float             # Mean signed deviation in mm
    bounding_box: np.ndarray          # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    point_count: int                  # Number of points in cluster
    estimated_area_mm2: float         # Surface area in square millimeters


def cluster_defect_sites(
    test_points: np.ndarray,
    signed_deviations: np.ndarray,
    tolerance: float = 0.20,
    eps: float = 2.5,                 # DBSCAN radius in mm
    min_samples: int = 8,             # Minimum cluster size
    total_surface_area_mm2: float = 12000.0,
    assembly_offset: Optional[np.ndarray] = None,
) -> List[DefectCluster]:
    """
    Extracts out-of-tolerance surface points and segments them into distinct 3D physical defect sites.
    """
    outlier_mask = np.abs(signed_deviations) > tolerance
    outlier_pts = test_points[outlier_mask]
    outlier_devs = signed_deviations[outlier_mask]

    if len(outlier_pts) < min_samples:
        return []

    # Apply DBSCAN in 3D Euclidean space
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(outlier_pts)

    unique_labels = set(labels) - {-1}  # Ignore noise (-1)
    clusters = []

    area_per_point = total_surface_area_mm2 / max(1, len(test_points))
    default_offset = assembly_offset if assembly_offset is not None else np.array([1612.54, -569.57, 1152.75])

    for cluster_id, lbl in enumerate(sorted(unique_labels)):
        c_mask = labels == lbl
        pts_c = outlier_pts[c_mask]
        devs_c = outlier_devs[c_mask]

        centroid_loc = np.mean(pts_c, axis=0)
        centroid_w = centroid_loc + default_offset

        b_min = np.min(pts_c, axis=0)
        b_max = np.max(pts_c, axis=0)
        bbox = np.array([b_min, b_max])

        span = b_max - b_min
        peak_dev = float(np.max(np.abs(devs_c)))
        mean_dev = float(np.mean(devs_c))
        pt_count = len(pts_c)
        est_area = pt_count * area_per_point

        # Defect Classification Heuristics
        # Check if near countersunk holes (Z ~ 1092 or 1112 in world, or span in local)
        world_z = centroid_w[2]
        if (abs(world_z - 1092.06) < 6.0 or abs(world_z - 1112.06) < 6.0) and peak_dev < 1.0:
            defect_type = "HOLE DEFORMATION"
        elif np.max(span) > 25.0:
            defect_type = "WARPING / GLOBAL SAG"
        elif mean_dev < -0.05:
            defect_type = "DENT / MATERIAL UNDERFILL"
        elif mean_dev > 0.05:
            defect_type = "BURR / MATERIAL EXCESS"
        else:
            defect_type = "SURFACE ANOMALY"

        clusters.append(DefectCluster(
            cluster_id=cluster_id + 1,
            defect_type=defect_type,
            centroid_local=centroid_loc,
            centroid_world=centroid_w,
            peak_deviation=peak_dev,
            mean_deviation=mean_dev,
            bounding_box=bbox,
            point_count=pt_count,
            estimated_area_mm2=round(est_area, 2),
        ))

    # Sort by peak severity descending
    clusters.sort(key=lambda c: c.peak_deviation, reverse=True)
    return clusters
