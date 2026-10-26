"""
Universal 3D Geometry and Mesh Loader.
Handles STL, OBJ, PLY, and point cloud formats, extracting uniform surface samples and normals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class GeometryData:
    """Universal container for 3D geometric representations (CAD, Mesh, Point Cloud)."""
    name: str
    file_path: str
    format: str
    vertices: np.ndarray             # (V, 3)
    faces: Optional[np.ndarray]      # (F, 3) or None
    normals: np.ndarray              # (V, 3) or (P, 3)
    points: np.ndarray               # (N, 3) high-density surface point cloud
    point_normals: np.ndarray        # (N, 3) outward normal vectors
    centroid: np.ndarray             # (3,) geometric center
    assembly_offset: np.ndarray      # (3,) offset if normalized from world CAD frame
    bounds_min: np.ndarray           # (3,)
    bounds_max: np.ndarray           # (3,)
    dimensions: np.ndarray           # (3,) [dx, dy, dz]
    is_watertight: bool = False

    def get_local_points(self) -> np.ndarray:
        """Return zero-centered points for registration."""
        return self.points - self.centroid

    def get_world_points(self) -> np.ndarray:
        """Return points in original world coordinate frame."""
        return self.points + self.assembly_offset


def load_3d_model(
    file_path: Union[str, Path],
    num_samples: int = 100000,
    normalize_origin: bool = True,
) -> GeometryData:
    """
    Universal loader for 3D models (STEP, STL, OBJ, PLY).
    Dispatches to step_loader for .stp/.step files, and trimesh for polygonal formats.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"3D Model not found: {file_path}")

    ext = path.suffix.lower()

    if ext in [".stp", ".step"]:
        from .step_loader import load_step_model
        return load_step_model(path, num_samples=num_samples, normalize_origin=normalize_origin)

    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"Empty scene in {path}")
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32) if hasattr(mesh, "faces") and mesh.faces is not None else None
    
    # Calculate bounds & dimensions
    b_min = vertices.min(axis=0)
    b_max = vertices.max(axis=0)
    dimensions = b_max - b_min
    centroid = (b_min + b_max) / 2.0

    # Sample surface point cloud and normals
    if faces is not None and len(faces) > 0:
        points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
        point_normals = mesh.face_normals[face_idx]
        vertex_normals = mesh.vertex_normals
        is_watertight = bool(mesh.is_watertight)
    else:
        # Point cloud fallback
        if len(vertices) > num_samples:
            idx = np.random.choice(len(vertices), num_samples, replace=False)
            points = vertices[idx]
        else:
            points = vertices
        # Estimate normals using k-nearest neighbors
        point_normals = _estimate_normals(points)
        vertex_normals = point_normals
        is_watertight = False

    assembly_offset = np.zeros(3, dtype=np.float64)
    if normalize_origin:
        # Check if model has a large assembly coordinate offset (> 100mm from origin)
        if np.linalg.norm(centroid) > 100.0:
            assembly_offset = centroid.copy()
            vertices = vertices - centroid
            points = points - centroid
            b_min = b_min - centroid
            b_max = b_max - centroid
            centroid = np.zeros(3, dtype=np.float64)

    return GeometryData(
        name=path.stem,
        file_path=str(path.resolve()),
        format=ext.replace(".", ""),
        vertices=vertices,
        faces=faces,
        normals=vertex_normals,
        points=points,
        point_normals=point_normals,
        centroid=centroid,
        assembly_offset=assembly_offset,
        bounds_min=b_min,
        bounds_max=b_max,
        dimensions=dimensions,
        is_watertight=is_watertight,
    )


def _estimate_normals(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Compute local surface normals using PCA on k-nearest neighbors."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=min(k, len(points)))
    
    normals = np.zeros_like(points)
    for i, idx in enumerate(idxs):
        neighbors = points[idx]
        cov = np.cov(neighbors - neighbors.mean(axis=0), rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]  # Eigenvector corresponding to smallest eigenvalue
        normals[i] = normal

    # Orient normals outward from centroid
    center = points.mean(axis=0)
    radial = points - center
    flip = (normals * radial).sum(axis=1) < 0
    normals[flip] *= -1.0
    return normals
