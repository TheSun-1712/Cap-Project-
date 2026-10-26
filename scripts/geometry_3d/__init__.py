"""
3D Metrology and Geometric Inspection Module for CATIA STEP AP203 CAD and Scanned 3D Models.
"""

from .step_loader import load_step_model, tessellate_step_to_mesh
from .mesh_loader import load_3d_model, GeometryData
from .alignment import align_point_clouds, align_meshes
from .comparator import compute_signed_deviation, MetrologyResult
from .gdt_inspector import inspect_gdt_features, GDTResult
from .clustering import cluster_defect_sites, DefectCluster

__all__ = [
    "load_step_model",
    "tessellate_step_to_mesh",
    "load_3d_model",
    "GeometryData",
    "align_point_clouds",
    "align_meshes",
    "compute_signed_deviation",
    "MetrologyResult",
    "inspect_gdt_features",
    "GDTResult",
    "cluster_defect_sites",
    "DefectCluster",
]
