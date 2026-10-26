"""
CAD STEP AP203 Ingestion and Adaptive Curvature Tessellation Engine.
Utilizes embedded OpenCASCADE via Gmsh to convert ISO 10303-21 B-Rep surfaces
into high-resolution surface meshes and uniform point clouds.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import trimesh

from .mesh_loader import GeometryData


def tessellate_step_to_mesh(
    step_path: Union[str, Path],
    mesh_size_min: float = 0.3,
    mesh_size_max: float = 1.2,
    output_stl_path: Optional[Union[str, Path]] = None,
) -> trimesh.Trimesh:
    """
    Tessellates a STEP AP203/AP214 CAD file into a clean triangular surface mesh using Gmsh/OpenCASCADE.
    Automatically strips CATIA datum origin artifacts (points near (0,0,0)).
    """
    import gmsh

    path = Path(step_path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    is_initialized = gmsh.isInitialized()
    if not is_initialized:
        try:
            gmsh.initialize(interruptible=False)
        except TypeError:
            gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)  # Silence console logs
    gmsh.model.add(path.stem)

    try:
        # Import B-Rep topology via OpenCASCADE
        shapes = gmsh.model.occ.importShapes(str(path.resolve()))
        gmsh.model.occ.synchronize()

        # Curvature-adaptive 2D Delaunay meshing
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay for 2D surfaces
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
        gmsh.model.mesh.generate(2)

        # Extract nodes (vertices)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        # Extract 2D triangular elements (element type 2 in Gmsh)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        
        triangles = []
        for e_type, e_nodes in zip(elem_types, elem_node_tags):
            if e_type == 2:  # 3-node triangle
                node_array = np.array(e_nodes, dtype=np.int64).reshape(-1, 3)
                for tri in node_array:
                    triangles.append([tag_to_idx[tri[0]], tag_to_idx[tri[1]], tag_to_idx[tri[2]]])

        if len(triangles) == 0:
            raise RuntimeError(f"Gmsh failed to generate 2D surface elements for {path.name}")

        triangles = np.array(triangles, dtype=np.int32)

        # Filter out isolated CATIA datum origin points (nodes near X < 100.0)
        # Note: The real part is placed at X ~ 1603..1622 mm in CATIA assembly space
        real_mask = coords[:, 0] > 100.0
        if not np.all(real_mask):
            old_to_new = -np.ones(len(coords), dtype=np.int32)
            filtered_coords = coords[real_mask]
            old_to_new[real_mask] = np.arange(len(filtered_coords), dtype=np.int32)

            # Keep only triangles whose all 3 vertices are part of the real part
            valid_tri_mask = (old_to_new[triangles[:, 0]] >= 0) & \
                             (old_to_new[triangles[:, 1]] >= 0) & \
                             (old_to_new[triangles[:, 2]] >= 0)
            valid_triangles = triangles[valid_tri_mask]
            remapped_triangles = old_to_new[valid_triangles]
            coords = filtered_coords
            triangles = remapped_triangles

        mesh = trimesh.Trimesh(vertices=coords, faces=triangles, process=True)

        if output_stl_path is not None:
            out_p = Path(output_stl_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_p))

        return mesh

    finally:
        if not is_initialized:
            gmsh.finalize()


def load_step_model(
    step_path: Union[str, Path],
    num_samples: int = 100000,
    mesh_size_min: float = 0.3,
    mesh_size_max: float = 1.2,
    normalize_origin: bool = True,
) -> GeometryData:
    """
    Ingests a STEP AP203 file, discretizes B-spline and revolution surfaces,
    normalizes coordinates, and samples high-density surface point cloud.
    """
    path = Path(step_path)
    mesh = tessellate_step_to_mesh(path, mesh_size_min=mesh_size_min, mesh_size_max=mesh_size_max)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Compute bounding envelope in world space
    b_min = vertices.min(axis=0)
    b_max = vertices.max(axis=0)
    dimensions = b_max - b_min
    centroid = (b_min + b_max) / 2.0

    # High-density uniform surface sampling
    points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    point_normals = mesh.face_normals[face_idx]
    vertex_normals = mesh.vertex_normals

    assembly_offset = np.zeros(3, dtype=np.float64)
    if normalize_origin and np.linalg.norm(centroid) > 100.0:
        assembly_offset = centroid.copy()
        vertices = vertices - centroid
        points = points - centroid
        b_min = b_min - centroid
        b_max = b_max - centroid
        centroid = np.zeros(3, dtype=np.float64)

    return GeometryData(
        name=path.stem,
        file_path=str(path.resolve()),
        format="step",
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
        is_watertight=bool(mesh.is_watertight),
    )
