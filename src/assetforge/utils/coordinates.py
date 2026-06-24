"""
Coordinate normalization for photogrammetry pipelines.

Prevents float32 precision issues with large coordinates (UTM, GPS data)
by normalizing scene coordinates to local frame during processing,
then restoring geographic projection at export.

Based on research findings:
    OpenMVS uses single-precision floats internally, causing catastrophic
    precision loss when coordinates exceed ~1km range. Normalizing to a
    local origin at the start prevents this, with inverse transformation
    applied at final export.
"""

import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger("assetforge")


@dataclass
class CoordinateTransform:
    """Stores transformation for coordinate normalization/restoration.

    Attributes:
        center: Original centroid of the scene (x, y, z)
        scale: Applied scale factor
        matrix: Full 4x4 similarity transformation matrix
    """

    center: tuple[float, float, float]
    scale: float
    matrix: list[list[float]]  # 4x4

    def save(self, path: str | Path) -> None:
        """Save transformation to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.debug(f"Coordinate transform saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CoordinateTransform":
        """Load transformation from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_scene_centroid(sfm_data_path: str | Path) -> tuple[float, float, float]:
    """Compute the centroid of all camera positions in sfm_data.json.

    Args:
        sfm_data_path: Path to OpenMVG sfm_data.json

    Returns:
        (cx, cy, cz) centroid of all camera centers

    Raises:
        FileNotFoundError: If sfm_data.json doesn't exist
        ValueError: If no cameras found in sfm_data
    """
    sfm_data_path = Path(sfm_data_path)
    if not sfm_data_path.exists():
        raise FileNotFoundError(f"sfm_data.json not found: {sfm_data_path}")

    with open(sfm_data_path) as f:
        data = json.load(f)

    # Extract extrinsics (camera poses)
    views = data.get("views", [])
    extrinsics = data.get("extrinsics", [])

    if not extrinsics:
        raise ValueError("No extrinsics (camera poses) found in sfm_data")

    # Build mapping: view key to extrinsic
    centers = []
    for ext in extrinsics:
        # The extrinsic value is a 4x4 rotation+translation matrix
        # Camera center = -R^T * t
        pose = ext.get("value", {}).get("rotation", [])
        if len(pose) != 16:
            continue

        # Convert to numpy 4x4
        R = np.array(pose).reshape(4, 4)
        # Camera center: C = -R[:3, :3].T * R[:3, 3]
        C = -R[:3, :3].T @ R[:3, 3]
        centers.append(C)

    if not centers:
        # Fall back to structure points if no camera poses
        structure = data.get("structure", {})
        for point_key, point_data in structure.items():
            X = point_data.get("value", {}).get("X", [])
            if len(X) == 3:
                centers.append(np.array(X, dtype=np.float64))

    if not centers:
        raise ValueError("No camera poses or structure points found in sfm_data")

    centers_arr = np.array(centers)
    centroid = tuple(centers_arr.mean(axis=0).tolist())
    logger.info(
        f"Scene centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
    )
    return centroid


def normalize_sfm_data(
    sfm_data_path: str | Path,
    output_path: str | Path,
) -> CoordinateTransform:
    """Normalize OpenMVG sfm_data.json coordinates to local frame.

    Reads the sfm_data, computes centroid and extent, then writes a
    normalized version with all coordinates shifted to origin and scaled.

    Args:
        sfm_data_path: Input OpenMVG sfm_data.json
        output_path: Output path for normalized sfm_data.json

    Returns:
        CoordinateTransform that can be used to restore original coordinates
    """
    sfm_data_path = Path(sfm_data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not sfm_data_path.exists():
        logger.warning(
            f"sfm_data.json not found at {sfm_data_path}, skipping normalization"
        )
        return CoordinateTransform(
            center=(0.0, 0.0, 0.0),
            scale=1.0,
            matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )

    with open(sfm_data_path) as f:
        data = json.load(f)

    # Compute centroid from camera poses
    center = compute_scene_centroid(sfm_data_path)
    cx, cy, cz = center

    # Compute bounding box diagonal for scaling
    all_points = []
    extrinsics = data.get("extrinsics", [])
    for ext in extrinsics:
        pose = ext.get("value", {}).get("rotation", [])
        if len(pose) == 16:
            R = np.array(pose).reshape(4, 4)
            C = -R[:3, :3].T @ R[:3, 3]
            all_points.append(C)

    structure = data.get("structure", {})
    for point_key, point_data in structure.items():
        X = point_data.get("value", {}).get("X", [])
        if len(X) == 3:
            all_points.append(np.array(X, dtype=np.float64))

    if all_points:
        pts = np.array(all_points)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)
        scale = 2.0 / diagonal if diagonal > 0 else 1.0
    else:
        scale = 1.0

    logger.info(
        f"Normalizing scene: center=({cx:.2f},{cy:.2f},{cz:.2f}), scale={scale:.6f}"
    )

    # Modify extrinsics (camera poses)
    for ext in extrinsics:
        pose = ext.get("value", {}).get("rotation", [])
        if len(pose) == 16:
            R = np.array(pose).reshape(4, 4)
            # Camera center before normalization
            C = -R[:3, :3].T @ R[:3, 3]
            # Normalize center
            C_norm = (C - np.array([cx, cy, cz])) * scale
            # New translation: t = -R * C_norm
            t_new = -R[:3, :3] @ C_norm
            R[:3, 3] = t_new
            ext["value"]["rotation"] = R.flatten().tolist()

    # Modify structure points (3D landmarks)
    for point_key, point_data in structure.items():
        X = point_data.get("value", {}).get("X", [])
        if len(X) == 3:
            pt = np.array(X, dtype=np.float64)
            pt_norm = (pt - np.array([cx, cy, cz])) * scale
            point_data["value"]["X"] = pt_norm.tolist()

    # Write normalized data
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Build transformation matrix
    transform_matrix = [
        [scale, 0, 0, -cx * scale],
        [0, scale, 0, -cy * scale],
        [0, 0, scale, -cz * scale],
        [0, 0, 0, 1],
    ]

    transform = CoordinateTransform(
        center=(cx, cy, cz),
        scale=scale,
        matrix=transform_matrix,
    )

    # Save transform alongside normalized data
    transform_path = output_path.with_name("coordinate_transform.json")
    transform.save(transform_path)

    logger.info(f"Normalized sfm_data written to {output_path}")
    return transform


def restore_coordinates(
    mesh_path: str | Path,
    transform: CoordinateTransform,
    output_path: str | Path,
) -> Path:
    """Apply inverse transformation to restore geographic coordinates on a mesh.

    For PLY/OBJ meshes, this transforms all vertex positions back to
    the original coordinate space.

    Args:
        mesh_path: Path to the mesh file (PLY or OBJ)
        transform: CoordinateTransform from normalization step
        output_path: Path for the restored mesh file

    Returns:
        Path to the restored mesh file
    """
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        logger.warning(
            f"Mesh not found at {mesh_path}, skipping coordinate restoration"
        )
        return mesh_path

    if transform.scale == 1.0 and all(c == 0.0 for c in transform.center):
        logger.info("No transformation needed, copying mesh as-is")
        import shutil

        shutil.copy2(mesh_path, output_path)
        return output_path

    ext = mesh_path.suffix.lower()

    if ext == ".ply":
        _restore_ply_coordinates(mesh_path, transform, output_path)
    elif ext == ".obj":
        _restore_obj_coordinates(mesh_path, transform, output_path)
    else:
        # Unsupported format, copy as-is
        logger.warning(
            f"Unsupported mesh format '{ext}' for coordinate restoration, copying"
        )
        import shutil

        shutil.copy2(mesh_path, output_path)

    logger.info(f"Coordinate restoration complete → {output_path}")
    return output_path


def _restore_ply_coordinates(
    mesh_path: Path, transform: CoordinateTransform, output_path: Path
) -> None:
    """Restore coordinates in a PLY file by transforming vertex positions."""
    cx, cy, cz = transform.center
    inv_scale = 1.0 / transform.scale

    with open(mesh_path) as f:
        content = f.read()

    lines = content.split("\n")
    header_end = 0
    vertex_count = 0
    vertex_props = []
    in_header = True
    vertex_start_idx = 0

    for i, line in enumerate(lines):
        if in_header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append(parts[-1])
            elif line == "end_header":
                header_end = i
                vertex_start_idx = i + 1
                in_header = False
        else:
            break

    if vertex_count == 0 or vertex_start_idx == 0:
        logger.warning("Could not parse PLY header, copying mesh as-is")
        import shutil

        shutil.copy2(mesh_path, output_path)
        return

    # Determine which columns are x, y, z
    x_idx = vertex_props.index("x") if "x" in vertex_props else -1
    y_idx = vertex_props.index("y") if "y" in vertex_props else -1
    z_idx = vertex_props.index("z") if "z" in vertex_props else -1

    if x_idx < 0 or y_idx < 0 or z_idx < 0:
        logger.warning("Could not find vertex coordinates in PLY, copying as-is")
        import shutil

        shutil.copy2(mesh_path, output_path)
        return

    out_lines = lines[:vertex_start_idx]

    for i in range(vertex_start_idx, vertex_start_idx + vertex_count):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) < max(x_idx, y_idx, z_idx) + 1:
            out_lines.append(lines[i])
            continue
        try:
            x = float(parts[x_idx]) * inv_scale + cx
            y = float(parts[y_idx]) * inv_scale + cy
            z = float(parts[z_idx]) * inv_scale + cz
            parts[x_idx] = f"{x:.6f}"
            parts[y_idx] = f"{y:.6f}"
            parts[z_idx] = f"{z:.6f}"
            out_lines.append(" ".join(parts))
        except (ValueError, IndexError):
            out_lines.append(lines[i])

    out_lines.extend(lines[vertex_start_idx + vertex_count :])

    with open(output_path, "w") as f:
        f.write("\n".join(out_lines))


def _restore_obj_coordinates(
    mesh_path: Path, transform: CoordinateTransform, output_path: Path
) -> None:
    """Restore coordinates in an OBJ file by transforming vertex positions."""
    cx, cy, cz = transform.center
    inv_scale = 1.0 / transform.scale

    with open(mesh_path) as f:
        content = f.read()

    lines = content.split("\n")
    out_lines = []

    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x = float(parts[1]) * inv_scale + cx
                    y = float(parts[2]) * inv_scale + cy
                    z = float(parts[3]) * inv_scale + cz
                    out_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
                    continue
                except ValueError:
                    pass
        out_lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(out_lines))
