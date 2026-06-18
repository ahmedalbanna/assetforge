"""
Gaussian Splatting backend — trains 3D Gaussians and extracts meshes.

Provides:
- Camera pose estimation (COLMAP)
- 3DGS training
- Mesh extraction (SuGaR / MILo)
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")


def estimate_poses(image_dir: str, output_dir: str) -> Path:
    """Estimate camera poses using COLMAP.

    Args:
        image_dir: Directory of input images
        output_dir: Output directory for COLMAP data

    Returns:
        Path to the COLMAP output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Estimating camera poses with COLMAP...")

    try:
        # COLMAP feature extraction
        subprocess.run(
            [
                "colmap",
                "feature_extractor",
                "--database_path",
                str(out / "database.db"),
                "--image_path",
                image_dir,
                "--SiftExtraction.use_gpu",
                "1",
            ],
            check=True,
            capture_output=True,
        )

        # COLMAP exhaustive matching
        subprocess.run(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(out / "database.db"),
                "--SiftMatching.use_gpu",
                "1",
            ],
            check=True,
            capture_output=True,
        )

        # COLMAP sparse mapper
        sparse_dir = out / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        subprocess.run(
            [
                "colmap",
                "mapper",
                "--database_path",
                str(out / "database.db"),
                "--image_path",
                image_dir,
                "--output_path",
                str(sparse_dir),
            ],
            check=True,
            capture_output=True,
        )

        # Convert to text format for 3DGS
        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(sparse_dir / "0"),
                "--output_path",
                str(sparse_dir / "0" / "points3D.txt"),
                "--output_type",
                "TXT",
            ],
            check=True,
            capture_output=True,
        )

        logger.info(f"COLMAP pose estimation complete → {out}")

    except (subprocess.FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "COLMAP is required for camera pose estimation.\n"
            "Install from: https://github.com/colmap/colmap\n"
            f"Error: {e}"
        )

    return out


def train_splats(
    colmap_dir: str,
    output_path: str,
    quality: str = "medium",
    max_splats: int = 1_000_000,
) -> Path:
    """Train 3D Gaussian Splatting from COLMAP output.

    Args:
        colmap_dir: COLMAP output directory
        output_path: Path to save trained splats (PLY)
        quality: Training quality ('low', 'medium', 'high')
        max_splats: Maximum number of Gaussians

    Returns:
        Path to the trained splat file
    """
    logger.info("Training 3D Gaussian Splatting...")

    iterations = {"low": 7000, "medium": 15000, "high": 30000}
    n_iter = iterations.get(quality, 15000)

    try:
        # Run 3D Gaussian Splatting trainer
        subprocess.run(
            [
                "python",
                "-m",
                "gaussian_splatting.train",
                "-s",
                colmap_dir,
                "-m",
                str(Path(output_path).parent),
                "--iterations",
                str(n_iter),
                "--max_gaussians",
                str(max_splats),
            ],
            check=True,
            capture_output=True,
        )

        logger.info(f"3DGS training complete → {output_path}")

    except (subprocess.FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "3D Gaussian Splatting trainer required.\n"
            "Install from: https://github.com/graphdeco-inria/gaussian-splatting\n"
            f"Error: {e}"
        )

    return Path(output_path)


def extract_mesh_milo(splat_path: str, output_path: str) -> Path:
    """Extract surface mesh from Gaussians using MILo.

    MILo (Mesh-In-the-Loop Gaussian Splatting) integrates mesh
    extraction as a differentiable loop during training.

    Args:
        splat_path: Path to trained splat file (PLY)
        output_path: Path to save extracted mesh (OBJ)

    Returns:
        Path to extracted mesh
    """
    logger.info("Extracting mesh using MILo...")

    try:
        subprocess.run(
            [
                "python",
                "-m",
                "milo.extract_mesh",
                "--splat",
                splat_path,
                "--output",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "MILo extraction requires the MILo package.\n"
            "See: https://github.com/Anttwo/MILo\n"
            f"Error: {e}"
        )

    return Path(output_path)


def extract_mesh_sugar(splat_path: str, output_path: str) -> Path:
    """Extract surface mesh from Gaussians using SuGaR.

    SuGaR (Surface-Aligned Gaussian Splatting) enforces Gaussian
    alignment to surfaces for clean mesh extraction.

    Args:
        splat_path: Path to trained splat file (PLY)
        output_path: Path to save extracted mesh (OBJ)

    Returns:
        Path to extracted mesh
    """
    logger.info("Extracting mesh using SuGaR...")

    try:
        subprocess.run(
            [
                "python",
                "-m",
                "sugar.extract_mesh",
                "--splat",
                splat_path,
                "--output",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "SuGaR extraction requires the SuGaR package.\n"
            "See: https://github.com/Anttwo/SuGaR\n"
            f"Error: {e}"
        )

    return Path(output_path)


def compress_ply(ply_path: str, output_path: str, quality: int = 80) -> Path:
    """Compress a 3DGS PLY file to SPZ format using the Rust backend.

    Parses the standard 3D Gaussian Splatting PLY format and compresses
    the structured data using Rust's SPZ compressor.

    Args:
        ply_path: Path to input PLY file
        output_path: Path to output .spz file
        quality: Compression quality (0-100)

    Returns:
        Path to the compressed .spz file
    """
    import struct

    logger.info(f"Compressing {ply_path} (quality={quality})...")

    with open(ply_path, "rb") as f:
        data = f.read()

    # Parse PLY header
    header_end = data.find(b"end_header\n")
    if header_end == -1:
        raise ValueError("Invalid PLY file: no end_header found")
    header = data[:header_end].decode("ascii")
    body = data[header_end + len(b"end_header\n") :]

    # Count vertices
    vertex_count = 0
    for line in header.split("\n"):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            break

    if vertex_count == 0:
        raise ValueError("No vertices found in PLY file")

    # Parse vertex properties
    props = []
    in_vertex = False
    for line in header.split("\n"):
        if line.startswith("element vertex"):
            in_vertex = True
        elif line.startswith("element") and not line.startswith("element vertex"):
            in_vertex = False
        if in_vertex and line.startswith("property"):
            parts = line.split()
            name = parts[-1]
            props.append(name)

    # Assume binary little-endian float format
    stride = len(props) * 4
    expected_size = vertex_count * stride

    if len(body) < expected_size:
        raise ValueError(
            f"PLY data truncated: expected {expected_size} bytes, got {len(body)}"
        )

    # Parse all vertex data
    positions = []
    rotations = []
    scales = []
    opacities = []
    colors = []

    for i in range(vertex_count):
        offset = i * stride
        vert_bytes = body[offset : offset + stride]
        vert_floats = struct.unpack("<" + "f" * len(props), vert_bytes)

        vals = dict(zip(props, vert_floats))

        # Position
        positions.append([vals.get("x", 0.0), vals.get("y", 0.0), vals.get("z", 0.0)])

        # Rotation (quaternion: rot_0 rot_1 rot_2 rot_3)
        rotations.append(
            [
                vals.get("rot_0", 0.0),
                vals.get("rot_1", 0.0),
                vals.get("rot_2", 0.0),
                vals.get("rot_3", 1.0),
            ]
        )

        # Scale
        scales.append(
            [
                vals.get("scale_0", 0.0),
                vals.get("scale_1", 0.0),
                vals.get("scale_2", 0.0),
            ]
        )

        # Opacity
        opacities.append(vals.get("opacity", 0.0))

        # Color (SH DC components)
        colors.append(
            [
                vals.get("f_dc_0", 0.0),
                vals.get("f_dc_1", 0.0),
                vals.get("f_dc_2", 0.0),
            ]
        )

    # Compress via Rust backend
    try:
        from assetforge_pyo3 import compress_splats as compress_rust
    except ImportError:
        logger.warning("Rust backend not available, using ZIP fallback")
        import zipfile

        out = Path(output_path)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(ply_path, "splat.ply")
        return out

    compressed = compress_rust(positions, rotations, scales, opacities, colors, quality)
    output = Path(output_path)
    output.write_bytes(compressed)

    ratio = len(compressed) / len(data) * 100
    logger.info(
        f"Compression complete: {len(data) / 1024:.1f} KB → "
        f"{len(compressed) / 1024:.1f} KB ({ratio:.1f}%)"
    )
    return output
