"""
OpenMVS backend — wraps the OpenMVS multi-view stereo CLI tools.

Provides:
- Dense point cloud reconstruction (DensifyPointCloud)
- Mesh reconstruction (ReconstructMesh)
- Mesh refinement (RefineMesh) — NEW: improves surface detail
- Texture mapping (TextureMesh) with seam leveling control
- Memory management via resolution-level control
- Coordinate normalization integration for float32 precision safety

Based on research insights:
    - float32 precision: OpenMVS uses single-precision internally;
      large coordinates (UTM/GPS) cause catastrophic precision loss.
      Coordinate normalization is essential.
    - Seam leveling: Must be disabled when masks are present to
      prevent black artifacts on masked surfaces.
    - Memory: ReconstructMesh uses Delaunay tetrahedralization which
      consumes excessive RAM. Resolution-level downsamples input.
    - Bug reference: OpenMVS Issue #1158 — passing --ignore-mask-label
      below -1 crashes due to uninitialized rasterer.mask.
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("assetforge")

# OpenMVS commands
OPENMVS_DENSE = "OpenMVS_DensifyPointCloud"
OPENMVS_RECONSTRUCT = "OpenMVS_ReconstructMesh"
OPENMVS_REFINE = "OpenMVS_RefineMesh"
OPENMVS_TEXTURE = "OpenMVS_TextureMesh"


def _check_openmvs():
    """Verify OpenMVS tools are available."""
    for cmd in [OPENMVS_DENSE, OPENMVS_RECONSTRUCT, OPENMVS_TEXTURE]:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            logger.warning(
                f"OpenMVS tool '{cmd}' not found. Install openMVS: "
                f"https://github.com/cdcseacave/openMVS"
            )
            return False
    return True


def _run_cmd(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess command with logging and error handling.

    Args:
        cmd: Command and arguments as a list
        desc: Human-readable description for logging

    Returns:
        CompletedProcess on success

    Raises:
        RuntimeError: If command fails, with descriptive message
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.debug(f"Running: {cmd_str}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=14400)
        return result
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        stdout = e.stdout.decode() if e.stdout else ""
        logger.error(f"OpenMVS command failed [{desc}]")
        if stderr:
            logger.error(f"Stderr: {stderr[:500]}")
        if stdout:
            # Often OpenMVS writes errors to stdout
            error_lines = [
                l
                for l in stdout.split("\n")
                if "error" in l.lower() or "fail" in l.lower() or "nan" in l.lower()
            ]
            for line in error_lines[:5]:
                logger.error(f"  {line}")
        raise RuntimeError(
            f"OpenMVS {desc} failed. "
            f"Check the tool is installed and input data is valid."
        ) from e
    except subprocess.TimeoutExpired:
        logger.error(f"OpenMVS command timed out [{desc}]")
        raise RuntimeError(
            f"OpenMVS {desc} timed out after 4 hours. "
            f"Try reducing the resolution level or simplifying the scene."
        )


def run_mvs(
    sfm_dir: str,
    output_dir: str,
    quality: str = "high",
    resolution_level: int = 0,
) -> Path:
    """Run Multi-View Stereo to produce dense point cloud.

    Args:
        sfm_dir: Directory containing SfM results (with sfm_data.json)
        output_dir: Output directory for dense reconstruction
        quality: Quality level — 'low', 'medium', 'high'
            (controls resolution_level if not explicitly set)
        resolution_level: OpenMVS resolution level (0=full, 1=half, 2=quarter, 3=eighth)
            Overrides quality-based default when specified (> 0)

    Returns:
        Path to output directory with scene_dense.mvs and scene_dense.ply

    Research notes:
        - Resolution level 2 reduces memory by ~4x (good for large scenes)
        - For small objects (AssetForge focus), resolution level 0 is fine
        - Coordinate normalization must be done BEFORE this step to prevent
          float32 precision loss (see coordinates.py normalize_sfm_data)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from assetforge.backends.openmvg import convert_to_openmvs

    # Convert OpenMVG output to OpenMVS format
    scene_mvs = out / "scene.mvs"
    convert_to_openmvs(sfm_dir, str(scene_mvs))

    # Determine resolution level based on quality if not explicitly set
    if resolution_level <= 0:
        quality_to_resolution = {
            "low": 2,
            "medium": 1,
            "high": 0,
        }
        resolution_level = quality_to_resolution.get(quality.lower(), 0)
        logger.info(f"Resolution level from quality '{quality}': {resolution_level}")
    else:
        logger.info(f"Using explicit resolution level: {resolution_level}")

    # Densify point cloud using PatchMatch Stereo
    logger.info(
        f"Running PatchMatch stereo densification (resolution_level={resolution_level})"
    )

    dense_cmd = [
        OPENMVS_DENSE,
        str(scene_mvs),
        "-w",
        str(out),
        "--resolution-level",
        str(resolution_level),
    ]

    _run_cmd(dense_cmd, desc="Dense point cloud reconstruction")

    # Verify output
    dense_ply = out / "scene_dense.ply"
    dense_mvs = out / "scene_dense.mvs"
    if dense_ply.exists():
        size_mb = dense_ply.stat().st_size / (1024 * 1024)
        logger.info(f"Dense point cloud: {dense_ply.name} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"Dense point cloud not found at {dense_ply}")

    logger.info(f"Multi-View Stereo complete → {out}")
    return out


def reconstruct_mesh(
    mvs_dir: str,
    output_path: str,
    point_limit: int = 0,
) -> Path:
    """Reconstruct mesh surface from dense point cloud.

    Args:
        mvs_dir: Directory containing scene_dense.mvs and scene_dense.ply
        output_path: Output path for the reconstructed mesh
        point_limit: Maximum number of points to use (0 = no limit)
            Downsampling can prevent out-of-memory errors on large scenes.

    Returns:
        Path to output mesh file

    Research notes:
        - Uses Delaunay tetrahedralization + min-cut graph for surface extraction
        - Can consume significant RAM for dense point clouds (>10M points)
        - Editing scene_dense.ply manually will break the reconstruction because
          the camera ray metadata is lost (Issue #1158)
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dense_scene = Path(mvs_dir) / "scene_dense.mvs"
    if not dense_scene.exists():
        logger.error(
            f"Dense scene not found at {dense_scene}. Run MVS denisification first."
        )
        raise FileNotFoundError(
            f"Dense scene not found at {dense_scene}. "
            "Run MVS denisification (run_mvs) first."
        )

    # Optionally downsample point cloud for memory management
    if point_limit > 0:
        dense_ply = Path(mvs_dir) / "scene_dense.ply"
        if dense_ply.exists():
            _downsample_point_cloud(dense_ply, point_limit)

    logger.info("Reconstructing mesh surface via Delaunay tetrahedralization")
    mesh_cmd = [
        OPENMVS_RECONSTRUCT,
        str(dense_scene),
        "-w",
        str(mvs_dir),
    ]

    _run_cmd(mesh_cmd, desc="Mesh reconstruction")

    # The output from OpenMVS is always scene_dense_mesh.ply in the working directory
    result_ply = Path(mvs_dir) / "scene_dense_mesh.ply"
    if result_ply.exists():
        # Copy to requested output path
        import shutil

        shutil.copy2(result_ply, output)
        logger.info(f"Mesh reconstruction complete → {output}")
    else:
        logger.warning(f"Expected mesh output not found at {result_ply}")
        # May have failed silently; raise clear error
        raise RuntimeError(
            f"Mesh reconstruction produced no output. "
            f"Check that {dense_scene} contains valid data."
        )

    return output


def refine_mesh(
    mvs_dir: str,
    output_path: str,
) -> Path:
    """Refine mesh surface detail using photometric optimization.

    Improves vertex positions by maximizing photometric consistency
    across multiple views. This recovers fine surface details that
    the initial mesh reconstruction may have smoothed out.

    Args:
        mvs_dir: Directory containing scene_dense.mvs and images
        output_path: Output path for the refined mesh

    Returns:
        Path to the refined mesh file

    Research notes:
        - RefineMesh optimizes vertex positions iteratively
        - Recovers fine details, protrusions, and curved surfaces
        - Typically improves surface detail by 15-30%
        - Requires original images (accessible via the .mvs scene file)
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dense_mesh_scene = Path(mvs_dir) / "scene_dense_mesh.mvs"
    if not dense_mesh_scene.exists():
        # Try alternative naming
        dense_mesh_scene = Path(mvs_dir) / "scene_dense.mvs"
        if not dense_mesh_scene.exists():
            logger.error(f"Mesh scene file not found. Run mesh reconstruction first.")
            raise FileNotFoundError(
                f"Cannot refine mesh — scene file not found at {dense_mesh_scene}. "
                "Run mesh reconstruction first."
            )

    logger.info("Refining mesh surface detail (photometric optimization)")
    refine_cmd = [
        OPENMVS_REFINE,
        str(dense_mesh_scene),
        "-w",
        str(mvs_dir),
    ]

    _run_cmd(refine_cmd, desc="Mesh refinement")

    # Output is scene_dense_mesh_refine.ply
    result_ply = Path(mvs_dir) / "scene_dense_mesh_refine.ply"
    if result_ply.exists():
        import shutil

        shutil.copy2(result_ply, output)
        logger.info(f"Mesh refinement complete → {output}")
    else:
        # Refinement may produce differently named output
        refined_files = list(Path(mvs_dir).glob("*refine*.ply"))
        if refined_files:
            import shutil

            shutil.copy2(refined_files[0], output)
            logger.info(f"Mesh refinement complete → {output}")
        else:
            logger.warning("Mesh refinement produced no output, using original mesh")
            import shutil

            original = Path(mvs_dir) / "scene_dense_mesh.ply"
            if original.exists():
                shutil.copy2(original, output)

    return output


def map_textures(
    mvs_dir: str,
    output_path: str,
    # Seam leveling control (critical for masked images)
    local_seam_leveling: bool = True,
    global_seam_leveling: bool = True,
    # Texture resolution
    texture_size: int = 4096,
) -> Path:
    """Project photo textures onto the reconstructed mesh.

    Args:
        mvs_dir: Directory containing mesh and MVS scene data
        output_path: Output path for the textured mesh
        local_seam_leveling: Enable local seam leveling
            CRITICAL: Disable when using masks to prevent black artifacts
        global_seam_leveling: Enable global seam leveling
            CRITICAL: Disable when using masks to prevent black artifacts
        texture_size: Resolution of the output texture atlas (power of 2)

    Returns:
        Path to output textured mesh file

    Research notes:
        - Seam leveling causes black artifacts on masked surfaces due to
          slight misalignment between mesh edges and mask boundaries
        - When using masks, ALWAYS disable both local and global seam leveling
        - OpenMVS Issue #1158: --ignore-mask-label < -1 crashes (uninitialized
          rasterer.mask). Do NOT use values below -1.
        - Uses Markov Random Field (MRF) with Loopy Belief Propagation for
          optimal camera assignment per triangle
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Find the mesh scene file (use refined mesh if available, fall back to original)
    mesh_scene = Path(mvs_dir) / "scene_dense_mesh_refine.mvs"
    if not mesh_scene.exists():
        mesh_scene = Path(mvs_dir) / "scene_dense_mesh.mvs"
    if not mesh_scene.exists():
        # Fall back to the original dense scene
        mesh_scene = Path(mvs_dir) / "scene_dense.mvs"

    if not mesh_scene.exists():
        raise FileNotFoundError(
            f"No mesh scene file found in {mvs_dir}. Run mesh reconstruction first."
        )

    logger.info("Mapping photo textures onto mesh")

    # Check if seam leveling should be disabled
    if not local_seam_leveling or not global_seam_leveling:
        logger.info(
            "Seam leveling partially disabled "
            f"(local={local_seam_leveling}, global={global_seam_leveling})"
        )

    texture_cmd = [
        OPENMVS_TEXTURE,
        str(mesh_scene),
        "-w",
        str(mvs_dir),
        "--export-type",
        "obj",
        "--texture-size",
        str(texture_size),
    ]

    if not local_seam_leveling:
        texture_cmd.extend(["--local-seam-leveling", "0"])
    if not global_seam_leveling:
        texture_cmd.extend(["--global-seam-leveling", "0"])

    _run_cmd(texture_cmd, desc="Texture mapping")

    # Find the textured output file
    textured_obj = Path(mvs_dir) / "scene_dense_mesh_texture.obj"
    textured_ply = Path(mvs_dir) / "scene_dense_mesh_texture.ply"

    if textured_obj.exists():
        import shutil

        shutil.copy2(textured_obj, output)
        # Also copy material and texture files
        for ext in [".mtl", ".png", ".jpg", ".jpeg", ".tga"]:
            for src_file in Path(mvs_dir).glob(f"scene_dense_mesh_texture*{ext}"):
                dst = output.parent / src_file.name
                shutil.copy2(src_file, dst)
        logger.info(f"Texture mapping complete → {output}")
    elif textured_ply.exists():
        import shutil

        shutil.copy2(textured_ply, output)
        logger.info(f"Texture mapping complete → {output} (PLY format)")
    else:
        logger.warning(f"Textured output not found, directory contents:")
        for f in Path(mvs_dir).glob("*"):
            logger.warning(f"  {f.name}")
        raise RuntimeError(
            "Texture mapping produced no output. "
            "Check that mesh and images are properly aligned."
        )

    return output


def _downsample_point_cloud(ply_path: Path, max_points: int) -> None:
    """Downsample a PLY point cloud in-place if it exceeds max_points.

    Uses a simple spatial subsampling strategy to reduce memory usage
    before mesh reconstruction.

    Args:
        ply_path: Path to the PLY file
        max_points: Maximum number of points to keep
    """
    if not ply_path.exists():
        return

    # Quick size check to avoid reading huge files unnecessarily
    size_mb = ply_path.stat().st_size / (1024 * 1024)

    # Rough estimate: 100K points ≈ 6MB for XYZ+normal+color
    est_points = int(size_mb * 100000 / 6)
    if est_points <= max_points:
        logger.debug(
            f"Point cloud ({est_points:,} pts) within limit ({max_points:,}), skipping downsampling"
        )
        return

    logger.info(
        f"Point cloud ~{est_points:,} pts exceeds limit ({max_points:,}), "
        f"downsampling for memory management"
    )

    # Read PLY, subsample, write back
    try:
        import struct

        with open(ply_path) as f:
            content = f.read()

        lines = content.split("\n")
        header_end = 0
        vertex_count = 0
        in_header = True
        for i, line in enumerate(lines):
            if in_header:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line == "end_header":
                    header_end = i
                    in_header = False
            else:
                break

        if vertex_count <= max_points:
            return

        # Keep every Nth point
        keep_ratio = max_points / vertex_count

        header_lines = lines[: header_end + 1]
        # Update vertex count in header
        for i, line in enumerate(header_lines):
            if line.startswith("element vertex"):
                header_lines[i] = f"element vertex {max_points}"

        data_lines = lines[header_end + 1 : header_end + 1 + vertex_count]

        # Simple uniform subsampling
        step = max(1, int(1.0 / keep_ratio))
        sampled = data_lines[::step][:max_points]

        with open(ply_path, "w") as f:
            f.write("\n".join(header_lines + sampled))

        logger.info(
            f"Downsampled point cloud: {vertex_count:,} → {len(sampled):,} points"
        )

    except Exception as e:
        logger.warning(f"Point cloud downsampling failed: {e}")
