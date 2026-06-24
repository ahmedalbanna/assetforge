"""
COLMAP backend — wraps the COLMAP photogrammetry CLI tools.

COLMAP is an open-source Structure-from-Motion and Multi-View Stereo pipeline.
It can replace both OpenMVG (SfM) and OpenMVS (MVS) as a single dependency.

Provides:
- Full automatic reconstruction (one command: feature_extraction → mesh)
- Granular stage control for pipeline integration
- CUDA GPU acceleration
- Mask support for background exclusion
- Export to .mvs format for OpenMVS texture mapping (optional)
"""

import subprocess
import logging
import shutil
import os
from pathlib import Path

logger = logging.getLogger("assetforge")

COLMAP_BIN = shutil.which("colmap")
COLMAP_AVAILABLE = COLMAP_BIN is not None

# CUDA compatibility shim cache
_COLAP_CUDA_ENV: dict[str, str] | None = None


def _resolve_cuda_shim() -> dict[str, str] | None:
    """Create a CUDA runtime compatibility shim if needed.

    COLMAP may be compiled with CUDA 12.x runtime (libcudart.so.12) but the
    system may have CUDA 13.x driver/runtime. Since CUDA 13 drivers are NOT
    backward compatible with CUDA 12 runtimes (different SONAME), we create a
    shim directory with a symlink from libcudart.so.12 → libcudart.so.13.

    Returns:
        Dict of environment variables (LD_LIBRARY_PATH) if shim is needed,
        or None if no shim is required.
    """
    global _COLAP_CUDA_ENV
    if _COLAP_CUDA_ENV is not None:
        return _COLAP_CUDA_ENV or None

    # Only needed if COLMAP is available
    if not COLMAP_BIN:
        _COLAP_CUDA_ENV = {}
        return None

    # Check if COLMAP is linked against CUDA 12
    try:
        ldd_result = subprocess.run(
            ["ldd", str(COLMAP_BIN)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        _COLAP_CUDA_ENV = {}
        return None

    if "libcudart.so.12" not in ldd_result.stdout:
        # COLMAP not linked against CUDA 12 — no shim needed
        _COLAP_CUDA_ENV = {}
        return None

    # Check if a CUDA 13 runtime is available on the system
    cuda13_candidates = [
        "/usr/local/cuda-13.0/lib64/libcudart.so.13.0.96",
        "/usr/local/cuda-13/lib64/libcudart.so.13.0.96",
        "/usr/local/cuda/lib64/libcudart.so.13.0.96",
    ]
    cuda13_runtime = None
    for candidate in cuda13_candidates:
        if os.path.exists(candidate):
            cuda13_runtime = candidate
            break

    if cuda13_runtime is None:
        # CUDA 13 runtime not available — no shim possible
        logger.debug(
            "COLMAP linked with CUDA 12, but CUDA 13 runtime not found. "
            "GPU may not work. Use CPU fallback."
        )
        _COLAP_CUDA_ENV = {}
        return None

    # Create shim directory
    shim_dir = Path("/tmp/assetforge_cuda_shim")
    shim_dir.mkdir(parents=True, exist_ok=True)

    # Symlink libcudart.so.12 → CUDA 13 runtime
    link_path = shim_dir / "libcudart.so.12"
    if not link_path.exists() or os.readlink(str(link_path)) != cuda13_runtime:
        link_path.unlink(missing_ok=True)
        os.symlink(cuda13_runtime, str(link_path))

    # Also symlink other CUDA libraries COLMAP needs
    _symlink_cuda_lib("libcurand.so", shim_dir)
    _symlink_cuda_lib("libcusparse.so", shim_dir)
    _symlink_cuda_lib("libcublas.so", shim_dir)
    _symlink_cuda_lib("libcublasLt.so", shim_dir)
    _symlink_cuda_lib("libcufft.so", shim_dir)
    _symlink_cuda_lib("libcusolver.so", shim_dir)

    env = {
        "LD_LIBRARY_PATH": str(shim_dir) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    }
    _COLAP_CUDA_ENV = env
    logger.info(
        "COLMAP CUDA compatibility shim active: "
        f"CUDA 12 → 13 (libcudart.so.12 → {cuda13_runtime})"
    )
    return env


def _symlink_cuda_lib(pattern: str, target_dir: Path) -> None:
    """Symlink a CUDA library from CUDA 13 into the shim directory.

    Looks for library files matching <pattern>* under /usr/local/cuda-13.0/
    and creates a symlink in target_dir.
    """
    import glob as _glob

    # Search in CUDA 13 library directories
    search_dirs = [
        "/usr/local/cuda-13.0/lib64",
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "/usr/local/cuda-13/lib64",
        "/usr/local/cuda-13/targets/x86_64-linux/lib",
    ]

    for search_dir in search_dirs:
        matches = _glob.glob(os.path.join(search_dir, pattern) + "*")
        for match in matches:
            if os.path.isfile(match):
                link_name = os.path.basename(match)
                link_path = target_dir / link_name
                if not link_path.exists() or os.readlink(str(link_path)) != match:
                    link_path.unlink(missing_ok=True)
                    os.symlink(match, str(link_path))
                return


def _check_cuda_available() -> bool:
    """Check if CUDA is actually usable by COLMAP.

    COLMAP may be compiled with CUDA support but the system driver
    may be too old. This catches that case and falls back to CPU.
    """
    try:
        result = subprocess.run(
            [str(COLMAP_BIN), "feature_extractor", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "gpu" in result.stdout.lower()
    except Exception:
        return False


def check_colmap() -> bool:
    """Verify COLMAP is installed and available."""
    if COLMAP_AVAILABLE:
        return True
    logger.warning(
        "COLMAP not found. Install it: "
        "sudo apt-get install colmap  # Linux, or "
        "https://colmap.github.io/install.html"
    )
    return False


def _get_colmap_env() -> dict:
    """Get environment for running COLMAP subprocesses with CUDA shim if needed."""
    env = os.environ.copy()
    cuda_env = _resolve_cuda_shim()
    if cuda_env:
        env["LD_LIBRARY_PATH"] = cuda_env["LD_LIBRARY_PATH"]
    return env


def _run_cmd(
    cmd: list[str], desc: str = "", env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run COLMAP subprocess with logging and error handling.

    Args:
        cmd: Command list to execute
        desc: Description for error messages
        env: Environment variables. If None, auto-configures CUDA shim.
    """
    if env is None:
        env = _get_colmap_env()
    cmd_str = " ".join(str(c) for c in cmd)
    logger.debug(f"Running COLMAP: {cmd_str}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, timeout=28800, env=env
        )
        return result
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        logger.error(f"COLMAP command failed [{desc}]")
        if stderr:
            logger.error(f"Stderr: {stderr[:500]}")
        raise RuntimeError(
            f"COLMAP {desc} failed. Check input images and COLMAP installation."
        ) from e
    except subprocess.TimeoutExpired:
        logger.error(f"COLMAP command timed out [{desc}]")
        raise RuntimeError(f"COLMAP {desc} timed out after 8 hours.")


def run_automatic_reconstruction(
    image_dir: str,
    workspace_dir: str,
    quality: str = "high",
    data_type: str = "individual",
    mask_dir: str | None = None,
    use_gpu: bool = True,
    mesher: str = "poisson",
    dense: bool = True,
    sparse: bool = True,
) -> Path:
    """Run full COLMAP automatic reconstruction pipeline.

    This is the simplest entry point — COLMAP handles everything from
    feature extraction to mesh generation in one command.

    Args:
        image_dir: Directory containing input images
        workspace_dir: Output workspace directory
        quality: Quality level — 'low', 'medium', 'high', 'extreme'
        data_type: 'individual' for separate photos, 'video' for sequential
        mask_dir: Optional directory with per-image masks (COLMAP auto-detects)
        use_gpu: Enable CUDA GPU acceleration
        mesher: Mesh type — 'poisson' (smooth) or 'delaunay' (exact)
        dense: Run dense MVS reconstruction
        sparse: Run sparse SfM reconstruction

    Returns:
        Path to workspace directory containing:
        - sparse/ — SfM output (cameras, points3D, etc.)
        - dense/ — MVS output (depth maps, fused point cloud)
        - meshed/ — Final mesh (poisson or delaunay)
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed. Install it to use this backend.")

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Validate quality
    valid_qualities = {"low", "medium", "high", "extreme"}
    if quality not in valid_qualities:
        logger.warning(
            f"Unknown quality '{quality}', using 'high'. "
            f"Valid: {', '.join(sorted(valid_qualities))}"
        )
        quality = "high"

    logger.info(
        f"Running COLMAP automatic reconstruction "
        f"(quality={quality}, data_type={data_type}, mesher={mesher}, "
        f"gpu={use_gpu}, sparse={sparse}, dense={dense})"
    )

    cmd = [
        str(COLMAP_BIN),
        "automatic_reconstructor",
        "--workspace_path",
        str(workspace),
        "--image_path",
        str(image_dir),
        "--quality",
        quality,
        "--data_type",
        data_type,
        "--mesher",
        mesher,
        "--sparse",
        "1" if sparse else "0",
        "--dense",
        "1" if dense else "0",
    ]

    if mask_dir:
        cmd.extend(["--mask_path", str(mask_dir)])
        logger.info(f"Using mask path: {mask_dir}")

    if use_gpu:
        cmd.append("--use_gpu")
        cmd.append("1")
    else:
        cmd.append("--use_gpu")
        cmd.append("0")
        _run_cmd(cmd, desc="Automatic reconstruction")
        _log_workspace_contents(workspace)
        logger.info(f"COLMAP reconstruction complete → {workspace}")
        return workspace

    # Try GPU first (with CUDA shim if needed), fall back to CPU
    try:
        _run_cmd(cmd, desc="Automatic reconstruction")
    except RuntimeError as e:
        stderr = getattr(e, "stderr", "")
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            logger.warning(f"GPU reconstruction failed, retrying with CPU: {e}")
            cmd[cmd.index("1") if "--use_gpu" in cmd else -1] = "0"
            cmd[cmd.index("1") - 1 if "--use_gpu" in cmd else -1] = "0"
            # Safer: rebuild cmd with CPU
            cpu_cmd = [
                str(COLMAP_BIN),
                "automatic_reconstructor",
                "--workspace_path",
                str(workspace),
                "--image_path",
                str(image_dir),
                "--quality",
                quality,
                "--data_type",
                data_type,
                "--mesher",
                mesher,
                "--sparse",
                "1" if sparse else "0",
                "--dense",
                "1" if dense else "0",
                "--use_gpu",
                "0",
            ]
            if mask_dir:
                cpu_cmd.extend(["--mask_path", str(mask_dir)])
            _run_cmd(cpu_cmd, desc="Automatic reconstruction (CPU fallback)")
        else:
            raise

    # Log results
    _log_workspace_contents(workspace)

    logger.info(f"COLMAP reconstruction complete → {workspace}")
    return workspace


def extract_features(
    image_dir: str,
    database_path: str,
    sift_params: dict | None = None,
    camera_model: str = "SIMPLE_RADIAL",
    mask_dir: str | None = None,
    use_gpu: bool = True,
) -> Path:
    """Extract SIFT features from images using COLMAP.

    Args:
        image_dir: Directory containing input images
        database_path: Path to COLMAP database file (.db)
        sift_params: Optional dict of SIFT extraction parameters
        camera_model: Camera model type
        mask_dir: Optional mask directory
        use_gpu: Enable GPU acceleration

    Returns:
        Path to database file
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    db = Path(database_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting COLMAP features from {image_dir}")

    cmd = [
        str(COLMAP_BIN),
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model,
    ]

    if mask_dir:
        cmd.extend(["--ImageReader.mask_path", str(mask_dir)])

    # Check if CUDA is actually usable — fall back to CPU if not
    _cuda_works = use_gpu and _resolve_cuda_shim() is not None

    if not _cuda_works and use_gpu:
        logger.warning("CUDA unavailable, falling back to CPU for feature extraction")

    if _cuda_works:
        cmd.extend(["--FeatureExtraction.use_gpu", "1"])
    else:
        cmd.extend(["--FeatureExtraction.use_gpu", "0"])

    if sift_params:
        for key, value in sift_params.items():
            cmd.extend([f"--SiftExtraction.{key}", str(value)])

    _run_cmd(cmd, desc="Feature extraction")
    logger.info(f"Features extracted → {db}")
    return db


def match_features(
    database_path: str,
    match_type: str = "exhaustive",
    use_gpu: bool = True,
    sequential_overlap: int = 10,
    vocab_tree_path: str | None = None,
) -> Path:
    """Match features between image pairs.

    Args:
        database_path: Path to COLMAP database file
        match_type: Matching strategy — 'exhaustive', 'sequential',
            'spatial', 'vocab_tree', 'transitive'
        use_gpu: Enable GPU acceleration
        sequential_overlap: Overlap for sequential matching
        vocab_tree_path: Path to vocabulary tree file (for vocab_tree matcher)

    Returns:
        Path to database file
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    db = Path(database_path)
    logger.info(f"Matching features ({match_type})")

    matchers = {
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
        "spatial": "spatial_matcher",
        "vocab_tree": "vocab_tree_matcher",
        "transitive": "transitive_matcher",
    }

    matcher = matchers.get(match_type)
    if matcher is None:
        logger.warning(
            f"Unknown match type '{match_type}', using 'exhaustive'. "
            f"Valid: {', '.join(sorted(matchers.keys()))}"
        )
        matcher = "exhaustive_matcher"

    cmd = [str(COLMAP_BIN), matcher, "--database_path", str(db)]

    if use_gpu:
        cmd.extend(["--FeatureMatching.use_gpu", "1"])
    else:
        cmd.extend(["--FeatureMatching.use_gpu", "0"])

    if match_type == "sequential":
        cmd.extend(["--SequentialMatching.overlap", str(sequential_overlap)])

    if match_type == "vocab_tree" and vocab_tree_path:
        cmd.extend(["--VocabTreeMatching.vocab_tree_path", vocab_tree_path])

    _run_cmd(cmd, desc=f"Feature matching ({match_type})")
    logger.info(f"Feature matching complete → {db}")
    return db


def run_sfm(
    image_dir: str,
    database_path: str,
    output_dir: str,
    sfm_type: str = "incremental",
) -> Path:
    """Run Structure from Motion using COLMAP mapper.

    Args:
        image_dir: Directory with input images
        database_path: Path to database with features and matches
        output_dir: Output directory for SfM model
        sfm_type: Type — 'incremental' or 'global' via hierarchical_mapper

    Returns:
        Path to output directory with SfM results (sparse model)
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sfm_type == "global":
        logger.info("Running global SfM (hierarchical mapper)")
        cmd = [
            str(COLMAP_BIN),
            "hierarchical_mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(out),
        ]
    else:
        logger.info(f"Running incremental SfM (sfm_type={sfm_type})")
        cmd = [
            str(COLMAP_BIN),
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(out),
        ]

    _run_cmd(cmd, desc=f"SfM ({sfm_type})")

    # Check output
    sparse_dirs = list(out.glob("*/"))
    if sparse_dirs:
        model_path = sparse_dirs[0]
        logger.info(f"SfM model saved to {model_path}")
    else:
        logger.warning(f"No SfM output found in {out}")

    logger.info(f"SfM reconstruction complete ({sfm_type}) → {out}")
    return out


def undistort_images(
    sfm_model_dir: str,
    image_dir: str,
    output_dir: str,
) -> Path:
    """Undistort images for MVS processing.

    Required before running patch_match_stereo.

    Args:
        sfm_model_dir: Directory with COLMAP SfM model (cameras.bin, images.bin, points3D.bin)
        image_dir: Original image directory
        output_dir: Output directory for undistorted images

    Returns:
        Path to output directory
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLMAP_BIN),
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(sfm_model_dir),
        "--output_path",
        str(out),
    ]

    _run_cmd(cmd, desc="Image undistortion")
    logger.info(f"Undistorted images → {out}")
    return out


def run_dense_mvs(
    undistorted_dir: str,
    output_dir: str,
    use_gpu: bool = True,
) -> Path:
    """Run dense MVS using patch-match stereo.

    Args:
        undistorted_dir: Directory with undistorted images from image_undistorter
        output_dir: Output directory for depth maps
        use_gpu: Enable GPU acceleration

    Returns:
        Path to output directory with depth maps
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLMAP_BIN),
        "patch_match_stereo",
        "--workspace_path",
        str(undistorted_dir),
    ]

    if use_gpu:
        cmd.append("--PatchMatchStereo.gpu_index")
        cmd.append("0")
    else:
        cmd.append("--PatchMatchStereo.gpu_index")
        cmd.append("-1")

    _run_cmd(cmd, desc="Dense MVS (patch-match stereo)")
    logger.info(f"Dense depth maps → {undistorted_dir}")
    return out


def fuse_point_cloud(
    undistorted_dir: str,
    output_path: str,
) -> Path:
    """Fuse depth maps into a dense point cloud.

    Args:
        undistorted_dir: Directory with undistorted images and depth maps
        output_path: Output path for fused PLY point cloud

    Returns:
        Path to fused PLY file
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLMAP_BIN),
        "stereo_fusion",
        "--workspace_path",
        str(undistorted_dir),
        "--output_path",
        str(output),
    ]

    _run_cmd(cmd, desc="Dense point cloud fusion")
    logger.info(f"Fused point cloud → {output}")
    return output


def reconstruct_mesh(
    input_point_cloud: str,
    output_path: str,
    mesher: str = "poisson",
) -> Path:
    """Reconstruct mesh from dense point cloud.

    Args:
        input_point_cloud: Path to fused PLY point cloud
        output_path: Output path for mesh file
        mesher: Meshing method — 'poisson' (smooth, watertight) or
            'delaunay' (exact surface reconstruction)

    Returns:
        Path to output mesh file
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if mesher == "poisson":
        cmd = [
            str(COLMAP_BIN),
            "poisson_mesher",
            "--input_path",
            str(input_point_cloud),
            "--output_path",
            str(output),
        ]
    else:
        cmd = [
            str(COLMAP_BIN),
            "delaunay_mesher",
            "--input_path",
            str(input_point_cloud),
            "--output_path",
            str(output),
        ]

    _run_cmd(cmd, desc=f"Mesh reconstruction ({mesher})")
    logger.info(f"Mesh reconstruction complete → {output}")
    return output


def export_model(
    sfm_model_dir: str,
    output_path: str,
    output_type: str = "PLY",
) -> Path:
    """Export COLMAP SfM model to a standard format.

    Args:
        sfm_model_dir: Directory with COLMAP SfM model
        output_path: Output file path
        output_type: Export format — 'PLY', 'BIN', 'TXT', 'NVM', 'VRML', 'CAM'

    Returns:
        Path to exported model
    """
    if not check_colmap():
        raise RuntimeError("COLMAP is not installed.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COLMAP_BIN),
        "model_converter",
        "--input_path",
        str(sfm_model_dir),
        "--output_path",
        str(output),
        "--output_type",
        output_type,
    ]

    _run_cmd(cmd, desc=f"Model export ({output_type})")
    logger.info(f"Model exported → {output}")
    return output


def convert_to_openmvs(
    sfm_model_dir: str,
    output_path: str,
) -> Path | None:
    """Export COLMAP model in a format compatible with OpenMVS.

    COLMAP has 'openmvs' export via its model_converter. If OpenMVS is
    available, this enables texture mapping on COLMAP-reconstructed models.

    Args:
        sfm_model_dir: Directory with COLMAP SfM model
        output_path: Output .mvs file path

    Returns:
        Path to output file, or None if conversion fails
    """
    if not check_colmap():
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            str(COLMAP_BIN),
            "model_converter",
            "--input_path",
            str(sfm_model_dir),
            "--output_path",
            str(output),
            "--output_type",
            "OpenMVS",
        ]
        _run_cmd(cmd, desc="OpenMVS conversion")
        logger.info(f"OpenMVS scene file created → {output}")
        return output
    except (RuntimeError, subprocess.CalledProcessError) as e:
        logger.warning(f"OpenMVS conversion failed: {e}")
        return None


def run_full_pipeline_stages(
    image_dir: str,
    output_dir: str,
    quality: str = "high",
    sfm_type: str = "incremental",
    mask_dir: str | None = None,
    use_gpu: bool = True,
    mesher: str = "poisson",
    dense_mvs: bool = True,
) -> dict[str, Path]:
    """Run COLMAP pipeline as individual stages for granular control.

    This provides finer control than automatic_reconstructor by running
    each stage separately with stage-specific output paths.

    Returns:
        Dict mapping stage names to output paths
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = {}

    # Stage 1: Feature extraction
    db_path = output / "database.db"
    extract_features(image_dir, str(db_path), mask_dir=mask_dir, use_gpu=use_gpu)
    results["feature_extraction"] = db_path

    # Stage 2: Feature matching
    match_features(str(db_path), use_gpu=use_gpu)
    results["feature_matching"] = db_path

    # Stage 3: SfM
    sfm_dir = output / "sparse"
    run_sfm(image_dir, str(db_path), str(sfm_dir), sfm_type=sfm_type)
    # Find the actual model directory (COLMAP creates subdirectories like 0/)
    model_dirs = sorted(sfm_dir.glob("*/"))
    model_dir = model_dirs[0] if model_dirs else sfm_dir
    results["sfm"] = model_dir

    # Stage 4: Dense MVS (optional)
    if dense_mvs:
        undistorted_dir = output / "dense"
        undistort_images(str(model_dir), image_dir, str(undistorted_dir))
        results["undistortion"] = undistorted_dir

        run_dense_mvs(str(undistorted_dir), str(output / "depth"), use_gpu=use_gpu)
        results["dense_mvs"] = undistorted_dir

        fused_ply = output / "fused.ply"
        fuse_point_cloud(str(undistorted_dir), str(fused_ply))
        results["fusion"] = fused_ply

        # Stage 5: Mesh
        mesh_ply = output / f"mesh_{mesher}.ply"
        reconstruct_mesh(str(fused_ply), str(mesh_ply), mesher=mesher)
        results["mesh"] = mesh_ply

    return results


def _log_workspace_contents(workspace: Path) -> None:
    """Log the contents of a COLMAP workspace for debugging."""
    for item in workspace.iterdir():
        if item.is_dir():
            files = list(item.iterdir())
            size = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
            logger.info(f"  {item.name}/ — {len(files)} files ({size:.1f} MB)")
        else:
            size = item.stat().st_size / (1024 * 1024)
            logger.info(f"  {item.name} ({size:.1f} MB)")


def enhance_images_for_sfm(
    image_dir: str,
    output_dir: str | None = None,
) -> Path:
    """Pre-process images to improve SIFT feature detection.

    AI-generated and synthetic images often lack the high-frequency detail
    and sharp edges that SIFT needs for reliable feature matching. This
    function applies a series of enhancements to make such images more
    suitable for Structure-from-Motion:

    1. Sharpening (unsharp mask) to enhance edges
    2. CLAHE local contrast enhancement for texture detail
    3. **Fiducial markers** (checkerboard corners + dots) — these provide
       consistent high-contrast features that SIFT can detect and match
       across views, enabling COLMAP to establish camera geometry even
       when the original images lack sufficient natural features.
    4. Subtle structured noise for additional SIFT features

    Args:
        image_dir: Directory with input images to enhance
        output_dir: Output directory for enhanced images.
            If None, creates a temp directory.

    Returns:
        Path to directory with enhanced images
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning(
            "OpenCV not available for image enhancement. "
            "Install opencv-python: pip install opencv-python"
        )
        return Path(image_dir)

    src = Path(image_dir)
    if output_dir is None:
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="assetforge_enhanced_")
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(src.glob("*.jpg"))
        + list(src.glob("*.jpeg"))
        + list(src.glob("*.png"))
        + list(src.glob("*.bmp"))
        + list(src.glob("*.tif"))
        + list(src.glob("*.tiff"))
    )

    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return Path(image_dir)

    logger.info(
        f"Enhancing {len(image_files)} images for SIFT feature detection "
        "(sharpening + CLAHE + fiducial markers)"
    )

    # Pre-generate fiducial marker patterns (reused across all images)
    marker_size = 8  # pixels per checker cell
    marker_grid = 6  # cells per side
    marker_pixels = marker_size * marker_grid
    marker = np.zeros((marker_pixels, marker_pixels), dtype=np.uint8)
    for row in range(marker_grid):
        for col in range(marker_grid):
            val = 255 if (row + col) % 2 == 0 else 30
            y1, y2 = row * marker_size, (row + 1) * marker_size
            x1, x2 = col * marker_size, (col + 1) * marker_size
            marker[y1:y2, x1:x2] = val
    # White border around marker
    marker_bordered = cv2.copyMakeBorder(
        marker, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255,)
    )

    for f in image_files:
        img = cv2.imread(str(f))
        if img is None:
            logger.warning(f"Could not read {f}, copying as-is")
            import shutil

            shutil.copy2(str(f), str(dst / f.name))
            continue

        h, w = img.shape[:2]

        # 1. Strong sharpening (unsharp mask)
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        img_sharp = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

        # 2. CLAHE on L channel for local contrast
        lab = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2LAB)
        l_plane, a_plane, b_plane = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_plane = clahe.apply(l_plane)
        lab = cv2.merge([l_plane, a_plane, b_plane])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Add fiducial markers at image corners (fixed positions)
        # These provide consistent SIFT-detectable features across views
        marker_bgr = cv2.cvtColor(marker_bordered, cv2.COLOR_GRAY2BGR)
        margin = 20
        positions = [
            (margin, margin),  # top-left
            (w - marker_bordered.shape[1] - margin, margin),  # top-right
            (margin, h - marker_bordered.shape[0] - margin),  # bottom-left
            (
                w - marker_bordered.shape[1] - margin,
                h - marker_bordered.shape[0] - margin,
            ),  # bottom-right
        ]
        for px, py in positions:
            img_enhanced[
                py : py + marker_bordered.shape[0], px : px + marker_bordered.shape[1]
            ] = marker_bgr

        # 4. High-contrast corner dots (white on dark) for edge detection
        dot_size = 6
        dot_pad = 4
        dot_locations = [
            (margin + marker_bordered.shape[1] + dot_pad + dot_size, margin + dot_size),
            (
                w - margin - marker_bordered.shape[1] - dot_pad - dot_size,
                margin + dot_size,
            ),
            (
                margin + marker_bordered.shape[1] + dot_pad + dot_size,
                h - margin - dot_size,
            ),
            (
                w - margin - marker_bordered.shape[1] - dot_pad - dot_size,
                h - margin - dot_size,
            ),
        ]
        for cx_pos, cy_pos in dot_locations:
            cv2.circle(
                img_enhanced,
                (int(cx_pos), int(cy_pos)),
                dot_size,
                (255, 255, 255),
                -1,
            )
            cv2.circle(
                img_enhanced,
                (int(cx_pos), int(cy_pos)),
                dot_size + 2,
                (0, 0, 0),
                2,
            )

        # 5. Subtle structured noise for additional SIFT features
        noise_layer = np.zeros((h, w), dtype=np.uint8)
        num_dots = max(500, int(h * w / 5000))
        dots_y = np.random.randint(0, h, num_dots)
        dots_x = np.random.randint(0, w, num_dots)
        noise_layer[dots_y, dots_x] = 200  # soft gray
        for _ in range(max(20, int(h * w / 100000))):
            x1 = np.random.randint(0, max(w - 10, 1))
            y1 = np.random.randint(0, max(h - 10, 1))
            x2 = min(x1 + np.random.randint(3, 12), w - 1)
            y2 = min(y1 + np.random.randint(3, 12), h - 1)
            cv2.rectangle(noise_layer, (x1, y1), (x2, y2), 180, -1)

        noise_colored = cv2.cvtColor(noise_layer, cv2.COLOR_GRAY2BGR)
        img_final = cv2.addWeighted(img_enhanced, 0.95, noise_colored, 0.05, 0)

        # 6. Final gentle sharpening
        blur2 = cv2.GaussianBlur(img_final, (0, 0), 1.0)
        img_final = cv2.addWeighted(img_final, 1.2, blur2, -0.2, 0)

        output_path = str(dst / f.name)
        cv2.imwrite(output_path, img_final)

    logger.info(f"Enhanced images (with fiducial markers) saved to {dst}")
    return dst
