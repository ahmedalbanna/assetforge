"""
OpenMVG backend — wraps the OpenMVG photogrammetry CLI tools.

Provides:
- Feature extraction (SIFT/ORB/AKAZE) with masking support
- Feature matching
- Structure from Motion (SfM) — incremental, incremental_v2, global
- Coordinate normalization integration for float32 precision safety
- OpenMVG-to-OpenMVS conversion

Based on research insights:
    - AC-RANSAC provides threshold-free robust estimation
    - Global SfM is faster and drift-free; incremental is more robust
    - Mask support prevents feature extraction in background regions
    - Coordinate normalization prevents float32 precision loss in OpenMVS
"""

import subprocess
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("assetforge")

# Default OpenMVG commands (adjust if not in PATH)
OPENMVG_SFM = "openMVG_main_SfMInit_ImageListing"
OPENMVG_FEATURES = "openMVG_main_ComputeFeatures"
OPENMVG_MATCHES = "openMVG_main_ComputeMatches"
OPENMVG_SFM_RECONSTRUCT = "openMVG_main_SfM"
OPENMVG_TO_OPENMVS = "openMVG_main_openMVG2openMVS"

# Valid SfM types
VALID_SFM_TYPES = {"incremental", "incremental_v2", "global"}
DEFAULT_SFM_TYPE = "incremental"

# Valid feature types
VALID_FEATURE_TYPES = {"sift", "orb", "akaze"}
DEFAULT_FEATURE_TYPE = "sift"


def _check_openmvg():
    """Verify core OpenMVG tools are available."""
    for cmd in [OPENMVG_SFM, OPENMVG_FEATURES, OPENMVG_MATCHES]:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, timeout=30, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            logger.warning(
                f"OpenMVG tool '{cmd}' not found. Install openMVG: "
                f"https://github.com/openMVG/openMVG"
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
        subprocess.CalledProcessError if command fails
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.debug(f"Running: {cmd_str}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        return result
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        logger.error(f"Command failed [{desc}]: {cmd_str}")
        if stderr:
            logger.error(f"Stderr: {stderr[:500]}")
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out [{desc}]: {cmd_str}")
        raise


def extract_features(
    input_dir: str,
    output_dir: str,
    feature_type: str = "sift",
    mask_dir: str | None = None,
    upright: bool = False,
) -> Path:
    """Extract features from all images in input_dir using OpenMVG.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for features and matches
        feature_type: Feature extractor type — 'sift', 'orb', or 'akaze'
        mask_dir: Optional directory with background masks
            (auto-detects global_mask.png or <image>_mask.png per image)
        upright: If True, disable rotation invariance (faster for upright captures)

    Returns:
        Path to output directory with sfm_data.json and computed features

    Research notes:
        - Masks (black pixels) prevent feature extraction in background regions
        - Global mask: {mask_dir}/global_mask.png
        - Per-image mask: {mask_dir}/{image_name}_mask.png
        - Upright features (-u 1) disables orientation estimation for speed
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Validate feature type
    feature_type = feature_type.lower()
    if feature_type not in VALID_FEATURE_TYPES:
        logger.warning(
            f"Unknown feature type '{feature_type}', falling back to 'sift'. "
            f"Valid options: {', '.join(sorted(VALID_FEATURE_TYPES))}"
        )
        feature_type = "sift"

    # Map feature types to OpenMVG descriptor names
    feature_map = {
        "sift": "SIFT",
        "orb": "ORB",
        "akaze": "AKAZE_FLOAT",
    }
    describer = feature_map[feature_type]

    # Step 1: Image listing
    logger.info(
        f"Step 1/3: Image listing (feature={feature_type}, describer={describer})"
    )
    sfm_data = out / "sfm_data.json"

    sfm_cmd = [
        OPENMVG_SFM,
        "-i",
        input_dir,
        "-o",
        str(out),
        "-d",
        describer,
    ]

    # Add mask support if mask_dir provided
    if mask_dir:
        mask_path = Path(mask_dir)
        if mask_path.is_dir():
            sfm_cmd.extend(["-m", str(mask_path)])
            logger.info(f"Using mask directory: {mask_path}")
            global_mask = mask_path / "global_mask.png"
            if global_mask.exists():
                logger.info(f"Found global mask: {global_mask}")
        else:
            logger.warning(f"Mask directory not found: {mask_dir}")

    _run_cmd(sfm_cmd, desc="Image listing")

    # Step 2: Compute features
    logger.info(f"Step 2/3: Computing {feature_type} features")
    features_cmd = [
        OPENMVG_FEATURES,
        "-i",
        str(sfm_data),
        "-o",
        str(out),
    ]

    # Add upright flag if specified
    if upright:
        features_cmd.extend(["-u", "1"])
        logger.info("Using upright features (rotation-invariance disabled)")

    _run_cmd(features_cmd, desc="Feature extraction")

    # Step 3: Compute matches
    logger.info("Step 3/3: Computing feature matches")
    matches_cmd = [
        OPENMVG_MATCHES,
        "-i",
        str(sfm_data),
        "-o",
        str(out),
    ]
    _run_cmd(matches_cmd, desc="Feature matching")

    logger.info(f"Features extracted to {out}")
    return out


def match_features(features_dir: str, output_dir: str) -> Path:
    """Geometric matching between image pairs.

    Note: In the full pipeline, this is integrated into extract_features().
    This function exists for granular stage control.

    Args:
        features_dir: Directory containing features (sfm_data.json + computed features)
        output_dir: Output directory for match data

    Returns:
        Path to output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sfm_data = Path(features_dir) / "sfm_data.json"

    if not sfm_data.exists():
        logger.warning(f"sfm_data.json not found at {sfm_data}, skipping matching")
        return out

    matches_cmd = [
        OPENMVG_MATCHES,
        "-i",
        str(sfm_data),
        "-o",
        str(out),
    ]
    _run_cmd(matches_cmd, desc="Geometric matching")

    # Copy sfm_data for downstream stages
    dest_sfm = out / "sfm_data.json"
    if not dest_sfm.exists():
        shutil.copy2(sfm_data, dest_sfm)

    logger.info(f"Feature matching complete → {out}")
    return out


def run_sfm(
    matches_dir: str,
    output_dir: str,
    sfm_type: str = "incremental",
) -> Path:
    """Run Structure from Motion to compute camera poses and sparse cloud.

    Args:
        matches_dir: Directory containing matches and sfm_data.json
        output_dir: Output directory for SfM results
        sfm_type: SfM algorithm type — 'incremental', 'incremental_v2', or 'global'

    Returns:
        Path to output directory with SfM results

    Research notes:
        - Incremental: Robust for sparse/imperfect data, but slow, drift-prone
        - Global: Fast, no drift, needs robust relative poses
        - IncrementalV2: Updated incremental with better initialization
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sfm_data = Path(matches_dir) / "sfm_data.json"

    if not sfm_data.exists():
        logger.error(f"sfm_data.json not found: {sfm_data}")
        raise FileNotFoundError(
            f"Cannot run SfM without sfm_data.json at {sfm_data}. "
            "Run feature extraction first."
        )

    # Map SfM type to command
    sfm_type = sfm_type.lower()
    if sfm_type not in VALID_SFM_TYPES:
        logger.warning(
            f"Unknown SfM type '{sfm_type}', falling back to 'incremental'. "
            f"Valid options: {', '.join(sorted(VALID_SFM_TYPES))}"
        )
        sfm_type = "incremental"

    # Use -s flag to specify SfM type
    sfm_cmd = [
        OPENMVG_SFM_RECONSTRUCT,
        "-i",
        str(sfm_data),
        "-m",
        matches_dir,
        "-o",
        str(out),
    ]

    # Add SfM type flag for global (incremental is the default)
    if sfm_type == "global":
        sfm_cmd.extend(["-s", "GLOBAL"])
        logger.info("Using Global SfM (faster, less drift)")
    elif sfm_type == "incremental_v2":
        sfm_cmd.extend(["-s", "INCREMENTAL_V2"])
        logger.info("Using IncrementalV2 SfM")
    else:
        logger.info("Using Incremental SfM (default, robust)")

    _run_cmd(sfm_cmd, desc=f"SfM reconstruction ({sfm_type})")

    # Log output files
    sfm_out_files = list(out.glob("*.ply")) + list(out.glob("*.json"))
    for f in sfm_out_files:
        logger.debug(f"  SfM output: {f.name}")

    logger.info(f"SfM reconstruction complete ({sfm_type}) → {out}")
    return out


def convert_to_openmvs(sfm_dir: str, output_path: str) -> Path:
    """Convert OpenMVG SfM output to OpenMVS format (.mvs).

    Args:
        sfm_dir: Directory containing SfM results (sfm_data.json + camera poses)
        output_path: Output path for the .mvs scene file

    Returns:
        Path to the generated .mvs file

    Raises:
        FileNotFoundError: If sfm_data.json is missing
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sfm_data = Path(sfm_dir) / "sfm_data.json"

    if not sfm_data.exists():
        raise FileNotFoundError(
            f"Cannot convert to OpenMVS — sfm_data.json not found at {sfm_data}"
        )

    cmd = [
        OPENMVG_TO_OPENMVS,
        "-i",
        str(sfm_data),
        "-o",
        str(output),
    ]

    _run_cmd(cmd, desc="OpenMVG to OpenMVS conversion")

    logger.info(f"OpenMVS scene file created → {output}")
    return output


def run_sfm_with_normalization(
    matches_dir: str,
    output_dir: str,
    sfm_type: str = "incremental",
) -> tuple[Path, object]:
    """Run SfM with coordinate normalization to prevent float32 precision loss.

    This is the recommended entry point for scenes with large coordinates
    (e.g., GPS/UTM data). It normalizes coordinates, runs SfM, and returns
    the transform for later restoration.

    Args:
        matches_dir: Directory containing matches
        output_dir: Output directory
        sfm_type: SfM algorithm type

    Returns:
        Tuple of (output_dir_path, CoordinateTransform)
    """
    from assetforge.utils.coordinates import normalize_sfm_data

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # First, normalize the sfm_data to prevent float32 precision issues
    sfm_data_path = Path(matches_dir) / "sfm_data.json"
    normalized_sfm = out / "sfm_data_normalized.json"

    if sfm_data_path.exists():
        transform = normalize_sfm_data(sfm_data_path, normalized_sfm)
        # Use the normalized sfm_data
        _run_sfm_with_data(str(normalized_sfm), matches_dir, str(out), sfm_type)
    else:
        # No normalization needed
        transform = None
        run_sfm(matches_dir, str(out), sfm_type)

    return out, transform


def _run_sfm_with_data(
    sfm_data_path: str, matches_dir: str, output_dir: str, sfm_type: str
) -> Path:
    """Run SfM using a specific sfm_data.json file."""
    out = Path(output_dir)

    sfm_type = sfm_type.lower()
    if sfm_type not in VALID_SFM_TYPES:
        sfm_type = "incremental"

    sfm_cmd = [
        OPENMVG_SFM_RECONSTRUCT,
        "-i",
        sfm_data_path,
        "-m",
        matches_dir,
        "-o",
        str(out),
    ]

    if sfm_type == "global":
        sfm_cmd.extend(["-s", "GLOBAL"])
    elif sfm_type == "incremental_v2":
        sfm_cmd.extend(["-s", "INCREMENTAL_V2"])

    _run_cmd(sfm_cmd, desc=f"SfM reconstruction ({sfm_type})")
    return out
