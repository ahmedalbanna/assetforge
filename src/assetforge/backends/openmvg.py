"""
OpenMVG backend — wraps the OpenMVG photogrammetry CLI tools.

Provides:
- Feature extraction (SIFT/ORB)
- Feature matching
- Structure from Motion (SfM)
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("assetforge")

# Default OpenMVG commands (adjust if not in PATH)
OPENMVG_SFM = "openMVG_main_SfMInit_ImageListing"
OPENMVG_FEATURES = "openMVG_main_ComputeFeatures"
OPENMVG_MATCHES = "openMVG_main_ComputeMatches"
OPENMVG_SFM_RECONSTRUCT = "openMVG_main_SfM"


def _check_openmvg():
    """Verify OpenMVG tools are available."""
    for cmd in [OPENMVG_SFM, OPENMVG_FEATURES, OPENMVG_MATCHES]:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, check=True)
        except (subprocess.FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                f"OpenMVG tool '{cmd}' not found. Install openMVG: "
                f"https://github.com/openMVG/openMVG"
            )
            return False
    return True


def extract_features(
    input_dir: str, output_dir: str, feature_type: str = "sift"
) -> Path:
    """Extract features from all images in input_dir using OpenMVG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Image listing
    sfm_data = out / "sfm_data.json"
    subprocess.run(
        [
            OPENMVG_SFM,
            "-i",
            input_dir,
            "-o",
            str(out),
            "-d",
            "SIFT" if feature_type.upper() == "SIFT" else "ORB",
        ],
        check=True,
        capture_output=True,
    )

    # Step 2: Compute features
    subprocess.run(
        [
            OPENMVG_FEATURES,
            "-i",
            str(sfm_data),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    # Step 3: Compute matches
    subprocess.run(
        [
            OPENMVG_MATCHES,
            "-i",
            str(sfm_data),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Features extracted to {out}")
    return out


def match_features(features_dir: str, output_dir: str) -> Path:
    """Geometric matching between image pairs.

    Note: In the full pipeline, this is integrated into extract_features().
    This function exists for granular stage control.
    """
    return Path(features_dir)


def run_sfm(matches_dir: str, output_dir: str) -> Path:
    """Run Structure from Motion to compute camera poses and sparse cloud."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sfm_data = Path(matches_dir) / "sfm_data.json"

    subprocess.run(
        [
            OPENMVG_SFM_RECONSTRUCT,
            "-i",
            str(sfm_data),
            "-m",
            matches_dir,
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"SfM reconstruction complete → {out}")
    return out
