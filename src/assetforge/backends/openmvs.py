"""
OpenMVS backend — wraps the OpenMVS multi-view stereo CLI tools.

Provides:
- Dense point cloud reconstruction
- Mesh reconstruction
- Texture mapping
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("assetforge")

# OpenMVS commands
OPENMVS_DENSE = "OpenMVS_DensifyPointCloud"
OPENMVS_RECONSTRUCT = "OpenMVS_ReconstructMesh"
OPENMVS_TEXTURE = "OpenMVS_TextureMesh"


def _check_openmvs():
    """Verify OpenMVS tools are available."""
    for cmd in [OPENMVS_DENSE, OPENMVS_RECONSTRUCT, OPENMVS_TEXTURE]:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, check=True)
        except (subprocess.FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                f"OpenMVS tool '{cmd}' not found. Install openMVS: "
                f"https://github.com/cdcseacave/openMVS"
            )
            return False
    return True


def run_mvs(sfm_dir: str, output_dir: str, quality: str = "high") -> Path:
    """Run Multi-View Stereo to produce dense point cloud."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert OpenMVG output to OpenMVS format
    scene_mvs = out / "scene.mvs"
    subprocess.run(
        [
            "openMVG_main_openMVG2openMVS",
            "-i",
            str(Path(sfm_dir) / "sfm_data.json"),
            "-o",
            str(scene_mvs),
        ],
        check=True,
        capture_output=True,
    )

    # Densify point cloud
    subprocess.run(
        [
            OPENMVS_DENSE,
            str(scene_mvs),
            "-w",
            str(out),
            "--resolution-level",
            "2" if quality == "high" else "1",
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Multi-View Stereo complete → {out}")
    return out


def reconstruct_mesh(mvs_dir: str, output_path: str) -> Path:
    """Reconstruct mesh surface from dense point cloud."""
    dense_scene = Path(mvs_dir) / "scene_dense.mvs"

    subprocess.run(
        [
            OPENMVS_RECONSTRUCT,
            str(dense_scene),
            "-w",
            str(mvs_dir),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Mesh reconstruction complete → {output_path}")
    return Path(output_path)


def map_textures(mesh_path: str, mvs_dir: str, output_path: str) -> Path:
    """Project photo textures onto the mesh."""
    subprocess.run(
        [
            OPENMVS_TEXTURE,
            str(Path(mvs_dir) / "scene_dense_mesh.mvs"),
            "-w",
            str(Path(output_path).parent),
        ],
        check=True,
        capture_output=True,
    )

    logger.info(f"Texture mapping complete → {output_path}")
    return Path(output_path)
