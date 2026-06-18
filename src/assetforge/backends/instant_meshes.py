"""
Instant Meshes backend — wrapper for the Instant Meshes CLI tool.

Instant Meshes is an interactive tool for quad-remeshing of
triangle meshes. It can be used as an alternative to Blender's
remesh modifier for high-quality quad topology.

Installation:
    https://github.com/wjakob/instant-meshes
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")


class InstantMeshesBackend:
    """Wrapper around Instant Meshes CLI for quad remeshing."""

    def __init__(self, binary_path: str = "instant-meshes"):
        self.binary_path = binary_path

    def ensure_available(self) -> bool:
        """Check if Instant Meshes is installed."""
        try:
            subprocess.run(
                [self.binary_path, "--help"],
                capture_output=True,
                check=False,
            )
            return True
        except FileNotFoundError:
            return False

    def remesh(
        self,
        input_mesh: str,
        output_mesh: str,
        target_face_count: int = 10000,
        crease_angle: float = 30.0,
        smooth_iterations: int = 5,
        deterministic: bool = True,
    ) -> Path:
        """Apply quad remeshing to a triangle mesh.

        Args:
            input_mesh: Path to input mesh (OBJ)
            output_mesh: Path to output mesh (OBJ)
            target_face_count: Target number of faces
            crease_angle: Crease preservation angle in degrees
            smooth_iterations: Number of smoothing iterations
            deterministic: Use deterministic solver

        Returns:
            Path to the output mesh
        """
        input_path = Path(input_mesh)
        output_path = Path(output_mesh)

        if not input_path.exists():
            raise FileNotFoundError(f"Input mesh not found: {input_path}")

        if not self.ensure_available():
            raise RuntimeError(
                "Instant Meshes not found. Install from:\n"
                "  https://github.com/wjakob/instant-meshes"
            )

        logger.info(
            f"Running Instant Meshes: {input_mesh} → {output_mesh} "
            f"({target_face_count} faces)"
        )

        cmd = [
            self.binary_path,
            str(input_path.resolve()),
            str(output_path.resolve()),
            "--faces",
            str(target_face_count),
            "--crease",
            str(crease_angle),
            "--smooth",
            str(smooth_iterations),
        ]

        if deterministic:
            cmd.append("--deterministic")

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Instant Meshes timed out after 300 seconds")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Instant Meshes failed (exit code {e.returncode}):\n"
                f"{e.stderr.decode() if e.stderr else 'Unknown error'}"
            )

        if not output_path.exists():
            raise RuntimeError(f"Instant Meshes did not produce output: {output_path}")

        logger.info(f"Instant Meshes complete → {output_path}")
        return output_path
