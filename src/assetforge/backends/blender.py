"""
Blender headless backend — wraps Blender CLI operations for:
- Mesh decimation
- Quad remeshing
- Texture baking
- Format conversion
- Mesh cleanup
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")

# Path to built-in Blender scripts
BLENDER_SCRIPTS_DIR = Path(__file__).parent / "blender_scripts"


class BlenderBackend:
    """Wrapper around Blender's headless CLI mode."""

    def __init__(self, blender_path: str = "blender"):
        self.blender_path = blender_path

    def _run_script(
        self, script_name: str, args: list[str]
    ) -> subprocess.CompletedProcess:
        """Run a Blender Python script in headless mode."""
        script_path = BLENDER_SCRIPTS_DIR / script_name

        if not script_path.exists():
            raise FileNotFoundError(
                f"Blender script not found: {script_path}. "
                f"Ensure {script_name} exists in {BLENDER_SCRIPTS_DIR}"
            )

        cmd = [
            self.blender_path,
            "--background",
            "--python",
            str(script_path),
            "--",
            *args,
        ]

        logger.debug(f"Running Blender: {' '.join(cmd)}")

        # Sanitize environment to avoid conda/PYTHONPATH conflicts with Blender's bundled Python
        clean_env = os.environ.copy()
        for var in ["PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"]:
            clean_env.pop(var, None)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # 10 minute timeout
                env=clean_env,
            )
            if result.stdout:
                logger.debug(f"Blender stdout: {result.stdout[:500]}")
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Blender script '{script_name}' timed out after 600s")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Blender script '{script_name}' failed:\n"
                f"stdout: {e.stdout[:500]}\n"
                f"stderr: {e.stderr[:500]}"
            )

    def ensure_available(self) -> bool:
        """Check if Blender is installed and accessible."""
        try:
            result = subprocess.run(
                [self.blender_path, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            version = (
                result.stdout.strip().split("\n")[0] if result.stdout else "unknown"
            )
            logger.info(f"Blender detected: {version}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                "Blender not found. Install Blender and add to PATH, "
                "or set ASSETFORGE_BLENDER_PATH environment variable."
            )
            return False

    def apply_decimate(
        self, input_mesh: str, output_mesh: str, ratio: float = 0.5
    ) -> Path:
        """Reduce polygon count using Blender's Decimate modifier."""
        self._run_script(
            "blender_decimate.py", [str(input_mesh), str(output_mesh), str(ratio)]
        )
        return Path(output_mesh)

    def apply_remesh(
        self, input_mesh: str, output_mesh: str, target_faces: int
    ) -> Path:
        """Apply quad remeshing."""
        self._run_script(
            "blender_remesh.py", [str(input_mesh), str(output_mesh), str(target_faces)]
        )
        return Path(output_mesh)

    def run_cleanup(self, input_mesh: str, output_mesh: str) -> Path:
        """Clean up mesh: remove floaters, close holes, smooth."""
        self._run_script("blender_cleanup.py", [str(input_mesh), str(output_mesh)])
        return Path(output_mesh)

    def bake_textures(
        self, low_poly: str, high_poly: str, output_dir: str, resolution: int = 2048
    ) -> Path:
        """Bake high-poly details to low-poly PBR textures."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._run_script(
            "blender_bake.py",
            [str(low_poly), str(high_poly), str(out), str(resolution)],
        )
        return out

    def export_format(self, input_mesh: str, output_path: str) -> Path:
        """Export mesh in the specified format (auto-detected from extension)."""
        self._run_script("blender_export.py", [str(input_mesh), str(output_path)])
        return Path(output_path)
