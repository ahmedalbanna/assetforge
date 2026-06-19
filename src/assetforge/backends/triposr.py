"""
TripoSR backend — generates 3D meshes from single images.

Uses the stabilityai/TripoSR model from HuggingFace for fast
single-image 3D reconstruction.

Model code: https://github.com/VAST-AI-Research/TripoSR
Model weights: https://huggingface.co/stabilityai/TripoSR

Installation (automated):
    python -c "from assetforge.backends.triposr import TripoSRBackend; b = TripoSRBackend(); b.generate('input.png', 'output.glb')"
"""

import logging
import sys
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")

TRIPOSR_REPO = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_DIR = Path.home() / ".cache" / "assetforge" / "triposr_repo"


class TripoSRBackend:
    """Wrapper around TripoSR for single-image 3D generation.

    Automatically downloads the model code from GitHub on first use.
    """

    MODEL_REPO = "stabilityai/TripoSR"
    MODEL_CONFIG = "config.yaml"
    MODEL_CKPT = "model.ckpt"

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._repo_ready = False
        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    TRIPOSR_DEPS = [
        "omegaconf>=2.3",
        "einops>=0.7",
        "trimesh>=4.0",
        "imageio",
        "imageio-ffmpeg",
        "rembg",
        "huggingface_hub",
        "transformers",
        "xatlas",
    ]

    def _ensure_repo(self):
        """Clone TripoSR repo and install dependencies if needed."""
        if self._repo_ready:
            return True

        # Check if tsr is already importable
        try:
            import tsr  # noqa

            self._repo_ready = True
            return True
        except ImportError:
            pass

        # Clone the repo
        if not TRIPOSR_DIR.exists():
            logger.info(f"Cloning TripoSR from {TRIPOSR_REPO}...")
            TRIPOSR_DIR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth=1", TRIPOSR_REPO, str(TRIPOSR_DIR)],
                check=True,
                capture_output=True,
            )
            logger.info("TripoSR repo cloned")

        # Add to path
        if str(TRIPOSR_DIR) not in sys.path:
            sys.path.insert(0, str(TRIPOSR_DIR))

        # Install Python dependencies automatically
        self._install_deps()

        # Create torchmcubes compatibility layer if needed
        try:
            import torchmcubes  # noqa
        except ImportError:
            self._install_mcubes_fallback()

        try:
            import tsr  # noqa

            self._repo_ready = True
            return True
        except ImportError as e:
            logger.error(f"Failed to import tsr module: {e}")
            return False

    def _install_deps(self):
        """Install required Python packages via pip."""
        missing = []
        for dep in self.TRIPOSR_DEPS:
            pkg_name = dep.split(">=")[0].split("==")[0].strip()
            try:
                __import__(pkg_name.replace("-", "_"))
            except ImportError:
                missing.append(dep)

        if not missing:
            return

        logger.info(f"Installing TripoSR dependencies: {', '.join(missing)}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + missing,
                check=True,
                capture_output=True,
                timeout=120,
            )
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
            logger.warning(
                f"Some dependencies failed to install:\n{stderr}\n"
                f"Try manually: pip install {' '.join(missing)}"
            )
        except subprocess.TimeoutExpired:
            logger.warning("Dependency install timed out. Try manually.")

    def _install_mcubes_fallback(self):
        """Create a torchmcubes compatibility module using trimesh marching cubes."""
        mcubes_code = """
import torch
import numpy as np
import trimesh

def marching_cubes(sdf_tensor, isovalue=0.0):
    \"\"\"Marching cubes using trimesh as fallback.\"\"\"
    sdf = sdf_tensor.detach().cpu().numpy() if sdf_tensor.requires_grad else sdf_tensor.cpu().numpy()
    
    # trimesh marching cubes
    vertices, faces, _, _ = trimesh.voxel.marching_cubes(
        sdf.squeeze(), 
        pitch=1.0,
    )
    
    if len(vertices) == 0 or len(faces) == 0:
        return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long)
    
    vertices_tensor = torch.from_numpy(vertices.astype(np.float32))
    faces_tensor = torch.from_numpy(faces.astype(np.int64))
    
    return vertices_tensor, faces_tensor
"""
        import types

        mod = types.ModuleType("torchmcubes")
        exec(mcubes_code, mod.__dict__)
        sys.modules["torchmcubes"] = mod
        logger.info("Installed torchmcubes fallback (trimesh-based marching cubes)")

    def generate(
        self,
        input_path: str,
        output_path: str,
        pbr: bool = False,
        tex_resolution: int = 2048,
    ) -> Path:
        """Generate a 3D mesh from a single image using TripoSR.

        Args:
            input_path: Path to input image
            output_path: Path to save output mesh
            pbr: Whether to generate PBR materials (TripoSR basic support)
            tex_resolution: Texture resolution

        Returns:
            Path to the output mesh
        """
        logger.info(f"Running TripoSR on {input_path}...")

        # Ensure repo is available
        if not self._ensure_repo():
            raise RuntimeError(
                "Failed to set up TripoSR. "
                f"Clone manually: git clone {TRIPOSR_REPO} "
                "and ensure dependencies: pip install omegaconf einops trimesh imageio"
            )

        try:
            import torch
            from PIL import Image
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground
            from huggingface_hub import hf_hub_download

            # Download model weights
            ckpt_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_CKPT,
            )
            config_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_CONFIG,
            )

            logger.info("Loading TripoSR model...")
            model = TSR.from_pretrained(
                self.MODEL_REPO,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(131072)
            model.to(self.device)
            model.eval()

            # Load and preprocess image
            logger.info("Preprocessing input image...")
            image = Image.open(input_path).convert("RGB")

            # Remove background
            try:
                image = remove_background(image)
                image = resize_foreground(image, 0.8)
            except Exception:
                import numpy as np

                logger.warning("Background removal failed, using raw image")
                # Convert to expected input format
                arr = np.array(image).astype(np.float32) / 255.0
                # Pad to square with white background
                size = max(arr.shape[0], arr.shape[1])
                square = np.ones((size, size, 3), dtype=np.float32)
                y_off = (size - arr.shape[0]) // 2
                x_off = (size - arr.shape[1]) // 2
                square[y_off : y_off + arr.shape[0], x_off : x_off + arr.shape[1]] = (
                    arr[:, :, :3]
                )
                image = Image.fromarray((square * 255).astype(np.uint8))

            # Run inference
            logger.info("Running TripoSR inference...")
            with torch.no_grad():
                scene_codes = model([image], device=self.device)

            logger.info("Extracting mesh...")
            meshes = model.extract_mesh(scene_codes, resolution=256)

            # Export mesh
            output = Path(output_path)
            output_format = output.suffix.lstrip(".")
            if not output_format:
                output_format = "obj"
                output = output.with_suffix(".obj")

            meshes[0].export(str(output))
            logger.info(f"TripoSR mesh saved to: {output}")

            # Optional texture baking
            if pbr:
                try:
                    logger.info("Baking texture...")
                    from tsr.bake_texture import bake_texture
                    import xatlas
                    import numpy as np

                    bake_output = bake_texture(
                        meshes[0], model, scene_codes[0], tex_resolution
                    )
                    tex_dir = output.parent / "textures"
                    tex_dir.mkdir(exist_ok=True)

                    # Export textured mesh
                    xatlas.export(
                        str(output),
                        meshes[0].vertices[bake_output["vmapping"]],
                        bake_output["indices"],
                        bake_output["uvs"],
                        meshes[0].vertex_normals[bake_output["vmapping"]],
                    )
                    # Save texture
                    texture = (bake_output["colors"] * 255.0).astype(np.uint8)
                    Image.fromarray(texture).transpose(Image.FLIP_TOP_BOTTOM).save(
                        str(tex_dir / "albedo.png")
                    )
                    logger.info(f"Textured mesh saved to: {output}")
                except Exception as e:
                    logger.warning(f"Texture baking failed: {e}")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                f"Install dependencies:\n"
                f"  pip install omegaconf einops trimesh imageio imageio-ffmpeg\n"
                f"  pip install git+https://github.com/VAST-AI-Research/TripoSR.git\n"
            )
        except Exception as e:
            raise RuntimeError(f"TripoSR inference failed: {e}")

        return Path(output_path)
