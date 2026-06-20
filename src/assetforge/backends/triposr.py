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
        "scikit-image",
        "imageio",
        "imageio-ffmpeg",
        "rembg",
        "huggingface_hub",
        "transformers>=4.48",
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

        # Fix numpy ABI compatibility — force-reinstall after other deps
        # to ensure all C extensions match the loaded numpy version
        try:
            import numpy

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "-q",
                    f"numpy=={numpy.__version__}",
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except Exception:
            pass  # Non-fatal; may just cause a import error later

    @staticmethod
    def _remap_vit_state_dict(state_dict: dict) -> dict:
        """Remap ViT state dict keys from transformers v4 naming to v5 naming.

        Transformers v5 renamed ViTModel's internal modules:
          v4: encoder.layer.{N}.attention.attention.{proj}.weight
          v5: layers.{N}.attention.{proj}.weight
        """
        import re

        remapped = {}
        for key, value in state_dict.items():
            # Match v4 ViT encoder patterns
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.attention\.attention\.",
                r".layers.\1.attention.",
                key,
            )
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.attention\.output\.",
                r".layers.\1.attention.output.",
                new_key,
            )
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.intermediate\.",
                r".layers.\1.intermediate.",
                new_key,
            )
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.output\.",
                r".layers.\1.output.",
                new_key,
            )
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.layernorm_before\.",
                r".layers.\1.layernorm_before.",
                new_key,
            )
            new_key = re.sub(
                r"\.encoder\.layer\.(\d+)\.layernorm_after\.",
                r".layers.\1.layernorm_after.",
                new_key,
            )
            new_key = re.sub(
                r"\.encoder\.layernorm\.",
                r".layernorm.",
                new_key,
            )
            remapped[new_key] = value
        return remapped

    def _install_mcubes_fallback(self):
        """Create a torchmcubes compatibility module using skimage marching cubes."""
        mcubes_code = """
import torch
import numpy as np
from skimage.measure import marching_cubes as sk_marching_cubes

def marching_cubes(sdf_tensor, isovalue=0.0):
    \"\"\"Marching cubes using skimage as fallback for torchmcubes.\"\"\"
    sdf = sdf_tensor.detach().cpu().numpy() if sdf_tensor.requires_grad else sdf_tensor.cpu().numpy()
    sdf = sdf.squeeze()
    
    # skimage marching cubes
    try:
        vertices, faces, _, _ = sk_marching_cubes(sdf, level=isovalue)
    except Exception:
        return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long)
    
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
        mesh_resolution: int = 256,
        foreground_ratio: float = 0.85,
        chunk_size: int = 131072,
        bake_from_original: bool = True,
    ) -> Path:
        """Generate a 3D mesh from a single image using TripoSR.

        Args:
            input_path: Path to input image
            output_path: Path to save output mesh
            pbr: Whether to generate PBR materials (TripoSR basic support)
            tex_resolution: Texture resolution (1024/2048/4096)
            mesh_resolution: Marching cubes mesh resolution (128/256/512).
                Higher = more geometric detail but slower and more VRAM.
            foreground_ratio: How much of the square canvas the object fills
                (0.5-1.0). Higher = more pixels on the object.
            chunk_size: Renderer chunk size for TripoSR. Lower = less VRAM.
            bake_from_original: Project original image onto mesh as texture.

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
            from transformers import __version__ as transformers_version

            major = int(transformers_version.split(".")[0])
            using_transformers_v5 = major >= 5

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
            if using_transformers_v5:
                from omegaconf import OmegaConf

                cfg = OmegaConf.load(config_path)
                OmegaConf.resolve(cfg)
                model = TSR(cfg)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ckpt = self._remap_vit_state_dict(ckpt)
                missing, unexpected = model.load_state_dict(ckpt, strict=False)
                if missing:
                    logger.warning(
                        f"Missing keys in TripoSR checkpoint (may affect quality): "
                        f"{[k for k in missing if 'image_tokenizer' not in k][:5]}"
                    )
            else:
                model = TSR.from_pretrained(
                    self.MODEL_REPO,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
            model.renderer.set_chunk_size(chunk_size)
            model.to(self.device)
            model.eval()

            # Load and preprocess image
            logger.info("Preprocessing input image...")
            image = Image.open(input_path).convert("RGB")

            # Remove background
            original_image = image.copy()
            try:
                image = remove_background(image)
                image = resize_foreground(image, foreground_ratio)
                # Convert RGBA back to RGB for ViT compatibility
                if image.mode == "RGBA":
                    # Composite onto white background
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
            except Exception:
                import numpy as np

                logger.warning("Background removal failed, using raw image")
                arr = np.array(image).astype(np.float32) / 255.0
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

            logger.info("Extracting mesh (resolution=%d)...", mesh_resolution)
            meshes = model.extract_mesh(
                scene_codes, has_vertex_color=False, resolution=mesh_resolution
            )

            # Export mesh
            output = Path(output_path)
            output_format = output.suffix.lstrip(".")
            if not output_format:
                output_format = "obj"
                output = output.with_suffix(".obj")

            meshes[0].export(str(output))
            logger.info(f"TripoSR mesh saved to: {output}")

            # Optional: project original image onto mesh as texture
            if bake_from_original and not pbr:
                try:
                    logger.info("Projecting original image onto mesh...")
                    tex_dir = output.parent / "textures"
                    tex_dir.mkdir(exist_ok=True)
                    self._project_texture(
                        mesh=meshes[0],
                        original_image=original_image,
                        tex_dir=tex_dir,
                        resolution=tex_resolution,
                    )
                except Exception as e:
                    logger.warning(f"Texture projection failed: {e}")

            # Optional TripoSR native texture baking
            if pbr:
                try:
                    logger.info("Baking PBR texture...")
                    from tsr.bake_texture import bake_texture
                    import xatlas
                    import numpy as np

                    bake_output = bake_texture(
                        meshes[0], model, scene_codes[0], tex_resolution
                    )
                    tex_dir = output.parent / "textures"
                    tex_dir.mkdir(exist_ok=True)

                    xatlas.export(
                        str(output),
                        meshes[0].vertices[bake_output["vmapping"]],
                        bake_output["indices"],
                        bake_output["uvs"],
                        meshes[0].vertex_normals[bake_output["vmapping"]],
                    )
                    texture = (bake_output["colors"] * 255.0).astype(np.uint8)
                    Image.fromarray(texture).transpose(Image.FLIP_TOP_BOTTOM).save(
                        str(tex_dir / "albedo.png")
                    )
                    logger.info(f"PBR textured mesh saved to: {output}")
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

    @staticmethod
    def _project_texture(mesh, original_image, tex_dir, resolution=2048):
        """Project original image onto mesh via simple front-view UV mapping."""
        import numpy as np
        from PIL import Image

        # Simple cylindrical-like projection from front view
        verts = mesh.vertices
        # Normalize vertices to [-1, 1] range
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        vrange = np.maximum(vmax - vmin, 1e-8)
        norm = (verts - vmin) / vrange  # [0, 1]

        # Project front view: use X (right) for U, Y (up) for V
        # TripoSR coordinate system: Y-up, X-right, Z-forward (toward viewer)
        uvs = np.stack([norm[:, 0], norm[:, 1]], axis=1)

        # Flip V for image coordinates
        uvs[:, 1] = 1.0 - uvs[:, 1]

        # Sample image at UV locations
        img = np.array(original_image.convert("RGB").resize((resolution, resolution)))
        h, w = img.shape[:2]
        uv_pixels = np.clip(uvs * [w, h], 0, [w - 1, h - 1]).astype(np.int32)

        # Create texture atlas by interpolating vertex colors
        # Simple approach: create a blank texture and flood-fill from UV map
        # For a proper implementation, we'd use barycentric interpolation
        # For now, create a simple vertex-color-based export
        vertex_colors = img[uv_pixels[:, 1], uv_pixels[:, 0]] / 255.0

        # Save vertex-colored mesh
        import trimesh

        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=vertex_colors,
        )
        colored_mesh.export(str(tex_dir.parent / "mesh_colored.obj"))
        logger.info(
            f"Vertex-colored mesh saved to: {tex_dir.parent / 'mesh_colored.obj'}"
        )
