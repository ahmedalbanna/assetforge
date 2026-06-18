"""
Hunyuan3D-2 backend — generates high-quality 3D meshes with PBR textures.

Uses Tencent's Hunyuan3D-2 model for single-image 3D generation
with superior texture quality.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")


class Hunyuan3DBackend:
    """Wrapper around Hunyuan3D-2 for single-image 3D generation."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

    def generate(
        self,
        input_path: str,
        output_path: str,
        pbr: bool = True,
        tex_resolution: int = 2048,
    ) -> Path:
        """Generate a 3D mesh from a single image with PBR textures.

        Args:
            input_path: Path to input image
            output_path: Path to save output mesh (GLB format)
            pbr: Generate PBR material maps (albedo, normal, roughness, metallic)
            tex_resolution: Texture resolution (e.g., 1024, 2048, 4096, 8192)

        Returns:
            Path to the output mesh
        """
        logger.info(f"Running Hunyuan3D-2 on {input_path}...")

        try:
            from hy3dgen.texgen import HunyuanTexGen
            from hy3dgen.rembg import BackgroundRemover
            from hy3dgen.shapegen import HunyuanShapeGen

            # Stage 1: Remove background from input
            remover = BackgroundRemover()
            image_rgba = remover.remove_background(input_path)

            # Stage 2: Generate 3D shape
            shape_gen = HunyuanShapeGen(device=self.device)
            mesh = shape_gen.generate(image_rgba)

            if pbr:
                # Stage 3: Generate PBR textures
                tex_gen = HunyuanTexGen(device=self.device)
                mesh = tex_gen.generate(
                    mesh,
                    image_rgba,
                    tex_resolution=tex_resolution,
                )

            # Export to GLB
            mesh.export(output_path)
            logger.info(f"Hunyuan3D-2 generation complete → {output_path}")

        except ImportError:
            raise ImportError(
                "Hunyuan3D-2 requires additional packages.\n"
                "Install with: pip install hy3dgen\n"
                "Or see: https://github.com/Tencent/Hunyuan3D-2"
            )

        return Path(output_path)
