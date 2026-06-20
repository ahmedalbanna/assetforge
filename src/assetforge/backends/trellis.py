"""
TRELLIS backend — generates high-quality 3D meshes with splat representation.

Uses Microsoft's TRELLIS 2 model for single-image 3D generation
with hybrid representation (mesh + splats + PBR materials).
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")


class TRELLISBackend:
    """Wrapper around TRELLIS 2 for single-image 3D generation."""

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
        guidance_scale: float = 7.5,
        inference_steps: int = 50,
        **kwargs,
    ) -> Path:
        """Generate a 3D asset from a single image using TRELLIS 2.

        Args:
            input_path: Path to input image
            output_path: Path to save output asset
            pbr: Generate PBR material maps
            tex_resolution: Texture resolution (1024/2048/4096)
            guidance_scale: Classifier-free guidance scale (3.0-15.0).
                Higher = more faithful to prompt/image, may lose diversity.
            inference_steps: Number of denoising steps (10-100).
                Higher = better quality but slower.

        Returns:
            Path to the output asset
        """
        logger.info(
            f"Running TRELLIS 2 on {input_path} "
            f"(guidance={guidance_scale}, steps={inference_steps})..."
        )

        try:
            import torch
            from PIL import Image
            from trellis2 import Trellis2Pipeline

            # Load pipeline
            pipe = Trellis2Pipeline.from_pretrained("microsoft/TRELLIS-2")
            pipe = pipe.to(self.device)

            # Load image
            image = Image.open(input_path).convert("RGB")

            # Run generation with quality parameters
            result = pipe(
                image,
                generate_mesh=True,
                generate_splats=True,
                generate_pbr=pbr,
                tex_resolution=tex_resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
            )

            # Export as GLB (contains mesh, textures, and optionally splats)
            result.export_glb(output_path)
            logger.info(f"TRELLIS 2 generation complete → {output_path}")

        except ImportError:
            raise ImportError(
                "TRELLIS 2 requires additional packages.\n"
                "Install with: pip install trellis2\n"
                "Or see: https://github.com/microsoft/TRELLIS"
            )

        return Path(output_path)
