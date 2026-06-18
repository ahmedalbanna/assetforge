"""
2D Sprite Pipeline — Extracts a game-ready sprite from a single image.

Stages:
1. Background removal (SAM2 or Rembg)
2. Edge cleanup (OpenCV)
3. Vectorization (Rust via PyO3) [optional]
4. Sprite rasterization at target sizes
5. Sprite sheet packing [optional]
"""

import logging
from pathlib import Path
from typing import Optional

from assetforge.pipelines.base import BasePipeline

logger = logging.getLogger("assetforge")


class Pipeline2D(BasePipeline):
    """Extract and prepare a 2D sprite from an image for game engines."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "./output",
        background_remover: str = "sam2",
        sprite_sizes: list[int] | None = None,
        pack_sheet: bool = False,
        generate_vector: bool = False,
        feather_pixels: int = 0,
    ):
        super().__init__(output_dir, name="2d-sprite")
        self.input_path = Path(input_path).resolve()
        self.background_remover = background_remover
        self.sprite_sizes = sprite_sizes or [64, 128, 256]
        self.pack_sheet = pack_sheet
        self.generate_vector = generate_vector
        self.feather_pixels = feather_pixels

        # Intermediate directories
        self.stages_dir = self.output_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the full 2D sprite pipeline."""
        logger.info(f"=== 2D Sprite Pipeline ===")
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Sprite sizes: {self.sprite_sizes}")

        # Validate input
        if not self.input_path.exists():
            logger.error(f"Input file not found: {self.input_path}")
            return self.stages

        # Stage 1: Background removal
        self.run_stage("background_removal", self._remove_background)

        # Stage 2: Edge cleanup
        self.run_stage("edge_cleanup", self._cleanup_edges)

        # Stage 3: Vectorization (optional)
        if self.generate_vector:
            self.run_stage("vectorization", self._vectorize)

        # Stage 4: Rasterize sprites
        self.run_stage("sprite_rasterization", self._rasterize_sprites)

        # Stage 5: Sprite sheet (optional)
        if self.pack_sheet:
            self.run_stage("sprite_sheet", self._pack_sprite_sheet)

        self.summary()
        return self.stages

    def _remove_background(self) -> list[Path]:
        """Remove background from input image using SAM2 or Rembg."""
        from assetforge.backends.segmentation import remove_background

        output_path = self.stages_dir / "01_foreground.png"
        remove_background(
            str(self.input_path),
            str(output_path),
            model=self.background_remover,
        )
        self._foreground_path = output_path
        return [output_path]

    def _cleanup_edges(self) -> list[Path]:
        """Clean up sprite edges using OpenCV."""
        import cv2
        import numpy as np

        input_img = cv2.imread(str(self._foreground_path), cv2.IMREAD_UNCHANGED)

        if self.feather_pixels > 0 and input_img.shape[2] == 4:
            alpha = input_img[:, :, 3]
            kernel_size = self.feather_pixels * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
            input_img[:, :, 3] = alpha

        output_path = self.stages_dir / "02_cleaned.png"
        cv2.imwrite(str(output_path), input_img)
        self._cleaned_path = output_path
        return [output_path]

    def _vectorize(self) -> list[Path]:
        """Convert cleaned bitmap to SVG vector paths using Rust backend."""
        try:
            from assetforge_pyo3 import vectorize_bitmap
        except ImportError:
            logger.warning("Rust PyO3 module not available. Skipping vectorization.")
            return []

        import cv2
        import numpy as np

        img = cv2.imread(str(self._cleaned_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        svg_content = vectorize_bitmap(
            binary.tobytes(),
            binary.shape[1],
            binary.shape[0],
            threshold=128,
            turdsize=2,
        )

        output_path = self.output_dir / "sprite.svg"
        with open(output_path, "w") as f:
            f.write(svg_content)

        return [output_path]

    def _rasterize_sprites(self) -> list[Path]:
        """Rasterize the cleaned sprite at each target size."""
        from PIL import Image

        sprite = Image.open(self._cleaned_path).convert("RGBA")
        output_paths = []

        for size in self.sprite_sizes:
            resized = sprite.resize((size, size), Image.LANCZOS)
            output_path = self.output_dir / f"sprite_{size}x{size}.png"
            resized.save(output_path, "PNG")
            output_paths.append(output_path)

        return output_paths

    def _pack_sprite_sheet(self) -> list[Path]:
        """Pack all sprite sizes into a single sprite sheet."""
        from PIL import Image

        sprite_images = []
        for size in self.sprite_sizes:
            path = self.output_dir / f"sprite_{size}x{size}.png"
            if path.exists():
                sprite_images.append(Image.open(path).convert("RGBA"))

        if not sprite_images:
            logger.warning("No sprites to pack into sheet.")
            return []

        # Simple horizontal strip layout
        total_width = sum(s.width for s in sprite_images)
        max_height = max(s.height for s in sprite_images)

        sheet = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))
        x_offset = 0
        for s in sprite_images:
            sheet.paste(s, (x_offset, 0), s)
            x_offset += s.width

        output_path = self.output_dir / "spritesheet.png"
        sheet.save(output_path, "PNG")

        # Generate atlas metadata
        import json

        atlas = {"frames": [], "meta": {"image": "spritesheet.png"}}
        x_offset = 0
        for i, size in enumerate(self.sprite_sizes):
            atlas["frames"].append(
                {
                    "filename": f"sprite_{size}x{size}.png",
                    "frame": {"x": x_offset, "y": 0, "w": size, "h": size},
                }
            )
            x_offset += size

        atlas_path = self.output_dir / "spritesheet.json"
        with open(atlas_path, "w") as f:
            json.dump(atlas, f, indent=2)

        return [output_path, atlas_path]
