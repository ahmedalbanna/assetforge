"""
2D Sprite Pipeline — Extracts a game-ready sprite from a single image.

Stages:
1. Background removal (SAM2 or Rembg)
2. Edge cleanup (OpenCV)
3. Vectorization (Rust outline or Python color) [optional]
4. Sprite rasterization at target sizes
5. Sprite sheet packing [optional]
"""

import logging
from pathlib import Path
from typing import Optional

from assetforge.pipelines.base import (
    BasePipeline,
    validate_input_image,
    validate_output_dir,
    validate_sprite_sizes,
    clamp,
)

logger = logging.getLogger("assetforge")

# Valid vectorization modes
VALID_VECTORIZE_MODES = {"outline", "color"}


class Pipeline2D(BasePipeline):
    """Extract and prepare a 2D sprite from an image for game engines."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "./output",
        # Background removal options
        background_remover: str = "rembg",
        bg_model: Optional[str] = None,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        # Sprite options
        sprite_sizes: list[int] | None = None,
        pack_sheet: bool = False,
        feather_pixels: int = 0,
        # Vectorization options
        generate_vector: bool = False,
        vectorize_mode: str = "outline",
        num_colors: int = 24,
        simplify_tolerance: float = 1.5,
        min_area: int = 10,
    ):
        super().__init__(output_dir, name="2d-sprite")
        self.input_path = validate_input_image(input_path, check_exists=False)
        self.background_remover = background_remover
        self.bg_model = bg_model
        self.alpha_matting = alpha_matting
        self.alpha_matting_foreground_threshold = clamp(
            alpha_matting_foreground_threshold,
            0,
            255,
            "alpha_matting_foreground_threshold",
        )
        self.alpha_matting_background_threshold = clamp(
            alpha_matting_background_threshold,
            0,
            255,
            "alpha_matting_background_threshold",
        )
        self.alpha_matting_erode_size = int(
            clamp(alpha_matting_erode_size, 0, 100, "alpha_matting_erode_size")
        )
        self.sprite_sizes = validate_sprite_sizes(sprite_sizes or [64, 128, 256])
        self.pack_sheet = pack_sheet
        self.generate_vector = generate_vector
        self.feather_pixels = int(clamp(feather_pixels, 0, 10, "feather_pixels"))
        self.vectorize_mode = vectorize_mode
        self.num_colors = int(clamp(num_colors, 4, 64, "num_colors"))
        self.simplify_tolerance = clamp(
            simplify_tolerance, 0.5, 5.0, "simplify_tolerance"
        )
        self.min_area = int(clamp(min_area, 1, 10000, "min_area"))
        self.output_dir = validate_output_dir(output_dir)

        # Validate background remover choice
        if background_remover not in ("sam2", "rembg"):
            raise ValueError(
                f"Unknown background remover: '{background_remover}'. "
                f"Use 'sam2' or 'rembg'."
            )

        # Validate vectorize mode
        if vectorize_mode not in VALID_VECTORIZE_MODES:
            raise ValueError(
                f"Unknown vectorize mode: '{vectorize_mode}'. "
                f"Use: {', '.join(sorted(VALID_VECTORIZE_MODES))}."
            )

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
            if self.vectorize_mode == "outline":
                self.run_stage("vectorization", self._vectorize_outline)
            else:
                self.run_stage("vectorization", self._vectorize_color)

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
            bg_model=self.bg_model,
            alpha_matting=self.alpha_matting,
            alpha_matting_foreground_threshold=self.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.alpha_matting_background_threshold,
            alpha_matting_erode_size=self.alpha_matting_erode_size,
        )
        self._foreground_path = output_path
        return [output_path]

    def _cleanup_edges(self) -> list[Path]:
        """Clean up sprite edges using OpenCV."""
        self.require_attr("_foreground_path", "background_removal")
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

    def _vectorize_outline(self) -> list[Path]:
        """Convert cleaned bitmap to SVG vector paths using Rust backend (outline mode)."""
        self.require_attr("_cleaned_path", "edge_cleanup")
        try:
            from assetforge_pyo3 import vectorize_bitmap
        except ImportError:
            logger.warning(
                "Rust PyO3 module not available for outline vectorization. "
                "Falling back to Python color mode (outline)..."
            )
            # Fallback: use Python implementation for outline
            return self._vectorize_color_outline_fallback()

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

    def _vectorize_color_outline_fallback(self) -> list[Path]:
        """Fallback outline vectorization using pure Python (OpenCV)."""
        import cv2
        import numpy as np

        img = cv2.imread(str(self._cleaned_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = binary.shape
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        ]

        for contour in contours:
            if len(contour) < 3:
                continue
            # Simplify
            epsilon = self.simplify_tolerance
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(approx) < 3:
                continue

            # Build path data
            path_data = f"M {approx[0][0][0]} {approx[0][0][1]}"
            for pt in approx[1:]:
                path_data += f" L {pt[0][0]} {pt[0][1]}"
            path_data += " Z"

            svg_parts.append(f'<path d="{path_data}" fill="black" />')

        svg_parts.append("</svg>")
        svg_content = "\n".join(svg_parts)

        output_path = self.output_dir / "sprite.svg"
        with open(output_path, "w") as f:
            f.write(svg_content)

        return [output_path]

    def _vectorize_color(self) -> list[Path]:
        """Convert cleaned bitmap to multi-color SVG using Python color vectorization."""
        self.require_attr("_cleaned_path", "edge_cleanup")

        from assetforge.backends.vectorize import (
            vectorize_color,
            ColorVectorizationConfig,
        )

        import cv2

        img = cv2.imread(str(self._cleaned_path), cv2.IMREAD_UNCHANGED)

        # Convert BGR to RGB for the vectorizer
        if img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        config = ColorVectorizationConfig(
            num_colors=self.num_colors,
            simplify_tolerance=self.simplify_tolerance,
            min_area=self.min_area,
        )

        svg_content = vectorize_color(img_rgb, config)

        output_path = self.output_dir / "sprite.svg"
        with open(output_path, "w") as f:
            f.write(svg_content)

        return [output_path]

    def _rasterize_sprites(self) -> list[Path]:
        """Rasterize the cleaned sprite at each target size."""
        self.require_attr("_cleaned_path", "edge_cleanup")
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
