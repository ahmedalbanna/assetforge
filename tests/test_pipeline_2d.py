"""
Integration tests for the 2D sprite pipeline.

Tests the full pipeline end-to-end using synthetic test images.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from assetforge.pipelines.pipeline_2d import Pipeline2D


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a simple test image with a colored square on a transparent background."""
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    # Draw a red square in the center
    pixels = np.array(img)
    pixels[32:96, 32:96] = [255, 0, 0, 255]  # Red opaque square
    img = Image.fromarray(pixels)

    path = tmp_path / "test_input.png"
    img.save(path)
    return path


def test_pipeline_2d_basic(tmp_path: Path, test_image: Path):
    """Test the 2D pipeline runs without errors."""
    output_dir = tmp_path / "output"

    pipeline = Pipeline2D(
        input_path=str(test_image),
        output_dir=str(output_dir),
        background_remover="rembg",
        sprite_sizes=[64, 128],
        pack_sheet=False,
        generate_vector=False,
    )
    results = pipeline.run()

    # Check all stages completed successfully
    assert all(r.success for r in results), (
        f"Pipeline stage failed: {[r for r in results if not r.success]}"
    )

    # Check output files exist
    assert (output_dir / "sprite_64x64.png").exists()
    assert (output_dir / "sprite_128x128.png").exists()


def test_pipeline_2d_with_spritesheet(tmp_path: Path, test_image: Path):
    """Test the 2D pipeline with sprite sheet generation."""
    output_dir = tmp_path / "output_sheet"

    pipeline = Pipeline2D(
        input_path=str(test_image),
        output_dir=str(output_dir),
        background_remover="rembg",
        sprite_sizes=[32, 64],
        pack_sheet=True,
    )
    results = pipeline.run()

    assert all(r.success for r in results)
    assert (output_dir / "spritesheet.png").exists()
    assert (output_dir / "spritesheet.json").exists()


def test_pipeline_2d_invalid_input(tmp_path: Path):
    """Test error handling with non-existent input."""
    pipeline = Pipeline2D(
        input_path=str(tmp_path / "nonexistent.png"),
        output_dir=str(tmp_path / "output"),
    )
    results = pipeline.run()
    # Should not crash — pipeline returns empty list early for invalid input
    assert len(results) == 0


def test_pipeline_2d_edge_cleanup(tmp_path: Path, test_image: Path):
    """Test edge cleanup with feathering."""
    output_dir = tmp_path / "output_feather"

    pipeline = Pipeline2D(
        input_path=str(test_image),
        output_dir=str(output_dir),
        background_remover="rembg",
        feather_pixels=3,
    )
    results = pipeline.run()

    assert all(r.success for r in results)
    assert (output_dir / "sprite_64x64.png").exists()
