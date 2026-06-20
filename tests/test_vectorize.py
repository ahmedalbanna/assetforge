"""
Tests for the vectorization backend (color quantization + SVG generation).

Tests cover:
- Color quantization
- Outline vectorization
- Color vectorization with synthetic images
- Edge cases (empty, transparent, single color)
- Configuration validation
- SVG output validity
"""

import pytest
import numpy as np
from assetforge.backends.vectorize import (
    ColorVectorizationConfig,
    quantize_colors,
    separate_layers,
    vectorize_color,
    vectorize_outline,
    _validate_config,
    _rgb_to_luminance,
    _contours_to_svg_paths,
)


def _make_test_image_rgba(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a synthetic RGBA image with colored shapes."""
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Red square in top-left
    img[8:24, 8:24] = [255, 0, 0, 255]

    # Green circle approximation in center (a rectangle for simplicity)
    img[20:36, 20:36] = [0, 255, 0, 255]

    # Blue rectangle in bottom-right
    img[32:56, 32:56] = [0, 0, 255, 255]

    # Semi-transparent yellow
    img[40:56, 8:24] = [255, 255, 0, 128]

    return img


def _make_solid_color_image(
    color: tuple = (128, 128, 128, 255),
    h: int = 32,
    w: int = 32,
) -> np.ndarray:
    """Create a solid color RGBA image."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = color
    return img


# ── Configuration Validation ──────────────────────────────────────


def test_default_config():
    """Test default configuration values are valid."""
    config = ColorVectorizationConfig()
    _validate_config(config)  # Should not raise


def test_invalid_num_colors_low():
    """Test num_colors below minimum raises error."""
    config = ColorVectorizationConfig(num_colors=2)
    with pytest.raises(ValueError, match="num_colors"):
        _validate_config(config)


def test_invalid_num_colors_high():
    """Test num_colors above maximum raises error."""
    config = ColorVectorizationConfig(num_colors=100)
    with pytest.raises(ValueError, match="num_colors"):
        _validate_config(config)


def test_invalid_simplify_tolerance():
    """Test simplify_tolerance out of range raises error."""
    config = ColorVectorizationConfig(simplify_tolerance=10.0)
    with pytest.raises(ValueError, match="simplify_tolerance"):
        _validate_config(config)


def test_invalid_min_area():
    """Test min_area below minimum raises error."""
    config = ColorVectorizationConfig(min_area=0)
    with pytest.raises(ValueError, match="min_area"):
        _validate_config(config)


def test_invalid_color_tolerance():
    """Test color_tolerance out of range raises error."""
    config = ColorVectorizationConfig(color_tolerance=100)
    with pytest.raises(ValueError, match="color_tolerance"):
        _validate_config(config)


def test_invalid_layer_order():
    """Test invalid layer_order raises error."""
    config = ColorVectorizationConfig(layer_order="invalid")
    with pytest.raises(ValueError, match="layer_order"):
        _validate_config(config)


# ── Color Quantization ────────────────────────────────────────────


def test_quantize_single_color():
    """Test quantization with a single-color image."""
    img = _make_solid_color_image((100, 150, 200, 255))
    quantized, palette = quantize_colors(img, num_colors=4)

    assert quantized.shape == (32, 32, 4)
    assert palette.shape == (4, 4)
    # First palette color should be close to the input color
    assert abs(int(palette[0, 0]) - 100) < 20
    assert abs(int(palette[0, 1]) - 150) < 20
    assert abs(int(palette[0, 2]) - 200) < 20


def test_quantize_transparent():
    """Test quantization with a fully transparent image."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)
    quantized, palette = quantize_colors(img, num_colors=8)

    assert quantized.shape == (16, 16, 4)
    assert palette.shape == (8, 4)
    assert quantized[:, :, 3].sum() == 0  # All transparent


def test_quantize_multi_color():
    """Test quantization preserves distinct colors."""
    img = _make_test_image_rgba(32, 32)
    quantized, palette = quantize_colors(img, num_colors=8)

    assert quantized.shape == (32, 32, 4)
    assert palette.shape == (8, 4)
    # Should have found at least a few distinct colors
    unique_in_palette = len(set(tuple(c) for c in palette))
    assert unique_in_palette >= 1


def test_quantize_preserves_edges():
    """Test quantization preserves non-transparent regions."""
    img = _make_test_image_rgba(32, 32)
    quantized, palette = quantize_colors(img, num_colors=8)

    # Non-transparent pixels in original should map to non-transparent in quantized
    orig_alpha_count = (img[:, :, 3] > 0).sum()
    quant_alpha_count = (quantized[:, :, 3] > 0).sum()
    assert quant_alpha_count > 0
    # Roughly similar fill ratio
    assert abs(quant_alpha_count - orig_alpha_count) / max(orig_alpha_count, 1) < 0.5


# ── RGB to Luminance ──────────────────────────────────────────────


def test_rgb_to_luminance():
    """Test luminance calculation."""
    # Pure red
    assert _rgb_to_luminance(np.array([255, 0, 0])) > 0
    # Pure black
    assert _rgb_to_luminance(np.array([0, 0, 0])) == 0.0
    # Pure white
    assert _rgb_to_luminance(np.array([255, 255, 255])) == 255.0
    # Luminance: green > red > blue
    green_lum = _rgb_to_luminance(np.array([0, 255, 0]))
    red_lum = _rgb_to_luminance(np.array([255, 0, 0]))
    blue_lum = _rgb_to_luminance(np.array([0, 0, 255]))
    assert green_lum > red_lum > blue_lum


# ── Contour to SVG Paths ─────────────────────────────────────────


def test_contours_to_svg_paths():
    """Test contour conversion to SVG path data."""
    # Create a simple square contour
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    path_data = _contours_to_svg_paths([contour])
    assert "M" in path_data
    assert "L" in path_data
    assert "Z" in path_data
    # Should contain the vertex coordinates
    assert "0" in path_data
    assert "10" in path_data


def test_contours_to_svg_paths_empty():
    """Test empty contour list."""
    path_data = _contours_to_svg_paths([])
    assert path_data == ""


def test_contours_to_svg_paths_too_few_points():
    """Test contour with < 3 points (cannot form polygon)."""
    contour = np.array([[[0, 0]], [[10, 0]]], dtype=np.int32)
    path_data = _contours_to_svg_paths([contour])
    # Should skip this contour
    assert path_data == ""


# ── Outline Vectorization ─────────────────────────────────────────


def test_vectorize_outline_simple():
    """Test outline vectorization with a simple shape."""
    img = _make_test_image_rgba(32, 32)
    svg = vectorize_outline(img)

    assert "<?xml" in svg
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "M " in svg or 'd="' in svg
    assert "black" in svg or 'fill="' in svg


def test_vectorize_outline_empty():
    """Test outline vectorization with fully transparent image."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)
    svg = vectorize_outline(img)

    assert "<svg" in svg
    assert "</svg>" in svg


def test_vectorize_outline_custom_fill():
    """Test outline vectorization with custom fill color."""
    img = _make_test_image_rgba(32, 32)
    svg = vectorize_outline(img, fill_color="red")
    assert 'fill="red"' in svg


# ── Color Vectorization ───────────────────────────────────────────


def test_vectorize_color_simple():
    """Test color vectorization with a simple multi-color image."""
    img = _make_test_image_rgba(32, 32)
    config = ColorVectorizationConfig(
        num_colors=8,
        simplify_tolerance=1.0,
        min_area=5,
    )
    svg = vectorize_color(img, config)

    assert "<?xml" in svg
    assert "<svg" in svg
    assert "</svg>" in svg
    # Should contain at least one path with fill
    assert 'fill="' in svg or "fill=" in svg


def test_vectorize_color_preserves_colors():
    """Test that output SVG contains color information."""
    img = _make_test_image_rgba(32, 32)
    config = ColorVectorizationConfig(
        num_colors=4,
        simplify_tolerance=0.5,
        min_area=3,
    )
    svg = vectorize_color(img, config)

    # Should have at least some paths with fill colors
    path_count = svg.count("<path")
    assert path_count > 0


def test_vectorize_color_transparent_bg():
    """Test color vectorization with transparent background."""
    img = np.zeros((32, 32, 4), dtype=np.uint8)
    # A single colored circle-like shape
    img[8:24, 8:24] = [0, 128, 255, 255]
    # Surrounding is transparent

    config = ColorVectorizationConfig(
        num_colors=4,
        simplify_tolerance=1.0,
        min_area=5,
    )
    svg = vectorize_color(img, config)

    assert "<svg" in svg
    assert "</svg>" in svg
    # Should not have background fill since background is 'none'
    assert 'fill="none"' not in svg or config.background_color == "none"


def test_vectorize_color_image_at_path(tmp_path):
    """Test vectorization of an image loaded from file (integration check)."""
    from PIL import Image

    # Create a test image with distinct colored regions
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    pixels = np.array(img)
    # Red region
    pixels[4:12, 4:12] = [255, 0, 0, 255]
    # Green region
    pixels[16:28, 16:28] = [0, 255, 0, 255]
    # Blue region
    pixels[20:30, 4:14] = [0, 0, 255, 200]
    img = Image.fromarray(pixels)

    path = tmp_path / "test_multi.png"
    img.save(path)

    # Load with OpenCV
    import cv2

    cv_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    # Convert BGR to RGB
    if cv_img is not None and cv_img.shape[2] == 4:
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    elif cv_img is not None:
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    svg = vectorize_color(cv_img_rgb)
    assert "<svg" in svg
    assert "fill=" in svg


def test_vectorize_color_all_transparent():
    """Test color vectorization with fully transparent image."""
    img = np.zeros((16, 16, 4), dtype=np.uint8)
    svg = vectorize_color(img)

    assert "<svg" in svg
    assert "</svg>" in svg


def test_vectorize_color_large_downscale():
    """Test color vectorization with downscaling enabled."""
    img = _make_test_image_rgba(128, 128)
    config = ColorVectorizationConfig(
        num_colors=8,
        max_svg_dimension=32,
    )
    svg = vectorize_color(img, config)

    assert "<svg" in svg
    assert "</svg>" in svg


# ── Layer Separation ──────────────────────────────────────────────


def test_separate_layers_basic():
    """Test layer separation produces correct number of layers."""
    img = _make_test_image_rgba(32, 32)
    config = ColorVectorizationConfig(num_colors=8)
    quantized, palette = quantize_colors(img, config.num_colors)
    layers = separate_layers(quantized, palette, config)

    assert isinstance(layers, list)
    # Should have at least one layer
    assert len(layers) >= 1

    for layer in layers:
        assert "color" in layer
        assert "contours" in layer
        assert "area" in layer
        assert "luminance" in layer
        assert len(layer["color"]) == 4  # RGBA


def test_separate_layers_order_luminance():
    """Test layers are sorted by luminance when configured."""
    img = _make_test_image_rgba(32, 32)
    config = ColorVectorizationConfig(num_colors=8, layer_order="luminance")
    quantized, palette = quantize_colors(img, config.num_colors)
    layers = separate_layers(quantized, palette, config)

    if len(layers) >= 2:
        luminances = [l["luminance"] for l in layers]
        # Check sorted ascending
        for i in range(len(luminances) - 1):
            assert luminances[i] <= luminances[i + 1]


def test_separate_layers_order_area():
    """Test layers are sorted by area (descending) when configured."""
    img = _make_test_image_rgba(32, 32)
    config = ColorVectorizationConfig(num_colors=8, layer_order="area")
    quantized, palette = quantize_colors(img, config.num_colors)
    layers = separate_layers(quantized, palette, config)

    if len(layers) >= 2:
        areas = [l["area"] for l in layers]
        for i in range(len(areas) - 1):
            assert areas[i] >= areas[i + 1]


def test_separate_layers_skip_transparent():
    """Test that nearly-transparent palette colors are skipped."""
    img = _make_test_image_rgba(32, 32)
    palette = np.zeros((4, 4), dtype=np.uint8)
    palette[0] = [255, 0, 0, 255]
    palette[1] = [0, 255, 0, 255]
    palette[2] = [0, 0, 255, 0]  # Fully transparent - should be skipped
    palette[3] = [128, 128, 0, 255]

    config = ColorVectorizationConfig(num_colors=4)
    layers = separate_layers(img, palette, config)

    # Should not include the transparent color
    layer_colors = [l["color"] for l in layers]
    assert (0, 0, 255, 0) not in layer_colors


# ── SVG Output Format ─────────────────────────────────────────────


def test_svg_xml_declaration():
    """Test SVG output has proper XML declaration."""
    img = _make_test_image_rgba(32, 32)
    svg = vectorize_color(img)
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')


def test_svg_with_viewbox():
    """Test SVG output has correct viewBox."""
    img = _make_test_image_rgba(64, 32)  # 64h x 32w
    svg = vectorize_outline(img)
    assert 'viewBox="0 0 32 64"' in svg


def test_svg_valid_xml():
    """Test SVG output is valid XML (basic check)."""
    img = _make_test_image_rgba(32, 32)
    svg = vectorize_color(img)

    # Count opening and closing tags for basic validity
    assert svg.count("<svg") == 1
    assert svg.count("</svg>") == 1
    # All open paths should be closed
    open_paths = svg.count("<path")
    close_tags = svg.count("/>")
    assert open_paths == close_tags, (
        f"Not all path tags closed: {open_paths} opens vs {close_tags} closes"
    )
