"""
Image vectorization backend — converts raster images to SVG vector graphics.

Supports:
- Outline mode: Single-color silhouette vectorization
- Color mode: Multi-layer vectorization with color quantization

Uses OpenCV for contour extraction and simplification, with pure-Python SVG generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("assetforge")


@dataclass
class ColorVectorizationConfig:
    """Configuration for color vectorization.

    Attributes:
        num_colors: Number of colors for k-means quantization (4-64).
        simplify_tolerance: Douglas-Peucker simplification epsilon in pixels (0.5-5.0).
        min_area: Minimum contour area in pixels (smaller contours are discarded).
        color_tolerance: Maximum Lab delta-E distance for pixel-to-color matching (1-30).
        layer_order: How to order layers in SVG: 'luminance' (dark→light),
                     'area' (largest→smallest), or 'index' (as quantized).
        background_color: SVG background fill (e.g., 'none', 'white', '#FFFFFF').
        max_svg_dimension: If > 0, downscale image before vectorization if max(H,W)
                           exceeds this value (improves performance).
    """

    num_colors: int = 24
    simplify_tolerance: float = 1.5
    min_area: int = 10
    color_tolerance: int = 10
    layer_order: str = "luminance"
    background_color: str = "none"
    max_svg_dimension: int = 0  # 0 = no downscale


def _validate_config(config: ColorVectorizationConfig) -> None:
    """Validate vectorization configuration parameters."""
    if not 4 <= config.num_colors <= 64:
        raise ValueError(
            f"num_colors must be between 4 and 64, got {config.num_colors}"
        )
    if not 0.5 <= config.simplify_tolerance <= 5.0:
        raise ValueError(
            f"simplify_tolerance must be between 0.5 and 5.0, "
            f"got {config.simplify_tolerance}"
        )
    if config.min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {config.min_area}")
    if not 1 <= config.color_tolerance <= 30:
        raise ValueError(
            f"color_tolerance must be between 1 and 30, got {config.color_tolerance}"
        )
    if config.layer_order not in ("luminance", "area", "index"):
        raise ValueError(
            f"layer_order must be 'luminance', 'area', or 'index', "
            f"got {config.layer_order}"
        )


def quantize_colors(
    image: np.ndarray,
    num_colors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce image colors using k-means clustering in Lab space.

    Args:
        image: RGBA uint8 array, shape (H, W, 4).
        num_colors: Number of target colors.

    Returns:
        Tuple of (quantized_image, palette) where:
        - quantized_image: RGBA uint8 array with reduced colors.
        - palette: (num_colors, 4) uint8 array of palette colors (RGBA).
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 4).astype(np.float32)

    # Only cluster non-transparent pixels (alpha > 0) to avoid background noise
    alpha_mask = pixels[:, 3] > 0
    if alpha_mask.sum() == 0:
        # Fully transparent image, return as-is
        palette = np.zeros((num_colors, 4), dtype=np.uint8)
        quantized = image.copy()
        return quantized, palette

    opaque_pixels = pixels[alpha_mask]

    # Convert RGB to Lab for perceptually uniform clustering
    rgb = opaque_pixels[:, :3] / 255.0
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)

    # Combine Lab + alpha for clustering
    features = np.column_stack(
        [
            lab,
            opaque_pixels[:, 3:4] / 255.0,  # alpha scaled 0-1
        ]
    )

    # Scale features: Lab values are ~0-100, alpha is 0-1
    # We weight Lab more heavily since color is more important
    k = min(num_colors, len(features))
    if k < 2:
        # Single color or tiny image
        palette_rgba = np.zeros((num_colors, 4), dtype=np.uint8)
        palette_rgba[0] = opaque_pixels[0].astype(np.uint8)
        quantized = np.zeros((h * w, 4), dtype=np.uint8)
        quantized[alpha_mask] = palette_rgba[0]
        quantized = quantized.reshape(h, w, 4)
        return quantized, palette_rgba

    # Use k-means with multiple attempts for better quality
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        features,
        k,
        None,
        criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # Map quantized colors back to original pixels (RGBA)
    palette_rgba = np.zeros((num_colors, 4), dtype=np.uint8)
    for i in range(k):
        # Find the original pixel closest to this cluster center
        cluster_mask = labels.flatten() == i
        if cluster_mask.any():
            cluster_pixels = opaque_pixels[cluster_mask]
            # Use mean of cluster pixels for more accurate color
            palette_rgba[i] = cluster_pixels.mean(axis=0).astype(np.uint8)

    # Build quantized image
    quantized = np.zeros((h * w, 4), dtype=np.uint8)
    if alpha_mask.any():
        opaque_labels = labels.flatten()
        for i in range(k):
            cluster_mask = opaque_labels == i
            if cluster_mask.any():
                quantized_pixels = quantized[alpha_mask]
                quantized_pixels[cluster_mask] = palette_rgba[i]
                quantized[alpha_mask] = quantized_pixels

    quantized = quantized.reshape(h, w, 4)

    # If we have fewer unique colors than requested, pad the palette
    if k < num_colors:
        palette_rgba[k:] = palette_rgba[k - 1]

    return quantized, palette_rgba


def _rgb_to_luminance(rgb: np.ndarray) -> float:
    """Calculate perceived luminance from an RGB triplet."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _lab_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calculate CIE76 delta-E color difference."""
    return float(np.sqrt(((lab1 - lab2) ** 2).sum()))


def _color_distance_sq(c1: np.ndarray, c2: np.ndarray) -> float:
    """Squared RGB distance with alpha weighting."""
    dr = float(c1[0]) - float(c2[0])
    dg = float(c1[1]) - float(c2[1])
    db = float(c1[2]) - float(c2[2])
    da = float(c1[3]) - float(c2[3])
    # Weight alpha less
    return dr * dr + dg * dg + db * db + 0.5 * da * da


def separate_layers(
    image: np.ndarray,
    palette: np.ndarray,
    config: ColorVectorizationConfig,
) -> list[dict]:
    """Separate image into color layers.

    For each palette color, creates a binary mask of matching pixels,
    then extracts and simplifies contours.

    Args:
        image: RGBA uint8 array, shape (H, W, 4).
        palette: (num_colors, 4) uint8 array of palette colors.
        config: Vectorization configuration.

    Returns:
        List of layer dicts, each containing:
        - 'color': RGBA tuple
        - 'contours': list of simplified contour arrays
        - 'area': total filled area in pixels
        - 'luminance': perceived brightness
    """
    h, w = image.shape[:2]
    layers = []

    for idx in range(len(palette)):
        color = palette[idx]
        r, g, b, a = int(color[0]), int(color[1]), int(color[2]), int(color[3])

        # Skip fully transparent colors
        if a < 5:
            continue

        # Create binary mask for this color with tolerance
        rgb_target = np.array([r, g, b], dtype=np.uint8)
        rgb_image = image[:, :, :3]

        # Use RGB distance for mask creation
        diff = np.abs(rgb_image.astype(np.int16) - rgb_target.astype(np.int16))
        mask = np.all(diff <= config.color_tolerance, axis=2)

        # Also require the pixel to be non-transparent in original
        mask = mask & (image[:, :, 3] > 0)

        # Remove tiny isolated blobs and fill small holes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter by area and simplify
        simplified_contours = []
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.min_area:
                continue
            total_area += area

            # Douglas-Peucker simplification
            epsilon = config.simplify_tolerance
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)

            # Simplification returned in shape (N, 1, 2)
            if len(approx) >= 3:
                simplified_contours.append(approx)

        if simplified_contours:
            lum = _rgb_to_luminance(np.array([r, g, b]))
            layers.append(
                {
                    "color": (r, g, b, a),
                    "contours": simplified_contours,
                    "area": total_area,
                    "luminance": lum,
                    "index": idx,
                }
            )

    # Sort layers
    if config.layer_order == "luminance":
        layers.sort(key=lambda x: x["luminance"])
    elif config.layer_order == "area":
        layers.sort(key=lambda x: x["area"], reverse=True)
    # 'index' = keep original order

    return layers


def _contours_to_svg_paths(contours: list[np.ndarray]) -> str:
    """Convert a list of OpenCV contours to SVG path data string.

    Each contour becomes a sub-path: M x y L x y ... Z

    Args:
        contours: List of contours, each shape (N, 1, 2).

    Returns:
        SVG path data string (the d attribute value).
    """
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        pts = contour[:, 0, :]  # Shape (N, 2)
        path_data = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            path_data += f" L {pt[0]} {pt[1]}"
        path_data += " Z"
        paths.append(path_data)
    return " ".join(paths)


def vectorize_color(
    image: np.ndarray,
    config: Optional[ColorVectorizationConfig] = None,
) -> str:
    """Convert a raster RGBA image to a multi-color SVG.

    Steps:
    1. Color quantization (k-means in Lab space)
    2. Layer separation (mask per color)
    3. Contour extraction (OpenCV)
    4. Path simplification (Douglas-Peucker)
    5. SVG generation with colored paths

    Args:
        image: RGBA uint8 array, shape (H, W, 4). Pixels with alpha=0
               are treated as transparent/background.
        config: Vectorization configuration (defaults if None).

    Returns:
        SVG string with multi-color vector paths.
    """
    if config is None:
        config = ColorVectorizationConfig()
    _validate_config(config)

    h, w = image.shape[:2]

    # Validate input
    if h == 0 or w == 0:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="0" height="0" viewBox="0 0 0 0"></svg>'
        )

    logger.info(
        "Vectorizing image: %dx%d, %d colors, simplify=%.1f, min_area=%d",
        w,
        h,
        config.num_colors,
        config.simplify_tolerance,
        config.min_area,
    )

    # Determine if we need to downscale for performance
    img = image
    scale_factor = 1.0
    if config.max_svg_dimension > 0:
        max_dim = max(h, w)
        if max_dim > config.max_svg_dimension:
            scale_factor = config.max_svg_dimension / max_dim
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info("Downscaled to %dx%d (factor=%.2f)", new_w, new_h, scale_factor)

    # Step 1: Color quantization
    quantized, palette = quantize_colors(img, config.num_colors)

    # Step 2 & 3: Separate layers and extract contours
    layers = separate_layers(quantized, palette, config)

    if not layers:
        logger.warning("No layers found in image — returning empty SVG")
        svg_h, svg_w = img.shape[:2]
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_w}" height="{svg_h}" '
            f'viewBox="0 0 {svg_w} {svg_h}" '
            f'fill="{config.background_color}">\n</svg>'
        )

    # Step 4 & 5: Generate SVG
    svg_h, svg_w = img.shape[:2]
    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">',
    ]

    # If background is not 'none', add a background rect
    if config.background_color.lower() != "none":
        svg_parts.append(
            f'<rect width="{svg_w}" height="{svg_h}" '
            f'fill="{config.background_color}" />'
        )

    for layer in layers:
        r, g, b, a = layer["color"]
        fill = f"rgb({r},{g},{b})"
        opacity = a / 255.0

        path_data = _contours_to_svg_paths(layer["contours"])
        if not path_data:
            continue

        if opacity < 1.0:
            svg_parts.append(
                f'<path d="{path_data}" fill="{fill}" fill-opacity="{opacity:.3f}" />'
            )
        else:
            svg_parts.append(f'<path d="{path_data}" fill="{fill}" />')

    svg_parts.append("</svg>")

    svg_content = "\n".join(svg_parts)
    logger.info(
        "Generated SVG: %d layers, %d bytes",
        len(layers),
        len(svg_content),
    )

    return svg_content


def vectorize_outline(
    image: np.ndarray,
    simplify_tolerance: float = 1.5,
    min_area: int = 10,
    fill_color: str = "black",
) -> str:
    """Convert a binary/alpha mask to a single-color outline SVG.

    This is a pure-Python alternative to the Rust outline vectorization.

    Args:
        image: RGBA uint8 array, shape (H, W, 4). Alpha channel is used as mask.
        simplify_tolerance: Douglas-Peucker simplification epsilon.
        min_area: Minimum contour area.
        fill_color: SVG fill color string (e.g., 'black', '#000000').

    Returns:
        SVG string with black vector paths.
    """
    h, w = image.shape[:2]

    # Use alpha channel as mask, or threshold grayscale
    if image.shape[2] >= 4:
        alpha = image[:, :, 3]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, alpha = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]

    path_parts = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue

        pts = approx[:, 0, :]
        path_data = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            path_data += f" L {pt[0]} {pt[1]}"
        path_data += " Z"
        path_parts.append(path_data)

    if path_parts:
        svg_parts.append(f'<path d="{" ".join(path_parts)}" fill="{fill_color}" />')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
