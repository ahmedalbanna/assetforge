"""
Image vectorization backend — converts raster images to SVG vector graphics.

Supports:
- Outline mode: Single-color silhouette vectorization
- Color mode: Multi-layer vectorization with color quantization
- Character-Aware mode: Feature-preserving vectorization with face/eye/mouth detection
- Anime Line Art: Separate stroke layer for thin line art (anime style)
- Procedural Enhancements: Eye highlights, detail preservation

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

        # Character-aware vectorization options
        enable_character_features: If True, use face/eye/mouth detection to enhance
                                   detail in facial regions during vectorization.
        feature_simplify_tolerance: Lower simplification tolerance for facial feature
                                    regions (e.g., 0.3-1.0) for more detail.
        feature_min_area: Smaller minimum contour area for feature regions.
        preserve_line_art: If True, extract and preserve thin line art (anime style)
                           as a separate SVG stroke layer.
        line_art_threshold: Canny edge detection threshold for line art extraction.
        line_art_color: Color string for the line art stroke layer (e.g., 'black').
        enable_eye_highlights: If True, add procedural circular highlights to eyes
                               for more expressive characters.
        eye_highlight_color: Color for procedural eye highlights (default white).
        semantic_detail_level: Controls how much extra detail to preserve in feature
                               regions: 'low', 'medium', or 'high'.
                               High = lowest simplify_tolerance for features.
    """

    num_colors: int = 24
    simplify_tolerance: float = 0.8
    min_area: int = 4
    color_tolerance: int = 10
    layer_order: str = "luminance"
    background_color: str = "none"
    max_svg_dimension: int = 0  # 0 = no downscale
    gap_fill: bool = True  # fill remaining transparent holes between layers

    # Character-aware vectorization
    enable_character_features: bool = False
    feature_simplify_tolerance: float = 0.5
    feature_min_area: int = 2
    preserve_line_art: bool = True
    line_art_threshold: int = 30
    line_art_color: str = "black"
    enable_eye_highlights: bool = True
    eye_highlight_color: str = "white"
    semantic_detail_level: str = "high"


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
    # Character feature validation
    if config.enable_character_features:
        if not 0.1 <= config.feature_simplify_tolerance <= 5.0:
            raise ValueError(
                f"feature_simplify_tolerance must be between 0.1 and 5.0, "
                f"got {config.feature_simplify_tolerance}"
            )
        if config.feature_min_area < 1:
            raise ValueError(
                f"feature_min_area must be >= 1, got {config.feature_min_area}"
            )
        if config.semantic_detail_level not in ("low", "medium", "high"):
            raise ValueError(
                f"semantic_detail_level must be 'low', 'medium', or 'high', "
                f"got {config.semantic_detail_level}"
            )
        if not 10 <= config.line_art_threshold <= 200:
            raise ValueError(
                f"line_art_threshold must be between 10 and 200, "
                f"got {config.line_art_threshold}"
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


def _extract_line_art(
    image: np.ndarray,
    threshold1: int = 30,
    threshold2: int = 90,
    min_line_length: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract thin line art from an image using edge detection.

    Uses Canny edge detection with morphological thinning to produce
    clean line art suitable for an SVG stroke layer (anime/manga style).

    Args:
        image: RGB uint8 array, shape (H, W, 3) or RGBA.
        threshold1: Lower Canny threshold (sensitivity to weak edges).
        threshold2: Upper Canny threshold (sensitivity to strong edges).
        min_line_length: Minimum contour length in pixels.

    Returns:
        Tuple of (contours, edge_mask) where:
        - contours: List of OpenCV contour arrays for edges.
        - edge_mask: Binary uint8 edge mask.
    """
    if image.shape[2] >= 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to preserve edges while smoothing color
    gray = cv2.bilateralFilter(gray, 7, 30, 30)

    # Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=3)

    # Morphological closure to connect nearby edge segments
    # Use minimal kernel to avoid erasing thin lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find edge contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by length
    line_contours = []
    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        if length >= min_line_length:
            # Simplify slightly for smoother lines
            epsilon = 0.5
            approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
            if len(approx) >= 2:
                line_contours.append(approx)

    return line_contours, edges


def _contours_to_svg_strokes(
    contours: list[np.ndarray],
    stroke_color: str = "black",
    stroke_width: float = 1.0,
) -> str:
    """Convert edge contours to SVG stroke paths.

    Unlike filled paths, stroke paths don't need to be closed polygons.
    Each contour becomes an open path (no Z command).

    Args:
        contours: List of contours, each shape (N, 1, 2).
        stroke_color: SVG stroke color string.
        stroke_width: Stroke width in SVG units.

    Returns:
        SVG path data string(s) suitable for a <path> with stroke.
    """
    paths = []
    for contour in contours:
        if len(contour) < 2:
            continue
        pts = contour[:, 0, :]
        path_data = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            path_data += f" L {pt[0]} {pt[1]}"
        paths.append(path_data)

    if not paths:
        return ""

    joined = " ".join(paths)
    return (
        f'<path d="{joined}" fill="none" '
        f'stroke="{stroke_color}" stroke-width="{stroke_width}" />'
    )


def _add_eye_highlight_paths(
    eye_regions: list[tuple[int, int, int, int]],
    highlight_color: str = "white",
    highlight_radius_ratio: float = 0.25,
) -> list[str]:
    """Generate procedural SVG paths for eye highlights.

    Adds small circular highlights near the top-center of each detected
    eye region, mimicking the specular reflection that gives eyes life.

    Args:
        eye_regions: List of (x, y, w, h) bounding boxes for eyes.
        highlight_color: SVG color for the highlight circles.
        highlight_radius_ratio: Radius of highlight relative to eye width.

    Returns:
        List of SVG path element strings.
    """
    elements = []
    for ex, ey, ew, eh in eye_regions:
        if ew < 4 or eh < 4:
            continue

        # Highlight position: upper-left quadrant of eye
        cx = ex + int(ew * 0.35)
        cy = ey + int(eh * 0.30)
        r = max(1, int(min(ew, eh) * highlight_radius_ratio))

        # Main highlight
        elements.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="{highlight_color}" '
            f'fill-opacity="0.8" />'
        )

        # Secondary smaller highlight (lower-right) for anime-style sparkle
        if ew >= 8 and eh >= 6:
            cx2 = ex + int(ew * 0.60)
            cy2 = ey + int(eh * 0.55)
            r2 = max(1, r // 2)
            elements.append(
                f'<circle cx="{cx2}" cy="{cy2}" r="{r2}" '
                f'fill="{highlight_color}" '
                f'fill-opacity="0.5" />'
            )

    return elements


def _assign_nearest_colors(
    image: np.ndarray,
    palette: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Assign every foreground pixel to its nearest palette color (Lab distance).

    Uses CIE76 delta-E in Lab color space for perceptually uniform matching.
    This guarantees 100% assignment — no pixel is left unmatched, which
    eliminates the gaps caused by fixed RGB tolerance thresholds.

    Args:
        image: RGBA uint8 array, shape (H, W, 4).
        palette: (N, 4) uint8 array of palette colors.
        fg_mask: Boolean mask of foreground pixels (H, W).

    Returns:
        Integer label map of shape (H, W) where each foreground pixel
        contains the index of its nearest palette color. Background pixels
        are labeled -1.
    """
    h, w = image.shape[:2]
    label_map = np.full((h, w), -1, dtype=np.int32)

    if not fg_mask.any():
        return label_map

    # Convert image RGB to Lab
    rgb = image[:, :, :3].astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)  # (H, W, 3) float32

    # Convert palette RGB to Lab
    pal_rgb = palette[:, :3].astype(np.float32) / 255.0
    pal_lab = cv2.cvtColor(pal_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(
        -1, 3
    )  # (N, 3) float32

    # Number of valid palette entries (alpha > 5)
    valid_indices = np.where(palette[:, 3] >= 5)[0]
    if len(valid_indices) == 0:
        return label_map

    valid_pal_lab = pal_lab[valid_indices]  # (M, 3)

    # For each foreground pixel, compute delta-E to all valid palette colors
    # and assign to the nearest one.
    # Flatten foreground pixels for vectorized distance computation.
    fg_ys, fg_xs = np.where(fg_mask)
    fg_pixels_lab = lab_image[fg_ys, fg_xs]  # (P, 3)

    # Compute squared Lab distance: (P, 1, 3) - (1, M, 3) → (P, M, 3) → sum
    diff = fg_pixels_lab[:, np.newaxis, :] - valid_pal_lab[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)  # (P, M)

    nearest = np.argmin(dist_sq, axis=1)  # (P,)
    label_map[fg_ys, fg_xs] = valid_indices[nearest]

    return label_map


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

    # Build a nearest-color assignment map using Lab distance.
    # This guarantees every foreground pixel is assigned to exactly one
    # palette color, eliminating gaps caused by RGB tolerance matching.
    fg_mask = image[:, :, 3] > 0
    label_map = _assign_nearest_colors(image, palette, fg_mask)

    for idx in range(len(palette)):
        color = palette[idx]
        r, g, b, a = int(color[0]), int(color[1]), int(color[2]), int(color[3])

        # Skip fully transparent colors
        if a < 5:
            continue

        # Create binary mask from the pre-computed nearest-color assignment
        mask = (label_map == idx) & fg_mask

        # Fill small transparent holes inside the color region.
        # NOTE: MORPH_OPEN is intentionally omitted — it erodes 1px from
        # all edges and destroys thin features (belt straps, weapon edges,
        # clothing folds, facial details), causing visible gaps in the SVG.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
        )

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

    # Character-aware mode: delegate to feature-enhanced vectorization
    if config.enable_character_features:
        logger.info("Character-aware vectorization enabled — detecting features...")
        return vectorize_color_character_aware(image, config)

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

    # Gap-fill safety net: check for any foreground pixels not covered
    # by any layer's contours and add a fill layer for them.
    if config.gap_fill and layers:
        svg_h, svg_w = img.shape[:2]
        layer_coverage = np.zeros((svg_h, svg_w), dtype=np.uint8)
        for layer in layers:
            cv2.drawContours(layer_coverage, layer["contours"], -1, 255, -1)
        original_fg = img[:, :, 3] > 0
        gap_pixels = original_fg & (layer_coverage == 0)
        gap_count = int(gap_pixels.sum())
        if gap_count > 0:
            # Assign each gap pixel to its nearest palette color and
            # generate a fill layer for the uncovered region.
            gap_labels = _assign_nearest_colors(img, palette, original_fg)
            gap_label_mask = gap_labels >= 0
            gap_with_label = gap_label_mask & (layer_coverage == 0)

            # Group gap pixels by their nearest palette color
            for idx in range(len(palette)):
                color = palette[idx]
                r, g, b, a = int(color[0]), int(color[1]), int(color[2]), int(color[3])
                if a < 5:
                    continue
                mask = (gap_labels == idx) & gap_with_label
                if not mask.any():
                    continue

                # Dilate the gap mask slightly to connect isolated pixels
                # into contiguous regions that form valid contours
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

                contours, _ = cv2.findContours(
                    dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                fill_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < config.min_area:
                        continue
                    epsilon = config.simplify_tolerance
                    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                    if len(approx) >= 3:
                        fill_contours.append(approx)

                if fill_contours:
                    # Check if this palette color already has a layer —
                    # append the gap contours to it
                    existing = next(
                        (l for l in layers if l["color"][:3] == (r, g, b)), None
                    )
                    if existing:
                        existing["contours"].extend(fill_contours)
                        existing["area"] += sum(
                            cv2.contourArea(c) for c in fill_contours
                        )
                    else:
                        lum = _rgb_to_luminance(np.array([r, g, b]))
                        layers.append(
                            {
                                "color": (r, g, b, a),
                                "contours": fill_contours,
                                "area": sum(cv2.contourArea(c) for c in fill_contours),
                                "luminance": lum,
                                "index": idx,
                            }
                        )
            logger.info("Gap-fill: covered %d orphaned foreground pixels", gap_count)

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


def _vectorize_feature_region(
    image_rgba: np.ndarray,
    region_mask: np.ndarray,
    config: ColorVectorizationConfig,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[dict]:
    """Vectorize a specific region with enhanced detail settings.

    Performs color quantization and layer separation on a sub-region
    of the image, using tighter simplification and smaller min area
    for more detail preservation.

    Args:
        image_rgba: Full RGBA uint8 image.
        region_mask: Binary uint8 mask (0/255) indicating the region.
        config: Base vectorization config (feature detail settings used).
        x_offset: X offset to add to contour coordinates.
        y_offset: Y offset to add to contour coordinates.

    Returns:
        List of layer dicts localized to the region.
    """
    # Extract region bounding box
    ys, xs = np.where(region_mask > 0)
    if len(ys) == 0:
        return []
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()

    # Pad slightly
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image_rgba.shape[1] - 1, x2 + pad)
    y2 = min(image_rgba.shape[0] - 1, y2 + pad)
    sub_w, sub_h = x2 - x1 + 1, y2 - y1 + 1

    if sub_w < 8 or sub_h < 8:
        return []

    # Extract sub-image
    sub_img = image_rgba[y1 : y2 + 1, x1 : x2 + 1].copy()

    # Create local mask for this region
    local_mask = region_mask[y1 : y2 + 1, x1 : x2 + 1].copy()

    # Apply mask: zero out pixels outside the region (they become transparent)
    sub_img[local_mask == 0] = [0, 0, 0, 0]

    # Create config with feature-level detail
    feature_config = ColorVectorizationConfig(
        num_colors=max(8, min(config.num_colors, 32)),
        simplify_tolerance=config.feature_simplify_tolerance,
        min_area=config.feature_min_area,
        color_tolerance=config.color_tolerance,
        layer_order=config.layer_order,
        background_color=config.background_color,
    )

    # Quantize colors on the sub-region
    quantized, palette = quantize_colors(sub_img, feature_config.num_colors)

    # Separate layers
    raw_layers = separate_layers(quantized, palette, feature_config)

    # Offset contour coordinates back to full image space
    offset_layers = []
    for layer in raw_layers:
        adjusted_contours = []
        for contour in layer["contours"]:
            pts = contour[:, 0, :].copy()
            pts[:, 0] += x1 + x_offset
            pts[:, 1] += y1 + y_offset
            adjusted = pts.reshape(-1, 1, 2).astype(np.int32)
            adjusted_contours.append(adjusted)

        offset_layers.append(
            {
                "color": layer["color"],
                "contours": adjusted_contours,
                "area": layer["area"],
                "luminance": layer["luminance"],
                "index": layer["index"],
                "feature_layer": True,  # Mark as feature overlay
            }
        )

    return offset_layers


def vectorize_color_character_aware(
    image: np.ndarray,
    config: ColorVectorizationConfig,
) -> str:
    """Vectorize an image with character feature-aware enhancement.

    Performs multi-pass vectorization:
    1. Standard pass: full image with base config (for background, clothes, hair)
    2. Feature pass: face/eyes/mouth regions with enhanced detail
    3. Line art pass: extract thin edges as separate stroke layer
    4. Procedural pass: add eye highlights

    Args:
        image: RGBA uint8 array, shape (H, W, 4).
        config: Vectorization configuration with character features enabled.

    Returns:
        SVG string with enhanced character details.
    """
    from assetforge.backends.character_features import (
        DetectionMethod,
        detect_character_features,
        enhance_feature_masks,
    )

    h, w = image.shape[:2]

    # Ensure RGBA
    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    # Step 1: Detect character features
    rgb_img = img[:, :, :3]
    features = detect_character_features(
        rgb_img,
        method=DetectionMethod.AUTO,
        min_face_size=30,
        min_confidence=0.4,
    )
    features = enhance_feature_masks(rgb_img, features, dilation_kernel=3)

    # SVG building
    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
    ]

    # Background rect if needed
    if config.background_color.lower() != "none":
        svg_parts.append(
            f'<rect width="{w}" height="{h}" fill="{config.background_color}" />'
        )

    # Step 2: Base vectorization (full image with standard settings)
    # Quantize colors
    quantized, palette = quantize_colors(img, config.num_colors)
    base_layers = separate_layers(quantized, palette, config)

    # If no face detected, just use standard vectorization
    if not features.face_detected:
        logger.info("No face detected — falling back to standard vectorization")
        # Still try line art if enabled
        all_layers = base_layers

        # Combine into SVG
        for layer in all_layers:
            r, g, b, a = layer["color"]
            fill = f"rgb({r},{g},{b})"
            opacity = a / 255.0
            path_data = _contours_to_svg_paths(layer["contours"])
            if not path_data:
                continue
            if opacity < 1.0:
                svg_parts.append(
                    f'<path d="{path_data}" fill="{fill}" '
                    f'fill-opacity="{opacity:.3f}" />'
                )
            else:
                svg_parts.append(f'<path d="{path_data}" fill="{fill}" />')

        # Line art layer
        if config.preserve_line_art:
            line_contours, _ = _extract_line_art(
                rgb_img,
                threshold1=config.line_art_threshold,
                threshold2=config.line_art_threshold * 3,
            )
            line_art = _contours_to_svg_strokes(
                line_contours, stroke_color=config.line_art_color
            )
            if line_art:
                svg_parts.append(line_art)

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    logger.info(
        "Character-aware vectorization: face detected (%d eyes, mouth=%s)",
        len(features.eye_regions),
        features.mouth_region is not None,
    )

    # Step 3: Extract feature-specific layers with higher detail
    feature_layers_map: dict[str, list[dict]] = {}

    for feature_name in ("face", "left_eye", "right_eye", "mouth"):
        mask = features.feature_masks.get(feature_name)
        if mask is not None and mask.sum() > 0:
            feature_layers = _vectorize_feature_region(
                img,
                mask,
                config,
            )
            if feature_layers:
                feature_layers_map[feature_name] = feature_layers

    # Step 4: Merge layers — base first, then feature overlays on top
    # Remove any base layers that overlap with feature regions entirely
    # (feature layers will be rendered on top, so base layers underneath are fine)

    # Write base layers
    for layer in base_layers:
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

    # Write face feature overlay layers
    # Order: face base → mouth → eyes (so eyes are on top)
    face_features_order = ["face", "mouth", "left_eye", "right_eye"]
    for feat_name in face_features_order:
        f_layers = feature_layers_map.get(feat_name, [])
        # Sort by luminance so darker details render on top
        f_layers.sort(key=lambda x: x["luminance"])
        for layer in f_layers:
            r, g, b, a = layer["color"]
            fill = f"rgb({r},{g},{b})"
            opacity = a / 255.0
            path_data = _contours_to_svg_paths(layer["contours"])
            if not path_data:
                continue
            if opacity < 1.0:
                svg_parts.append(
                    f'<path d="{path_data}" fill="{fill}" '
                    f'fill-opacity="{opacity:.3f}" />'
                )
            else:
                svg_parts.append(f'<path d="{path_data}" fill="{fill}" />')

    # Step 5: Line art layer (anime style strokes)
    if config.preserve_line_art:
        line_contours, _ = _extract_line_art(
            rgb_img,
            threshold1=config.line_art_threshold,
            threshold2=config.line_art_threshold * 3,
        )
        line_art = _contours_to_svg_strokes(
            line_contours, stroke_color=config.line_art_color
        )
        if line_art:
            svg_parts.append(line_art)

    # Step 6: Procedural eye highlights
    if config.enable_eye_highlights and features.eye_regions:
        highlights = _add_eye_highlight_paths(
            features.eye_regions,
            highlight_color=config.eye_highlight_color,
        )
        svg_parts.extend(highlights)

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


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
