"""
Smart Defaults — AI-driven auto-configuration for the 2D pipeline.

Analyzes input images and automatically selects optimal parameters,
eliminating the need for manual tuning. Single ``quality`` preset replaces
10+ individual settings.

Presets:
    - **fast**: Quick preview, low-poly outline
    - **balanced**: General sprites with color vectorization (*default*)
    - **high**: Detailed characters with alpha matting and fine detail
    - **ultra**: Maximum quality, SAM2 segmentation, tiny detail preservation
    - **auto**: Analyzes the image and selects best preset automatically

Usage:
    >>> from assetforge.backends.smart_defaults import resolve_2d_config
    >>>
    >>> # Get all pipeline params from a single quality string
    >>> params = resolve_2d_config("balanced")
    >>> params["num_colors"]
    16
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("assetforge")

# ── Quality Presets ─────────────────────────────────────────────────────────
# Each preset defines a complete set of 2D pipeline parameters.
# Individual CLI/UI flags override these when explicitly provided.

QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "label": "Fast — quick preview, low detail",
        "background_remover": "rembg",
        "bg_model": "u2netp",
        "alpha_matting": False,
        "alpha_matting_foreground_threshold": 240,
        "alpha_matting_background_threshold": 10,
        "alpha_matting_erode_size": 10,
        "vectorize_mode": "outline",
        "num_colors": 8,
        "simplify_tolerance": 2.0,
        "min_area": 20,
        "enable_character_features": False,
        "feature_simplify_tolerance": 0.5,
        "feature_min_area": 2,
        "preserve_line_art": False,
        "line_art_threshold": 30,
        "enable_eye_highlights": False,
        "semantic_detail_level": "low",
        "sprite_sizes": [64, 128],
        "feather_pixels": 0,
        "pack_sheet": False,
        "gap_fill": True,
    },
    "balanced": {
        "label": "Balanced — general-purpose sprites (default)",
        "background_remover": "rembg",
        "bg_model": "isnet-general-use",
        "alpha_matting": False,
        "alpha_matting_foreground_threshold": 240,
        "alpha_matting_background_threshold": 10,
        "alpha_matting_erode_size": 10,
        "vectorize_mode": "color",
        "num_colors": 16,
        "simplify_tolerance": 0.8,
        "min_area": 4,
        "enable_character_features": False,  # auto-enabled if face detected
        "feature_simplify_tolerance": 0.5,
        "feature_min_area": 2,
        "preserve_line_art": True,
        "line_art_threshold": 30,
        "enable_eye_highlights": True,
        "semantic_detail_level": "medium",
        "sprite_sizes": [64, 128, 256],
        "feather_pixels": 1,
        "pack_sheet": True,
        "gap_fill": True,
    },
    "high": {
        "label": "High — detailed characters and fine art",
        "background_remover": "rembg",
        "bg_model": "isnet-general-use",
        "alpha_matting": True,
        "alpha_matting_foreground_threshold": 240,
        "alpha_matting_background_threshold": 10,
        "alpha_matting_erode_size": 10,
        "vectorize_mode": "color",
        "num_colors": 32,
        "simplify_tolerance": 0.5,
        "min_area": 2,
        "enable_character_features": True,
        "feature_simplify_tolerance": 0.3,
        "feature_min_area": 1,
        "preserve_line_art": True,
        "line_art_threshold": 25,
        "enable_eye_highlights": True,
        "semantic_detail_level": "high",
        "sprite_sizes": [128, 256, 512],
        "feather_pixels": 2,
        "pack_sheet": True,
        "gap_fill": True,
    },
    "ultra": {
        "label": "Ultra — maximum quality, SAM2, finest detail",
        "background_remover": "sam2",
        "bg_model": None,
        "alpha_matting": True,
        "alpha_matting_foreground_threshold": 245,
        "alpha_matting_background_threshold": 5,
        "alpha_matting_erode_size": 5,
        "vectorize_mode": "color",
        "num_colors": 48,
        "simplify_tolerance": 0.3,
        "min_area": 1,
        "enable_character_features": True,
        "feature_simplify_tolerance": 0.2,
        "feature_min_area": 1,
        "preserve_line_art": True,
        "line_art_threshold": 20,
        "enable_eye_highlights": True,
        "semantic_detail_level": "high",
        "sprite_sizes": [256, 512, 1024],
        "feather_pixels": 3,
        "pack_sheet": True,
        "gap_fill": True,
    },
}

VALID_QUALITY_LEVELS = set(QUALITY_PRESETS.keys()) | {"auto"}


# ── Image Analysis ──────────────────────────────────────────────────────────


def analyze_image(
    image_path: Optional[str] = None,
    image_array: Optional[np.ndarray] = None,
) -> dict:
    """Analyze an input image and return semantic features.

    Uses lightweight heuristics (no model downloads) to classify the image:
    - **has_face**: whether a face is detected (uses OpenCV Haar cascade)
    - **is_anime**: whether the image appears to be anime/manga style
    - **complexity**: edge-based complexity score (0.0–1.0)
    - **megapixels**: input resolution in megapixels

    Args:
        image_path: Path to the input image (optional if image_array given).
        image_array: RGBA/BGR uint8 array (optional if image_path given).

    Returns:
        Dict of image features.
    """
    result: dict = {
        "has_face": False,
        "is_anime": False,
        "complexity": 0.5,
        "megapixels": 0.0,
        "width": 0,
        "height": 0,
        "aspect_ratio": 1.0,
    }

    import cv2

    if image_path is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        # Ensure 3-channel BGR for analysis
        if img is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif image_array is not None:
        img = image_array.copy()
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        return result

    if img is None or img.size == 0:
        return result

    h, w = img.shape[:2]
    result["width"] = w
    result["height"] = h
    result["megapixels"] = round(h * w / 1_000_000, 2)
    result["aspect_ratio"] = round(w / h, 2) if h > 0 else 1.0

    # ── Anime detection via color palette analysis ──
    # Anime typically has: limited palette, flat shading (low gradient variance),
    # distinct saturation/luminance profile
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flatten to 1D color array for palette analysis
    pixel_colors = img.reshape(-1, 3)

    # Sample 10k random pixels for analysis
    if len(pixel_colors) > 10_000:
        indices = np.random.choice(len(pixel_colors), 10_000, replace=False)
        sampled = pixel_colors[indices]
    else:
        sampled = pixel_colors

    # Count unique colors (discrete quantization bins)
    quantized = (sampled // 16).astype(np.uint8)  # 16^3 = 4096 bins
    unique_bins = len(np.unique(quantized, axis=0))
    unique_ratio = unique_bins / max(len(sampled), 1)

    # Gradient variance: anime has fewer gradients (flat shading)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_variance = float(np.var(grad_mag))

    # Heuristic: low unique color ratio + low gradient variance → anime
    # Real photos have many subtle color variations
    result["is_anime"] = unique_ratio < 0.15 and grad_variance < 2000

    # ── Complexity score ──
    # Based on edge density and color diversity
    edge_count = float((grad_mag > 50).sum())
    edge_density = edge_count / (h * w)
    result["complexity"] = min(1.0, unique_ratio * 3 + edge_density * 2)

    # ── Face detection ──
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            logger.debug("Haar cascade not available for face detection")
        else:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            result["has_face"] = len(faces) > 0
    except Exception:
        logger.debug("Face detection failed", exc_info=True)

    logger.debug(
        "Image analysis: %dx%d (%.2f MP), anime=%s, face=%s, complexity=%.2f",
        w,
        h,
        result["megapixels"],
        result["is_anime"],
        result["has_face"],
        result["complexity"],
    )

    return result


def select_best_rembg_model(
    image_path: Optional[str] = None, analysis: Optional[dict] = None
) -> str:
    """Select the best Rembg model based on image content.

    Args:
        image_path: Path to input image (used if analysis not provided).
        analysis: Pre-computed image analysis dict.

    Returns:
        Rembg model name.
    """
    if analysis is None:
        analysis = analyze_image(image_path)

    if analysis.get("is_anime"):
        return "isnet-anime"
    if analysis.get("has_face"):
        return "u2net_human_seg"
    return "isnet-general-use"


def auto_enable_alpha_matting(
    image_path: Optional[str] = None, analysis: Optional[dict] = None
) -> bool:
    """Determine whether alpha matting should be enabled.

    Alpha matting improves edges for hair, fur, and semi-transparent regions
    but adds significant processing time.

    Args:
        image_path: Path to input image (used if analysis not provided).
        analysis: Pre-computed image analysis dict.

    Returns:
        True if alpha matting is recommended.
    """
    if analysis is None:
        analysis = analyze_image(image_path)

    # Enable for faces/portraits (hair) and complex edges
    return bool(analysis.get("has_face") or analysis.get("complexity", 0) > 0.7)


def auto_sprite_sizes(
    image_path: Optional[str] = None,
    analysis: Optional[dict] = None,
    quality: str = "balanced",
) -> list[int]:
    """Determine appropriate sprite output sizes based on input resolution and quality.

    Args:
        image_path: Path to input image (used if analysis not provided).
        analysis: Pre-computed image analysis dict.
        quality: Quality preset name.

    Returns:
        List of sprite sizes in pixels.
    """
    if analysis is None:
        analysis = analyze_image(image_path)

    mp = analysis.get("megapixels", 1.0)
    max_dim = max(analysis.get("width", 512), analysis.get("height", 512))

    if quality == "fast":
        return [64, 128]
    elif quality == "high":
        sizes = [128, 256, 512]
        if max_dim >= 1024:
            sizes.append(1024)
        return sizes
    elif quality == "ultra":
        sizes = [256, 512]
        if mp >= 2.0:
            sizes.append(1024)
        if mp >= 4.0:
            sizes.append(2048)
        return sizes
    else:  # balanced
        if mp >= 4.0:
            return [128, 256, 512]
        elif mp >= 1.0:
            return [64, 128, 256]
        else:
            return [64, 128]


def auto_vectorize_config(
    analysis: Optional[dict] = None,
    quality: str = "balanced",
) -> dict:
    """Auto-tune vectorization parameters based on image features.

    Args:
        analysis: Pre-computed image analysis dict.
        quality: Quality preset name.

    Returns:
        Dict with num_colors, simplify_tolerance, min_area, etc.
    """
    if analysis is None:
        analysis = {}

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    complexity = analysis.get("complexity", 0.5)

    # Scale num_colors with complexity
    base_colors = preset.get("num_colors", 16)
    if complexity > 0.7:
        num_colors = min(base_colors * 2, 64)
    elif complexity < 0.3:
        num_colors = max(base_colors // 2, 4)
    else:
        num_colors = base_colors

    return {
        "num_colors": num_colors,
        "simplify_tolerance": preset.get("simplify_tolerance", 0.8),
        "min_area": preset.get("min_area", 4),
        "enable_character_features": preset.get("enable_character_features", False),
        "feature_simplify_tolerance": preset.get("feature_simplify_tolerance", 0.5),
        "feature_min_area": preset.get("feature_min_area", 2),
        "preserve_line_art": preset.get("preserve_line_art", True),
        "line_art_threshold": preset.get("line_art_threshold", 30),
        "enable_eye_highlights": preset.get("enable_eye_highlights", True),
        "semantic_detail_level": preset.get("semantic_detail_level", "medium"),
        "gap_fill": preset.get("gap_fill", True),
    }


def resolve_preset_quality(quality: str, image_path: Optional[str] = None) -> str:
    """Resolve a ``quality`` string to an actual preset name.

    - ``"auto"`` → analyze image, pick best preset
    - ``""`` (empty) → ``"balanced"`` (safe default)
    - Valid preset name → returned as-is
    - Invalid → ``"balanced"`` with a warning

    Args:
        quality: Quality preset name or "auto" or "".
        image_path: Path to input image (required for "auto").

    Returns:
        Resolved preset name.
    """
    if not quality or quality not in VALID_QUALITY_LEVELS:
        if quality:
            logger.warning(
                "Unknown quality preset '%s', falling back to 'balanced'", quality
            )
        return "balanced"

    if quality == "auto":
        analysis = analyze_image(image_path) if image_path else {}
        # Select preset based on image features
        # High-res + high complexity → high quality
        # Face → high (to preserve details)
        # Low-res + low complexity → fast
        mp = analysis.get("megapixels", 0)
        complexity = analysis.get("complexity", 0.5)
        has_face = analysis.get("has_face", False)

        if mp >= 4.0 or (has_face and complexity > 0.6):
            return "ultra"
        elif mp >= 2.0 or has_face or complexity > 0.7:
            return "high"
        elif complexity < 0.3 and mp < 0.5:
            return "fast"
        else:
            return "balanced"

    return quality


def resolve_2d_config(
    quality: str = "",
    image_path: Optional[str] = None,
    user_overrides: Optional[dict] = None,
) -> dict:
    """Resolve a complete 2D pipeline configuration from a quality preset.

    This is the main entry point for quality-based configuration.
    Combines preset defaults + image analysis + user overrides.

    Args:
        quality: Quality preset ("fast", "balanced", "high", "ultra", "auto", "").
        image_path: Path to input image (for auto-detection).
        user_overrides: Individual parameter overrides (takes precedence).

    Returns:
        Dict of all pipeline parameters.
    """
    # 1. Resolve quality to a preset name
    preset_name = resolve_preset_quality(quality, image_path)
    preset = dict(QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["balanced"]))

    # 2. Analyze image if available
    analysis = analyze_image(image_path) if image_path else None

    # 3. Apply image-aware overrides
    if analysis:
        # Auto-select Rembg model
        if preset["background_remover"] == "rembg":
            preset["bg_model"] = select_best_rembg_model(analysis=analysis)

        # Auto-enable character features if face detected
        if analysis.get("has_face") and not preset.get("enable_character_features"):
            preset["enable_character_features"] = True
            preset["semantic_detail_level"] = "high"

        # Auto-enable alpha matting for complex edges
        if analysis.get("has_face") or analysis.get("complexity", 0) > 0.7:
            preset["alpha_matting"] = auto_enable_alpha_matting(analysis=analysis)

        # Auto-tune sprite sizes
        preset["sprite_sizes"] = auto_sprite_sizes(
            analysis=analysis, quality=preset_name
        )

        # Auto-tune vectorization
        vec_config = auto_vectorize_config(analysis=analysis, quality=preset_name)
        preset.update(vec_config)

    # 4. Apply user overrides (highest priority)
    if user_overrides:
        for key, value in user_overrides.items():
            if value is not None and key in preset:
                preset[key] = value

    # Remove internal keys
    preset.pop("label", None)

    return preset
