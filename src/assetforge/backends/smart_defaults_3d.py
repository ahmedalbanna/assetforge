"""
Smart Defaults for 3D Pipelines — AI-driven auto-configuration.

Analyzes input images, photo sets, and videos to automatically select
optimal parameters for 3D generation, photogrammetry, and Gaussian
splatting pipelines. Single ``quality`` preset replaces 5+ individual settings.

Presets (shared across all 3D pipelines):
    - **fast**: Quick preview, lowest quality, fastest
    - **balanced**: General-purpose, good quality (*default*)
    - **high**: Detailed output, longer processing
    - **ultra**: Maximum quality, longest processing
    - **auto**: Analyzes the input and selects best preset automatically

Usage:
    >>> from assetforge.backends.smart_defaults_3d import (
    ...     resolve_3d_gen_config,
    ...     resolve_3d_photo_config,
    ...     resolve_3dgs_config,
    ... )
    >>> params = resolve_3d_gen_config("balanced", "/path/to/image.png")
    >>> params["model_name"]
    'auto'
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")

# ── Shared 3D Quality Presets ───────────────────────────────────────────────
# These map to pipeline-specific QUALITY_PRESETS in each pipeline module.
# The resolve functions below translate to the actual param names each pipeline expects.

PRESET_DESCRIPTIONS: dict[str, dict] = {
    "fast": {
        "label": "Fast — quick preview, lowest quality",
        "intent": "speed",
    },
    "balanced": {
        "label": "Balanced — general-purpose 3D assets (default)",
        "intent": "balanced",
    },
    "high": {
        "label": "High — detailed assets, longer processing",
        "intent": "quality",
    },
    "ultra": {
        "label": "Ultra — maximum quality, longest processing",
        "intent": "quality",
    },
}

VALID_QUALITY_LEVELS = set(PRESET_DESCRIPTIONS.keys()) | {"auto"}


# ── Input Analysis ──────────────────────────────────────────────────────────


def analyze_3d_input_image(
    image_path: Optional[str] = None,
) -> dict:
    """Analyze a single input image for 3D generation auto-configuration.

    Uses lightweight heuristics (no model downloads) to classify the image:
    - **has_content**: whether the image has meaningful pixel content
    - **is_anime**: whether the image appears to be anime/manga style
    - **has_face**: whether a face is detected
    - **complexity**: edge-based complexity score (0.0–1.0)
    - **megapixels**: input resolution in megapixels
    - **aspect_ratio**: width/height ratio
    - **is_square**: whether the image is roughly square

    Args:
        image_path: Path to the input image.

    Returns:
        Dict of image features relevant to 3D generation.
    """
    result: dict = {
        "has_content": True,
        "is_anime": False,
        "has_face": False,
        "complexity": 0.5,
        "megapixels": 0.0,
        "width": 0,
        "height": 0,
        "aspect_ratio": 1.0,
        "is_square": True,
    }

    if image_path is None:
        return result

    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read image: %s", image_path)
            result["has_content"] = False
            return result

        # Convert to 3-channel BGR for analysis
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            # Check if mostly transparent
            alpha = img[:, :, 3]
            if np.mean(alpha > 0) < 0.05:
                result["has_content"] = False
                return result
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        result["width"] = w
        result["height"] = h
        result["megapixels"] = round(h * w / 1_000_000, 2)
        result["aspect_ratio"] = round(w / h, 2) if h > 0 else 1.0
        result["is_square"] = abs(1.0 - result["aspect_ratio"]) < 0.15

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Anime detection via color palette analysis ──
        pixel_colors = img.reshape(-1, 3)
        if len(pixel_colors) > 10_000:
            indices = np.random.choice(len(pixel_colors), 10_000, replace=False)
            sampled = pixel_colors[indices]
        else:
            sampled = pixel_colors

        quantized = (sampled // 16).astype(np.uint8)
        unique_bins = len(np.unique(quantized, axis=0))
        unique_ratio = unique_bins / max(len(sampled), 1)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_variance = float(np.var(grad_mag))

        result["is_anime"] = unique_ratio < 0.15 and grad_variance < 2000

        # ── Complexity score ──
        edge_count = float((grad_mag > 50).sum())
        edge_density = edge_count / (h * w)
        result["complexity"] = min(1.0, unique_ratio * 3 + edge_density * 2)

        # ── Face detection ──
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                )
                result["has_face"] = len(faces) > 0
        except Exception:
            pass

        logger.debug(
            "3D input analysis: %dx%d (%.2f MP), anime=%s, face=%s, complexity=%.2f",
            w,
            h,
            result["megapixels"],
            result["is_anime"],
            result["has_face"],
            result["complexity"],
        )

    except ImportError:
        logger.debug("OpenCV not available for 3D input analysis")

    return result


def analyze_photo_set(photo_dir: str) -> dict:
    """Analyze a directory of photos for photogrammetry configuration.

    Args:
        photo_dir: Path to directory containing input photos.

    Returns:
        Dict with: num_photos, avg_resolution, total_megapixels, has_overlap_estimate.
    """
    result: dict = {
        "num_photos": 0,
        "avg_megapixels": 0.0,
        "total_megapixels": 0.0,
        "avg_aspect_ratio": 1.0,
        "min_dimension": 0,
        "max_dimension": 0,
    }

    photo_dir_path = Path(photo_dir)
    if not photo_dir_path.is_dir():
        return result

    # Find image files
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
    photos = sorted(
        [p for p in photo_dir_path.iterdir() if p.suffix.lower() in extensions]
    )
    result["num_photos"] = len(photos)

    if not photos:
        return result

    # Sample a few photos for resolution analysis
    try:
        from PIL import Image

        total_mp = 0.0
        aspect_ratios = []
        min_dim = float("inf")
        max_dim = 0

        # Analyze up to 20 photos for speed
        sample = photos[:20]
        for p in sample:
            try:
                img = Image.open(p)
                w, h = img.size
                mp = w * h / 1_000_000
                total_mp += mp
                aspect_ratios.append(w / h if h > 0 else 1.0)
                min_dim = min(min_dim, w, h)
                max_dim = max(max_dim, w, h)
            except Exception:
                continue

        if sample:
            result["avg_megapixels"] = round(total_mp / len(sample), 2)
            result["total_megapixels"] = round(total_mp, 2)
            result["avg_aspect_ratio"] = round(
                sum(aspect_ratios) / len(aspect_ratios), 2
            )
            result["min_dimension"] = int(min_dim) if min_dim != float("inf") else 0
            result["max_dimension"] = int(max_dim)

    except ImportError:
        logger.debug("PIL not available for photo set analysis")

    logger.debug(
        "Photo set analysis: %d photos, avg %.2f MP, total %.2f MP",
        result["num_photos"],
        result["avg_megapixels"],
        result["total_megapixels"],
    )
    return result


def analyze_video(video_path: str) -> dict:
    """Analyze a video file for 3DGS auto-configuration.

    Args:
        video_path: Path to the input video file.

    Returns:
        Dict with: duration_s, frame_count, fps, resolution, has_motion.
    """
    result: dict = {
        "duration_s": 0.0,
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
    }

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return result

        result["fps"] = cap.get(cv2.CAP_PROP_FPS)
        result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if result["fps"] > 0:
            result["duration_s"] = round(result["frame_count"] / result["fps"], 2)

        cap.release()

    except ImportError:
        logger.debug("OpenCV not available for video analysis")

    logger.debug(
        "Video analysis: %.1fs, %d frames @ %.1f fps, %dx%d",
        result["duration_s"],
        result["frame_count"],
        result["fps"],
        result["width"],
        result["height"],
    )
    return result


# ── Preset Resolution ───────────────────────────────────────────────────────


def resolve_preset_quality(
    quality: str,
    input_path: Optional[str] = None,
    pipeline_type: str = "gen",
) -> str:
    """Resolve a ``quality`` string to an actual preset name.

    - ``"auto"`` → analyze input, pick best preset
    - ``""`` (empty) → ``"balanced"`` (safe default)
    - Valid preset name → returned as-is
    - Invalid → ``"balanced"`` with a warning

    Args:
        quality: Quality preset name or "auto" or "".
        input_path: Path to input (image, photo dir, or video).
        pipeline_type: Pipeline type — "gen", "photo", or "gs".

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
        # Analyze input based on pipeline type
        if pipeline_type == "gen":
            analysis = analyze_3d_input_image(input_path) if input_path else {}
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

        elif pipeline_type == "photo":
            analysis = analyze_photo_set(input_path) if input_path else {}
            num_photos = analysis.get("num_photos", 0)
            total_mp = analysis.get("total_megapixels", 0)

            if total_mp >= 100 or num_photos >= 100:
                return "ultra"
            elif total_mp >= 30 or num_photos >= 40:
                return "high"
            elif total_mp < 5 or num_photos < 10:
                return "fast"
            else:
                return "balanced"

        elif pipeline_type == "gs":
            analysis = {}
            if input_path:
                from assetforge.utils.io import detect_input_type

                input_type = detect_input_type(input_path)
                if input_type == "video":
                    analysis = analyze_video(input_path)
                else:
                    analysis = analyze_photo_set(input_path)

            duration = analysis.get("duration_s", 0)
            num_frames = analysis.get("frame_count", 0)
            num_photos = analysis.get("num_photos", 0)

            total_inputs = max(num_frames, num_photos)
            if duration > 60 or total_inputs > 500:
                return "ultra"
            elif duration > 20 or total_inputs > 100:
                return "high"
            elif duration < 5 or total_inputs < 20:
                return "fast"
            else:
                return "balanced"

    return quality


# ── Pipeline-Specific Config Resolvers ──────────────────────────────────────


def resolve_3d_gen_config(
    quality: str = "",
    image_path: Optional[str] = None,
    user_overrides: Optional[dict] = None,
) -> dict:
    """Resolve a complete 3D-Gen pipeline configuration from a quality preset.

    Combines preset defaults + image analysis + user overrides.

    Args:
        quality: Quality preset ("fast", "balanced", "high", "ultra", "auto", "").
        image_path: Path to input image (for auto-detection).
        user_overrides: Individual parameter overrides (takes precedence).

    Returns:
        Dict of all 3D-Gen pipeline parameters.
    """
    from assetforge.pipelines.pipeline_3d_gen import QUALITY_PRESETS as GEN_PRESETS

    # 1. Resolve quality to a preset name
    preset_name = resolve_preset_quality(quality, image_path, pipeline_type="gen")
    preset = dict(GEN_PRESETS.get(preset_name, GEN_PRESETS["balanced"]))

    # 2. Analyze image if available
    analysis = analyze_3d_input_image(image_path) if image_path else None

    # 3. Apply image-aware overrides
    if analysis:
        # Anime images benefit from simpler models with fewer steps
        if analysis.get("is_anime") and preset.get("inference_steps", 50) > 30:
            preset["inference_steps"] = max(preset["inference_steps"] // 2, 25)
            preset["guidance_scale"] = max(preset.get("guidance_scale", 5.0) - 1.0, 3.0)

        # Face images need more geometric detail
        if analysis.get("has_face"):
            preset["mesh_resolution"] = max(preset.get("mesh_resolution", 256), 256)
            preset["retopology"] = True

        # Low-res images don't need high texture resolution
        if analysis.get("megapixels", 1.0) < 0.5:
            preset["texture_resolution"] = min(
                preset.get("texture_resolution", 2048), 1024
            )

    # 4. Apply user overrides (highest priority)
    if user_overrides:
        for key, value in user_overrides.items():
            if value is not None and key in preset:
                preset[key] = value

    # Remove internal keys
    preset.pop("label", None)

    return preset


def resolve_3d_photo_config(
    quality: str = "",
    photo_dir: Optional[str] = None,
    user_overrides: Optional[dict] = None,
) -> dict:
    """Resolve a complete 3D-Photo pipeline configuration from a quality preset.

    Args:
        quality: Quality preset ("fast", "balanced", "high", "ultra", "auto", "").
        photo_dir: Path to input photo directory (for auto-detection).
        user_overrides: Individual parameter overrides (takes precedence).

    Returns:
        Dict of all 3D-Photo pipeline parameters.
    """
    from assetforge.pipelines.pipeline_3d_photo import QUALITY_PRESETS as PHOTO_PRESETS

    # 1. Resolve quality to a preset name
    preset_name = resolve_preset_quality(quality, photo_dir, pipeline_type="photo")
    preset = dict(PHOTO_PRESETS.get(preset_name, PHOTO_PRESETS["balanced"]))

    # 2. Analyze photo set if available
    analysis = analyze_photo_set(photo_dir) if photo_dir else None

    # 3. Apply photo-set-aware overrides
    if analysis:
        num_photos = analysis.get("num_photos", 0)
        avg_mp = analysis.get("avg_megapixels", 0)

        # Few photos → use better features to maximize matches
        if num_photos < 15:
            preset["feature_type"] = "sift"

        # High-res photos → no need for ultra-high mesh quality
        if avg_mp > 10:
            preset["mesh_quality"] = "medium"

        # Many photos → reduce target faces proportionally
        if num_photos > 100 and preset.get("retopology"):
            preset["target_faces"] = min(preset.get("target_faces", 15000), 30000)

    # 4. Apply user overrides
    if user_overrides:
        for key, value in user_overrides.items():
            if value is not None and key in preset:
                preset[key] = value

    return preset


def resolve_3dgs_config(
    quality: str = "",
    input_path: Optional[str] = None,
    user_overrides: Optional[dict] = None,
) -> dict:
    """Resolve a complete 3DGS pipeline configuration from a quality preset.

    Args:
        quality: Quality preset ("fast", "balanced", "high", "ultra", "auto", "").
        input_path: Path to input (video or photo dir, for auto-detection).
        user_overrides: Individual parameter overrides (takes precedence).

    Returns:
        Dict of all 3DGS pipeline parameters.
    """
    from assetforge.pipelines.pipeline_3dgs import QUALITY_PRESETS as GS_PRESETS

    # 1. Resolve quality to a preset name
    preset_name = resolve_preset_quality(quality, input_path, pipeline_type="gs")
    preset = dict(GS_PRESETS.get(preset_name, GS_PRESETS["balanced"]))

    # 2. Analyze video or photo set if available
    analysis = {}
    if input_path:
        from assetforge.utils.io import detect_input_type

        input_type = detect_input_type(input_path)
        if input_type == "video":
            analysis = analyze_video(input_path)
        else:
            analysis = analyze_photo_set(input_path)

    # 3. Apply input-aware overrides
    if analysis:
        duration = analysis.get("duration_s", 0)
        num_frames = analysis.get("frame_count", 0)
        num_photos = analysis.get("num_photos", 0)
        total = max(num_frames, num_photos)

        # Long video → scale up splats proportionally
        if duration > 30 or total > 200:
            preset["max_splats"] = max(preset.get("max_splats", 500_000), 1_000_000)

        # Short video → fewer splats, fast training
        if duration < 3 and duration > 0:
            preset["max_splats"] = min(preset.get("max_splats", 100_000), 100_000)
            preset["train_quality"] = "low"

    # 4. Apply user overrides
    if user_overrides:
        for key, value in user_overrides.items():
            if value is not None and key in preset:
                preset[key] = value

    return preset
