"""
Segmentation backend — removes backgrounds using SAM2 or Rembg.

Supports:
- SAM 2 (Meta's Segment Anything Model 2) — high quality, needs GPU
- Rembg (U²-Net, IS-Net) — lightweight, works on CPU, multiple models
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")

# Valid Rembg model names
REMBG_MODELS = {
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "isnet-general-use",
    "isnet-anime",
    "silueta",
    "sam",
}

# Default Rembg model
_DEFAULT_REMBG_MODEL = "isnet-general-use"


def remove_background(
    input_path: str,
    output_path: str,
    model: str = "sam2",
    device: Optional[str] = None,
    bg_model: Optional[str] = None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    post_process_mask: bool = False,
) -> Path:
    """Remove background from an image using the specified model.

    Args:
        input_path: Path to input image
        output_path: Path to save the output RGBA PNG
        model: 'sam2' or 'rembg'
        device: 'cuda', 'cpu', or None (auto-detect)
        bg_model: Rembg specific model name (e.g., 'isnet-general-use', 'u2net', etc.)
        alpha_matting: Enable alpha matting for finer edges
        alpha_matting_foreground_threshold: Foreground threshold (0-255)
        alpha_matting_background_threshold: Background threshold (0-255)
        alpha_matting_erode_size: Erosion kernel size for alpha matting
        post_process_mask: Apply post-processing to the mask

    Returns:
        Path to the output image
    """
    if model == "sam2":
        return _remove_background_sam2(input_path, output_path, device)
    elif model == "rembg":
        return _remove_background_rembg(
            input_path,
            output_path,
            bg_model=bg_model,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask,
        )
    else:
        raise ValueError(
            f"Unknown background removal model: {model}. Use 'sam2' or 'rembg'."
        )


def _remove_background_sam2(
    input_path: str, output_path: str, device: Optional[str] = None
) -> Path:
    """Remove background using SAM 2 model."""
    logger.info("Using SAM 2 for background removal...")

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import numpy as np
        from PIL import Image

        checkpoint = _get_sam2_checkpoint()
        model_cfg = "sam2_hiera_l.yaml"

        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

        image = Image.open(input_path).convert("RGB")
        np_image = np.array(image)

        predictor.set_image(np_image)

        h, w = np_image.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]
        rgba = np.dstack((np_image, mask * 255)).astype(np.uint8)
        result = Image.fromarray(rgba, "RGBA")
        result.save(output_path)

        logger.info(f"SAM 2 segmentation complete → {output_path}")

    except ImportError as e:
        raise ImportError(
            f"SAM 2 requires additional packages: pip install sam2\n"
            f"Or use --bg-remover rembg for a lighter alternative.\n"
            f"Original error: {e}"
        )

    return Path(output_path)


def _remove_background_rembg(
    input_path: str,
    output_path: str,
    bg_model: Optional[str] = None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    post_process_mask: bool = False,
) -> Path:
    """Remove background using Rembg with configurable model and alpha matting."""
    model_name = bg_model or _DEFAULT_REMBG_MODEL

    logger.info(
        "Using Rembg for background removal (model=%s, alpha_matting=%s)...",
        model_name,
        alpha_matting,
    )

    try:
        from rembg import remove as rembg_remove, new_session
        from PIL import Image

        # Create session with the specified model
        session = new_session(model_name=model_name)

        with open(input_path, "rb") as f:
            input_data = f.read()

        output_data = rembg_remove(
            input_data,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask,
        )

        with open(output_path, "wb") as f:
            f.write(output_data)

        logger.info(f"Rembg background removal complete → {output_path}")

    except ImportError:
        raise ImportError("Rembg is required. Install with: pip install rembg")

    return Path(output_path)


def _get_sam2_checkpoint() -> str:
    """Get or download the SAM 2 model checkpoint."""
    from assetforge.models.registry import download_model

    return str(download_model("sam2"))
