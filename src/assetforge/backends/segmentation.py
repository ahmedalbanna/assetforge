"""
Segmentation backend — removes backgrounds using SAM2 or Rembg.

Supports:
- SAM 2 (Meta's Segment Anything Model 2) — high quality, needs GPU
- Rembg (U²-Net) — lightweight, works on CPU
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assetforge")


def remove_background(
    input_path: str,
    output_path: str,
    model: str = "sam2",
    device: Optional[str] = None,
) -> Path:
    """Remove background from an image using the specified model.

    Args:
        input_path: Path to input image
        output_path: Path to save the output RGBA PNG
        model: 'sam2' or 'rembg'
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        Path to the output image
    """
    if model == "sam2":
        return _remove_background_sam2(input_path, output_path, device)
    elif model == "rembg":
        return _remove_background_rembg(input_path, output_path)
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
        # SAM2 model loading
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import numpy as np
        from PIL import Image

        # Load model (downloaded on first use)
        checkpoint = _get_sam2_checkpoint()
        model_cfg = "sam2_hiera_l.yaml"

        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

        # Load and process image
        image = Image.open(input_path).convert("RGB")
        np_image = np.array(image)

        # Auto-generate a center point prompt for the main object
        predictor.set_image(np_image)

        # Get center point of image as prompt
        h, w = np_image.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        # Use the best mask
        mask = masks[0]

        # Apply mask to create RGBA output
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


def _remove_background_rembg(input_path: str, output_path: str) -> Path:
    """Remove background using Rembg (U²-Net)."""
    logger.info("Using Rembg for background removal...")

    try:
        from rembg import remove as rembg_remove
        from PIL import Image

        with open(input_path, "rb") as f:
            input_data = f.read()

        output_data = rembg_remove(input_data)

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
