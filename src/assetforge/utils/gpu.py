"""
GPU detection and management utilities.

Detects CUDA capability, VRAM, and recommends model selection
based on available hardware.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect_gpu() -> dict:
    """Detect GPU capabilities.

    Returns:
        dict with GPU information keys:
        - available: bool
        - device_count: int
        - device_name: str or None
        - vram_total_gb: float or None
        - compute_capability: tuple or None
        - cuda_version: str or None
    """
    result = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "vram_total_gb": 0.0,
        "compute_capability": None,
        "cuda_version": None,
        "pytorch_version": None,
    }

    try:
        import torch

        result["pytorch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            result["available"] = True
            result["device_count"] = torch.cuda.device_count()
            result["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["vram_total_gb"] = round(props.total_memory / 1e9, 2)
            result["compute_capability"] = (
                props.major,
                props.minor,
            )
    except ImportError:
        logger.warning("PyTorch not installed. GPU acceleration unavailable.")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")

    return result


def print_gpu_info(gpu_info: dict) -> None:
    """Pretty-print GPU information."""
    import json

    print("=" * 50)
    print("  AssetForge — System Information")
    print("=" * 50)

    if gpu_info["available"]:
        print(f"  GPU:          {gpu_info['device_name']}")
        print(f"  VRAM:         {gpu_info['vram_total_gb']} GB")
        print(f"  CUDA Compute: {gpu_info['compute_capability']}")
        print(f"  CUDA Version: {gpu_info['cuda_version']}")
        print(f"  PyTorch:      {gpu_info['pytorch_version']}")
    else:
        print("  GPU:          NOT AVAILABLE")
        print("  Running on CPU only — some pipelines may be slow.")

    print("=" * 50)


def select_3d_model(vram_gb: float) -> str:
    """Auto-select the best 3D generative model based on available VRAM."""
    if vram_gb >= 8.0:
        return "trellis"  # Best quality, needs >8GB
    elif vram_gb >= 4.0:
        return "hunyuan3d-2"  # Good quality with PBR textures
    else:
        return "triposr"  # Fast, works in limited VRAM


def get_optimal_device() -> str:
    """Return the best available device string for PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
    except ImportError:
        pass
    return "cpu"
