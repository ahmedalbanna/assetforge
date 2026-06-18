"""
AssetForge — Convert real-world images into game-ready digital assets.

Supports:
- 2D sprite extraction (background removal → vectorization → sprite export)
- 3D photogrammetry (multi-view → SfM → MVS → mesh → retopology)
- 3D generative AI (single image → mesh → textures)
- 3D Gaussian splatting (photos → 3DGS → compression → export)
"""

__version__ = "0.1.0"
__author__ = "AssetForge Contributors"

from assetforge.utils.gpu import (
    detect_gpu,
    print_gpu_info,
    select_3d_model,
    get_optimal_device,
)
