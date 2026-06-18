"""
Configuration management for AssetForge.

Supports:
- Default configuration
- Environment variable overrides
- TOML/YAML config file
"""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AssetForgeConfig:
    """Global configuration for AssetForge."""

    # Paths
    output_dir: str = "./output"
    cache_dir: str = "~/.cache/assetforge"
    model_dir: str = "~/.cache/assetforge/models"

    # GPU
    gpu_device: int = 0
    use_cuda: bool = True
    use_rust_acceleration: bool = True

    # Default model selection
    background_remover: str = "sam2"  # sam2 | rembg
    gen3d_model: str = "auto"  # auto | triposr | hunyuan3d-2 | trellis

    # Output defaults
    output_format: str = "glb"
    texture_resolution: int = 2048
    target_faces_3d: int = 15000

    # Blender
    blender_path: str = "blender"

    # Logging
    verbose: bool = False
    quiet: bool = False

    @classmethod
    def from_env(cls) -> "AssetForgeConfig":
        """Load configuration from environment variables."""
        config = cls()

        if "ASSETFORGE_OUTPUT_DIR" in os.environ:
            config.output_dir = os.environ["ASSETFORGE_OUTPUT_DIR"]
        if "ASSETFORGE_CACHE_DIR" in os.environ:
            config.cache_dir = os.environ["ASSETFORGE_CACHE_DIR"]
        if "ASSETFORGE_GPU_DEVICE" in os.environ:
            config.gpu_device = int(os.environ["ASSETFORGE_GPU_DEVICE"])
        if "ASSETFORGE_NO_CUDA" in os.environ:
            config.use_cuda = os.environ["ASSETFORGE_NO_CUDA"].lower() not in (
                "1",
                "true",
                "yes",
            )
        if "ASSETFORGE_BLENDER_PATH" in os.environ:
            config.blender_path = os.environ["ASSETFORGE_BLENDER_PATH"]

        return config

    def resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
        self.model_dir = str(Path(self.model_dir).expanduser().resolve())
