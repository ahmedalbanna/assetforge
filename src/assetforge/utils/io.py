"""
File I/O utilities for AssetForge.

Handles image loading, format detection, and output directory management.
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

# Supported input image formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_MESH_FORMATS = {".obj", ".fbx", ".glb", ".gltf", ".stl", ".ply"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path string to an absolute Path."""
    return Path(path).expanduser().resolve()


def ensure_output_dir(path: Union[str, Path]) -> Path:
    """Create output directory if it doesn't exist and return the path."""
    out = resolve_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def detect_input_type(path: Union[str, Path]) -> str:
    """Detect whether input is an image, directory, video, or mesh file.

    Returns:
        One of: 'image', 'directory', 'video', 'mesh', 'unknown'
    """
    p = resolve_path(path)

    if p.is_dir():
        return "directory"
    if p.is_file():
        ext = p.suffix.lower()
        if ext in SUPPORTED_IMAGE_FORMATS:
            return "image"
        if ext in SUPPORTED_MESH_FORMATS:
            return "mesh"
        if ext in SUPPORTED_VIDEO_FORMATS:
            return "video"
    return "unknown"


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image file as a PIL Image."""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGBA")


def load_image_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load an image as a numpy array (H, W, 4)."""
    img = load_image(path)
    return np.array(img, dtype=np.uint8)


def save_png(image: Union[Image.Image, np.ndarray], path: Union[str, Path]) -> Path:
    """Save an image as PNG and return the output path."""
    p = resolve_path(path)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(p, format="PNG")
    return p


def get_image_files(directory: Union[str, Path]) -> list[Path]:
    """Get all supported image files from a directory, sorted by name."""
    d = resolve_path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    files = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            files.append(f)
    return files
