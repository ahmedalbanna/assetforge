"""
Base pipeline class providing common infrastructure.

All pipelines inherit from this base class which provides:
- Stage sequencing and timing
- Output directory management
- Logging
- Error handling
- Input validation utilities
"""

import logging
import os
import time
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("assetforge")


# ── Input Validation Utilities ────────────────────────────────────


# Allowed file extensions for input types
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".fbx", ".ply", ".stl"}
ALLOWED_OUTPUT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".glb",
    ".obj",
    ".fbx",
    ".svg",
    ".spz",
    ".ply",
}

# Maximum file size (500 MB)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024


def _resolve_path(path: str | Path) -> Path:
    """Resolve and normalize a path, checking for path traversal.

    This is the base path resolution that performs safety checks
    but does NOT require the path to exist (existence should be
    checked at runtime by the caller).

    Args:
        path: File path to resolve

    Returns:
        Resolved Path object

    Raises:
        ValueError: If the path contains path traversal or is unsafe
    """
    p = Path(path).expanduser().resolve()

    # Path traversal check
    path_str = str(p)
    if ".." in path_str.split(os.sep):
        raise ValueError(f"Path contains '..' traversal: {p}")

    return p


def validate_input_path(path: str | Path, check_exists: bool = False) -> Path:
    """Validate and resolve an input file path.

    Args:
        path: Input file path to validate
        check_exists: If True, also checks the file exists and its size

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If check_exists is True and the file does not exist
        ValueError: If the path is invalid or too large
    """
    p = _resolve_path(path)

    if check_exists:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if not p.is_file():
            raise ValueError(f"Input path is not a file: {p}")
        try:
            file_size = p.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"File too large: {p.name} ({file_size / 1024 / 1024:.1f} MB). "
                    f"Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB"
                )
        except OSError as e:
            raise ValueError(f"Cannot access file: {p} — {e}")

    return p


def validate_input_image(path: str | Path, check_exists: bool = False) -> Path:
    """Validate an input image path with extension check.

    Args:
        path: Input image path
        check_exists: If True, also checks the file exists

    Returns:
        Resolved Path object

    Raises:
        ValueError: If extension is not a supported image format
    """
    p = validate_input_path(path, check_exists=check_exists)

    ext = p.suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}' for '{p.name}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    return p


def validate_input_mesh(path: str | Path, check_exists: bool = False) -> Path:
    """Validate an input mesh path with extension check.

    Args:
        path: Input mesh path
        check_exists: If True, also checks the file exists

    Returns:
        Resolved Path object

    Raises:
        ValueError: If extension is not a supported mesh format
    """
    p = validate_input_path(path, check_exists=check_exists)

    ext = p.suffix.lower()
    if ext not in ALLOWED_MESH_EXTENSIONS:
        raise ValueError(
            f"Unsupported mesh format '{ext}' for '{p.name}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_MESH_EXTENSIONS))}"
        )

    return p


def validate_input_dir(path: str | Path, check_exists: bool = False) -> Path:
    """Validate and resolve an input directory path.

    Args:
        path: Input directory path to validate
        check_exists: If True, also checks the directory exists

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If check_exists is True and the directory does not exist
        ValueError: If the path is invalid or not a directory
    """
    p = _resolve_path(path)

    if check_exists:
        if not p.exists():
            raise FileNotFoundError(f"Input directory not found: {p}")
        if not p.is_dir():
            raise ValueError(f"Input path is not a directory: {p}")

    return p


def validate_output_dir(path: str | Path) -> Path:
    """Validate and create an output directory.

    Ensures the output directory is within a reasonable location
    (not in system directories) and creates it if needed.

    Args:
        path: Output directory path

    Returns:
        Resolved Path object
    """
    p = Path(path).expanduser().resolve()

    # Prevent writing to sensitive system directories
    # Block writing directly to critical system directories only.
    # Root (/) is excluded from blocking since all paths are under it.
    system_dirs = {"/etc", "/bin", "/sbin", "/usr", "/boot", "/sys", "/proc", "/dev"}
    # Only check the root-most path, not every parent (since all paths are under /)
    resolved_str = str(p)
    if resolved_str in system_dirs or any(
        resolved_str.startswith(d + "/") or resolved_str == d for d in system_dirs
    ):
        raise ValueError(f"Cannot write output to system directory: {p}")

    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_sprite_sizes(sizes: list[int]) -> list[int]:
    """Validate sprite size values.

    Args:
        sizes: List of sprite dimensions in pixels

    Returns:
        Validated list of sizes

    Raises:
        ValueError: If any size is invalid
    """
    if not sizes:
        raise ValueError("Sprite sizes list cannot be empty")

    validated = []
    for s in sizes:
        if not isinstance(s, int) or s <= 0:
            raise ValueError(f"Invalid sprite size: {s}. Must be a positive integer.")
        if s > 8192:
            raise ValueError(f"Sprite size too large: {s}. Maximum allowed: 8192")
        validated.append(s)

    return sorted(validated)


def clamp(value: float, min_val: float, max_val: float, name: str = "value") -> float:
    """Clamp a numeric value within bounds.

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter (for error messages)

    Returns:
        Clamped value

    Raises:
        ValueError: If value is outside bounds
    """
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value


# ── Pipeline Base Class ───────────────────────────────────────────


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    name: str
    success: bool
    duration_s: float
    output_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePipeline(ABC):
    """Abstract base class for all asset pipelines."""

    def __init__(self, output_dir: str | Path, name: str = "pipeline"):
        self.name = name
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[StageResult] = []

    @abstractmethod
    def run(self) -> list[StageResult]:
        """Execute all stages of the pipeline."""
        ...

    def require_attr(self, attr_name: str, stage_name: str = "") -> None:
        """Check that a required attribute exists (set by a previous stage).

        Raises a clear error if the attribute is missing, which indicates
        the previous stage failed and this stage should be skipped.

        Args:
            attr_name: Name of the attribute to check
            stage_name: Human-readable name of the stage that should have set it
        """
        if not hasattr(self, attr_name):
            prev = stage_name or attr_name.strip("_").replace("_", " ")
            raise RuntimeError(
                f"Required attribute '{attr_name}' not found. "
                f"Previous stage '{prev}' likely failed. "
                f"This stage is being skipped."
            )

    def run_stage(self, name: str, fn, *args, **kwargs) -> StageResult:
        """Run a single pipeline stage with timing and error handling."""
        logger.info(f"[{self.name}] Stage: {name}...")
        start = time.time()

        try:
            result = fn(*args, **kwargs)
            duration = time.time() - start

            stage_result = StageResult(
                name=name,
                success=True,
                duration_s=round(duration, 2),
                output_paths=result
                if isinstance(result, list)
                else [result]
                if result
                else [],
            )
            logger.info(f"[{self.name}] ✓ {name} ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            logger.error(f"[{self.name}] ✗ {name} failed after {duration:.2f}s: {e}")
            stage_result = StageResult(
                name=name,
                success=False,
                duration_s=round(duration, 2),
                error=str(e),
            )

        self.stages.append(stage_result)
        return stage_result

    def summary(self) -> None:
        """Print a summary of all stages."""
        total_time = sum(s.duration_s for s in self.stages)
        successes = sum(1 for s in self.stages if s.success)
        failures = sum(1 for s in self.stages if not s.success)

        logger.info("=" * 50)
        logger.info(f"Pipeline '{self.name}' complete")
        logger.info(f"  Stages: {successes} succeeded, {failures} failed")
        logger.info(f"  Total time: {total_time:.2f}s")

        if failures > 0:
            for s in self.stages:
                if not s.success:
                    logger.warning(f"  Failed: {s.name} — {s.error}")

        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 50)
