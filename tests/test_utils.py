"""
Tests for utility modules.
"""

from pathlib import Path
import tempfile
from assetforge.utils.io import (
    ensure_output_dir,
    detect_input_type,
    load_image,
    get_image_files,
)


def test_ensure_output_dir(tmp_path: Path):
    """Test output directory creation and return."""
    out = ensure_output_dir(tmp_path / "nested" / "dir" / "path")
    assert out.exists()
    assert out.is_dir()


def test_detect_input_type(tmp_path: Path):
    """Test input type detection."""
    # Image
    img = tmp_path / "test.jpg"
    img.write_text("fake image")
    assert detect_input_type(str(img)) == "image"

    # Directory
    assert detect_input_type(str(tmp_path)) == "directory"

    # Unknown extension
    unknown = tmp_path / "test.xyz"
    unknown.write_text("unknown")
    assert detect_input_type(str(unknown)) == "unknown"


def test_detect_input_type_nonexistent(tmp_path: Path):
    """Test detection with non-existent file."""
    assert detect_input_type(str(tmp_path / "nonexistent.png")) == "unknown"


def test_get_image_files(tmp_path: Path):
    """Test image file discovery in directory."""
    # Create test files
    (tmp_path / "img1.jpg").write_text("")
    (tmp_path / "img2.png").write_text("")
    (tmp_path / "notes.txt").write_text("")  # Should be ignored
    (tmp_path / "subdir").mkdir()

    files = get_image_files(str(tmp_path))
    assert len(files) == 2
    assert all(f.suffix in [".jpg", ".png"] for f in files)
