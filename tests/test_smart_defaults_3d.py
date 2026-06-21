"""
Tests for the 3D Smart Defaults module.

Tests validate auto-configuration logic for all three 3D pipelines:
- 3D-Gen (single image → 3D model)
- 3D-Photo (multi-image → photogrammetry)
- 3DGS (photos/video → Gaussian splats)
"""

import pytest
from pathlib import Path
from assetforge.backends.smart_defaults_3d import (
    analyze_3d_input_image,
    analyze_photo_set,
    analyze_video,
    resolve_preset_quality,
    resolve_3d_gen_config,
    resolve_3d_photo_config,
    resolve_3dgs_config,
)


class TestAnalyze3DInputImage:
    """Tests for single-image analysis for 3D generation."""

    def test_no_image_path(self):
        """Test analysis with no image path returns safe defaults."""
        result = analyze_3d_input_image(None)
        assert result["has_content"] is True
        assert result["megapixels"] == 0.0

    def test_nonexistent_image(self):
        """Test analysis with nonexistent image returns safe defaults."""
        result = analyze_3d_input_image("/nonexistent/image.png")
        assert result["has_content"] is False  # can't read

    def test_with_synthetic_image(self, tmp_path):
        """Test analysis with a synthetic test image."""
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "test.png"
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 200
        arr[50:200, 50:200] = [100, 150, 200]
        Image.fromarray(arr, "RGB").save(str(img_path))

        result = analyze_3d_input_image(str(img_path))
        assert result["has_content"] is True
        assert result["megapixels"] == 0.07  # 256*256/1_000_000
        assert result["width"] == 256
        assert result["height"] == 256
        assert result["is_square"] is True


class TestAnalyzePhotoSet:
    """Tests for photo set analysis for photogrammetry."""

    def test_empty_directory(self, tmp_path):
        """Test analysis of empty directory."""
        empty_dir = tmp_path / "empty_photos"
        empty_dir.mkdir()
        result = analyze_photo_set(str(empty_dir))
        assert result["num_photos"] == 0

    def test_nonexistent_directory(self):
        """Test analysis of nonexistent directory."""
        result = analyze_photo_set("/nonexistent/photos")
        assert result["num_photos"] == 0

    def test_with_synthetic_photos(self, tmp_path):
        """Test analysis with synthetic test photos."""
        import numpy as np
        from PIL import Image

        photo_dir = tmp_path / "synthetic_photos"
        photo_dir.mkdir()
        for i in range(5):
            arr = np.ones((128, 128, 3), dtype=np.uint8) * 200
            Image.fromarray(arr, "RGB").save(str(photo_dir / f"view_{i}.jpg"))

        result = analyze_photo_set(str(photo_dir))
        assert result["num_photos"] == 5
        assert result["avg_megapixels"] > 0
        assert result["total_megapixels"] > 0


class TestAnalyzeVideo:
    """Tests for video analysis for 3DGS."""

    def test_nonexistent_video(self):
        """Test analysis of nonexistent video."""
        result = analyze_video("/nonexistent/video.mp4")
        assert result["duration_s"] == 0.0
        assert result["frame_count"] == 0


class TestResolvePresetQuality:
    """Tests for quality preset resolution."""

    def test_empty_quality_returns_balanced(self):
        """Test empty quality returns 'balanced'."""
        assert resolve_preset_quality("") == "balanced"

    def test_none_returns_balanced(self):
        """Test None-quality returns 'balanced'."""
        assert resolve_preset_quality("") == "balanced"

    def test_valid_preset_returned_as_is(self):
        """Test valid preset names are returned unchanged."""
        for preset in ["fast", "balanced", "high", "ultra"]:
            assert resolve_preset_quality(preset) == preset

    def test_invalid_preset_falls_back_to_balanced(self):
        """Test invalid preset falls back to balanced."""
        assert resolve_preset_quality("nonexistent") == "balanced"

    def test_auto_with_gen_pipeline(self, tmp_path):
        """Test auto mode for gen pipeline with an image."""
        import numpy as np
        from PIL import Image

        img_path = tmp_path / "auto_test.png"
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
        Image.fromarray(arr, "RGB").save(str(img_path))

        preset = resolve_preset_quality("auto", str(img_path), pipeline_type="gen")
        assert preset in ("fast", "balanced", "high", "ultra")

    def test_auto_with_photo_pipeline(self, tmp_path):
        """Test auto mode for photo pipeline with a small photo set."""
        import numpy as np
        from PIL import Image

        photo_dir = tmp_path / "auto_photos"
        photo_dir.mkdir()
        for i in range(10):
            arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
            Image.fromarray(arr, "RGB").save(str(photo_dir / f"view_{i}.jpg"))

        preset = resolve_preset_quality("auto", str(photo_dir), pipeline_type="photo")
        assert preset in ("fast", "balanced", "high", "ultra")

    def test_auto_with_gs_pipeline(self):
        """Test auto mode for GS pipeline."""
        preset = resolve_preset_quality("auto", pipeline_type="gs")
        assert preset in ("fast", "balanced", "high", "ultra")


class TestResolve3DGenConfig:
    """Tests for 3D-Gen config resolution."""

    def test_balanced_preset_returns_all_params(self):
        """Test balanced preset returns expected parameters."""
        config = resolve_3d_gen_config("balanced")
        assert config["model"] == "auto"
        assert config["mesh_resolution"] == 256
        assert config["inference_steps"] == 50
        assert config["retopology"] is True
        assert config["generate_pbr"] is True
        assert config["texture_resolution"] == 2048

    def test_fast_preset_minimal_params(self):
        """Test fast preset returns minimal parameters."""
        config = resolve_3d_gen_config("fast")
        assert config["model"] == "triposr"
        assert config["mesh_resolution"] == 128
        assert config["inference_steps"] == 25
        assert config["retopology"] is False
        assert config["generate_pbr"] is False

    def test_ultra_preset_max_params(self):
        """Test ultra preset returns maximum parameters."""
        config = resolve_3d_gen_config("ultra")
        assert config["model"] == "trellis"
        assert config["mesh_resolution"] == 512
        assert config["inference_steps"] == 75
        assert config["texture_resolution"] == 8192

    def test_user_overrides_take_precedence(self):
        """Test user overrides override preset values."""
        config = resolve_3d_gen_config("fast", user_overrides={"model": "trellis"})
        assert config["model"] == "trellis"  # override
        assert config["mesh_resolution"] == 128  # from preset


class TestResolve3DPhotoConfig:
    """Tests for 3D-Photo config resolution."""

    def test_balanced_preset_returns_all_params(self):
        """Test balanced preset returns expected parameters."""
        config = resolve_3d_photo_config("balanced")
        assert config["feature_type"] == "sift"
        assert config["mesh_quality"] == "medium"
        assert config["retopology"] is True
        assert config["target_faces"] == 15000

    def test_fast_preset_minimal_params(self):
        """Test fast preset returns minimal parameters."""
        config = resolve_3d_photo_config("fast")
        assert config["feature_type"] == "orb"
        assert config["mesh_quality"] == "low"
        assert config["retopology"] is False

    def test_ultra_preset(self):
        """Test ultra preset."""
        config = resolve_3d_photo_config("ultra")
        assert config["feature_type"] == "sift"
        assert config["mesh_quality"] == "high"
        assert config["retopology"] is True
        assert config["target_faces"] == 50000


class TestResolve3DGSConfig:
    """Tests for 3DGS config resolution."""

    def test_balanced_preset_returns_all_params(self):
        """Test balanced preset returns expected parameters."""
        config = resolve_3dgs_config("balanced")
        assert config["train_quality"] == "medium"
        assert config["max_splats"] == 500_000
        assert config["extract_mesh"] is True
        assert config["compress"] is True

    def test_fast_preset_minimal_params(self):
        """Test fast preset returns minimal parameters."""
        config = resolve_3dgs_config("fast")
        assert config["train_quality"] == "low"
        assert config["max_splats"] == 100_000
        assert config["extract_mesh"] is False
        assert config["compress"] is False

    def test_ultra_preset(self):
        """Test ultra preset."""
        config = resolve_3dgs_config("ultra")
        assert config["train_quality"] == "high"
        assert config["max_splats"] == 2_000_000
        assert config["extract_mesh"] is True
        assert config["compress"] is True
