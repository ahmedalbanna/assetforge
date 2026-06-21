"""
Integration tests for the 3D Photogrammetry Pipeline.

Tests validate pipeline construction, stage sequencing, and
error handling for missing external dependencies.
"""

import pytest
from pathlib import Path
from assetforge.pipelines.pipeline_3d_photo import Pipeline3DPhoto


@pytest.fixture
def photo_set_dir(tmp_path):
    """Create a temporary directory with test images for photogrammetry."""
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    for i in range(3):
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 200
        arr[20:44, 20:44] = [100, 150, 200]
        Image.fromarray(arr, "RGB").save(str(img_dir / f"view_{i:02d}.jpg"))
    return img_dir


class TestPipeline3DPhoto:
    """Tests for the photogrammetry pipeline."""

    def test_pipeline_initialization(self, photo_set_dir):
        """Test that the pipeline can be constructed with valid inputs."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_out",
            feature_type="sift",
            mesh_quality="medium",
        )
        assert pipeline.input_dir == Path(photo_set_dir)
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "medium"
        assert pipeline.retopology is False

    def test_pipeline_initialization_with_retopology(self, photo_set_dir):
        """Test pipeline construction with retopology enabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_out",
            retopology=True,
            target_faces=5000,
            output_format="obj",
        )
        assert pipeline.retopology is True
        assert pipeline.target_faces == 5000
        assert pipeline.output_format == "obj"

    def test_pipeline_run_no_openmvg(self, photo_set_dir):
        """Test that pipeline handles missing tools gracefully."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir=str(photo_set_dir.parent / "output"),
        )
        results = pipeline.run()
        # The pipeline should handle missing tools gracefully
        assert len(results) > 0
        # First stage (feature extraction) should fail because OpenMVG is not installed
        # but the pipeline should continue and report the failure
        stage_results = {r.name: r for r in results}
        assert "feature_extraction" in stage_results

    def test_pipeline_output_dir_creation(self, photo_set_dir):
        """Test that output directory is created."""
        out_dir = photo_set_dir.parent / "photo_output"
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir=str(out_dir),
        )
        assert out_dir.exists()

    def test_pipeline_invalid_input_dir(self):
        """Test pipeline with non-existent input directory."""
        pipeline = Pipeline3DPhoto(
            input_dir="/nonexistent/path",
            output_dir="/tmp/test_invalid",
        )
        results = pipeline.run()
        # Pipeline returns empty stages when input missing
        assert len(results) == 0

    # ── Quality Preset Tests ──────────────────────────────────────────

    def test_quality_preset_fast(self, photo_set_dir):
        """Test 'fast' quality preset applies correct parameters."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_fast",
            quality="fast",
        )
        assert pipeline.quality_preset == "fast"
        assert pipeline.feature_type == "orb"
        assert pipeline.mesh_quality == "low"
        assert pipeline.retopology is False

    def test_quality_preset_balanced(self, photo_set_dir):
        """Test 'balanced' quality preset applies correct parameters."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_balanced",
            quality="balanced",
        )
        assert pipeline.quality_preset == "balanced"
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "medium"
        assert pipeline.retopology is True
        assert pipeline.target_faces == 15000
        assert pipeline.output_format == "glb"

    def test_quality_preset_high(self, photo_set_dir):
        """Test 'high' quality preset applies correct parameters."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_high",
            quality="high",
        )
        assert pipeline.quality_preset == "high"
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "high"
        assert pipeline.retopology is True
        assert pipeline.target_faces == 30000

    def test_quality_preset_ultra(self, photo_set_dir):
        """Test 'ultra' quality preset applies correct parameters."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_photo_ultra",
            quality="ultra",
        )
        assert pipeline.quality_preset == "ultra"
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "high"
        assert pipeline.retopology is True
        assert pipeline.target_faces == 50000

    def test_quality_preset_fully_overrides_individual(self, photo_set_dir):
        """Test that quality preset overrides all individual params."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_override",
            quality="fast",
            mesh_quality="high",  # individual param (overridden by preset)
        )
        # Quality preset fully overrides individual params when set
        assert pipeline.quality_preset == "fast"
        assert pipeline.feature_type == "orb"  # from preset
        assert pipeline.mesh_quality == "low"  # from preset, individual ignored

    def test_quality_preset_unknown(self, photo_set_dir):
        """Test unknown quality preset logs warning and uses individual params."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_unknown",
            quality="nonexistent",
            feature_type="sift",
            mesh_quality="high",
        )
        # Should fall back to individual params
        assert pipeline.quality_preset == "nonexistent"
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "high"
        assert pipeline.retopology is False
