"""
Integration tests for the Gaussian Splatting Pipeline.

Tests validate pipeline construction, stage sequencing, and
error handling for missing external dependencies.
"""

import pytest
from pathlib import Path
from assetforge.pipelines.pipeline_3dgs import Pipeline3DGS


@pytest.fixture
def photo_dir(tmp_path):
    """Create a temporary directory with test images for 3DGS."""
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "photos_3dgs"
    img_dir.mkdir()
    for i in range(3):
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 200
        arr[20:44, 20:44] = [100, 150, 200]
        Image.fromarray(arr, "RGB").save(str(img_dir / f"frame_{i:04d}.jpg"))
    return img_dir


class TestPipeline3DGS:
    """Tests for the Gaussian splatting pipeline."""

    def test_pipeline_initialization(self, photo_dir):
        """Test that the pipeline can be constructed with valid inputs."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_3dgs_out",
            train_quality="low",
        )
        assert pipeline.input_path == Path(photo_dir)
        assert pipeline.train_quality == "low"
        assert pipeline.extract_mesh is False
        assert pipeline.compress is False

    def test_pipeline_initialization_with_options(self, photo_dir):
        """Test pipeline construction with all optional features."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_3dgs_full",
            extract_mesh=True,
            mesh_method="milo",
            compress=True,
            train_quality="high",
            max_splats=500_000,
        )
        assert pipeline.extract_mesh is True
        assert pipeline.mesh_method == "milo"
        assert pipeline.compress is True
        assert pipeline.train_quality == "high"
        assert pipeline.max_splats == 500_000

    def test_pipeline_run_no_colmap(self, photo_dir):
        """Test that pipeline handles missing COLMAP gracefully."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir=str(photo_dir.parent / "3dgs_output"),
            train_quality="low",
        )
        results = pipeline.run()
        # Pipeline should handle missing tools
        assert len(results) > 0
        stage_results = {r.name: r for r in results}
        assert "camera_poses" in stage_results

    def test_pipeline_video_input(self, tmp_path):
        """Test pipeline handles video input detection."""
        # Create a small test video (just need the path to exist-ish)
        import numpy as np
        import cv2

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 10, (32, 32))
        for _ in range(5):
            frame = np.ones((32, 32, 3), dtype=np.uint8) * 128
            out.write(frame)
        out.release()

        pipeline = Pipeline3DGS(
            input_path=str(video_path),
            output_dir=str(tmp_path / "video_output"),
        )
        assert pipeline.input_path.exists()

    def test_pipeline_invalid_input(self):
        """Test pipeline with non-existent input."""
        pipeline = Pipeline3DGS(
            input_path="/nonexistent/path",
            output_dir="/tmp/test_3dgs_invalid",
        )
        results = pipeline.run()
        # Pipeline returns empty stages when input missing
        assert len(results) == 0

    # ── Quality Preset Tests ──────────────────────────────────────────

    def test_quality_preset_fast(self, photo_dir):
        """Test 'fast' quality preset applies correct parameters."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_gs_fast",
            quality="fast",
        )
        assert pipeline.quality_preset == "fast"
        assert pipeline.train_quality == "low"
        assert pipeline.max_splats == 100_000
        assert pipeline.extract_mesh is False
        assert pipeline.compress is False

    def test_quality_preset_balanced(self, photo_dir):
        """Test 'balanced' quality preset applies correct parameters."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_gs_balanced",
            quality="balanced",
        )
        assert pipeline.quality_preset == "balanced"
        assert pipeline.train_quality == "medium"
        assert pipeline.max_splats == 500_000
        assert pipeline.extract_mesh is True
        assert pipeline.mesh_method == "milo"
        assert pipeline.compress is True

    def test_quality_preset_high(self, photo_dir):
        """Test 'high' quality preset applies correct parameters."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_gs_high",
            quality="high",
        )
        assert pipeline.quality_preset == "high"
        assert pipeline.train_quality == "high"
        assert pipeline.max_splats == 1_000_000
        assert pipeline.extract_mesh is True
        assert pipeline.mesh_method == "sugar"

    def test_quality_preset_ultra(self, photo_dir):
        """Test 'ultra' quality preset applies correct parameters."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_gs_ultra",
            quality="ultra",
        )
        assert pipeline.quality_preset == "ultra"
        assert pipeline.train_quality == "high"
        assert pipeline.max_splats == 2_000_000
        assert pipeline.extract_mesh is True
        assert pipeline.compress is True

    def test_quality_preset_fully_overrides_individual(self, photo_dir):
        """Test that quality preset overrides all individual params."""
        pipeline = Pipeline3DGS(
            input_path=str(photo_dir),
            output_dir="/tmp/test_gs_override",
            quality="fast",
            train_quality="high",  # individual param (overridden by preset)
        )
        # Quality preset fully overrides individual params when set
        assert pipeline.quality_preset == "fast"
        assert pipeline.train_quality == "low"  # from preset, individual ignored
        assert pipeline.max_splats == 100_000  # from preset
        assert pipeline.extract_mesh is False  # from preset
