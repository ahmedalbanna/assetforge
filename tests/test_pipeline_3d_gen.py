"""
Integration tests for the 3D Generative AI Pipeline.

Tests validate pipeline construction, model selection, and
error handling for unavailable generative models.
"""

import pytest
from pathlib import Path
from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen


@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image for 3D generation."""
    import numpy as np
    from PIL import Image

    arr = np.ones((128, 128, 4), dtype=np.uint8) * 255
    arr[30:90, 30:90] = [100, 150, 200, 255]
    img_path = tmp_path / "test_concept.png"
    Image.fromarray(arr, "RGBA").save(str(img_path))
    return img_path


class TestPipeline3DGen:
    """Tests for the 3D generative AI pipeline."""

    def test_pipeline_initialization(self, test_image):
        """Test that the pipeline can be constructed with valid inputs."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_out",
            model_name="triposr",
        )
        assert pipeline.input_path == Path(test_image)
        assert pipeline.model_name == "triposr"
        assert pipeline.retopology is False

    def test_pipeline_initialization_with_options(self, test_image):
        """Test pipeline construction with all options."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_out_full",
            model_name="auto",
            retopology=True,
            target_faces=10000,
            generate_pbr=True,
            texture_resolution=1024,
            output_format="obj",
        )
        assert pipeline.model_name == "auto"
        assert pipeline.retopology is True
        assert pipeline.target_faces == 10000
        assert pipeline.generate_pbr is True
        assert pipeline.texture_resolution == 1024
        assert pipeline.output_format == "obj"

    def test_pipeline_run_no_model(self, test_image):
        """Test that pipeline handles missing models gracefully."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir=str(test_image.parent / "gen_output"),
            model_name="triposr",
        )
        results = pipeline.run()
        # The pipeline should handle the TripoSR stub gracefully
        assert len(results) > 0
        stage_results = {r.name: r for r in results}
        assert "ai_generation" in stage_results

    def test_pipeline_unknown_model(self, test_image):
        """Test that an unknown model name logs error but doesn't crash."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_unknown",
            model_name="nonexistent_model",
        )
        results = pipeline.run()
        # Pipeline should handle unknown model gracefully without raising
        assert len(results) > 0

    def test_pipeline_invalid_input(self):
        """Test pipeline with non-existent input."""
        pipeline = Pipeline3DGen(
            input_path="/nonexistent/image.png",
            output_dir="/tmp/test_invalid",
        )
        results = pipeline.run()
        # Pipeline returns empty stages when input missing
        assert len(results) == 0

    # ── Quality Preset Tests ──────────────────────────────────────────

    def test_quality_preset_fast(self, test_image):
        """Test 'fast' quality preset applies correct parameters."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_fast",
            quality="fast",
        )
        assert pipeline.quality_preset == "fast"
        assert pipeline.model_name == "triposr"
        assert pipeline.mesh_resolution == 128
        assert pipeline.inference_steps == 25
        assert pipeline.retopology is False
        assert pipeline.generate_pbr is False

    def test_quality_preset_balanced(self, test_image):
        """Test 'balanced' quality preset applies correct parameters."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_balanced",
            quality="balanced",
        )
        assert pipeline.quality_preset == "balanced"
        assert pipeline.model_name == "auto"
        assert pipeline.mesh_resolution == 256
        assert pipeline.inference_steps == 50
        assert pipeline.retopology is True

    def test_quality_preset_high(self, test_image):
        """Test 'high' quality preset applies correct parameters."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_high",
            quality="high",
        )
        assert pipeline.quality_preset == "high"
        assert pipeline.model_name == "hunyuan3d-2"
        assert pipeline.mesh_resolution == 512
        assert pipeline.inference_steps == 50
        assert pipeline.retopology is True
        assert pipeline.generate_pbr is True

    def test_quality_preset_ultra(self, test_image):
        """Test 'ultra' quality preset applies correct parameters."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_ultra",
            quality="ultra",
        )
        assert pipeline.quality_preset == "ultra"
        assert pipeline.model_name == "trellis"
        assert pipeline.mesh_resolution == 512
        assert pipeline.inference_steps == 75
        assert pipeline.retopology is True
        assert pipeline.generate_pbr is True

    def test_quality_preset_fully_overrides_individual(self, test_image):
        """Test that quality preset overrides all individual params."""
        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_gen_override",
            quality="fast",
            model_name="trellis",  # individual param (overridden by preset)
        )
        # Quality preset fully overrides individual params when set
        assert pipeline.quality_preset == "fast"
        assert pipeline.model_name == "triposr"  # from preset, individual ignored
        assert pipeline.mesh_resolution == 128  # from preset
        assert pipeline.inference_steps == 25  # from preset
