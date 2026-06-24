"""
Tests for the DreamGaussian backend.

Validates initialization, fallback behavior, and parameter handling.
Actual model inference is not tested (requires GPU + heavy deps).
"""

import pytest
from pathlib import Path


class TestDreamGaussianBackend:
    """Tests for DreamGaussianBackend."""

    def test_import(self):
        """Test that the module can be imported."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        assert DreamGaussianBackend is not None

    def test_init_default_device(self):
        """Test initialization with default device detection."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        backend = DreamGaussianBackend()
        assert backend.device in ("cuda", "cpu")
        assert backend._repo_ready is False

    def test_init_cpu(self):
        """Test initialization with explicit CPU device."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        backend = DreamGaussianBackend(device="cpu")
        assert backend.device == "cpu"

    def test_dreamgaussian_deps_defined(self):
        """Test that DreamGaussian dependencies are defined."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        assert len(DreamGaussianBackend.DREAMGAUSSIAN_DEPS) > 0
        assert "diffusers" in " ".join(DreamGaussianBackend.DREAMGAUSSIAN_DEPS)
        assert "nvdiffrast" in " ".join(DreamGaussianBackend.DREAMGAUSSIAN_DEPS)

    def test_generate_raises_without_full_deps(self, tmp_path):
        """Test that generate raises RuntimeError when deps not fully available."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        backend = DreamGaussianBackend(device="cpu")

        input_path = tmp_path / "test.png"
        # Create a minimal dummy image
        import numpy as np
        from PIL import Image

        arr = np.ones((64, 64, 3), dtype=np.uint8) * 200
        Image.fromarray(arr, "RGB").save(str(input_path))

        output_path = tmp_path / "output.glb"

        # Without all deps, generate should fail gracefully
        with pytest.raises(RuntimeError):
            backend.generate(str(input_path), str(output_path))

    def test_repo_constants(self):
        """Test that repo constants are properly defined."""
        from assetforge.backends.dreamgaussian import (
            DREAMGAUSSIAN_REPO,
            DREAMGAUSSIAN_DIR,
        )

        assert "dreamgaussian" in DREAMGAUSSIAN_REPO
        assert "dreamgaussian" in str(DREAMGAUSSIAN_DIR)
        assert DREAMGAUSSIAN_DIR.is_absolute()

    def test_backend_str(self):
        """Test string representation."""
        from assetforge.backends.dreamgaussian import DreamGaussianBackend

        backend = DreamGaussianBackend(device="cpu")
        assert "DreamGaussian" in backend.__class__.__name__


class TestDreamGaussianPipelineIntegration:
    """Tests DreamGaussian integration with the pipeline."""

    def test_pipeline_model_selection(self, test_image):
        """Test that pipeline supports dreamgaussian model."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_dreamgaussian",
            model_name="dreamgaussian",
        )
        assert pipeline.model_name == "dreamgaussian"

    def test_pipeline_resolve_chain_contains_dreamgaussian(self, test_image):
        """Test that model resolution chain includes dreamgaussian."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_dg_chain",
            model_name="auto",
        )
        chain = pipeline._resolve_model_chain()
        model_names = [m[0] for m in chain]
        assert "dreamgaussian" in model_names
        # dreamgaussian should come after hunyuan3d-2 and before triposr
        assert model_names.index("hunyuan3d-2") < model_names.index("dreamgaussian")
        assert model_names.index("dreamgaussian") < model_names.index("triposr")

    def test_pipeline_resolve_chain_dreamgaussian_fallback(self, test_image):
        """Test that dreamgaussian falls back to triposr."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_dg_fallback",
            model_name="dreamgaussian",
        )
        chain = pipeline._resolve_model_chain()
        model_names = [m[0] for m in chain]
        assert model_names[0] == "dreamgaussian"
        assert "triposr" in model_names  # Fallback

    def test_model_registry_has_dreamgaussian(self):
        """Test that model registry has DreamGaussian entry."""
        from assetforge.models.registry import MODEL_REGISTRY

        assert "dreamgaussian" in MODEL_REGISTRY
        info = MODEL_REGISTRY["dreamgaussian"]
        assert info.pipeline_type == "3d_gen"
        assert info.min_vram_gb == 8.0


@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image."""
    import numpy as np
    from PIL import Image

    arr = np.ones((128, 128, 4), dtype=np.uint8) * 255
    arr[30:90, 30:90] = [100, 150, 200, 255]
    img_path = tmp_path / "test_concept.png"
    Image.fromarray(arr, "RGBA").save(str(img_path))
    return img_path
