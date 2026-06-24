"""
Tests for TripoSR NeRF feature extraction and texture baking capabilities.

Validates:
- NeRF feature saving format
- Feature loading and validation
- Integration with PipelineBake for editing workflow
"""

import pickle
import pytest
from pathlib import Path


@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image."""
    import numpy as np
    from PIL import Image

    arr = np.ones((128, 128, 3), dtype=np.uint8) * 200
    arr[30:90, 30:90] = [100, 150, 200]
    img_path = tmp_path / "test_input.png"
    Image.fromarray(arr, "RGB").save(str(img_path))
    return img_path


class TestTripoSRBackend:
    """Tests for TripoSR backend."""

    def test_import(self):
        """Test that the module can be imported."""
        from assetforge.backends.triposr import TripoSRBackend

        assert TripoSRBackend is not None

    def test_init(self):
        """Test basic initialization."""
        from assetforge.backends.triposr import TripoSRBackend

        backend = TripoSRBackend(device="cpu")
        assert backend.device == "cpu"
        assert backend._repo_ready is False

    def test_nerf_feature_keys(self):
        """Test that _save_nerf_features produces expected dictionary keys."""
        from assetforge.backends.triposr import TripoSRBackend
        import numpy as np

        backend = TripoSRBackend(device="cpu")

        # Create mock model, scene_codes, mesh
        class MockMesh:
            def __init__(self):
                self.vertices = np.array(
                    [
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [0.5, 0.5, -0.5],
                        [-0.5, 0.5, -0.5],
                    ],
                    dtype=np.float32,
                )
                self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
                self.vertex_normals = np.array(
                    [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=np.float32
                )

        class MockModel:
            def __init__(self):
                pass

        class MockSceneCode:
            def cpu(self):
                return "scene_code_stub"

        mesh = MockMesh()
        model = MockModel()
        scene_codes = [MockSceneCode()]

        # Create mock original image
        from PIL import Image
        import numpy as np

        orig_img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)

        # Test saving
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pkl_path = f.name

        try:
            result = backend._save_nerf_features(
                model=model,
                scene_codes=scene_codes,
                mesh=mesh,
                original_image=orig_img,
                output_path=pkl_path,
                method="triplane",
            )

            # Verify the file was created
            assert Path(pkl_path).exists()

            # Load and verify contents
            with open(pkl_path, "rb") as f:
                features = pickle.load(f)

            assert features["method"] == "triplane"
            assert "vertices" in features
            assert "faces" in features
            assert "vertex_colors" in features or features["vertex_colors"] is None
            assert "camera_radius" in features
            assert "fov" in features
            assert features["camera_radius"] == 2.0
            assert features["fov"] == 49.0
            assert "original_image_bytes" in features
            assert result == Path(pkl_path)

        finally:
            Path(pkl_path).unlink(missing_ok=True)

    def test_nerf_features_load_with_pipeline_bake(self, tmp_path):
        """Test that NeRF features can be loaded by PipelineBake."""
        import numpy as np

        # Create mock NeRF features
        features = {
            "method": "triplane",
            "vertices": np.random.rand(100, 3).astype(np.float32),
            "faces": np.random.randint(0, 100, (50, 3)).astype(np.int32),
            "vertex_colors": np.random.rand(100, 3).astype(np.float32),
            "camera_radius": 2.0,
            "fov": 49.0,
            "num_views": 8,
        }

        pkl_path = tmp_path / "test_features.pkl"
        with open(str(pkl_path), "wb") as f:
            pickle.dump(features, f)

        # Create a simple mesh that PipelineBake can load
        import trimesh

        mesh = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        mesh_path = tmp_path / "test_mesh.obj"
        mesh.export(str(mesh_path))

        # Test that PipelineBake accepts it
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(mesh_path),
            output_dir=str(tmp_path / "bake_output"),
            nerf_features_path=str(pkl_path),
            use_blender=False,
            resolution=64,
        )
        assert pipeline.nerf_features_path == pkl_path
        # Mesh is loaded during run(), not at init
        assert not hasattr(pipeline, "_mesh")

    def test_generate_has_save_nerf_param(self):
        """Test that generate() accepts save_nerf_features parameter."""
        from assetforge.backends.triposr import TripoSRBackend
        import inspect

        sig = inspect.signature(TripoSRBackend.generate)
        assert "save_nerf_features" in sig.parameters
        param = sig.parameters["save_nerf_features"]
        assert param.default is True

    def test_nerf_features_invalid_path(self):
        """Test that invalid NeRF features path raises FileNotFoundError."""
        from assetforge.backends.triposr import TripoSRBackend

        backend = TripoSRBackend(device="cpu")

        with pytest.raises(FileNotFoundError):
            from assetforge.pipelines.pipeline_bake import PipelineBake
            import trimesh
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".obj") as f:
                mesh = trimesh.creation.box()
                mesh.export(f.name)
                PipelineBake(
                    mesh_path=f.name,
                    output_dir="/tmp/test_bad_nerf",
                    nerf_features_path="/nonexistent/features.pkl",
                )

    def test_triposr_deps_defined(self):
        """Test that TripoSR dependencies are defined."""
        from assetforge.backends.triposr import TripoSRBackend

        assert len(TripoSRBackend.TRIPOSR_DEPS) > 0
        assert "omegaconf" in " ".join(TripoSRBackend.TRIPOSR_DEPS)
        assert "xatlas" in " ".join(TripoSRBackend.TRIPOSR_DEPS)


class TestTripoSRPipelineIntegration:
    """Tests TripoSR integration with the 3D gen pipeline."""

    def test_pipeline_supports_triposr(self, test_image):
        """Test that pipeline works with triposr model."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_triposr_pipe",
            model_name="triposr",
        )
        assert pipeline.model_name == "triposr"

    def test_pipeline_resolve_chain_starts_with_triposr(self, test_image):
        """Test that triposr chain only includes triposr (no fallback up)."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_tr_chain",
            model_name="triposr",
        )
        chain = pipeline._resolve_model_chain()
        model_names = [m[0] for m in chain]
        assert model_names[0] == "triposr"
        assert len(model_names) == 1  # No fallback after triposr

    def test_pipeline_resolve_chain_auto_includes_triposr(self, test_image):
        """Test that auto chain ends with triposr as final fallback."""
        from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

        pipeline = Pipeline3DGen(
            input_path=str(test_image),
            output_dir="/tmp/test_auto_chain",
            model_name="auto",
        )
        chain = pipeline._resolve_model_chain()
        model_names = [m[0] for m in chain]
        assert "triposr" in model_names
        assert model_names[-1] == "triposr"  # Last resort
