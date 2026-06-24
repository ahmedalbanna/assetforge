"""
Tests for the Texture Baking Pipeline.

Validates initialization, parameter handling, and error cases
for both NeRF-based and image-projection baking.
"""

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


@pytest.fixture
def test_mesh(tmp_path):
    """Create a simple test mesh (a cube)."""
    import trimesh
    import numpy as np

    # Create a simple cube
    mesh = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    mesh_path = tmp_path / "test_mesh.obj"
    mesh.export(str(mesh_path))
    return mesh_path


@pytest.fixture
def test_nerf_features(tmp_path):
    """Create a mock NeRF features pickle file."""
    import pickle
    import numpy as np

    features = {
        "method": "triplane",
        "vertices": np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        "faces": np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 5, 6],
                [1, 6, 2],
                [5, 4, 7],
                [5, 7, 6],
                [4, 0, 3],
                [4, 3, 7],
                [3, 2, 6],
                [3, 6, 7],
                [4, 5, 1],
                [4, 1, 0],
            ],
            dtype=np.int32,
        ),
        "vertex_colors": np.random.rand(8, 3).astype(np.float32),
        "camera_radius": 2.0,
        "fov": 49.0,
        "num_views": 8,
    }
    pkl_path = tmp_path / "nerf_features.pkl"
    with open(str(pkl_path), "wb") as f:
        pickle.dump(features, f)
    return pkl_path


class TestPipelineBake:
    """Tests for the texture baking pipeline."""

    def test_import(self):
        """Test that the pipeline module can be imported."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        assert PipelineBake is not None

    def test_init_image_projection(self, test_mesh, test_image):
        """Test initialization with image projection."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir="/tmp/test_bake_img",
            texture_image_path=str(test_image),
        )
        assert pipeline.mesh_path == test_mesh
        assert pipeline.texture_image_path == test_image
        assert pipeline.nerf_features_path is None

    def test_init_nerf_baking(self, test_mesh, test_nerf_features):
        """Test initialization with NeRF feature baking."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir="/tmp/test_bake_nerf",
            nerf_features_path=str(test_nerf_features),
        )
        assert pipeline.nerf_features_path == test_nerf_features
        assert pipeline.texture_image_path is None

    def test_init_no_source_raises(self, test_mesh):
        """Test that missing source raises ValueError."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        with pytest.raises(ValueError, match="nerf_features_path.*texture_image_path"):
            PipelineBake(
                mesh_path=str(test_mesh),
                output_dir="/tmp/test_bake_none",
            )

    def test_init_with_quality_preset(self, test_mesh, test_image):
        """Test that quality preset overrides resolution."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir="/tmp/test_bake_quality",
            texture_image_path=str(test_image),
            quality="ultra",
            resolution=512,  # Should be overridden to 8192
        )
        assert pipeline.resolution == 8192
        assert pipeline.uv_padding == 16

    def test_init_resolution_clamping(self, test_mesh, test_image):
        """Test that resolution is properly clamped."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        with pytest.raises(ValueError, match="resolution"):
            PipelineBake(
                mesh_path=str(test_mesh),
                output_dir="/tmp/test_bake_clamp",
                texture_image_path=str(test_image),
                resolution=99999,  # Out of range
            )

    def test_init_uv_padding_clamping(self, test_mesh, test_image):
        """Test that UV padding is properly clamped."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        with pytest.raises(ValueError, match="uv_padding"):
            PipelineBake(
                mesh_path=str(test_mesh),
                output_dir="/tmp/test_bake_pad",
                texture_image_path=str(test_image),
                uv_padding=999,  # Out of range
            )

    def test_init_invalid_mesh_raises(self):
        """Test that invalid mesh path raises FileNotFoundError."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        with pytest.raises(FileNotFoundError):
            PipelineBake(
                mesh_path="/nonexistent/mesh.obj",
                output_dir="/tmp/test_bake_invalid",
                texture_image_path="/nonexistent/img.png",
            )

    def test_quality_presets(self):
        """Test that all quality presets are properly defined."""
        from assetforge.pipelines.pipeline_bake import QUALITY_PRESETS

        assert "fast" in QUALITY_PRESETS
        assert "balanced" in QUALITY_PRESETS
        assert "high" in QUALITY_PRESETS
        assert "ultra" in QUALITY_PRESETS

        # Check resolution progression
        assert (
            QUALITY_PRESETS["fast"]["resolution"]
            < QUALITY_PRESETS["balanced"]["resolution"]
        )
        assert (
            QUALITY_PRESETS["balanced"]["resolution"]
            < QUALITY_PRESETS["high"]["resolution"]
        )
        assert (
            QUALITY_PRESETS["high"]["resolution"]
            <= QUALITY_PRESETS["ultra"]["resolution"]
        )

    def test_pipeline_run_image_projection(self, test_mesh, test_image):
        """Test that pipeline runs with image projection."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir=str(test_image.parent / "bake_output"),
            texture_image_path=str(test_image),
            use_blender=False,  # Skip Blender dependency
            resolution=64,
        )
        results = pipeline.run()
        assert len(results) > 0

        # Each stage should have executed
        stage_names = [r.name for r in results]
        assert "load_mesh" in stage_names
        assert "uv_unwrap" in stage_names
        assert "bake_texture_image_projection" in stage_names
        assert "export" in stage_names

    def test_pipeline_runs_with_nerf(self, test_mesh, test_nerf_features):
        """Test that pipeline runs with NeRF features."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir=str(test_nerf_features.parent / "nerf_bake"),
            nerf_features_path=str(test_nerf_features),
            use_blender=False,
            resolution=64,
        )
        results = pipeline.run()
        assert len(results) > 0

        stage_names = [r.name for r in results]
        assert "load_mesh" in stage_names
        assert "bake_texture_nerf" in stage_names

    def test_output_directory_created(self, test_mesh, test_image):
        """Test that output directory is created."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        output_dir = test_image.parent / "bake_out"
        pipeline = PipelineBake(
            mesh_path=str(test_mesh),
            output_dir=str(output_dir),
            texture_image_path=str(test_image),
            use_blender=False,
            resolution=64,
        )
        pipeline.run()
        assert output_dir.exists()
        assert (output_dir / "stages").exists()

    def test_load_mesh_invalid_file(self, test_nerf_features):
        """Test that loading an invalid mesh fails."""
        from assetforge.pipelines.pipeline_bake import PipelineBake

        with pytest.raises((FileNotFoundError, RuntimeError)):
            pipeline = PipelineBake(
                mesh_path="/nonexistent/mesh.obj",
                output_dir="/tmp/test_bad_mesh",
                nerf_features_path=str(test_nerf_features),
            )
            pipeline.run()
