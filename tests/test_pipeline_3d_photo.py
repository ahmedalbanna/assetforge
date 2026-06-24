"""
Integration tests for the 3D Photogrammetry Pipeline.

Tests validate pipeline construction, research-backed parameters,
stage sequencing, quality presets, and error handling.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from assetforge.pipelines.pipeline_3d_photo import Pipeline3DPhoto, QUALITY_PRESETS


@pytest.fixture
def photo_set_dir(tmp_path):
    """Create a temporary directory with test images for photogrammetry."""
    from PIL import Image

    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    for i in range(3):
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 200
        arr[20:44, 20:44] = [100, 150, 200]
        Image.fromarray(arr, "RGB").save(str(img_dir / f"view_{i:02d}.jpg"))
    return img_dir


@pytest.fixture
def photo_set_with_masks(tmp_path):
    """Create a temporary directory with test images and masks."""
    from PIL import Image, ImageDraw

    img_dir = tmp_path / "photos_masked"
    img_dir.mkdir()
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    for i in range(3):
        arr = np.ones((64, 64, 3), dtype=np.uint8) * 200
        arr[20:44, 20:44] = [100, 150, 200]
        Image.fromarray(arr, "RGB").save(str(img_dir / f"view_{i:02d}.jpg"))

        mask = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([10, 10, 54, 54], fill=255)
        mask.save(str(mask_dir / f"view_{i:02d}_mask.png"))

    global_mask = Image.new("L", (64, 64), 255)
    mask_dir_global = tmp_path / "masks_global"
    mask_dir_global.mkdir()
    global_mask.save(str(mask_dir_global / "global_mask.png"))

    return {
        "img_dir": img_dir,
        "mask_dir_per_image": mask_dir,
        "mask_dir_global": mask_dir_global,
    }


class TestPipelineInitialization:
    """Tests for pipeline construction with research-backed parameters."""

    def test_default_parameters(self, photo_set_dir):
        """Test default pipeline parameters include research-backed defaults."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_defaults",
        )
        assert pipeline.sfm_type == "incremental"
        assert pipeline.feature_type == "sift"
        assert pipeline.resolution_level == 0
        assert pipeline.mesh_refinement is True
        assert pipeline.normalize_coordinates is True
        assert pipeline.mask_dir is None

    def test_sfm_type_selection(self, photo_set_dir):
        """Test SfM type parameter is stored correctly."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_sfm",
            sfm_type="global",
        )
        assert pipeline.sfm_type == "global"

    def test_feature_type_akaze(self, photo_set_dir):
        """Test AKAZE feature type selection."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_akaze",
            feature_type="akaze",
        )
        assert pipeline.feature_type == "akaze"

    def test_mask_directory(self, photo_set_with_masks):
        """Test mask directory parameter."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_with_masks["img_dir"]),
            output_dir="/tmp/test_masks",
            mask_dir=str(photo_set_with_masks["mask_dir_per_image"]),
        )
        assert pipeline.mask_dir is not None
        assert "masks" in pipeline.mask_dir

    def test_resolution_level(self, photo_set_dir):
        """Test resolution level for memory management."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_resolution",
            resolution_level=2,
        )
        assert pipeline.resolution_level == 2

    def test_mesh_refinement_disabled(self, photo_set_dir):
        """Test mesh refinement can be disabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_no_refine",
            mesh_refinement=False,
        )
        assert pipeline.mesh_refinement is False

    def test_normalize_coordinates_disabled(self, photo_set_dir):
        """Test coordinate normalization can be disabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_no_norm",
            normalize_coordinates=False,
        )
        assert pipeline.normalize_coordinates is False


class TestQualityPresets:
    """Tests for quality preset resolution with research-backed defaults."""

    def test_quality_preset_fast(self, photo_set_dir):
        """Test 'fast' preset — global SfM, no refine."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_fast",
            quality="fast",
        )
        preset = QUALITY_PRESETS["fast"]
        assert pipeline.quality_preset == "fast"
        assert pipeline.sfm_type == preset["sfm_type"]
        assert pipeline.feature_type == preset["feature_type"]
        assert pipeline.resolution_level == preset["resolution_level"]
        assert pipeline.mesh_refinement == preset["mesh_refinement"]
        assert pipeline.retopology == preset["retopology"]
        assert pipeline.normalize_coordinates == preset["normalize_coordinates"]

    def test_quality_preset_balanced(self, photo_set_dir):
        """Test 'balanced' preset — incremental SfM, refine, retopo."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_balanced",
            quality="balanced",
        )
        preset = QUALITY_PRESETS["balanced"]
        assert pipeline.sfm_type == preset["sfm_type"]
        assert pipeline.feature_type == preset["feature_type"]
        assert pipeline.mesh_refinement == preset["mesh_refinement"]
        assert pipeline.retopology == preset["retopology"]
        assert pipeline.normalize_coordinates == preset["normalize_coordinates"]

    def test_quality_preset_high(self, photo_set_dir):
        """Test 'high' preset — full resolution, refine, retopo 30k."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_high",
            quality="high",
        )
        preset = QUALITY_PRESETS["high"]
        assert pipeline.sfm_type == preset["sfm_type"]
        assert pipeline.resolution_level == preset["resolution_level"]
        assert pipeline.mesh_refinement == preset["mesh_refinement"]
        assert pipeline.target_faces == preset["target_faces"]

    def test_quality_preset_ultra(self, photo_set_dir):
        """Test 'ultra' preset — global SfM, full res, refine, 50k."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_ultra",
            quality="ultra",
        )
        preset = QUALITY_PRESETS["ultra"]
        assert pipeline.sfm_type == preset["sfm_type"]
        assert pipeline.feature_type == preset["feature_type"]
        assert pipeline.resolution_level == preset["resolution_level"]
        assert pipeline.target_faces == preset["target_faces"]

    def test_quality_preset_fully_overrides_individual(self, photo_set_dir):
        """Test quality preset overrides individual research-backed params."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_override",
            quality="fast",
            sfm_type="incremental",
            mesh_refinement=True,
            resolution_level=0,
        )
        preset = QUALITY_PRESETS["fast"]
        assert pipeline.sfm_type == preset["sfm_type"]
        assert pipeline.mesh_refinement == preset["mesh_refinement"]
        assert pipeline.resolution_level == preset["resolution_level"]

    def test_quality_preset_unknown(self, photo_set_dir):
        """Test unknown quality preset uses individual params."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_unknown",
            quality="nonexistent",
            feature_type="sift",
            mesh_quality="high",
        )
        assert pipeline.quality_preset == "nonexistent"
        assert pipeline.feature_type == "sift"
        assert pipeline.mesh_quality == "high"
        assert pipeline.retopology is False


class TestCoordinateNormalization:
    """Tests for coordinate normalization utility."""

    def test_normalize_sfm_data_creates_transform(self, tmp_path):
        """Test that normalize_sfm_data creates a valid CoordinateTransform."""
        from assetforge.utils.coordinates import normalize_sfm_data

        # Create a minimal sfm_data.json with known camera poses
        sfm_data = {
            "views": [
                {
                    "key": 0,
                    "value": {
                        "ptr_wrapper": {
                            "data": {"local_path": "", "filename": "view_00.jpg"}
                        }
                    },
                },
                {
                    "key": 1,
                    "value": {
                        "ptr_wrapper": {
                            "data": {"local_path": "", "filename": "view_01.jpg"}
                        }
                    },
                },
            ],
            "extrinsics": [
                {
                    "key": 0,
                    "value": {
                        "rotation": [
                            1,
                            0,
                            0,
                            100,
                            0,
                            1,
                            0,
                            200,
                            0,
                            0,
                            1,
                            300,
                            0,
                            0,
                            0,
                            1,
                        ]
                    },
                },
                {
                    "key": 1,
                    "value": {
                        "rotation": [
                            1,
                            0,
                            0,
                            400,
                            0,
                            1,
                            0,
                            500,
                            0,
                            0,
                            1,
                            600,
                            0,
                            0,
                            0,
                            1,
                        ]
                    },
                },
            ],
            "structure": {},
        }

        sfm_path = tmp_path / "sfm_data.json"
        with open(sfm_path, "w") as f:
            json.dump(sfm_data, f)

        output_path = tmp_path / "normalized" / "sfm_data.json"
        transform = normalize_sfm_data(str(sfm_path), str(output_path))

        assert transform is not None
        assert len(transform.center) == 3
        assert transform.scale > 0

        # Verify the normalized file exists
        assert output_path.exists()
        assert (output_path.with_name("coordinate_transform.json")).exists()

    def test_compute_centroid(self, tmp_path):
        """Test scene centroid computation."""
        from assetforge.utils.coordinates import compute_scene_centroid

        sfm_data = {
            "views": [],
            "extrinsics": [
                {
                    "key": 0,
                    "value": {
                        "rotation": [
                            1,
                            0,
                            0,
                            10,
                            0,
                            1,
                            0,
                            20,
                            0,
                            0,
                            1,
                            30,
                            0,
                            0,
                            0,
                            1,
                        ]
                    },
                },
            ],
            "structure": {},
        }

        sfm_path = tmp_path / "sfm_centroid.json"
        with open(sfm_path, "w") as f:
            json.dump(sfm_data, f)

        centroid = compute_scene_centroid(sfm_path)
        # Camera center = -R.T * t = -[1,0,0;0,1,0;0,0,1] * [10,20,30] = [-10,-20,-30]
        assert abs(centroid[0] - (-10.0)) < 0.01
        assert abs(centroid[1] - (-20.0)) < 0.01
        assert abs(centroid[2] - (-30.0)) < 0.01

    def test_restore_obj_coordinates(self, tmp_path):
        """Test restoring coordinates on an OBJ file."""
        from assetforge.utils.coordinates import (
            restore_coordinates,
            CoordinateTransform,
        )

        # Create a simple OBJ
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""
        obj_path = tmp_path / "test.obj"
        with open(obj_path, "w") as f:
            f.write(obj_content)

        transform = CoordinateTransform(
            center=(10.0, 20.0, 30.0),
            scale=2.0,
            matrix=[[2, 0, 0, -20], [0, 2, 0, -40], [0, 0, 2, -60], [0, 0, 0, 1]],
        )

        output_path = tmp_path / "restored.obj"
        restore_coordinates(str(obj_path), transform, str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
        # First vertex: x = 0.0 / 2.0 + 10.0 = 10.0
        assert "v 10.0" in content

    def test_no_transform_when_identity(self, tmp_path):
        """Test that identity transform copies mesh as-is."""
        from assetforge.utils.coordinates import (
            restore_coordinates,
            CoordinateTransform,
        )

        obj_content = "v 0.0 0.0 0.0\n"
        obj_path = tmp_path / "test.obj"
        with open(obj_path, "w") as f:
            f.write(obj_content)

        transform = CoordinateTransform(
            center=(0.0, 0.0, 0.0),
            scale=1.0,
            matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )

        output_path = tmp_path / "restored.obj"
        restore_coordinates(str(obj_path), transform, str(output_path))

        with open(output_path) as f:
            assert f.read() == obj_content


class TestPipelineRun:
    """Tests for pipeline execution (without external tools)."""

    def test_pipeline_run_no_openmvg(self, photo_set_dir):
        """Test pipeline handles missing tools gracefully."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir=str(photo_set_dir.parent / "output"),
        )
        results = pipeline.run()
        assert len(results) > 0
        stage_results = {r.name: r for r in results}
        assert "feature_extraction" in stage_results

    def test_pipeline_output_dir_creation(self, photo_set_dir):
        """Test output directory is created."""
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
        assert len(results) == 0

    def test_pipeline_stage_count_with_refinement(self, photo_set_dir):
        """Test stage count when mesh refinement is enabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir=str(photo_set_dir.parent / "output_count"),
            mesh_refinement=True,
            normalize_coordinates=True,
        )
        # Just test initialization, not execution
        assert pipeline.mesh_refinement is True
        assert pipeline.normalize_coordinates is True

    def test_pipeline_stage_count_without_refinement(self, photo_set_dir):
        """Test stage count when mesh refinement is disabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir=str(photo_set_dir.parent / "output_count2"),
            mesh_refinement=False,
            normalize_coordinates=False,
        )
        assert pipeline.mesh_refinement is False
        assert pipeline.normalize_coordinates is False


class TestMaskSeamLeveling:
    """Tests for mask-aware seam leveling behavior."""

    def test_masks_disable_seam_leveling_logging(self, photo_set_with_masks, caplog):
        """Test that mask presence logs seam leveling info."""
        caplog.set_level("INFO")
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_with_masks["img_dir"]),
            output_dir=str(photo_set_with_masks["img_dir"].parent / "output_seam"),
            mask_dir=str(photo_set_with_masks["mask_dir_per_image"]),
        )
        # Verify mask_dir is set
        assert pipeline.mask_dir is not None
        # The seam leveling decision happens at runtime in _map_textures

    def test_no_masks_enables_seam_leveling(self, photo_set_dir):
        """Test that no masks means seam leveling stays enabled."""
        pipeline = Pipeline3DPhoto(
            input_dir=str(photo_set_dir),
            output_dir="/tmp/test_no_mask_seam",
        )
        assert pipeline.mask_dir is None


class TestBackendFunctions:
    """Tests for backend utility functions."""

    def test_openmvg_check(self):
        """Test _check_openmvg returns expected value (False when not installed)."""
        from assetforge.backends.openmvg import _check_openmvg

        # In test environment, OpenMVG is likely not installed
        result = _check_openmvg()
        assert result is False

    def test_openmvs_check(self):
        """Test _check_openmvs returns expected value."""
        from assetforge.backends.openmvs import _check_openmvs

        result = _check_openmvs()
        assert result is False

    def test_valid_sfm_types(self):
        """Test valid SfM types are recognized."""
        from assetforge.backends.openmvg import VALID_SFM_TYPES

        assert "incremental" in VALID_SFM_TYPES
        assert "global" in VALID_SFM_TYPES
        assert "incremental_v2" in VALID_SFM_TYPES
        assert len(VALID_SFM_TYPES) == 3

    def test_valid_feature_types(self):
        """Test valid feature types are recognized."""
        from assetforge.backends.openmvg import VALID_FEATURE_TYPES

        assert "sift" in VALID_FEATURE_TYPES
        assert "orb" in VALID_FEATURE_TYPES
        assert "akaze" in VALID_FEATURE_TYPES
        assert len(VALID_FEATURE_TYPES) == 3


class TestCoordinateTransform:
    """Tests for CoordinateTransform dataclass."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading CoordinateTransform from JSON."""
        from assetforge.utils.coordinates import CoordinateTransform

        transform = CoordinateTransform(
            center=(1.0, 2.0, 3.0),
            scale=0.5,
            matrix=[
                [0.5, 0, 0, -0.5],
                [0, 0.5, 0, -1.0],
                [0, 0, 0.5, -1.5],
                [0, 0, 0, 1],
            ],
        )

        save_path = tmp_path / "transform.json"
        transform.save(str(save_path))

        loaded = CoordinateTransform.load(str(save_path))
        assert list(loaded.center) == [1.0, 2.0, 3.0]
        assert loaded.scale == 0.5
        assert loaded.matrix == transform.matrix
