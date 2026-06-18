"""
3D Photogrammetry Pipeline — Reconstructs 3D models from multi-angle photos.

Stages:
1. Feature extraction (OpenMVG)
2. Feature matching (OpenMVG)
3. Structure from Motion (OpenMVG)
4. Multi-View Stereo (OpenMVS)
5. Mesh reconstruction (OpenMVS)
6. Texture mapping (OpenMVS)
7. Retopology (Blender headless) [optional]
8. Export to game format
"""

import logging
from pathlib import Path

from assetforge.pipelines.base import BasePipeline

logger = logging.getLogger("assetforge")


class Pipeline3DPhoto(BasePipeline):
    """Reconstruct a 3D model from multiple photos using photogrammetry."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str = "./output",
        feature_type: str = "sift",
        mesh_quality: str = "high",
        retopology: bool = False,
        target_faces: int = 15000,
        output_format: str = "glb",
    ):
        super().__init__(output_dir, name="3d-photogrammetry")
        self.input_dir = Path(input_dir).resolve()
        self.feature_type = feature_type
        self.mesh_quality = mesh_quality
        self.retopology = retopology
        self.target_faces = target_faces
        self.output_format = output_format

        self.stages_dir = self.output_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the full photogrammetry pipeline."""
        logger.info(f"=== 3D Photogrammetry Pipeline ===")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")

        if not self.input_dir.is_dir():
            logger.error(f"Input directory not found: {self.input_dir}")
            return self.stages

        # Stage 1: Feature extraction
        self.run_stage("feature_extraction", self._extract_features)

        # Stage 2: Feature matching
        self.run_stage("feature_matching", self._match_features)

        # Stage 3: Structure from Motion
        self.run_stage("sfm", self._run_sfm)

        # Stage 4: Multi-View Stereo
        self.run_stage("mvs", self._run_mvs)

        # Stage 5: Mesh reconstruction
        self.run_stage("mesh_reconstruction", self._reconstruct_mesh)

        # Stage 6: Texture mapping
        self.run_stage("texture_mapping", self._map_textures)

        # Stage 7: Retopology (optional)
        if self.retopology:
            self.run_stage("retopology", self._apply_retopology)

        # Stage 8: Export
        self.run_stage("export", self._export_model)

        self.summary()
        return self.stages

    def _extract_features(self) -> list[Path]:
        """Extract SIFT/ORB features from all input images using OpenMVG."""
        from assetforge.backends.openmvg import extract_features

        output_path = self.stages_dir / "features"
        extract_features(
            str(self.input_dir),
            str(output_path),
            feature_type=self.feature_type,
        )
        return [output_path]

    def _match_features(self) -> list[Path]:
        """Geometric matching between image pairs."""
        from assetforge.backends.openmvg import match_features

        features_dir = self.stages_dir / "features"
        output_path = self.stages_dir / "matches"
        match_features(str(features_dir), str(output_path))
        return [output_path]

    def _run_sfm(self) -> list[Path]:
        """Structure from Motion — estimate camera poses and sparse cloud."""
        from assetforge.backends.openmvg import run_sfm

        matches_dir = self.stages_dir / "matches"
        output_path = self.stages_dir / "sfm"
        run_sfm(str(matches_dir), str(output_path))
        return [output_path]

    def _run_mvs(self) -> list[Path]:
        """Multi-View Stereo — dense point cloud reconstruction."""
        from assetforge.backends.openmvs import run_mvs

        sfm_dir = self.stages_dir / "sfm"
        output_path = self.stages_dir / "mvs"
        run_mvs(str(sfm_dir), str(output_path), quality=self.mesh_quality)
        return [output_path]

    def _reconstruct_mesh(self) -> list[Path]:
        """Reconstruct mesh surface from dense point cloud."""
        from assetforge.backends.openmvs import reconstruct_mesh

        mvs_dir = self.stages_dir / "mvs"
        output_path = self.stages_dir / "mesh_raw.obj"
        reconstruct_mesh(str(mvs_dir), str(output_path))
        self._reconstructed_mesh_path = Path(output_path)
        return [output_path]

    def _map_textures(self) -> list[Path]:
        """Project photo textures onto the reconstructed mesh."""
        self.require_attr("_reconstructed_mesh_path", "mesh_reconstruction")
        from assetforge.backends.openmvs import map_textures

        output_path = self.stages_dir / "mesh_textured.obj"
        map_textures(
            str(self._reconstructed_mesh_path),
            str(self.stages_dir / "mvs"),
            str(output_path),
        )
        self._textured_mesh_path = Path(output_path)
        return [output_path]

    def _apply_retopology(self) -> list[Path]:
        """Apply quad remeshing via Blender headless for game-ready topology."""
        self.require_attr("_textured_mesh_path", "texture_mapping")
        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        input_mesh = self._textured_mesh_path
        output_mesh = self.stages_dir / "mesh_retopology.obj"

        blender.apply_decimate(
            str(input_mesh),
            str(output_mesh),
            ratio=min(1.0, self.target_faces / 100000.0),
        )
        self._retopologized_path = output_mesh

        # Apply quad remeshing
        quads_mesh = self.stages_dir / "mesh_quads.obj"
        blender.apply_remesh(
            str(output_mesh),
            str(quads_mesh),
            target_faces=self.target_faces,
        )
        self._final_mesh_path = quads_mesh
        return [output_mesh, quads_mesh]

    def _export_model(self) -> list[Path]:
        """Export the final model in the requested game format."""
        mesh_path = (
            getattr(self, "_final_mesh_path", None)
            or getattr(self, "_textured_mesh_path", None)
            or getattr(self, "_reconstructed_mesh_path", None)
        )
        if mesh_path is None:
            raise RuntimeError(
                "No mesh to export — all reconstruction stages failed. "
                "Check previous stages for details."
            )

        output_ext = f".{self.output_format}"
        output_path = self.output_dir / f"model{output_ext}"

        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        blender.export_format(str(mesh_path), str(output_path))
        return [output_path]
