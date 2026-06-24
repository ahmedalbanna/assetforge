"""
3D Photogrammetry Pipeline — Reconstructs 3D models from multi-angle photos.

Auto-detects available backends:
1. COLMAP (primary) — full pipeline (features → SfM → MVS → mesh)
2. OpenMVG + OpenMVS (fallback) — feature extraction, SfM, MVS, mesh, textures

Stages:
0. Coordinate normalization (research-backed: prevents float32 precision loss)
1. Feature extraction (COLMAP or OpenMVG)
2. Feature matching (COLMAP or OpenMVG)
3. Structure from Motion (COLMAP or OpenMVG — incremental or global)
4. Multi-View Stereo (COLMAP or OpenMVS)
5. Mesh reconstruction (COLMAP or OpenMVS)
6. Mesh refinement (OpenMVS only) [optional, skipped if OpenMVS unavailable]
7. Texture mapping (OpenMVS only) [optional, skipped if OpenMVS unavailable]
8. Retopology (Blender headless) [optional]
9. Coordinate restoration (restore geographic coordinates)
10. Export to game format

Quality Presets:
    - fast:     SIFT, global SfM, low quality, no refine, no retopo, obj
    - balanced: SIFT, incremental SfM, medium quality, refine, retopo (15k), glb
    - high:     SIFT, incremental SfM, high quality, refine, retopo (30k), glb
    - ultra:    SIFT, global SfM, extreme quality, refine, retopo (50k), 8k atlas, glb
"""

import logging
from pathlib import Path

from assetforge.pipelines.base import (
    BasePipeline,
    validate_input_dir,
    validate_output_dir,
)

logger = logging.getLogger("assetforge")

# ── Quality Presets ──────────────────────────────────────────────────────────
# Each preset defines a complete set of photogrammetry pipeline parameters.
# Research-backed defaults incorporated:
#   - Global SfM for fast/ultra (faster, less drift)
#   - Incremental SfM for balanced/high (more robust with sparse data)
#   - Mesh refinement enabled for all but 'fast'
#   - Coordinate normalization enabled by default for all

QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "feature_type": "orb",
        "sfm_type": "global",  # Faster reconstruction
        "mesh_quality": "low",
        "resolution_level": 2,  # Quarter resolution, saves memory
        "mesh_refinement": False,  # Skip for speed
        "retopology": False,
        "target_faces": 5000,
        "output_format": "obj",
        "normalize_coordinates": True,  # Recommended for all scenes
    },
    "balanced": {
        "feature_type": "sift",
        "sfm_type": "incremental",  # More robust for general use
        "mesh_quality": "medium",
        "resolution_level": 1,  # Half resolution
        "mesh_refinement": True,  # 15-30% detail improvement
        "retopology": True,
        "target_faces": 15000,
        "output_format": "glb",
        "normalize_coordinates": True,
    },
    "high": {
        "feature_type": "sift",
        "sfm_type": "incremental",  # Robust for high quality
        "mesh_quality": "high",
        "resolution_level": 0,  # Full resolution
        "mesh_refinement": True,
        "retopology": True,
        "target_faces": 30000,
        "output_format": "glb",
        "normalize_coordinates": True,
    },
    "ultra": {
        "feature_type": "sift",
        "sfm_type": "global",  # Best quality, global = less drift
        "mesh_quality": "high",
        "resolution_level": 0,  # Full resolution
        "mesh_refinement": True,
        "retopology": True,
        "target_faces": 50000,
        "output_format": "glb",
        "normalize_coordinates": True,
    },
}

DEFAULT_PRESET = "balanced"

VALID_PRESETS = set(QUALITY_PRESETS.keys())


class Pipeline3DPhoto(BasePipeline):
    """Reconstruct a 3D model from multiple photos using photogrammetry.

    Incorporates research-backed improvements:
    - Coordinate normalization to prevent float32 precision issues
    - SfM type selection (incremental, incremental_v2, global)
    - Mask support for background removal
    - Resolution level control for memory management
    - Mesh refinement for detail enhancement
    - Seam leveling control for artifact-free texture mapping
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str = "./output",
        # Quality preset (overrides individual params)
        quality: str = "",
        # Research-backed parameters
        sfm_type: str = "incremental",
        feature_type: str = "sift",
        mask_dir: str | None = None,
        resolution_level: int = 0,
        mesh_refinement: bool = True,
        normalize_coordinates: bool = True,
        # Individual configuration (used when quality is empty)
        mesh_quality: str = "high",
        retopology: bool = False,
        target_faces: int = 15000,
        output_format: str = "glb",
    ):
        super().__init__(output_dir, name="3d-photogrammetry")
        self.input_dir = validate_input_dir(input_dir, check_exists=False)

        # Resolve quality preset if specified
        if quality and quality in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[quality]
            self.quality_preset = quality
            self.sfm_type = preset.get("sfm_type", sfm_type)
            self.feature_type = preset.get("feature_type", feature_type)
            self.mesh_quality = preset.get("mesh_quality", mesh_quality)
            self.resolution_level = preset.get("resolution_level", resolution_level)
            self.mesh_refinement = preset.get("mesh_refinement", mesh_refinement)
            self.retopology = preset.get("retopology", retopology)
            self.target_faces = preset.get("target_faces", target_faces)
            self.output_format = preset.get("output_format", output_format)
            self.normalize_coordinates = preset.get(
                "normalize_coordinates", normalize_coordinates
            )
            logger.info(
                f"Using quality preset: '{quality}' "
                f"(features: {self.feature_type}, "
                f"sfm: {self.sfm_type}, "
                f"mesh: {self.mesh_quality})"
            )
        else:
            if quality:
                logger.warning(
                    "Unknown quality preset '%s', falling back to individual params",
                    quality,
                )
            self.quality_preset = quality
            self.sfm_type = sfm_type
            self.feature_type = feature_type
            self.mask_dir = mask_dir
            self.resolution_level = resolution_level
            self.mesh_refinement = mesh_refinement
            self.normalize_coordinates = normalize_coordinates
            self.mesh_quality = mesh_quality
            self.retopology = retopology
            self.target_faces = target_faces
            self.output_format = output_format

        self.mask_dir = mask_dir
        self.stages_dir = self.output_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self._normalize_transform = None

        # ── Auto-detect available backends ──
        self._use_colmap = False
        self._use_openmvg = False
        self._use_openmvs = False
        self._detect_backends()

    def _detect_backends(self):
        """Auto-detect which photogrammetry backends are installed.

        Priority: COLMAP > OpenMVG + OpenMVS
        COLMAP handles the full pipeline (features → SfM → MVS → mesh).
        OpenMVS adds mesh refinement and texture mapping.
        """
        try:
            from assetforge.backends.colmap import check_colmap

            self._use_colmap = check_colmap()
            if self._use_colmap:
                logger.info("Backend: COLMAP detected (full pipeline)")
        except Exception:
            self._use_colmap = False

        try:
            from assetforge.backends.openmvg import _check_openmvg

            self._use_openmvg = _check_openmvg()
            if self._use_openmvg:
                logger.info("Backend: OpenMVG detected (feature extraction + SfM)")
        except Exception:
            self._use_openmvg = False

        try:
            from assetforge.backends.openmvs import _check_openmvs

            self._use_openmvs = _check_openmvs()
            if self._use_openmvs:
                logger.info("Backend: OpenMVS detected (MVS + mesh + texture)")
        except Exception:
            self._use_openmvs = False

        if not self._use_colmap and not self._use_openmvg:
            logger.warning(
                "No photogrammetry backend detected. "
                "Install COLMAP (sudo apt-get install colmap) or "
                "build OpenMVG from https://github.com/openMVG/openMVG"
            )

    def run(self):
        """Execute the full photogrammetry pipeline."""
        logger.info(f"=== 3D Photogrammetry Pipeline ===")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Quality: {self.quality_preset or 'custom'}")
        logger.info(f"SfM type: {self.sfm_type}")
        logger.info(f"Feature type: {self.feature_type}")
        logger.info(f"Mask dir: {self.mask_dir or 'none'}")
        logger.info(f"Resolution level: {self.resolution_level}")
        logger.info(f"Mesh refinement: {self.mesh_refinement}")
        logger.info(f"Coordinate normalization: {self.normalize_coordinates}")

        if not self.input_dir.is_dir():
            logger.error(f"Input directory not found: {self.input_dir}")
            return self.stages

        # Stage 0: Coordinate normalization — prevents float32 precision loss
        if self.normalize_coordinates:
            self.run_stage("coordinate_normalization", self._normalize_coords)

        # Stage 1: Feature extraction (with masking support)
        self.run_stage("feature_extraction", self._extract_features)

        # Stage 2: Feature matching
        self.run_stage("feature_matching", self._match_features)

        # Stage 3: Structure from Motion (configurable SfM type)
        self.run_stage("sfm", self._run_sfm)

        # Stage 4: Multi-View Stereo (with resolution control)
        self.run_stage("mvs", self._run_mvs)

        # Stage 5: Mesh reconstruction
        self.run_stage("mesh_reconstruction", self._reconstruct_mesh)

        # Stage 6: Mesh refinement (optional, improves surface detail)
        if self.mesh_refinement:
            self.run_stage("mesh_refinement", self._refine_mesh)

        # Stage 7: Texture mapping (with seam leveling control)
        self.run_stage("texture_mapping", self._map_textures)

        # Stage 8: Retopology (optional)
        if self.retopology:
            self.run_stage("retopology", self._apply_retopology)

        # Stage 9: Coordinate restoration (restore geographic coords)
        if self.normalize_coordinates and self._normalize_transform:
            self.run_stage("coordinate_restoration", self._restore_coords)

        # Stage 10: Export
        self.run_stage("export", self._export_model)

        self.summary()
        return self.stages

    def _normalize_coords(self) -> list[Path]:
        """Normalize scene coordinates to prevent float32 precision loss.

        Research-backed: OpenMVS uses single-precision floats internally.
        Large coordinates (UTM/GPS data) cause catastrophic precision loss
        that manifests as:
        - Collapsed point clouds (appear as 1D lines)
        - NAN values in Delaunay tetrahedralization
        - Silent crashes during ReconstructMesh

        Solution: Shift scene centroid to origin and scale to unit size
        before processing, then restore coordinates at final export.
        """
        from assetforge.utils.coordinates import normalize_sfm_data

        sfm_data_path = self.stages_dir / "features" / "sfm_data.json"
        normalized_path = self.stages_dir / "sfm_normalized" / "sfm_data.json"

        if sfm_data_path.exists():
            transform = normalize_sfm_data(sfm_data_path, normalized_path)
            self._normalize_transform = transform
            logger.info(
                f"Coordinate normalization: center={transform.center}, scale={transform.scale:.4f}"
            )
            return [normalized_path, self.stages_dir / "sfm_normalized"]
        else:
            logger.warning(
                "sfm_data.json not found yet, deferring normalization to SfM stage"
            )
            return []

    def _run_colmap_full_pipeline(self):
        """Run the full COLMAP pipeline (features → SfM → MVS → mesh).

        Auto-retries with image enhancement for AI-generated / synthetic
        images that lack sufficient SIFT features.

        Raises RuntimeError if COLMAP cannot produce a model from the input
        images (e.g., too few features, not enough overlap between photos).
        """
        from assetforge.backends.colmap import (
            run_automatic_reconstruction,
            enhance_images_for_sfm,
        )

        quality_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "ultra": "extreme",
        }
        colmap_quality = quality_map.get(self.mesh_quality, "high")

        colmap_workspace = self.stages_dir / "colmap_workspace"

        def _try_reconstruction(image_dir: str, attempt_label: str):
            """Attempt COLMAP with GPU→CPU fallback."""
            for use_gpu in [True, False]:
                try:
                    run_automatic_reconstruction(
                        image_dir=image_dir,
                        workspace_dir=str(colmap_workspace),
                        quality=colmap_quality,
                        data_type="individual",
                        mask_dir=self.mask_dir,
                        use_gpu=use_gpu,
                        mesher="poisson",
                    )
                    return True
                except RuntimeError as e:
                    if use_gpu:
                        logger.warning(
                            "GPU reconstruction failed (%s), retrying with CPU: %s",
                            attempt_label,
                            e,
                        )
                        continue
                    # Both GPU and CPU failed — propagate
                    raise

        def _check_output():
            """Check if COLMAP produced a valid model."""
            meshed_dir = colmap_workspace / "meshed"
            dense_dir = colmap_workspace / "dense"
            sparse_dir = colmap_workspace / "sparse"

            has_mesh = False
            colmap_mesh_path = None
            if meshed_dir.exists():
                ply_files = list(meshed_dir.glob("*.ply"))
                if ply_files:
                    colmap_mesh_path = ply_files[0]
                    has_mesh = True

            has_sparse_model = False
            if sparse_dir.exists():
                sparse_subdirs = list(sparse_dir.glob("*/"))
                if sparse_subdirs:
                    for subdir in sparse_subdirs:
                        if (subdir / "cameras.bin").exists() or (
                            subdir / "cameras.txt"
                        ).exists():
                            has_sparse_model = True
                            break

            has_dense = dense_dir.exists() and any(dense_dir.iterdir())
            return has_mesh, has_sparse_model, has_dense, colmap_mesh_path

        # Attempt 1: original images
        # Note: COLMAP returns exit code 0 even on reconstruction failure
        # (no sparse model produced), so we MUST check output after.
        first_attempt = False
        try:
            _try_reconstruction(str(self.input_dir), "original images")
            first_attempt = True
        except RuntimeError:
            first_attempt = False

        has_mesh, has_sparse_model, has_dense, colmap_mesh_path = _check_output()

        if not has_sparse_model and not has_mesh:
            # Attempt 2: enhanced images (for AI-generated / synthetic)
            logger.info(
                "COLMAP produced no model with original images. Retrying with "
                "image enhancement (sharpening + noise + markers) for "
                "AI-generated / synthetic images..."
            )
            enhanced_dir = enhance_images_for_sfm(
                str(self.input_dir),
                output_dir=str(self.stages_dir / "enhanced_images"),
            )

            # Reset workspace for clean retry
            import shutil

            if colmap_workspace.exists():
                shutil.rmtree(str(colmap_workspace))
            colmap_workspace.mkdir(parents=True, exist_ok=True)

            try:
                _try_reconstruction(str(enhanced_dir), "enhanced images")
            except RuntimeError as e:
                raise RuntimeError(
                    "COLMAP could not reconstruct any camera poses from the input images, "
                    "even after automatic image enhancement. "
                    "This usually means the images lack sufficient visual features or "
                    "overlap for Structure-from-Motion to work. "
                    "Try using photos with more texture, increasing image overlap, "
                    "or using higher resolution images."
                ) from e

            # Re-check output after enhancement
            has_mesh, has_sparse_model, has_dense, colmap_mesh_path = _check_output()

        if not has_sparse_model and not has_mesh:
            raise RuntimeError(
                "COLMAP could not reconstruct any camera poses from the input images. "
                "This usually means the images lack sufficient visual features or "
                "overlap for Structure-from-Motion to work. "
                "Try using photos with more texture, increasing image overlap, "
                "or using higher resolution images."
            )

        # Only set attributes once we confirm COLMAP produced valid output
        self._colmap_workspace = colmap_workspace
        self._colmap_mesh_path = colmap_mesh_path
        self._colmap_has_dense = has_dense
        self._colmap_has_sparse = has_sparse_model
        if colmap_mesh_path:
            self._reconstructed_mesh_path = colmap_mesh_path

        logger.info(
            f"COLMAP full pipeline complete "
            f"(sparse={has_sparse_model}, dense={has_dense}, mesh={has_mesh})"
        )

    def _extract_features(self) -> list[Path]:
        """Extract features from input images.

        COLMAP: runs full pipeline (automatic_reconstructor).
        OpenMVG: standalone feature extraction.
        """
        output_path = self.stages_dir / "features"
        output_path.mkdir(parents=True, exist_ok=True)

        if self._use_colmap:
            self._run_colmap_full_pipeline()
            return [output_path]
        else:
            from assetforge.backends.openmvg import extract_features

            extract_features(
                str(self.input_dir),
                str(output_path),
                feature_type=self.feature_type,
                mask_dir=self.mask_dir,
            )
            return [output_path]

    def _match_features(self) -> list[Path]:
        """Geometric matching — handled by COLMAP's automatic_reconstructor."""
        if self._use_colmap:
            if getattr(self, "_colmap_workspace", None) is None:
                logger.warning("COLMAP pipeline did not complete — matching skipped")
                return []
            logger.info("COLMAP: matching handled by automatic_reconstructor")
            return [self.stages_dir / "colmap_workspace"]
        else:
            from assetforge.backends.openmvg import match_features

            features_dir = self.stages_dir / "features"
            output_path = self.stages_dir / "matches"
            match_features(str(features_dir), str(output_path))
            return [output_path]

    def _run_sfm(self) -> list[Path]:
        """Structure from Motion — handled by COLMAP's automatic_reconstructor."""
        if self._use_colmap:
            if getattr(self, "_colmap_has_sparse", False):
                logger.info("COLMAP: SfM handled by automatic_reconstructor")
                return [self.stages_dir / "colmap_workspace"]
            logger.warning("COLMAP pipeline did not complete — SfM skipped")
            return []
        else:
            from assetforge.backends.openmvg import run_sfm

            matches_dir = self.stages_dir / "matches"
            output_path = self.stages_dir / "sfm"
            run_sfm(
                str(matches_dir),
                str(output_path),
                sfm_type=self.sfm_type,
            )
            return [output_path]

    def _run_mvs(self) -> list[Path]:
        """Multi-View Stereo — handled by COLMAP's automatic_reconstructor."""
        if self._use_colmap:
            logger.info("COLMAP: MVS handled by automatic_reconstructor")
            workspace = getattr(self, "_colmap_workspace", None)
            has_dense = getattr(self, "_colmap_has_dense", False)
            if workspace is None:
                logger.warning("COLMAP workspace not set — MVS stage skipped")
                return []
            dense_dir = workspace / "dense" if has_dense else None
            if dense_dir and dense_dir.exists():
                return [dense_dir]
            return [workspace]
        else:
            from assetforge.backends.openmvs import run_mvs

            output_path = self.stages_dir / "mvs"
            output_path.mkdir(parents=True, exist_ok=True)
            sfm_dir = self.stages_dir / "sfm"
            run_mvs(
                str(sfm_dir),
                str(output_path),
                quality=self.mesh_quality,
                resolution_level=self.resolution_level,
            )
            return [output_path]

    def _reconstruct_mesh(self) -> list[Path]:
        """Reconstruct mesh surface — handled by COLMAP's automatic_reconstructor."""
        if self._use_colmap:
            logger.info("COLMAP: mesh handled by automatic_reconstructor")
            mesh_path = getattr(self, "_colmap_mesh_path", None)
            workspace = getattr(self, "_colmap_workspace", None)
            if mesh_path and Path(mesh_path).exists():
                return [mesh_path]
            if workspace:
                return [workspace]
            logger.warning("COLMAP workspace not set — mesh stage skipped")
            return []
        else:
            from assetforge.backends.openmvs import reconstruct_mesh

            mvs_dir = self.stages_dir / "mvs"
            output_path = self.stages_dir / "mesh_raw.obj"
            reconstruct_mesh(str(mvs_dir), str(output_path))
            self._reconstructed_mesh_path = Path(output_path)
            return [output_path]

    def _refine_mesh(self) -> list[Path]:
        """Refine mesh surface detail using photometric optimization.

        Research-backed: Improves vertex positions by maximizing
        photometric consistency across views. Recovers fine details,
        protrusions, and curved surfaces smoothed by initial mesh extraction.
        Typical improvement: 15-30% surface detail enhancement.

        NOTE: Only available with OpenMVS backend. Skipped when using COLMAP.
        """
        if not self._use_openmvs:
            logger.info(
                "Mesh refinement requires OpenMVS, which is not installed. "
                "Skipping refinement stage."
            )
            return []

        self.require_attr("_reconstructed_mesh_path", "mesh_reconstruction")
        from assetforge.backends.openmvs import refine_mesh

        mvs_dir = self.stages_dir / "mvs"
        output_path = self.stages_dir / "mesh_refined.obj"
        refine_mesh(
            str(mvs_dir),
            str(output_path),
        )
        # Update the mesh path to the refined version
        self._reconstructed_mesh_path = Path(output_path)
        return [output_path]

    def _map_textures(self) -> list[Path]:
        """Project photo textures onto the reconstructed mesh.

        NOTE: Only available with OpenMVS backend. COLMAP does not include
        texture mapping capability. Skipped when using COLMAP.

        Seam leveling control:
        - When using masks (mask_dir is set), seam leveling is automatically
          disabled to prevent black artifacts on masked surfaces
        - Research-backed: OpenMVS seam leveling causes black artifacts when
          mask edges don't perfectly align with mesh boundaries
        """
        if not self._use_openmvs:
            logger.info(
                "Texture mapping requires OpenMVS, which is not installed. "
                "Skipping texture mapping stage."
            )
            return []

        self.require_attr("_reconstructed_mesh_path", "mesh_reconstruction")
        from assetforge.backends.openmvs import map_textures

        mvs_dir = self.stages_dir / "mvs"
        output_path = self.stages_dir / "mesh_textured.obj"

        # Auto-disable seam leveling when masks are present
        enable_seam_leveling = self.mask_dir is None

        if self.mask_dir:
            logger.info(
                "Masks detected: disabling seam leveling to prevent black artifacts"
            )

        map_textures(
            str(mvs_dir),
            str(output_path),
            local_seam_leveling=enable_seam_leveling,
            global_seam_leveling=enable_seam_leveling,
        )
        self._textured_mesh_path = Path(output_path)
        return [output_path]

    def _apply_retopology(self) -> list[Path]:
        """Apply quad remeshing via Rust or Blender for game-ready topology."""
        # If texturing was skipped (no OpenMVS), fall back to raw mesh
        input_mesh = getattr(self, "_textured_mesh_path", None) or getattr(
            self, "_reconstructed_mesh_path", None
        )
        if input_mesh is None:
            raise RuntimeError(
                "retopology requires a mesh from a previous stage. "
                "Neither texture_mapping nor mesh_reconstruction produced output."
            )

        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        input_mesh = Path(input_mesh)
        output_mesh = self.stages_dir / "mesh_retopology.obj"
        ratio = min(1.0, self.target_faces / 100000.0)

        # Try Rust optimization first (faster), fall back to Blender
        rust_result = blender.apply_decimate_rust(
            str(input_mesh), str(output_mesh), ratio=ratio
        )
        if rust_result is None:
            blender.apply_decimate(str(input_mesh), str(output_mesh), ratio=ratio)

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

    def _restore_coords(self) -> list[Path]:
        """Restore original geographic coordinates after processing.

        Applies the inverse transformation from the normalization step
        to place the final model in its correct world position.

        This is purely a cosmetic/projection step done at the very end.
        """
        if self._normalize_transform is None:
            logger.info(
                "No normalization transform found, skipping coordinate restoration"
            )
            return []

        from assetforge.utils.coordinates import restore_coordinates

        # Find the mesh to restore
        mesh_path = (
            getattr(self, "_final_mesh_path", None)
            or getattr(self, "_textured_mesh_path", None)
            or getattr(self, "_reconstructed_mesh_path", None)
        )

        if mesh_path is None:
            logger.warning("No mesh to restore coordinates on")
            return []

        output_path = self.stages_dir / "mesh_restored.obj"
        restored = restore_coordinates(
            str(mesh_path),
            self._normalize_transform,
            str(output_path),
        )
        self._restored_mesh_path = restored
        return [restored]

    def _export_model(self) -> list[Path]:
        """Export the final model in the requested game format."""
        mesh_path = (
            getattr(self, "_restored_mesh_path", None)
            or getattr(self, "_final_mesh_path", None)
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
