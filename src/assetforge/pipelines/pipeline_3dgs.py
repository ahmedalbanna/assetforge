"""
3D Gaussian Splatting Pipeline — Train and export Gaussian splats.

Stages:
1. Camera pose estimation (COLMAP / OpenMVG)
2. 3DGS training
3. Mesh extraction via SuGaR/MILo [optional]
4. SPZ compression [optional]
5. GLTF export with KHR_gaussian_splatting extension

Quality Presets:
    - fast:     Lowest splat count, no mesh, no compression, quickest
    - balanced: Medium splats, MILo mesh, SPZ compress, good quality
    - high:     High splats, SuGaR mesh, SPZ compress, detailed
    - ultra:    Max splats, SuGaR mesh, SPZ compress, finest detail
"""

import logging
from pathlib import Path

from assetforge.pipelines.base import (
    BasePipeline,
    validate_input_path,
)

logger = logging.getLogger("assetforge")

# ── Quality Presets ──────────────────────────────────────────────────────────
# Each preset controls training quality, splat count, mesh extraction,
# and compression for the 3DGS pipeline.

QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "train_quality": "low",
        "max_splats": 100_000,
        "extract_mesh": False,
        "mesh_method": "milo",
        "compress": False,
    },
    "balanced": {
        "train_quality": "medium",
        "max_splats": 500_000,
        "extract_mesh": True,
        "mesh_method": "milo",
        "compress": True,
    },
    "high": {
        "train_quality": "high",
        "max_splats": 1_000_000,
        "extract_mesh": True,
        "mesh_method": "sugar",
        "compress": True,
    },
    "ultra": {
        "train_quality": "high",
        "max_splats": 2_000_000,
        "extract_mesh": True,
        "mesh_method": "sugar",
        "compress": True,
    },
}

DEFAULT_PRESET = "balanced"

VALID_PRESETS = set(QUALITY_PRESETS.keys())


class Pipeline3DGS(BasePipeline):
    """Train 3D Gaussian Splatting from photos or video."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "./output",
        # Quality preset (overrides individual params)
        quality: str = "",
        # Individual configuration (used when quality is empty)
        train_quality: str = "medium",
        extract_mesh: bool = False,
        mesh_method: str = "milo",
        compress: bool = False,
        max_splats: int = 1_000_000,
    ):
        super().__init__(output_dir, name="3d-gaussian-splatting")
        self.input_path = validate_input_path(input_path, check_exists=False)

        # Resolve quality preset if specified
        if quality and quality in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[quality]
            self.quality_preset = quality
            self.train_quality = preset.get("train_quality", train_quality)
            self.max_splats = preset.get("max_splats", max_splats)
            self.extract_mesh = preset.get("extract_mesh", extract_mesh)
            self.mesh_method = preset.get("mesh_method", mesh_method)
            self.compress = preset.get("compress", compress)
            logger.info(
                f"Using quality preset: '{quality}' "
                f"(train: {self.train_quality}, splats: {self.max_splats})"
            )
        else:
            if quality:
                logger.warning(
                    "Unknown quality preset '%s', falling back to individual params",
                    quality,
                )
            self.quality_preset = quality
            self.train_quality = train_quality
            self.max_splats = max_splats
            self.extract_mesh = extract_mesh
            self.mesh_method = mesh_method
            self.compress = compress

        self.stages_dir = self.output_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the Gaussian splatting pipeline."""
        logger.info(f"=== 3D Gaussian Splatting Pipeline ===")
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_dir}")

        if not self.input_path.exists():
            logger.error(f"Input not found: {self.input_path}")
            return self.stages

        # Stage 1: Camera pose estimation
        self.run_stage("camera_poses", self._estimate_poses)

        # Stage 2: Train 3D Gaussians
        self.run_stage("train_splats", self._train_splats)

        # Stage 3: Mesh extraction (optional)
        if self.extract_mesh:
            self.run_stage("mesh_extraction", self._extract_mesh)

        # Stage 4: Compression (optional)
        if self.compress:
            self.run_stage("compression", self._compress_splats)

        # Stage 5: GLTF export
        self.run_stage("export", self._export_gltf)

        self.summary()
        return self.stages

    def _estimate_poses(self) -> list[Path]:
        """Estimate camera poses from input images using COLMAP or OpenMVG."""
        # Check if input is a video file — extract frames if so
        from assetforge.utils.io import detect_input_type

        input_type = detect_input_type(str(self.input_path))

        if input_type == "video":
            # Extract frames from video
            import cv2

            video_path = self.input_path
            frames_dir = self.stages_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(frames_dir / f"frame_{frame_count:04d}.jpg"), frame)
                frame_count += 1
            cap.release()
            logger.info(f"Extracted {frame_count} frames from video")
            image_dir = frames_dir
        else:
            image_dir = self.input_path

        # Run COLMAP for pose estimation
        from assetforge.backends.gaussian_splat import estimate_poses

        output_path = self.stages_dir / "poses"
        estimate_poses(str(image_dir), str(output_path))
        self._poses_dir = Path(output_path)
        return [output_path]

    def _train_splats(self) -> list[Path]:
        """Train 3D Gaussian representation from posed images."""
        self.require_attr("_poses_dir", "camera_poses")
        from assetforge.backends.gaussian_splat import train_splats

        output_path = self.stages_dir / "splat.ply"
        train_splats(
            str(self._poses_dir),
            str(output_path),
            quality=self.train_quality,
            max_splats=self.max_splats,
        )
        self._splat_path = Path(output_path)
        return [output_path]

    def _extract_mesh(self) -> list[Path]:
        """Extract surface mesh from trained Gaussians using SuGaR or MILo."""
        self.require_attr("_splat_path", "train_splats")
        if self.mesh_method == "milo":
            from assetforge.backends.gaussian_splat import extract_mesh_milo

            output_path = self.stages_dir / "mesh_milo.obj"
            extract_mesh_milo(str(self._splat_path), str(output_path))
        elif self.mesh_method == "sugar":
            from assetforge.backends.gaussian_splat import extract_mesh_sugar

            output_path = self.stages_dir / "mesh_sugar.obj"
            extract_mesh_sugar(str(self._splat_path), str(output_path))
        else:
            raise ValueError(f"Unknown mesh method: {self.mesh_method}")

        self._mesh_path = Path(output_path)
        return [output_path]

    def _compress_splats(self) -> list[Path]:
        """Compress Gaussian splats to SPZ format using Rust backend."""
        self.require_attr("_splat_path", "train_splats")

        try:
            from assetforge_pyo3 import compress_splats
        except ImportError:
            logger.warning(
                "Rust PyO3 module not available. Using fallback compression."
            )
            # Fallback: simple zip compression
            import zipfile

            output_path = self.stages_dir / "splat.spz"
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(str(self._splat_path), "splat.ply")
            return [output_path]

        # Parse PLY binary (simplified — in production use a proper PLY parser)
        with open(self._splat_path, "rb") as f:
            ply_data = f.read()

        compressed = compress_splats(ply_data, quality=80)
        output_path = self.stages_dir / "splat.spz"
        with open(output_path, "wb") as f:
            f.write(compressed)

        self._spz_path = Path(output_path)
        return [output_path]

    def _export_gltf(self) -> list[Path]:
        """Export Gaussian splats as GLTF with KHR_gaussian_splatting extension."""
        splat_path = getattr(self, "_spz_path", None) or getattr(
            self, "_splat_path", None
        )
        if splat_path is None:
            raise RuntimeError(
                "No splat data to export — training stage likely failed. "
                "Check 'train_splats' stage for details."
            )

        try:
            from assetforge_pyo3 import export_splat_gltf
        except ImportError:
            logger.warning("Rust PyO3 module not available. Copying PLY as output.")
            import shutil

            output_path = self.output_dir / "splat.ply"
            shutil.copy2(str(splat_path), str(output_path))
            return [output_path]

        output_path = self.output_dir / "splat.glb"
        export_splat_gltf(str(splat_path), str(output_path))
        return [output_path]
