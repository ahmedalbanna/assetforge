"""
3D Generative AI Pipeline — Generates 3D models from a single image using AI.

Stages:
1. AI 3D generation (TripoSR / Hunyuan3D-2 / TRELLIS)
2. Mesh cleanup (Blender headless)
3. Retopology (Blender headless) [optional]
4. UV unwrap + texture baking
5. Export to game format
"""

import logging
from pathlib import Path

from assetforge.pipelines.base import BasePipeline
from assetforge.utils.gpu import detect_gpu, select_3d_model

logger = logging.getLogger("assetforge")


class Pipeline3DGen(BasePipeline):
    """Generate a 3D model from a single image using generative AI."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "./output",
        model_name: str = "auto",
        retopology: bool = False,
        target_faces: int = 20000,
        generate_pbr: bool = False,
        texture_resolution: int = 2048,
        output_format: str = "glb",
    ):
        super().__init__(output_dir, name="3d-generative")
        self.input_path = Path(input_path).resolve()
        self.model_name = model_name
        self.retopology = retopology
        self.target_faces = target_faces
        self.generate_pbr = generate_pbr
        self.texture_resolution = texture_resolution
        self.output_format = output_format

        self.stages_dir = self.output_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute the full generative 3D pipeline."""
        logger.info(f"=== 3D Generative AI Pipeline ===")
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_dir}")

        if not self.input_path.exists():
            logger.error(f"Input file not found: {self.input_path}")
            return self.stages

        # Auto-select model if needed
        if self.model_name == "auto":
            gpu = detect_gpu()
            if gpu["available"]:
                self.model_name = select_3d_model(gpu["vram_total_gb"])
            else:
                self.model_name = "triposr"  # Lightest model
            logger.info(f"Auto-selected model: {self.model_name}")

        # Stage 1: AI 3D generation
        self.run_stage("ai_generation", self._generate_3d)

        # Stage 2: Mesh cleanup
        self.run_stage("mesh_cleanup", self._cleanup_mesh)

        # Stage 3: Retopology (optional)
        if self.retopology:
            self.run_stage("retopology", self._apply_retopology)

        # Stage 4: Texture baking (optional)
        if self.generate_pbr:
            self.run_stage("texture_baking", self._bake_textures)

        # Stage 5: Export
        self.run_stage("export", self._export_model)

        self.summary()
        return self.stages

    def _generate_3d(self) -> list[Path]:
        """Run the selected AI model to generate 3D mesh from image."""
        if self.model_name == "triposr":
            from assetforge.backends.triposr import TripoSRBackend

            backend = TripoSRBackend()
        elif self.model_name == "hunyuan3d-2":
            from assetforge.backends.hunyuan3d import Hunyuan3DBackend

            backend = Hunyuan3DBackend()
        elif self.model_name == "trellis":
            from assetforge.backends.trellis import TRELLISBackend

            backend = TRELLISBackend()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        output_path = self.stages_dir / f"raw_{self.model_name}.glb"
        backend.generate(
            str(self.input_path),
            str(output_path),
            pbr=self.generate_pbr,
            tex_resolution=self.texture_resolution,
        )
        self._raw_mesh_path = Path(output_path)
        return [output_path]

    def _cleanup_mesh(self) -> list[Path]:
        """Remove floaters, close holes, smooth the raw mesh."""
        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        output_path = self.stages_dir / "mesh_cleaned.obj"
        blender.run_cleanup(str(self._raw_mesh_path), str(output_path))
        self._cleaned_mesh_path = Path(output_path)
        return [output_path]

    def _apply_retopology(self) -> list[Path]:
        """Decimate and remesh for game-optimized topology."""
        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()

        ratio = min(1.0, self.target_faces / 50000.0)
        decimated = self.stages_dir / "mesh_decimated.obj"
        blender.apply_decimate(
            str(self._cleaned_mesh_path), str(decimated), ratio=ratio
        )

        quads = self.stages_dir / "mesh_quads.obj"
        blender.apply_remesh(str(decimated), str(quads), target_faces=self.target_faces)
        self._final_mesh_path = quads
        return [decimated, quads]

    def _bake_textures(self) -> list[Path]:
        """Bake PBR texture maps from the generated mesh."""
        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        mesh = getattr(self, "_final_mesh_path", self._cleaned_mesh_path)
        tex_dir = self.stages_dir / "textures"

        blender.bake_textures(
            str(mesh),
            str(self._raw_mesh_path),
            str(tex_dir),
            resolution=self.texture_resolution,
        )
        return list(tex_dir.glob("*"))

    def _export_model(self) -> list[Path]:
        """Export the final model in game-ready format."""
        mesh = getattr(self, "_final_mesh_path", self._cleaned_mesh_path)
        output_ext = f".{self.output_format}"
        output_path = self.output_dir / f"model{output_ext}"

        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        blender.export_format(str(mesh), str(output_path))
        return [output_path]
