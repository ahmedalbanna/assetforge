"""
3D Generative AI Pipeline — Generates 3D models from a single image using AI.

Stages:
0. Input preprocessing (smart crop, background removal options)
1. AI 3D generation (TripoSR / Hunyuan3D-2 / TRELLIS)
2. Mesh cleanup (Blender headless)
3. Retopology (Blender headless) [optional]
4. UV unwrap + texture baking
5. Export to game format
"""

import logging
from pathlib import Path

from assetforge.pipelines.base import (
    BasePipeline,
    validate_input_image,
    validate_output_dir,
    clamp,
)
from assetforge.utils.gpu import detect_gpu, select_3d_model

logger = logging.getLogger("assetforge")

# Quality presets: presets determine model + defaults for all quality knobs
QUALITY_PRESETS = {
    "fast": {
        "model": "triposr",
        "mesh_resolution": 128,
        "inference_steps": 25,
        "guidance_scale": 3.0,
        "foreground_ratio": 0.9,
        "chunk_size": 65536,
        "texture_resolution": 1024,
        "retopology": False,
        "generate_pbr": False,
        "target_faces": 10000,
        "bake_from_original": False,
    },
    "balanced": {
        "model": "auto",  # picks best available (auto GPU-based)
        "mesh_resolution": 256,
        "inference_steps": 50,
        "guidance_scale": 5.0,
        "foreground_ratio": 0.85,
        "chunk_size": 131072,
        "texture_resolution": 2048,
        "retopology": True,
        "generate_pbr": True,
        "target_faces": 15000,
        "bake_from_original": True,
    },
    "high": {
        "model": "hunyuan3d-2",
        "mesh_resolution": 512,
        "inference_steps": 50,
        "guidance_scale": 7.5,
        "foreground_ratio": 0.82,
        "chunk_size": 262144,
        "texture_resolution": 4096,
        "retopology": True,
        "generate_pbr": True,
        "target_faces": 30000,
        "bake_from_original": True,
    },
    "ultra": {
        "model": "trellis",
        "mesh_resolution": 512,
        "inference_steps": 75,
        "guidance_scale": 10.0,
        "foreground_ratio": 0.8,
        "chunk_size": 524288,
        "texture_resolution": 8192,
        "retopology": True,
        "generate_pbr": True,
        "target_faces": 50000,
        "bake_from_original": True,
    },
}

# Default preset used when no preset specified
DEFAULT_PRESET = "balanced"


class Pipeline3DGen(BasePipeline):
    """Generate a 3D model from a single image using generative AI."""

    def __init__(
        self,
        input_path: str,
        output_dir: str = "./output",
        # Quality preset (overrides individual params)
        quality: str = "",
        # Model selection
        model_name: str = "auto",
        # Geometric fidelity
        mesh_resolution: int = 256,
        foreground_ratio: float = 0.85,
        chunk_size: int = 131072,
        # Sampling quality (TRELLIS/Hunyuan)
        inference_steps: int = 50,
        guidance_scale: float = 5.0,
        # Texture quality
        texture_resolution: int = 2048,
        generate_pbr: bool = False,
        bake_from_original: bool = True,
        # Retopology
        retopology: bool = False,
        target_faces: int = 20000,
        # Output format
        output_format: str = "glb",
    ):
        super().__init__(output_dir, name="3d-generative")
        # Validate inputs (extension check only, existence checked at runtime)
        self.input_path = validate_input_image(input_path, check_exists=False)
        self.output_dir = validate_output_dir(output_dir)

        # Validate and clamp numeric parameters
        self.mesh_resolution = int(clamp(mesh_resolution, 64, 1024, "mesh_resolution"))
        self.foreground_ratio = clamp(foreground_ratio, 0.5, 1.0, "foreground_ratio")
        self.chunk_size = int(clamp(chunk_size, 4096, 1_048_576, "chunk_size"))
        self.inference_steps = int(clamp(inference_steps, 1, 200, "inference_steps"))
        self.guidance_scale = clamp(guidance_scale, 1.0, 30.0, "guidance_scale")
        self.texture_resolution = int(
            clamp(texture_resolution, 256, 16384, "texture_resolution")
        )
        self.target_faces = int(clamp(target_faces, 100, 5_000_000, "target_faces"))

        # Model name is validated at runtime in _resolve_model_chain()
        # Output format is validated at runtime in the export stage

        # Resolve quality preset if specified
        if quality and quality in QUALITY_PRESETS:
            preset = QUALITY_PRESETS[quality]
            self.model_name = preset.get("model", model_name)
            self.mesh_resolution = preset.get("mesh_resolution", mesh_resolution)
            self.inference_steps = preset.get("inference_steps", inference_steps)
            self.guidance_scale = preset.get("guidance_scale", guidance_scale)
            self.foreground_ratio = preset.get("foreground_ratio", foreground_ratio)
            self.chunk_size = preset.get("chunk_size", chunk_size)
            self.texture_resolution = preset.get(
                "texture_resolution", texture_resolution
            )
            self.generate_pbr = preset.get("generate_pbr", generate_pbr)
            self.bake_from_original = preset.get(
                "bake_from_original", bake_from_original
            )
            self.retopology = preset.get("retopology", retopology)
            self.target_faces = preset.get("target_faces", target_faces)
            self.quality_preset = quality
            logger.info(f"Using quality preset: '{quality}' (model: {self.model_name})")
        else:
            self.model_name = model_name
            self.mesh_resolution = mesh_resolution
            self.inference_steps = inference_steps
            self.guidance_scale = guidance_scale
            self.foreground_ratio = foreground_ratio
            self.chunk_size = chunk_size
            self.texture_resolution = texture_resolution
            self.generate_pbr = generate_pbr
            self.bake_from_original = bake_from_original
            self.retopology = retopology
            self.target_faces = target_faces
            self.quality_preset = quality

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

        # Stage 0: Input preprocessing (optional)
        self.run_stage("preprocess", self._preprocess_input)

        # Stage 1: AI 3D generation
        self.run_stage("ai_generation", self._generate_3d)

        # Stage 2: Mesh cleanup
        self.run_stage("mesh_cleanup", self._cleanup_mesh)

        # Stage 3: Retopology (optional)
        if self.retopology:
            self.run_stage("retopology", self._apply_retopology)

        # Stage 4: Texture baking (optional)
        if self.generate_pbr or self.bake_from_original:
            self.run_stage("texture_baking", self._bake_textures)

        # Stage 5: Export
        self.run_stage("export", self._export_model)

        self.summary()
        return self.stages

    def _preprocess_input(self) -> list[Path]:
        """Preprocess the input image for better 3D generation results."""
        import numpy as np
        from PIL import Image, ImageOps

        image = Image.open(self.input_path).convert("RGB")
        w, h = image.size

        # Smart crop: detect rough object bounds and center-crop
        try:
            from rembg import remove

            # Quick background removal to find object bounds
            rgba = remove(image)
            alpha = np.array(rgba.split()[3])
            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                # Add 10% padding
                pad_y = int((y_max - y_min) * 0.1)
                pad_x = int((x_max - x_min) * 0.1)
                y_min = max(0, y_min - pad_y)
                y_max = min(h, y_max + pad_y)
                x_min = max(0, x_min - pad_x)
                x_max = min(w, x_max + pad_x)
                image = image.crop((x_min, y_min, x_max, y_max))
                logger.info(f"Cropped image to object bounds: {image.size}")
        except Exception:
            logger.debug("Smart crop skipped (rembg not available)")

        # Pad to square (improves TripoSR/TRELLIS results)
        size = max(image.size)
        padded = ImageOps.pad(
            image, (size, size), color=(255, 255, 255), centering=(0.5, 0.5)
        )

        output_path = self.stages_dir / "input_preprocessed.png"
        padded.save(output_path)
        logger.info(f"Preprocessed input saved to: {output_path}")

        # Store preprocessed path for later stages
        self._preprocessed_input = output_path

        return [output_path]

    def _generate_3d(self) -> list[Path]:
        """Run the selected AI model to generate 3D mesh from image.

        Falls back to lighter models if the primary model is unavailable.
        Fallback chain: trellis → hunyuan3d-2 → triposr
        """
        # Use preprocessed input if available
        input_path = str(getattr(self, "_preprocessed_input", None) or self.input_path)

        models_to_try = self._resolve_model_chain()

        last_error = None
        for model_name, backend_class_path in models_to_try:
            try:
                logger.info(f"Attempting model: {model_name}...")
                module_path, class_name = backend_class_path
                import importlib

                module = importlib.import_module(module_path)
                backend_cls = getattr(module, class_name)
                backend = backend_cls()

                output_path = self.stages_dir / f"raw_{model_name}.glb"
                backend.generate(
                    input_path,
                    str(output_path),
                    pbr=self.generate_pbr,
                    tex_resolution=self.texture_resolution,
                    mesh_resolution=self.mesh_resolution,
                    foreground_ratio=self.foreground_ratio,
                    chunk_size=self.chunk_size,
                    bake_from_original=self.bake_from_original,
                )
                self._raw_mesh_path = Path(output_path)
                self.model_name = model_name
                logger.info(f"✓ Model '{model_name}' succeeded")
                return [output_path]

            except ImportError as e:
                logger.warning(f"Model '{model_name}' not available: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Model '{model_name}' failed: {e}")
                last_error = e
                continue

        # All models failed
        raise RuntimeError(
            f"All AI models failed to generate a 3D mesh.\n"
            f"  Requested model: {self.model_name}\n"
            f"  Last error: {last_error}\n\n"
            f"Try installing one of the required packages:\n"
            f"  - TripoSR:   (auto-setup by backend)\n"
            f"  - Hunyuan3D: pip install hy3dgen\n"
            f"  - TRELLIS:   pip install trellis2"
        )

    def _resolve_model_chain(self) -> list[tuple[str, tuple[str, str]]]:
        """Resolve the ordered list of models to attempt.

        Returns list of (model_name, (module_path, class_name)) tuples.
        """
        # Model registry: (model_name, (module, class))
        all_models = {
            "trellis": ("assetforge.backends.trellis", "TRELLISBackend"),
            "hunyuan3d-2": ("assetforge.backends.hunyuan3d", "Hunyuan3DBackend"),
            "triposr": ("assetforge.backends.triposr", "TripoSRBackend"),
        }

        if self.model_name == "auto":
            return [
                ("trellis", all_models["trellis"]),
                ("hunyuan3d-2", all_models["hunyuan3d-2"]),
                ("triposr", all_models["triposr"]),
            ]
        elif self.model_name in all_models:
            ordered_keys = ["trellis", "hunyuan3d-2", "triposr"]
            start_idx = ordered_keys.index(self.model_name)
            return [(k, all_models[k]) for k in ordered_keys[start_idx:]]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _cleanup_mesh(self) -> list[Path]:
        """Remove floaters, close holes, smooth the raw mesh."""
        self.require_attr("_raw_mesh_path", "ai_generation")
        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        output_path = self.stages_dir / "mesh_cleaned.obj"
        blender.run_cleanup(str(self._raw_mesh_path), str(output_path))
        self._cleaned_mesh_path = Path(output_path)
        return [output_path]

    def _apply_retopology(self) -> list[Path]:
        """Decimate and remesh for game-optimized topology."""
        self.require_attr("_cleaned_mesh_path", "mesh_cleanup")
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
        """Bake PBR or original-image texture maps onto the mesh."""
        self.require_attr("_raw_mesh_path", "ai_generation")
        from assetforge.backends.blender import BlenderBackend
        import numpy as np
        from PIL import Image

        mesh = getattr(self, "_final_mesh_path", None) or getattr(
            self, "_cleaned_mesh_path", self._raw_mesh_path
        )
        tex_dir = self.stages_dir / "textures"
        tex_dir.mkdir(parents=True, exist_ok=True)

        outputs = []

        # Project original image texture
        if self.bake_from_original:
            try:
                logger.info("Projecting original image onto mesh...")
                input_img = str(
                    getattr(self, "_preprocessed_input", None) or self.input_path
                )
                # Try Blender bake first, fall back to simple vertex color
                try:
                    blender = BlenderBackend()
                    blender.bake_textures(
                        str(mesh),
                        input_img,
                        str(tex_dir),
                        resolution=self.texture_resolution,
                    )
                    outputs.extend(list(tex_dir.glob("*")))
                except Exception:
                    logger.warning(
                        "Blender texture bake unavailable, using vertex colors"
                    )
                    # Fallback: vertex color projection
                    import trimesh

                    m = trimesh.load(str(mesh))
                    img = np.array(
                        Image.open(input_img).resize(
                            (self.texture_resolution, self.texture_resolution)
                        )
                    )
                    verts = m.vertices
                    vmin = verts.min(axis=0)
                    vmax = verts.max(axis=0)
                    vrange = np.maximum(vmax - vmin, 1e-8)
                    norm = (verts - vmin) / vrange
                    uvs = np.stack([norm[:, 0], 1.0 - norm[:, 1]], axis=1)
                    h, w = img.shape[:2]
                    uv_pixels = np.clip(uvs * [w, h], 0, [w - 1, h - 1]).astype(
                        np.int32
                    )
                    vertex_colors = img[uv_pixels[:, 1], uv_pixels[:, 0]] / 255.0
                    colored = trimesh.Trimesh(
                        vertices=m.vertices,
                        faces=m.faces,
                        vertex_colors=vertex_colors,
                    )
                    colored_path = tex_dir.parent / "mesh_textured.obj"
                    colored.export(str(colored_path))
                    outputs.append(colored_path)
                    logger.info(f"Vertex-colored mesh saved to: {colored_path}")
            except Exception as e:
                logger.warning(f"Original-image texture projection failed: {e}")

        # PBR baking via Blender (if requested)
        if self.generate_pbr:
            try:
                logger.info("Baking PBR textures via Blender...")
                blender = BlenderBackend()
                blender.bake_textures(
                    str(mesh),
                    str(getattr(self, "_preprocessed_input", None) or self.input_path),
                    str(tex_dir),
                    resolution=self.texture_resolution,
                )
                outputs.extend(list(tex_dir.glob("*")))
            except Exception as e:
                logger.warning(f"PBR baking failed: {e}")

        return outputs

    def _export_model(self) -> list[Path]:
        """Export the final model in game-ready format."""
        # Fall back through mesh chain: final > cleaned > raw
        mesh = (
            getattr(self, "_final_mesh_path", None)
            or getattr(self, "_cleaned_mesh_path", None)
            or getattr(self, "_raw_mesh_path", None)
        )
        if mesh is None:
            raise RuntimeError(
                "No mesh to export — all generation stages failed. "
                "Check ai_generation stage for details."
            )

        output_ext = f".{self.output_format}"
        output_path = self.output_dir / f"model{output_ext}"

        from assetforge.backends.blender import BlenderBackend

        blender = BlenderBackend()
        blender.export_format(str(mesh), str(output_path))
        return [output_path]
