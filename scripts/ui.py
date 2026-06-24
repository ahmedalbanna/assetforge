#!/usr/bin/env python3
"""
AssetForge Web UI — Try all pipelines through a browser interface.

Usage:
    python3 scripts/ui.py              # Launch (default: http://localhost:7860)
    python3 scripts/ui.py --port 8080  # Custom port
    python3 scripts/ui.py --share      # Create public link (gradio share)
"""

import os
import sys
import time
import json
import logging
import tempfile
import argparse
from pathlib import Path
from typing import Optional

# Ensure the assetforge package is importable
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("assetforge-ui")

try:
    import gradio as gr
except ImportError:
    print("Gradio is required. Install with: pip install gradio")
    sys.exit(1)


# ── Pipeline wrappers ───────────────────────────────────────────────


def run_2d_pipeline(
    input_image,
    quality: str,
    # Background removal options
    bg_remover: str,
    bg_model: str,
    alpha_matting: bool,
    alpha_fg: int,
    alpha_bg: int,
    alpha_erode: int,
    # Sprite options
    sprite_sizes: str,
    pack_sheet: bool,
    generate_svg: bool,
    vectorize_mode: str,
    num_colors: int,
    simplify_tolerance: float,
    min_area: int,
    feather: int,
    # Character feature options
    enable_character_features: bool,
    feature_simplify_tolerance: float,
    feature_min_area: int,
    preserve_line_art: bool,
    line_art_threshold: int,
    enable_eye_highlights: bool,
    semantic_detail_level: str,
    progress=gr.Progress(),
):
    """Run 2D sprite pipeline and return output paths."""
    if input_image is None:
        return None, None, None, None, "⚠️ Please upload an image first."

    from assetforge.pipelines.pipeline_2d import Pipeline2D

    sizes = [int(s.strip()) for s in sprite_sizes.split(",")]
    out_dir = tempfile.mkdtemp(prefix="assetforge_2d_")

    progress(0.1, desc="Initializing pipeline...")

    # Only pass bg_model if using rembg
    bg_model_val = bg_model if bg_remover == "rembg" and bg_model else None

    pipeline = Pipeline2D(
        input_path=input_image,
        output_dir=out_dir,
        quality=quality,
        background_remover=bg_remover,
        bg_model=bg_model_val,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_fg,
        alpha_matting_background_threshold=alpha_bg,
        alpha_matting_erode_size=alpha_erode,
        sprite_sizes=sizes,
        pack_sheet=pack_sheet,
        generate_vector=generate_svg,
        vectorize_mode=vectorize_mode,
        num_colors=num_colors,
        simplify_tolerance=simplify_tolerance,
        min_area=min_area,
        feather_pixels=feather,
        # Character feature options
        enable_character_features=enable_character_features,
        feature_simplify_tolerance=feature_simplify_tolerance,
        feature_min_area=feature_min_area,
        preserve_line_art=preserve_line_art,
        line_art_threshold=line_art_threshold,
        enable_eye_highlights=enable_eye_highlights,
        semantic_detail_level=semantic_detail_level,
    )

    progress(0.3, desc="Running 2D pipeline...")
    results = pipeline.run()

    all_ok = all(r.success for r in results)
    if not all_ok:
        errors = [f"{r.name}: {r.error}" for r in results if not r.success]
        return None, None, None, None, f"❌ Pipeline failed:\n" + "\n".join(errors)

    progress(0.9, desc="Collecting outputs...")

    # Collect output files
    sprite_files = []
    svg_file = None
    sheet_file = None
    sheet_json = None

    for r in results:
        for p in r.output_paths:
            p = Path(p)
            if not p.exists():
                continue
            name = p.name
            if name.endswith(".svg"):
                svg_file = str(p)
            elif "spritesheet" in name.lower():
                if name.endswith(".json"):
                    sheet_json = str(p)
                else:
                    sheet_file = str(p)
            elif name.startswith("sprite_"):
                sprite_files.append(str(p))
            elif "foreground" in name or "cleaned" in name:
                sprite_files.append(str(p))

    # Read SVG content for inline preview
    svg_html = None
    if svg_file and Path(svg_file).exists():
        try:
            svg_content = Path(svg_file).read_text(encoding="utf-8")
            # Embed SVG inline for preview (responsive via max-width)
            svg_html = f'<div style="background:#f5f5f5; border-radius:8px; padding:8px; max-height:400px; overflow:auto;"><div style="max-width:100%;">{svg_content}</div></div>'
        except Exception:
            svg_html = None

    progress(1.0, desc="Done!")
    return (
        sprite_files,
        svg_file,
        svg_html,
        sheet_file,
        "✅ Pipeline completed successfully!",
    )


def run_3d_gen_pipeline(
    input_image,
    model_name: str,
    pbr: bool,
    mesh_resolution: int,
    texture_resolution: int,
    use_retopology: bool,
    progress=gr.Progress(),
):
    """Run 3D generative pipeline."""
    if input_image is None:
        return None, None, "⚠️ Please upload an image first."

    from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

    out_dir = tempfile.mkdtemp(prefix="assetforge_3dgen_")

    progress(0.1, desc="Initializing 3D generation...")
    pipeline = Pipeline3DGen(
        input_path=input_image,
        output_dir=out_dir,
        model_name=model_name,
        generate_pbr=pbr,
        mesh_resolution=mesh_resolution,
        texture_resolution=texture_resolution,
        retopology=use_retopology,
    )

    progress(0.3, desc="Generating 3D model (this may take a while)...")
    try:
        results = pipeline.run()
    except Exception as e:
        return None, None, f"❌ Generation failed: {e}"

    all_ok = all(r.success for r in results)
    if not all_ok:
        errors = [f"{r.name}: {r.error}" for r in results if not r.success]
        return None, None, "❌ " + "\n".join(errors)

    progress(0.9, desc="Collecting outputs...")

    # Find the final output model
    model_file = None
    nerf_features_file = None
    for r in results:
        for p in r.output_paths:
            p = Path(p)
            if p.exists() and p.suffix in (".glb", ".obj", ".gltf"):
                model_file = str(p)

    # Look for NeRF features in stages directory
    nerf_candidates = list(Path(out_dir).rglob("nerf_features.pkl"))
    if nerf_candidates:
        nerf_features_file = str(nerf_candidates[0])

    progress(1.0, desc="Done!")
    return model_file, nerf_features_file, "✅ Model generated successfully!"


def run_3d_scan_pipeline(
    input_images: list,
    quality: str,
    sfm_type: str,
    feature_type: str,
    mask_dir: str,
    resolution_level: int,
    mesh_refinement: bool,
    normalize_coords: bool,
    mesh_quality: str,
    retopology: bool,
    target_faces: int,
    output_format: str,
    progress=gr.Progress(),
):
    """Run 3D photogrammetry pipeline from uploaded images."""
    if not input_images:
        return None, "⚠️ Please upload photos first."

    import tempfile
    import shutil

    # Save uploaded files to a temp directory
    out_dir = tempfile.mkdtemp(prefix="assetforge_3dscan_")
    input_dir = Path(out_dir) / "photos"
    input_dir.mkdir(parents=True, exist_ok=True)

    for img in input_images:
        shutil.copy2(img, input_dir / Path(img).name)

    from assetforge.pipelines.pipeline_3d_photo import Pipeline3DPhoto

    progress(0.1, desc="Initializing photogrammetry pipeline...")

    pipeline = Pipeline3DPhoto(
        input_dir=str(input_dir),
        output_dir=str(Path(out_dir) / "output"),
        quality=quality,
        sfm_type=sfm_type,
        feature_type=feature_type,
        mask_dir=mask_dir if mask_dir else None,
        resolution_level=resolution_level,
        mesh_refinement=mesh_refinement,
        normalize_coordinates=normalize_coords,
        mesh_quality=mesh_quality,
        retopology=retopology,
        target_faces=target_faces,
        output_format=output_format,
    )

    progress(0.2, desc="Running photogrammetry (this may take a while)...")
    try:
        results = pipeline.run()
    except Exception as e:
        return None, f"❌ Pipeline failed: {e}"

    all_ok = all(r.success for r in results)
    if not all_ok:
        errors = [f"{r.name}: {r.error}" for r in results if not r.success]
        return None, "❌ " + "\n".join(errors)

    progress(0.9, desc="Collecting outputs...")

    # Find the final output model
    model_file = None
    for r in results:
        for p in r.output_paths:
            p = Path(p)
            if p.exists() and p.suffix in (".glb", ".obj", ".gltf", ".ply", ".fbx"):
                model_file = str(p)
                break

    progress(1.0, desc="Done!")
    if model_file and Path(model_file).exists():
        size_kb = Path(model_file).stat().st_size / 1024
        return model_file, f"✅ Model reconstructed ({size_kb:.1f} KB)"
    return model_file, "✅ Pipeline completed (check output directory for models)"


def run_retopo_pipeline(
    input_mesh,
    target_faces: int,
    decimate_ratio: float,
    quad_remesh: bool,
    output_format: str,
    progress=gr.Progress(),
):
    """Run retopology pipeline."""
    if input_mesh is None:
        return None, "⚠️ Please upload a mesh file first."

    from assetforge.backends.blender import BlenderBackend

    out_dir = tempfile.mkdtemp(prefix="assetforge_retopo_")
    output_path = Path(out_dir) / f"optimized.{output_format}"

    progress(0.1, desc="Initializing Blender...")
    blender = BlenderBackend()
    if not blender.ensure_available():
        return None, "❌ Blender is not installed. Install it for retopology features."

    # Stage 1: Decimate
    progress(0.3, desc="Decimating mesh...")
    decimated = Path(out_dir) / "decimated.obj"
    try:
        blender.apply_decimate(str(input_mesh), str(decimated), ratio=decimate_ratio)
    except Exception as e:
        return None, f"❌ Decimation failed: {e}"

    # Stage 2: Quad remesh (optional)
    if quad_remesh:
        progress(0.5, desc="Quad remeshing...")
        remeshed = Path(out_dir) / "remeshed.obj"
        try:
            blender.apply_remesh(
                str(decimated), str(remeshed), target_faces=target_faces
            )
            final = remeshed
        except Exception as e:
            return None, f"❌ Quad remesh failed: {e}"
    else:
        final = decimated

    # Stage 3: Export
    progress(0.8, desc="Exporting...")
    try:
        blender.export_format(str(final), str(output_path))
    except Exception as e:
        return None, f"❌ Export failed: {e}"

    # Cleanup temp decimated file
    if decimated.exists() and decimated != output_path:
        decimated.unlink()

    progress(1.0, desc="Done!")
    size_kb = output_path.stat().st_size / 1024
    return str(output_path), f"✅ Mesh optimized ({size_kb:.1f} KB)"


def run_bake_pipeline(
    input_mesh,
    source_type: str,
    nerf_features_file,
    texture_image,
    quality: str,
    resolution: int,
    output_format: str,
    progress=gr.Progress(),
):
    """Run texture baking pipeline."""
    if input_mesh is None:
        return None, "⚠️ Please upload a mesh file first."

    from assetforge.pipelines.pipeline_bake import PipelineBake

    out_dir = tempfile.mkdtemp(prefix="assetforge_bake_")

    nerf_path = None
    tex_path = None
    if source_type == "NeRF Features (.pkl)":
        if nerf_features_file is None:
            return None, "⚠️ Please upload a NeRF features (.pkl) file."
        nerf_path = nerf_features_file
    else:
        if texture_image is None:
            return None, "⚠️ Please upload a texture image."
        tex_path = texture_image

    progress(0.1, desc="Initializing texture baking...")
    pipeline = PipelineBake(
        mesh_path=input_mesh,
        output_dir=out_dir,
        nerf_features_path=nerf_path,
        texture_image_path=tex_path,
        quality=quality,
        resolution=resolution,
        output_format=output_format,
    )

    progress(0.3, desc="Baking textures (this may take a moment)...")
    try:
        results = pipeline.run()
    except Exception as e:
        return None, f"❌ Baking failed: {e}"

    all_ok = all(r.success for r in results)
    if not all_ok:
        errors = [f"{r.name}: {r.error}" for r in results if not r.success]
        return None, "❌ " + "\n".join(errors)

    progress(0.9, desc="Collecting outputs...")

    # Find the final textured output
    output_file = None
    for r in results:
        for p in r.output_paths:
            p = Path(p)
            if p.exists() and p.suffix in (".glb", ".obj", ".fbx"):
                output_file = str(p)
                break

    progress(1.0, desc="Done!")
    if output_file:
        size_kb = Path(output_file).stat().st_size / 1024
        return output_file, f"✅ Textures baked successfully ({size_kb:.1f} KB)"
    return output_file, "✅ Baking complete (check output directory)"


def get_system_info():
    """Gather system information for the Info tab."""
    info = {}

    # GPU
    try:
        from assetforge.utils.gpu import detect_gpu

        gpu = detect_gpu()
        info["GPU"] = gpu
    except Exception as e:
        info["GPU"] = {"error": str(e)}

    # Versions
    info["Versions"] = {}
    try:
        info["Versions"]["Python"] = sys.version.split()[0]
    except Exception:
        pass
    try:
        import torch

        info["Versions"]["PyTorch"] = torch.__version__
    except Exception:
        pass
    try:
        import gradio

        info["Versions"]["Gradio"] = gradio.__version__
    except Exception:
        pass
    try:
        from assetforge import __version__

        info["Versions"]["AssetForge"] = __version__
    except Exception:
        info["Versions"]["AssetForge"] = "0.1.0"

    # Installed backends
    info["Backends"] = {}
    backends = {
        "Rembg": "rembg",
        "Blender": "blender",
        "OpenMVG": "openMVG_main_SfM",
        "OpenMVS": "DensifyPointCloud",
        "COLMAP": "colmap",
        "Instant Meshes": "instant-meshes",
    }
    import shutil

    for name, cmd in backends.items():
        info["Backends"][name] = shutil.which(cmd) is not None

    # AI model backends (Python packages)
    info["AI Backends"] = {}
    ai_backends = {
        "TripoSR": "tsr",
        "DreamGaussian": "dreamgaussian",
        "Hunyuan3D-2": "hy3dgen",
        "TRELLIS": "trellis",
    }
    for name, pkg in ai_backends.items():
        try:
            __import__(pkg)
            info["AI Backends"][name] = True
        except ImportError:
            info["AI Backends"][name] = False

    # Model registry
    try:
        from assetforge.models.registry import list_available_models

        info["Models"] = list_available_models()
    except Exception as e:
        info["Models"] = {"error": str(e)}

    return info


def format_system_info(info: dict) -> str:
    """Format system info as markdown."""
    lines = []

    # GPU
    gpu = info.get("GPU", {})
    if gpu.get("available"):
        lines.append(f"### GPU: ✅ {gpu['device_name']}")
        lines.append(f"- VRAM: {gpu['vram_total_gb']:.1f} GB")
        lines.append(f"- CUDA: {gpu.get('cuda_version', 'N/A')}")
        lines.append(f"- PyTorch: {gpu.get('pytorch_version', 'N/A')}")
    else:
        lines.append("### GPU: ❌ Not available (running on CPU)")
        if "reason" in gpu:
            lines.append(f"- {gpu['reason']}")

    # Versions
    lines.append("\n### Versions")
    versions = info.get("Versions", {})
    for k, v in versions.items():
        lines.append(f"- **{k}:** {v}")

    # Backends
    lines.append("\n### Installed Backends")
    backends = info.get("Backends", {})
    for name, installed in backends.items():
        icon = "✅" if installed else "⬜"
        lines.append(f"- {icon} {name}")
    lines.append("")

    # Models
    models = info.get("Models", {})
    if models and "error" not in models:
        lines.append("\n### Cached Models")
        for name, path in models.items():
            status = "✅" if path else "⬜"
            lines.append(f"- {status} {name}")
    elif isinstance(models, dict) and "error" in models:
        lines.append(f"\n### Models: {models['error']}")

    # AI Backends
    ai_backends = info.get("AI Backends", {})
    if ai_backends:
        lines.append("\n### AI Model Backends")
        for name, installed in ai_backends.items():
            icon = "✅" if installed else "⬜"
            lines.append(f"- {icon} {name}")

    # Texture Baking
    lines.append("\n### Texture Baking")
    lines.append("- ✅ PipelineBake (standalone texture baking)")
    lines.append("- ✅ NeRF feature extraction (TripoSR)")
    lines.append("- ✅ UV unwrapping via xatlas/trimesh")
    lines.append("- ✅ Image projection fallback")

    return "\n".join(lines)


# ── Build the UI ────────────────────────────────────────────────────


def create_ui():
    """Create the Gradio web interface."""
    with gr.Blocks(title="AssetForge") as demo:
        # ── Header ──
        gr.HTML(
            """
            <div class="app-title">
                <h1>🎮 AssetForge</h1>
                <p>Convert real-world images into game-ready digital assets</p>
            </div>
            """
        )

        # ── Tabs ──
        with gr.Tabs():
            # ──── TAB 1: 2D Sprite ────
            with gr.Tab("🎨 2D Sprites"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        input_2d = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            height=300,
                        )

                        # ── AI Quality Preset (controls everything) ──
                        with gr.Group():
                            gr.Markdown("**🎯 AI Quality Preset**")
                            gr.Markdown(
                                "Set quality and let AI choose the best settings. "
                                "Override any setting below in Advanced.",
                            )
                            quality = gr.Radio(
                                label="Quality",
                                choices=[
                                    ("Fast — quick preview, low detail", "fast"),
                                    ("Balanced — general-purpose sprites", "balanced"),
                                    ("High — detailed characters, fine art", "high"),
                                    ("Ultra — maximum quality, SAM2", "ultra"),
                                    ("Auto — analyze image, pick best", "auto"),
                                ],
                                value="balanced",
                                info="Auto-configures all pipeline parameters",
                            )

                        # ── Background Removal (simplified) ──
                        with gr.Group():
                            gr.Markdown("**Background Removal**")
                            bg_remover = gr.Dropdown(
                                label="Backend",
                                choices=["rembg", "sam2"],
                                value="rembg",
                            )
                            bg_model = gr.Dropdown(
                                label="Rembg Model",
                                choices=[
                                    "isnet-general-use",
                                    "u2net",
                                    "u2netp",
                                    "u2net_human_seg",
                                    "isnet-anime",
                                    "silueta",
                                ],
                                value="isnet-general-use",
                                visible=True,
                                info="Only applies when Backend = rembg",
                            )

                        # ── Sprite Options (simplified) ──
                        with gr.Group():
                            gr.Markdown("**Sprites**")
                            sprite_sizes = gr.Textbox(
                                label="Sprite Sizes",
                                value="64,128,256",
                                info="Comma-separated pixel sizes",
                            )
                            with gr.Row():
                                pack_sheet = gr.Checkbox(
                                    label="Pack Spritesheet", value=True
                                )
                                generate_svg = gr.Checkbox(
                                    label="Generate SVG", value=True
                                )

                        # ── Advanced Settings (collapsible) ──
                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            gr.Markdown(
                                "Override individual settings. When set, these "
                                "take precedence over the quality preset.",
                            )
                            with gr.Group():
                                gr.Markdown("**Alpha Matting**")
                                alpha_matting = gr.Checkbox(
                                    label="Enable Alpha Matting",
                                    value=False,
                                    info="Smoother edges (Rembg only)",
                                )
                                with gr.Column(visible=False) as alpha_opts:
                                    alpha_fg = gr.Slider(
                                        label="Alpha FG Threshold",
                                        minimum=0,
                                        maximum=255,
                                        value=240,
                                        step=1,
                                    )
                                    alpha_bg = gr.Slider(
                                        label="Alpha BG Threshold",
                                        minimum=0,
                                        maximum=255,
                                        value=10,
                                        step=1,
                                    )
                                    alpha_erode = gr.Slider(
                                        label="Alpha Erode Size",
                                        minimum=0,
                                        maximum=50,
                                        value=10,
                                        step=1,
                                    )

                            with gr.Group():
                                gr.Markdown("**Vectorization**")
                                feather = gr.Slider(
                                    label="Edge Feather (px)",
                                    minimum=0,
                                    maximum=5,
                                    step=1,
                                    value=1,
                                )
                                vectorize_mode = gr.Dropdown(
                                    label="SVG Mode",
                                    choices=["outline", "color"],
                                    value="color",
                                    interactive=True,
                                    info="Outline = silhouette, Color = multi-layer",
                                )
                                with gr.Column(visible=True) as vec_opts:
                                    num_colors = gr.Slider(
                                        label="Color Count",
                                        minimum=4,
                                        maximum=64,
                                        value=24,
                                        step=1,
                                        info="Number of colors for color mode",
                                    )
                                    simplify_tolerance = gr.Slider(
                                        label="Simplify Tolerance",
                                        minimum=0.5,
                                        maximum=5.0,
                                        value=0.8,
                                        step=0.1,
                                        info="Higher = smoother but less detail",
                                    )
                                    min_area = gr.Slider(
                                        label="Min Area (px)",
                                        minimum=1,
                                        maximum=100,
                                        value=4,
                                        step=1,
                                        info="Minimum contour size to include",
                                    )

                            with gr.Group():
                                gr.Markdown(
                                    "**Character Features (face-aware detail)**"
                                )
                                enable_character_features = gr.Checkbox(
                                    label="Enable Character Features",
                                    value=False,
                                    info="Detect face, eyes, mouth and preserve detail",
                                )
                                with gr.Column(visible=False) as char_opts:
                                    feature_simplify_tolerance = gr.Slider(
                                        label="Feature Simplify Tolerance",
                                        minimum=0.1,
                                        maximum=3.0,
                                        value=0.5,
                                        step=0.1,
                                        info="Lower = more detail in eyes/mouth",
                                    )
                                    feature_min_area = gr.Slider(
                                        label="Feature Min Area (px)",
                                        minimum=1,
                                        maximum=30,
                                        value=2,
                                        step=1,
                                        info="Minimum contour size for facial features",
                                    )
                                    with gr.Row():
                                        preserve_line_art = gr.Checkbox(
                                            label="Line Art (anime strokes)",
                                            value=True,
                                            info="Extract thin edges as stroke layer",
                                        )
                                        enable_eye_highlights = gr.Checkbox(
                                            label="Eye Highlights",
                                            value=True,
                                            info="Add procedural eye highlights",
                                        )
                                    line_art_threshold = gr.Slider(
                                        label="Line Art Threshold",
                                        minimum=10,
                                        maximum=150,
                                        value=30,
                                        step=5,
                                        info="Lower = more lines detected",
                                    )
                                    semantic_detail_level = gr.Dropdown(
                                        label="Detail Level",
                                        choices=["low", "medium", "high"],
                                        value="high",
                                        info="Extra detail for facial features",
                                    )

                                # Show/hide character feature options based on checkbox
                                enable_character_features.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=[enable_character_features],
                                    outputs=[char_opts],
                                )

                        run_btn_2d = gr.Button(
                            "🚀 Generate Sprites", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        status_2d = gr.Markdown(
                            "Ready. Upload an image and click generate.",
                            elem_classes="status-msg",
                        )
                        sprite_gallery = gr.Gallery(
                            label="Generated Sprites",
                            columns=4,
                            height=300,
                            object_fit="contain",
                        )
                        with gr.Accordion("SVG Preview", open=True):
                            svg_preview = gr.HTML(
                                value='<p style="color:#888; text-align:center; padding:20px;">✨ Generate an SVG to see preview here.</p>',
                                label="SVG Preview",
                            )
                        with gr.Row():
                            svg_download = gr.File(label="Download SVG", visible=True)
                            sheet_download = gr.File(
                                label="Download Spritesheet", visible=True
                            )

                # ── Event wiring ──

                # Show/hide alpha matting options
                alpha_matting.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[alpha_matting],
                    outputs=[alpha_opts],
                )

                # Show/hide bg_model based on backend choice
                bg_remover.change(
                    fn=lambda x: gr.update(visible=(x == "rembg")),
                    inputs=[bg_remover],
                    outputs=[bg_model],
                )

                # Show/hide color-specific options based on vectorize mode
                vectorize_mode.change(
                    fn=lambda x: gr.update(visible=(x == "color")),
                    inputs=[vectorize_mode],
                    outputs=[vec_opts],
                )

                run_btn_2d.click(
                    fn=run_2d_pipeline,
                    inputs=[
                        input_2d,
                        quality,
                        bg_remover,
                        bg_model,
                        alpha_matting,
                        alpha_fg,
                        alpha_bg,
                        alpha_erode,
                        sprite_sizes,
                        pack_sheet,
                        generate_svg,
                        vectorize_mode,
                        num_colors,
                        simplify_tolerance,
                        min_area,
                        feather,
                        enable_character_features,
                        feature_simplify_tolerance,
                        feature_min_area,
                        preserve_line_art,
                        line_art_threshold,
                        enable_eye_highlights,
                        semantic_detail_level,
                    ],
                    outputs=[
                        sprite_gallery,
                        svg_download,
                        svg_preview,
                        sheet_download,
                        status_2d,
                    ],
                )

            # ──── TAB 2: 3D Generative ────
            with gr.Tab("🧊 3D Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        input_3d = gr.Image(
                            label="Upload Concept Art",
                            type="filepath",
                            height=300,
                        )
                        with gr.Group():
                            model_choice = gr.Dropdown(
                                label="AI Model",
                                choices=[
                                    ("Auto (recommended)", "auto"),
                                    ("TripoSR — fastest, <0.5s", "triposr"),
                                    (
                                        "DreamGaussian — textured, iterative",
                                        "dreamgaussian",
                                    ),
                                    ("Hunyuan3D-2 — high quality PBR", "hunyuan3d-2"),
                                    ("TRELLIS — ultra quality", "trellis"),
                                ],
                                value="auto",
                                info="Auto selects best model for your GPU",
                            )
                            with gr.Row():
                                pbr_toggle = gr.Checkbox(
                                    label="PBR Textures",
                                    value=False,
                                    info="Generate albedo/normal/roughness maps",
                                )
                                retopology_toggle = gr.Checkbox(
                                    label="Retopology",
                                    value=False,
                                    info="Optimize mesh for game engines",
                                )

                        # Advanced settings
                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            gr.Markdown("**Geometric & Texture Quality**")
                            with gr.Row():
                                mesh_res_3d = gr.Slider(
                                    label="Mesh Resolution",
                                    minimum=128,
                                    maximum=512,
                                    step=128,
                                    value=256,
                                    info="Higher = more geometric detail",
                                )
                                tex_res_3d = gr.Slider(
                                    label="Texture Resolution",
                                    minimum=512,
                                    maximum=4096,
                                    step=512,
                                    value=2048,
                                    info="Output texture size (px)",
                                )

                        # NeRF features output for later texture baking
                        nerf_features_output = gr.File(
                            label="NeRF Features (for texture baking)",
                            visible=True,
                            interactive=False,
                        )
                        gr.Markdown(
                            "*NeRF features saved automatically. Use 'Texture Bake' tab "
                            "to re-bake after editing the mesh in Blender.*",
                        )

                        run_btn_3d = gr.Button(
                            "🚀 Generate 3D Model", variant="primary", size="lg"
                        )
                        # Examples
                        gr.Markdown("### Examples")
                        gr.Examples(
                            examples=[
                                [str(REPO_DIR / "examples" / "sample_2d_input.png")],
                            ],
                            inputs=[input_3d],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        status_3d = gr.Markdown("Ready.")
                        model_viewer = gr.Model3D(
                            label="3D Model",
                            clear_color=(0.2, 0.2, 0.2, 1.0),
                            height=500,
                        )
                        model_download = gr.File(label="Download Model")

                run_btn_3d.click(
                    fn=run_3d_gen_pipeline,
                    inputs=[
                        input_3d,
                        model_choice,
                        pbr_toggle,
                        mesh_res_3d,
                        tex_res_3d,
                        retopology_toggle,
                    ],
                    outputs=[model_viewer, nerf_features_output, status_3d],
                ).then(
                    fn=lambda m: m,
                    inputs=[model_viewer],
                    outputs=[model_download],
                )

            # ──── TAB 3: 3D Photogrammetry Scan ────
            with gr.Tab("📷 3D Scan"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        input_scan_images = gr.File(
                            label="Upload Photos (from multiple angles)",
                            file_count="multiple",
                            file_types=[
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".webp",
                                ".bmp",
                                ".tiff",
                            ],
                        )

                        # ── Quality Preset ──
                        with gr.Group():
                            gr.Markdown("**🎯 Quality Preset**")
                            scan_quality = gr.Radio(
                                label="Quality",
                                choices=[
                                    (
                                        "Fast — ORB, global SfM, low mesh, no refine",
                                        "fast",
                                    ),
                                    (
                                        "Balanced — SIFT, incremental SfM, medium mesh",
                                        "balanced",
                                    ),
                                    (
                                        "High — SIFT, incremental SfM, high mesh, refine",
                                        "high",
                                    ),
                                    (
                                        "Ultra — SIFT, global SfM, high mesh, refine",
                                        "ultra",
                                    ),
                                ],
                                value="balanced",
                            )

                        # ── Research-backed Parameters ──
                        with gr.Accordion("🔬 Research-Backed Settings", open=False):
                            gr.Markdown(
                                "**Based on analysis of OpenMVG/OpenMVS internals**"
                            )
                            with gr.Group():
                                gr.Markdown("**SfM Configuration**")
                                scan_sfm_type = gr.Dropdown(
                                    label="SfM Algorithm",
                                    choices=[
                                        (
                                            "Incremental — robust, handles sparse data",
                                            "incremental",
                                        ),
                                        ("Global — faster, no drift", "global"),
                                        (
                                            "Incremental V2 — updated incremental",
                                            "incremental_v2",
                                        ),
                                    ],
                                    value="incremental",
                                    info="Global is 2-5x faster; Incremental is more robust",
                                )
                                scan_feature_type = gr.Dropdown(
                                    label="Feature Extractor",
                                    choices=[
                                        (
                                            "SIFT — best quality, scale-invariant",
                                            "sift",
                                        ),
                                        ("ORB — fast, good for real-time", "orb"),
                                        ("AKAZE — middle ground", "akaze"),
                                    ],
                                    value="sift",
                                )
                                scan_mask_dir = gr.Textbox(
                                    label="Mask Directory (optional)",
                                    placeholder="/path/to/masks",
                                    info="global_mask.png or <image>_mask.png per image",
                                )

                            with gr.Group():
                                gr.Markdown("**Memory Management**")
                                scan_resolution_level = gr.Slider(
                                    label="Resolution Level (0=full, 2=quarter)",
                                    minimum=0,
                                    maximum=3,
                                    step=1,
                                    value=0,
                                    info="Higher = less memory, lower quality",
                                )
                                with gr.Row():
                                    scan_mesh_refinement = gr.Checkbox(
                                        label="Mesh Refinement (15-30% detail boost)",
                                        value=True,
                                    )
                                    scan_normalize_coords = gr.Checkbox(
                                        label="Normalize Coordinates (prevents precision loss)",
                                        value=True,
                                    )

                            with gr.Group():
                                gr.Markdown("**Seam Leveling Control**")
                                gr.Markdown(
                                    "*Automatically disabled when masks are present "
                                    "to prevent black artifacts (research-backed)*",
                                )

                        # ── Basic Settings ──
                        with gr.Accordion("Basic Settings", open=True):
                            scan_mesh_quality = gr.Dropdown(
                                label="Mesh Quality",
                                choices=["low", "medium", "high"],
                                value="high",
                            )
                            scan_retopology = gr.Checkbox(
                                label="Apply Retopology (game-ready topology)",
                                value=False,
                            )
                            with gr.Row():
                                scan_target_faces = gr.Number(
                                    label="Target Faces",
                                    value=15000,
                                    precision=0,
                                    minimum=100,
                                    maximum=500000,
                                )
                                scan_output_format = gr.Dropdown(
                                    label="Output Format",
                                    choices=["glb", "obj", "fbx", "ply"],
                                    value="glb",
                                )

                        run_btn_scan = gr.Button(
                            "🚀 Reconstruct 3D Model", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        status_scan = gr.Markdown(
                            "Ready. Upload photos and click reconstruct."
                        )
                        scan_model_viewer = gr.Model3D(
                            label="Reconstructed Model",
                            clear_color=(0.2, 0.2, 0.2, 1.0),
                            height=500,
                        )
                        scan_model_download = gr.File(label="Download Model")

                run_btn_scan.click(
                    fn=run_3d_scan_pipeline,
                    inputs=[
                        input_scan_images,
                        scan_quality,
                        scan_sfm_type,
                        scan_feature_type,
                        scan_mask_dir,
                        scan_resolution_level,
                        scan_mesh_refinement,
                        scan_normalize_coords,
                        scan_mesh_quality,
                        scan_retopology,
                        scan_target_faces,
                        scan_output_format,
                    ],
                    outputs=[scan_model_viewer, status_scan],
                ).then(
                    fn=lambda m: m,
                    inputs=[scan_model_viewer],
                    outputs=[scan_model_download],
                )

            # ──── TAB 4: Texture Baking ────
            with gr.Tab("🎨 Texture Bake"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        bake_mesh_input = gr.File(
                            label="Upload Mesh (OBJ/GLB/GLTF/FBX/PLY)",
                            file_types=[
                                ".obj",
                                ".glb",
                                ".gltf",
                                ".fbx",
                                ".ply",
                                ".stl",
                            ],
                        )

                        with gr.Group():
                            bake_source_type = gr.Radio(
                                label="Texture Source",
                                choices=[
                                    "NeRF Features (.pkl)",
                                    "Image Projection",
                                ],
                                value="Image Projection",
                                info="NeRF features from 3D Gen pipeline; Image for simple projection",
                            )

                        # NeRF features input (shown when NeRF selected)
                        bake_nerf_input = gr.File(
                            label="NeRF Features (.pkl)",
                            visible=False,
                            file_types=[".pkl"],
                        )

                        # Texture image input (shown when Image selected)
                        bake_image_input = gr.Image(
                            label="Texture Image",
                            type="filepath",
                            visible=True,
                            height=200,
                        )

                        # Quality & output settings
                        with gr.Group():
                            gr.Markdown("**Baking Settings**")
                            bake_quality = gr.Radio(
                                label="Quality",
                                choices=[
                                    ("Fast — 1024px", "fast"),
                                    ("Balanced — 2048px", "balanced"),
                                    ("High — 4096px", "high"),
                                    ("Ultra — 8192px", "ultra"),
                                ],
                                value="balanced",
                            )
                            with gr.Row():
                                bake_resolution = gr.Slider(
                                    label="Resolution",
                                    minimum=256,
                                    maximum=8192,
                                    step=256,
                                    value=2048,
                                    visible=False,  # Hidden when using preset
                                )
                                bake_format = gr.Radio(
                                    label="Output Format",
                                    choices=["glb", "obj", "fbx"],
                                    value="glb",
                                )

                        run_btn_bake = gr.Button(
                            "🎨 Bake Textures", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        status_bake = gr.Markdown("Ready.")
                        bake_output_viewer = gr.Model3D(
                            label="Textured Model",
                            clear_color=(0.2, 0.2, 0.2, 1.0),
                            height=500,
                        )
                        bake_output_file = gr.File(label="Download Textured Model")

                # Show/hide inputs based on source type
                bake_source_type.change(
                    fn=lambda x: (
                        gr.update(visible=(x == "NeRF Features (.pkl)")),
                        gr.update(visible=(x == "Image Projection")),
                    ),
                    inputs=[bake_source_type],
                    outputs=[bake_nerf_input, bake_image_input],
                )

                run_btn_bake.click(
                    fn=run_bake_pipeline,
                    inputs=[
                        bake_mesh_input,
                        bake_source_type,
                        bake_nerf_input,
                        bake_image_input,
                        bake_quality,
                        bake_resolution,
                        bake_format,
                    ],
                    outputs=[bake_output_viewer, status_bake],
                ).then(
                    fn=lambda m: m,
                    inputs=[bake_output_viewer],
                    outputs=[bake_output_file],
                )

            # ──── TAB 5: Retopology ────
            with gr.Tab("🔧 Retopology"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        input_mesh = gr.File(
                            label="Upload Mesh (OBJ/GLB/FBX)",
                            file_types=[
                                ".obj",
                                ".glb",
                                ".gltf",
                                ".fbx",
                                ".ply",
                                ".stl",
                            ],
                        )
                        with gr.Group():
                            target_faces = gr.Number(
                                label="Target Faces",
                                value=15000,
                                precision=0,
                                minimum=100,
                                maximum=500000,
                            )
                            decimate_ratio = gr.Slider(
                                label="Decimation Ratio",
                                minimum=0.01,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                info="1.0 = no change, 0.01 = max reduction",
                            )
                            quad_remesh = gr.Checkbox(
                                label="Quad Remeshing", value=True
                            )
                            output_format = gr.Radio(
                                label="Output Format",
                                choices=["obj", "glb", "fbx"],
                                value="obj",
                            )
                        run_btn_retopo = gr.Button(
                            "🚀 Optimize Mesh", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        status_retopo = gr.Markdown("Ready.")
                        mesh_viewer = gr.Model3D(
                            label="Optimized Mesh",
                            clear_color=(0.2, 0.2, 0.2, 1.0),
                            height=500,
                        )
                        mesh_download = gr.File(label="Download Optimized Mesh")

                run_btn_retopo.click(
                    fn=run_retopo_pipeline,
                    inputs=[
                        input_mesh,
                        target_faces,
                        decimate_ratio,
                        quad_remesh,
                        output_format,
                    ],
                    outputs=[mesh_viewer, status_retopo],
                ).then(
                    fn=lambda m: m,
                    inputs=[mesh_viewer],
                    outputs=[mesh_download],
                )

            # ──── TAB 4: Info ────
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("### System Information")
                info_output = gr.Markdown("*Click refresh to load system info.*")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

                def refresh_info():
                    info = get_system_info()
                    return format_system_info(info)

                refresh_btn.click(fn=refresh_info, outputs=[info_output])

                # Auto-load on page load
                demo.load(fn=refresh_info, outputs=[info_output])

    return demo


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="AssetForge Web UI — Try all pipelines through a browser."
    )
    parser.add_argument(
        "--port", "-p", type=int, default=7860, help="Port to run the UI on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()

    print(f"\n  🎮 AssetForge Web UI")
    print(f"  {'=' * 40}")
    print(f"  Starting on http://{args.host}:{args.port}")
    if args.share:
        print(f"  Public share link will be generated...")
    print()

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .app-title { text-align: center; padding: 0.5rem 0; }
        .app-title h1 { font-size: 2rem; margin: 0; }
        .app-title p { color: #666; margin: 0; }
        .output-image { max-height: 400px; }
        .status-msg { padding: 0.5rem; border-radius: 0.5rem; }
        .success-msg { background: #d4edda; color: #155724; }
        .error-msg { background: #f8d7da; color: #721c24; }
        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
