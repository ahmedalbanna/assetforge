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
    bg_remover: str,
    sprite_sizes: str,
    pack_sheet: bool,
    generate_svg: bool,
    feather: int,
    progress=gr.Progress(),
):
    """Run 2D sprite pipeline and return output paths."""
    if input_image is None:
        return None, None, None, "⚠️ Please upload an image first."

    from assetforge.pipelines.pipeline_2d import Pipeline2D

    sizes = [int(s.strip()) for s in sprite_sizes.split(",")]
    out_dir = tempfile.mkdtemp(prefix="assetforge_2d_")

    progress(0.1, desc="Initializing pipeline...")
    pipeline = Pipeline2D(
        input_path=input_image,
        output_dir=out_dir,
        background_remover=bg_remover,
        sprite_sizes=sizes,
        pack_sheet=pack_sheet,
        generate_vector=generate_svg,
        feather_pixels=feather,
    )

    progress(0.3, desc="Running 2D pipeline...")
    results = pipeline.run()

    all_ok = all(r.success for r in results)
    if not all_ok:
        errors = [f"{r.name}: {r.error}" for r in results if not r.success]
        return None, None, None, f"❌ Pipeline failed:\n" + "\n".join(errors)

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

    progress(1.0, desc="Done!")
    return sprite_files, svg_file, sheet_file, "✅ Pipeline completed successfully!"


def run_3d_gen_pipeline(
    input_image,
    model_name: str,
    pbr: bool,
    progress=gr.Progress(),
):
    """Run 3D generative pipeline."""
    if input_image is None:
        return None, "⚠️ Please upload an image first."

    from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

    out_dir = tempfile.mkdtemp(prefix="assetforge_3dgen_")

    progress(0.1, desc="Initializing 3D generation...")
    pipeline = Pipeline3DGen(
        input_path=input_image,
        output_dir=out_dir,
        model_name=model_name,
        generate_pbr=pbr,
    )

    progress(0.3, desc="Generating 3D model (this may take a while)...")
    try:
        results = pipeline.run()
    except Exception as e:
        return None, f"❌ Generation failed: {e}"

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
            if p.exists() and p.suffix in (".glb", ".obj", ".gltf"):
                model_file = str(p)
                break

    progress(1.0, desc="Done!")
    return model_file, "✅ Model generated successfully!"


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
                        with gr.Group():
                            bg_remover = gr.Dropdown(
                                label="Background Remover",
                                choices=["rembg", "sam2"],
                                value="rembg",
                            )
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
                            feather = gr.Slider(
                                label="Edge Feather (px)",
                                minimum=0,
                                maximum=5,
                                step=1,
                                value=1,
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
                        with gr.Row():
                            svg_download = gr.File(label="SVG Vector", visible=True)
                            sheet_download = gr.File(label="Spritesheet", visible=True)

                run_btn_2d.click(
                    fn=run_2d_pipeline,
                    inputs=[
                        input_2d,
                        bg_remover,
                        sprite_sizes,
                        pack_sheet,
                        generate_svg,
                        feather,
                    ],
                    outputs=[sprite_gallery, svg_download, sheet_download, status_2d],
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
                                choices=["auto", "triposr", "hunyuan3d-2", "trellis"],
                                value="auto",
                                info="Auto selects best model for your GPU",
                            )
                            pbr_toggle = gr.Checkbox(
                                label="Generate PBR Textures",
                                value=False,
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
                    inputs=[input_3d, model_choice, pbr_toggle],
                    outputs=[model_viewer, status_3d],
                ).then(
                    fn=lambda m: m,
                    inputs=[model_viewer],
                    outputs=[model_download],
                )

            # ──── TAB 3: Retopology ────
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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
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
