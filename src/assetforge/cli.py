"""
AssetForge CLI — Command-line interface for the asset generation pipelines.

Uses Typer (built on Click) for a modern CLI experience.
"""

import typer
from pathlib import Path
from typing import Optional
from assetforge.utils.gpu import detect_gpu, print_gpu_info
from assetforge.utils.logging import setup_logging

app = typer.Typer(
    name="assetforge",
    help="Convert real-world images into game-ready digital assets.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        from assetforge import __version__

        typer.echo(f"AssetForge v{__version__}")
        raise typer.Exit()


@app.callback()
def cli_callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress non-error output"
    ),
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version"
    ),
):
    """AssetForge: Image-to-asset pipeline tool for game developers."""
    setup_logging(verbose=verbose, quiet=quiet)


# ──────────────────────────────────────────────
# 2D Sprite Pipeline
# ──────────────────────────────────────────────


@app.command(name="2d")
def cmd_2d(
    input: str = typer.Argument(..., help="Input image path (JPG/PNG)"),
    output: str = typer.Option("./output", "-o", "--output", help="Output directory"),
    background_remover: str = typer.Option(
        "sam2", "--bg-remover", help="Background removal model: 'sam2' or 'rembg'"
    ),
    sprite_sizes: str = typer.Option(
        "64,128,256", "--sprite-sizes", help="Comma-separated output sprite sizes"
    ),
    sheet: bool = typer.Option(
        False, "--sheet", help="Pack sprites into a sprite sheet"
    ),
    vectorize: bool = typer.Option(
        False, "--vectorize", help="Also output SVG vector paths"
    ),
    feather: int = typer.Option(0, "--feather", help="Edge feather radius in pixels"),
):
    """Extract a 2D sprite from an image and prepare it for game engines."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    sizes = [int(s.strip()) for s in sprite_sizes.split(",")]
    pipeline = Pipeline2D(
        input_path=input,
        output_dir=output,
        background_remover=background_remover,
        sprite_sizes=sizes,
        pack_sheet=sheet,
        generate_vector=vectorize,
        feather_pixels=feather,
    )
    pipeline.run()


# ──────────────────────────────────────────────
# 3D Photogrammetry Pipeline
# ──────────────────────────────────────────────


@app.command(name="3d-scan")
def cmd_3d_scan(
    input: str = typer.Argument(..., help="Directory of multi-angle photos"),
    output: str = typer.Option("./output", "-o", "--output", help="Output directory"),
    feature_type: str = typer.Option(
        "sift", "--feature-type", help="Feature extractor: 'sift' or 'orb'"
    ),
    mesh_quality: str = typer.Option(
        "high", "--mesh-quality", help="Mesh quality: 'low', 'medium', 'high'"
    ),
    retopology: bool = typer.Option(
        False, "--retopology", help="Apply quad remeshing for game use"
    ),
    target_faces: int = typer.Option(
        15000, "--target-faces", help="Target triangle count"
    ),
    format: str = typer.Option(
        "glb", "--format", help="Output format: 'glb', 'obj', 'fbx'"
    ),
):
    """Reconstruct a 3D model from multiple photos using photogrammetry."""
    from assetforge.pipelines.pipeline_3d_photo import Pipeline3DPhoto

    pipeline = Pipeline3DPhoto(
        input_dir=input,
        output_dir=output,
        feature_type=feature_type,
        mesh_quality=mesh_quality,
        retopology=retopology,
        target_faces=target_faces,
        output_format=format,
    )
    pipeline.run()


# ──────────────────────────────────────────────
# 3D Generative AI Pipeline
# ──────────────────────────────────────────────


@app.command(name="3d-gen")
def cmd_3d_gen(
    input: str = typer.Argument(
        ..., help="Single input image (concept art, reference)"
    ),
    output: str = typer.Option("./output", "-o", "--output", help="Output directory"),
    # Quality preset (overrides individual settings)
    quality: str = typer.Option(
        "",
        "--quality",
        "-q",
        help="Quality preset: 'fast', 'balanced', 'high', 'ultra'. "
        "Overrides per-parameter flags. If empty, uses per-param defaults.",
    ),
    # Model selection
    model: str = typer.Option(
        "auto",
        "--model",
        "-m",
        help="AI model: 'auto', 'triposr', 'hunyuan3d-2', 'trellis'",
    ),
    # Geometric fidelity
    mesh_resolution: int = typer.Option(
        256,
        "--mesh-res",
        "-mr",
        help="Marching cubes resolution (128/256/512). Higher = more detail but slower.",
    ),
    foreground_ratio: float = typer.Option(
        0.85,
        "--foreground-ratio",
        "-fg",
        help="Object fill ratio (0.5-1.0). Higher = more pixels on object.",
    ),
    chunk_size: int = typer.Option(
        131072,
        "--chunk-size",
        "-cs",
        help="Renderer chunk size. Lower = less VRAM, higher = faster.",
    ),
    # Sampling quality
    steps: int = typer.Option(
        50,
        "--steps",
        "-s",
        help="Denoising steps (10-100). Higher = better quality but slower.",
    ),
    guidance: float = typer.Option(
        5.0,
        "--guidance",
        "-g",
        help="CFG scale (3.0-15.0). Higher = more faithful to input.",
    ),
    # Texture quality
    tex_resolution: int = typer.Option(
        2048,
        "--tex-resolution",
        "-tr",
        help="Texture resolution (1024/2048/4096/8192)",
    ),
    pbr: bool = typer.Option(
        False,
        "--pbr",
        help="Generate full PBR material maps (albedo, normal, roughness)",
    ),
    bake_original: bool = typer.Option(
        True,
        "--bake-original/--no-bake-original",
        help="Project original input image onto mesh as texture",
    ),
    # Retopology
    retopology: bool = typer.Option(
        False, "--retopology", "-r", help="Optimize mesh topology for game engines"
    ),
    target_faces: int = typer.Option(
        20000,
        "--target-faces",
        "-tf",
        help="Target triangle/quad count after retopology",
    ),
    # Output format
    format: str = typer.Option(
        "glb", "--format", "-f", help="Output format: 'glb', 'obj', 'fbx'"
    ),
):
    """Generate a 3D model from a single image using AI.

    Quality presets (--quality):
    \b
      fast      \tTripoSR, low res, no PBR (fastest)
      balanced  \tAuto model, medium res, PBR (default)
      high      \tHunyuan3D-2, high res, PBR + retopology
      ultra     \tTRELLIS, max res, PBR + retopology

    For fine control, use individual flags instead of --quality.
    """
    from assetforge.pipelines.pipeline_3d_gen import Pipeline3DGen

    pipeline = Pipeline3DGen(
        input_path=input,
        output_dir=output,
        quality=quality,
        model_name=model,
        mesh_resolution=mesh_resolution,
        foreground_ratio=foreground_ratio,
        chunk_size=chunk_size,
        inference_steps=steps,
        guidance_scale=guidance,
        texture_resolution=tex_resolution,
        generate_pbr=pbr,
        bake_from_original=bake_original,
        retopology=retopology,
        target_faces=target_faces,
        output_format=format,
    )
    pipeline.run()


# ──────────────────────────────────────────────
# Gaussian Splatting Pipeline
# ──────────────────────────────────────────────


@app.command(name="3dgs")
def cmd_3dgs(
    input: str = typer.Argument(..., help="Photo directory or video file"),
    output: str = typer.Option("./output", "-o", "--output", help="Output directory"),
    extract_mesh: bool = typer.Option(
        False, "--extract-mesh", help="Extract surface mesh from splats"
    ),
    mesh_method: str = typer.Option(
        "milo", "--mesh-method", help="Mesh extraction: 'milo' or 'sugar'"
    ),
    compress: bool = typer.Option(
        False, "--compress", help="Compress using SPZ format"
    ),
    quality: str = typer.Option(
        "medium", "--quality", help="Training quality: 'low', 'medium', 'high'"
    ),
    max_splats: int = typer.Option(
        1_000_000, "--max-splats", help="Maximum number of Gaussians"
    ),
):
    """Train 3D Gaussian Splatting from photos for real-time rendering."""
    from assetforge.pipelines.pipeline_3dgs import Pipeline3DGS

    pipeline = Pipeline3DGS(
        input_path=input,
        output_dir=output,
        extract_mesh=extract_mesh,
        mesh_method=mesh_method,
        compress=compress,
        quality=quality,
        max_splats=max_splats,
    )
    pipeline.run()


# ──────────────────────────────────────────────
# Standalone Retopology Command
# ──────────────────────────────────────────────


@app.command()
def retopo(
    input: str = typer.Argument(..., help="Input mesh file (OBJ/GLB/FBX)"),
    output: str = typer.Argument(..., help="Output mesh file (OBJ/GLB)"),
    target_faces: int = typer.Option(
        15000, "--target-faces", "-f", help="Target triangle count"
    ),
    decimate_ratio: float = typer.Option(
        0.5, "--ratio", "-r", help="Decimation ratio (0.0-1.0)"
    ),
    quad_remesh: bool = typer.Option(
        False, "--quads", "-q", help="Apply quad remeshing after decimation"
    ),
    format: str = typer.Option(
        "obj", "--format", help="Output format: 'obj', 'glb', 'fbx'"
    ),
):
    """Optimize a mesh for game engines — decimate, remesh, and export."""
    from assetforge.pipelines.base import StageResult

    # Ensure output has the right extension
    output_path = Path(output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")

    from assetforge.backends.blender import BlenderBackend

    blender = BlenderBackend()

    if not blender.ensure_available():
        raise typer.Exit(1)

    typer.echo(f"Input: {input}")
    typer.echo(f"Output: {output_path}")
    typer.echo(f"Target faces: {target_faces}")

    # Stage 1: Decimate
    typer.echo("Stage 1/3: Decimating mesh...")
    decimated = Path(str(output_path) + ".decimated.obj")
    try:
        blender.apply_decimate(str(input), str(decimated), ratio=decimate_ratio)
    except Exception as e:
        typer.echo(f"Decimation failed: {e}", err=True)
        raise typer.Exit(1)

    # Stage 2: Quad remesh (optional)
    if quad_remesh:
        typer.echo("Stage 2/3: Quad remeshing...")
        remeshed = Path(str(output_path) + ".remeshed.obj")
        try:
            blender.apply_remesh(
                str(decimated), str(remeshed), target_faces=target_faces
            )
            final_mesh = remeshed
        except Exception as e:
            typer.echo(f"Quad remesh failed: {e}", err=True)
            final_mesh = decimated
    else:
        final_mesh = decimated

    # Stage 3: Export
    typer.echo("Stage 3/3: Exporting...")
    try:
        blender.export_format(str(final_mesh), str(output_path))
    except Exception as e:
        typer.echo(f"Export failed: {e}", err=True)
        raise typer.Exit(1)

    # Cleanup temp files
    if decimated.exists() and decimated != Path(output_path):
        decimated.unlink()
    if quad_remesh and "remeshed" in locals():
        remeshed_path = Path(str(output_path) + ".remeshed.obj")
        if remeshed_path.exists() and remeshed_path != Path(output_path):
            remeshed_path.unlink()

    size_kb = output_path.stat().st_size / 1024
    typer.echo(f"✓ Mesh optimized and saved to: {output_path} ({size_kb:.1f} KB)")


# ──────────────────────────────────────────────
# Standalone Splat Compression Command
# ──────────────────────────────────────────────


@app.command()
def compress(
    input: str = typer.Argument(..., help="Input PLY file from 3DGS training"),
    output: str = typer.Argument(..., help="Output .spz compressed file"),
    quality: int = typer.Option(
        80, "--quality", "-q", help="Compression quality (0-100)", min=0, max=100
    ),
):
    """Compress Gaussian splats to SPZ format for efficient delivery."""
    from assetforge.backends.gaussian_splat import compress_ply

    output_path = compress_ply(input, output, quality=quality)
    size_kb = output_path.stat().st_size / 1024
    typer.echo(f"✓ Compressed to: {output_path} ({size_kb:.1f} KB)")


# ──────────────────────────────────────────────
# Utility Commands
# ──────────────────────────────────────────────


@app.command()
def info():
    """Show system information: GPU, CUDA, installed models."""
    gpu = detect_gpu()
    print_gpu_info(gpu)


# Entry point
def main():
    app()


if __name__ == "__main__":
    main()
