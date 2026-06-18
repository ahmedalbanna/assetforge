#!/usr/bin/env python3
"""
AssetForge Demo — end-to-end showcase of all working pipelines.

Usage:
    python3 scripts/demo.py

This will:
1. Run the 2D sprite pipeline (background removal → sprites → SVG)
2. Run retopo optimization on the test mesh
3. Display summary results
"""

import time
import sys
from pathlib import Path

# Ensure the assetforge package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEP = "━" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def step(msg: str):
    print(f"  ● {msg}")


def ok(msg: str):
    print(f"  ✓ {msg}")


def demo_2d_pipeline():
    """Run the 2D sprite pipeline end-to-end."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    section("1. 2D Sprite Pipeline")
    print("  Input:   examples/sample_2d_input.png")
    print("  Output:  demo_output/2d_sprites/")

    start = time.time()
    pipeline = Pipeline2D(
        input_path="examples/sample_2d_input.png",
        output_dir="demo_output/2d_sprites",
        sprite_sizes=[64, 128, 256],
        background_remover="rembg",
        pack_sheet=True,
    )
    results = pipeline.run()
    elapsed = time.time() - start

    all_ok = all(r.success for r in results)
    if all_ok:
        ok(f"Pipeline completed in {elapsed:.2f}s ({len(results)} stages)")
        for r in results:
            for p in r.output_paths:
                p = Path(p)
                if p.exists():
                    print(
                        f"       → {p.relative_to(Path.cwd())} ({p.stat().st_size / 1024:.1f} KB)"
                    )
    else:
        print("  ✗ Pipeline failed")
        for r in results:
            if not r.success:
                print(f"    ✗ {r.name}: {r.error}")
        return False
    return True


def demo_retopo():
    """Run retopo optimization on the test mesh."""
    from assetforge.backends.blender import BlenderBackend

    section("2. Mesh Retopology")
    print("  Input:   examples/sample_mesh.obj (400 verts)")
    print("  Output:  demo_output/retopo/")

    start = time.time()
    blender = BlenderBackend()
    if not blender.ensure_available():
        print("  ✗ Blender not available, skipping")
        return False

    out_dir = Path("demo_output/retopo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decimate by 50%
    step("Decimating mesh (50%)...")
    half = out_dir / "retopo_half.obj"
    blender.apply_decimate("examples/sample_mesh.obj", str(half), ratio=0.5)
    ok(f"Saved: {half} ({half.stat().st_size / 1024:.1f} KB)")

    # Decimate by 90%
    step("Decimating mesh (90% reduction)...")
    tenth = out_dir / "retopo_10pct.obj"
    blender.apply_decimate("examples/sample_mesh.obj", str(tenth), ratio=0.1)
    ok(f"Saved: {tenth} ({tenth.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - start
    ok(f"Retopo completed in {elapsed:.2f}s")
    return True


def demo_info():
    """Show system info."""
    from assetforge.cli import app
    from typer.testing import CliRunner

    section("3. System Information")
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    print(result.stdout)


def main():
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │         AssetForge — Demo Suite             │")
    print("  │  Real-world image → game-ready assets       │")
    print("  └─────────────────────────────────────────────┘")

    success = True
    success &= demo_2d_pipeline()
    success &= demo_retopo()

    print(f"\n{SEP}")
    if success:
        print("  ✅ All pipelines completed successfully!")
        print(f"\n  Output directory: demo_output/")
        print(f"  Run `tree demo_output/` to see all files.")
    else:
        print("  ⚠️  Some pipelines had issues (see above).")
    print(SEP)
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
