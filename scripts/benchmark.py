#!/usr/bin/env python3
"""
AssetForge Benchmark Suite.

Measures pipeline performance across different configurations:
- End-to-end time per pipeline
- Per-stage timing
- Output file sizes
- Quality metrics (PSNR, IoU)
"""

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")


@dataclass
class BenchmarkResult:
    pipeline: str
    config: dict
    duration_s: float
    output_files: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    peak_vram_gb: float = 0.0
    error: str | None = None


def benchmark_2d_pipeline(
    input_path: str, output_dir: str, **kwargs
) -> BenchmarkResult:
    """Run and benchmark the 2D sprite pipeline."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    config = {"input": input_path, **kwargs}
    start = time.time()

    try:
        pipeline = Pipeline2D(input_path=input_path, output_dir=output_dir, **kwargs)
        results = pipeline.run()

        duration = time.time() - start
        success = all(r.success for r in results)

        output_files = {}
        if success:
            for r in results:
                for p in r.output_paths:
                    p = Path(p)
                    if p.exists():
                        output_files[p.name] = str(p)

        return BenchmarkResult(
            pipeline="2d",
            config=config,
            duration_s=round(duration, 2),
            output_files=output_files,
            error=None if success else "Stage failed",
        )
    except Exception as e:
        duration = time.time() - start
        return BenchmarkResult(
            pipeline="2d",
            config=config,
            duration_s=round(duration, 2),
            error=str(e),
        )


def benchmark_retopo(input_path: str, output_path: str, **kwargs) -> BenchmarkResult:
    """Benchmark the standalone retopo command."""
    from assetforge.backends.blender import BlenderBackend

    config = {"input": input_path, "output": output_path, **kwargs}
    start = time.time()

    try:
        blender = BlenderBackend()
        if not blender.ensure_available():
            raise RuntimeError("Blender not available")

        decimate_ratio = kwargs.get("ratio", 0.5)
        output = Path(output_path)
        temp_dec = output.with_suffix(".decimated.obj")

        blender.apply_decimate(
            str(Path(input_path).resolve()),
            str(temp_dec.resolve()),
            ratio=decimate_ratio,
        )
        blender.export_format(str(temp_dec.resolve()), str(output.resolve()))

        if temp_dec.exists() and temp_dec != output:
            temp_dec.unlink()

        duration = time.time() - start
        size_kb = output.stat().st_size / 1024 if output.exists() else 0

        return BenchmarkResult(
            pipeline="retopo",
            config=config,
            duration_s=round(duration, 2),
            output_files={output.name: str(output)},
            metrics={"size_kb": round(size_kb, 1)},
        )
    except Exception as e:
        duration = time.time() - start
        return BenchmarkResult(
            pipeline="retopo",
            config=config,
            duration_s=round(duration, 2),
            error=str(e),
        )


def run_all_benchmarks(
    output_dir: str = "./benchmark_results",
) -> list[BenchmarkResult]:
    """Run a standard set of benchmarks."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    # 2D benchmarks
    input_2d = Path("examples/sample_2d_input.png")
    if input_2d.exists():
        logger.info("Benchmarking 2D pipeline (Rembg)...")
        results.append(
            benchmark_2d_pipeline(
                str(input_2d), str(out / "2d_rembg"), background_remover="rembg"
            )
        )

        logger.info("Benchmarking 2D pipeline (Rembg + sheet)...")
        results.append(
            benchmark_2d_pipeline(
                str(input_2d),
                str(out / "2d_sheet"),
                background_remover="rembg",
                pack_sheet=True,
            )
        )

    # Retopo benchmark
    input_mesh = Path("examples/sample_mesh.obj")
    if input_mesh.exists():
        logger.info("Benchmarking retopo (decimate 50%)...")
        results.append(
            benchmark_retopo(str(input_mesh), str(out / "retopo_half.obj"), ratio=0.5)
        )

        logger.info("Benchmarking retopo (decimate 90%)...")
        results.append(
            benchmark_retopo(str(input_mesh), str(out / "retopo_90pct.obj"), ratio=0.1)
        )

    # Save raw JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }

    report_path = out / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nBenchmark report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  AssetForge Benchmark Summary")
    print("=" * 60)
    print(f"  {'Pipeline':<20s} {'Duration':>10s}  {'Status':>8s}  {'Details':<30s}")
    print("  " + "-" * 70)
    for r in results:
        status = "✓ PASS" if r.error is None else "✗ FAIL"
        detail = r.error or ""
        if not detail and r.output_files:
            # show file sizes
            sizes = []
            for name, path in r.output_files.items():
                p = Path(path)
                if p.exists():
                    sizes.append(f"{p.stat().st_size / 1024:.1f}KB")
            if sizes:
                detail = ", ".join(sizes)
        print(f"  {r.pipeline:<20s} {r.duration_s:>8.2f}s  {status:>8s}  {detail:<30s}")
    print("=" * 60)

    # Save markdown summary
    md_path = out / "benchmark_report.md"
    with open(md_path, "w") as f:
        f.write("# AssetForge Benchmark Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Pipeline | Duration (s) | Status | Details |\n")
        f.write("|----------|-------------|--------|--------|\n")
        for r in results:
            status = "✅" if r.error is None else "❌"
            detail = r.error or ""
            if not detail and r.output_files:
                sizes = []
                for name, path in r.output_files.items():
                    p = Path(path)
                    if p.exists():
                        sizes.append(f"{name}: {p.stat().st_size / 1024:.1f}KB")
                detail = "; ".join(sizes)
            f.write(f"| {r.pipeline} | {r.duration_s:.2f}s | {status} | {detail} |\n")
        f.write("\n")
    logger.info(f"Markdown report saved to: {md_path}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
